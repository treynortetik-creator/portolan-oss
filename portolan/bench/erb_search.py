"""EnterpriseRAG-Bench (ERB) as Portolan's search exam: database, embedder,
and the retrieval passes. The pure half (corpus reading, question
selection, ERB's metrics, BM25) is `portolan.bench.erb`; its docstring
carries the provenance of every definition and the no-training rule.

Three things this module enforces rather than documents:

1. **Only `portolan_erb` on a local server.** `assert_erb_database` refuses
   any other database name and any non-local host, so a stray
   `DATABASE_URL` can never point this loader at Railway, a dev graph, or
   a `portolan_test_*` database. The URL is built from host/port only;
   `DATABASE_URL` is never read.
2. **Spend is capped and metered.** `OpenRouterEmbedder` records every
   request's real `usage.cost` in `portolan.training.spend.SpendMeter`
   under corpus key `erb_exam`, and refuses to send a request once that
   corpus's spend reaches the budget. `estimate_embedding_cost` runs
   before any paid call; the CLI stops when the estimate exceeds the cap.
3. **The index is Portolan's own.** Rows are exactly what
   `portolan.search.index.index_note` writes (`chunk_note` with the note
   title as chunk 0, the note body hash in `note_index_state`), only
   batched across notes and written with COPY. The embedding model is
   Portolan's local default, `BAAI/bge-large-en-v1.5` (1024-d,
   `portolan.search.embed.FastEmbedEmbedder`), served by OpenRouter
   because local ONNX on the dev machine managed 0.5 chunks/s against a
   ~1.2M-chunk corpus (write-up has the numbers). OpenRouter has no BAA:
   fine for a public synthetic benchmark, never for customer data.
"""

from __future__ import annotations

import json
import math
import os
import random
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx
import psycopg

from portolan.bench.erb import (
    TOP_K,
    Bm25Index,
    ErbDoc,
    aggregate,
    document_recall_pct,
    fact_numeral_gate,
    first_gold_rank,
    invalid_extra_docs,
    invent_number,
)
from portolan.search.chunk import chunk_note
from portolan.search.index import _body_hash
from portolan.training.spend import SpendMeter
from portolan.util.redact import redact_dsn

__all__ = [
    "BGE_QUERY_INSTRUCTION",
    "EMBED_MODEL",
    "EMBED_PRICE_PER_MILLION",
    "ERB_DB_NAME",
    "INGEST_TIMESTAMP",
    "SPEND_CORPUS",
    "EF_SEARCH_GRID",
    "EF_SEARCH_RECALL_TOLERANCE",
    "VECTOR_WEIGHT_GRID",
    "BudgetExceeded",
    "CachingEmbedder",
    "LoadReport",
    "OpenRouterEmbedder",
    "assert_erb_database",
    "choose_ef_search",
    "choose_vector_weight",
    "chunk_texts_for",
    "ensure_erb_database",
    "erb_database_url",
    "estimate_embedding_cost",
    "load_documents",
    "read_openrouter_key",
    "run_exact_passes",
    "exact_vector_scan",
    "fuse",
    "fusion_rows",
    "latency_summary",
    "load_scan",
    "run_ef_sweep",
    "run_passes",
    "run_weight_sweep",
    "save_scan",
    "title_boost_levels",
    "tune_fusion_weights",
]

ERB_DB_NAME = "portolan_erb"
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", ""}
SPEND_CORPUS = "erb_exam"
# OpenRouter's name for the model Portolan runs locally via fastembed.
# Price read from GET https://openrouter.ai/api/v1/embeddings/models on
# 2026-09-22: "prompt": "0.00000001" per token = $0.01 per million.
EMBED_MODEL = "baai/bge-large-en-v1.5"
EMBED_PRICE_PER_MILLION = 0.01
EMBED_DIM = 1024
_OPENROUTER_EMBEDDINGS_URL = "https://openrouter.ai/api/v1/embeddings"
# bge-large-en-v1.5 model card's retrieval instruction for short queries.
# Portolan's FastEmbedEmbedder deliberately sends none (see its
# docstring); the bench measures both, as an ablation.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
# Every ERB note gets the same `updated`, as a bulk ingest would stamp it,
# so Portolan's recency multiplier is identical for every candidate and
# cannot reorder anything: ERB has no recency questions to reward it.
INGEST_TIMESTAMP = datetime(2026, 9, 22, tzinfo=UTC)


# ---------------------------------------------------------------------
# Database guard
# ---------------------------------------------------------------------


def erb_database_url(host: str = "localhost", port: int = 5432, user: str | None = None) -> str:
    netloc = f"{user}@{host}:{port}" if user else f"{host}:{port}"
    return urlunsplit(("postgresql", netloc, f"/{ERB_DB_NAME}", "", ""))


def assert_erb_database(url: str) -> None:
    """Raise unless `url` names exactly `portolan_erb` on a local host."""
    parts = urlsplit(url)
    dbname = parts.path.lstrip("/").split("?", 1)[0]
    host = parts.hostname or ""
    if dbname != ERB_DB_NAME:
        raise ValueError(
            f"refusing database {dbname!r} ({redact_dsn(url)}): the ERB exam only ever "
            f"writes to {ERB_DB_NAME!r}"
        )
    if host not in _LOCAL_HOSTS:
        raise ValueError(
            f"refusing host {host!r} ({redact_dsn(url)}): the ERB exam only runs against "
            "a local Postgres"
        )


def ensure_erb_database(url: str, *, fresh: bool = False) -> bool:
    """Create `portolan_erb` if missing (or drop and recreate it with
    `fresh`), then return True if it was (re)created. Guarded by
    `assert_erb_database` before anything is touched."""
    assert_erb_database(url)
    parts = urlsplit(url)
    maintenance = urlunsplit((parts.scheme, parts.netloc, "/postgres", "", ""))
    conn = psycopg.connect(maintenance, autocommit=True)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (ERB_DB_NAME,))
            exists = cur.fetchone() is not None
            if exists and fresh:
                cur.execute(
                    "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                    "WHERE datname = %s AND pid <> pg_backend_pid()",
                    (ERB_DB_NAME,),
                )
                cur.execute(f'DROP DATABASE "{ERB_DB_NAME}"')
                exists = False
            if not exists:
                cur.execute(f'CREATE DATABASE "{ERB_DB_NAME}"')
                return True
    finally:
        conn.close()
    return False


# ---------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------


class BudgetExceeded(RuntimeError):
    """The `erb_exam` corpus spend reached the cap; no further paid call."""


class InputTooLong(RuntimeError):
    """The provider rejected an input over the model's 512-token window
    (DeepInfra errors where fastembed would silently truncate)."""


def read_openrouter_key(env_file: Path | None) -> str:
    """`OPENROUTER_API_KEY` from the environment, else from `env_file`.
    The value is returned, never printed or logged."""
    value = os.environ.get("OPENROUTER_API_KEY")
    if value:
        return value.strip()
    if env_file is not None and env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("OPENROUTER_API_KEY="):
                value = line.split("=", 1)[1].strip().strip('"').strip("'")
                if value:
                    return value
    raise RuntimeError(
        "OPENROUTER_API_KEY not found in the environment or the given --env-file "
        "(its value is never printed)"
    )


def _decode_embedding(value: Any) -> list[float]:
    """A base64 little-endian float32 string (`encoding_format: base64`),
    or a plain JSON list if a provider ignores the format request."""
    if isinstance(value, str):
        import base64

        import numpy as np

        return np.frombuffer(base64.b64decode(value), dtype="<f4").tolist()
    return list(value)


class OpenRouterEmbedder:
    """`Embedder` over OpenRouter's OpenAI-compatible `/api/v1/embeddings`.

    `embed()` splits its input into requests of at most
    `max_inputs_per_request`, keeps the response order by `index`, checks
    every vector is `EMBED_DIM` wide, and records each response's
    `usage.cost` (falling back to `prompt_tokens x price` if a response
    ever omits it) in `meter` under `corpus`. Before each request it
    raises `BudgetExceeded` if that corpus's spend is already at
    `budget_usd`. 429/5xx and transport errors retry with backoff.
    Thread-safe: the loader calls `embed()` from a worker pool.
    """

    dim = EMBED_DIM

    def __init__(
        self,
        api_key: str,
        *,
        model: str = EMBED_MODEL,
        meter: SpendMeter | None = None,
        corpus: str = SPEND_CORPUS,
        budget_usd: float = 5.0,
        price_per_million: float = EMBED_PRICE_PER_MILLION,
        max_inputs_per_request: int = 64,
        max_retries: int = 6,
        timeout: float = 120.0,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.model = model
        self.meter = meter
        self.corpus = corpus
        self.budget_usd = budget_usd
        self.price_per_million = price_per_million
        self.max_inputs_per_request = max_inputs_per_request
        self.max_retries = max_retries
        self._sleep = sleep
        self._client = httpx.Client(
            transport=transport,
            timeout=timeout,
            headers={"Authorization": f"Bearer {api_key}"},
        )
        self._lock = threading.Lock()
        self.requests = 0
        self.prompt_tokens = 0
        self.spent_usd = 0.0
        self.providers: Counter = Counter()
        self.retries = 0
        self.truncated_inputs = 0

    def close(self) -> None:
        self._client.close()

    def _check_budget(self) -> None:
        spent = self.meter.corpus_usd(self.corpus) if self.meter else self.spent_usd
        if spent >= self.budget_usd:
            raise BudgetExceeded(
                f"{self.corpus} spend ${spent:.4f} has reached the ${self.budget_usd:.2f} cap"
            )

    def _post(self, inputs: list[str]) -> dict[str, Any]:
        # base64 float32 instead of JSON floats: a quarter of the bytes and
        # no per-float JSON parsing (measured 2026-09-22: 353 KB vs 1.39 MB
        # for 64 inputs, identical vectors, cosine 1.0).
        body = {"model": self.model, "input": inputs, "encoding_format": "base64"}
        last_error = "no attempt made"
        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.post(_OPENROUTER_EMBEDDINGS_URL, json=body)
            except httpx.TransportError as exc:
                last_error = f"{type(exc).__name__}"
            else:
                if response.status_code == 200:
                    payload = response.json()
                    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
                        return payload
                    last_error = f"200 without data: {str(payload)[:200]}"
                elif response.status_code in (408, 409, 425, 429) or response.status_code >= 500:
                    last_error = f"HTTP {response.status_code}"
                elif response.status_code == 400 and "input tokens" in response.text:
                    raise InputTooLong(response.text[:200])
                else:
                    raise RuntimeError(
                        f"OpenRouter embeddings HTTP {response.status_code}: {response.text[:300]}"
                    )
            if attempt < self.max_retries:
                with self._lock:
                    self.retries += 1
                self._sleep(min(60.0, 2.0**attempt) + random.random())
        raise RuntimeError(f"OpenRouter embeddings failed after retries: {last_error}")

    def _embed_request(self, inputs: list[str]) -> list[list[float]]:
        self._check_budget()
        payload = self._post(inputs)
        data = sorted(payload["data"], key=lambda item: item.get("index", 0))
        if len(data) != len(inputs):
            raise RuntimeError(f"asked for {len(inputs)} embeddings, got {len(data)}")
        vectors = [_decode_embedding(item["embedding"]) for item in data]
        for vec in vectors:
            if len(vec) != self.dim:
                raise RuntimeError(f"expected {self.dim}-d embeddings, got {len(vec)}")
        usage = payload.get("usage") or {}
        tokens = int(usage.get("prompt_tokens") or usage.get("total_tokens") or 0)
        cost = usage.get("cost")
        amount = float(cost) if cost is not None else tokens * self.price_per_million / 1e6
        with self._lock:
            self.requests += 1
            self.prompt_tokens += tokens
            self.spent_usd += amount
            self.providers[str(payload.get("provider") or "unknown")] += 1
        if self.meter is not None:
            self.meter.record(model=self.model, corpus=self.corpus, amount_usd=amount)
        return vectors

    def _embed_fitting(self, inputs: list[str]) -> list[list[float]]:
        """`_embed_request`, recovering from an over-long input the way
        fastembed would have handled it locally -- by truncating it: split
        the request until the offending input is alone, then cut 10% of its
        characters at a time until the provider accepts it. Rare (the
        chunker caps chunks at 1,200 characters; a few token-dense ones
        exceed 512 bge tokens) and counted in `truncated_inputs`."""
        try:
            return self._embed_request(inputs)
        except InputTooLong:
            if len(inputs) > 1:
                mid = len(inputs) // 2
                return self._embed_fitting(inputs[:mid]) + self._embed_fitting(inputs[mid:])
            text = inputs[0]
            for _ in range(20):
                text = text[: max(1, int(len(text) * 0.9))]
                try:
                    vectors = self._embed_request([text])
                except InputTooLong:
                    continue
                with self._lock:
                    self.truncated_inputs += 1
                return vectors
            raise

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        step = self.max_inputs_per_request
        for i in range(0, len(texts), step):
            out.extend(self._embed_fitting(list(texts[i : i + step])))
        return out


class CachingEmbedder:
    """Memoizes `embed()` per text, so one question embedded for the fused
    pass is not paid for again by the vector-only pass. An optional
    `prefix` is prepended before embedding (the BGE instruction ablation)."""

    def __init__(self, inner: Any, prefix: str = "") -> None:
        self.inner = inner
        self.prefix = prefix
        self.dim = inner.dim
        self._cache: dict[str, list[float]] = {}

    def embed(self, texts: list[str]) -> list[list[float]]:
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            vectors = self.inner.embed([self.prefix + t for t in missing])
            self._cache.update(zip(missing, vectors, strict=True))
        return [self._cache[t] for t in texts]


# ---------------------------------------------------------------------
# Plan (cost estimate) and load
# ---------------------------------------------------------------------


def chunk_texts_for(doc: ErbDoc) -> list[tuple[str | None, str, int]]:
    """(heading, text, token_estimate) for every chunk `index_note` would
    write for this document as a note titled `doc.title` with body
    `doc.content` -- the same `chunk_note` call, title as chunk 0."""
    return [
        (c.heading, c.text, c.token_estimate)
        for c in chunk_note(doc.content, title=doc.title, description=None)
    ]


def find_bge_tokenizer() -> Path | None:
    """fastembed's cached `tokenizer.json` for bge-large-en-v1.5, if the
    model was ever downloaded on this machine (`FASTEMBED_CACHE_PATH`, else
    fastembed's default `<tmp>/fastembed_cache`)."""
    import tempfile

    roots = [
        os.environ.get("FASTEMBED_CACHE_PATH"),
        Path(tempfile.gettempdir()) / "fastembed_cache",
    ]
    for root in roots:
        if not root:
            continue
        base = Path(root) / "models--qdrant--bge-large-en-v1.5-onnx" / "snapshots"
        if base.is_dir():
            for snap in sorted(base.iterdir()):
                candidate = snap / "tokenizer.json"
                if candidate.is_file():
                    return candidate
    return None


@dataclass
class PlanReport:
    per_source: dict[str, dict[str, int]] = field(default_factory=dict)
    tokenizer: str = "chars/3 (no local tokenizer found; deliberately pessimistic)"
    price_per_million: float = EMBED_PRICE_PER_MILLION

    @property
    def total_tokens(self) -> int:
        return sum(v["tokens"] for v in self.per_source.values())

    @property
    def estimated_usd(self) -> float:
        return self.total_tokens * self.price_per_million / 1e6

    def as_dict(self) -> dict[str, Any]:
        return {
            "per_source": self.per_source,
            "tokenizer": self.tokenizer,
            "price_per_million_usd": self.price_per_million,
            "total_documents": sum(v["documents"] for v in self.per_source.values()),
            "total_chunks": sum(v["chunks"] for v in self.per_source.values()),
            "total_chars": sum(v["chars"] for v in self.per_source.values()),
            "total_tokens": self.total_tokens,
            "estimated_usd": round(self.estimated_usd, 4),
        }


def estimate_embedding_cost(
    docs: Iterable[ErbDoc],
    *,
    tokenizer_path: Path | None = None,
    price_per_million: float = EMBED_PRICE_PER_MILLION,
    progress: Callable[[str], None] | None = None,
) -> PlanReport:
    """Chunk every document exactly as the loader will and count what the
    embeddings will cost: exact bge tokens (with [CLS]/[SEP], which the
    provider bills) when the local tokenizer is available, else a
    pessimistic chars/3."""
    tok = None
    report = PlanReport(price_per_million=price_per_million)
    if tokenizer_path is not None:
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(str(tokenizer_path))
        tok.enable_truncation(512)
        report.tokenizer = f"bge-large-en-v1.5 tokenizer ({tokenizer_path.name}), truncation 512"
    pending: list[tuple[str, str]] = []

    def _flush() -> None:
        if not pending:
            return
        texts = [t for _, t in pending]
        if tok is not None:
            counts = [len(enc.ids) for enc in tok.encode_batch(texts)]
        else:
            counts = [max(1, len(t) // 3) for t in texts]
        for (source, _), n in zip(pending, counts, strict=True):
            report.per_source[source]["tokens"] += n
        pending.clear()

    for n_docs, doc in enumerate(docs, 1):
        stats = report.per_source.setdefault(
            doc.source, {"documents": 0, "chunks": 0, "chars": 0, "tokens": 0}
        )
        stats["documents"] += 1
        for _heading, text, _est in chunk_texts_for(doc):
            stats["chunks"] += 1
            stats["chars"] += len(text)
            pending.append((doc.source, text))
        if len(pending) >= 4096:
            _flush()
        if progress and n_docs % 20000 == 0:
            progress(f"  plan: {n_docs} documents chunked")
    _flush()
    return report


@dataclass
class LoadReport:
    documents_seen: int = 0
    documents_skipped_already_indexed: int = 0
    notes_written: int = 0
    chunks_written: int = 0
    stopped_by_budget: bool = False
    seconds: float = 0.0
    seconds_preparing: float = 0.0
    seconds_waiting_for_embeddings: float = 0.0
    seconds_writing: float = 0.0


def _already_indexed(conn: psycopg.Connection) -> set[str]:
    with conn.cursor() as cur:
        cur.execute("SELECT note_slug FROM note_index_state")
        return {row[0] for row in cur.fetchall()}


def _write_batch(
    conn: psycopg.Connection,
    batch: list[tuple[ErbDoc, list[tuple[str | None, str, int]], list[list[float]]]],
    updated: datetime,
) -> int:
    """One transaction per batch: notes, their chunks (with vectors), and
    `note_index_state` -- the same three writes `index_note` makes, so a
    crash mid-load leaves only whole, indexed notes behind."""
    import numpy as np

    chunks_written = 0
    with conn.transaction():
        with conn.cursor() as cur:
            with cur.copy(
                "COPY notes (slug, type, title, status, created, updated, body_md) FROM STDIN"
            ) as copy:
                for doc, _chunks, _vecs in batch:
                    copy.write_row(
                        (doc.dsid, "source", doc.title, "living", updated, updated, doc.content)
                    )
            with cur.copy(
                "COPY chunks (note_slug, heading, text, token_count, embedding) "
                "FROM STDIN WITH (FORMAT BINARY)"
            ) as copy:
                copy.set_types(["text", "text", "text", "int4", "vector"])
                for doc, chunks, vecs in batch:
                    for (heading, text, est), vec in zip(chunks, vecs, strict=True):
                        copy.write_row((doc.dsid, heading, text, est, np.asarray(vec, np.float32)))
                        chunks_written += 1
            with cur.copy(
                "COPY note_index_state (note_slug, body_hash, indexed_at) FROM STDIN"
            ) as copy:
                for doc, _chunks, _vecs in batch:
                    copy.write_row((doc.dsid, _body_hash(doc.content), updated))
    return chunks_written


def load_documents(
    conn: psycopg.Connection,
    docs: Iterable[ErbDoc],
    embedder: Any,
    *,
    batch_docs: int = 300,
    workers: int = 8,
    request_size: int = 64,
    updated: datetime = INGEST_TIMESTAMP,
    progress: Callable[[str], None] | None = None,
) -> LoadReport:
    """Chunk, embed and write every document not already indexed.

    Pipelined: while batch N's embedding requests are in flight on the
    worker pool, batch N-1 is written and batch N+1 is parsed/chunked.
    Resumable: notes already in `note_index_state` are skipped. Stops
    cleanly (after writing every batch whose embeddings completed) on
    `BudgetExceeded`.
    """
    from pgvector.psycopg import register_vector

    register_vector(conn)
    report = LoadReport()
    started = time.monotonic()
    done = _already_indexed(conn)
    pool = ThreadPoolExecutor(max_workers=workers)
    pending: tuple[list[tuple[ErbDoc, list]], list[Future]] | None = None

    def _submit(items: list[tuple[ErbDoc, list]]) -> list[Future]:
        texts = [text for _doc, chunks in items for _h, text, _e in chunks]
        return [
            pool.submit(embedder.embed, texts[i : i + request_size])
            for i in range(0, len(texts), request_size)
        ]

    def _finish(items: list[tuple[ErbDoc, list]], futures: list[Future]) -> None:
        vectors: list[list[float]] = []
        t_wait = time.monotonic()
        for fut in futures:
            vectors.extend(fut.result())
        report.seconds_waiting_for_embeddings += time.monotonic() - t_wait
        batch = []
        pos = 0
        for doc, chunks in items:
            batch.append((doc, chunks, vectors[pos : pos + len(chunks)]))
            pos += len(chunks)
        t_write = time.monotonic()
        report.chunks_written += _write_batch(conn, batch, updated)
        report.seconds_writing += time.monotonic() - t_write
        report.notes_written += len(batch)
        if progress:
            rate = report.chunks_written / max(1e-9, time.monotonic() - started)
            spent = getattr(embedder, "spent_usd", 0.0)
            progress(
                f"  load: {report.notes_written} notes, {report.chunks_written} chunks, "
                f"{rate:.0f} chunks/s, ${spent:.4f} this run, "
                f"retries {getattr(embedder, 'retries', 0)}, "
                f"truncated {getattr(embedder, 'truncated_inputs', 0)}, "
                f"prep/wait/write {report.seconds_preparing:.0f}/"
                f"{report.seconds_waiting_for_embeddings:.0f}/{report.seconds_writing:.0f}s"
            )

    def _batches() -> Iterator[list[tuple[ErbDoc, list]]]:
        items: list[tuple[ErbDoc, list]] = []
        t_prep = time.monotonic()
        for doc in docs:
            report.documents_seen += 1
            if doc.dsid in done:
                report.documents_skipped_already_indexed += 1
                continue
            items.append((doc, chunk_texts_for(doc)))
            if len(items) >= batch_docs:
                report.seconds_preparing += time.monotonic() - t_prep
                yield items
                items = []
                t_prep = time.monotonic()
        if items:
            report.seconds_preparing += time.monotonic() - t_prep
            yield items

    try:
        for items in _batches():
            futures = _submit(items)
            if pending is not None:
                _finish(*pending)
            pending = (items, futures)
        if pending is not None:
            _finish(*pending)
            pending = None
    except BudgetExceeded:
        report.stopped_by_budget = True
        if progress:
            progress("  load: stopped -- erb_exam budget reached; completed batches are kept")
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        report.seconds = round(time.monotonic() - started, 1)
    return report


# ---------------------------------------------------------------------
# Passes
# ---------------------------------------------------------------------

PASS_NAMES = (
    "fused",
    "portolan_fts",
    "portolan_vector",
    "bm25",
    "rrf_bm25_vector",
    "portolan_vector_bge_instruction",
    "fused_ef1000",
)
# Vector-only pass: `_vector_candidates(k)` fetches 3k nearest chunks and
# keeps each note's best; k=34 fetches 102 chunks, the analogue of ERB's
# vector baseline fetching 100 chunks (`vector_retrieval.py --chunk-limit`)
# before keeping the first 10 distinct documents.
VECTOR_ONLY_K = 34
# pgvector 0.8 without iterative scans returns at most `hnsw.ef_search`
# rows from an HNSW scan. `fused` runs `search()` as production does, at
# the configured `search_hnsw_ef_search`; the vector-only passes use 200 so
# a 102-chunk fetch can actually return 102 chunks, and `fused_ef1000`
# uses pgvector's maximum to measure what the approximate index costs.
VECTOR_ONLY_EF_SEARCH = 200
EF_SEARCH_MAX = 1000


def _row(question: dict[str, Any], retrieved: list[str], seconds: float) -> dict[str, Any]:
    expected = question.get("expected_doc_ids") or []
    top = list(retrieved[:TOP_K])
    recall = document_recall_pct(top, expected)
    return {
        "question_id": question["question_id"],
        "question_type": question["question_type"],
        "source_types": sorted(question.get("source_types") or []),
        "expected_doc_ids": list(expected),
        "retrieved_doc_ids": top,
        "document_recall_pct": None if recall is None else round(recall, 2),
        "invalid_extra_docs": invalid_extra_docs(top, expected),
        "first_gold_rank": first_gold_rank(top, expected),
        "seconds": round(seconds, 3),
    }


def title_boost_levels(
    conn: psycopg.Connection, query: str, slugs: Iterable[str]
) -> dict[str, float]:
    """Each slug's title boost as a fraction of the top RRF score
    (`_title_boost` with `rrf_max` 1.0): numbers only, so fusion can be
    re-run offline for other weights without the titles themselves."""
    from portolan.search import query as q

    terms = q._terms(query)
    titles = _titles(conn, slugs)
    levels = {s: q._title_boost(terms, titles.get(s, ""), 1.0) for s in titles}
    return {s: v for s, v in levels.items() if v}


def fuse(
    fts_list: Sequence[str],
    vector_list: Sequence[str],
    title_levels: dict[str, float],
    *,
    fts_weight: float | None = None,
    vector_weight: float | None = None,
) -> list[str]:
    """Portolan's own fusion arithmetic (`_rrf`, then the title boost scaled
    by the top RRF score, then `(-score, slug)`; recency is a constant here,
    see INGEST_TIMESTAMP) over two externally supplied candidate lists.
    Weights default to `search()`'s own."""
    from portolan.search import query as q

    fused, _via, scores = q._rrf(
        [
            (list(fts_list), q.FTS_WEIGHT if fts_weight is None else fts_weight, "fts"),
            (
                list(vector_list),
                q.VECTOR_WEIGHT if vector_weight is None else vector_weight,
                "vector",
            ),
        ]
    )
    if not fused:
        return []
    rrf_max = max(scores.values())
    final = {s: scores[s] + title_levels.get(s, 0.0) * rrf_max for s in fused}
    return sorted(final, key=lambda s: (-final[s], s))


def _titles(conn: psycopg.Connection, slugs: Iterable[str]) -> dict[str, str]:
    slugs = list(dict.fromkeys(slugs))
    if not slugs:
        return {}
    with conn.cursor() as cur:
        cur.execute("SELECT slug, title FROM notes WHERE slug = ANY(%s)", (slugs,))
        return dict(cur.fetchall())


@dataclass
class GateTally:
    hits_checked: int = 0
    hits_verbatim: int = 0
    hits_gate_exact: int = 0
    hits_unresolvable: int = 0
    facts_with_numerals: int = 0
    facts_pass_cited_chunks: int = 0
    facts_pass_retrieved_docs: int = 0
    facts_pass_gold_docs: int = 0
    invented_facts: int = 0
    invented_facts_caught_with_gold_docs: int = 0

    def as_dict(self) -> dict[str, Any]:
        def pct(a: int, b: int) -> float | None:
            return round(a / b * 100, 2) if b else None

        return {
            "provenance_hits_checked": self.hits_checked,
            "provenance_hits_verbatim_in_original_file": self.hits_verbatim,
            "provenance_hits_unresolvable": self.hits_unresolvable,
            "provenance_verbatim_pct": pct(self.hits_verbatim, self.hits_checked),
            "provenance_hits_pass_quote_gate_exact_rule": self.hits_gate_exact,
            "provenance_quote_gate_exact_pct": pct(self.hits_gate_exact, self.hits_checked),
            "numeral_facts": self.facts_with_numerals,
            "numeral_facts_pass_with_cited_chunks": self.facts_pass_cited_chunks,
            "numeral_facts_pass_with_cited_chunks_pct": pct(
                self.facts_pass_cited_chunks, self.facts_with_numerals
            ),
            "numeral_facts_pass_with_full_retrieved_docs": self.facts_pass_retrieved_docs,
            "numeral_facts_pass_with_full_retrieved_docs_pct": pct(
                self.facts_pass_retrieved_docs, self.facts_with_numerals
            ),
            "numeral_facts_pass_with_gold_docs": self.facts_pass_gold_docs,
            "numeral_facts_pass_with_gold_docs_pct": pct(
                self.facts_pass_gold_docs, self.facts_with_numerals
            ),
            "invented_numeral_facts": self.invented_facts,
            "invented_numeral_facts_caught_with_gold_docs": (
                self.invented_facts_caught_with_gold_docs
            ),
            "invented_numeral_facts_caught_with_gold_docs_pct": pct(
                self.invented_facts_caught_with_gold_docs, self.invented_facts
            ),
        }


def run_passes(
    conn: psycopg.Connection,
    embedder: Any,
    questions: Sequence[dict[str, Any]],
    *,
    bm25: Bm25Index | None,
    original_text: Callable[[str], str | None],
    instruction_embedder: Any | None = None,
    passes: Sequence[str] = PASS_NAMES,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run every requested pass over every question and return per-pass
    rows, per-pass aggregates (overall, by type, by source), fused-pass
    diagnostics and the citation-gate tally.

    Passes (all top 10, all over the one `portolan_erb` index except bm25):

    - `fused` -- `portolan.search.search(conn, embedder, q, k=10)`, as is
      (the configured `search_hnsw_ef_search`).
    - `portolan_fts` -- its keyword leg alone (`_fts_candidates`, the same
      30-candidate pool `search()` asks for).
    - `portolan_vector` -- its vector leg alone (`_vector_candidates`,
      102 nearest chunks, first 10 distinct notes).
    - `bm25` -- `Bm25Index` over the same documents' same text.
    - `rrf_bm25_vector` -- Portolan's fusion arithmetic with BM25's top 30
      in place of the keyword leg (vector leg exactly as `fused` has it).
    - `portolan_vector_bge_instruction` -- `portolan_vector` with the BGE
      model card's query instruction prepended.
    - `fused_ef1000` -- `fused` with `hnsw.ef_search` at pgvector's max.

    `original_text(dsid)` re-reads a document's indexed text from the ERB
    file on disk (not from the database), so the provenance check follows
    the chain hit -> chunk -> original file.
    """
    from portolan import config
    from portolan.search import query as q
    from portolan.search.query import search

    production_ef_search = config.settings.search_hnsw_ef_search
    unknown = set(passes) - set(PASS_NAMES)
    if unknown:
        raise ValueError(f"unknown pass(es): {sorted(unknown)}")
    if ({"bm25", "rrf_bm25_vector"} & set(passes)) and bm25 is None:
        raise ValueError("bm25 passes need a Bm25Index")
    if "portolan_vector_bge_instruction" in passes and instruction_embedder is None:
        raise ValueError("portolan_vector_bge_instruction needs instruction_embedder")

    rows: dict[str, list[dict[str, Any]]] = {name: [] for name in passes}
    gate = GateTally()
    via_counts: Counter = Counter()
    fts_zero = 0
    fts_pool = max(TOP_K * 3, q.FTS_CANDIDATE_POOL)

    def _timed(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[Any, float]:
        t0 = time.monotonic()
        value = fn(*args, **kwargs)
        return value, time.monotonic() - t0

    for i, question in enumerate(questions, 1):
        text = question["question"]

        if "fused" in passes:
            result, dt = _timed(search, conn, embedder, text, k=TOP_K)
            rows["fused"].append(_row(question, [h.note_slug for h in result.hits], dt))
            _gate_question(gate, via_counts, question, result.hits, original_text)

        if "fused_ef1000" in passes:
            result, dt = _timed(search, conn, embedder, text, k=TOP_K, ef_search=EF_SEARCH_MAX)
            rows["fused_ef1000"].append(_row(question, [h.note_slug for h in result.hits], dt))

        if "portolan_fts" in passes:
            (fts_rank, _), dt = _timed(q._fts_candidates, conn, text, fts_pool)
            fts_zero += int(not fts_rank)
            rows["portolan_fts"].append(_row(question, fts_rank, dt))

        if "portolan_vector" in passes:
            (vec_rank, _), dt = _timed(
                q._vector_candidates,
                conn,
                embedder,
                text,
                VECTOR_ONLY_K,
                ef_search=VECTOR_ONLY_EF_SEARCH,
            )
            rows["portolan_vector"].append(_row(question, vec_rank, dt))

        if "portolan_vector_bge_instruction" in passes:
            (vec_rank, _), dt = _timed(
                q._vector_candidates,
                conn,
                instruction_embedder,
                text,
                VECTOR_ONLY_K,
                ef_search=VECTOR_ONLY_EF_SEARCH,
            )
            rows["portolan_vector_bge_instruction"].append(_row(question, vec_rank, dt))

        if "bm25" in passes or "rrf_bm25_vector" in passes:
            bm25_top, dt = _timed(bm25.search, text, fts_pool)
            if "bm25" in passes:
                rows["bm25"].append(_row(question, bm25_top, dt))
            if "rrf_bm25_vector" in passes:
                # The vector list exactly as `search()` builds it (k=10,
                # the configured ef_search), then Portolan's own fusion.
                t0 = time.monotonic()
                vec10, _ = q._vector_candidates(
                    conn, embedder, text, TOP_K, ef_search=production_ef_search
                )
                levels = title_boost_levels(conn, text, [*bm25_top, *vec10])
                fused_list = fuse(bm25_top, vec10, levels)
                dt2 = time.monotonic() - t0
                rows["rrf_bm25_vector"].append(_row(question, fused_list, dt + dt2))

        if progress and i % 10 == 0:
            progress(f"  run: {i}/{len(questions)} questions")

    return _summarize(rows, via_counts, fts_zero, gate)


def _gate_question(
    gate: GateTally,
    via_counts: Counter,
    question: dict[str, Any],
    hits: Sequence[Any],
    original_text: Callable[[str], str | None],
) -> None:
    """Portolan's citation gate, retrieval form, on one fused result.

    Provenance: every hit's `best_chunk` (the passage Portolan cites) is
    checked against the original ERB file's indexed text twice: byte-exact
    substring, and the quote gate's own exact rule
    (`portolan.extract.quote_gate._quote_supported_by_text`: whitespace
    collapsed, glyphs folded, case folded, no fuzzy matching). They differ
    when `chunk_note` packs two paragraphs into one chunk: it joins them
    without the blank line that separated them in the body, so the chunk
    is no longer a byte-exact span of the source.
    Numerals: each gold answer fact carrying a checkable numeral is run
    through the chat answer path's numbers guard against (a) the chunks
    cited, (b) the full retrieved documents, (c) the gold documents; and
    the same fact with one number changed (`invent_number`) against (c),
    which the guard must reject.
    """
    from portolan.extract.quote_gate import _quote_supported_by_text

    for h in hits:
        via_counts["+".join(h.via) or "expand"] += 1
        gate.hits_checked += 1
        original = original_text(h.note_slug)
        if original is None:
            gate.hits_unresolvable += 1
            continue
        if h.best_chunk and h.best_chunk in original:
            gate.hits_verbatim += 1
        if _quote_supported_by_text(h.best_chunk, original):
            gate.hits_gate_exact += 1
    cited_chunks = [h.best_chunk for h in hits]
    retrieved_docs = [original_text(h.note_slug) or "" for h in hits]
    gold_docs = [original_text(d) or "" for d in question.get("expected_doc_ids") or []]
    for fact in question.get("answer_facts") or []:
        verdict = fact_numeral_gate(fact, cited_chunks)
        if verdict is None:
            continue
        gate.facts_with_numerals += 1
        gate.facts_pass_cited_chunks += int(verdict[0])
        full = fact_numeral_gate(fact, retrieved_docs)
        gate.facts_pass_retrieved_docs += int(bool(full and full[0]))
        gold = fact_numeral_gate(fact, gold_docs)
        gate.facts_pass_gold_docs += int(bool(gold and gold[0]))
        invented = invent_number(fact)
        verdict_invented = fact_numeral_gate(invented, gold_docs) if invented else None
        if verdict_invented is not None:
            gate.invented_facts += 1
            gate.invented_facts_caught_with_gold_docs += int(not verdict_invented[0])


# ---------------------------------------------------------------------
# Exact vector leg (one streaming pass for every query at once)
# ---------------------------------------------------------------------

EXACT_PASS_NAMES = (
    "fused_exact",
    "portolan_fts",
    "portolan_vector_exact",
    "portolan_vector_bge_instruction_exact",
    "bm25",
    "rrf_bm25_vector_exact",
)


@dataclass(frozen=True)
class ChunkHit:
    note_slug: str
    chunk_id: int
    distance: float


def _as_float32(value: Any) -> Any:
    """A pgvector value as a float32 numpy array (`Vector` in text mode,
    an ndarray in binary mode, or a plain list)."""
    import numpy as np

    if hasattr(value, "to_numpy"):
        value = value.to_numpy()
    return np.asarray(value, dtype=np.float32)


def exact_vector_scan(
    conn: psycopg.Connection,
    query_vectors: dict[str, list[float]],
    n_chunks: int,
    *,
    batch_rows: int = 20000,
    progress: Callable[[str], None] | None = None,
) -> dict[str, list[ChunkHit]]:
    """The `n_chunks` nearest chunks to every query vector by cosine
    distance, computed exactly in one sequential pass over `chunks`.

    This is what `portolan.search.query._vector_candidates` returns when
    Postgres answers it with a sequential scan (no HNSW index): ordered by
    `embedding <=> query` ascending, ties by `notes.updated DESC` (constant
    here, see INGEST_TIMESTAMP) then `note_slug ASC`. Streaming every
    query at once turns 157 x 3 table scans (about 45 s each on the dev
    machine) into one.
    """
    import heapq

    import numpy as np
    from pgvector.psycopg import register_vector

    register_vector(conn)
    keys = list(query_vectors)
    q = np.asarray([query_vectors[k] for k in keys], dtype=np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    best: dict[str, list[tuple[float, str, int]]] = {k: [] for k in keys}
    seen = 0
    with conn.transaction(), conn.cursor(name="erb_exact_scan") as cur:
        cur.itersize = batch_rows
        cur.execute("SELECT id, note_slug, embedding FROM chunks WHERE embedding IS NOT NULL")
        while True:
            rows = cur.fetchmany(batch_rows)
            if not rows:
                break
            ids = [r[0] for r in rows]
            slugs = [r[1] for r in rows]
            m = np.stack([_as_float32(r[2]) for r in rows])
            m /= np.linalg.norm(m, axis=1, keepdims=True)
            dist = 1.0 - (m @ q.T)  # rows x queries
            take = min(n_chunks, len(rows))
            part = np.argpartition(dist, take - 1, axis=0)[:take]
            for j, key in enumerate(keys):
                cand = [(float(dist[i, j]), slugs[i], ids[i]) for i in part[:, j]]
                best[key] = heapq.nsmallest(
                    n_chunks, best[key] + cand, key=lambda t: (t[0], t[1], t[2])
                )
            seen += len(rows)
            if progress and seen % (batch_rows * 10) == 0:
                progress(f"  exact scan: {seen:,} chunks")
    return {k: [ChunkHit(s, i, d) for d, s, i in v] for k, v in best.items()}


def save_scan(path: Path, scan: dict[str, list[ChunkHit]]) -> None:
    """`exact_vector_scan`'s result as JSON (slugs, chunk ids, distances:
    no text), so a later run can reuse a scan that takes minutes."""
    rows = {k: [[h.note_slug, h.chunk_id, h.distance] for h in v] for k, v in scan.items()}
    write_json(path, rows)


def load_scan(path: Path) -> dict[str, list[ChunkHit]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {k: [ChunkHit(s, int(i), float(d)) for s, i, d in v] for k, v in data.items()}


def notes_from_chunks(hits: Sequence[ChunkHit], n_chunks: int) -> list[ChunkHit]:
    """`_vector_candidates`' post-processing over its first `n_chunks`
    rows: stop at the similarity floor, keep each note's first (nearest)
    chunk, in order."""
    from portolan.search.query import VECTOR_MIN_SIMILARITY

    out: list[ChunkHit] = []
    seen: set[str] = set()
    for h in hits[:n_chunks]:
        if (1.0 - h.distance) < VECTOR_MIN_SIMILARITY:
            break
        if h.note_slug in seen:
            continue
        seen.add(h.note_slug)
        out.append(h)
    return out


def _chunk_texts(conn: psycopg.Connection, chunk_ids: Iterable[int]) -> dict[int, str]:
    ids = list(dict.fromkeys(chunk_ids))
    if not ids:
        return {}
    with conn.cursor() as cur:
        cur.execute("SELECT id, text FROM chunks WHERE id = ANY(%s)", (ids,))
        return dict(cur.fetchall())


@dataclass
class _Hit:
    note_slug: str
    best_chunk: str
    via: list[str]


def run_exact_passes(
    conn: psycopg.Connection,
    questions: Sequence[dict[str, Any]],
    *,
    plain: dict[str, list[ChunkHit]],
    instructed: dict[str, list[ChunkHit]] | None,
    bm25: Bm25Index | None,
    original_text: Callable[[str], str | None],
    progress: Callable[[str], None] | None = None,
    fusion_routing: bool | None = None,
) -> dict[str, Any]:
    """The same passes as `run_passes`, with the vector leg taken from
    `exact_vector_scan` (keyed by question id) instead of a pgvector query.

    `fused_exact` is `portolan.search.search()`'s ranking, step for step:
    the real keyword leg (`_fts_leg`, pool 30), the vector leg's
    first 30 chunks deduped to notes (`search(k=10)` fetches 3k = 30),
    `_rrf` with `search()`'s weights (per query from `_route_weights` when
    `fusion_routing` is on; None reads `search_fusion_routing`), `_title_boost`,
    the recency multiplier (identical for every note here), `(-score, slug)` order, top 10; the
    cited chunk per hit is the keyword leg's chunk when it matched, else
    the vector leg's nearest chunk, else chunk 0, as `search()` picks it.
    `tests/test_bench_erb.py` checks it against `search()` itself.
    """
    from portolan.search import query as q

    rows: dict[str, list[dict[str, Any]]] = {
        "fused_exact": [],
        "portolan_fts": [],
        "portolan_vector_exact": [],
        "bm25": [],
        "rrf_bm25_vector_exact": [],
    }
    if instructed is not None:
        rows["portolan_vector_bge_instruction_exact"] = []
    gate = GateTally()
    via_counts: Counter = Counter()
    fts_zero = 0
    fts_pool = max(TOP_K * 3, q.FTS_CANDIDATE_POOL)
    legs: dict[str, dict[str, Any]] = {}
    if fusion_routing is None:
        fusion_routing = q._settings().search_fusion_routing

    for i, question in enumerate(questions, 1):
        qid = question["question_id"]
        text = question["question"]
        t0 = time.monotonic()
        fts_rank, fts_chunks, coverage = q._fts_leg(conn, text, fts_pool)
        dt_fts = time.monotonic() - t0
        fts_zero += int(not fts_rank)
        rows["portolan_fts"].append(_row(question, fts_rank, dt_fts))

        vec30 = notes_from_chunks(plain[qid], 3 * TOP_K)
        vec_list = [h.note_slug for h in vec30]
        t0 = time.monotonic()
        levels = title_boost_levels(conn, text, [*fts_rank, *vec_list])
        fts_w, vec_w = (
            q._route_weights(coverage) if fusion_routing else (q.FTS_WEIGHT, q.VECTOR_WEIGHT)
        )
        fused = fuse(fts_rank, vec_list, levels, fts_weight=fts_w, vector_weight=vec_w)[:TOP_K]
        via = {s: [] for s in fused}
        for s in fused:
            if s in fts_rank:
                via[s].append("fts")
            if s in vec_list:
                via[s].append("vector")
        nearest = {h.note_slug: h.chunk_id for h in vec30}
        texts = _chunk_texts(conn, [nearest[s] for s in fused if s in nearest])
        hits = []
        for s in fused:
            if s in fts_chunks:
                chunk = fts_chunks[s][1]
            elif s in nearest:
                chunk = texts.get(nearest[s], "")
            else:
                chunk = q._fallback_chunk(conn, s)[1]
            hits.append(_Hit(note_slug=s, best_chunk=chunk, via=sorted(via[s])))
        dt = dt_fts + (time.monotonic() - t0)
        rows["fused_exact"].append(_row(question, fused, dt))
        _gate_question(gate, via_counts, question, hits, original_text)
        legs[qid] = {"keyword": fts_rank, "vector": vec_list, "title_levels": levels}

        vec102 = [h.note_slug for h in notes_from_chunks(plain[qid], 3 * VECTOR_ONLY_K)]
        rows["portolan_vector_exact"].append(_row(question, vec102, 0.0))
        if instructed is not None:
            inst = [h.note_slug for h in notes_from_chunks(instructed[qid], 3 * VECTOR_ONLY_K)]
            rows["portolan_vector_bge_instruction_exact"].append(_row(question, inst, 0.0))

        if bm25 is not None:
            t0 = time.monotonic()
            bm25_top = bm25.search(text, fts_pool)
            rows["bm25"].append(_row(question, bm25_top, time.monotonic() - t0))
            bm25_levels = title_boost_levels(conn, text, bm25_top)
            fused_b = fuse(bm25_top, vec_list, {**bm25_levels, **levels})
            rows["rrf_bm25_vector_exact"].append(_row(question, fused_b, 0.0))
            legs[qid]["bm25"] = bm25_top
            legs[qid]["title_levels"] = {**bm25_levels, **levels}

        if progress and i % 10 == 0:
            progress(f"  run: {i}/{len(questions)} questions")

    out = _summarize(rows, via_counts, fts_zero, gate)
    out["legs"] = legs
    out["fusion_weights"] = {"fts": q.FTS_WEIGHT, "vector": q.VECTOR_WEIGHT}
    out["fusion_routing"] = fusion_routing
    return out


# Vector-leg weights tried against a keyword weight of 1.0 (search()'s
# before this grid: 1.2). Not 1.0 itself: at equal weights every pair of
# hits at the same rank in the two legs ties exactly and the slug decides.
FUSION_GRID_VECTOR_WEIGHTS = (0.2, 0.25, 0.33, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.5)


def fusion_rows(
    questions: Sequence[dict[str, Any]],
    legs: dict[str, dict[str, Any]],
    *,
    fts_weight: float,
    vector_weight: float,
    keyword_leg: str = "keyword",
) -> list[dict[str, Any]]:
    """Per-question rows for `fuse` over the recorded legs of
    `run_exact_passes` (ids and title-boost levels only), at the given
    weights. `keyword_leg="bm25"` fuses the BM25 baseline instead."""
    out = []
    for question in questions:
        leg = legs[question["question_id"]]
        ranked = fuse(
            leg[keyword_leg],
            leg["vector"],
            leg["title_levels"],
            fts_weight=fts_weight,
            vector_weight=vector_weight,
        )
        out.append(_row(question, ranked[:TOP_K], 0.0))
    return out


def tune_fusion_weights(
    dev_questions: Sequence[dict[str, Any]],
    legs: dict[str, dict[str, Any]],
    *,
    grid: Sequence[float] = FUSION_GRID_VECTOR_WEIGHTS,
    fts_weight: float = 1.0,
) -> dict[str, Any]:
    """Every vector weight in `grid` on the dev questions only, and the one
    chosen by a rule fixed before looking: highest mean recall@10, then
    highest MRR@10, then the weight closest to the keyword weight."""
    table = [
        {
            "fts_weight": fts_weight,
            "vector_weight": w,
            "dev": aggregate(
                fusion_rows(dev_questions, legs, fts_weight=fts_weight, vector_weight=w)
            ),
        }
        for w in grid
    ]
    chosen = max(
        table,
        key=lambda r: (
            r["dev"]["average_recall_pct"],
            r["dev"]["ours_mrr_at_10"],
            -abs(math.log(r["vector_weight"] / fts_weight)),
        ),
    )
    return {"grid": table, "chosen": chosen}


# ---------------------------------------------------------------------
# hnsw.ef_search on the production path
# ---------------------------------------------------------------------

EF_SEARCH_GRID = (40, 100, 200, 400)
# Dev recall within this many points of the grid's best counts as a tie,
# and a tie goes to the smallest ef_search (the least index work).
EF_SEARCH_RECALL_TOLERANCE = 0.5


def run_ef_sweep(
    conn: psycopg.Connection,
    embedder: Any,
    questions: Sequence[dict[str, Any]],
    *,
    ef_values: Sequence[int] = EF_SEARCH_GRID,
    progress: Callable[[str], None] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """`search(k=10)` itself, through the HNSW index, at every
    `hnsw.ef_search` in `ef_values`: `{str(ef): rows}`, rows as `_row`
    (ids, numbers, seconds per search). Each question runs every value
    back to back, the order rotated by one per question, so no value
    always runs on pages another value just read into the cache."""
    from portolan.search.query import search

    values = [int(v) for v in ef_values]
    rows: dict[str, list[dict[str, Any]]] = {str(v): [] for v in values}
    for i, question in enumerate(questions):
        shift = i % len(values)
        for ef in values[shift:] + values[:shift]:
            t0 = time.monotonic()
            result = search(conn, embedder, question["question"], k=TOP_K, ef_search=ef)
            dt = time.monotonic() - t0
            rows[str(ef)].append(_row(question, [h.note_slug for h in result.hits], dt))
        if progress and (i + 1) % 20 == 0:
            progress(f"  ef sweep: {i + 1}/{len(questions)} questions")
    for value_rows in rows.values():
        value_rows.sort(key=lambda r: r["question_id"])
    return rows


def latency_summary(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    """p50 and p95 (nearest rank) and mean of the rows' `seconds`, in ms."""
    secs = sorted(r["seconds"] for r in rows)
    if not secs:
        return {"p50_ms": 0.0, "p95_ms": 0.0, "mean_ms": 0.0}

    def _rank(p: float) -> float:
        return secs[max(0, math.ceil(p * len(secs)) - 1)] * 1000

    return {
        "p50_ms": round(_rank(0.50), 1),
        "p95_ms": round(_rank(0.95), 1),
        "mean_ms": round(sum(secs) / len(secs) * 1000, 1),
    }


def choose_ef_search(
    dev_recall: dict[int, float], *, tolerance: float = EF_SEARCH_RECALL_TOLERANCE
) -> int:
    """The smallest ef_search whose dev recall@10 is within `tolerance`
    points of the best dev recall in the grid: dev numbers only."""
    best = max(dev_recall.values())
    return min(ef for ef, recall in dev_recall.items() if recall >= best - tolerance)


# ---------------------------------------------------------------------
# Fusion weight on the production path
# ---------------------------------------------------------------------

# Vector-leg weights against a keyword weight of 1.0.
VECTOR_WEIGHT_GRID = (0.2, 0.4, 0.6, 0.8, 1.0)


def run_weight_sweep(
    conn: psycopg.Connection,
    embedder: Any,
    questions: Sequence[dict[str, Any]],
    *,
    vector_weights: Sequence[float] = VECTOR_WEIGHT_GRID,
    ef_search: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """`search(k=10)` itself at keyword weight 1.0 and each vector weight
    (swapped in by `fusion_weights`, so recency and the title boost are
    search()'s own), HNSW at `ef_search` (None: the configured value):
    `{str(weight): rows}`, rows as `_row`."""
    from portolan.bench.demo_search import fusion_weights
    from portolan.search.query import search

    rows: dict[str, list[dict[str, Any]]] = {str(w): [] for w in vector_weights}
    for i, question in enumerate(questions):
        for w in vector_weights:
            with fusion_weights(1.0, w):
                t0 = time.monotonic()
                result = search(conn, embedder, question["question"], k=TOP_K, ef_search=ef_search)
                dt = time.monotonic() - t0
            rows[str(w)].append(_row(question, [h.note_slug for h in result.hits], dt))
        if progress and (i + 1) % 20 == 0:
            progress(f"  weight sweep: {i + 1}/{len(questions)} questions")
    return rows


def choose_vector_weight(
    erb_dev_recall: dict[float, float],
    paraphrase_recall: dict[float, float],
    demo_recall: dict[float, float],
) -> float:
    """Among weights that keep every demo question found (recall 100), the
    one with the highest min(ERB dev recall, paraphrase recall); ties to
    the higher paraphrase recall, then the higher ERB dev recall, then the
    smaller weight. ERB held-out numbers never enter."""
    allowed = [w for w in erb_dev_recall if demo_recall.get(w) == 100.0]
    if not allowed:
        raise ValueError("no vector weight keeps all demo questions in the top 10")
    return max(
        allowed,
        key=lambda w: (
            min(erb_dev_recall[w], paraphrase_recall[w]),
            paraphrase_recall[w],
            erb_dev_recall[w],
            -w,
        ),
    )


def _summarize(
    rows: dict[str, list[dict[str, Any]]], via_counts: Counter, fts_zero: int, gate: GateTally
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for name, name_rows in rows.items():
        if not name_rows:
            continue
        by_type: dict[str, list] = {}
        by_source: dict[str, list] = {}
        for r in name_rows:
            by_type.setdefault(r["question_type"], []).append(r)
            by_source.setdefault("+".join(r["source_types"]), []).append(r)
        summary[name] = {
            "overall": aggregate(name_rows),
            "by_type": {k: aggregate(v) for k, v in sorted(by_type.items())},
            "by_source": {k: aggregate(v) for k, v in sorted(by_source.items())},
            "mean_seconds": round(sum(r["seconds"] for r in name_rows) / len(name_rows), 3),
        }
    return {
        "summary": summary,
        "rows": {k: v for k, v in rows.items() if v},
        "fused_diagnostics": {
            "hit_via_counts": dict(via_counts),
            "portolan_fts_zero_candidate_questions": fts_zero,
        },
        "citation_gate": gate.as_dict() if gate.hits_checked else None,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)
