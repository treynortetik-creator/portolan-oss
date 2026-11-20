"""Reranker and query-routing evaluation over recorded retrieval legs.

`search()` is `_legs` (the two retrieval legs, the database's work) then
`_rank` (fusion, title boost, recency, the optional rerank). This module
records each question's legs once, scores every candidate note a
configuration could rerank with each model once, and then runs `_rank`
itself for every configuration, with a lookup table standing in for the
model. The rank is therefore search()'s own code; only the model's
arithmetic is cached, and `tests/test_rerank_eval.py` checks that the
table gives what the model gives.

Recorded legs hold note text, so they stay in `var/` (gitignored), as ERB
text must: results written for `01-spec/` carry ids and numbers only.

Latency per configuration = the question's legs time + `_rank`'s own time
+ the model time of the notes in its rerank pool, each note timed when it
was scored, one pair per call.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import psycopg

from portolan.bench.erb import TOP_K, aggregate, document_recall_pct, first_gold_rank
from portolan.search import query as q
from portolan.search.rerank import OnnxReranker, Reranker

__all__ = [
    "LookupReranker",
    "ROUTE_GRID",
    "candidate_passages",
    "choose",
    "evaluate",
    "latency_summary",
    "load_legs",
    "pool_slugs",
    "record_legs",
    "route_profile",
    "save_legs",
    "score_candidates",
]

# Keyword profile fixed at the weights ERB dev chose (1.0 / 0.2, bench write-up
# "Fusion weight"); the grid moves the coverage threshold and the vector
# weight of the semantic profile.
ROUTE_KEYWORD_WEIGHTS = (1.0, 0.2)
ROUTE_GRID: tuple[tuple[float, float], ...] = tuple(
    (threshold, semantic_vector)
    for threshold in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
    for semantic_vector in (1.0, 1.5, 2.0)
)


def record_legs(
    conn: psycopg.Connection,
    embedder: Any,
    questions: Sequence[dict[str, Any]],
    *,
    k: int = TOP_K,
    progress: Callable[[str], None] | None = None,
) -> dict[str, dict[str, Any]]:
    """`_legs` for every question (the query vector embedded first, so the
    time is the database's): `{qid: {"legs": {...}, "seconds": s}}`."""
    out: dict[str, dict[str, Any]] = {}
    for i, x in enumerate(questions):
        embedder.embed([x["question"]])
        t0 = time.monotonic()
        legs = q._legs(conn, embedder, x["question"], k)
        out[x["question_id"]] = {"legs": asdict(legs), "seconds": time.monotonic() - t0}
        if progress and (i + 1) % 20 == 0:
            progress(f"  legs: {i + 1}/{len(questions)}")
    return out


def save_legs(path: Path, legs: dict[str, dict[str, Any]]) -> None:
    _write_json(path, legs)


def load_legs(path: Path) -> dict[str, tuple[q.Legs, float]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for qid, rec in raw.items():
        d = rec["legs"]
        legs = q.Legs(
            query=d["query"],
            fts_rank=list(d["fts_rank"]),
            fts_chunks={s: (h, t) for s, (h, t) in d["fts_chunks"].items()},
            vector_rank=list(d["vector_rank"]),
            vector_chunks={s: (h, t) for s, (h, t) in d["vector_chunks"].items()},
            coverage=float(d["coverage"]),
        )
        out[qid] = (legs, float(rec["seconds"]))
    return out


@contextlib.contextmanager
def route_profile(threshold: float, semantic_vector: float) -> Iterator[None]:
    """`_route_weights`' threshold and profiles set for the block, then restored."""
    saved = (q.ROUTE_COVERAGE_THRESHOLD, q.ROUTE_KEYWORD_WEIGHTS, q.ROUTE_SEMANTIC_WEIGHTS)
    q.ROUTE_COVERAGE_THRESHOLD = threshold
    q.ROUTE_KEYWORD_WEIGHTS = ROUTE_KEYWORD_WEIGHTS
    q.ROUTE_SEMANTIC_WEIGHTS = (1.0, semantic_vector)
    try:
        yield
    finally:
        q.ROUTE_COVERAGE_THRESHOLD, q.ROUTE_KEYWORD_WEIGHTS, q.ROUTE_SEMANTIC_WEIGHTS = saved


def pool_slugs(
    conn: psycopg.Connection, legs: q.Legs, *, pool: int, fusion_routing: bool
) -> list[str]:
    """The notes `_rank` would hand a reranker with this pool: its own order
    before the rerank, cut at `pool`."""
    result = q._rank(conn, legs, k=pool, fusion_routing=fusion_routing, reranker=False)
    return [h.note_slug for h in result.hits]


def candidate_passages(conn: psycopg.Connection, legs: q.Legs, slugs: Sequence[str]) -> list[str]:
    """What `_rank` passes the reranker for each of `slugs`
    (`_rerank_passages`, with the same chunk lookup `_rank` builds)."""
    return q._rerank_passages(conn, list(slugs), {**legs.vector_chunks, **legs.fts_chunks})


def score_candidates(
    conn: psycopg.Connection,
    reranker: Reranker,
    legs: q.Legs,
    slugs: Sequence[str],
    done: dict[str, list[float]],
) -> dict[str, list[float]]:
    """`[score, seconds, tokens]` per slug not already in `done`, one pair per
    call so each note's time is its own (tokens 0 for a model without a
    tokenizer)."""
    todo = [s for s in slugs if s not in done]
    passages = candidate_passages(conn, legs, todo)
    if isinstance(reranker, OnnxReranker) and todo:
        tokens = reranker.token_counts(legs.query, passages)
    else:
        tokens = [0] * len(todo)
    out: dict[str, list[float]] = {}
    for slug, passage, n in zip(todo, passages, tokens, strict=True):
        t0 = time.monotonic()
        [score] = reranker.score(legs.query, [passage])
        out[slug] = [score, time.monotonic() - t0, n]
    return out


class LookupReranker:
    """A reranker that returns recorded scores: `table[passage]` for one
    question. Raises on a passage that was never scored."""

    def __init__(self, name: str, query: str, by_passage: dict[str, float]) -> None:
        self.name = name
        self.query = query
        self.by_passage = by_passage
        self.passages_seen: list[str] = []

    def score(self, query: str, passages: list[str]) -> list[float]:
        assert query == self.query
        self.passages_seen.extend(passages)
        return [self.by_passage[p] for p in passages]


def evaluate(
    conn: psycopg.Connection,
    questions: Sequence[dict[str, Any]],
    legs: dict[str, tuple[q.Legs, float]],
    *,
    fusion_routing: bool,
    model: str | None = None,
    scores: dict[str, dict[str, list[float]]] | None = None,
    pool: int = 30,
) -> list[dict[str, Any]]:
    """`_rank(k=10)` for every question under one configuration, rows in the
    ERB bench's shape (ids and numbers only) plus the latency parts."""
    rows = []
    for x in questions:
        qid = x["question_id"]
        leg, leg_seconds = legs[qid]
        reranker: Any = False
        model_seconds = 0.0
        if model is not None:
            assert scores is not None
            slugs = pool_slugs(conn, leg, pool=max(pool, TOP_K), fusion_routing=fusion_routing)
            passages = candidate_passages(conn, leg, slugs)
            recorded = scores[qid]
            reranker = LookupReranker(
                model,
                leg.query,
                {p: recorded[s][0] for s, p in zip(slugs, passages, strict=True)},
            )
            model_seconds = sum(recorded[s][1] for s in slugs)
        t0 = time.monotonic()
        result = q._rank(
            conn,
            leg,
            k=TOP_K,
            fusion_routing=fusion_routing,
            reranker=reranker,
            rerank_pool=pool,
        )
        rank_seconds = time.monotonic() - t0
        retrieved = [h.note_slug for h in result.hits]
        expected = list(x.get("expected_doc_ids") or ())
        recall = document_recall_pct(retrieved, expected)
        rows.append(
            {
                "question_id": qid,
                "question_type": x.get("question_type"),
                "expected_doc_ids": expected,
                "retrieved_doc_ids": retrieved,
                "document_recall_pct": None if recall is None else round(recall, 2),
                "invalid_extra_docs": None,
                "first_gold_rank": first_gold_rank(retrieved, expected),
                "coverage": round(leg.coverage, 4),
                "seconds": leg_seconds + rank_seconds + model_seconds,
                "model_seconds": model_seconds,
            }
        )
    return rows


def latency_summary(rows: Sequence[dict[str, Any]], key: str = "seconds") -> dict[str, float]:
    """p50 and p95 (nearest rank) of `key`, in ms."""
    secs = sorted(r[key] for r in rows)
    if not secs:
        return {"p50_ms": 0.0, "p95_ms": 0.0}

    def _rank(p: float) -> float:
        return round(secs[max(0, math.ceil(p * len(secs)) - 1)] * 1000, 1)

    return {"p50_ms": _rank(0.50), "p95_ms": _rank(0.95)}


def choose(candidates: dict[str, dict[str, float]]) -> str:
    """The orchestrator's rule (bench write-up, "Fusion weight"): among
    configurations whose demo recall is 100, the highest min(ERB dev recall,
    paraphrase recall); ties to the higher paraphrase recall, then the higher
    ERB dev recall, then the name. ERB held-out never enters."""
    allowed = [n for n, c in candidates.items() if c["demo"] == 100.0]
    if not allowed:
        raise ValueError("no configuration keeps every demo question in the top 10")
    return max(
        sorted(allowed),
        key=lambda n: (
            min(candidates[n]["erb_dev"], candidates[n]["paraphrase"]),
            candidates[n]["paraphrase"],
            candidates[n]["erb_dev"],
        ),
    )


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {**aggregate(list(rows)), **latency_summary(rows)}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, path)


write_json = _write_json
