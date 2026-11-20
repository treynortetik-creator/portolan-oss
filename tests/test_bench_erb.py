"""Tests for the EnterpriseRAG-Bench search exam (`portolan.bench.erb`,
`portolan.bench.erb_search`, checklist item 27).

No network (the embedder only ever sees an `httpx.MockTransport`), no ERB
text (every document and question here is invented for the test), and
the ERB clone is not needed except by the one smoke test at the bottom,
which skips when it is absent. DB-backed tests use the shared `db`
fixture (a `portolan_test_*` database), never `portolan_erb`.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from portolan.bench.erb import (
    Bm25Index,
    DocumentFieldError,
    ErbDoc,
    IterStats,
    aggregate,
    bm25_tokenize,
    default_erb_root,
    document_recall_pct,
    extract_document_content,
    fact_numeral_gate,
    first_gold_rank,
    invalid_extra_docs,
    iter_documents,
    load_questions,
    load_uuid_index,
    select_in_slice_questions,
)
from portolan.bench.erb_search import (
    BudgetExceeded,
    CachingEmbedder,
    OpenRouterEmbedder,
    assert_erb_database,
    chunk_texts_for,
    erb_database_url,
    estimate_embedding_cost,
    load_documents,
    read_openrouter_key,
    run_exact_passes,
    run_passes,
)
from portolan.search.chunk import chunk_note
from portolan.search.embed import FakeEmbedder
from portolan.search.index import index_pending
from portolan.training.spend import SpendMeter

# ---------------------------------------------------------------------
# extract_document_content (port of ERB src/utils/document_content.py)
# ---------------------------------------------------------------------


def test_extract_single_content_field_is_used_bare():
    doc = {"title_field_name": "t", "content_field_names": ["body"], "t": "Hi", "body": "x y"}
    assert extract_document_content(doc) == ("Hi", "x y")


def test_extract_multiple_fields_are_labelled_and_lists_joined():
    doc = {
        "title_field_name": "subject",
        "content_field_names": ["summary", "items"],
        "subject": "Weekly sync",
        "summary": "short",
        "items": ["one", "two"],
    }
    title, content = extract_document_content(doc)
    assert title == "Weekly sync"
    assert content == "summary:\nshort\n\nitems:\none\ntwo"


@pytest.mark.parametrize(
    "doc",
    [
        {"content_field_names": ["b"], "b": "x"},
        {"title_field_name": "t", "content_field_names": ["b"], "b": "x"},
        {"title_field_name": "t", "t": "x"},
        {"title_field_name": "t", "t": "x", "content_field_names": []},
        {"title_field_name": "t", "t": "x", "content_field_names": "b", "b": "y"},
        {"title_field_name": "t", "t": "x", "content_field_names": ["missing"]},
    ],
)
def test_extract_rejects_bad_labels(doc):
    with pytest.raises(DocumentFieldError):
        extract_document_content(doc)


# ---------------------------------------------------------------------
# iter_documents
# ---------------------------------------------------------------------


def _write_doc(root: Path, rel: str, dsid: str, title: str, body: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "title_field_name": "title",
                "content_field_names": ["body"],
                "title": title,
                "body": body,
                "dataset_doc_uuid": dsid,
            }
        )
    )


@pytest.fixture()
def tiny_corpus(tmp_path: Path) -> Path:
    sources = tmp_path / "sources"
    _write_doc(sources, "jira/b.json", "dsid_b", "Pump alarm ticket", "The pump alarm fired twice.")
    _write_doc(sources, "jira/a.json", "dsid_a", "Renewal note", "Renewal moved to March.")
    _write_doc(sources, "hubspot/x.json", "dsid_x", "Acme account", "Acme runs 47 pumps.")
    _write_doc(sources, "hubspot/dup.json", "dsid_x", "Acme duplicate", "A second copy.")
    (sources / "gmail").mkdir(parents=True)
    (sources / "gmail" / "broken.json").write_text("{not json")
    _write_doc(sources, "slack/ignored.json", "dsid_s", "Not a slice", "never read")
    return sources


def test_iter_documents_fixed_order_and_first_duplicate_wins(tiny_corpus: Path):
    stats = IterStats()
    docs = list(iter_documents(tiny_corpus, ("hubspot", "jira", "gmail"), stats=stats))
    assert [(d.source, d.dsid) for d in docs] == [
        ("hubspot", "dsid_x"),
        ("jira", "dsid_a"),
        ("jira", "dsid_b"),
    ]
    assert docs[0].relpath == "hubspot/dup.json"  # sorted path order, first wins
    assert stats.duplicate_dsid_skipped == 1
    assert stats.parse_failures == 1
    assert stats.files == 5
    assert docs[1].index_text == "Renewal note\n\nRenewal moved to March."


def test_iter_documents_prefers_the_uuid_index_path_for_a_duplicate(tiny_corpus: Path):
    uuid_index = {"dsid_x": "hubspot/x.json"}
    docs = list(iter_documents(tiny_corpus, ("hubspot",), uuid_index=uuid_index))
    assert [d.relpath for d in docs] == ["hubspot/x.json"]
    assert docs[0].title == "Acme account"


# ---------------------------------------------------------------------
# Question selection
# ---------------------------------------------------------------------


def _questions() -> list[dict]:
    return [
        {"question_id": "q1", "question_type": "basic", "source_types": ["jira"],
         "expected_doc_ids": ["dsid_a"]},
        {"question_id": "q2", "question_type": "semantic", "source_types": ["gmail", "jira"],
         "expected_doc_ids": ["dsid_b", "dsid_m"]},
        {"question_id": "q3", "question_type": "basic", "source_types": ["slack"],
         "expected_doc_ids": ["dsid_s"]},
        {"question_id": "q4", "question_type": "basic", "source_types": ["jira", "slack"],
         "expected_doc_ids": ["dsid_a", "dsid_s"]},
        {"question_id": "q5", "question_type": "info_not_found", "source_types": [],
         "expected_doc_ids": []},
    ]  # fmt: skip


def test_select_in_slice_questions_rule_a_and_cross_check():
    uuid_index = {
        "dsid_a": "jira/a.json",
        "dsid_b": "jira/b.json",
        "dsid_m": "gmail/m.json",
        "dsid_s": "slack/s.json",
    }
    sel = select_in_slice_questions(_questions(), ("jira", "gmail"), uuid_index=uuid_index)
    assert [q["question_id"] for q in sel.questions] == ["q1", "q2"]
    assert sel.without_gold == 1
    assert sel.by_gold_paths == 2
    assert sel.derivations_agree is True
    assert sel.by_type == {"basic": 1, "semantic": 1}
    assert sel.by_source_combo == {"jira": 1, "gmail+jira": 1}
    assert sel.distinct_gold_docs == 3
    assert "selected: 2" in sel.describe(("jira", "gmail"))


def test_select_flags_disagreement_between_the_two_rules():
    uuid_index = {"dsid_a": "slack/a.json", "dsid_b": "jira/b.json", "dsid_m": "gmail/m.json"}
    sel = select_in_slice_questions(_questions(), ("jira", "gmail"), uuid_index=uuid_index)
    assert sel.derivations_agree is False


def test_load_questions_skips_blank_lines(tmp_path: Path):
    path = tmp_path / "q.jsonl"
    path.write_text('{"question_id": "q1"}\n\n{"question_id": "q2"}\n')
    assert [q["question_id"] for q in load_questions(path)] == ["q1", "q2"]


# ---------------------------------------------------------------------
# ERB's retrieval metrics
# ---------------------------------------------------------------------


def test_document_recall_matches_erb_definition():
    assert document_recall_pct(["a", "b", "c"], ["a", "z"]) == 50.0
    assert document_recall_pct(["a", "a", "z"], ["a", "z"]) == 100.0
    assert document_recall_pct(["a"], []) is None


def test_invalid_extra_docs_counts_distinct_non_gold_non_valid():
    assert invalid_extra_docs(["a", "b", "b", "c"], ["a"]) == 2
    assert invalid_extra_docs(["a", "b", "c"], ["a"], valid=["c"]) == 1
    assert invalid_extra_docs(["a"], []) is None


def test_first_gold_rank_is_one_based_after_dedupe():
    assert first_gold_rank(["x", "x", "g"], ["g"]) == 2
    assert first_gold_rank(["x"], ["g"]) is None


def test_aggregate_skips_undefined_and_reports_hits_and_mrr():
    rows = [
        {"document_recall_pct": 100.0, "invalid_extra_docs": 9, "first_gold_rank": 1},
        {"document_recall_pct": 50.0, "invalid_extra_docs": 8, "first_gold_rank": 4},
        {"document_recall_pct": 0.0, "invalid_extra_docs": 10, "first_gold_rank": None},
        {"document_recall_pct": None, "invalid_extra_docs": None, "first_gold_rank": None},
    ]
    agg = aggregate(rows)
    assert agg["count"] == 4
    assert agg["average_recall_pct"] == 50.0
    assert agg["average_invalid_extra_docs"] == 9.0
    assert agg["ours_hit_at_1_pct"] == 25.0
    assert agg["ours_hit_at_5_pct"] == 50.0
    assert agg["ours_mrr_at_10"] == round((1 + 0.25) / 4, 4)


# ---------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------


def test_bm25_tokenize_keeps_dotted_and_underscored_identifiers():
    assert bm25_tokenize("Stream.Timebox_finalized fired; re-try 10.5 don't") == [
        "stream.timebox_finalized",
        "fired",
        "re",
        "try",
        "10.5",
        "don't",
    ]


def test_bm25_matches_the_okapi_formula_on_a_tiny_corpus():
    import math

    idx = Bm25Index()
    idx.add("d1", "pump alarm pump")
    idx.add("d2", "renewal alarm")
    idx.add("d3", "unrelated words here")
    idx.finalize()
    n, avgdl = 3, (3 + 2 + 3) / 3

    def expected(tf: int, df: int, dl: int) -> float:
        idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
        return idf * tf * 2.2 / (tf + 1.2 * (1 - 0.75 + 0.75 * dl / avgdl))

    scores = idx.scores("pump")
    assert scores[0] == pytest.approx(expected(2, 1, 3), rel=1e-5)
    assert scores[1] == 0 and scores[2] == 0
    # OR semantics: a document matching only one query term still ranks.
    assert idx.search("pump renewal", 10) == ["d1", "d2"] or idx.search("pump renewal", 10) == [
        "d2",
        "d1",
    ]
    assert idx.search("nothing matches", 10) == []


def test_bm25_ties_break_by_insertion_order():
    idx = Bm25Index()
    for dsid in ("z", "y", "x"):
        idx.add(dsid, "same text")
    idx.finalize()
    assert idx.search("same", 2) == ["z", "y"]


# ---------------------------------------------------------------------
# Citation gate, numeral half
# ---------------------------------------------------------------------


def test_fact_numeral_gate():
    assert fact_numeral_gate("The limit is 10 MiB per file.", ["max_file_size: 10 MiB"]) == (
        True,
        [],
    )
    passes, missing = fact_numeral_gate("The limit is 10 MiB per file.", ["no numbers"])
    assert passes is False and missing == ["10"]
    assert fact_numeral_gate("The owner is Dana.", ["anything"]) is None
    assert fact_numeral_gate("Filed on 2026-03-12 as SUP-100.", ["x"]) is None


# ---------------------------------------------------------------------
# Database guard
# ---------------------------------------------------------------------


def test_erb_database_url_is_local_portolan_erb():
    assert erb_database_url() == "postgresql://localhost:5432/portolan_erb"
    assert_erb_database(erb_database_url())
    assert_erb_database("postgresql://u:secret@127.0.0.1:5432/portolan_erb")


@pytest.mark.parametrize(
    "url",
    [
        "postgresql://localhost:5432/portolan",
        "postgresql://localhost:5432/portolan_test_123",
        "postgresql://localhost:5432/portolan_erb_copy",
        "postgresql://u:hunter2@db.railway.internal:5432/portolan_erb",
        "postgresql://u:hunter2@example.com/portolan_erb",
    ],
)
def test_assert_erb_database_refuses_everything_else_without_leaking_passwords(url):
    with pytest.raises(ValueError) as exc:
        assert_erb_database(url)
    assert "hunter2" not in str(exc.value)


def test_read_openrouter_key_from_env_file_only_when_env_unset(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text('OTHER=1\nOPENROUTER_API_KEY="sk-test-123"\n')
    assert read_openrouter_key(env) == "sk-test-123"
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-from-env")
    assert read_openrouter_key(env) == "sk-from-env"
    monkeypatch.delenv("OPENROUTER_API_KEY")
    with pytest.raises(RuntimeError) as exc:
        read_openrouter_key(tmp_path / "missing.env")
    assert "sk-" not in str(exc.value)


# ---------------------------------------------------------------------
# OpenRouterEmbedder (MockTransport only)
# ---------------------------------------------------------------------


def _mock_embeddings(calls: list, *, fail_first: int = 0, dim: int = 1024, with_cost=True):
    state = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        state["n"] += 1
        body = json.loads(request.content)
        calls.append(body)
        if state["n"] <= fail_first:
            return httpx.Response(429, json={"error": "slow down"})
        inputs = body["input"]
        data = [
            {"index": i, "embedding": [float(len(t))] + [0.0] * (dim - 1)}
            for i, t in enumerate(inputs)
        ]
        data.reverse()  # the client must re-order by index
        usage = {"prompt_tokens": 10 * len(inputs)}
        if with_cost:
            usage["cost"] = 1e-7 * len(inputs)
        return httpx.Response(200, json={"data": data, "usage": usage, "provider": "Mock"})

    return httpx.MockTransport(handler)


def test_embedder_batches_orders_and_meters(tmp_path: Path):
    calls: list = []
    meter = SpendMeter(tmp_path / "spend.json")
    emb = OpenRouterEmbedder(
        "sk-test",
        meter=meter,
        budget_usd=1.0,
        max_inputs_per_request=2,
        transport=_mock_embeddings(calls),
        sleep=lambda s: None,
    )
    vecs = emb.embed(["a", "bb", "ccc"])
    assert [v[0] for v in vecs] == [1.0, 2.0, 3.0]
    assert [len(c["input"]) for c in calls] == [2, 1]
    assert calls[0]["model"] == "baai/bge-large-en-v1.5"
    assert emb.requests == 2 and emb.prompt_tokens == 30
    assert meter.corpus_usd("erb_exam") == pytest.approx(3e-7)
    snap = meter.snapshot()
    assert snap["per_model_usd"]["baai/bge-large-en-v1.5"] == pytest.approx(3e-7)


def test_embedder_falls_back_to_tokens_times_price_without_cost(tmp_path: Path):
    meter = SpendMeter(tmp_path / "spend.json")
    emb = OpenRouterEmbedder(
        "sk-test",
        meter=meter,
        transport=_mock_embeddings([], with_cost=False),
        sleep=lambda s: None,
    )
    emb.embed(["a"])
    assert meter.corpus_usd("erb_exam") == pytest.approx(10 * 0.01 / 1e6)


def test_embedder_retries_429_then_succeeds(tmp_path: Path):
    calls: list = []
    slept: list = []
    emb = OpenRouterEmbedder(
        "sk-test",
        transport=_mock_embeddings(calls, fail_first=2),
        sleep=slept.append,
    )
    assert len(emb.embed(["x"])) == 1
    assert len(calls) == 3 and len(slept) == 2


def test_embedder_refuses_to_send_once_the_corpus_budget_is_spent(tmp_path: Path):
    calls: list = []
    meter = SpendMeter(tmp_path / "spend.json")
    meter.record(model="m", corpus="erb_exam", amount_usd=5.0)
    meter.record(model="m", corpus="other_corpus", amount_usd=50.0)
    emb = OpenRouterEmbedder(
        "sk-test", meter=meter, budget_usd=5.0, transport=_mock_embeddings(calls)
    )
    with pytest.raises(BudgetExceeded):
        emb.embed(["x"])
    assert calls == []
    # Another corpus's spend on the shared meter does not count against this cap.
    under = OpenRouterEmbedder(
        "sk-test", meter=meter, budget_usd=6.0, transport=_mock_embeddings(calls)
    )
    under.embed(["x"])
    assert len(calls) == 1


def test_embedder_asks_for_base64_and_decodes_float32():
    import base64

    import numpy as np

    calls: list = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body)
        vec = np.arange(1024, dtype="<f4") / 1024
        data = [
            {"index": i, "embedding": base64.b64encode(vec.tobytes()).decode()}
            for i, _ in enumerate(body["input"])
        ]
        return httpx.Response(200, json={"data": data, "usage": {"prompt_tokens": 3}})

    emb = OpenRouterEmbedder("sk-test", transport=httpx.MockTransport(handler))
    [vec] = emb.embed(["x"])
    assert calls[0]["encoding_format"] == "base64"
    assert len(vec) == 1024 and vec[1] == pytest.approx(1 / 1024)


def test_embedder_truncates_an_input_the_provider_says_is_too_long():
    calls: list = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body["input"])
        if any(len(t) > 50 for t in body["input"]):
            return httpx.Response(
                400, text='{"error": "You passed 513 input tokens, max is 512 input tokens"}'
            )
        data = [{"index": i, "embedding": [1.0] * 1024} for i, _ in enumerate(body["input"])]
        return httpx.Response(200, json={"data": data, "usage": {"prompt_tokens": 1}})

    emb = OpenRouterEmbedder("sk-test", transport=httpx.MockTransport(handler))
    vecs = emb.embed(["short", "x" * 60, "also short"])
    assert len(vecs) == 3
    assert emb.truncated_inputs == 1
    assert calls[-1] == ["also short"]  # split, truncated the long one, kept order


def test_embedder_rejects_wrong_dimension():
    emb = OpenRouterEmbedder("sk-test", transport=_mock_embeddings([], dim=384))
    with pytest.raises(RuntimeError, match="1024"):
        emb.embed(["x"])


def test_embedder_non_retryable_error_raises_without_the_key():
    def handler(request):
        return httpx.Response(401, json={"error": "bad key"})

    emb = OpenRouterEmbedder("sk-secret-value", transport=httpx.MockTransport(handler))
    with pytest.raises(RuntimeError) as exc:
        emb.embed(["x"])
    assert "sk-secret-value" not in str(exc.value)


def test_caching_embedder_embeds_each_text_once_and_applies_prefix():
    seen: list[list[str]] = []

    class Inner:
        dim = 3

        def embed(self, texts):
            seen.append(list(texts))
            return [[float(len(t)), 0.0, 0.0] for t in texts]

    cache = CachingEmbedder(Inner(), prefix="Q: ")
    first = cache.embed(["ab", "ab", "c"])
    second = cache.embed(["c"])
    assert seen == [["Q: ab", "Q: c"]]
    assert first[0] == [5.0, 0.0, 0.0] and second == [[4.0, 0.0, 0.0]]


# ---------------------------------------------------------------------
# Plan / estimate
# ---------------------------------------------------------------------


def _docs() -> list[ErbDoc]:
    return [
        ErbDoc(
            "dsid_1",
            "jira",
            "jira/1.json",
            "Pump alarm ticket",
            "The pump alarm fired twice.\n\nIt was reset at 09:00.",
        ),
        ErbDoc(
            "dsid_2",
            "hubspot",
            "hubspot/2.json",
            "Acme account",
            "Acme runs 47 pumps across three sites and renews in March.",
        ),
        ErbDoc(
            "dsid_3",
            "gmail",
            "gmail/3.json",
            "Re: renewal pricing",
            "Pricing for the renewal is flat. Legal wants the new MSA first.",
        ),
    ]


def test_chunk_texts_for_is_index_note_chunking():
    doc = _docs()[0]
    expected = chunk_note(doc.content, title=doc.title, description=None)
    assert chunk_texts_for(doc) == [(c.heading, c.text, c.token_estimate) for c in expected]
    assert chunk_texts_for(doc)[0][1] == "Pump alarm ticket"
    # The finding the provenance check surfaced: packed paragraphs lose the
    # blank line between them, so the chunk is not a byte-exact span.
    body_chunk = chunk_texts_for(doc)[1][1]
    assert body_chunk == "The pump alarm fired twice.\nIt was reset at 09:00."
    assert body_chunk not in doc.content


def test_estimate_without_tokenizer_is_chars_over_three():
    report = estimate_embedding_cost(_docs(), tokenizer_path=None, price_per_million=0.01)
    texts = [t for d in _docs() for _h, t, _e in chunk_texts_for(d)]
    assert report.total_tokens == sum(max(1, len(t) // 3) for t in texts)
    out = report.as_dict()
    assert out["total_documents"] == 3
    assert out["total_chunks"] == len(texts)
    assert out["estimated_usd"] == round(report.total_tokens * 0.01 / 1e6, 4)
    assert set(out["per_source"]) == {"jira", "hubspot", "gmail"}


# ---------------------------------------------------------------------
# Loader and passes (db fixture: a portolan_test_* database)
# ---------------------------------------------------------------------


def test_load_documents_writes_what_index_note_would(db):
    report = load_documents(db, _docs(), FakeEmbedder(), batch_docs=2, workers=2)
    assert report.notes_written == 3 and not report.stopped_by_budget
    with db.cursor() as cur:
        cur.execute("SELECT slug, type, title, status, body_md FROM notes ORDER BY slug")
        notes = cur.fetchall()
        cur.execute(
            "SELECT note_slug, heading, text, token_count, embedding IS NOT NULL "
            "FROM chunks ORDER BY note_slug, id"
        )
        chunks = cur.fetchall()
    assert [n[0] for n in notes] == ["dsid_1", "dsid_2", "dsid_3"]
    assert all(n[1] == "source" and n[3] == "living" for n in notes)
    expected_chunks = [
        (d.dsid, h, t, e, True)
        for d in sorted(_docs(), key=lambda d: d.dsid)
        for h, t, e in chunk_texts_for(d)
    ]
    assert chunks == expected_chunks
    assert report.chunks_written == len(expected_chunks)
    # Portolan's own incremental indexer sees nothing left to do: the body
    # hashes written match exactly what index_note would have recorded.
    assert index_pending(db, FakeEmbedder()).notes_considered == 0


def test_load_documents_is_resumable(db):
    load_documents(db, _docs()[:1], FakeEmbedder())
    report = load_documents(db, _docs(), FakeEmbedder())
    assert report.documents_skipped_already_indexed == 1
    assert report.notes_written == 2


def test_load_documents_stops_cleanly_on_budget(db):
    class OneBatchThenBroke:
        dim = 1024

        def __init__(self):
            self.inner = FakeEmbedder()
            self.calls = 0

        def embed(self, texts):
            self.calls += 1
            if self.calls > 1:
                raise BudgetExceeded("cap reached")
            return self.inner.embed(texts)

    report = load_documents(
        db, _docs(), OneBatchThenBroke(), batch_docs=1, workers=1, request_size=64
    )
    assert report.stopped_by_budget
    with db.cursor() as cur:
        cur.execute("SELECT count(*) FROM notes")
        written = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM note_index_state")
        assert cur.fetchone()[0] == written
    assert written == report.notes_written == 1


def test_run_passes_end_to_end_on_a_tiny_index(db):
    docs = _docs()
    load_documents(db, docs, FakeEmbedder())
    bm25 = Bm25Index()
    for d in docs:
        bm25.add(d.dsid, d.index_text)
    bm25.finalize()
    by_id = {d.dsid: d.index_text for d in docs}
    questions = [
        {
            "question_id": "q1",
            "question_type": "basic",
            "source_types": ["hubspot"],
            "question": "How many pumps does Acme run?",
            "expected_doc_ids": ["dsid_2"],
            "answer_facts": ["Acme runs 47 pumps."],
        },
        {
            "question_id": "q2",
            "question_type": "semantic",
            "source_types": ["gmail"],
            "question": "What does legal want before the renewal?",
            "expected_doc_ids": ["dsid_3"],
            "answer_facts": ["Legal wants the new MSA first."],
        },
    ]
    out = run_passes(
        db,
        CachingEmbedder(FakeEmbedder()),
        questions,
        bm25=bm25,
        original_text=by_id.get,
        instruction_embedder=CachingEmbedder(FakeEmbedder(), prefix="Represent: "),
    )
    assert set(out["summary"]) == set(out["rows"])
    for name, rows in out["rows"].items():
        assert [r["question_id"] for r in rows] == ["q1", "q2"], name
        for r in rows:
            assert len(r["retrieved_doc_ids"]) <= 10
    fused = {r["question_id"]: r for r in out["rows"]["fused"]}
    assert fused["q1"]["document_recall_pct"] == 100.0
    assert out["rows"]["bm25"][0]["retrieved_doc_ids"][0] == "dsid_2"
    gate = out["citation_gate"]
    assert gate["provenance_hits_checked"] > 0
    # Every cited chunk passes the quote gate's exact rule...
    assert gate["provenance_quote_gate_exact_pct"] == 100.0
    # ...but not always byte-exact: dsid_1's two paragraphs are packed into
    # one chunk without the blank line between them (see _gate_question).
    assert gate["provenance_hits_verbatim_in_original_file"] <= gate["provenance_hits_checked"]
    assert gate["numeral_facts"] == 1
    assert gate["numeral_facts_pass_with_gold_docs"] == 1


def _more_docs() -> list[ErbDoc]:
    extra = [
        ("dsid_4", "jira", "Pump firmware rollback", "Rolled back pump firmware after alarms."),
        ("dsid_5", "hubspot", "Acme renewal forecast", "Acme renewal forecast moved to March."),
        ("dsid_6", "gmail", "Re: pump alarm follow-up", "Legal asked about the alarm contract."),
        ("dsid_7", "fireflies", "Weekly pump review", "Pumps, alarms and renewals were reviewed."),
    ]
    return _docs() + [ErbDoc(d, s, f"{s}/{d}.json", t, c) for d, s, t, c in extra]


def test_exact_passes_reproduce_search_on_a_small_index(db):
    from portolan.bench.erb_search import exact_vector_scan, notes_from_chunks
    from portolan.search import query as q
    from portolan.search.query import search

    docs = _more_docs()
    load_documents(db, docs, FakeEmbedder())
    questions = [
        {
            "question_id": f"q{i}",
            "question_type": "basic",
            "source_types": ["jira"],
            "question": text,
            "expected_doc_ids": ["dsid_1"],
            "answer_facts": [],
        }
        for i, text in enumerate(
            [
                "pump alarm",
                "When does the Acme renewal happen?",
                "What did legal want about the renewal pricing?",
            ]
        )
    ]
    emb = FakeEmbedder()
    vecs = {x["question_id"]: emb.embed([x["question"]])[0] for x in questions}
    scan = exact_vector_scan(db, vecs, 102, batch_rows=3)
    for x in questions:
        pg_rank, _ = q._vector_candidates(db, emb, x["question"], 10)
        assert [h.note_slug for h in notes_from_chunks(scan[x["question_id"]], 30)] == pg_rank
    by_id = {d.dsid: d.index_text for d in docs}
    out = run_exact_passes(
        db, questions, plain=scan, instructed=None, bm25=None, original_text=by_id.get
    )
    for x, row in zip(questions, out["rows"]["fused_exact"], strict=True):
        real = [h.note_slug for h in search(db, emb, x["question"], k=10).hits]
        assert row["retrieved_doc_ids"] == real, x["question"]
    assert out["citation_gate"]["provenance_quote_gate_exact_pct"] == 100.0


def test_ef_sweep_runs_search_at_each_value_on_a_small_index(db):
    from portolan.bench.erb_search import run_ef_sweep
    from portolan.search.query import search

    load_documents(db, _more_docs(), FakeEmbedder())
    questions = [
        {
            "question_id": f"q{i}",
            "question_type": "basic",
            "source_types": ["jira"],
            "question": text,
            "expected_doc_ids": ["dsid_1"],
            "answer_facts": [],
        }
        for i, text in enumerate(["pump alarm", "When does the Acme renewal happen?"])
    ]
    emb = FakeEmbedder()
    rows = run_ef_sweep(db, emb, questions, ef_values=(1, 100))
    assert list(rows) == ["1", "100"]
    for ef, ef_rows in rows.items():
        assert [r["question_id"] for r in ef_rows] == ["q0", "q1"]
        for x, row in zip(questions, ef_rows, strict=True):
            real = search(db, emb, x["question"], k=10, ef_search=int(ef)).hits
            assert row["retrieved_doc_ids"] == [h.note_slug for h in real]
            assert row["seconds"] >= 0


def test_choose_ef_search_takes_the_smallest_value_within_tolerance_of_the_best():
    from portolan.bench.erb_search import choose_ef_search

    dev = {40: 75.64, 100: 75.64, 200: 76.92, 400: 78.21}
    assert choose_ef_search(dev, tolerance=0.5) == 400
    assert choose_ef_search(dev, tolerance=1.3) == 200
    assert choose_ef_search(dev, tolerance=3.0) == 40
    assert choose_ef_search({40: 70.0, 100: 70.0}, tolerance=0.0) == 40


def test_choose_vector_weight_maximises_the_worse_of_erb_dev_and_paraphrase():
    from portolan.bench.erb_search import choose_vector_weight

    # The 2026-09-23 sweep: ERB dev falls as the vector weight rises,
    # paraphrase recall only moves at 1.0.
    erb = {0.2: 76.92, 0.4: 75.64, 0.6: 75.64, 0.8: 75.64, 1.0: 73.08}
    para = {0.2: 46.67, 0.4: 46.67, 0.6: 46.67, 0.8: 46.67, 1.0: 73.33}
    demo = dict.fromkeys(erb, 100.0)
    assert choose_vector_weight(erb, para, demo) == 1.0
    # a weight that loses a demo question is never chosen
    assert choose_vector_weight(erb, para, {**demo, 1.0: 95.0}) == 0.2
    # equal minimums: the higher paraphrase recall, then the higher ERB dev
    assert choose_vector_weight({0.2: 60.0, 1.0: 70.0}, {0.2: 70.0, 1.0: 60.0}, demo) == 0.2
    assert choose_vector_weight({0.2: 70.0, 1.0: 71.0}, {0.2: 60.0, 1.0: 60.0}, demo) == 1.0
    with pytest.raises(ValueError):
        choose_vector_weight(erb, para, dict.fromkeys(erb, 95.0))


def test_latency_summary_is_nearest_rank_in_milliseconds():
    from portolan.bench.erb_search import latency_summary

    rows = [{"seconds": s / 1000} for s in range(1, 21)]  # 1..20 ms
    assert latency_summary(rows) == {"p50_ms": 10.0, "p95_ms": 19.0, "mean_ms": 10.5}
    assert latency_summary([]) == {"p50_ms": 0.0, "p95_ms": 0.0, "mean_ms": 0.0}


def test_run_passes_rejects_unknown_pass(db):
    with pytest.raises(ValueError):
        run_passes(db, FakeEmbedder(), [], bm25=None, original_text=lambda d: None, passes=["x"])


# ---------------------------------------------------------------------
# Smoke test against the real clone (skipped when it is absent)
# ---------------------------------------------------------------------


def test_real_clone_selects_157_questions_when_present():
    root = default_erb_root()
    if not (root / "questions.jsonl").is_file():
        pytest.skip(f"ERB clone not present at {root}")
    uuid_index = load_uuid_index(root)
    sel = select_in_slice_questions(load_questions(root / "questions.jsonl"), uuid_index=uuid_index)
    assert len(sel.questions) == 157
    assert sel.derivations_agree is True
