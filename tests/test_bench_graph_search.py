"""The graph retrieval bench's question sets and its regression floor on the demo graph
(`portolan.bench.graph_search`, write-up `01-spec/bench-graph-retrieval-2026-09-23.md`).

One seeded demo graph (the fake extractor over `corpus/out`, FakeEmbedder: word overlap
only, so this pins the graph modes and the fusion, not a semantic model). Measured
2026-09-25 with this embedder and the proportional title boost: on the 26 multi-hop
questions mode "off" meets every answer requirement on 19 (73.1%) and entity mode on
21 (80.8%); on the 35 single-hop questions both find 25 (71.4%).
"""

from __future__ import annotations

from datetime import date

import pytest

from portolan.bench.demo_search import seed_demo_graph
from portolan.bench.graph_search import (
    MULTI_HOP_QUESTIONS,
    GraphQuestion,
    aggregate_rows,
    evaluate,
    score_hits,
    single_hop_questions,
    window_questions,
)
from portolan.search.embed import FakeEmbedder


def test_score_hits_counts_requirements_and_leaves_bridges_out_of_the_answer_numbers():
    q = GraphQuestion(
        "q", "?", "hierarchy", (("acct",), ("sig-a", "action-a"), ("sig-b",))
    )  # fmt: skip
    scored = score_hits(["action-a", "x", "acct"], q, k=10, entity_slugs=frozenset({"acct"}))
    assert scored["requirement_recall_pct"] == pytest.approx(200 / 3)
    assert scored["supported"] is False
    assert scored["answer_recall_pct"] == 50.0
    assert scored["first_gold_rank"] == 1
    assert scored["unmet"] == [["sig-b"]]
    # The same retrieval cut at k=2 loses the bridge.
    assert score_hits(["action-a", "x", "acct"], q, k=2)["requirement_recall_pct"] == (
        pytest.approx(100 / 3)
    )


def test_aggregate_rows_reports_support_and_latency():
    rows = [
        {"requirement_recall_pct": 100.0, "supported": True, "answer_recall_pct": 100.0,
         "answer_supported": True, "first_gold_rank": 1, "seconds": 0.1, "pool_size": 4},
        {"requirement_recall_pct": 50.0, "supported": False, "answer_recall_pct": 0.0,
         "answer_supported": False, "first_gold_rank": None, "seconds": 0.3, "pool_size": 0},
    ]  # fmt: skip
    out = aggregate_rows(rows)
    assert out["requirement_recall_pct"] == 75.0
    assert out["supported_pct"] == 50.0
    assert out["mrr_at_10"] == 0.5
    assert out["p50_ms"] == 200.0


def test_graph_modes_on_the_demo_graph(db, tmp_path):
    embedder = FakeEmbedder()
    seed_demo_graph(db, tmp_path, embedder)
    with db.cursor() as cur:
        cur.execute("SELECT slug FROM notes")
        slugs = {row[0] for row in cur.fetchall()}

    multi = MULTI_HOP_QUESTIONS
    assert len(multi) >= 20
    assert len({q.qid for q in multi}) == len(multi)
    assert all(len(q.requirements) >= 2 for q in multi)
    assert {s for q in multi for req in q.requirements for s in req} <= slugs

    single = single_hop_questions()
    off_multi = evaluate(db, embedder, multi, mode="off")["overall"]
    entity_multi = evaluate(db, embedder, multi, mode="entity")["overall"]
    # Query routing is on by default (`search_fusion_routing`). 80.77 before a keyword
    # rank counted by the note's share of the question (`KEYWORD_SHARE_FULL`).
    assert off_multi["answer_supported_pct"] == 92.31
    # Entity mode was 84.62 before that change and is 88.46 after it: better than it
    # was, no longer ahead of plain search on this embedder (word overlap only).
    assert entity_multi["answer_supported_pct"] >= 84.62
    assert entity_multi["answer_recall_pct"] > off_multi["answer_recall_pct"]
    off_single = evaluate(db, embedder, single, mode="off")["overall"]
    entity_single = evaluate(db, embedder, single, mode="entity")["overall"]
    assert entity_single["requirement_recall_pct"] == off_single["requirement_recall_pct"]

    window = window_questions(db, since=date(2026, 8, 1))
    assert window and all(q.window for q in window)
    plain = evaluate(db, embedder, window, mode="off")["overall"]
    windowed = evaluate(db, embedder, window, mode="off", use_window=True)["overall"]
    assert windowed["window_precision_pct"] == 100.0
    assert plain["window_precision_pct"] < 50.0
    assert windowed["requirement_recall_pct"] == 100.0
