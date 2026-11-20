"""The reranker bench (`portolan.bench.rerank_eval`): recorded legs plus a
score table must rank exactly as `search()` does with the model itself, or
the bench measures something production never runs."""

from __future__ import annotations

from datetime import date

import pytest

from portolan.bench import rerank_eval as ev
from portolan.search.embed import FakeEmbedder
from portolan.search.index import reindex_all
from portolan.search.query import search
from portolan.search.rerank import FakeReranker

NOTES = [
    ("pump-alarm-alpha", "Pump alarm alpha", "The alpha pump alarm sounded twice at night."),
    ("pump-alarm-bravo", "Pump alarm bravo", "Bravo ward reported one pump alarm."),
    ("pump-recall", "Pump recall", "Corvane pulled its pump from the market after a recall."),
    ("invoice-dispute", "Invoice dispute", "A duplicate charge on the March pump invoice."),
    ("night-alarms", "Night alarms", "Nurses want quieter alarms on night shifts."),
    ("renewal-date", "Renewal date", "The pump service contract renews in October."),
]
QUESTIONS = [
    {"question_id": "q1", "question": "pump alarm at night", "expected_doc_ids": ["night-alarms"]},
    {
        "question_id": "q2",
        "question": "which pump was recalled",
        "expected_doc_ids": ["pump-recall"],
    },
    {"question_id": "q3", "question": "invoice charge", "expected_doc_ids": ["invoice-dispute"]},
]


@pytest.fixture()
def graph(db):
    with db.cursor() as cur:
        for slug, title, body in NOTES:
            cur.execute(
                "INSERT INTO notes (slug, type, title, body_md, updated) "
                "VALUES (%s, 'signal', %s, %s, %s)",
                (slug, title, body, date(2026, 9, 1)),
            )
    reindex_all(db, FakeEmbedder())
    return db


@pytest.mark.parametrize("routing", [False, True])
@pytest.mark.parametrize("pool", [3, 10])
def test_recorded_scores_rank_exactly_as_the_model_does(graph, tmp_path, routing, pool):
    embedder = FakeEmbedder()
    ev.save_legs(tmp_path / "legs.json", ev.record_legs(graph, embedder, QUESTIONS))
    legs = ev.load_legs(tmp_path / "legs.json")

    model = FakeReranker()
    scores: dict[str, dict[str, list[float]]] = {}
    for x in QUESTIONS:
        leg = legs[x["question_id"]][0]
        # search(k=10) reranks max(k, pool) notes
        slugs = ev.pool_slugs(graph, leg, pool=max(pool, 10), fusion_routing=routing)
        scores[x["question_id"]] = ev.score_candidates(graph, model, leg, slugs, {})

    rows = ev.evaluate(
        graph, QUESTIONS, legs, fusion_routing=routing, model="fake", scores=scores, pool=pool
    )
    direct = [
        [
            h.note_slug
            for h in search(
                graph,
                embedder,
                x["question"],
                k=10,
                fusion_routing=routing,
                reranker=FakeReranker(),
                rerank_pool=pool,
            ).hits
        ]
        for x in QUESTIONS
    ]
    assert [r["retrieved_doc_ids"] for r in rows] == direct
    assert all(r["model_seconds"] >= 0 for r in rows)

    plain = ev.evaluate(graph, QUESTIONS, legs, fusion_routing=routing)
    assert [r["retrieved_doc_ids"] for r in plain] == [
        [
            h.note_slug
            for h in search(
                graph, embedder, x["question"], k=10, fusion_routing=routing, reranker=False
            ).hits
        ]
        for x in QUESTIONS
    ]


def test_a_note_never_scored_fails_loudly(graph, tmp_path):
    ev.save_legs(tmp_path / "legs.json", ev.record_legs(graph, FakeEmbedder(), QUESTIONS[:1]))
    legs = ev.load_legs(tmp_path / "legs.json")
    with pytest.raises(KeyError):
        ev.evaluate(
            graph, QUESTIONS[:1], legs, fusion_routing=False, model="m", scores={"q1": {}}, pool=3
        )


def test_choose_keeps_demo_whole_then_maximises_the_worse_of_dev_and_paraphrase():
    rows = {
        "live": {"erb_dev": 73.0, "paraphrase": 73.3, "demo": 100.0},
        "keyword": {"erb_dev": 77.0, "paraphrase": 46.7, "demo": 100.0},
        "rerank": {"erb_dev": 80.0, "paraphrase": 80.0, "demo": 95.0},
        "both": {"erb_dev": 75.0, "paraphrase": 74.0, "demo": 100.0},
        "tie": {"erb_dev": 76.0, "paraphrase": 74.0, "demo": 100.0},
    }
    assert ev.choose(rows) == "tie"  # min 74 twice; then paraphrase ties; then ERB dev
    with pytest.raises(ValueError):
        ev.choose({"x": {"erb_dev": 1.0, "paraphrase": 1.0, "demo": 90.0}})
