"""Regression floor for search on the demo graph (`portolan.bench.
demo_search`): 20 questions a rep would ask, over the graph the web tests
seed, with `FakeEmbedder` (word overlap only, so this pins the keyword leg
and the fusion, not a semantic model). Measured 2026-09-22 with this
embedder: main before the BM25 keyword leg found 18 of 20 in the top 10
(MRR@10 0.658); the BM25 leg at weights 1.0 / 0.2 finds 20 of 20 (MRR@10
0.875). 01-spec/bench-erb-search-2026-09-22.md, "Demo graph check".
At equal weights (2026-09-23, "Fusion weight") it still finds 20 of 20,
MRR@10 0.7375: this embedder's word-hash vector leg now counts as much as
the keyword leg. With local bge-large the same graph scores MRR@10 0.883.

The paraphrase set is scored with a real embedding model by the bench
script (write-up, "Production path"); here only its premise is pinned:
its gold notes exist and share few words with the questions.
"""

from __future__ import annotations

import pytest

from portolan.bench.demo_search import (
    DEMO_QUESTIONS,
    PARAPHRASE_QUESTIONS,
    evaluate_demo_search,
    fusion_weights,
    lexeme_overlap,
    seed_demo_graph,
)
from portolan.search import query as q
from portolan.search.embed import FakeEmbedder

MRR_FLOOR = 0.70
# Measured 2026-09-23: 0.075 for the paraphrases, 0.750 for DEMO_QUESTIONS.
PARAPHRASE_MEAN_OVERLAP_CEILING = 0.15


def test_every_demo_question_finds_its_note_in_the_top_ten(db, tmp_path):
    embedder = FakeEmbedder()
    seed_demo_graph(db, tmp_path, embedder)
    with db.cursor() as cur:
        cur.execute("SELECT slug FROM notes")
        slugs = {row[0] for row in cur.fetchall()}
    assert {s for q in DEMO_QUESTIONS for s in q.expected} <= slugs

    out = evaluate_demo_search(db, embedder)

    missed = [r["question_id"] for r in out["rows"] if r["first_gold_rank"] is None]
    assert missed == []
    assert out["overall"]["average_recall_pct"] == 100.0
    assert out["overall"]["ours_mrr_at_10"] >= MRR_FLOOR


def test_paraphrase_questions_ask_about_the_graph_in_other_words(db, tmp_path):
    seed_demo_graph(db, tmp_path, FakeEmbedder())
    with db.cursor() as cur:
        cur.execute("SELECT slug FROM notes")
        slugs = {row[0] for row in cur.fetchall()}
    assert {s for x in PARAPHRASE_QUESTIONS for s in x.expected} <= slugs
    assert len({x.qid for x in PARAPHRASE_QUESTIONS}) == 15

    def mean_overlap(questions) -> float:
        return sum(lexeme_overlap(db, x) for x in questions) / len(questions)

    assert mean_overlap(PARAPHRASE_QUESTIONS) <= PARAPHRASE_MEAN_OVERLAP_CEILING
    assert mean_overlap(DEMO_QUESTIONS) >= 0.5


def test_fusion_weights_sets_and_always_restores_the_rrf_weights():
    saved = (q.FTS_WEIGHT, q.VECTOR_WEIGHT)
    with pytest.raises(RuntimeError), fusion_weights(1.0, 1.0):
        assert (q.FTS_WEIGHT, q.VECTOR_WEIGHT) == (1.0, 1.0)
        raise RuntimeError
    assert (q.FTS_WEIGHT, q.VECTOR_WEIGHT) == saved
