"""The ERB exam's weight tuning: a fixed dev/held-out split, fusion over the
recorded legs, the pre-registered choice rule, and the gate's invented-number
check. Pure, no database, no ERB data."""

from __future__ import annotations

from pathlib import Path

from portolan.bench.erb import fact_numeral_gate, invent_number, split_dev_heldout
from portolan.bench.erb_search import (
    ChunkHit,
    fuse,
    fusion_rows,
    load_scan,
    save_scan,
    tune_fusion_weights,
)
from portolan.search import query as q

IDS = [f"qst_{i:04d}" for i in range(157)]


def test_split_is_fixed_disjoint_and_halved():
    dev, held = split_dev_heldout(IDS)
    assert (len(dev), len(held)) == (78, 79)
    assert set(dev).isdisjoint(held) and set(dev) | set(held) == set(IDS)
    assert split_dev_heldout(reversed(IDS)) == (dev, held)
    assert split_dev_heldout(IDS, seed="another seed") != (dev, held)


def test_fuse_is_search_arithmetic_with_title_boost():
    kw, vec = ["a", "b"], ["b", "c"]
    plain = fuse(kw, vec, {}, fts_weight=1.0, vector_weight=1.0)
    assert plain == ["b", "a", "c"]
    # a full title match adds TITLE_BOOST_ALL x the top RRF score
    boosted = fuse(kw, vec, {"a": q.TITLE_BOOST_ALL}, fts_weight=1.0, vector_weight=1.0)
    assert boosted == ["a", "b", "c"]
    # weights default to search()'s own
    assert fuse(kw, vec, {}) == fuse(
        kw, vec, {}, fts_weight=q.FTS_WEIGHT, vector_weight=q.VECTOR_WEIGHT
    )


def _question(qid: str, gold: str) -> dict:
    return {
        "question_id": qid,
        "question_type": "basic",
        "source_types": ["jira"],
        "expected_doc_ids": [gold],
    }


def test_tuning_chooses_on_dev_by_the_fixed_rule():
    # gold is the keyword leg's 11th hit and absent from the vector leg:
    # only a light vector weight lets it into the top 10
    vector = [f"v{i}" for i in range(10)]
    keyword = [*vector, "gold"]
    legs = {"q1": {"keyword": keyword, "vector": vector, "title_levels": {}}}
    questions = [_question("q1", "v0")]
    tuned = tune_fusion_weights(questions, legs, grid=(0.5, 1.0, 2.0))
    # every weight finds v0 first: ties go to recall, then MRR, then 1:1
    assert tuned["chosen"]["vector_weight"] == 1.0
    rows = fusion_rows(questions, legs, fts_weight=1.0, vector_weight=1.0)
    assert rows[0]["first_gold_rank"] == 1

    # gold is the keyword leg's second hit: rank 2 at 0.5, 3 at 1.0 (ties go
    # to the lower slug), pushed out of the top 10 at 2.0
    legs = {"q1": {"keyword": ["k1", "gold"], "vector": vector, "title_levels": {}}}
    tuned = tune_fusion_weights([_question("q1", "gold")], legs, grid=(0.5, 1.0, 2.0))
    assert tuned["chosen"]["vector_weight"] == 0.5
    assert [r["dev"]["ours_mrr_at_10"] for r in tuned["grid"]] == [0.5, 0.3333, 0.0]


def test_scan_cache_round_trips(tmp_path: Path):
    scan = {"q1": [ChunkHit("n1", 7, 0.25), ChunkHit("n2", 9, 0.5)], "q1#instruction": []}
    save_scan(tmp_path / "scan.json", scan)
    assert load_scan(tmp_path / "scan.json") == scan


def test_invented_number_changes_one_checkable_number():
    assert invent_number("Median latency is 250ms.") == "Median latency is 251ms."
    assert invent_number("The plan costs $5,000 a month.") == "The plan costs $6,000 a month."
    assert invent_number("Filed on 2026-03-12 as SUP-100.") is None
    fact = "Median latency is 250ms."
    assert fact_numeral_gate(fact, ["p50: 250 ms"]) == (True, [])
    assert fact_numeral_gate(invent_number(fact), ["p50: 250 ms"]) == (False, ["251"])
