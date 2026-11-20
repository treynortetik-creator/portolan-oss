"""`portolan.training.bakeoff_scoring`: matching one labeler's facts to a
reference labeler's by quote span and type, micro precision/recall, and the
two-of-three majority vote the local-teacher bake-off reports
(`01-spec/local-teachers-bakeoff-2026-09-22.md`)."""

from __future__ import annotations

import pytest

from portolan.training.bakeoff_scoring import (
    LabelFact,
    facts_match,
    match_facts,
    overlap_ratio,
    precision_recall,
    vote_facts,
)
from portolan.training.records import TrainingFact


def _f(fact_type: str, start: int, end: int) -> LabelFact:
    return LabelFact(fact_type, (start, end))


def test_overlap_ratio_is_shared_characters_over_the_shorter_span() -> None:
    assert overlap_ratio((0, 10), (5, 25)) == 0.5
    assert overlap_ratio((0, 10), (2, 6)) == 1.0
    assert overlap_ratio((0, 10), (10, 20)) == 0.0
    assert overlap_ratio((3, 3), (0, 10)) == 0.0


def test_match_needs_half_the_shorter_span_and_the_same_type() -> None:
    assert facts_match(_f("risk", 0, 10), _f("risk", 5, 25)) == 0.5
    assert facts_match(_f("risk", 0, 10), _f("risk", 6, 25)) == 0.0
    assert facts_match(_f("risk", 0, 10), _f("request", 0, 10)) == 0.0
    assert facts_match(_f("risk", 0, 10), _f("request", 0, 10), require_type=False) == 1.0


def test_spanless_facts_match_only_each_other_on_the_same_locator() -> None:
    a = LabelFact("pricing", None, "row-3")
    assert facts_match(a, LabelFact("pricing", None, "row-3")) == 1.0
    assert facts_match(a, LabelFact("pricing", None, "row-4")) == 0.0
    assert facts_match(a, _f("pricing", 0, 5)) == 0.0


def test_matching_is_one_to_one_strongest_first() -> None:
    predicted = [_f("risk", 0, 10), _f("risk", 2, 10)]
    reference = [_f("risk", 2, 10)]
    assert match_facts(predicted, reference) == [(0, 0)]  # tie on 1.0 -> lower index
    predicted = [_f("risk", 0, 20), _f("risk", 10, 20)]
    reference = [_f("risk", 10, 20), _f("risk", 0, 9)]
    assert match_facts(predicted, reference) == [(0, 1), (1, 0)]  # both, not greedy (0, 0)


def test_precision_recall_is_micro_averaged_and_counts_missing_documents() -> None:
    predicted = {"d1": [_f("risk", 0, 10), _f("praise", 50, 60)], "d3": [_f("risk", 0, 4)]}
    reference = {"d1": [_f("risk", 0, 10)], "d2": [_f("request", 0, 10)]}
    score = precision_recall(predicted, reference, documents=["d1", "d2", "d3"])
    assert (score.matched, score.predicted, score.reference) == (1, 3, 2)
    assert score.precision == pytest.approx(1 / 3)
    assert score.recall == pytest.approx(1 / 2)
    assert score.f1 == pytest.approx(2 / 5)
    empty = precision_recall({}, {}, documents=["d1"])
    assert empty.precision is None and empty.recall is None and empty.f1 is None


def test_vote_keeps_facts_two_labelers_reached() -> None:
    a = [_f("risk", 0, 10), _f("praise", 40, 50)]
    b = [_f("risk", 2, 10), _f("request", 40, 50)]
    c = [_f("risk", 0, 8), _f("praise", 42, 50), _f("pricing", 90, 99)]
    voted = vote_facts([a, b, c])
    assert voted == [_f("risk", 0, 10), _f("praise", 40, 50)]
    assert vote_facts([a, b, c], min_support=3) == [_f("risk", 0, 10)]


def test_one_labeler_cannot_vote_twice_for_a_cluster() -> None:
    a = [_f("risk", 0, 10), _f("risk", 1, 10)]
    assert vote_facts([a, []]) == []


def test_label_fact_from_a_training_fact() -> None:
    fact = TrainingFact(
        fact_type="risk", claim="c", quote="q", quote_span=(4, 5), locator="L2", entities=[]
    )
    assert LabelFact.from_training_fact(fact) == LabelFact("risk", (4, 5), "L2", "c")
