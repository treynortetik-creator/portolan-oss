"""Tests for `portolan.training.agreement` -- the silver-labeling
agreement gate (silver-labels build instructions item 2 and item 6: "same
type + overlapping quote kept; type mismatch dropped; quote not in text
dropped"). No network: every fact is a real `portolan.extract.schema.Fact`
constructed directly, so these tests exercise the same field validation a
live extraction result would already have passed.
"""

from __future__ import annotations

from portolan.extract.schema import Fact
from portolan.training.agreement import evaluate_document, match_facts, quotes_overlap

_DOC_TEXT = (
    "[L1] alice: my pump keeps beeping at night and I can't figure out why\n"
    "[L2] bob: that sounds like the battery alarm, have you checked the charger\n"
    "[L3] alice: I'll send you a screenshot of the error code tomorrow\n"
)


def _fact(
    fact_id: str,
    fact_type: str,
    claim: str,
    quote: str | None,
    locator: str = "L1",
    sentiment: int = 0,
    confidence: float = 0.8,
) -> Fact:
    return Fact(
        id=fact_id,
        type=fact_type,
        claim=claim,
        locator=locator,
        quote=quote,
        sentiment=sentiment,
        confidence=confidence,
    )


# --------------------------------------------------------------------------
# quotes_overlap
# --------------------------------------------------------------------------


def test_quotes_overlap_exact_match() -> None:
    assert quotes_overlap("my pump keeps beeping", "my pump keeps beeping") is True


def test_quotes_overlap_substring_either_direction() -> None:
    assert quotes_overlap("my pump keeps beeping at night", "pump keeps beeping") is True
    assert quotes_overlap("pump keeps beeping", "my pump keeps beeping at night") is True


def test_quotes_overlap_glyph_folded() -> None:
    # curly quotes / em dash on one side, plain ASCII on the other
    assert quotes_overlap("I can’t figure it out", "I can't figure it out") is True


def test_quotes_overlap_false_when_unrelated() -> None:
    assert quotes_overlap("my pump keeps beeping", "the charger is broken") is False


def test_quotes_overlap_false_when_either_missing() -> None:
    assert quotes_overlap(None, "my pump keeps beeping") is False
    assert quotes_overlap("my pump keeps beeping", None) is False
    assert quotes_overlap("", "my pump keeps beeping") is False


# --------------------------------------------------------------------------
# match_facts
# --------------------------------------------------------------------------


def test_match_facts_same_type_overlapping_quote_kept() -> None:
    fact_a = _fact("a1", "complaint", "pump beeps", "my pump keeps beeping at night")
    fact_b = _fact("b1", "complaint", "device alarms", "pump keeps beeping")
    pairs = match_facts([fact_a], [fact_b])
    assert pairs == [(fact_a, fact_b)]


def test_match_facts_type_mismatch_not_matched() -> None:
    fact_a = _fact("a1", "complaint", "pump beeps", "my pump keeps beeping at night")
    fact_b = _fact("b1", "request", "wants a fix", "my pump keeps beeping at night")
    assert match_facts([fact_a], [fact_b]) == []


def test_match_facts_quote_disagreement_not_matched() -> None:
    fact_a = _fact("a1", "complaint", "pump beeps", "my pump keeps beeping at night")
    fact_b = _fact("b1", "complaint", "charger issue", "the charger is broken")
    assert match_facts([fact_a], [fact_b]) == []


def test_match_facts_does_not_double_match_one_b_fact() -> None:
    shared_quote = "my pump keeps beeping at night"
    fact_a1 = _fact("a1", "complaint", "pump beeps 1", shared_quote)
    fact_a2 = _fact("a2", "complaint", "pump beeps 2", shared_quote)
    fact_b1 = _fact("b1", "complaint", "device alarms", shared_quote)
    pairs = match_facts([fact_a1, fact_a2], [fact_b1])
    assert pairs == [(fact_a1, fact_b1)]


# --------------------------------------------------------------------------
# evaluate_document
# --------------------------------------------------------------------------


def test_evaluate_document_kept_fact_both_teachers_agree_and_quote_in_text() -> None:
    fact_a = _fact(
        "a1",
        "complaint",
        "customer reports beeping pump",
        "my pump keeps beeping at night",
        locator="L1",
    )
    fact_b = _fact(
        "b1",
        "complaint",
        "pump beeping issue",
        "pump keeps beeping at night",
        locator="L1",
    )
    outcome = evaluate_document([fact_a], [fact_b], _DOC_TEXT)
    assert len(outcome.kept) == 1
    kept = outcome.kept[0]
    assert kept["fact_type"] == "complaint"
    assert kept["claim"] == "customer reports beeping pump"
    assert kept["locator"] == "L1"
    assert outcome.dropped_disagreement == 0
    assert outcome.dropped_quote_gate == 0
    assert outcome.agreement == 1.0
    assert outcome.is_negative is False


def test_evaluate_document_type_mismatch_dropped_as_disagreement() -> None:
    fact_a = _fact("a1", "complaint", "pump beeps", "my pump keeps beeping at night")
    fact_b = _fact("b1", "request", "wants callback", "my pump keeps beeping at night")
    outcome = evaluate_document([fact_a], [fact_b], _DOC_TEXT)
    assert outcome.kept == []
    assert outcome.dropped_disagreement == 2
    assert outcome.dropped_quote_gate == 0
    assert outcome.is_negative is True


def test_evaluate_document_quote_not_in_document_text_dropped() -> None:
    # Both teachers agree on type and their quotes overlap each other,
    # but neither quote is actually present in the real document text --
    # must be dropped by the final quote_is_supported check, not kept.
    fabricated = "the warranty was extended by six full years"
    fact_a = _fact("a1", "commitment", "warranty extended", fabricated)
    fact_b = _fact("b1", "commitment", "extended warranty", fabricated)
    outcome = evaluate_document([fact_a], [fact_b], _DOC_TEXT)
    assert outcome.kept == []
    assert outcome.dropped_disagreement == 0
    assert outcome.dropped_quote_gate == 1
    assert outcome.is_negative is True


def test_evaluate_document_both_teachers_zero_facts_is_negative_full_agreement() -> None:
    outcome = evaluate_document([], [], _DOC_TEXT)
    assert outcome.kept == []
    assert outcome.dropped_disagreement == 0
    assert outcome.dropped_quote_gate == 0
    assert outcome.agreement == 1.0
    assert outcome.is_negative is True


def test_evaluate_document_partial_agreement_ratio() -> None:
    matched_quote = "my pump keeps beeping at night"
    fact_a1 = _fact("a1", "complaint", "pump beeps", matched_quote)
    fact_a2 = _fact("a2", "request", "wants a fix", "I'll send you a screenshot", locator="L3")
    fact_b1 = _fact("b1", "complaint", "device alarms", "pump keeps beeping")
    outcome = evaluate_document([fact_a1, fact_a2], [fact_b1], _DOC_TEXT)
    assert len(outcome.kept) == 1
    assert outcome.dropped_disagreement == 1  # fact_a2 has no partner
    # denom = max(2, 1) = 2; kept = 1 -> agreement 0.5
    assert outcome.agreement == 0.5
