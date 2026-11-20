"""Tests for `portolan.extract.quote_gate`, the containment logic moved
out of `portolan.bench.extraction` 2026-09-19 ("mechanical quote gate to
production", `01-spec/extraction-final-2026-09-18.md`'s production pick).
`tests/test_bench_extraction.py` keeps exercising the exact same code
(re-exported into that module unchanged, byte for byte); these tests
import directly from the new home instead, and add coverage for
`quote_is_supported`, the public gate `portolan.pipeline` calls that the
bench itself never needed (it stops at the bare booleans/scores and
builds its own five-way verdict on top)."""

from __future__ import annotations

from portolan.extract.quote_gate import (
    FUZZY_QUOTE_THRESHOLD,
    _fuzzy_quote_containment,
    _quote_supported_by_text,
    quote_is_supported,
)

_SOURCE_TEXT = (
    "[00:01:00] Sam Okafor: We can provide packing information and a "
    "delivery schedule.\n"
    "[00:01:05] Derek Hollis: Include the dimensions and weight of the "
    "shipping cases.\n"
    "[00:01:15] Sam Okafor: We serve Ashgrove Medical Center and shipped "
    "40 pumps in March 2026.\n"
)


def test_fuzzy_quote_threshold_is_075() -> None:
    assert FUZZY_QUOTE_THRESHOLD == 0.75


def test_quote_supported_by_text_exact_match() -> None:
    quote = "Include the dimensions and weight of the shipping cases."
    assert _quote_supported_by_text(quote, _SOURCE_TEXT) is True


def test_quote_supported_by_text_glyph_folds_curly_apostrophes() -> None:
    text_with_curly = "I’ll get back to you once I’ve checked with the team.\n"
    quote_with_straight = "I'll get back to you once I've checked with the team."
    assert _quote_supported_by_text(quote_with_straight, text_with_curly) is True


def test_quote_supported_by_text_none_when_no_quote_given() -> None:
    assert _quote_supported_by_text(None, _SOURCE_TEXT) is None
    assert _quote_supported_by_text("   ", _SOURCE_TEXT) is None


def test_quote_supported_by_text_false_for_unrelated_words() -> None:
    quote = "We are extremely satisfied with the onboarding experience overall."
    assert _quote_supported_by_text(quote, _SOURCE_TEXT) is False


def test_fuzzy_quote_containment_one_word_changed_clears_threshold() -> None:
    text = "I’ll get back to you once I’ve checked with the team.\n"
    quote = "I'll get back to you once I've verified with the team."
    assert _quote_supported_by_text(quote, text) is False
    assert _fuzzy_quote_containment(quote, text) >= 0.75


def test_fuzzy_quote_containment_unrelated_text_stays_low() -> None:
    quote = "We are extremely satisfied with the onboarding experience overall."
    assert _fuzzy_quote_containment(quote, _SOURCE_TEXT) < 0.75


# --------------------------------------------------------------------
# quote_is_supported -- the public gate, not part of the pre-move module
# --------------------------------------------------------------------


def test_quote_is_supported_true_for_an_exact_quote() -> None:
    quote = "Include the dimensions and weight of the shipping cases."
    assert quote_is_supported(quote, _SOURCE_TEXT) is True


def test_quote_is_supported_true_for_a_near_verbatim_quote() -> None:
    text = "I’ll get back to you once I’ve checked with the team.\n"
    quote = "I'll get back to you once I've verified with the team."
    assert quote_is_supported(quote, text) is True


def test_quote_is_supported_false_for_unrelated_words() -> None:
    quote = "We are extremely satisfied with the onboarding experience overall."
    assert quote_is_supported(quote, _SOURCE_TEXT) is False


def test_quote_is_supported_false_for_missing_or_blank_quote() -> None:
    assert quote_is_supported(None, _SOURCE_TEXT) is False
    assert quote_is_supported("", _SOURCE_TEXT) is False
    assert quote_is_supported("   ", _SOURCE_TEXT) is False


def test_quote_is_supported_false_for_empty_source_text() -> None:
    assert quote_is_supported("anything", "") is False
