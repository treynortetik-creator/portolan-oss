"""Tests for `portolan.training.quote_span.resolve_quote_span`
(feature/labeling-exact-quotes) -- the exact-span resolver a kept fact's
quote must clear before it's trusted as something a student model could
learn to copy verbatim. Pure functions, no network, no corpus fixture."""

from __future__ import annotations

from portolan.training.quote_span import resolve_quote_span


def _assert_resolves(quote: str, text: str, *, tier: str) -> None:
    resolved = resolve_quote_span(quote, text)
    assert resolved is not None, f"expected {quote!r} to resolve against {text!r}"
    start, end = resolved.span
    assert text[start:end] == resolved.quote
    assert resolved.tier == tier


# --------------------------------------------------------------------------
# exact
# --------------------------------------------------------------------------


def test_exact_substring_resolves_at_tier_exact() -> None:
    text = "the pump beeped three times and shut off"
    _assert_resolves("the pump beeped three times", text, tier="exact")


def test_exact_tier_returns_the_first_occurrence() -> None:
    text = "beep beep beep"
    resolved = resolve_quote_span("beep", text)
    assert resolved is not None
    assert resolved.span == (0, 4)


# --------------------------------------------------------------------------
# normalized: whitespace, case, curly quotes/dashes, NFKC
# --------------------------------------------------------------------------


def test_whitespace_run_collapses_to_match() -> None:
    text = "the pump   beeped\nthree times"
    _assert_resolves("the pump beeped three times", text, tier="normalized")


def test_case_difference_resolves_to_original_case_span() -> None:
    text = "The Pump Beeped Three Times"
    resolved = resolve_quote_span("the pump beeped three times", text)
    assert resolved is not None
    assert resolved.tier == "normalized"
    # The stored quote is the EXACT original-text substring, not the
    # caller's lowercase spelling.
    assert resolved.quote == "The Pump Beeped Three Times"


def test_curly_quotes_fold_to_straight() -> None:
    text = "she said “the device malfunctioned” during use"
    _assert_resolves('she said "the device malfunctioned"', text, tier="normalized")


def test_em_dash_folds_to_hyphen() -> None:
    text = "the device failed — twice"
    _assert_resolves("the device failed - twice", text, tier="normalized")


def test_ellipsis_glyph_folds_to_three_dots_for_containment() -> None:
    # Not a stitched quote (no literal "..." split intended here) --
    # just the ellipsis GLYPH normalizing the same way on both sides.
    text = "wait… what happened"
    _assert_resolves("wait... what happened", text, tier="normalized")


def test_nfkc_fullwidth_digits_normalize() -> None:
    text = "order number １２３ confirmed"  # fullwidth "123"
    _assert_resolves("order number 123 confirmed", text, tier="normalized")


# --------------------------------------------------------------------------
# mime: quoted-printable artifacts
# --------------------------------------------------------------------------


def test_mime_soft_space_20_resolves() -> None:
    text = "made by=20 organizations such as Enron"
    _assert_resolves("made by organizations such as Enron", text, tier="mime")


def test_mime_soft_line_break_joins_with_no_gap() -> None:
    text = "the mig=\nration plan is ready"
    _assert_resolves("the migration plan is ready", text, tier="mime")


def test_mime_crlf_soft_break_joins_with_no_gap() -> None:
    text = "the mig=\r\nration plan is ready"
    _assert_resolves("the migration plan is ready", text, tier="mime")


def test_mime_tab_and_cr_codes_collapse_as_whitespace() -> None:
    text = "shipped=09 on time=0D confirmed"
    _assert_resolves("shipped on time confirmed", text, tier="mime")


def test_mime_tier_only_tried_after_normalized_fails() -> None:
    # No MIME artifacts at all -- still resolves, just at "normalized".
    text = "the plan   is ready"
    _assert_resolves("the plan is ready", text, tier="normalized")


# --------------------------------------------------------------------------
# stitched quotes ("..." splices two spans) -- first segment only
# --------------------------------------------------------------------------


def test_stitched_quote_resolves_to_first_segment_only() -> None:
    text = "Shipper shall pay Transporter on or before the tenth day after invoice"
    resolved = resolve_quote_span(
        "Shipper shall pay Transporter ...on or before the tenth day", text
    )
    assert resolved is not None
    assert resolved.tier == "stitched:exact"
    assert resolved.quote == "Shipper shall pay Transporter"
    start, end = resolved.span
    assert text[start:end] == resolved.quote


def test_stitched_quote_first_segment_can_itself_need_normalization() -> None:
    text = "The   Device Failed during use ... and was later replaced"
    resolved = resolve_quote_span("the device failed during use...something else entirely", text)
    assert resolved is not None
    assert resolved.tier == "stitched:normalized"
    assert resolved.quote == "The   Device Failed during use"


def test_stitched_quote_with_ellipsis_glyph_splits_too() -> None:
    text = "first part of the sentence stands alone. second part is unrelated"
    resolved = resolve_quote_span("first part of the sentence stands alone… unrelated tail", text)
    assert resolved is not None
    assert resolved.tier.startswith("stitched:")
    assert resolved.quote == "first part of the sentence stands alone"


def test_stitched_quote_dropped_when_first_segment_unresolvable() -> None:
    text = "completely unrelated document content here"
    resolved = resolve_quote_span("nothing like this exists ...unrelated tail", text)
    assert resolved is None


# --------------------------------------------------------------------------
# no resolution at all -> None
# --------------------------------------------------------------------------


def test_paraphrase_with_substituted_word_does_not_resolve() -> None:
    text = "the pump beeped three times and shut off"
    resolved = resolve_quote_span("the pump chimed three times and shut off", text)
    assert resolved is None


def test_empty_quote_returns_none() -> None:
    assert resolve_quote_span("", "some text") is None
    assert resolve_quote_span("   ", "some text") is None


def test_none_quote_returns_none() -> None:
    assert resolve_quote_span(None, "some text") is None


def test_empty_text_returns_none() -> None:
    assert resolve_quote_span("a quote", "") is None


def test_quote_longer_than_text_returns_none() -> None:
    assert resolve_quote_span("a very long quote indeed", "short") is None
