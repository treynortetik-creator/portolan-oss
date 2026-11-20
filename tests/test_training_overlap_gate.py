"""Tests for `portolan.training.overlap_gate` -- the mechanical keep gate
for `--mode single` (coordinator course-correction, 2026-09-21, final
design item 2, then a scope correction the same day: numerals/names must
be grounded in the FULL record `text` -- a name or date from elsewhere in
the same document is a correct label, not a fabrication -- and
content-word coverage passes on either the local quote-plus-sentence
score (>= `threshold`) or the full-document score (>= `full_text_
threshold`), whichever is more lenient). No network, no corpus: every
case here is a plain string.
"""

from __future__ import annotations

from portolan.training.overlap_gate import (
    CalibrationCase,
    choose_threshold,
    content_word_score,
    evaluate_threshold,
    extract_surrounding_sentence,
    numerals_and_names_covered,
    overlap_gate_reason,
    passes_overlap_gate,
    sweep_thresholds,
    tokenize_claim,
)

_DOC_TEXT = (
    "[L1] the customer called about a late shipment. The pump beeped three times and shut off "
    "on 2026-08-04, according to Dana Whitlock. No adverse effects were reported."
)


# --------------------------------------------------------------------------
# tokenize_claim
# --------------------------------------------------------------------------


def test_tokenize_claim_splits_content_numerals_names() -> None:
    tokens = tokenize_claim("A rep said Whitlock reported the pump beeped 3 times on 2026-08-04.")
    # "Whitlock" is a real name, not claim-initial, not a role word.
    assert "whitlock" in tokens.names
    assert "3" in tokens.numerals
    assert "2026-08-04" in tokens.numerals
    assert "pump" in tokens.content
    assert "beeped" in tokens.content
    # stopwords removed
    assert "the" not in tokens.content
    assert "on" not in tokens.content


def test_tokenize_claim_empty_claim() -> None:
    tokens = tokenize_claim("")
    assert tokens.content == ()
    assert tokens.numerals == ()
    assert tokens.names == ()


# --------------------------------------------------------------------------
# Name-rule normalization (fix/overlap-gate-paraphrase, 2026-09-22): a
# capitalized claim-initial word, a generic role word (possessive-
# stripped), and a possessive name form must all compare/exclude
# correctly -- the real Kimi K2 finding this fix was written against.
# --------------------------------------------------------------------------


def test_tokenize_claim_excludes_claim_initial_capitalized_word() -> None:
    # "Virtual" opens the claim -- excluded even though it's capitalized
    # and not a stopword; "Onboarding" and "Suite" (word 2+ of the same
    # proper-noun phrase) still count.
    tokens = tokenize_claim("Virtual Onboarding Suite is priced at $299 per month.")
    assert "virtual" not in tokens.names
    assert "onboarding" in tokens.names
    assert "suite" in tokens.names


def test_tokenize_claim_excludes_generic_role_words() -> None:
    tokens = tokenize_claim("It seems the Customer talked to a Rep about the Vendor's pricing.")
    assert "customer" not in tokens.names
    assert "rep" not in tokens.names
    assert "vendor" not in tokens.names


def test_tokenize_claim_strips_possessive_before_role_word_check() -> None:
    # "Customer's" (mid-claim, so not exempted by claim-initial position)
    # must still be recognized as the role word "customer" after
    # stripping the possessive, not kept as a distinct "name".
    tokens = tokenize_claim("Regarding pricing, Customer's turnover rate is high.")
    assert "customer" not in tokens.names
    assert "customer's" not in tokens.names


def test_numerals_and_names_covered_matches_possessive_either_direction() -> None:
    # claim states the name bare, document uses the possessive form.
    _numerals_ok, names_ok = numerals_and_names_covered(
        "Regarding onboarding, SmartOnboard pricing is too high.",
        "the call covered SmartOnboard's pricing plan",
    )
    assert names_ok is True
    # and the reverse: claim uses the possessive, document states it bare.
    _numerals_ok, names_ok = numerals_and_names_covered(
        "Regarding onboarding, SmartOnboard's pricing is too high.",
        "the call covered SmartOnboard pricing plans",
    )
    assert names_ok is True


def test_numerals_and_names_covered_still_catches_fabricated_name_mid_claim() -> None:
    # a role word doesn't exempt a REAL name sitting next to it.
    _numerals_ok, names_ok = numerals_and_names_covered(
        "Regarding the account, Rep Kenji Mori confirmed the issue.",
        "the pump beeped three times and shut off",
    )
    assert names_ok is False


# --------------------------------------------------------------------------
# Numeral normalization (fix/overlap-gate-paraphrase, 2026-09-22): a
# thousands-separator comma must not split a numeral into tokens that no
# longer match the same number spelled without one.
# --------------------------------------------------------------------------


def test_numerals_and_names_covered_matches_thousands_comma_either_direction() -> None:
    numerals_ok, _names_ok = numerals_and_names_covered(
        "The deal is worth $1,200 annually.", "the account is valued at $1200 per year"
    )
    assert numerals_ok is True
    numerals_ok, _names_ok = numerals_and_names_covered(
        "The deal is worth $1200 annually.", "the account is valued at $1,200 per year"
    )
    assert numerals_ok is True


def test_tokenize_claim_normalizes_thousands_comma_numeral() -> None:
    tokens = tokenize_claim("Pricing came to $1,200 total.")
    assert "1200" in tokens.numerals
    assert "1" not in tokens.numerals
    assert "200" not in tokens.numerals


def test_ordinary_sentence_comma_is_not_treated_as_a_thousands_separator() -> None:
    # no digit immediately before the comma -- must be left alone.
    tokens = tokenize_claim("The pricing, 5 tiers in total, was confusing.")
    assert "5" in tokens.numerals


# --------------------------------------------------------------------------
# extract_surrounding_sentence
# --------------------------------------------------------------------------


def test_extract_surrounding_sentence_finds_best_overlapping_sentence() -> None:
    result = extract_surrounding_sentence("pump beeped three times and shut off", _DOC_TEXT)
    assert "pump beeped three times" in result


def test_extract_surrounding_sentence_empty_inputs() -> None:
    assert extract_surrounding_sentence("", _DOC_TEXT) == ""
    assert extract_surrounding_sentence("quote", "") == "quote"


# --------------------------------------------------------------------------
# content_word_score / numerals_and_names_covered
# --------------------------------------------------------------------------


def test_content_word_score_full_coverage() -> None:
    score = content_word_score("the pump beeped", "the pump beeped three times and shut off")
    assert score == 1.0


def test_content_word_score_low_coverage() -> None:
    score = content_word_score(
        "the invoice was overdue and billing escalated", "the pump beeped three times"
    )
    assert score < 0.5


def test_content_word_score_empty_content_bucket_is_full_score() -> None:
    # a claim that's entirely stopwords/numerals -- nothing content-like to fail to cover
    assert content_word_score("it was the", "anything at all") == 1.0


def test_numerals_and_names_covered_true_when_present() -> None:
    numerals_ok, names_ok = numerals_and_names_covered(
        "Dana Whitlock reported it beeped 3 times", "Dana Whitlock said it beeped 3 times total"
    )
    assert numerals_ok is True
    assert names_ok is True


def test_numerals_and_names_covered_false_when_numeral_absent() -> None:
    numerals_ok, _names_ok = numerals_and_names_covered(
        "the pump beeped, costing $8,400", "the pump beeped three times and shut off"
    )
    assert numerals_ok is False


def test_numerals_and_names_covered_false_when_name_absent() -> None:
    _numerals_ok, names_ok = numerals_and_names_covered(
        "Kenji Mori confirmed the pump beeped", "the pump beeped three times and shut off"
    )
    assert names_ok is False


def test_numerals_and_names_covered_checked_against_whatever_text_is_passed() -> None:
    # the caller decides the scope (local sentence vs. full document) --
    # this function itself just checks presence in whatever it's given.
    claim = "Dana Whitlock reported it beeped"
    assert numerals_and_names_covered(claim, "no names here at all") == (True, False)
    assert numerals_and_names_covered(claim, "elsewhere in the doc, Dana Whitlock signed off") == (
        True,
        True,
    )


# --------------------------------------------------------------------------
# passes_overlap_gate
# --------------------------------------------------------------------------


def test_passes_overlap_gate_true_for_faithful_matching_claim() -> None:
    assert (
        passes_overlap_gate(
            claim="The pump beeped three times and shut off.",
            quote="The pump beeped three times and shut off",
            doc_text=_DOC_TEXT,
            threshold=0.6,
        )
        is True
    )


def test_passes_overlap_gate_false_when_quote_not_in_document() -> None:
    assert (
        passes_overlap_gate(
            claim="The pump beeped three times.",
            quote="a sentence never present anywhere in the document",
            doc_text=_DOC_TEXT,
            threshold=0.3,
        )
        is False
    )


def test_passes_overlap_gate_false_when_added_numeral_missing() -> None:
    assert (
        passes_overlap_gate(
            claim="The pump beeped three times, costing $9,900 in damages.",
            quote="The pump beeped three times and shut off",
            doc_text=_DOC_TEXT,
            threshold=0.3,
        )
        is False
    )


def test_passes_overlap_gate_false_when_added_name_missing() -> None:
    assert (
        passes_overlap_gate(
            claim="Kenji Mori confirmed the pump beeped three times.",
            quote="The pump beeped three times and shut off",
            doc_text=_DOC_TEXT,
            threshold=0.3,
        )
        is False
    )


def test_passes_overlap_gate_respects_threshold() -> None:
    claim = "The billing department escalated an overdue invoice about the pump."
    quote = "The pump beeped three times and shut off"
    # low threshold: passes despite weak overlap; high threshold: fails
    # both the local AND full-document (0.8 default) bar
    assert passes_overlap_gate(claim=claim, quote=quote, doc_text=_DOC_TEXT, threshold=0.1) is True
    assert passes_overlap_gate(claim=claim, quote=quote, doc_text=_DOC_TEXT, threshold=0.9) is False


# --------------------------------------------------------------------------
# overlap_gate_reason -- the drop-log reason string behind each
# passes_overlap_gate bool (feature/labeling-exact-quotes, build
# instructions item 2: "quote gate, overlap gate, numeral or name rule").
# --------------------------------------------------------------------------


def test_overlap_gate_reason_none_when_gate_passes() -> None:
    assert (
        overlap_gate_reason(
            claim="The pump beeped three times and shut off.",
            quote="The pump beeped three times and shut off",
            doc_text=_DOC_TEXT,
            threshold=0.6,
        )
        is None
    )


def test_overlap_gate_reason_quote_gate_when_quote_absent() -> None:
    assert (
        overlap_gate_reason(
            claim="The pump beeped three times.",
            quote="a sentence never present anywhere in the document",
            doc_text=_DOC_TEXT,
            threshold=0.3,
        )
        == "quote_gate"
    )


def test_overlap_gate_reason_numeral_when_added_numeral_missing() -> None:
    assert (
        overlap_gate_reason(
            claim="The pump beeped three times, costing $9,900 in damages.",
            quote="The pump beeped three times and shut off",
            doc_text=_DOC_TEXT,
            threshold=0.3,
        )
        == "overlap_gate_numeral"
    )


def test_overlap_gate_reason_name_when_added_name_missing() -> None:
    assert (
        overlap_gate_reason(
            claim="Kenji Mori confirmed the pump beeped three times.",
            quote="The pump beeped three times and shut off",
            doc_text=_DOC_TEXT,
            threshold=0.3,
        )
        == "overlap_gate_name"
    )


def test_overlap_gate_reason_content_when_neither_bar_clears() -> None:
    doc_text = "The pump beeped three times and shut off."
    claim = "The billing department flagged an unrelated overdue invoice entirely."
    quote = "The pump beeped three times and shut off"
    assert (
        overlap_gate_reason(
            claim=claim, quote=quote, doc_text=doc_text, threshold=0.5, full_text_threshold=0.8
        )
        == "overlap_gate_content"
    )


def test_overlap_gate_reason_agrees_with_passes_overlap_gate_bool() -> None:
    cases = [
        ("The pump beeped three times and shut off.", "The pump beeped three times and shut off"),
        ("Kenji Mori confirmed the pump beeped.", "The pump beeped three times and shut off"),
        ("The pump beeped, costing $9,900.", "The pump beeped three times and shut off"),
        ("x", "a sentence never present anywhere in the document"),
    ]
    for claim, quote in cases:
        reason = overlap_gate_reason(claim=claim, quote=quote, doc_text=_DOC_TEXT, threshold=0.3)
        passed = passes_overlap_gate(claim=claim, quote=quote, doc_text=_DOC_TEXT, threshold=0.3)
        assert (reason is None) == passed


# --------------------------------------------------------------------------
# Scope correction (coordinator, 2026-09-21, same day as the 0.5 default):
# numerals/names checked against the FULL doc_text, not just the local
# quote-plus-sentence window; content coverage passes on either bar.
# --------------------------------------------------------------------------

_MULTI_SENTENCE_DOC = (
    "[L1] Dana Whitlock opened a ticket on 2026-08-04 about a late shipment. "
    "[L2] the pump beeped three times and shut off. "
    "[L3] No adverse effects were reported."
)


def test_passes_overlap_gate_keeps_claim_combining_quote_with_name_from_elsewhere_in_doc() -> None:
    # "Dana Whitlock" and "2026-08-04" are in L1, the quote is in L2 --
    # a real extractor reading the whole record can legitimately combine
    # them. Must be KEPT, not treated as a fabrication.
    claim = "Dana Whitlock reported the pump beeped three times and shut off on 2026-08-04."
    quote = "the pump beeped three times and shut off"
    assert (
        passes_overlap_gate(claim=claim, quote=quote, doc_text=_MULTI_SENTENCE_DOC, threshold=0.5)
        is True
    )


def test_passes_overlap_gate_still_drops_name_absent_from_whole_document() -> None:
    # "Kenji Mori" appears nowhere in _MULTI_SENTENCE_DOC at all -- still
    # a fabrication even under the widened, full-document scope.
    claim = "Kenji Mori reported the pump beeped three times and shut off."
    quote = "the pump beeped three times and shut off"
    assert (
        passes_overlap_gate(claim=claim, quote=quote, doc_text=_MULTI_SENTENCE_DOC, threshold=0.5)
        is False
    )


def test_passes_overlap_gate_full_text_threshold_rescues_low_local_score() -> None:
    # local score (against the quote's own sentence) is 0.167 -- most of
    # the claim's content words are in the OTHER sentence, not the
    # quote's; full-document score is 0.833 (5 of 6 content words
    # present somewhere in doc_text).
    doc_text = "The billing department flagged an overdue matter. The pump beeped three times."
    claim = "The billing department flagged an overdue invoice about the pump."
    quote = "The pump beeped three times"
    # local score (0.167) fails any reasonable local threshold ...
    assert passes_overlap_gate(claim=claim, quote=quote, doc_text=doc_text, threshold=0.99) is True
    # ... it's the full_text_threshold path (default 0.8, cleared by
    # 0.833) doing the rescuing -- proof: raising it past 0.833 flips the
    # decision back to dropped.
    assert (
        passes_overlap_gate(
            claim=claim, quote=quote, doc_text=doc_text, threshold=0.99, full_text_threshold=0.9
        )
        is False
    )


def test_passes_overlap_gate_drops_when_neither_bar_clears() -> None:
    doc_text = "The pump beeped three times and shut off."
    claim = "The billing department flagged an unrelated overdue invoice entirely."
    quote = "The pump beeped three times and shut off"
    assert (
        passes_overlap_gate(
            claim=claim, quote=quote, doc_text=doc_text, threshold=0.5, full_text_threshold=0.8
        )
        is False
    )


# --------------------------------------------------------------------------
# Calibration table: evaluate_threshold / sweep_thresholds / choose_threshold
# --------------------------------------------------------------------------


def _cases() -> list[CalibrationCase]:
    gold_doc = "the pump beeped three times and shut off completely during the site visit"
    return [
        CalibrationCase(
            label="gold",
            claim="The pump beeped three times and shut off.",
            quote="the pump beeped three times and shut off",
            doc_text=gold_doc,
            expected_keep=True,
        ),
        CalibrationCase(
            label="gold",
            claim="The device shut off completely during the visit.",
            quote="shut off completely during the site visit",
            doc_text=gold_doc,
            expected_keep=True,
        ),
        CalibrationCase(
            label="quote_swapped",
            claim="The pump beeped three times and shut off.",
            quote="the invoice was overdue by ninety days",
            doc_text="the invoice was overdue by ninety days and billing escalated it",
            expected_keep=False,
        ),
        CalibrationCase(
            label="added_detail",
            claim="The pump beeped three times, costing $8,400 in damages.",
            quote="the pump beeped three times and shut off",
            doc_text=gold_doc,
            expected_keep=False,
        ),
    ]


def test_evaluate_threshold_computes_gold_keep_and_corruption_drop_rates() -> None:
    stats = evaluate_threshold(_cases(), threshold=0.5)
    assert stats.threshold == 0.5
    assert stats.gold_keep_rate == 1.0
    assert stats.corruption_drop_rate_by_label["quote_swapped"] == 1.0
    assert stats.corruption_drop_rate_by_label["added_detail"] == 1.0
    assert stats.overall_corruption_drop_rate == 1.0


def test_evaluate_threshold_no_gold_cases_defaults_to_full_keep_rate() -> None:
    only_corruptions = [c for c in _cases() if not c.expected_keep]
    stats = evaluate_threshold(only_corruptions, threshold=0.5)
    assert stats.gold_keep_rate == 1.0  # vacuously true, nothing to fail


def test_evaluate_threshold_no_corruption_cases_defaults_to_zero_drop_rate() -> None:
    only_gold = [c for c in _cases() if c.expected_keep]
    stats = evaluate_threshold(only_gold, threshold=0.5)
    assert stats.overall_corruption_drop_rate == 0.0
    assert stats.corruption_drop_rate_by_label == {}


def test_sweep_thresholds_returns_one_row_per_threshold() -> None:
    rows = sweep_thresholds(_cases(), [0.2, 0.5, 0.8])
    assert [r.threshold for r in rows] == [0.2, 0.5, 0.8]


def test_choose_threshold_picks_highest_drop_rate_meeting_gold_floor() -> None:
    rows = sweep_thresholds(_cases(), [0.1, 0.3, 0.5, 0.95])
    chosen = choose_threshold(rows, min_gold_keep_rate=0.95)
    assert chosen is not None
    assert chosen.gold_keep_rate >= 0.95


def test_choose_threshold_none_when_nothing_meets_gold_floor() -> None:
    # a gold case that can never pass at any reasonable threshold
    impossible = [
        CalibrationCase(
            label="gold",
            claim="X",
            quote="unrelated text",
            doc_text="totally different",
            expected_keep=True,
        )
    ]
    rows = sweep_thresholds(impossible, [0.1, 0.5, 0.9])
    assert choose_threshold(rows, min_gold_keep_rate=0.95) is None


def test_choose_threshold_ties_broken_toward_higher_threshold() -> None:
    cases = _cases()
    # both 0.3 and 0.6 clear the gold floor and achieve the same full
    # corruption-drop rate on this fixture -- the stricter (higher)
    # threshold should win the tie.
    rows = sweep_thresholds(cases, [0.3, 0.6])
    chosen = choose_threshold(rows, min_gold_keep_rate=0.95)
    assert chosen is not None
    assert chosen.threshold == 0.6
