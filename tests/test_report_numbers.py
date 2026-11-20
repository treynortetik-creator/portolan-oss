"""Tests for `portolan.report.numbers` (the numbers rule, polish wave
2026-09-12: `01-spec/decisions-log.md` 2026-09-12 "Numbers rule").
"""

from __future__ import annotations

from portolan.report.numbers import NUMERAL_NOT_SUPPLIED, NumberFacts, check_numerals


def test_number_facts_as_table_lists_every_fact_in_order():
    facts = NumberFacts()
    facts.add("notes_created", 3)
    facts.add("account_arr", 1420000, "1,420,000")

    table = facts.as_table()
    lines = table.splitlines()
    assert lines[0] == "- notes_created: 3"
    assert lines[1] == "- account_arr: 1,420,000"


def test_number_facts_as_table_when_empty_says_so_explicitly():
    facts = NumberFacts()
    assert "no figures" in facts.as_table().lower()
    assert not facts


def test_check_numerals_allows_dates_years_ids_times_ordinals_and_supplied_figures():
    facts = NumberFacts()
    facts.add("account_arr", 1420000, "1,420,000")
    facts.add("open_actions", 2)

    text = (
        "On 2026-08-04, Ashgrove Health (CRM A-1010) called at 00:14:32 "
        "about its 21st renewal in 2026. ARR is $1,420,000 across 2 open actions."
    )
    assert check_numerals(text, facts) == []


def test_check_numerals_ignores_citation_markers_and_wikilinks():
    facts = NumberFacts()
    text = (
        "Larkspur Memorial Hospital had a late delivery. "
        "^[s-ticket-031 @ T-031#1] See [[larkspur-memorial-hospital-2026]] for detail."
    )
    assert check_numerals(text, facts) == []


def test_check_numerals_rejects_a_number_not_in_the_supplied_figures():
    facts = NumberFacts()
    facts.add("account_arr", 1420000, "1,420,000")

    text = "This account has 17 open complaints and ARR of $1,420,000."
    violations = check_numerals(text, facts)
    assert violations == ["17"]


def test_check_numerals_rejects_a_fabricated_dollar_figure():
    facts = NumberFacts()  # nothing computed for this text at all
    text = "Estimated pipeline impact is roughly $50,000 this quarter."
    violations = check_numerals(text, facts)
    # The leading `$` is decorative punctuation, stripped like any other
    # edge junk before the core numeral is judged and reported.
    assert violations == ["50,000"]


def test_check_numerals_rejects_fabricated_magnitude_figures():
    """Finding 3 (blocker): before this fix, none of these four became
    candidates at all -- the old rule skipped any word with a letter in
    it, so a fabricated `$8.2M` was never even checked. Test gap 1's four
    named strings, each with nothing computed to back it."""
    facts = NumberFacts()
    assert check_numerals("Churn risk across $8.2M of ARR.", facts) == ["8.2M"]
    assert check_numerals("About 450k in pipeline.", facts) == ["450k"]
    assert check_numerals("A 3x increase in complaints.", facts) == ["3x"]
    assert check_numerals("Late deliveries put 12bn at risk.", facts) == ["12bn"]


def test_check_numerals_allows_a_magnitude_matching_a_supplied_bare_rendering():
    """ "the numeric part to match a supplied rendering and the suffix to
    be consistent with it" (finding 3's fix): a fact rendered bare as
    `"8.2"` licenses the model writing `"$8.2M"` -- the suffix's own
    legitimacy is already settled by it being one `_MAGNITUDE_RE`
    recognises at all."""
    facts = NumberFacts()
    facts.add("account_arr_millions", 8.2, "8.2")
    assert check_numerals("This account is worth $8.2M in ARR.", facts) == []


def test_check_numerals_rejects_a_magnitude_whose_numeric_part_is_not_supplied():
    """A magnitude suffix must not bypass the check for an unrelated,
    genuinely supplied figure -- only its OWN numeric part licenses it."""
    facts = NumberFacts()
    facts.add("account_arr", 1420000, "1,420,000")
    assert check_numerals("ARR is roughly $9.9M this quarter.", facts) == ["9.9M"]


def test_harvest_allows_restating_a_magnitude_figure_verbatim():
    facts = NumberFacts()
    facts.harvest("The account renewed at $8.2M ARR on 2026-08-04.")
    assert check_numerals("Renewal ARR is $8.2M.", facts) == []


def test_check_numerals_still_allows_ordinals_and_ids_with_magnitude_like_suffixes():
    facts = NumberFacts()
    text = "Its 21st renewal (CRM A-1010) closed with no complaints."
    assert check_numerals(text, facts) == []


def test_check_numerals_never_flags_small_counting_words():
    facts = NumberFacts()
    text = "A few customers raised one complaint about a couple of late shipments."
    assert check_numerals(text, facts) == []


def test_numeral_not_supplied_constant_matches_the_reason_string():
    assert NUMERAL_NOT_SUPPLIED == "numeral_not_supplied"


def test_harvest_allows_restating_a_number_already_in_the_given_context():
    facts = NumberFacts()
    claims_context = "Larkspur Memorial Hospital paid $50,000 for the renewal on 2026-08-04."
    facts.harvest(claims_context)

    text = "Larkspur Memorial Hospital's renewal was worth $50,000."
    assert check_numerals(text, facts) == []


def test_harvest_does_not_allow_a_number_absent_from_the_context():
    facts = NumberFacts()
    facts.harvest("Larkspur Memorial Hospital reported a late delivery.")

    text = "This cost the account $50,000 in penalties."
    assert check_numerals(text, facts) == ["50,000"]


def test_harvest_returns_self_for_chaining():
    facts = NumberFacts()
    assert facts.harvest("no numbers here") is facts


def test_harvest_with_a_marker_allows_restating_it_only_alongside_that_citation():
    """Test gap 10 (review finding 9): a number harvested WITH a marker
    only licenses a sentence that also cites that same marker -- not
    "somewhere in context", "in the claim you cited"."""
    facts = NumberFacts()
    facts.harvest(
        "Larkspur Memorial Hospital paid $50,000 for the renewal on 2026-08-04.",
        marker=("s-crm-042", "row#7"),
    )

    faithful = "The renewal was worth $50,000. ^[s-crm-042 @ row#7]"
    assert check_numerals(faithful, facts) == []


def test_harvest_with_a_marker_does_not_license_recombination():
    """The same fabricated-looking check as above, but the sentence cites
    a DIFFERENT claim than the one the number was harvested from -- this
    is "recombination", not restatement, and must still be rejected even
    though `$50,000` is, in isolation, one of `facts`' known renderings."""
    facts = NumberFacts()
    facts.harvest(
        "Larkspur Memorial Hospital paid $50,000 for the renewal on 2026-08-04.",
        marker=("s-crm-042", "row#7"),
    )

    recombined = "This other account also lost $50,000. ^[s-ticket-099 @ T-099#1]"
    assert check_numerals(recombined, facts) == ["50,000"]

    # No citation at all is the same failure mode -- nothing to check the
    # marker against.
    uncited = "The renewal was worth $50,000."
    assert check_numerals(uncited, facts) == ["50,000"]


def test_harvest_without_a_marker_still_licenses_unconditionally():
    """Backward compatibility (review finding 9): a caller that hasn't
    been updated to pass `marker=` keeps the original, looser behaviour
    -- licensed anywhere in that surface's text, no citation required."""
    facts = NumberFacts()
    facts.harvest("Larkspur Memorial Hospital paid $50,000 for the renewal.")
    assert check_numerals("The renewal was worth $50,000.", facts) == []


def test_add_stays_unconditional_even_if_the_same_rendering_is_also_harvested_with_a_marker():
    """A genuinely computed figure (`add()`) is licensed everywhere in
    the surface -- a coincidental marker-bound harvest of the same
    rendering elsewhere must not narrow that."""
    facts = NumberFacts()
    facts.add("notes_created", 3)
    facts.harvest("3 accounts mentioned a competitor.", marker=("s-x", "loc#1"))
    assert check_numerals("3 notes were created this window.", facts) == []


# ---------------------------------------------------------------------
# Bughunt-2026-09-18 finding 3: the gate used to exempt only ISO dates
# (`YYYY-MM-DD`), `HH:MM[:SS]` times, and a bare four-digit year -- every
# other unambiguous date form the corpus and the prompts actually use
# (a month name spelled out, a short `YYYY-MM` month-year, a slash date,
# an ISO week id) fell through to an ordinary numeral check and rejected
# the whole surrounding text, even though "dates ... are never numerals
# for this purpose" (01-spec/surfacing-and-weekly-report.md "Numbers")
# and `ACCOUNT_SYNTHESIS_SYSTEM` explicitly asks for "the date of last
# contact" in "absolute dates". One table, positive (must pass with
# nothing computed) and negative (a non-date numeral nearby must still
# be rejected).
# ---------------------------------------------------------------------

_DATE_POSITIVE_CASES = [
    ("ISO date", "Last contact was 2026-09-09."),
    ("month day, year", "Last contact was September 9, 2026."),
    ("abbreviated month day", "Last contact was Sept 9."),
    ("day month year", "Last contact was 9 September 2026."),
    ("day abbreviated month", "Last contact was 9 Sept."),
    ("month-year short ISO", "Quiet as of 2026-08."),
    ("month year, spelled out", "Quiet as of August 2026."),
    ("ISO week id", "Nothing happened during 2026-W35."),
    ("slash date", "Renewed on 09/30/2026."),
    ("bare time", "Logged at 09:02:51 today."),
    ("bare year", "First seen in 2026."),
    ("ordinal day", "Renews on the 21st."),
]

_DATE_NEGATIVE_CASES = [
    ("unsupplied count", "14 late deliveries this week.", "14"),
    ("unsupplied money", "We lost $8.3M in ARR.", "8.3M"),
    ("unsupplied percentage", "Up 12% week over week.", "12"),
    (
        "unsupplied count next to a real date",
        "On September 9, 2026, the account raised 14 complaints.",
        "14",
    ),
    (
        "unsupplied ARR next to a month-year",
        "As of 2026-08 the account was worth $500k.",
        "500k",
    ),
]


def test_check_numerals_exempts_every_absolute_date_form():
    for _label, text in _DATE_POSITIVE_CASES:
        assert check_numerals(text, NumberFacts()) == [], text


def test_check_numerals_still_rejects_non_date_numerals_near_date_like_text():
    for _label, text, expected_violation in _DATE_NEGATIVE_CASES:
        assert check_numerals(text, NumberFacts()) == [expected_violation], text


def test_check_numerals_month_day_exemption_does_not_swallow_a_following_year():
    """A greedy day-number match inside "Month Day" must not partially
    consume a standalone `YYYY` that follows with no day at all --
    "August 2026" (no day) already passed before this fix (the month has
    no digit, the year matches the bare-year rule on its own); the day/
    year `(?!\\d)` guards exist only to keep it passing once the "Month
    Day[, Year]" phrase-stripping regex was added alongside it."""
    assert check_numerals("As of August 2026 the account was quiet.", NumberFacts()) == []
    assert check_numerals("As of 2026 the account was quiet.", NumberFacts()) == []


def test_check_numerals_strips_quote_marks_around_a_supplied_figure():
    """Finding 3's own repro: a model quoting a customer's exact words
    around an already-licensed number (`"3"`, straight or curly quotes)
    must not turn that verbatim restatement into a fabrication."""
    facts = NumberFacts()
    facts.add("count", 3, "3")
    assert check_numerals('The count was "3" last week.', facts) == []
    assert check_numerals("The count was “3” last week.", facts) == []
    # Still rejected when 3 genuinely isn't supplied -- the quotes are
    # stripped as edge junk before comparison, so the reported violation
    # is the bare numeral, same as any other candidate.
    assert check_numerals('The count was "3" last week.', NumberFacts()) == ["3"]


def test_check_numerals_allows_a_dash_range_when_both_endpoints_are_supplied():
    """Finding 3's own repro: "3-4 accounts" is two numerals joined by a
    hyphen or en dash, not one fabricated token -- each side is checked
    against `facts` independently."""
    facts = NumberFacts()
    facts.add("lo", 3, "3")
    facts.add("hi", 4, "4")
    assert check_numerals("3-4 accounts affected.", facts) == []
    assert check_numerals("3–4 accounts affected.", facts) == []
    # Still rejected when the range isn't backed by two supplied figures.
    assert check_numerals("3–4 accounts affected.", NumberFacts()) == ["3–4"]


def test_check_numerals_present_markers_parameter_overrides_the_text_scan():
    """Bughunt-2026-09-18 finding 1's mechanism, at the `numbers.py` unit
    level (see `tests/test_report_questions.py` for the end-to-end
    version): a caller whose citation form isn't an inline `^[source @
    locator]` marker in the checked text at all (`portolan.report.
    questions.answer_standing_questions`'s `cites` list of note slugs)
    can pass its own `present_markers` and have a marker-bound harvest
    licensed by THAT, instead of by `citation_markers_in(text)` (which
    would find nothing in text that never carries an inline marker)."""
    facts = NumberFacts()
    facts.harvest("Ashgrove reported 14 late deliveries.", marker=("ashgrove-signal", "cites"))

    # No inline marker in the text at all -- the default (text-scanned)
    # behaviour would find nothing and reject "14".
    text = "Ashgrove reported 14 late deliveries."
    assert check_numerals(text, facts) == ["14"]

    # Passing the caller's own notion of what this text cites licenses it.
    assert check_numerals(text, facts, present_markers={("ashgrove-signal", "cites")}) == []

    # A DIFFERENT marker in present_markers does not license it --
    # this is still "cite the claim you're restating", not "cite anything".
    assert check_numerals(text, facts, present_markers={("some-other-note", "cites")}) == ["14"]
