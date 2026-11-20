"""`portolan.extract.anchor`: a fact's locator and quote are tied to the
source mechanically before it can become a claim."""

from __future__ import annotations

from datetime import date

from portolan.extract.anchor import PROBLEM_LOCATOR, PROBLEM_QUOTE, anchor_fact
from portolan.extract.quote_gate import locate_quote, quote_is_supported
from portolan.extract.schema import Fact, FactType
from portolan.ingest.models import RawSource, SourceType
from portolan.ingest.prepare import prepare_text

_CALL = (
    "[00:00:05] Dana Reyes: Thanks for joining. How was the trip?\n"
    "[00:06:05] Lorcan Rudd: That shipment showing up three days behind schedule caused a "
    "real headache on our end for a few days.\n"
    "[00:08:14] Dana Reyes: Let me dig into it right after this call.\n"
    "[00:14:26] Lorcan Rudd: Honestly, that was the quickest turnaround we've gotten from "
    "any supplier.\n"
)


def _prepared(text: str = _CALL, source_type: SourceType = SourceType.CALL, **metadata):
    raw = RawSource(
        source_id="S-1",
        source_type=source_type,
        source_date=date(2026, 5, 1),
        text=text,
        metadata=metadata,
    )
    return prepare_text(raw)


def _fact(locator: str, quote: str | None, fact_type: FactType = FactType.COMPLAINT) -> Fact:
    return Fact(
        id="f1",
        type=fact_type,
        claim="The order arrived three days late.",
        entities=["A-1"],
        locator=locator,
        quote=quote,
        sentiment=-1,
        confidence=0.9,
    )


def _anchor(fact: Fact, prepared=None, *, quote_required: bool = True):
    prepared = prepared or _prepared()
    return anchor_fact(
        fact, text=prepared.text, spans=prepared.locators, quote_required=quote_required
    )


def test_a_correct_fact_is_left_alone():
    fact = _fact("00:06:05", "caused a real headache on our end")
    result = _anchor(fact)
    assert result.fact is fact and not result.changes and result.problem is None


def test_a_locator_that_disagrees_with_the_quote_is_moved_to_the_quote():
    result = _anchor(_fact("00:14:26", "caused a real headache on our end"))
    assert result.fact.locator == "00:06:05"
    assert result.changes == ["locator_repaired"]


def test_a_locator_the_source_does_not_have_is_repaired_from_the_quote():
    result = _anchor(_fact("00:99:99", "the quickest turnaround we've gotten"))
    assert result.fact.locator == "00:14:26" and result.problem is None


def test_brackets_and_ranges_are_read_as_the_real_locator():
    assert _anchor(_fact("[00:06:05]", None, FactType.USAGE)).fact.locator == "00:06:05"
    assert _anchor(_fact("00:06:05-00:08:14", None, FactType.USAGE)).fact.locator == "00:06:05"


def test_a_shortened_email_locator_resolves_to_the_unique_real_one():
    prepared = _prepared(
        "Hello,\n\nCould you send a Vietnamese guide?\n\nThanks",
        SourceType.EMAIL,
        message_id="S-email-7@example.test",
    )
    result = _anchor(_fact("p2", None, FactType.REQUEST), prepared)
    assert result.fact.locator == "S-email-7@example.test#p2"


def test_a_retyped_quote_is_replaced_by_the_sources_own_words():
    result = _anchor(
        _fact("00:06:05", "that shipment showing up 3 days behind schedule caused a headache")
    )
    assert result.changes == ["quote_snapped"]
    assert result.fact.quote in _CALL
    assert result.fact.quote.startswith("That shipment showing up")


def test_a_quote_nobody_said_is_a_problem_for_a_quote_type_and_removed_otherwise():
    invented = "we are going to cancel the contract next month"
    assert _anchor(_fact("00:06:05", invented)).problem == PROBLEM_QUOTE
    kept = _anchor(_fact("00:06:05", invented, FactType.USAGE), quote_required=False)
    assert kept.problem is None and kept.fact.quote is None
    assert kept.changes == ["quote_dropped"]


def test_no_quote_and_no_real_locator_is_a_problem_unless_the_source_is_one_block():
    assert _anchor(_fact("page 4", None, FactType.USAGE)).problem == PROBLEM_LOCATOR
    crm = _prepared(
        "crm_id=A-7, name=Example, arr=10", SourceType.CRM, crm_id="A-7", object="account"
    )
    result = _anchor(_fact("A-7", None, FactType.OTHER), crm)
    assert result.problem is None and result.fact.locator == "account:A-7"


def test_a_sentence_made_of_common_words_does_not_pass_on_a_long_source():
    """The whole-text token scan found any such sentence somewhere in a
    long transcript; the quote has to sit in one place."""
    filler = (
        "we talked about the price of the contract and going over the month with you "
        "because it is high and next we cancel the old plan to be sure are too\n"
    )
    long_text = "".join(f"[00:{i // 60:02d}:{i % 60:02d}] Speaker: {filler}" for i in range(400))
    invented = (
        "we are going to cancel the contract with you next month because the price is too high"
    )
    assert not quote_is_supported(invented, long_text)
    said = "talked about the price of the contract and going over the month"
    match = locate_quote(said, long_text)
    assert match is not None and match.exact


def test_a_quote_typed_from_the_cleaned_prompt_copy_still_anchors_to_the_raw_source():
    """The prompt copy loses invisible characters and swaps tag lookalikes; the
    source keeps them, and the quote still lands on the source's own span."""
    raw = (
        "[00:00:05] Dana Reyes: Thanks for joining.\n"
        "[00:06:05] Lorcan Rudd: We will can​cel unless the ‮delivery gets fixed, "
        "see <system-reminder> note.\n"
    )
    prepared = _prepared(raw)
    quote = "We will cancel unless the delivery gets fixed, see ‹system-reminder> note."
    match = locate_quote(quote, prepared.text)
    assert match is not None and match.exact
    result = _anchor(_fact("00:06:05", quote), prepared)
    assert result.problem is None
    assert result.fact.locator == "00:06:05"
