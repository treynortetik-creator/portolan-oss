"""Tests for `portolan.ingest.prepare.prepare_text`.

Locator formats and the "strips quoted text" requirement come from
`01-spec/extraction-schema-v0.md`, `01-spec/provenance-chain.md`'s locator
table, and the wave A2 build instructions. The email-thread test below
uses the real corpus file so the assertion is against actual planted
boilerplate, not a synthetic stand-in.
"""

from __future__ import annotations

from datetime import date

from ingest_helpers import load_email

from portolan.extract.schema import SourceType
from portolan.ingest.models import RawSource
from portolan.ingest.prepare import prepare_text


def test_call_keeps_timestamp_prefixes_as_locators():
    raw = RawSource(
        source_id="S-call-test",
        source_type=SourceType.CALL,
        source_date=date(2026, 3, 19),
        text=(
            "[00:00:00] Sam Okafor: Hey, can you hear me?\n"
            "[00:00:04] Maria Chen: Yes, loud and clear.\n"
            "[00:00:08] Sam Okafor: Great.\n"
        ),
    )

    prepared = prepare_text(raw)

    assert prepared.text == raw.text  # kept, not rewritten
    assert [span.locator for span in prepared.locators] == ["00:00:00", "00:00:04", "00:00:08"]
    # Each span's text starts with its own timestamp line.
    for span in prepared.locators:
        assert prepared.text[span.start : span.end].startswith(f"[{span.locator}]")


def test_call_preamble_before_first_timestamp_gets_header_span():
    # F14: a title/attendee-list preamble (as the real corpus files carry)
    # falls outside every timestamped span unless it gets one of its own.
    raw = RawSource(
        source_id="S-call-preamble",
        source_type=SourceType.CALL,
        source_date=date(2026, 3, 19),
        text=(
            "Ashgrove QBR — 2026-08-14\n"
            "Attendees: Sam Okafor, Maria Chen\n"
            "\n"
            "[00:00:00] Sam Okafor: Hey, can you hear me?\n"
        ),
    )

    prepared = prepare_text(raw)

    assert prepared.locators[0].locator == "header"
    header_span = prepared.locators[0]
    assert "Ashgrove QBR" in prepared.text[header_span.start : header_span.end]
    assert [span.locator for span in prepared.locators] == ["header", "00:00:00"]


def test_call_with_no_preamble_gets_no_header_span():
    raw = RawSource(
        source_id="S-call-nopreamble",
        source_type=SourceType.CALL,
        source_date=date(2026, 3, 19),
        text="[00:00:00] Sam Okafor: Hey, can you hear me?\n",
    )

    prepared = prepare_text(raw)

    assert [span.locator for span in prepared.locators] == ["00:00:00"]


def test_call_multiline_utterance_extends_previous_locator():
    raw = RawSource(
        source_id="S-call-multiline",
        source_type=SourceType.CALL,
        source_date=date(2026, 3, 19),
        text=(
            "[00:00:00] Sam Okafor: This is a long thought\n"
            "that continues onto a second physical line\n"
            "before the next speaker.\n"
            "[00:00:10] Maria Chen: Got it.\n"
        ),
    )

    prepared = prepare_text(raw)

    assert len(prepared.locators) == 2
    first_span_text = prepared.text[prepared.locators[0].start : prepared.locators[0].end]
    assert "continues onto a second physical line" in first_span_text


def test_ticket_gets_one_locator_per_comment():
    raw = RawSource(
        source_id="S-ticket-019",
        source_type=SourceType.TICKET,
        source_date=date(2026, 5, 12),
        text="unused when metadata['comments'] is present",
        metadata={
            "comments": [
                {"number": 1, "author": "Owen Castellane", "body": "First comment body."},
                {"number": 2, "author": "Priya Natarajan", "body": "Second comment body."},
            ]
        },
    )

    prepared = prepare_text(raw)

    # F8: the ticket id is used exactly as given -- no invented "T-" prefix.
    locators = [span.locator for span in prepared.locators]
    assert locators == ["S-ticket-019#1", "S-ticket-019#2"]
    assert "First comment body." in prepared.text
    assert "Second comment body." in prepared.text
    span1 = prepared.locators[0]
    assert "First comment body." in prepared.text[span1.start : span1.end]


def test_ticket_comment_missing_number_falls_back_to_position():
    # F11: a comment dict without "number" used to raise KeyError and take
    # the whole pipeline stage down with it.
    raw = RawSource(
        source_id="S-ticket-nonumber",
        source_type=SourceType.TICKET,
        source_date=date(2026, 5, 12),
        text="unused",
        metadata={"comments": [{"body": "Only comment, no number field."}]},
    )

    prepared = prepare_text(raw)

    assert prepared.locators[0].locator == "S-ticket-nonumber#1"
    assert "Only comment, no number field." in prepared.text


def test_ticket_without_structured_comments_falls_back_to_one_locator():
    raw = RawSource(
        source_id="S-ticket-plain",
        source_type=SourceType.TICKET,
        source_date=date(2026, 5, 12),
        text="A ticket body with no structured comments.",
    )

    prepared = prepare_text(raw)

    assert len(prepared.locators) == 1
    assert prepared.locators[0].locator == "S-ticket-plain#1"


def test_email_thread_strips_quoted_original_under_30_percent():
    # S-email-021-thread: a ten-reply nested-quote thread
    # (corpus/truth-table.yaml `filter.quoted_text_stripped`). Only the
    # newest reply is not quoted; everything else is `>`-prefixed.
    raw = load_email("S-email-021-thread")

    prepared = prepare_text(raw)

    assert len(prepared.text) < 0.30 * len(raw.text)
    assert ">" not in prepared.text
    assert "Of course" in prepared.text  # the one unquoted reply survives
    assert "Quick-Start" not in prepared.text  # deep in the quoted chain


def test_email_paragraph_locators_use_message_id_and_paragraph_number():
    # F8: settled to match the manifest's own form exactly
    # (`S-email-005@halvardmedical.com#p2`), not the earlier `msg:...¶n`.
    raw = RawSource(
        source_id="S-email-simple",
        source_type=SourceType.EMAIL,
        source_date=date(2026, 8, 25),
        text="First paragraph of the reply.\n\nSecond paragraph of the reply.\n",
        metadata={"message_id": "S-email-simple@halvardmedical.com"},
    )

    prepared = prepare_text(raw)

    assert [span.locator for span in prepared.locators] == [
        "S-email-simple@halvardmedical.com#p1",
        "S-email-simple@halvardmedical.com#p2",
    ]
    assert "First paragraph" in prepared.text
    assert "Second paragraph" in prepared.text


def test_email_quoted_block_with_real_text_after_it_survives():
    # F1: a `>` block anywhere but the trailing edge of the body -- with
    # unquoted content after it -- must never be stripped. This is the
    # planted case from the review: an email quoting a poem the ICU lead
    # wants kept, with a real paragraph following.
    raw = RawSource(
        source_id="S-email-poem",
        source_type=SourceType.EMAIL,
        source_date=date(2026, 8, 25),
        text=(
            "Sharing the poem our ICU lead wants in the training deck:\n\n"
            "> Roses are red\n"
            "> Violets are blue\n"
            "> Sharps go here\n\n"
            "Let me know if you need the source file too.\n"
        ),
        metadata={"message_id": "m-poem"},
    )

    prepared = prepare_text(raw)

    assert "Roses are red" in prepared.text
    assert "Violets are blue" in prepared.text
    assert "Sharps go here" in prepared.text
    assert "Let me know if you need the source file too." in prepared.text
    assert not prepared.stripping_skipped


def test_email_entirely_quoted_body_never_comes_out_empty():
    # F1: an email whose whole body is a `>`-quoted customer complaint used
    # to come out as text='' with zero locators -- an empty envelope
    # reaching extraction. Falls back to the unstripped original instead.
    raw = RawSource(
        source_id="S-email-allquoted",
        source_type=SourceType.EMAIL,
        source_date=date(2026, 8, 25),
        text=("> The pump alarm has been going off at odd hours all week.\n> Please advise.\n"),
        metadata={"message_id": "m-allquoted"},
    )

    prepared = prepare_text(raw)

    assert prepared.text.strip() != ""
    assert prepared.locators
    assert prepared.removed_chars == 0
    assert prepared.stripping_skipped
    assert "pump alarm" in prepared.text


def test_email_strips_on_wrote_block():
    raw = RawSource(
        source_id="S-email-onwrote",
        source_type=SourceType.EMAIL,
        source_date=date(2026, 8, 25),
        text=(
            "Sounds good, thanks.\n\n"
            "On Mon, Aug 25, 2026 at 9:00 AM, Dana Whitlock wrote:\n"
            "> Original message content that should not survive.\n"
        ),
        metadata={"message_id": "m1"},
    )

    prepared = prepare_text(raw)

    assert "should not survive" not in prepared.text
    assert "Sounds good" in prepared.text


def test_email_strips_signature_after_dash_dash():
    raw = RawSource(
        source_id="S-email-sig",
        source_type=SourceType.EMAIL,
        source_date=date(2026, 8, 25),
        text=("The actual message.\n\n-- \nSam Okafor\nAccount Rep, Halvard Medical\n"),
        metadata={"message_id": "m1"},
    )

    prepared = prepare_text(raw)

    assert "The actual message." in prepared.text
    assert "Account Rep" not in prepared.text


def test_email_bare_dash_dash_mid_message_is_not_a_signature_delimiter():
    # F12: a bare "--" divider used mid-message (an outage summary's
    # section break) must not truncate the real content that follows it --
    # only "-- " (trailing space, RFC 3676) or a bare "--" that really is
    # the last block counts as a signature delimiter.
    raw = RawSource(
        source_id="S-email-dashrule",
        source_type=SourceType.EMAIL,
        source_date=date(2026, 8, 25),
        text=(
            "Outage summary, part one.\n\n--\n\nOutage summary, part two: root cause identified.\n"
        ),
        metadata={"message_id": "m1"},
    )

    prepared = prepare_text(raw)

    assert "Outage summary, part one." in prepared.text
    assert "root cause identified" in prepared.text


def test_email_bare_dash_dash_as_last_block_is_stripped():
    raw = RawSource(
        source_id="S-email-dashend",
        source_type=SourceType.EMAIL,
        source_date=date(2026, 8, 25),
        text="The actual message.\n\n--\nSam Okafor\n",
        metadata={"message_id": "m1"},
    )

    prepared = prepare_text(raw)

    assert "The actual message." in prepared.text
    assert "Sam Okafor" not in prepared.text


def test_email_strips_unsubscribe_footer_paragraph():
    raw = RawSource(
        source_id="S-email-newsletter-footer",
        source_type=SourceType.EMAIL,
        source_date=date(2026, 8, 25),
        text=(
            "Here is the real content of the message.\n\n"
            "You are receiving this because you subscribed. "
            "Click here to unsubscribe from future emails.\n"
        ),
        metadata={"message_id": "m1"},
    )

    prepared = prepare_text(raw)

    assert "real content" in prepared.text
    assert "unsubscribe" not in prepared.text.lower()
    assert prepared.removed_chars > 0


def test_email_customer_unsubscribe_request_is_not_stripped():
    # Bug hunt 2026-09-18, finding 7: a paragraph containing "unsubscribe"
    # is only boilerplate when it reads like a footer -- a customer's own
    # first-person request using the same word must survive verbatim,
    # even as the message's last paragraph.
    raw = RawSource(
        source_id="S-email-customer-unsub",
        source_type=SourceType.EMAIL,
        source_date=date(2026, 8, 25),
        text=(
            "Hi team, quick request.\n\n"
            "Please unsubscribe our night-shift nurses from the daily digest. "
            "This is the third time I have asked.\n"
        ),
        metadata={"message_id": "m1"},
    )

    prepared = prepare_text(raw)

    assert "Please unsubscribe our night-shift nurses" in prepared.text
    assert "third time I have asked" in prepared.text
    assert prepared.removed_chars <= 1  # only the trailing newline lost in the join round-trip


def test_email_customer_unsubscribe_question_is_not_stripped():
    raw = RawSource(
        source_id="S-email-customer-unsub-question",
        source_type=SourceType.EMAIL,
        source_date=date(2026, 8, 25),
        text=(
            "Hi team, quick question.\n\n"
            "How do we unsubscribe just the billing distribution list?\n"
        ),
        metadata={"message_id": "m1"},
    )

    prepared = prepare_text(raw)

    assert "How do we unsubscribe just the billing distribution list?" in prepared.text
    assert prepared.removed_chars <= 1  # only the trailing newline lost in the join round-trip


def test_email_real_content_containing_a_disclaimer_keyword_is_not_stripped_when_not_trailing():
    # A disclaimer keyword earlier in the message (not the trailing
    # footer block) is left alone entirely -- only a TRAILING run of
    # recognized footer paragraphs is ever checked. This paragraph has
    # neither a first-person pronoun nor a question mark, so only the
    # trailing-block restriction protects it.
    raw = RawSource(
        source_id="S-email-keyword-mid-message",
        source_type=SourceType.EMAIL,
        source_date=date(2026, 8, 25),
        text=(
            "Reminder to the team: unsubscribe requests from the vendor's "
            "mailing list are handled by IT on Fridays.\n\n"
            "Otherwise everything looks good for Thursday.\n"
        ),
        metadata={"message_id": "m1"},
    )

    prepared = prepare_text(raw)

    assert "unsubscribe requests from the vendor's mailing list" in prepared.text
    assert "everything looks good for Thursday" in prepared.text
    assert prepared.removed_chars <= 1  # only the trailing newline lost in the join round-trip


def test_crm_row_locator_falls_back_to_object_and_crm_id_without_a_path():
    raw = RawSource(
        source_id="S-crm-A1010",
        source_type=SourceType.CRM,
        source_date=date(2026, 9, 4),
        text="Ashgrove Health, A-1010, system, ashgrovehealth.org",
        metadata={"object": "account", "crm_id": "A-1010"},
    )

    prepared = prepare_text(raw)

    assert len(prepared.locators) == 1
    assert prepared.locators[0].locator == "account:A-1010"
    span = prepared.locators[0]
    assert prepared.text[span.start : span.end] == prepared.text
    # F2: the marker must actually be in the text, not just recorded as an
    # offset over unmodified text -- an extractor can't cite a locator it
    # never sees.
    assert prepared.text.startswith("[account:A-1010]")
    assert "Ashgrove Health" in prepared.text


def test_crm_row_locator_uses_manifest_path_form_when_given():
    # F8: settled to match corpus/out/manifest.json's own answer-key form
    # exactly, e.g. `crm/opportunities.csv#O-3001`.
    raw = RawSource(
        source_id="S-crm-A1010",
        source_type=SourceType.CRM,
        source_date=date(2026, 9, 4),
        text="Ashgrove Health, A-1010, system, ashgrovehealth.org",
        metadata={"object": "account", "crm_id": "A-1010", "path": "crm/accounts.csv"},
    )

    prepared = prepare_text(raw)

    assert prepared.locators[0].locator == "crm/accounts.csv#A-1010"
    assert prepared.text.startswith("[crm/accounts.csv#A-1010]")


def test_rss_item_locator_is_url_and_date():
    raw = RawSource(
        source_id="S-rss-004",
        source_type=SourceType.RSS,
        source_date=date(2026, 8, 26),
        text="Corvane Systems recalls C-40 pump line\n\nDetails...",
        metadata={"link": "https://feeds.example-newswire.test/portolan-corpus/s-rss-004"},
    )

    prepared = prepare_text(raw)

    assert prepared.locators[0].locator == (
        "url:https://feeds.example-newswire.test/portolan-corpus/s-rss-004 2026-08-26"
    )
    # F2: the marker must actually be in the text.
    assert prepared.text.startswith(
        "[url:https://feeds.example-newswire.test/portolan-corpus/s-rss-004 2026-08-26]"
    )
    assert "Corvane Systems" in prepared.text


def test_doc_uses_page_paragraph_locator_when_page_markers_present():
    raw = RawSource(
        source_id="S-doc-test",
        source_type=SourceType.DOC,
        source_date=date(2026, 8, 1),
        text=(
            "[page 1]\nFirst paragraph on page one.\n\nSecond paragraph on page one.\n\n"
            "[page 2]\nOnly paragraph on page two.\n"
        ),
    )

    prepared = prepare_text(raw)

    locators = [span.locator for span in prepared.locators]
    assert locators == ["p1¶1", "p1¶2", "p2¶1"]
    # F2: the marker must actually be injected into the text the extractor
    # reads, not just recorded as an offset over unmodified prose.
    assert "[p1¶1] First paragraph on page one." in prepared.text
    assert "[p1¶2] Second paragraph on page one." in prepared.text
    assert "[p2¶1] Only paragraph on page two." in prepared.text
    for span in prepared.locators:
        assert prepared.text[span.start : span.end].startswith(f"[{span.locator}]")


def test_doc_falls_back_to_line_numbers_without_page_markers():
    raw = RawSource(
        source_id="S-doc-nolines",
        source_type=SourceType.DOC,
        source_date=date(2026, 8, 1),
        text="First line.\nSecond line.\n\nFourth line (line 3 was blank).\n",
    )

    prepared = prepare_text(raw)

    locators = [span.locator for span in prepared.locators]
    assert locators == ["L1", "L2", "L4"]
    assert "[L1] First line." in prepared.text
    assert "[L2] Second line." in prepared.text
    assert "[L4] Fourth line (line 3 was blank)." in prepared.text
    for span in prepared.locators:
        assert prepared.text[span.start : span.end].startswith(f"[{span.locator}]")
