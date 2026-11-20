"""Tests for `portolan.web.review_presenter` -- pure functions, no
database: every test builds the same enriched-row dict shape `portolan.
web.routes.review` assembles (see `present_row`'s own docstring for the
contract) and checks the resulting `ReviewCard`.

the maintainer's own complaint about the old page (`01-spec/`, review-queue-
usability fix, 2026-09-18): "'unknown mention Dayton.' I don't know what
the fuck that means. 'Fact ID F1.' What the fuck does that mean?" These
tests pin, per `kind`, that a card's `headline` reads as a plain-English
sentence a non-engineer can act on, that raw ids never leak into it, and
that every choice is both offered and explained.
"""

from __future__ import annotations

from datetime import date, datetime

from portolan.web.review_presenter import (
    highlight_numerals,
    kind_counts,
    present_row,
    present_rows,
)

CREATED = datetime(2026, 5, 14, 9, 30)


def _base_row(**overrides):
    row = {
        "id": 42,
        "kind": "other",
        "payload": {},
        "source_slug": None,
        "status": "open",
        "created": CREATED,
        "resolved_at": None,
        "resolved_by": None,
    }
    row.update(overrides)
    return row


# ---------------------------------------------------------------------
# unknown_entity
# ---------------------------------------------------------------------


def test_unknown_entity_headline_names_the_mention_and_source_in_plain_english():
    row = _base_row(
        kind="unknown_entity",
        source_slug="S-call-014",
        payload={
            "mention": "Dayton",
            "context": {
                "fact_id": "f1",
                "fact_type": "risk",
                "claim": "Dayton distribution is running behind on the T2 rollout.",
                "quote": "Dayton's been slipping on every T2 delivery this month.",
                "locator": "00:14:32",
                "valid_from": "2026-05-14",
            },
        },
        source={
            "source_type": "call",
            "source_date": date(2026, 5, 14),
            "participants": [{"name": "Jane Doe"}],
            "summary": None,
            "account_name": "Northlake Health",
        },
        context_snippet={"before": "Let's talk logistics.", "after": "We need a fix by Friday."},
    )

    card = present_row(row)

    assert "Dayton" in card.headline
    assert "Northlake Health" in card.headline
    assert "sales call" in card.headline
    assert "2026-05-14" in card.headline
    # Never the raw fact id or a bare "Fact ID F1" -- the maintainer's own example.
    assert "f1" not in card.headline.lower()
    assert "Fact ID" not in card.headline

    assert card.claim_text == "Dayton distribution is running behind on the T2 rollout."
    assert card.quote == "Dayton's been slipping on every T2 delivery this month."
    assert card.context_before == "Let's talk logistics."
    assert card.context_after == "We need a fix by Friday."
    assert card.fact_type_label == "a risk"
    assert card.open_url == "/originals/S-call-014?at=00%3A14%3A32"
    assert card.open_label

    labels = {c.label for c in card.choices}
    assert "This is another name for…" in labels
    assert "Handled, nothing to change" in labels
    assert "Ignore this mention" in labels
    for choice in card.choices:
        assert choice.description  # every choice explained in a sentence

    # The raw fact id lives in the technical fold, never the main card.
    technical = dict(card.technical)
    assert technical["fact id"] == "f1"
    assert technical["review row id"] == "42"


def test_unknown_entity_with_a_legacy_payload_degrades_gracefully():
    """A row written before this fix (`context={"fact_id": "f9"}` only,
    no claim/quote/locator) must still render a sane card, not crash --
    and, since no `evidence_backfill` key means "nothing was found"
    (route-usability follow-up, 2026-09-18), it now surfaces the
    plain-language "couldn't find it" note rather than a silent gap."""
    row = _base_row(
        kind="unknown_entity",
        source_slug="S-call-099",
        payload={"mention": "a brand new name", "context": {"fact_id": "f9"}},
    )

    card = present_row(row)

    assert "a brand new name" in card.headline
    assert card.claim_text is None
    assert card.quote is None
    assert card.open_url == "/originals/S-call-099"  # no locator -- no ?at=
    assert card.show_alias_picker is True
    assert card.evidence_unavailable_note is not None
    assert "no longer has the exact sentence" in card.evidence_unavailable_note


def test_unknown_entity_with_a_legacy_payload_uses_the_route_backfilled_evidence():
    """When `portolan.web.routes.review._enrich_rows_for_presentation`
    already recovered the fact (via `portolan.web.queries.
    backfill_fact_evidence`) and stashed it on `row['evidence_backfill']`,
    the card shows that evidence exactly as if it had always been on the
    payload -- no "couldn't find it" note."""
    row = _base_row(
        kind="unknown_entity",
        source_slug="S-call-099",
        payload={"mention": "Dayton", "context": {"fact_id": "f9", "fact_type": None}},
        evidence_backfill={
            "claim": "Dayton distribution is running behind on the T2 rollout.",
            "quote": "Dayton's been slipping on every T2 delivery this month.",
            "fact_type": "risk",
            "locator": "00:14:32",
        },
    )

    card = present_row(row)

    assert card.claim_text == "Dayton distribution is running behind on the T2 rollout."
    assert card.quote == "Dayton's been slipping on every T2 delivery this month."
    assert card.fact_type_label == "a risk"
    assert card.open_url == "/originals/S-call-099?at=00%3A14%3A32"
    assert card.evidence_unavailable_note is None


def test_unknown_entity_with_a_legacy_payload_and_backfill_explicitly_not_found():
    """`evidence_backfill=None` (the route attempted recovery and found
    nothing, as opposed to the key being absent) reads identically to
    "absent" -- both mean no evidence to show."""
    row = _base_row(
        kind="unknown_entity",
        source_slug="S-call-099",
        payload={"mention": "Dayton", "context": {"fact_id": "f9"}},
        evidence_backfill=None,
    )

    card = present_row(row)

    assert card.claim_text is None
    assert card.evidence_unavailable_note is not None


# ---------------------------------------------------------------------
# ambiguous_entity
# ---------------------------------------------------------------------


def test_ambiguous_entity_document_mention_names_the_candidates():
    row = _base_row(
        kind="ambiguous_entity",
        source_slug="S-email-058",
        payload={
            "mention": "Ashgrove",
            "candidates": ["A-1010", "A-1011"],
            "fact_type": "complaint",
            "claim": "Ashgrove said the new dashboard is slow.",
            "quote": "The dashboard has been crawling all week.",
            "locator": "p2¶1",
        },
        source={
            "source_type": "email",
            "source_date": date(2026, 5, 10),
            "participants": [],
            "summary": None,
            "account_name": None,
        },
    )
    entities_by_crm_id = {
        "A-1010": {"canonical_name": "Ashgrove Health", "kind": "account"},
        "A-1011": {"canonical_name": "Ashgrove Medical Center", "kind": "account"},
    }

    card = present_row(row, entities_by_crm_id=entities_by_crm_id)

    assert "Ashgrove" in card.headline
    assert "couldn't tell which one" in card.headline
    assert "Ashgrove Health" in card.headline or any(
        c["name"] == "Ashgrove Health" for c in card.candidates
    )
    names = {c["name"] for c in card.candidates}
    assert names == {"Ashgrove Health", "Ashgrove Medical Center"}
    assert card.claim_text == "Ashgrove said the new dashboard is slow."
    assert card.open_url == "/originals/S-email-058?at=p2%C2%B61"


def test_ambiguous_entity_document_mention_with_a_legacy_payload_backfills_evidence():
    row = _base_row(
        kind="ambiguous_entity",
        source_slug="S-email-058",
        payload={"mention": "Ashgrove", "candidates": ["A-1010"]},  # legacy: no claim/quote
        evidence_backfill={
            "claim": "Ashgrove said the new dashboard is slow.",
            "quote": "The dashboard has been crawling all week.",
            "fact_type": "complaint",
            "locator": "p2¶1",
        },
    )

    entities_by_crm_id = {"A-1010": {"canonical_name": "Ashgrove Health", "kind": "account"}}
    card = present_row(row, entities_by_crm_id=entities_by_crm_id)

    assert card.claim_text == "Ashgrove said the new dashboard is slow."
    assert card.quote == "The dashboard has been crawling all week."
    assert card.fact_type_label == "a complaint"
    assert card.evidence_unavailable_note is None


def test_ambiguous_entity_document_mention_with_a_legacy_payload_and_no_backfill_shows_the_note():
    row = _base_row(
        kind="ambiguous_entity",
        source_slug="S-email-058",
        payload={"mention": "Ashgrove", "candidates": ["A-1010"]},
        evidence_backfill=None,
    )

    card = present_row(row, entities_by_crm_id={})

    assert card.claim_text is None
    assert card.evidence_unavailable_note is not None
    assert "no longer has the exact sentence" in card.evidence_unavailable_note


def test_ambiguous_entity_crm_import_reads_as_a_crm_dedup_not_a_document_mention():
    row = _base_row(
        kind="ambiguous_entity",
        source_slug=None,
        payload={
            "mention": "Ashgrove",
            "reason": "same normalised name, different email domain",
            "candidates": ["ashgrove-health", "ashgrove-medical"],
        },
    )

    card = present_row(row, entities_by_crm_id={})

    assert "importing the CRM" in card.headline
    assert "Ashgrove" in card.headline
    assert card.open_url is None
    # Candidates unresolved (no matching registry entity) fall back to the
    # raw id rather than disappearing.
    assert {c["name"] for c in card.candidates} == {"ashgrove-health", "ashgrove-medical"}
    # No source to look a fact up against at all -- never the "couldn't
    # find it" note, which only applies to a document mention.
    assert card.evidence_unavailable_note is None


def test_ambiguous_entity_candidates_missing_from_the_registry_fall_back_to_the_raw_id():
    row = _base_row(
        kind="ambiguous_entity",
        source_slug=None,
        payload={"mention": "Ashgrove", "candidates": ["ashgrove-health", "ashgrove-medical"]},
    )

    card = present_row(row, entities_by_crm_id={})

    assert "ashgrove-health" in {c["name"] for c in card.candidates}


# ---------------------------------------------------------------------
# failed_consolidation
# ---------------------------------------------------------------------


def test_failed_consolidation_names_the_note_and_lists_reasons():
    row = _base_row(
        kind="failed_consolidation",
        payload={
            "note_slug": "account-northlake-health",
            "reasons": ["unknown citation marker(s): [S-1 @ loc]", "41 lines (limit 40)"],
        },
        note_title="Northlake Health",
    )

    card = present_row(row)

    assert "Northlake Health" in card.headline
    assert "quality check" in card.headline
    assert card.reasons == [
        "unknown citation marker(s): [S-1 @ loc]",
        "41 lines (limit 40)",
    ]
    assert card.open_url == "/notes/account-northlake-health"
    labels = {c.label for c in card.choices}
    assert labels == {"Handled, nothing to change", "Ignore this"}


# ---------------------------------------------------------------------
# theme_resolution
# ---------------------------------------------------------------------


def test_theme_resolution_reads_as_a_yes_no_question_with_the_evidence():
    row = _base_row(
        kind="theme_resolution",
        payload={
            "theme_slug": "theme-late-deliveries",
            "resolution_claim_id": 501,
            "resolved_week": "2026-05-11",
            "evidence": "the problem is fixed now",
            "detail": "the judge's own internal reason text -- must never render on the page",
        },
        proposal={
            "theme_name": "Late Deliveries",
            "account_slug": "account-northlake-health",
            "world_date": date(2026, 5, 12),
            "claim_text": "Northlake said the late-delivery problem is fixed now.",
            "quote": None,
            "fact_type": "praise",
            "source_slug": "S-call-020",
            "locator": "00:04:10",
        },
    )

    card = present_row(row)

    assert "Late Deliveries" in card.headline
    assert "2026-05-12" in card.headline
    assert card.quote == "the problem is fixed now"
    assert "must never render on the page" not in card.headline
    assert "must never render on the page" not in card.why_it_matters
    assert all("must never render" not in v for _, v in card.technical)
    assert card.account_slug == "account-northlake-health"
    assert card.theme_slug == "theme-late-deliveries"

    labels = {c.label for c in card.choices}
    assert labels == {"Confirm resolution", "Reject"}
    for choice in card.choices:
        assert choice.description


def test_theme_resolution_with_no_proposal_context_does_not_crash():
    row = _base_row(
        kind="theme_resolution",
        payload={},
        proposal=None,
    )

    card = present_row(row)

    assert card.headline
    labels = {c.label for c in card.choices}
    assert labels == {"Confirm resolution", "Reject"}


# ---------------------------------------------------------------------
# quote_unsupported (mechanical quote gate, build item 3, 2026-09-19)
# ---------------------------------------------------------------------


def test_quote_unsupported_shows_the_claim_quote_locator_and_choices():
    row = _base_row(
        kind="quote_unsupported",
        source_slug="s-call-042",
        payload={
            "reason": "The model's quote for this fact was not found in the source text.",
            "fact": {
                "id": "F9",
                "type": "complaint",
                "claim": "Northlake Health reported repeated outages in August 2026.",
                "entities": ["Northlake Health"],
                "locator": "00:12:30",
                "quote": "it just kept going down on us",
                "speaker": None,
                "valid_from": None,
                "valid_to": None,
                "severity": None,
                "sentiment": -1,
                "confidence": 0.7,
            },
        },
        source={"source_type": "call", "participants": [], "account_name": "Northlake Health"},
    )

    card = present_row(row)

    assert "not in the source" in card.headline
    assert card.claim_text == "Northlake Health reported repeated outages in August 2026."
    assert card.quote == "it just kept going down on us"
    assert card.fact_type_label == "a complaint"
    assert card.open_url == "/originals/s-call-042?at=00%3A12%3A30"
    assert any(loc == "locator" and val == "00:12:30" for loc, val in card.technical)

    labels = {c.label for c in card.choices}
    assert labels == {"Keep as claim", "Dismiss"}
    for choice in card.choices:
        assert choice.description
    dismiss = next(c for c in card.choices if c.label == "Dismiss")
    assert dismiss.action == "dismiss"
    keep = next(c for c in card.choices if c.label == "Keep as claim")
    assert keep.action == "resolve"


def test_quote_unsupported_with_no_fact_payload_does_not_crash():
    row = _base_row(kind="quote_unsupported", payload={"reason": "test"})

    card = present_row(row)

    assert card.headline
    assert card.claim_text is None
    assert card.quote is None
    labels = {c.label for c in card.choices}
    assert labels == {"Keep as claim", "Dismiss"}


# ---------------------------------------------------------------------
# other / numeral_not_supplied
# ---------------------------------------------------------------------


def test_numeral_rejection_shows_rejected_text_and_the_allowed_figures():
    row = _base_row(
        payload={
            "reason": "numeral_not_supplied",
            "surface": "headline",
            "note_slug": None,
            "detail": "numeral(s) not in the supplied figures: 733",
            "violations": ["733"],
            "text": "Northlake grew by 733 seats this quarter.",
            "allowed_figures": "- notes_created: 3\n- account_arr: 1,420,000",
        }
    )

    card = present_row(row)

    assert "headline" in card.headline
    assert card.rejected_text == "Northlake grew by 733 seats this quarter."
    assert card.offending_numerals == ["733"]
    assert card.allowed_figures == [
        ("notes created", "3"),
        ("account arr", "1,420,000"),
    ]


def test_highlight_numerals_wraps_only_the_offending_words():
    text = "Northlake grew by 733 seats this quarter."
    marked = highlight_numerals(text, ["733"])
    assert "<mark>733</mark>" in str(marked)
    assert "Northlake grew by" in str(marked)


def test_highlight_numerals_escapes_the_rest_of_the_text():
    marked = highlight_numerals("<script>alert(1)</script> saw 5", ["5"])
    assert "<script>" not in str(marked)
    assert "&lt;script&gt;" in str(marked)
    assert "<mark>5</mark>" in str(marked)


def test_highlight_numerals_with_no_text_returns_empty():
    assert str(highlight_numerals(None, ["5"])) == ""
    assert str(highlight_numerals("", ["5"])) == ""


# ---------------------------------------------------------------------
# other / resolved_language_without_a_resolved_theme
# ---------------------------------------------------------------------


def test_resolved_language_rejection_shows_the_dropped_sentence():
    row = _base_row(
        payload={
            "reason": "resolved_language_without_a_resolved_theme",
            "surface": "headline",
            "note_slug": None,
            "detail": "The pricing concerns theme is now fully resolved.",
        }
    )

    card = present_row(row)

    assert "resolved" in card.headline
    assert card.rejected_text == "The pricing concerns theme is now fully resolved."


# ---------------------------------------------------------------------
# other / failed account synthesis
# ---------------------------------------------------------------------


def test_failed_account_synthesis_names_the_account_and_the_reason():
    row = _base_row(
        payload={
            "reason": "failed account synthesis",
            "note_slug": "account-northlake-health",
            "detail": "cites a source not in the given set",
        },
        note_title="Northlake Health",
    )

    card = present_row(row)

    assert "Northlake Health" in card.headline
    assert "cites a source not in the given set" in card.headline
    assert card.open_url == "/notes/account-northlake-health"


# ---------------------------------------------------------------------
# other / arr_not_parseable
# ---------------------------------------------------------------------


def test_arr_not_parseable_names_the_account_and_the_bad_value():
    row = _base_row(
        payload={
            "reason": "arr_not_parseable",
            "crm_id": "A-1010",
            "detail": "CRM accounts export: arr='N/A' for A-1010 is not a number",
        }
    )

    card = present_row(row)

    assert "A-1010" in card.headline
    assert "not zero" in card.headline
    assert card.detail == "CRM accounts export: arr='N/A' for A-1010 is not a number"


# ---------------------------------------------------------------------
# other / contradicts verdicts
# ---------------------------------------------------------------------


def test_contradiction_blocked_reads_as_a_plain_sentence():
    row = _base_row(
        payload={
            "reason": "contradicts verdict blocked: claim was already superseded",
            "note_slug": "signal-risk-northlake",
            "old_claim_id": 10,
            "attempted_new_claim_id": 11,
        }
    )

    card = present_row(row)

    assert "already been corrected" in card.headline
    assert card.open_url == "/notes/signal-risk-northlake"


def test_contradiction_downgraded_reads_as_a_plain_sentence():
    row = _base_row(
        payload={
            "reason": (
                "contradicts verdict downgraded to extends: not both explicitly dated, "
                "or not strictly later"
            ),
            "note_slug": "signal-risk-northlake",
            "old_claim_id": 10,
            "new_claim_id": 12,
        }
    )

    card = present_row(row)

    assert "couldn't prove it" in card.headline


# ---------------------------------------------------------------------
# other / unrecognised reason -- the fallback, never a raw dict repr.
# ---------------------------------------------------------------------


def test_unrecognised_other_row_falls_back_to_plain_language_plus_the_raw_payload_in_the_fold():
    row = _base_row(payload={"note_slug": "f1", "reasons": ["numeral_not_supplied"]})

    card = present_row(row)

    assert "didn't record a plain-language reason" in card.headline
    assert "{'note_slug'" not in card.headline
    technical = dict(card.technical)
    assert technical["note_slug"] == "f1"
    assert "numeral_not_supplied" in technical["reasons"]


def test_unrecognised_kind_does_not_crash():
    row = _base_row(kind="some_future_kind", payload={"whatever": "value"})
    card = present_row(row)
    assert card.headline


# ---------------------------------------------------------------------
# present_rows / kind_counts
# ---------------------------------------------------------------------


def test_present_rows_dispatches_every_row():
    rows = [
        _base_row(id=1, kind="unknown_entity", payload={"mention": "Dayton", "context": {}}),
        _base_row(id=2, kind="failed_consolidation", payload={"note_slug": "x", "reasons": []}),
    ]
    cards = present_rows(rows)
    assert [c.row_id for c in cards] == [1, 2]


def test_kind_counts_groups_by_category_in_first_seen_order():
    rows = [
        _base_row(id=1, kind="unknown_entity", payload={"mention": "A", "context": {}}),
        _base_row(id=2, kind="failed_consolidation", payload={"note_slug": "x", "reasons": []}),
        _base_row(id=3, kind="unknown_entity", payload={"mention": "B", "context": {}}),
    ]
    cards = present_rows(rows)
    counts = kind_counts(cards)
    assert counts == [("Unknown name", 2), ("Tidy-up failed", 1)]


# ---------------------------------------------------------------------
# owner_proposal (01-spec/actions-and-owners-2026-09-19.md, Phase 1)
# ---------------------------------------------------------------------


def test_owner_proposal_names_the_top_candidate_and_its_evidence():
    row = _base_row(
        kind="owner_proposal",
        payload={
            "account_crm_id": "A-1011",
            "account_name": "Ashgrove Medical Center",
            "candidates": [
                {
                    "crm_id": "sam-okafor",
                    "name": "Sam Okafor",
                    "source_count": 3,
                    "last_seen": "2026-08-31",
                    "commitments": 1,
                },
                {
                    "crm_id": "dana-whitlock",
                    "name": "Dana Whitlock",
                    "source_count": 1,
                    "last_seen": "2026-03-02",
                    "commitments": 0,
                },
            ],
        },
    )

    card = present_row(row)

    assert "Sam Okafor" in card.headline
    assert "Ashgrove Medical Center" in card.headline
    assert "3 sources" in card.headline
    assert "2026-08-31" in card.headline
    assert "1 commitment" in card.headline
    assert card.category_label == "Proposed owner"
    assert card.crm_id == "A-1011"
    assert [c["crm_id"] for c in card.owner_candidates] == ["sam-okafor", "dana-whitlock"]

    labels = {c.label for c in card.choices}
    assert labels == {"Confirm owner", "Dismiss"}
    confirm = next(c for c in card.choices if c.label == "Confirm owner")
    assert confirm.action == "resolve"
    dismiss = next(c for c in card.choices if c.label == "Dismiss")
    assert dismiss.action == "dismiss"


def test_owner_proposal_with_no_candidates_does_not_crash():
    row = _base_row(
        kind="owner_proposal",
        payload={"account_crm_id": "A-1011", "account_name": "Ashgrove Medical Center"},
    )

    card = present_row(row)

    assert card.headline
    assert card.owner_candidates == []
