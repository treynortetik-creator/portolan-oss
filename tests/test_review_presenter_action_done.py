"""The `action_done` review card (`portolan.report.action_done`). Pure, no
database, like every other presenter test."""

from __future__ import annotations

from datetime import datetime

from portolan.web.review_presenter import category_label_for, present_row


def _row(**overrides):
    row = {
        "id": 7,
        "kind": "action_done",
        "status": "open",
        "created": datetime(2026, 9, 22, 9, 0, 0),
        "source_slug": "S-email-9",
        "payload": {
            "action_id": 3,
            "action_slug": "action-3-follow-up-with-support",
            "what": "Follow up with support about the late delivery",
            "account": "ashgrove-medical-center",
            "owner_crm_id": "sam-okafor",
            "action_date": "2026-08-20",
            "claim_id": 11,
            "evidence": {
                "claim_id": 11,
                "text": "Sam Okafor emailed the support team about the late delivery.",
                "quote": "I emailed support",
                "source_slug": "S-email-9",
                "locator": "p2",
            },
            "fact_type": "commitment",
            "speaker": "Sam Okafor",
            "claim_date": "2026-08-21",
            "finding_id": 5,
            "run_id": "action-done-abc",
        },
        "done_context": {
            "account_name": "Ashgrove Medical Center",
            "owner_name": "Sam Okafor",
            "status": "open",
            "due": None,
            "finding_text": "1 commitment claims recorded for Ashgrove after 2026-08-20; 0 before.",
        },
    }
    row.update(overrides)
    return row


def test_the_card_names_the_claim_the_action_and_the_original() -> None:
    card = present_row(_row())

    assert card.kind == "action_done"
    assert card.category_label == "Proposed done" == category_label_for("action_done", None)
    assert card.headline == (
        "Sam Okafor recorded something on 2026-08-21 that looks like it answers the open action "
        "“Follow up with support about the late delivery” for Ashgrove Medical Center, so "
        "Portolan proposes marking it done."
    )
    assert card.claim_text == "Sam Okafor emailed the support team about the late delivery."
    assert card.quote == "I emailed support"
    assert card.open_url == "/originals/S-email-9?at=p2"
    assert card.done_action_slug == "action-3-follow-up-with-support"
    assert card.done_finding_text.startswith("1 commitment claims")
    assert [c.label for c in card.choices] == ["Confirm done", "Dismiss"]


def test_a_card_without_enrichment_still_renders() -> None:
    card = present_row(_row(done_context=None, payload={"action_id": 3}))

    assert "(no description)" in card.headline
    assert "(unknown account)" in card.headline
    assert card.claim_text is None
    assert card.open_url == "/originals/S-email-9"
