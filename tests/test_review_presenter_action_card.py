"""Tests for the `action_card` review card, revised contract (`01-spec/
actions-and-owners-2026-09-19.md`, "Phase 2, revised contract"). Pure --
no database, matching every other presenter test's own shape
(`tests/test_review_presenter.py`).
"""

from __future__ import annotations

from datetime import datetime

from portolan.web.review_presenter import present_row


def _row(**overrides):
    row = {
        "id": 1,
        "kind": "action_card",
        "status": "open",
        "created": datetime(2026, 9, 19, 12, 0, 0),
        "reason": None,
        "payload": {
            "card": {
                "run_id": "run-1",
                "brief_version": "researcher-brief-v4",
                "candidate_slug": "theme-rising:theme-late-deliveries:ashgrove-medical-center",
                "account": "ashgrove-medical-center",
                "verdict": "action",
                "what": "Follow up with Ashgrove Medical Center about late deliveries.",
                "owner": "sam-okafor",
                "theme": "theme-late-deliveries",
                "due": "2026-10-01",
                "due_rule": "next_report",
                "evidence": [
                    {
                        "claim_id": 10,
                        "text": "Ashgrove Medical Center received a late delivery.",
                        "quote": "late delivery",
                        "source_slug": "S-call-1",
                        "locator": "00:01:00",
                    }
                ],
                "findings": [42],
                "inference": [
                    {
                        "sentence": "No follow-up has been logged yet.",
                        "evidence_refs": [0],
                        "finding_refs": [0],
                    }
                ],
                "confidence": "medium",
                "confidence_refs": {"evidence_refs": [0], "finding_refs": []},
                "trail": {"tool_call_count": 5, "wall_time_s": 12.3, "model": "claude-sonnet-5"},
            }
        },
        "action_context": {
            "account_name": "Ashgrove Medical Center",
            "theme_name": "Late deliveries",
            "owner_name": "Sam Okafor",
            "evidence": [
                {
                    "claim_id": 10,
                    "text": "Ashgrove Medical Center received a late delivery.",
                    "quote": "late delivery",
                    "source_slug": "S-call-1",
                    "locator": "00:01:00",
                }
            ],
            "findings": [
                {
                    "id": 42,
                    "tool": "absence_check",
                    "arguments": {"fact_types": ["commitment"]},
                    "result": {"count": 0},
                }
            ],
            "inference": [
                {
                    "sentence": "No follow-up has been logged yet.",
                    "evidence_refs": [0],
                    "finding_refs": [0],
                    "evidence": [
                        {
                            "claim_id": 10,
                            "text": "Ashgrove Medical Center received a late delivery.",
                            "quote": "late delivery",
                            "source_slug": "S-call-1",
                            "locator": "00:01:00",
                        }
                    ],
                    "findings": [
                        {
                            "id": 42,
                            "tool": "absence_check",
                            "arguments": {"fact_types": ["commitment"]},
                            "result": {"count": 0},
                        }
                    ],
                }
            ],
            "confidence_refs": {"evidence": [], "findings": []},
            "trail": {"tool_call_count": 5, "wall_time_s": 12.3, "model": "claude-sonnet-5"},
            "owner_choices": [
                {"crm_id": "sam-okafor", "name": "Sam Okafor"},
                {"crm_id": "priya-natarajan", "name": "Priya Natarajan"},
            ],
        },
    }
    row.update(overrides)
    return row


def test_action_card_headline_names_the_account_and_the_action():
    card = present_row(_row())
    assert "Ashgrove Medical Center" in card.headline
    assert "Follow up with Ashgrove Medical Center about late deliveries." in card.headline


def test_action_card_carries_verdict_owner_theme_due_and_confidence():
    card = present_row(_row())
    assert card.action_verdict == "action"
    assert card.action_owner_name == "Sam Okafor"
    assert card.action_owner_crm_id == "sam-okafor"
    assert card.action_theme_name == "Late deliveries"
    assert card.action_theme_slug == "theme-late-deliveries"
    assert card.action_due == "2026-10-01"
    assert card.action_due_rule == "next_report"
    assert card.action_confidence == "medium"


def test_action_card_inference_carries_its_own_resolved_refs():
    card = present_row(_row())
    assert len(card.action_inference) == 1
    sentence = card.action_inference[0]
    assert sentence["sentence"] == "No follow-up has been logged yet."
    assert sentence["evidence"][0]["source_slug"] == "S-call-1"
    assert sentence["findings"][0]["tool"] == "absence_check"


def test_action_card_evidence_and_findings_lists():
    card = present_row(_row())
    assert len(card.action_evidence) == 1
    assert card.action_evidence[0]["text"] == "Ashgrove Medical Center received a late delivery."
    assert len(card.action_findings) == 1
    assert card.action_findings[0]["tool"] == "absence_check"


def test_action_card_trail_is_carried_through():
    card = present_row(_row())
    assert card.action_trail["tool_call_count"] == 5


def test_action_card_offers_confirm_and_dismiss_choices():
    card = present_row(_row())
    actions = {c.action for c in card.choices}
    assert actions == {"resolve", "dismiss"}


def test_action_card_owner_choices_for_the_edit_form():
    card = present_row(_row())
    assert {c["crm_id"] for c in card.action_owner_choices} == {"sam-okafor", "priya-natarajan"}


def test_action_card_category_label():
    card = present_row(_row())
    assert card.category_label == "Proposed action"


def test_action_card_trigger_stale_verdict_shows_a_different_why_it_matters():
    row = _row()
    row["payload"]["card"]["verdict"] = "trigger_stale"
    card = present_row(row)
    assert card.action_verdict == "trigger_stale"
    assert "stale trigger" in card.headline.lower()


def test_action_card_degrades_gracefully_with_no_enrichment():
    row = _row()
    del row["action_context"]
    card = present_row(row)
    assert card.action_what == "Follow up with Ashgrove Medical Center about late deliveries."
    assert card.action_inference == []
    assert card.action_owner_choices == []
