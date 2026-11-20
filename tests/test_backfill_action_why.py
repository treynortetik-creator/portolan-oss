"""Tests for `scripts/backfill_action_why.py` (build instructions, item
4: "for every resolved `action_card` review row with `payload['action_
id']` or a matching `actions.what`, persist the why ... and, where step 3
applies, merge the duplicate").

`scripts/` isn't a Python package (matching `tests/test_set_owners.py`'s
own setup) -- loaded by file path, skipped where the container image
excludes it.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import date, datetime, timedelta

from portolan.merge.actions import merge_action
from portolan.notes import NoteRow, NoteStatus, NoteType, action_note_slug, link_related, write_note
from portolan.notes.persist import ensure_entity_note, read_note
from portolan.registry import Registry
from portolan.report.action_why import build_why_context
from portolan.util import clock as clock_module
from tests._paths import require_repo_path
from tests.merge_helpers import (
    ASHGROVE_MEDICAL,
    SAM_OKAFOR,
    build_action,
    loaded_registry_fixture,  # noqa: F401
    make_source_row,
)

_SCRIPT = require_repo_path("scripts", "backfill_action_why.py")


def _load_backfill():
    spec = importlib.util.spec_from_file_location("backfill_action_why_script", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["backfill_action_why_script"] = module
    spec.loader.exec_module(module)
    return module.backfill_action_why


def _queue_resolved_action_card(
    db, *, card: dict, action_id: int | None, resolved_by: str = "Old Editor"
) -> int:
    payload = {"card": card, "week": "2026-W35"}
    if action_id is not None:
        payload["action_id"] = action_id
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO review_queue (kind, payload, status, resolved_at, resolved_by) "
            "VALUES ('action_card', %s::jsonb, 'resolved', %s, %s) RETURNING id",
            (json.dumps(payload), datetime.now() - timedelta(days=1), resolved_by),
        )
        row_id = cur.fetchone()[0]
    db.commit()
    return row_id


def _create_legacy_confirmed_action(
    db,
    registry: Registry,
    *,
    account_note_slug: str,
    what: str,
    source_slug: str,
    locator: str,
    owner_crm_id: str,
    due: date,
    editor: str = "Old Editor",
) -> tuple[int, str]:
    """Reproduces exactly what `confirm_action_card` used to write
    BEFORE this build (no `action_why`, no duplicate check) -- the shape
    of the "already wrongly duplicated" action the backfill script has to
    find and merge away."""
    owner_note = ensure_entity_note(db, registry, owner_crm_id)
    with db.cursor() as cur:
        cur.execute("SELECT id FROM registry_entities WHERE crm_id = %s", (owner_crm_id,))
        owner_dbid = cur.fetchone()[0]
        cur.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM actions")
        n = cur.fetchone()[0]
    slug = action_note_slug(n, what)
    now = clock_module.now()
    note = NoteRow(
        slug=slug,
        type=NoteType.ACTION,
        title=what[:120],
        description=f"Confirmed action from a researched card for [[{account_note_slug}]].",
        status=NoteStatus.LIVING,
        created=now,
        updated=now,
        related=[account_note_slug, owner_note.slug],
    )
    write_note(db, note, [], changed_by=editor, reason="legacy action card confirmed")
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO actions (note_slug, what, owner_entity_id, due, status, "
            "source_slug, locator) VALUES (%s, %s, %s, %s, 'open', %s, %s) RETURNING id",
            (slug, what, owner_dbid, due, source_slug, locator),
        )
        action_id = cur.fetchone()[0]
    db.commit()
    link_related(db, account_note_slug, slug, changed_by=editor, reason="new action")
    return action_id, slug


def _merge_pipeline_action(
    db, registry: Registry, *, what: str, locator: str
) -> tuple[int, str, int]:
    source = make_source_row(db, f"s-backfill-{locator}", source_date=date(2026, 8, 10))
    action = build_action(what, owner_mention=None, due=date(2026, 9, 1), locator=locator)
    outcome = merge_action(db, registry, action, source, entity_id=ASHGROVE_MEDICAL)
    with db.cursor() as cur:
        cur.execute("SELECT id FROM claims WHERE note_slug = %s", (outcome.note_slug,))
        claim_id = cur.fetchone()[0]
    return outcome.action_id, outcome.note_slug, claim_id


def _card_for(
    *, account: str, claim_id: int, claim_text: str, source_slug: str, locator: str, what: str
) -> dict:
    return {
        "run_id": "backfill-run",
        "candidate_slug": "backfill-candidate",
        "brief_version": "researcher-brief-v6",
        "account": account,
        "verdict": "action",
        "what": what,
        "evidence": [
            {
                "claim_id": claim_id,
                "text": claim_text,
                "quote": None,
                "source_slug": source_slug,
                "locator": locator,
            }
        ],
        "findings": [],
        "inference": [],
        "owner": SAM_OKAFOR,
        "due": "2026-11-01",
        "due_rule": "next_report",
        "theme": None,
        "confidence": "high",
        "confidence_refs": {"evidence_refs": [0], "finding_refs": []},
    }


def test_backfill_persists_why_for_a_resolved_row_with_no_duplicate(loaded_registry: Registry, db):
    backfill_action_why = _load_backfill()
    action_id, _note_slug, claim_id = _merge_pipeline_action(
        db, loaded_registry, what="Send a written delivery root-cause summary.", locator="p1"
    )
    account_slug = ensure_entity_note(db, loaded_registry, ASHGROVE_MEDICAL).slug
    with db.cursor() as cur:
        cur.execute(
            "SELECT claim_text, source_slug, locator FROM claims WHERE id = %s", (claim_id,)
        )
        claim_text, source_slug, locator = cur.fetchone()
    card = _card_for(
        account=account_slug,
        claim_id=claim_id,
        claim_text=claim_text,
        source_slug=source_slug,
        locator=locator,
        what="Send a written delivery root-cause summary.",
    )
    _queue_resolved_action_card(db, card=card, action_id=action_id)

    counts = backfill_action_why(db)

    assert counts["persisted"] == 1
    assert counts["merged_duplicate"] == 0
    why = build_why_context(db, action_id)
    assert why is not None
    assert why["confirmed_by"] == "Old Editor"
    assert why["evidence"][0]["claim_id"] == claim_id

    # Idempotent.
    second = backfill_action_why(db)
    assert second["persisted"] == 1
    with db.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM action_why WHERE action_id = %s", (action_id,))
        assert cur.fetchone()[0] == 1


def test_backfill_merges_a_duplicate_action_into_the_older_one(loaded_registry: Registry, db):
    """The worked example (build instructions): action-15 duplicated
    action-14 because both cite the same claim. Here: `pipeline_action`
    is the older, pre-existing action (claim on its own note); `legacy_
    action` is what the old, pre-fix `confirm_action_card` wrongly
    created alongside it, from a card whose evidence includes that same
    claim."""
    backfill_action_why = _load_backfill()
    pipeline_action_id, _pipeline_note_slug, pipeline_claim_id = _merge_pipeline_action(
        db,
        loaded_registry,
        what="Review the request and follow up with available pricing options.",
        locator="p2",
    )
    account_slug = ensure_entity_note(db, loaded_registry, ASHGROVE_MEDICAL).slug
    with db.cursor() as cur:
        cur.execute(
            "SELECT claim_text, source_slug, locator FROM claims WHERE id = %s",
            (pipeline_claim_id,),
        )
        claim_text, source_slug, locator = cur.fetchone()

    legacy_action_id, legacy_note_slug = _create_legacy_confirmed_action(
        db,
        loaded_registry,
        account_note_slug=account_slug,
        what="Follow up with Ashgrove Medical on pricing.",
        source_slug=source_slug,
        locator=locator,
        owner_crm_id=SAM_OKAFOR,
        due=date(2026, 10, 1),
        editor="Old Editor",
    )
    assert legacy_action_id > pipeline_action_id

    card = _card_for(
        account=account_slug,
        claim_id=pipeline_claim_id,
        claim_text=claim_text,
        source_slug=source_slug,
        locator=locator,
        what="Follow up with Ashgrove Medical on pricing.",
    )
    _queue_resolved_action_card(db, card=card, action_id=legacy_action_id)

    with db.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM actions")
        (before_count,) = cur.fetchone()

    counts = backfill_action_why(db)

    assert counts["merged_duplicate"] == 1
    assert counts["persisted"] == 0
    with db.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM actions")
        (after_count,) = cur.fetchone()
    assert after_count == before_count  # nothing new inserted or deleted

    with db.cursor() as cur:
        cur.execute("SELECT status, due FROM actions WHERE id = %s", (legacy_action_id,))
        legacy_status, _legacy_due = cur.fetchone()
        cur.execute("SELECT status, due FROM actions WHERE id = %s", (pipeline_action_id,))
        pipeline_status, pipeline_due = cur.fetchone()
    assert legacy_status == "dropped"
    assert pipeline_status == "open"
    assert pipeline_due == date(2026, 11, 1)  # absorbed from the card

    legacy_note, _claims = read_note(db, legacy_note_slug)
    assert legacy_note.status == NoteStatus.ARCHIVED

    why = build_why_context(db, pipeline_action_id)
    assert why is not None
    assert why["evidence"][0]["claim_id"] == pipeline_claim_id

    with db.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM changelog WHERE kind = 'action_duplicate_merged'")
        assert cur.fetchone()[0] == 1

    # Idempotent: a second run merges nothing further.
    second_counts = backfill_action_why(db)
    assert second_counts["merged_duplicate"] == 1  # still matched, just a no-op merge
    with db.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM changelog WHERE kind = 'action_duplicate_merged'")
        assert cur.fetchone()[0] == 1


def test_backfill_skips_a_row_with_no_card_and_a_row_with_no_action_match(db):
    backfill_action_why = _load_backfill()
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO review_queue (kind, payload, status, resolved_at, resolved_by) "
            "VALUES ('action_card', '{}'::jsonb, 'resolved', now(), 'Old Editor')"
        )
        cur.execute(
            "INSERT INTO review_queue (kind, payload, status, resolved_at, resolved_by) "
            "VALUES ('action_card', %s::jsonb, 'resolved', now(), 'Old Editor')",
            (json.dumps({"card": {"what": "No action ever matches this text."}}),),
        )
    db.commit()

    counts = backfill_action_why(db)

    assert counts["skipped_no_card"] == 1
    assert counts["skipped_no_action_match"] == 1
    assert counts["persisted"] == 0
    assert counts["merged_duplicate"] == 0
