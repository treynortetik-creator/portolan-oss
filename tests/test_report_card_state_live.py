"""The report page reads each suggested action card's state from the review
queue at view time (demo beat 7, 2026-09-22: confirm a card on `/review`,
then open the weekly report). Before, the report said "proposed" until
someone paid to regenerate it."""

from __future__ import annotations

from datetime import UTC, datetime

import psycopg
from fastapi.testclient import TestClient

from portolan.notes import NoteRow, NoteStatus, NoteType, write_note
from portolan.notes.persist import ensure_entity_note, read_note
from portolan.registry import Registry
from portolan.report.researcher import action_card_report_lines, refresh_action_card_lines
from tests.merge_helpers import ASHGROVE_MEDICAL
from tests.test_review_action_card_route import (  # noqa: F401
    _merge_complaint,
    _queue_action_card,
    auth_cookies,
    loaded_registry_fixture,
    review_client,
)

WEEK = "2026-W36"
SLUG = f"report-{WEEK}"


def _write_report(db: psycopg.Connection) -> None:
    """A W36 report written the way `build_weekly` writes one: the card
    lines as they read at generation time, under Suggested actions."""
    lines = "\n".join(f"- {line}" for line in action_card_report_lines(db, week=WEEK))
    body = f"### Executive summary\n\n**Suggested actions**\n{lines}\n- Call Ashgrove.\n"
    now = datetime.now(UTC)
    note = NoteRow(
        slug=SLUG,
        type=NoteType.REPORT,
        title=f"Weekly report — {WEEK}",
        description="fixture",
        status=NoteStatus.LIVING,
        created=now,
        updated=now,
        synthesis=body,
    )
    write_note(db, note, [], changed_by="report", reason="weekly report")
    with db.cursor() as cur:
        cur.execute("INSERT INTO reports (week, note_slug) VALUES (%s, %s)", (WEEK, SLUG))
    db.commit()


def _queue_two_cards(db: psycopg.Connection, registry: Registry) -> tuple[int, int]:
    account_slug = ensure_entity_note(db, registry, ASHGROVE_MEDICAL).slug
    claim_id, source_slug, claim_text, quote = _merge_complaint(db, registry)
    first = _queue_action_card(
        db,
        account_slug=account_slug,
        claim_id=claim_id,
        claim_text=claim_text,
        quote=quote,
        source_slug=source_slug,
        week=WEEK,
    )
    second = _queue_action_card(
        db,
        account_slug=account_slug,
        claim_id=claim_id,
        claim_text=claim_text,
        quote=quote,
        source_slug=source_slug,
        week=WEEK,
    )
    with db.cursor() as cur:
        cur.execute(
            "UPDATE review_queue SET payload = jsonb_set(payload, '{card,what}', "
            "to_jsonb('Ask Ashgrove about the second shipment.'::text)) WHERE id = %s",
            (second,),
        )
    db.commit()
    return first, second


def test_confirm_then_dismiss_shows_on_the_report_without_a_regeneration(
    review_client: TestClient,  # noqa: F811
    db: psycopg.Connection,
    auth_cookies,  # noqa: F811
    loaded_registry: Registry,
) -> None:
    first, second = _queue_two_cards(db, loaded_registry)
    _write_report(db)

    before = review_client.get(f"/reports/{SLUG}", cookies=auth_cookies).text
    assert before.count("(proposed, confidence: medium)") == 2

    confirm = review_client.post(
        f"/review/{first}/resolve", cookies=auth_cookies, follow_redirects=False
    )
    assert "action_confirmed" in confirm.headers["location"]
    review_client.post(f"/review/{second}/dismiss", cookies=auth_cookies, follow_redirects=False)

    with db.cursor() as cur:
        cur.execute("SELECT due FROM actions ORDER BY id DESC LIMIT 1")
        (due,) = cur.fetchone()
    stored = read_note(db, SLUG)[0].synthesis or ""
    assert stored.count("(proposed, confidence: medium)") == 2  # the text was not rewritten

    after = review_client.get(f"/reports/{SLUG}", cookies=auth_cookies).text
    assert "(proposed, confidence: medium)" not in after
    assert (
        "Follow up on the late delivery. "
        f"(confirmed by Test Editor, owner Sam Okafor, due {due.isoformat()})" in after
    )
    assert "Ask Ashgrove about the second shipment. (dismissed)" in after
    assert "Call Ashgrove." in after

    md = review_client.get(f"/reports/{SLUG}.md", cookies=auth_cookies).text
    assert f"(confirmed by Test Editor, owner Sam Okafor, due {due.isoformat()})" in md
    assert "(dismissed)" in md


def test_a_line_with_no_current_card_is_left_as_written(db: psycopg.Connection) -> None:
    body = "**Suggested actions**\n- [action] Something old. (proposed, confidence: low)\n"
    assert refresh_action_card_lines(db, body, week=WEEK) == body
