"""Pilot activity log: what people do in the app, for the 30-day pilot's
pass and kill lines (01-spec/red-team-2026-09-24.md sec 6). Migration 0037.

`record_activity` writes one `activity_events` row. It never raises: a
failed write is logged and the person's request carries on. `user_name` is
the login display name (`Session.editor`) or, for an email open or click,
the recipient label the link was signed with. No IP address, user agent or
query string is stored.

`detail` per kind (small fields only):
- `page_view`: none; `route` is the route template, `target` its first path
  parameter (a note slug, account slug or conversation id).
- `chat_question`: `question`, `answer_chars`, `cited_sentences`,
  `citations`, `withheld`, `latency_ms`, `cost_usd`, `model`, `mode`
  (`answered`, `gaps`, `nothing_recorded`, `index_broken` or `error`),
  `conversation_id`, `message_id`, `regenerate`.
- `citation_click`: `message_id`; `target` is the source slug.
- `review_decision`: `decision` (`resolve`, `dismiss`, `add_alias`),
  `row_kind`, `notice`; `target` is the review row id.
- `action_confirm`: `how` (`card_confirmed`, `done_confirmed`, `marked_done`,
  `marked_dropped`); `target` is the action id or review row id.
- `feedback`: `rating`, `message_id`.
- `email_open` / `email_click`: `email_id`, and for a click the `url`.

`purge_expired` applies `Settings.activity_retention_days`; `delete_user`
removes one person's rows from all three pilot tables. Both are on the CLI
(`python -m portolan.activity`).
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

__all__ = [
    "KINDS",
    "delete_user",
    "purge_expired",
    "record_activity",
]

_logger = logging.getLogger(__name__)

# Must match the CHECK on activity_events.kind (0037_pilot_activity.sql).
KINDS = frozenset(
    {
        "page_view",
        "chat_question",
        "citation_click",
        "review_decision",
        "action_confirm",
        "feedback",
        "email_open",
        "email_click",
    }
)


def record_activity(
    conn: psycopg.Connection,
    *,
    kind: str,
    user_name: str | None,
    route: str | None = None,
    target: str | None = None,
    detail: dict[str, Any] | None = None,
    occurred_at: datetime | None = None,
) -> bool:
    """One `activity_events` row; True when written. Runs in its own
    transaction (a savepoint when the caller is inside one), so a failure
    rolls back only this row and never the caller's work."""
    if kind not in KINDS:
        raise ValueError(f"unknown activity kind {kind!r}")
    try:
        with conn.transaction(), conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO activity_events
                    (occurred_at, user_name, kind, route, target, detail)
                VALUES (coalesce(%s, now()), %s, %s, %s, %s, %s)
                """,
                (occurred_at, user_name, kind, route, target, Jsonb(detail or {})),
            )
    except psycopg.Error:
        _logger.exception("activity event not recorded (kind=%s)", kind)
        return False
    return True


def purge_expired(
    conn: psycopg.Connection, *, retention_days: int, now: datetime | None = None
) -> dict[str, int]:
    """Delete `activity_events` and `answer_feedback` rows older than
    `retention_days`. Eval candidates are kept: a person curates them.
    `retention_days <= 0` deletes nothing."""
    if retention_days <= 0:
        return {"activity_events": 0, "answer_feedback": 0}
    cutoff = (now or datetime.now(UTC)) - timedelta(days=retention_days)
    with conn.transaction(), conn.cursor() as cur:
        cur.execute("DELETE FROM activity_events WHERE occurred_at < %s", (cutoff,))
        events = cur.rowcount
        cur.execute("DELETE FROM answer_feedback WHERE updated_at < %s", (cutoff,))
        feedback = cur.rowcount
    return {"activity_events": events, "answer_feedback": feedback}


def delete_user(conn: psycopg.Connection, user_name: str) -> dict[str, int]:
    """Every pilot-table row for `user_name`: activity events, feedback, and
    the eval candidates copied from their questions. Chat conversations are
    the person's own history and are deleted from the chat sidebar."""
    with conn.transaction(), conn.cursor() as cur:
        cur.execute("DELETE FROM eval_candidates WHERE user_name = %s", (user_name,))
        candidates = cur.rowcount
        cur.execute("DELETE FROM answer_feedback WHERE user_name = %s", (user_name,))
        feedback = cur.rowcount
        cur.execute("DELETE FROM activity_events WHERE user_name = %s", (user_name,))
        events = cur.rowcount
    return {
        "activity_events": events,
        "answer_feedback": feedback,
        "eval_candidates": candidates,
    }
