"""Backfill `action_why` for actions confirmed before that table existed
(follow-up to `01-spec/actions-and-owners-2026-09-19.md`'s Phase 2,
2026-09-20 build instructions, item 4: "for every resolved `action_card`
review row with `payload['action_id']` or a matching `actions.what`,
persist the why ... and, where step 3 applies, merge the duplicate").

Every `review_queue` row of `kind='action_card'` and `status='resolved'`
represents a card a human already confirmed through `/review`, before
`portolan.report.researcher.confirm_action_card` started persisting
`action_why` (and, separately, before it started catching the "no
duplicates" case at all -- the live graph's own action-15 duplicating
action-14, `01-spec/actions-and-owners-2026-09-19.md`'s worked example).
For each such row:

1. Resolve which `actions` row it confirmed: `payload['action_id']`
   (stamped at confirm time going forward) when present, else the same
   fallback `portolan.report.researcher._confirmed_action_state` already
   uses -- the most recent `actions` row whose `what` matches the card's
   own `what` text byte-for-byte.
2. Check for a duplicate the SAME way `confirm_action_card` does now
   (`portolan.report.action_why.find_duplicate_open_action`): does any
   evidence claim id already live on some OTHER open action's own note?
   If so, the OLDER of the two (by `actions.id`) is kept, the newer is
   merged into it (`portolan.report.action_why.merge_duplicate_action`:
   owner/due move onto the survivor, the newer note is archived -- never
   deleted -- and a changelog row records it).
3. Persist the why (`portolan.report.action_why.persist_action_why`) onto
   whichever action survives, attributed to the row's own `resolved_by`
   (the human who actually clicked Confirm) and `resolved_at` -- never
   "now", and never this script's own name.

**Idempotent.** Both `persist_action_why` (an upsert) and `merge_
duplicate_action` (a no-op past the first merge) are safe to call again;
running this script twice changes nothing the second time.

**Prints counts only.** Never a card's text, a claim id, or the
connection string (`01-spec/CLAUDE.md`'s "never print deployment
variables" rule) -- just how many rows fell into each bucket.

**Never run against the live database from an agent session.** the maintainer
runs this one himself, the same way `scripts/set_owners.py` is never run
here directly. Prove it first on a scratch copy:

    createdb portolan_why_scratch
    pg_dump portolan_reseed_20260919 | psql portolan_why_scratch
    uv run python scripts/backfill_action_why.py \\
        --database-url postgresql://localhost/portolan_why_scratch
    dropdb portolan_why_scratch

Usage:
    uv run python scripts/backfill_action_why.py --database-url "$DATABASE_URL"
"""

from __future__ import annotations

import argparse
import os
from typing import Any

import psycopg

from portolan.report.action_why import (
    find_duplicate_open_action,
    merge_duplicate_action,
    persist_action_why,
)

__all__ = ["backfill_action_why"]


def _resolve_action_id(
    conn: psycopg.Connection, *, payload: dict[str, Any], card: dict[str, Any]
) -> int | None:
    """`payload['action_id']` when stamped, else the most recent `actions`
    row whose `what` matches the card's own `what` verbatim -- the same
    fallback order `action_card_report_lines`/`_confirmed_action_state`
    (`portolan.report.researcher`) already use to find a resolved card's
    real action."""
    action_id = payload.get("action_id")
    if action_id is not None:
        return int(action_id)
    what = card.get("what")
    if not what:
        return None
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM actions WHERE what = %s ORDER BY id DESC LIMIT 1", (what,))
        row = cur.fetchone()
    return row[0] if row is not None else None


def backfill_action_why(conn: psycopg.Connection) -> dict[str, int]:
    """Idempotent. Returns counts only -- never a card's text or a claim
    id, so a caller can safely print the result."""
    counts = {
        "total_rows": 0,
        "persisted": 0,
        "merged_duplicate": 0,
        "skipped_no_card": 0,
        "skipped_no_action_match": 0,
    }

    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, payload, resolved_by, resolved_at FROM review_queue "
            "WHERE kind = 'action_card' AND status = 'resolved' ORDER BY id"
        )
        rows = cur.fetchall()

    for row_id, payload, resolved_by, resolved_at in rows:
        counts["total_rows"] += 1
        payload = payload or {}
        card = payload.get("card") or {}
        if not card:
            counts["skipped_no_card"] += 1
            continue

        action_id = _resolve_action_id(conn, payload=payload, card=card)
        if action_id is None:
            counts["skipped_no_action_match"] += 1
            continue

        editor = resolved_by or "unknown"
        evidence_ids = [
            e.get("claim_id") for e in (card.get("evidence") or []) if e.get("claim_id") is not None
        ]
        account_note_slug = card.get("account")
        duplicate_id = None
        if account_note_slug:
            duplicate_id = find_duplicate_open_action(
                conn,
                account_note_slug=account_note_slug,
                evidence_claim_ids=evidence_ids,
                exclude_action_id=action_id,
            )

        if duplicate_id is not None:
            keep_id, drop_id = sorted((action_id, duplicate_id))
            merge_duplicate_action(
                conn,
                keep_action_id=keep_id,
                drop_action_id=drop_id,
                card=card,
                editor=f"backfill (originally confirmed by {editor})",
                reason="duplicate action merged by backfill_action_why",
                commit=True,
            )
            persist_action_why(
                conn,
                action_id=keep_id,
                card=card,
                editor=editor,
                review_row_id=row_id,
                confirmed_at=resolved_at,
                commit=True,
            )
            counts["merged_duplicate"] += 1
            continue

        persist_action_why(
            conn,
            action_id=action_id,
            card=card,
            editor=editor,
            review_row_id=row_id,
            confirmed_at=resolved_at,
            commit=True,
        )
        counts["persisted"] += 1

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill action_why for action cards confirmed before it existed."
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help="Postgres connection string. Defaults to $DATABASE_URL. Never printed.",
    )
    args = parser.parse_args()

    database_url = args.database_url or os.environ.get("DATABASE_URL")
    if not database_url:
        raise SystemExit("no database url given: pass --database-url or set DATABASE_URL")

    conn = psycopg.connect(database_url)
    try:
        counts = backfill_action_why(conn)
    finally:
        conn.close()

    for name, value in counts.items():
        print(f"{name}={value}")


if __name__ == "__main__":
    main()
