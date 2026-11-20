"""Feed 1: the diff (`01-spec/surfacing-and-weekly-report.md` "1. The diff").
Pure SQL, no model call.

- `notes_created` / `notes_updated`: a note is "updated" only if it existed
  before the window; otherwise it is "created", never both.
- `superseded_facts`: claims whose `superseded_at` falls in the window.
- `quiet_accounts`: accounts with no claim mentioning them in 30/60/90 days.

`build_diff` assembles all four into one `DiffFeed`.

`clock`: `notes.created`/`updated` are wall-clock write times, so a corpus
backfilled in one sitting shows every note as written "today".
`clock="world"` measures the same created-vs-updated split on
`claims.recorded` (the source's own date) instead. `superseded_facts` and
`quiet_accounts` take no `clock`: neither reads `notes.created`/`updated`.
"""

from __future__ import annotations

from datetime import date

import psycopg

from portolan.report.models import DiffFeed, NoteChange, QuietAccounts, SupersededFact
from portolan.util import clock as clock_module

__all__ = [
    "QUIET_ACCOUNT_THRESHOLDS",
    "build_diff",
    "notes_created",
    "notes_updated",
    "quiet_accounts",
    "superseded_facts",
]


def _note_change_rows(conn: psycopg.Connection, slugs: list[str]) -> list[NoteChange]:
    if not slugs:
        return []
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT slug, type, title, created, updated, status
            FROM notes WHERE slug = ANY(%s)
            ORDER BY updated DESC, slug
            """,
            (slugs,),
        )
        rows = cur.fetchall()
    return [
        NoteChange(
            slug=slug,
            type=type_,
            title=title,
            # Deployment-zone dates; `.date()` would use the session's zone.
            created=clock_module.world_date(created),
            updated=clock_module.world_date(updated),
            status=status,
        )
        for slug, type_, title, created, updated, status in rows
    ]


_VALID_CLOCKS = ("record", "world")


def _check_clock(clock: str) -> None:
    if clock not in _VALID_CLOCKS:
        raise ValueError(f"clock must be one of {_VALID_CLOCKS!r}, got {clock!r}")


def notes_created(
    conn: psycopg.Connection, *, since: date, as_of: date, clock: str = "record"
) -> list[NoteChange]:
    """Notes that did not exist before the window. `clock="record"`:
    `created` in `[since, as_of]`. `clock="world"`: a claim `recorded` in
    `[since, as_of]` and none before `since` (a note with no claims never
    appears in world mode)."""
    _check_clock(clock)
    if clock == "record":
        query = (
            f"SELECT slug FROM notes "
            f"WHERE {clock_module.world_date_sql('created')} BETWEEN %s AND %s"
        )
        params = (since, as_of)
    else:
        query = f"""
            SELECT c.note_slug
            FROM claims c
            WHERE {clock_module.world_date_sql("c.recorded")} BETWEEN %s AND %s
              AND NOT EXISTS (
                  SELECT 1 FROM claims c2
                  WHERE c2.note_slug = c.note_slug
                    AND {clock_module.world_date_sql("c2.recorded")} < %s
              )
            GROUP BY c.note_slug
        """
        params = (since, as_of, since)
    with conn.cursor() as cur:
        cur.execute(query, params)
        slugs = [r[0] for r in cur.fetchall()]
    return _note_change_rows(conn, slugs)


def notes_updated(
    conn: psycopg.Connection, *, since: date, as_of: date, clock: str = "record"
) -> list[NoteChange]:
    """Notes that existed before the window and were touched inside it; a
    note created in the window is reported only by `notes_created`.
    `clock="record"`: `updated` in `[since, as_of]`, `created` before `since`,
    and `version > 1` (written again after its first write). `clock="world"`:
    a claim `recorded` in `[since, as_of]` and another before `since`."""
    _check_clock(clock)
    if clock == "record":
        query = f"""
            SELECT slug FROM notes
            WHERE {clock_module.world_date_sql("updated")} BETWEEN %s AND %s
              AND {clock_module.world_date_sql("created")} < %s
              AND version > 1
        """
        params = (since, as_of, since)
    else:
        query = f"""
            SELECT c.note_slug
            FROM claims c
            WHERE {clock_module.world_date_sql("c.recorded")} BETWEEN %s AND %s
              AND EXISTS (
                  SELECT 1 FROM claims c2
                  WHERE c2.note_slug = c.note_slug
                    AND {clock_module.world_date_sql("c2.recorded")} < %s
              )
            GROUP BY c.note_slug
        """
        params = (since, as_of, since)
    with conn.cursor() as cur:
        cur.execute(query, params)
        slugs = [r[0] for r in cur.fetchall()]
    return _note_change_rows(conn, slugs)


def superseded_facts(conn: psycopg.Connection, *, since: date, as_of: date) -> list[SupersededFact]:
    """Claims superseded in the window (record-time clock, spec sec 2)."""
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT old.note_slug, old.id, old.superseded_by, old.claim_text,
                   new.claim_text, new.source_slug, new.locator, old.superseded_at
            FROM claims old
            JOIN claims new ON new.id = old.superseded_by
            WHERE {clock_module.world_date_sql("old.superseded_at")} BETWEEN %s AND %s
            ORDER BY old.superseded_at
            """,
            (since, as_of),
        )
        rows = cur.fetchall()
    return [
        SupersededFact(
            note_slug=row[0],
            old_claim_id=row[1],
            new_claim_id=row[2],
            old_text=row[3],
            new_text=row[4],
            new_source_slug=row[5],
            new_locator=row[6],
            superseded_at=row[7],
        )
        for row in rows
    ]


# Public so prose callers can give the model these day counts as citable
# `NumberFacts`; otherwise "30 days" in prose fails the numeral gate.
QUIET_ACCOUNT_THRESHOLDS = (30, 60, 90)


def quiet_accounts(conn: psycopg.Connection, as_of: date) -> QuietAccounts:
    """Living, active accounts with no claim mentioning them in the last
    30/60/90 days as of `as_of`.

    "Mentioning" is any `[[account-slug]]` in `claims.claim_text`, not just
    claims homed on the account note. Inactive accounts
    (`registry_entities.active = false`, e.g. churned) are excluded: their
    silence is already explained.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT n.slug
            FROM notes n
            JOIN registry_entities re ON re.note_slug = n.slug
            WHERE n.type = 'account' AND n.status = 'living' AND re.active = true
            ORDER BY n.slug
            """
        )
        account_slugs = [r[0] for r in cur.fetchall()]

    # One query for every account rather than one LIKE scan per account.
    last_touch: dict[str, date | None] = dict.fromkeys(account_slugs)
    if account_slugs:
        with conn.cursor() as cur:
            # The `as_of` bound keeps a re-run of a past week reproducible. It
            # sits in the ON clause: a WHERE would make this an inner join and
            # drop never-touched accounts.
            cur.execute(
                f"""
                SELECT a.slug, MAX(c.recorded)
                FROM unnest(%s::text[]) AS a(slug)
                LEFT JOIN claims c ON c.claim_text LIKE '%%[[' || a.slug || ']]%%'
                    AND {clock_module.world_date_sql("c.recorded")} <= %s
                GROUP BY a.slug
                """,
                (account_slugs, as_of),
            )
            for slug, max_recorded in cur.fetchall():
                # Deployment-zone date; `.date()` would use the session's zone.
                last_touch[slug] = clock_module.world_date(max_recorded) if max_recorded else None

    result = QuietAccounts(as_of=as_of)
    buckets = {30: result.quiet_30, 60: result.quiet_60, 90: result.quiet_90}
    for slug in account_slugs:
        last = last_touch[slug]
        age_days = None if last is None else (as_of - last).days
        for threshold in QUIET_ACCOUNT_THRESHOLDS:
            if last is None or age_days >= threshold:
                buckets[threshold].append(slug)
    return result


def build_diff(
    conn: psycopg.Connection,
    *,
    since: date,
    as_of: date | None = None,
    clock: str = "record",
) -> DiffFeed:
    """Assemble the diff feed for one window. `clock` (`"record"` or
    `"world"`) applies to `notes_created`/`notes_updated` only."""
    _check_clock(clock)
    as_of = as_of or clock_module.today()
    return DiffFeed(
        since=since,
        as_of=as_of,
        notes_created=notes_created(conn, since=since, as_of=as_of, clock=clock),
        notes_updated=notes_updated(conn, since=since, as_of=as_of, clock=clock),
        facts_superseded=superseded_facts(conn, since=since, as_of=as_of),
        quiet_accounts=quiet_accounts(conn, as_of),
    )
