"""Coverage ledger: per connector, per day (spec sec 6.2, build item 6).

`record` is additive, not idempotent: every call adds its counts to the
`(connector, day)` row, so calling it twice counts twice. The addition
happens inside the one upsert's `DO UPDATE SET`, with no read step, so two
concurrent first writes for the same row both land.
"""

from __future__ import annotations

from datetime import date

import psycopg
from psycopg.types.json import Jsonb


def record(
    conn: psycopg.Connection,
    connector: str,
    day: date,
    *,
    seen: int = 0,
    filtered: dict[str, int] | None = None,
    extracted: int = 0,
    partial: int = 0,
    failed: int = 0,
    sampled: int = 0,
) -> None:
    """Upsert `coverage_ledger`, adding these counts to the day's totals.

    `filtered` is a reason -> count map (e.g. `{"junk:auto_reply": 1}`),
    summed key by key into the existing `filtered` jsonb; a bare jsonb `||`
    would replace a shared key's count instead of adding to it.
    """
    filtered = filtered or {}
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO coverage_ledger
                (connector, day, seen, filtered, extracted, partial, failed, sampled)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (connector, day) DO UPDATE SET
                seen = coverage_ledger.seen + EXCLUDED.seen,
                extracted = coverage_ledger.extracted + EXCLUDED.extracted,
                partial = coverage_ledger.partial + EXCLUDED.partial,
                failed = coverage_ledger.failed + EXCLUDED.failed,
                sampled = coverage_ledger.sampled + EXCLUDED.sampled,
                filtered = COALESCE((
                    SELECT jsonb_object_agg(
                        key,
                        COALESCE((coverage_ledger.filtered ->> key)::int, 0)
                        + COALESCE((EXCLUDED.filtered ->> key)::int, 0)
                    )
                    FROM (
                        SELECT jsonb_object_keys(coverage_ledger.filtered) AS key
                        UNION
                        SELECT jsonb_object_keys(EXCLUDED.filtered) AS key
                    ) AS keys
                ), '{}'::jsonb)
            """,
            (
                connector,
                day,
                seen,
                Jsonb(filtered),
                extracted,
                partial,
                failed,
                sampled,
            ),
        )
    conn.commit()


def latest_day(conn: psycopg.Connection) -> date | None:
    """The newest `day` on the ledger, or `None` when it is empty. The
    pipeline records a source's own date as `day`, not its ingest date."""
    with conn.cursor() as cur:
        cur.execute("SELECT max(day) FROM coverage_ledger")
        row = cur.fetchone()
    return row[0] if row else None


def summary(conn: psycopg.Connection, since: date) -> list[dict]:
    """Coverage rows on or after `since`, oldest first (spec sec 6.2)."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT connector, day, seen, filtered, extracted, partial, failed, sampled
            FROM coverage_ledger
            WHERE day >= %s
            ORDER BY day, connector
            """,
            (since,),
        )
        rows = cur.fetchall()

    return [
        {
            "connector": row[0],
            "day": row[1],
            "seen": row[2],
            "filtered": row[3],
            "extracted": row[4],
            "partial": row[5],
            "failed": row[6],
            "sampled": row[7],
        }
        for row in rows
    ]
