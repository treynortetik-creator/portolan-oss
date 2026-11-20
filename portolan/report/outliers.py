"""Feed 4: outliers (`01-spec/surfacing-and-weekly-report.md` "4. Outliers").
No model call: a claim is an outlier by *not* being in `theme_members`, so
the set is recomputed from that absence on every call; nothing stores it.
"""

from __future__ import annotations

from datetime import date

import psycopg

from portolan.report.accounts import home_account_slug
from portolan.report.models import OutlierClaim
from portolan.util import clock

__all__ = ["list_outliers"]


def list_outliers(
    conn: psycopg.Connection, *, since: date, as_of: date | None = None
) -> list[OutlierClaim]:
    """Signal claims recorded in `[since, as_of]` that are attached to no
    theme ("listed plainly with account and source")."""
    as_of = as_of or clock.today()
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT c.id, c.note_slug, c.fact_type, c.claim_text, c.source_slug,
                   c.locator, c.recorded
            FROM claims c
            JOIN notes n ON n.slug = c.note_slug
            WHERE n.type = 'signal'
              AND c.archived = false
              -- Public signals get their own report section (portolan.report.public_signals).
              AND c.fact_type <> 'public_signal'
              AND {clock.world_date_sql("c.recorded")} BETWEEN %s AND %s
              AND NOT EXISTS (SELECT 1 FROM theme_members tm WHERE tm.claim_id = c.id)
            ORDER BY c.recorded, c.id
            """,
            (since, as_of),
        )
        rows = cur.fetchall()

    outliers: list[OutlierClaim] = []
    for claim_id, note_slug, fact_type, claim_text, source_slug, locator, recorded in rows:
        account = home_account_slug(
            conn, note_slug=note_slug, fact_type=fact_type, claim_text=claim_text
        )
        outliers.append(
            OutlierClaim(
                claim_id=claim_id,
                note_slug=note_slug,
                account_slug=account,
                fact_type=fact_type,
                claim_text=claim_text,
                source_slug=source_slug,
                locator=locator,
                # Deployment-zone date; `.date()` would use the session's zone.
                recorded=clock.world_date(recorded),
            )
        )
    return outliers
