"""Feed: public signals this week (`01-spec/external-signals-2026-09-23.md`,
"In the weekly report"). Pure SQL, no model call.

Every `public_signal` claim that `portolan.signals` wrote in the window, grouped
by account. The window follows the report's clock like the diff feed:
`record` keys on when the scan judged the item (when Portolan learned it),
`world` on the claim's `recorded` date (the item's own publish date, or the
snapshot date for a website change).
"""

from __future__ import annotations

from datetime import date

import psycopg

from portolan.report.accounts import home_account_slug
from portolan.report.models import PublicSignal
from portolan.signals import PUBLIC_SIGNAL_FACT_TYPE
from portolan.util import clock as clock_module

__all__ = ["public_signals_feed", "render_public_signals"]


def public_signals_feed(
    conn: psycopg.Connection, *, since: date, as_of: date, clock: str = "record"
) -> list[PublicSignal]:
    """The window's public signals, oldest first (module docstring)."""
    window = (
        clock_module.world_date_sql("si.judged_at")
        if clock == "record"
        else clock_module.world_date_sql("c.recorded")
    )
    # One row per claim: a claim several items corroborate is listed once, with the
    # first item that produced it.
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT * FROM (
                SELECT DISTINCT ON (c.id)
                       c.id, c.note_slug, c.fact_type, c.claim_text, c.source_slug, c.locator,
                       c.recorded, sic.category, si.item_kind, si.url
                FROM signal_items si
                JOIN signal_item_claims sic ON sic.item_id = si.id
                JOIN claims c ON c.id = sic.claim_id
                WHERE si.status = 'relevant'
                  AND c.fact_type = %s
                  AND c.archived = false
                  AND {window} BETWEEN %s AND %s
                ORDER BY c.id, si.id
            ) AS per_claim
            ORDER BY recorded, id
            """,
            (PUBLIC_SIGNAL_FACT_TYPE, since, as_of),
        )
        rows = cur.fetchall()
    out = []
    for (
        claim_id,
        note_slug,
        fact_type,
        claim_text,
        source_slug,
        locator,
        recorded,
        category,
        item_kind,
        url,
    ) in rows:
        out.append(
            PublicSignal(
                claim_id=claim_id,
                account_slug=home_account_slug(
                    conn, note_slug=note_slug, fact_type=fact_type, claim_text=claim_text
                ),
                category=category,
                item_kind=item_kind,
                claim_text=claim_text,
                source_slug=source_slug,
                locator=locator,
                url=url,
                recorded=clock_module.world_date(recorded),
            )
        )
    return out


def render_public_signals(signals: list[PublicSignal]) -> list[str]:
    """The report section's lines: one group per account (alphabetical), one
    cited bullet per signal, oldest first."""
    lines = [
        "### Public signals this week",
        "_Public, second-hand information (news, LinkedIn company pages, the account's own "
        "website), each checked by a relevance judge. Not something the customer told us._",
        "",
    ]
    if not signals:
        lines.append("- none this window")
        return lines
    by_account: dict[str, list[PublicSignal]] = {}
    for signal in signals:
        by_account.setdefault(signal.account_slug or "", []).append(signal)
    for account in sorted(by_account, key=lambda a: (a == "", a)):
        lines.append(f"**[[{account}]]**" if account else "**(no account)**")
        lines.append("")
        for signal in by_account[account]:
            lines.append(
                f"- {signal.claim_text} ^[{signal.source_slug} @ {signal.locator}] "
                f"({signal.recorded:%Y-%m-%d})"
            )
        lines.append("")
    return lines
