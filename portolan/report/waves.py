"""Feed 5: waves. A theme groups claims by content (embeddings); a wave is a
plain `GROUP BY fact_type, product family, week` over signal claims in the
report window, with an account-count floor. No embedding, no model call.

Waves catch accounts that raise the same kind of claim in the same week
even when their wording is unrelated, without loosening the theme
clustering gate.
"""

from __future__ import annotations

from datetime import date

import psycopg

from portolan.notes.render import WIKILINK_RE
from portolan.report.accounts import home_account_slug
from portolan.report.models import WaveRow, WavesFeed
from portolan.report.themes import (
    _account_parent_slugs,
    _all_signal_details,
    _linked_arr,
    _representative_quotes,
    _top_level_accounts,
)
from portolan.report.util import week_start
from portolan.util import clock as clock_module

__all__ = ["WAVE_MIN_ACCOUNTS", "compute_waves"]

# Account count is a wave's only signal (no content check), so its floor
# is one higher than the theme floor `CLUSTER_MIN_ACCOUNTS`.
WAVE_MIN_ACCOUNTS = 3


def _product_family(slug: str) -> str:
    """The slug segment before its first hyphen (`tidewater-t2` -> `tidewater`),
    same rule as `themes._product_family`."""
    return slug.split("-", 1)[0]


def _resolved_product_families(conn: psycopg.Connection, claim_text: str) -> set[str]:
    """Every product family the claim's `[[slug]]` product/competitor mentions
    resolve to; a claim naming two families joins both waves."""
    mentioned = [m.group(1) for m in WIKILINK_RE.finditer(claim_text)]
    if not mentioned:
        return set()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT slug FROM notes WHERE type IN ('product', 'competitor') AND slug = ANY(%s)",
            (mentioned,),
        )
        return {_product_family(r[0]) for r in cur.fetchall()}


def _customer_roots(conn: psycopg.Connection) -> dict[str, str]:
    """`{account_note_slug: customer_key}`: the id of the account's topmost
    registry ancestor, whether or not that ancestor has a note. Accounts of
    one health system share a key. An account missing here is its own
    customer."""
    with conn.cursor() as cur:
        cur.execute(
            """
            WITH RECURSIVE chain AS (
                SELECT note_slug, id AS ancestor_id, parent_id, 0 AS depth
                FROM registry_entities
                WHERE kind = 'account' AND note_slug IS NOT NULL
                UNION ALL
                SELECT chain.note_slug, parent.id, parent.parent_id, chain.depth + 1
                FROM chain JOIN registry_entities parent ON parent.id = chain.parent_id
                WHERE chain.depth < 32
            )
            SELECT DISTINCT ON (note_slug) note_slug, ancestor_id
            FROM chain ORDER BY note_slug, depth DESC
            """
        )
        return {slug: f"entity-{ancestor_id}" for slug, ancestor_id in cur.fetchall()}


def compute_waves(
    conn: psycopg.Connection,
    *,
    since: date,
    as_of: date,
    arr_by_note_slug: dict[str, float],
) -> WavesFeed:
    """Every `(week_of, fact_type, product_family)` group of signal claims
    recorded in `[since, as_of]` (week taken from `valid_from`, else
    `recorded`) that touches `>= WAVE_MIN_ACCOUNTS` distinct customers:
    accounts under one topmost registry ancestor count once, whether or
    not that ancestor is in the group (`_customer_roots`). The ARR roll-up
    is `_top_level_accounts`'. A claim with no home account or no
    product/competitor mention joins no wave."""
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT c.id, c.note_slug, c.fact_type, c.claim_text, c.recorded, c.valid_from
            FROM claims c
            JOIN notes n ON n.slug = c.note_slug
            WHERE n.type = 'signal'
              AND c.archived = false
              -- A wave is customers reporting the same thing; public signals are not that.
              AND c.fact_type <> 'public_signal'
              AND {clock_module.world_date_sql("c.recorded")} BETWEEN %s AND %s
            ORDER BY c.recorded, c.id
            """,
            (since, as_of),
        )
        rows = cur.fetchall()

    groups: dict[tuple[date, str, str], tuple[set[int], set[str]]] = {}
    for claim_id, note_slug, fact_type, claim_text, recorded, valid_from in rows:
        world_date = valid_from or clock_module.world_date(recorded)
        week_of = week_start(world_date)
        account = home_account_slug(
            conn, note_slug=note_slug, fact_type=fact_type, claim_text=claim_text
        )
        if account is None:
            continue
        for family in _resolved_product_families(conn, claim_text):
            key = (week_of, fact_type, family)
            claim_ids, accounts = groups.setdefault(key, (set(), set()))
            claim_ids.add(claim_id)
            accounts.add(account)

    # Read the registry maps once for the run, not once per group.
    parent_of = _account_parent_slugs(conn)
    customer_of = _customer_roots(conn)

    waves: list[WaveRow] = []
    for (week_of, fact_type, family), (claim_ids, accounts) in sorted(groups.items()):
        customers = {customer_of.get(account, account) for account in accounts}
        if len(customers) < WAVE_MIN_ACCOUNTS:
            continue
        # Passed to `_linked_arr` so the roll-up is not computed twice.
        top_level, rolled_up = _top_level_accounts(conn, accounts, parent_of=parent_of)
        linked_arr, _rolled_up_again, unknown_arr_count = _linked_arr(
            conn,
            accounts,
            arr_by_note_slug,
            top_level_and_rolled_up=(top_level, rolled_up),
        )
        waves.append(
            WaveRow(
                week_of=week_of,
                fact_type=fact_type,
                product_family=family,
                account_slugs=sorted(accounts),
                linked_arr=linked_arr,
                claim_ids=sorted(claim_ids),
                rolled_up=rolled_up,
                unknown_arr_count=unknown_arr_count,
                quotes=_representative_quotes(conn, claim_ids),
                all_signals=_all_signal_details(conn, claim_ids),
            )
        )
    return WavesFeed(since=since, as_of=as_of, waves=waves)
