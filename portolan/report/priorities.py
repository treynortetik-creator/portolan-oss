"""Ranked theme priorities (`01-spec/actions-and-owners-2026-09-19.md`
"Phase 3: ranked priorities").

Order: accounts touched (distinct leaf accounts with a member claim in the
`[since, as_of]` window), then ARR touched (CRM-export ARR rolled up to
top-level accounts, so a facility and its parent system count once), then
movement (`MOVEMENT_PRIORITY_RANK`), then `theme_slug`. Regions come from
the CRM export.

The weekly report and the Atlas theme folder both rank through
`compute_theme_priorities`; Atlas uses the latest report's window
(`theme_priorities_for_atlas`) so the two show the same numbers.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date

import psycopg

from portolan.report.accounts import home_account_slug
from portolan.report.models import ThemePriority, ThemeSummary
from portolan.report.numbers import NumberFacts
from portolan.report.themes import (
    _account_parent_slugs,
    _linked_arr,
    _top_level_accounts,
    theme_trend,
)
from portolan.report.util import week_bounds

__all__ = [
    "MOVEMENT_PRIORITY_RANK",
    "add_priority_facts",
    "compute_theme_priorities",
    "format_priority_line",
    "format_priority_stats",
    "latest_report_window",
    "living_theme_summaries_for_window",
    "theme_priorities_for_atlas",
]

# "resolution proposed" ranks last: like `resolved`, it needs the least
# fresh attention.
MOVEMENT_PRIORITY_RANK: dict[str, int] = {
    "new": 0,
    "rising": 1,
    "falling": 2,
    "resolved": 3,
    "resolution proposed": 4,
}


def _touched_accounts_in_window(
    conn: psycopg.Connection, theme_slug: str, *, since: date, as_of: date
) -> set[str]:
    """Leaf accounts of this theme's member claims whose `theme_members.added`
    falls in `[since, as_of]` (not the theme's all-time accounts)."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT c.note_slug, c.fact_type, c.claim_text
            FROM theme_members tm
            JOIN claims c ON c.id = tm.claim_id
            WHERE tm.theme_slug = %s
              AND tm.added BETWEEN %s AND %s
              AND c.archived = false
            """,
            (theme_slug, since, as_of),
        )
        rows = cur.fetchall()

    accounts: set[str] = set()
    for note_slug, fact_type, claim_text in rows:
        account = home_account_slug(
            conn, note_slug=note_slug, fact_type=fact_type, claim_text=claim_text
        )
        if account is not None:
            accounts.add(account)
    return accounts


def compute_theme_priorities(
    conn: psycopg.Connection,
    *,
    themes: Iterable[ThemeSummary],
    since: date,
    as_of: date,
    arr_by_note_slug: dict[str, float],
    region_by_note_slug: dict[str, str],
) -> list[ThemePriority]:
    """Every theme in `themes` that touched at least one account in
    `[since, as_of]`, ranked `account_count` desc, `arr` desc,
    `MOVEMENT_PRIORITY_RANK`, `theme_slug` asc. Themes with no account
    touched in the window are dropped.

    Only `slug`, `name` and `movement` are read from each `ThemeSummary`,
    so a caller may pass placeholders for the rest."""
    parent_of = _account_parent_slugs(conn)

    rows: list[ThemePriority] = []
    for t in themes:
        touched = _touched_accounts_in_window(conn, t.slug, since=since, as_of=as_of)
        if not touched:
            continue
        top_level, rolled_up = _top_level_accounts(conn, touched, parent_of=parent_of)
        arr, rolled_up, unknown_count = _linked_arr(
            conn,
            touched,
            arr_by_note_slug,
            top_level_and_rolled_up=(top_level, rolled_up),
        )
        regions = sorted({region_by_note_slug[a] for a in touched if a in region_by_note_slug})
        rows.append(
            ThemePriority(
                theme_slug=t.slug,
                theme_name=t.name,
                movement=t.movement,
                account_slugs=sorted(touched),
                account_count=len(touched),
                arr=arr,
                arr_unknown_accounts=unknown_count,
                regions=regions,
                rolled_up=rolled_up,
            )
        )

    rows.sort(
        key=lambda p: (
            -p.account_count,
            -p.arr,
            MOVEMENT_PRIORITY_RANK.get(p.movement, len(MOVEMENT_PRIORITY_RANK)),
            p.theme_slug,
        )
    )
    return rows


def format_priority_stats(p: ThemePriority) -> str:
    """ "N accounts, $X in annual revenue in <regions> -- movement": a ranked
    row without the theme name, which the report and Atlas each link their
    own way."""
    accounts_word = "account" if p.account_count == 1 else "accounts"
    regions_txt = ", ".join(p.regions) if p.regions else "region unknown"
    arr_note = ""
    if p.arr_unknown_accounts:
        plural = "account" if p.arr_unknown_accounts == 1 else "accounts"
        arr_note = f" ({p.arr_unknown_accounts} {plural} excluded, ARR unknown)"
    return (
        f"{p.account_count} {accounts_word}, ${p.arr:,.0f} in annual revenue "
        f"in {regions_txt}{arr_note} — {p.movement}"
    )


def format_priority_line(p: ThemePriority) -> str:
    """One ranked row for the report body: `[[theme-slug]]` then
    `format_priority_stats`."""
    return f"[[{p.theme_slug}]] — {format_priority_stats(p)}"


def add_priority_facts(facts: NumberFacts, priorities: Iterable[ThemePriority]) -> None:
    """Register each ranked row's account count and ARR in `facts` (via
    `add`: these are computed aggregates, not numbers quoted from a claim),
    so prose stating them passes `check_numerals`."""
    for p in priorities:
        facts.add(f"priority:{p.theme_slug}:accounts", p.account_count, str(p.account_count))
        facts.add(f"priority:{p.theme_slug}:arr", p.arr, f"{p.arr:,.0f}")


def latest_report_window(conn: psycopg.Connection) -> tuple[date, date] | None:
    """`[since, as_of]` for the most recent weekly report (the fixed-width
    `YYYY-Www` string sorts chronologically), or `None` when there is none."""
    with conn.cursor() as cur:
        cur.execute("SELECT week FROM reports ORDER BY week DESC LIMIT 1")
        row = cur.fetchone()
    if row is None:
        return None
    return week_bounds(row[0])


def living_theme_summaries_for_window(
    conn: psycopg.Connection, as_of: date, *, clock: str = "record"
) -> list[ThemeSummary]:
    """Every living theme's slug, name and movement as of `as_of`, with
    placeholders for the other `ThemeSummary` fields: enough for
    `compute_theme_priorities` without the cost of `theme_summary()`."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT slug, title FROM notes WHERE type = 'theme' AND status = 'living' ORDER BY slug"
        )
        rows = cur.fetchall()
    return [
        ThemeSummary(
            slug=slug,
            name=title,
            signal_count=0,
            linked_arr=0.0,
            first_seen=None,
            last_seen=None,
            movement=theme_trend(conn, slug, as_of, clock=clock),
        )
        for slug, title in rows
    ]


def theme_priorities_for_atlas(conn: psycopg.Connection) -> list[ThemePriority]:
    """The ranked-priority list for the Atlas theme folder, over the latest
    report's window; `[]` when no report exists.

    ARR and region are read from the CRM export at page-render time via a
    `FileConnector` over `portolan.web.demo.DEFAULT_CORPUS_PATH`. The imports
    are local so `portolan.report` does not import the web package at load."""
    window = latest_report_window(conn)
    if window is None:
        return []
    since, as_of = window

    from portolan.connectors.file_connector import FileConnector
    from portolan.web.demo import DEFAULT_CORPUS_PATH

    themes = living_theme_summaries_for_window(conn, as_of)
    connector = FileConnector(DEFAULT_CORPUS_PATH)

    from portolan.report.accounts import (
        account_arr_by_note_slug,
        account_region_by_note_slug,
        load_account_arr_via_connector,
        load_account_regions_via_connector,
    )

    arr_by_crm_id = load_account_arr_via_connector(connector)
    arr_by_note_slug = account_arr_by_note_slug(conn, arr_by_crm_id)
    region_by_crm_id = load_account_regions_via_connector(connector)
    region_by_note_slug = account_region_by_note_slug(conn, region_by_crm_id)

    return compute_theme_priorities(
        conn,
        themes=themes,
        since=since,
        as_of=as_of,
        arr_by_note_slug=arr_by_note_slug,
        region_by_note_slug=region_by_note_slug,
    )
