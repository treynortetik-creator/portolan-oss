"""The action researcher's finding tools (`01-spec/actions-and-owners-
2026-09-19.md`, "Phase 2, revised contract", "Tools"), kept apart from
`researcher.py`'s agent-SDK wiring so tests can call them against Postgres.

Each tool (`absence_check`, `count_by`, `timeline`, `renewal`) records its
arguments and result in `researcher_findings` before returning; calling it
again with the stored arguments recomputes the same answer, up to graph
changes. `_claims_in_scope` resolves every live claim's account with
`home_account_slug` on each call (O(claims) queries).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal

import psycopg
from psycopg.types.json import Jsonb

from portolan.report.accounts import home_account_slug
from portolan.report.util import iso_week_str
from portolan.util import clock as clock_module

__all__ = [
    "AccountLookups",
    "GroupBy",
    "absence_arguments",
    "absence_check",
    "absence_result",
    "count_by",
    "load_latest_open_opportunities",
    "record_finding",
    "renewal",
    "timeline",
]

GroupBy = Literal["account", "facility", "region", "branch", "week", "speaker"]

_ISO_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


def record_finding(
    conn: psycopg.Connection,
    *,
    run_id: str,
    candidate_slug: str,
    tool: str,
    arguments: dict[str, Any],
    result: dict[str, Any],
    commit: bool = True,
) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO researcher_findings (run_id, candidate_slug, tool, arguments, result)
            VALUES (%s, %s, %s, %s::jsonb, %s::jsonb)
            RETURNING id
            """,
            (run_id, candidate_slug, tool, Jsonb(arguments), Jsonb(result)),
        )
        finding_id = cur.fetchone()[0]
    if commit:
        conn.commit()
    return finding_id


def _claim_rows(
    conn: psycopg.Connection, *, fact_types: list[str] | None = None
) -> list[dict[str, Any]]:
    with conn.cursor() as cur:
        if fact_types:
            cur.execute(
                "SELECT id, note_slug, fact_type, claim_text, quote, speaker, recorded, "
                "source_slug, locator FROM claims "
                "WHERE archived = false AND superseded_at IS NULL AND fact_type = ANY(%s)",
                (fact_types,),
            )
        else:
            cur.execute(
                "SELECT id, note_slug, fact_type, claim_text, quote, speaker, recorded, "
                "source_slug, locator FROM claims "
                "WHERE archived = false AND superseded_at IS NULL"
            )
        rows = cur.fetchall()
    return [
        {
            "id": r[0],
            "note_slug": r[1],
            "fact_type": r[2],
            "claim_text": r[3],
            "quote": r[4],
            "speaker": r[5],
            "recorded": r[6],
            "source_slug": r[7],
            "locator": r[8],
        }
        for r in rows
    ]


def _theme_member_claim_ids(conn: psycopg.Connection, theme_slug: str) -> set[int]:
    with conn.cursor() as cur:
        cur.execute("SELECT claim_id FROM theme_members WHERE theme_slug = %s", (theme_slug,))
        return {r[0] for r in cur.fetchall()}


def _resolved_account(conn: psycopg.Connection, claim: dict[str, Any]) -> str | None:
    return home_account_slug(
        conn,
        note_slug=claim["note_slug"],
        fact_type=claim["fact_type"],
        claim_text=claim["claim_text"],
    )


def _claims_in_scope(
    conn: psycopg.Connection,
    *,
    account_slug: str | None,
    theme_slug: str | None,
    fact_types: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Every live claim belonging to `account_slug` OR a member of
    `theme_slug` (the grounding gate's scope rule); pass at least one."""
    theme_ids = _theme_member_claim_ids(conn, theme_slug) if theme_slug else set()
    rows = _claim_rows(conn, fact_types=fact_types)
    result = []
    for claim in rows:
        if claim["id"] in theme_ids:
            result.append(claim)
            continue
        if account_slug is not None and _resolved_account(conn, claim) == account_slug:
            result.append(claim)
    return result


def _account_parent_slugs(conn: psycopg.Connection) -> dict[str, str | None]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT child.note_slug, parent.note_slug "
            "FROM registry_entities child "
            "LEFT JOIN registry_entities parent ON parent.id = child.parent_id "
            "WHERE child.kind = 'account' AND child.note_slug IS NOT NULL"
        )
        return dict(cur.fetchall())


def _top_level_account_slug(slug: str, parent_of: dict[str, str | None]) -> str:
    seen: set[str] = set()
    current = slug
    while True:
        parent = parent_of.get(current)
        if not parent or current in seen:
            return current
        seen.add(current)
        current = parent


# ---------------------------------------------------------------------
# absence_check
# ---------------------------------------------------------------------


def absence_check(
    conn: psycopg.Connection,
    *,
    run_id: str,
    candidate_slug: str,
    account_slug: str | None,
    theme_slug: str | None,
    fact_types: list[str],
    after_date: date,
    commit: bool = True,
) -> tuple[int, dict[str, Any]]:
    """ "are there claims of those types for that account and theme after
    the date; returns the count and the latest date, or zero".

    Also returns `count_before`/`had_matches_before`, so a sentence shaped
    "X existed before, nothing since" can cite both halves.
    """
    claims = _claims_in_scope(conn, account_slug=account_slug, theme_slug=theme_slug)
    result = absence_result(claims, fact_types=fact_types, after_date=after_date)
    finding_id = record_finding(
        conn,
        run_id=run_id,
        candidate_slug=candidate_slug,
        tool="absence_check",
        arguments=absence_arguments(
            account_slug=account_slug,
            theme_slug=theme_slug,
            fact_types=fact_types,
            after_date=after_date,
        ),
        result=result,
        commit=commit,
    )
    return finding_id, result


def absence_result(
    claims_in_scope: list[dict[str, Any]], *, fact_types: list[str], after_date: date
) -> dict[str, Any]:
    """`absence_check`'s result from claims already scoped to an account or
    theme. Shared with `portolan.report.action_done`, which scopes every
    open action's account in one pass and records the same finding."""
    in_type = [c for c in claims_in_scope if c["fact_type"] in fact_types]
    matches = [c for c in in_type if clock_module.world_date(c["recorded"]) > after_date]
    before = [c for c in in_type if clock_module.world_date(c["recorded"]) <= after_date]
    matches.sort(key=lambda c: c["recorded"])
    latest = clock_module.world_date(matches[-1]["recorded"]).isoformat() if matches else None
    return {
        "count": len(matches),
        "latest_date": latest,
        "claim_ids": [c["id"] for c in matches],
        "count_before": len(before),
        "had_matches_before": len(before) > 0,
        "claim_ids_before": [c["id"] for c in before],
    }


def absence_arguments(
    *, account_slug: str | None, theme_slug: str | None, fact_types: list[str], after_date: date
) -> dict[str, Any]:
    return {
        "account_slug": account_slug,
        "theme_slug": theme_slug,
        "fact_types": fact_types,
        "after_date": after_date.isoformat(),
    }


# ---------------------------------------------------------------------
# count_by
# ---------------------------------------------------------------------


@dataclass
class AccountLookups:
    """The CRM-export lookups (ARR, region, branch) and account parent map
    that `count_by`/`renewal` need; the graph stores none of the three."""

    arr_by_note_slug: dict[str, float]
    region_by_note_slug: dict[str, str]
    branch_by_note_slug: dict[str, str]
    parent_of: dict[str, str | None]


def count_by(
    conn: psycopg.Connection,
    *,
    run_id: str,
    candidate_slug: str,
    lookups: AccountLookups,
    fact_type: str | None = None,
    theme_slug: str | None = None,
    group_by: GroupBy = "account",
    since: date | None = None,
    until: date | None = None,
    commit: bool = True,
) -> tuple[int, dict[str, Any]]:
    """Claim counts by `group_by`, filtered by `fact_type` and/or
    `theme_slug`; grouping by account adds each group's ARR from
    `lookups` (the CRM export), not the registry."""
    if not fact_type and not theme_slug:
        raise ValueError("count_by needs at least one of fact_type or theme_slug")

    theme_ids = _theme_member_claim_ids(conn, theme_slug) if theme_slug else None
    rows = _claim_rows(conn, fact_types=[fact_type] if fact_type else None)
    if theme_ids is not None:
        rows = [r for r in rows if r["id"] in theme_ids]
    if since is not None:
        rows = [r for r in rows if clock_module.world_date(r["recorded"]) >= since]
    if until is not None:
        rows = [r for r in rows if clock_module.world_date(r["recorded"]) <= until]

    counts: dict[str, int] = {}
    for claim in rows:
        if group_by == "week":
            group = iso_week_str(clock_module.world_date(claim["recorded"]))
        elif group_by == "speaker":
            group = claim["speaker"] or "(unknown)"
        else:
            account_slug = _resolved_account(conn, claim)
            if account_slug is None:
                group = "(unattributed)"
            elif group_by == "facility":
                group = account_slug
            elif group_by == "account":
                group = _top_level_account_slug(account_slug, lookups.parent_of)
            elif group_by == "region":
                group = lookups.region_by_note_slug.get(account_slug, "(unknown)")
            elif group_by == "branch":
                group = lookups.branch_by_note_slug.get(account_slug, "(unknown)")
            else:  # pragma: no cover -- Literal exhausts this at type-check time
                raise ValueError(f"unknown group_by {group_by!r}")
        counts[group] = counts.get(group, 0) + 1

    groups = [{"group": g, "count": n} for g, n in sorted(counts.items())]
    if group_by == "account":
        for entry in groups:
            entry["arr"] = lookups.arr_by_note_slug.get(entry["group"], 0.0)

    result = {"groups": groups, "total": len(rows)}
    arguments = {
        "fact_type": fact_type,
        "theme_slug": theme_slug,
        "group_by": group_by,
        "since": since.isoformat() if since else None,
        "until": until.isoformat() if until else None,
    }
    finding_id = record_finding(
        conn,
        run_id=run_id,
        candidate_slug=candidate_slug,
        tool="count_by",
        arguments=arguments,
        result=result,
        commit=commit,
    )
    return finding_id, result


# ---------------------------------------------------------------------
# timeline
# ---------------------------------------------------------------------


def timeline(
    conn: psycopg.Connection,
    *,
    run_id: str,
    candidate_slug: str,
    account_slug: str | None,
    theme_slug: str | None,
    since: date | None = None,
    until: date | None = None,
    commit: bool = True,
) -> tuple[int, dict[str, Any]]:
    """ "dated claims in order with type and speaker"."""
    if account_slug is None and theme_slug is None:
        raise ValueError("timeline needs account_slug or theme_slug")
    claims = _claims_in_scope(conn, account_slug=account_slug, theme_slug=theme_slug)
    if since is not None:
        claims = [c for c in claims if clock_module.world_date(c["recorded"]) >= since]
    if until is not None:
        claims = [c for c in claims if clock_module.world_date(c["recorded"]) <= until]
    claims.sort(key=lambda c: c["recorded"])

    entries = [
        {
            "claim_id": c["id"],
            "date": clock_module.world_date(c["recorded"]).isoformat(),
            "fact_type": c["fact_type"],
            "speaker": c["speaker"],
            "claim_text": c["claim_text"],
            "quote": c["quote"],
            "source_slug": c["source_slug"],
            "locator": c["locator"],
        }
        for c in claims
    ]
    result = {"entries": entries, "count": len(entries)}
    arguments = {
        "account_slug": account_slug,
        "theme_slug": theme_slug,
        "since": since.isoformat() if since else None,
        "until": until.isoformat() if until else None,
    }
    finding_id = record_finding(
        conn,
        run_id=run_id,
        candidate_slug=candidate_slug,
        tool="timeline",
        arguments=arguments,
        result=result,
        commit=commit,
    )
    return finding_id, result


# ---------------------------------------------------------------------
# renewal
# ---------------------------------------------------------------------

_RENEWAL_KEYWORD_RE = re.compile(r"renew|close date|closes|contract", re.IGNORECASE)
_OPEN_STAGE_PREFIX = "closed"


def load_latest_open_opportunities(root) -> dict[str, dict[str, str]]:
    """`{account_crm_id: row}`: the current open opportunity per account.

    The export directory `root` holds dated snapshots (`opportunities.csv`,
    `opportunities-2026-08-21.csv`, ...), so every `opportunities*.csv` is
    read and, per account, the open-stage row with a `close_date` and the
    latest `export_date` wins. The one loader for candidates and the
    `renewal` tool, so both see the same snapshot."""
    import csv as csv_module

    best_by_account: dict[str, dict[str, str]] = {}
    best_export_date_by_account: dict[str, str] = {}
    for opportunities_csv in sorted(root.glob("opportunities*.csv")):
        if not opportunities_csv.is_file():
            continue
        with opportunities_csv.open(newline="", encoding="utf-8") as fh:
            for row in csv_module.DictReader(fh):
                account_crm_id = (row.get("account_crm_id") or "").strip()
                if not account_crm_id:
                    continue
                stage = (row.get("stage") or "").strip().lower()
                if stage.startswith(_OPEN_STAGE_PREFIX):
                    continue
                raw_close = (row.get("close_date") or "").strip()
                if not raw_close:
                    continue
                export_date = (row.get("export_date") or "").strip()
                if (
                    account_crm_id not in best_by_account
                    or export_date > best_export_date_by_account[account_crm_id]
                ):
                    best_by_account[account_crm_id] = row
                    best_export_date_by_account[account_crm_id] = export_date
    return best_by_account


def _load_opportunity(root, account_crm_id: str) -> dict[str, str] | None:
    """`load_latest_open_opportunities` for one account."""
    return load_latest_open_opportunities(root).get(account_crm_id)


def renewal(
    conn: psycopg.Connection,
    *,
    run_id: str,
    candidate_slug: str,
    account_slug: str,
    connector: Any,
    lookups: AccountLookups,
    commit: bool = True,
) -> tuple[int, dict[str, Any]]:
    """The account's current open opportunity (close date, name, stage)
    from the CRM export, its ARR from `lookups`, and every in-scope claim
    that states an ISO date alongside renewal wording, with claim ids."""
    from pathlib import Path

    with conn.cursor() as cur:
        cur.execute(
            "SELECT crm_id FROM registry_entities WHERE note_slug = %s AND kind = 'account'",
            (account_slug,),
        )
        row = cur.fetchone()
    account_crm_id = row[0] if row else None

    opportunity: dict[str, str] | None = None
    root = getattr(connector, "root", None) if connector is not None else None
    if root is not None and account_crm_id:
        opportunity = _load_opportunity(Path(root) / "crm", account_crm_id)

    claims = _claims_in_scope(conn, account_slug=account_slug, theme_slug=None)
    renewal_claims = []
    for c in claims:
        haystack = f"{c['claim_text']} {c['quote'] or ''}"
        if _ISO_DATE_RE.search(haystack) and _RENEWAL_KEYWORD_RE.search(haystack):
            renewal_claims.append(
                {
                    "claim_id": c["id"],
                    "claim_text": c["claim_text"],
                    "quote": c["quote"],
                    "dates_mentioned": _ISO_DATE_RE.findall(haystack),
                }
            )

    result = {
        "close_date": opportunity.get("close_date") if opportunity else None,
        "opportunity_name": opportunity.get("name") if opportunity else None,
        "stage": opportunity.get("stage") if opportunity else None,
        "arr": lookups.arr_by_note_slug.get(account_slug, 0.0),
        "renewal_claims": renewal_claims,
    }
    arguments = {"account_slug": account_slug}
    finding_id = record_finding(
        conn,
        run_id=run_id,
        candidate_slug=candidate_slug,
        tool="renewal",
        arguments=arguments,
        result=result,
        commit=commit,
    )
    return finding_id, result
