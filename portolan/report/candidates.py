"""Phase 2 candidates: what deserves an action this week (`01-spec/
actions-and-owners-2026-09-19.md`, "Candidates by code, not by the
model").

Pure SQL and Python, no model call. Inputs are graph queries, the CRM
opportunities export, and feeds the weekly report already computed (quiet
accounts, theme priorities, the actions feed, standing-question answers),
which are read, never recomputed.

Five rules, verbatim from the spec:

1. **theme_rising** -- a theme that is new or rising.
2. **renewal_risk** -- an account with a renewal inside the horizon and a
   quiet window or an open complaint.
3. **commitment_overdue** -- a commitment past due.
4. **standing_question** -- a standing question with a new answer.
5. **pricing_or_competitor** -- a pricing or competitor fact at an
   account above an ARR floor.

Each `Candidate` carries its `slug`, the `claim_ids` that triggered it,
and the account owner's `owner_crm_id` (`None` when unassigned).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Literal

import psycopg

from portolan.report.accounts import home_account_slug
from portolan.report.models import (
    ActionsFeed,
    DiffFeed,
    StandingQuestionAnswer,
    ThemePriority,
)
from portolan.util import clock as clock_module

__all__ = [
    "PRICING_COMPETITOR_ARR_FLOOR",
    "RENEWAL_HORIZON_DAYS",
    "Candidate",
    "find_candidates",
]

CandidateRule = Literal[
    "theme_rising",
    "renewal_risk",
    "commitment_overdue",
    "standing_question",
    "pricing_or_competitor",
]

# Rule 2's "renewal inside the horizon": an open opportunity closing within
# this many days of `as_of`. A chosen default (~9 months), not derived.
RENEWAL_HORIZON_DAYS = 270

# Rule 5's ARR floor. Calibrated on the synthetic corpus; a deployment
# should set it from its own ARR distribution.
PRICING_COMPETITOR_ARR_FLOOR = 500_000.0

_OPEN_OPPORTUNITY_STAGE_PREFIX = "closed"


@dataclass
class Candidate:
    """One thing the researcher should investigate. `theme_slug` is `None`
    when the rule has no theme to point at; the grounding gate then scopes
    a card's claims to `account_slug` alone."""

    slug: str
    rule: CandidateRule
    account_slug: str
    reason: str
    claim_ids: list[int] = field(default_factory=list)
    theme_slug: str | None = None
    owner_crm_id: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)


def _account_owner_crm_id(conn: psycopg.Connection, account_slug: str) -> str | None:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT owner.crm_id FROM registry_entities account "
            "LEFT JOIN registry_entities owner ON owner.id = account.owner_entity_id "
            "WHERE account.note_slug = %s AND account.kind = 'account'",
            (account_slug,),
        )
        row = cur.fetchone()
    return row[0] if row else None


def _account_name(conn: psycopg.Connection, account_slug: str) -> str:
    with conn.cursor() as cur:
        cur.execute("SELECT title FROM notes WHERE slug = %s", (account_slug,))
        row = cur.fetchone()
    return row[0] if row else account_slug


def _theme_for_claim(conn: psycopg.Connection, claim_id: int) -> str | None:
    """The theme `claim_id` belongs to (oldest `added` if more than one),
    or `None`: best-effort context for rules that are not theme-scoped."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT theme_slug FROM theme_members WHERE claim_id = %s ORDER BY added LIMIT 1",
            (claim_id,),
        )
        row = cur.fetchone()
    return row[0] if row else None


def _theme_window_claims_by_top_level_account(
    conn: psycopg.Connection,
    theme_slug: str,
    *,
    since: date,
    as_of: date,
    rolled_up: dict[str, str],
) -> dict[str, list[int]]:
    """`{top_level_account_slug: [claim_id, ...]}` for every claim that
    joined `theme_slug` in `[since, as_of]`, each account mapped through
    `rolled_up` (`ThemePriority.rolled_up`, `{child: parent}`)."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT c.id, c.note_slug, c.fact_type, c.claim_text
            FROM theme_members tm
            JOIN claims c ON c.id = tm.claim_id
            WHERE tm.theme_slug = %s AND tm.added BETWEEN %s AND %s AND c.archived = false
            """,
            (theme_slug, since, as_of),
        )
        rows = cur.fetchall()

    by_account: dict[str, list[int]] = {}
    for claim_id, note_slug, fact_type, claim_text in rows:
        account = home_account_slug(
            conn, note_slug=note_slug, fact_type=fact_type, claim_text=claim_text
        )
        if account is None:
            continue
        top_level = rolled_up.get(account, account)
        by_account.setdefault(top_level, []).append(claim_id)
    return by_account


def _theme_rising_candidates(
    conn: psycopg.Connection, priorities: list[ThemePriority], *, since: date, as_of: date
) -> list[Candidate]:
    """Rule 1: a theme that is new or rising. One candidate per theme,
    scoped to the top-level account with the most claims that joined it
    this window (the spec names one owner per candidate)."""
    candidates: list[Candidate] = []
    for p in priorities:
        if p.movement not in ("new", "rising"):
            continue
        by_account = _theme_window_claims_by_top_level_account(
            conn, p.theme_slug, since=since, as_of=as_of, rolled_up=p.rolled_up
        )
        if not by_account:
            continue
        account_slug, claim_ids = max(by_account.items(), key=lambda item: (len(item[1]), item[0]))
        candidates.append(
            Candidate(
                slug=f"theme-{p.movement}:{p.theme_slug}:{account_slug}",
                rule="theme_rising",
                account_slug=account_slug,
                theme_slug=p.theme_slug,
                claim_ids=sorted(claim_ids),
                owner_crm_id=_account_owner_crm_id(conn, account_slug),
                reason=(
                    f"theme {p.theme_name!r} is {p.movement} this window, touching "
                    f"{p.account_count} account(s) and ${p.arr:,.0f} in ARR; "
                    f"{_account_name(conn, account_slug)} drove the most activity"
                ),
                detail={"movement": p.movement, "account_count": p.account_count, "arr": p.arr},
            )
        )
    return candidates


def _open_complaint_claims_by_account(conn: psycopg.Connection) -> dict[str, list[int]]:
    """Every open complaint claim (not archived, never superseded), by
    account. No time bound: "open" means nothing superseded it."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, note_slug, fact_type, claim_text FROM claims "
            "WHERE fact_type = 'complaint' AND archived = false AND superseded_at IS NULL"
        )
        rows = cur.fetchall()
    by_account: dict[str, list[int]] = {}
    for claim_id, note_slug, fact_type, claim_text in rows:
        account = home_account_slug(
            conn, note_slug=note_slug, fact_type=fact_type, claim_text=claim_text
        )
        if account is not None:
            by_account.setdefault(account, []).append(claim_id)
    return by_account


def _load_open_renewals(
    crm_root: Path, *, as_of: date, horizon_days: int
) -> dict[str, tuple[date, str]]:
    """`{account_crm_id: (close_date, opportunity_name)}` for every open
    opportunity closing in `[as_of, as_of + horizon_days]`. Uses the same
    loader as the researcher's `renewal` tool, so both read the latest
    snapshot per account."""
    from portolan.report.researcher_tools import load_latest_open_opportunities

    result: dict[str, tuple[date, str]] = {}
    horizon_end = as_of + timedelta(days=horizon_days)
    for account_crm_id, row in load_latest_open_opportunities(crm_root).items():
        raw_close = (row.get("close_date") or "").strip()
        try:
            close_date = date.fromisoformat(raw_close)
        except ValueError:
            continue
        if not (as_of <= close_date <= horizon_end):
            continue
        result[account_crm_id] = (close_date, row.get("name") or "")
    return result


def _account_slugs_by_crm_id(conn: psycopg.Connection) -> dict[str, str]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT crm_id, note_slug FROM registry_entities "
            "WHERE kind = 'account' AND active AND crm_id IS NOT NULL AND note_slug IS NOT NULL"
        )
        return dict(cur.fetchall())


def _renewal_risk_candidates(
    conn: psycopg.Connection,
    *,
    as_of: date,
    diff: DiffFeed,
    connector: Any,
) -> list[Candidate]:
    """Rule 2: an account with a renewal inside the horizon AND (a quiet
    window OR an open complaint)."""
    root = getattr(connector, "root", None) if connector is not None else None
    if root is None:
        return []
    crm_root = Path(root) / "crm"
    renewals = _load_open_renewals(crm_root, as_of=as_of, horizon_days=RENEWAL_HORIZON_DAYS)
    if not renewals:
        return []

    account_slugs_by_crm_id = _account_slugs_by_crm_id(conn)
    quiet_30 = set(diff.quiet_accounts.quiet_30)
    complaints_by_account = _open_complaint_claims_by_account(conn)

    candidates: list[Candidate] = []
    for account_crm_id, (close_date, opp_name) in renewals.items():
        account_slug = account_slugs_by_crm_id.get(account_crm_id)
        if account_slug is None:
            continue
        is_quiet = account_slug in quiet_30
        complaint_claims = complaints_by_account.get(account_slug, [])
        if not is_quiet and not complaint_claims:
            continue
        trigger = []
        if is_quiet:
            trigger.append("no claim has mentioned it in 30 days")
        if complaint_claims:
            trigger.append(f"{len(complaint_claims)} open complaint(s)")
        candidate_theme_slug = (
            _theme_for_claim(conn, complaint_claims[0]) if complaint_claims else None
        )
        candidates.append(
            Candidate(
                slug=f"renewal-risk:{account_slug}",
                rule="renewal_risk",
                account_slug=account_slug,
                theme_slug=candidate_theme_slug,
                claim_ids=sorted(complaint_claims),
                owner_crm_id=_account_owner_crm_id(conn, account_slug),
                reason=(
                    f"{opp_name or account_slug} renews {close_date.isoformat()}, within "
                    f"{RENEWAL_HORIZON_DAYS} days of {as_of.isoformat()}, and "
                    + " and ".join(trigger)
                ),
                detail={
                    "close_date": close_date.isoformat(),
                    "quiet_30": is_quiet,
                    "open_complaints": len(complaint_claims),
                },
            )
        )
    return candidates


def _account_slug_for_note(conn: psycopg.Connection, note_slug: str) -> str | None:
    """The account an action note belongs to: an account note body-linking
    to it, else the parent account of a contact note that does; `None` if
    neither. Other notes that link to the action (signals, reports) are
    skipped, and ties go to the first slug, so the answer does not depend on
    the query plan."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT CASE WHEN n.type = 'account' THEN re.note_slug ELSE parent_re.note_slug END
            FROM links l
            JOIN notes n ON n.slug = l.from_slug
            LEFT JOIN registry_entities re ON re.note_slug = n.slug
            LEFT JOIN registry_entities parent_re ON parent_re.id = re.parent_id
            WHERE l.to_slug = %s AND l.kind = 'body'
              AND n.type IN ('account', 'contact')
              AND CASE WHEN n.type = 'account' THEN re.note_slug ELSE parent_re.note_slug END
                  IS NOT NULL
            ORDER BY n.type = 'account' DESC, n.slug
            LIMIT 1
            """,
            (note_slug,),
        )
        row = cur.fetchone()
    return row[0] if row is not None else None


def _commitment_overdue_candidates(
    conn: psycopg.Connection, actions: ActionsFeed, *, as_of: date
) -> list[Candidate]:
    """Rule 3: a commitment past due -- every action in `actions.overdue`."""
    candidates: list[Candidate] = []
    for row in actions.overdue:
        account_slug = _account_slug_for_note(conn, row.note_slug)
        if account_slug is None:
            continue
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM claims WHERE note_slug = %s ORDER BY recorded LIMIT 1",
                (row.note_slug,),
            )
            claim_row = cur.fetchone()
        claim_ids = [claim_row[0]] if claim_row else []
        days_overdue = (as_of - row.due).days if row.due else None
        candidates.append(
            Candidate(
                slug=f"commitment-overdue:{row.id}",
                rule="commitment_overdue",
                account_slug=account_slug,
                theme_slug=_theme_for_claim(conn, claim_ids[0]) if claim_ids else None,
                claim_ids=claim_ids,
                owner_crm_id=row.owner_crm_id,
                reason=(
                    f"{row.what!r} was due {row.due.isoformat() if row.due else 'unknown'} "
                    f"and is still open as of {as_of.isoformat()}"
                    + (f" ({days_overdue} days overdue)" if days_overdue else "")
                ),
                detail={"action_id": row.id, "due": row.due.isoformat() if row.due else None},
            )
        )
    return candidates


def _standing_question_candidates(
    conn: psycopg.Connection, questions: list[StandingQuestionAnswer]
) -> list[Candidate]:
    """Rule 4: a standing question whose answer cites at least one note.
    Scoped to the account most of the cited notes' claims belong to; no
    attributable account means no candidate (an action needs an account)."""
    candidates: list[Candidate] = []
    for q in questions:
        if not q.citations:
            continue
        claim_ids: list[int] = []
        account_counts: dict[str, int] = {}
        with conn.cursor() as cur:
            for note_slug in q.citations:
                cur.execute(
                    "SELECT id, fact_type, claim_text FROM claims "
                    "WHERE note_slug = %s AND archived = false",
                    (note_slug,),
                )
                note_claims = cur.fetchall()
                for claim_id, fact_type, claim_text in note_claims:
                    claim_ids.append(claim_id)
                    account = home_account_slug(
                        conn, note_slug=note_slug, fact_type=fact_type, claim_text=claim_text
                    )
                    if account is not None:
                        account_counts[account] = account_counts.get(account, 0) + 1
        if not account_counts:
            continue
        account_slug = max(account_counts.items(), key=lambda item: (item[1], item[0]))[0]
        theme_slug = None
        for claim_id in claim_ids:
            theme_slug = _theme_for_claim(conn, claim_id)
            if theme_slug is not None:
                break
        candidates.append(
            Candidate(
                slug=f"standing-question:{q.question_id}:{account_slug}",
                rule="standing_question",
                account_slug=account_slug,
                theme_slug=theme_slug,
                claim_ids=sorted(set(claim_ids)),
                owner_crm_id=_account_owner_crm_id(conn, account_slug),
                reason=(
                    f"{q.reader}'s standing question {q.question!r} got a new answer this "
                    f"window, citing {len(q.citations)} note(s)"
                ),
                detail={"question_id": q.question_id, "reader": q.reader},
            )
        )
    return candidates


def _pricing_or_competitor_candidates(
    conn: psycopg.Connection,
    *,
    since: date,
    as_of: date,
    arr_by_note_slug: dict[str, float],
    arr_floor: float,
) -> list[Candidate]:
    """Rule 5: a pricing or competitor fact at an account above an ARR
    floor, recorded this window."""
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT id, note_slug, fact_type, claim_text FROM claims
            WHERE fact_type IN ('pricing', 'competitor_mention')
              AND archived = false
              AND {clock_module.world_date_sql("recorded")} BETWEEN %s AND %s
            """,
            (since, as_of),
        )
        rows = cur.fetchall()

    by_account: dict[str, list[int]] = {}
    for claim_id, note_slug, fact_type, claim_text in rows:
        account = home_account_slug(
            conn, note_slug=note_slug, fact_type=fact_type, claim_text=claim_text
        )
        if account is None:
            continue
        if arr_by_note_slug.get(account, 0.0) < arr_floor:
            continue
        by_account.setdefault(account, []).append(claim_id)

    candidates: list[Candidate] = []
    for account_slug, claim_ids in by_account.items():
        theme_slug = None
        for claim_id in claim_ids:
            theme_slug = _theme_for_claim(conn, claim_id)
            if theme_slug is not None:
                break
        candidates.append(
            Candidate(
                slug=f"pricing-or-competitor:{account_slug}",
                rule="pricing_or_competitor",
                account_slug=account_slug,
                theme_slug=theme_slug,
                claim_ids=sorted(claim_ids),
                owner_crm_id=_account_owner_crm_id(conn, account_slug),
                reason=(
                    f"{len(claim_ids)} pricing/competitor fact(s) recorded this window at "
                    f"{_account_name(conn, account_slug)}, ARR "
                    f"${arr_by_note_slug.get(account_slug, 0.0):,.0f} (floor ${arr_floor:,.0f})"
                ),
                detail={"arr": arr_by_note_slug.get(account_slug, 0.0)},
            )
        )
    return candidates


def find_candidates(
    conn: psycopg.Connection,
    *,
    since: date,
    as_of: date,
    diff: DiffFeed,
    priorities: list[ThemePriority],
    questions: list[StandingQuestionAnswer],
    actions: ActionsFeed,
    arr_by_note_slug: dict[str, float] | None = None,
    connector: Any = None,
    arr_floor: float = PRICING_COMPETITOR_ARR_FLOOR,
) -> list[Candidate]:
    """Every candidate for this report window, all five rules, sorted by
    (rule, slug) so a rerun on the same graph gives the same list.

    `connector` is duck-typed for `.root`; opportunities are read from
    `<root>/crm/` via `researcher_tools.load_latest_open_opportunities`.
    With no connector, rule 2 (renewal_risk) is skipped.
    """
    arr_by_note_slug = arr_by_note_slug or {}

    candidates: list[Candidate] = []
    candidates.extend(_theme_rising_candidates(conn, priorities, since=since, as_of=as_of))
    candidates.extend(_renewal_risk_candidates(conn, as_of=as_of, diff=diff, connector=connector))
    candidates.extend(_commitment_overdue_candidates(conn, actions, as_of=as_of))
    candidates.extend(_standing_question_candidates(conn, questions))
    candidates.extend(
        _pricing_or_competitor_candidates(
            conn, since=since, as_of=as_of, arr_by_note_slug=arr_by_note_slug, arr_floor=arr_floor
        )
    )
    candidates.sort(key=lambda c: (c.rule, c.slug))
    return candidates
