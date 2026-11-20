"""The CRM truth check: a CRM change that moves a deal or an account forward
while a recent call on the same account family said something that cuts the
other way (`01-spec/monday-email-2026-09-25.md`).

All rules, no model:
- a "forward" change is one `portolan.crm.changes` scored `direction = +1`
  (stage toward a close, amount up, close date pulled in, health or forecast
  up, a new opportunity);
- a call claim "cuts the other way" when its fact type is in
  `NEGATIVE_FACT_TYPES` or its text matches `NEGATIVE_TEXT_RE`;
- "recent" is from `LOOKBACK_DAYS` before the CRM export date to the end of
  the email's week; "the account family" is the account, its parents and its
  children in the registry.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta

import psycopg

from portolan.crm.changes import ChangeRow
from portolan.report.accounts import home_account_slug
from portolan.util import clock

__all__ = [
    "LOOKBACK_DAYS",
    "NEGATIVE_FACT_TYPES",
    "NEGATIVE_TEXT_RE",
    "CallClaim",
    "TruthFlag",
    "account_families",
    "call_claims",
    "is_negative",
    "truth_flags",
]

LOOKBACK_DAYS = 30
NEGATIVE_FACT_TYPES = frozenset({"risk"})
NEGATIVE_TEXT_RE = re.compile(
    r"\bbudgets?\b.{0,20}\b(frozen|freeze|cut|slashed|on hold)\b"
    r"|\b(freeze|frozen|on hold|paus\w*|cancel\w*|churn\w*|walk away)\b"
    r"|\b(second supplier|another vendor|switch\w* (to|vendors?))\b"
    r"|\bevaluat\w+ \w+(?: \w+)? (pumps?|systems?|vendors?|alternatives?|competitors?)\b"
    r"|\bnot (going to )?renew\w*\b"
    r"|\braised at the \d{4} renewal\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CallClaim:
    claim_id: int
    note_slug: str
    note_type: str
    account_slug: str | None
    fact_type: str
    claim_text: str
    source_slug: str
    locator: str
    day: date
    speaker: str | None
    quote: str | None = None


def call_claims(conn: psycopg.Connection, *, start: date, end: date) -> list[CallClaim]:
    """Live claims from call transcripts whose world date is in `[start,
    end]`, each with the account it belongs to (`home_account_slug`)."""
    day_sql = clock.world_date_sql("c.recorded")
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT c.id, c.note_slug, c.fact_type, c.claim_text, c.source_slug, c.locator,
                   {day_sql}, c.speaker, n.type, c.quote
            FROM claims c
            JOIN sources s ON s.slug = c.source_slug
            JOIN notes n ON n.slug = c.note_slug
            WHERE s.source_type = 'call' AND NOT c.archived AND c.superseded_at IS NULL
              AND {day_sql} BETWEEN %s AND %s
            ORDER BY {day_sql}, c.id
            """,
            (start, end),
        )
        rows = cur.fetchall()
    return [
        CallClaim(
            claim_id=r[0],
            note_slug=r[1],
            note_type=r[8],
            account_slug=home_account_slug(conn, note_slug=r[1], fact_type=r[2], claim_text=r[3]),
            fact_type=r[2],
            claim_text=r[3],
            source_slug=r[4],
            locator=r[5],
            day=r[6],
            speaker=r[7],
            quote=r[9],
        )
        for r in rows
    ]


def account_families(conn: psycopg.Connection) -> dict[str, set[str]]:
    """`{account crm_id: note slugs of it, its ancestors and its descendants}`."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, crm_id, parent_id, note_slug FROM registry_entities "
            "WHERE kind = 'account' AND crm_id IS NOT NULL"
        )
        rows = cur.fetchall()
    by_id = {r[0]: r for r in rows}
    children: dict[int, list[int]] = {}
    for r in rows:
        if r[2] is not None:
            children.setdefault(r[2], []).append(r[0])

    families: dict[str, set[str]] = {}
    for entity_id, crm_id, _parent, _slug in rows:
        members = {entity_id}
        cursor = by_id[entity_id][2]
        while cursor is not None and cursor in by_id and cursor not in members:
            members.add(cursor)
            cursor = by_id[cursor][2]
        stack = [entity_id]
        while stack:
            for child in children.get(stack.pop(), []):
                if child not in members:
                    members.add(child)
                    stack.append(child)
        families[crm_id] = {by_id[m][3] for m in members if by_id[m][3]}
    return families


def is_negative(claim: CallClaim) -> bool:
    return claim.fact_type in NEGATIVE_FACT_TYPES or bool(NEGATIVE_TEXT_RE.search(claim.claim_text))


@dataclass(frozen=True)
class TruthFlag:
    change: ChangeRow
    call: CallClaim


def truth_flags(
    changes: list[ChangeRow],
    calls: list[CallClaim],
    families: dict[str, set[str]],
    *,
    window_end: date,
    per_change: int = 1,
) -> list[TruthFlag]:
    """For each forward change, the most recent `per_change` negative call
    claims on its account family in the lookback window. `calls` must
    already cover that window."""
    flags: list[TruthFlag] = []
    for change in changes:
        if change.direction <= 0:
            continue
        family = families.get(change.account_crm_id, set())
        earliest = change.as_of - timedelta(days=LOOKBACK_DAYS)
        hits = [
            c
            for c in calls
            if c.account_slug in family and earliest <= c.day <= window_end and is_negative(c)
        ]
        hits.sort(key=lambda c: (c.day, c.claim_id), reverse=True)
        flags.extend(TruthFlag(change=change, call=c) for c in hits[:per_change])
    return flags
