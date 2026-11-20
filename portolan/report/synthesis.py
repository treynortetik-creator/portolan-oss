"""Account synthesis (`01-spec/surfacing-and-weekly-report.md` "per-account
synthesis"). Only accounts mentioned by a claim recorded in the window are
regenerated, so cost tracks activity.

The prompt's claims are the account note's own plus every claim on a signal
or action note linked from it (or named `signal-<fact_type>-<account>`),
newest first, capped at `_MAX_CLAIMS`; account-anchored facts live on
signal notes, not the account note. The note's own `## Claims` section is
unchanged.

A new synthesis is kept only if it cites no marker outside the claims given,
stays within `_MAX_LINES` lines, and passes `check_numerals` on each cited
segment (`_cited_segments`), so a claim's number is licensed only next to
that claim's own marker (ARR and the open-action count are the only computed
figures, licensed anywhere). On failure the old synthesis stays and a
`review_queue` row is written.
"""

from __future__ import annotations

import json
from datetime import date

import psycopg
from pydantic import BaseModel, ConfigDict

from portolan.extract.schema import FactType
from portolan.llm.base import LLMClient
from portolan.notes import read_note, render_note, write_note
from portolan.notes.render import CITATION_RE, signal_note_slug
from portolan.report.models import SynthesisOutcome, SynthesisReport
from portolan.report.numbers import NumberFacts, check_numerals, queue_numeral_rejection
from portolan.report.util import validate_citations
from portolan.util import clock

__all__ = ["ACCOUNT_SYNTHESIS_SYSTEM", "refresh_account_synthesis"]

ACCOUNT_SYNTHESIS_SYSTEM = (
    "You write the top-of-note synthesis for one customer account: what "
    "they care about, what they complain about, open requests, a risk "
    "read, key contacts, and the date of last contact. Base every "
    "sentence only on the claims and open actions given below -- cite the "
    "claim's own source with its exact marker, copied verbatim, never "
    "invented or reformatted. Under 25 lines. Plain language, absolute "
    "dates, never 'recently' or 'this week'. Every number you state must "
    "be copied verbatim from the 'Figures you may state' list below -- "
    "never compute, round, sum, or estimate a number yourself. Reply as "
    "JSON."
)

_MAX_CLAIMS = 60
_MAX_LINES = 25


class _AccountSynthesis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    synthesis: str


def _account_synthesis_json_schema() -> dict:
    return _AccountSynthesis.model_json_schema()


def _grown_accounts(conn: psycopg.Connection, since: date, as_of: date) -> list[str]:
    """Living accounts with at least one claim recorded in the window that
    mentions them as `[[account-slug]]` (not only claims homed on the account)."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT slug FROM notes WHERE type = 'account' AND status = 'living' ORDER BY slug"
        )
        account_slugs = [r[0] for r in cur.fetchall()]
    if not account_slugs:
        return []

    # One query for every account rather than one LIKE round-trip each.
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT a.slug
            FROM unnest(%s::text[]) AS a(slug)
            WHERE EXISTS (
                SELECT 1 FROM claims c
                WHERE c.claim_text LIKE '%%[[' || a.slug || ']]%%'
                  AND {clock.world_date_sql("c.recorded")} BETWEEN %s AND %s
            )
            ORDER BY a.slug
            """,
            (account_slugs, since, as_of),
        )
        return [r[0] for r in cur.fetchall()]


def _registry_fields(conn: psycopg.Connection, account_note_slug: str) -> dict:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT re.canonical_name, re.crm_id, re.account_kind, parent.canonical_name
            FROM registry_entities re
            LEFT JOIN registry_entities parent ON parent.id = re.parent_id
            WHERE re.note_slug = %s
            """,
            (account_note_slug,),
        )
        row = cur.fetchone()
    if row is None:
        return {}
    name, crm_id, account_kind, parent_name = row
    return {
        "name": name,
        "crm_id": crm_id,
        "account_kind": account_kind,
        "parent_name": parent_name,
    }


def _open_actions_for_account(conn: psycopg.Connection, account_note_slug: str) -> list[tuple]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT a.what, a.due, a.status
            FROM actions a
            JOIN links l ON l.to_slug = a.note_slug AND l.kind = 'body'
            WHERE l.from_slug = %s AND a.status IN ('open', 'overdue')
            ORDER BY a.id
            """,
            (account_note_slug,),
        )
        return cur.fetchall()


def _linked_signal_and_action_notes(conn: psycopg.Connection, account_slug: str) -> list[str]:
    """Signal/action notes linked from the account by a `body` edge, plus
    any note named exactly `signal-<fact_type>-<account_slug>`. Exact names
    over every `FactType`, not a LIKE pattern: `signal-%-halvard` would also
    match another account's `signal-billing-west-halvard`."""
    candidate_slugs = [signal_note_slug(ft.value, account_slug) for ft in FactType]
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT n.slug
            FROM notes n
            WHERE n.type IN ('signal', 'action')
              AND (
                    EXISTS (
                        SELECT 1 FROM links l
                        WHERE l.from_slug = %s AND l.to_slug = n.slug AND l.kind = 'body'
                    )
                 OR n.slug = ANY(%s)
              )
            ORDER BY n.slug
            """,
            (account_slug, candidate_slugs),
        )
        return [r[0] for r in cur.fetchall()]


def _account_claim_pool(conn: psycopg.Connection, account_slug: str, account_claims: list) -> list:
    """`account_claims` plus every claim on a linked signal/action note.
    Prompt and citation allow-list only; `write_note` still gets
    `account_claims` alone."""
    pool = list(account_claims)
    for note_slug in _linked_signal_and_action_notes(conn, account_slug):
        _note, claims = read_note(conn, note_slug)
        pool.extend(claims)
    return pool


def _build_user_prompt(
    fields: dict, claims: list, actions: list[tuple], arr: float | None, facts: NumberFacts
) -> str:
    lines = [f"Account: {fields.get('name', '?')} (CRM {fields.get('crm_id', '?')})"]
    if fields.get("account_kind"):
        lines.append(f"Kind: {fields['account_kind']}")
    if fields.get("parent_name"):
        lines.append(f"Parent: {fields['parent_name']}")
    if arr is not None:
        lines.append(f"Annual recurring revenue: ${arr:,.0f}")
    lines.append("")
    lines.append("Claims (newest first):")
    for claim_text, source_slug, locator, recorded in claims:
        lines.append(f"- ({recorded:%Y-%m-%d}) {claim_text} ^[{source_slug} @ {locator}]")
    lines.append("")
    lines.append("Open actions:")
    if actions:
        for what, due, status in actions:
            due_txt = f", due {due:%Y-%m-%d}" if due else ""
            lines.append(f"- {what} ({status}{due_txt})")
    else:
        lines.append("(none)")
    lines.append("")
    lines.append("Figures you may state (use no other numbers):")
    lines.append(facts.as_table())
    return "\n".join(lines)


# Punctuation right after a marker still belongs to the segment it closes.
_SEGMENT_TAIL = ".,;:!?)"


def _cited_segments(text: str) -> list[str]:
    """`text` cut after each citation marker, line by line: each segment is
    the text its closing marker vouches for. Text after a line's last marker
    is a segment with no marker."""
    segments: list[str] = []
    for line in text.splitlines():
        start = 0
        for match in CITATION_RE.finditer(line):
            end = match.end()
            while end < len(line) and line[end] in _SEGMENT_TAIL:
                end += 1
            segments.append(line[start:end])
            start = end
        if line[start:].strip():
            segments.append(line[start:])
    return segments


def _mechanical_check(new_synthesis: str, allowed_markers: set[tuple[str, str]]) -> str | None:
    """`None` if `new_synthesis` passes; otherwise the reason it failed."""
    return validate_citations(new_synthesis, allowed_markers, max_lines=_MAX_LINES)


def _queue_failed_synthesis(conn: psycopg.Connection, account_slug: str, reason: str) -> None:
    payload = {
        "reason": "failed account synthesis",
        "note_slug": account_slug,
        "detail": reason,
    }
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO review_queue (kind, payload) VALUES ('other', %s::jsonb)",
            (json.dumps(payload),),
        )
    conn.commit()


def refresh_account_synthesis(
    conn: psycopg.Connection,
    llm: LLMClient,
    *,
    since: date,
    as_of: date | None = None,
    arr_by_note_slug: dict[str, float] | None = None,
    embedder=None,
    prose_model: str | None = None,
) -> SynthesisReport:
    """Rewrite the synthesis of every account active in `[since, as_of]`.

    `embedder`: forwarded to `write_note` so a rewritten synthesis is
    re-embedded now; `None` leaves that to the next reindex. `prose_model`:
    model for the one `llm.complete()` per account (`None` = the client's
    default)."""
    as_of = as_of or clock.today()
    arr_by_note_slug = arr_by_note_slug or {}
    outcomes: list[SynthesisOutcome] = []

    for account_slug in _grown_accounts(conn, since, as_of):
        note, claims = read_note(conn, account_slug)
        claim_pool = _account_claim_pool(conn, account_slug, claims)
        ordered = sorted(claim_pool, key=lambda c: c.recorded, reverse=True)[:_MAX_CLAIMS]
        claim_rows = [(c.claim_text, c.source_slug, c.locator, c.recorded) for c in ordered]
        allowed_markers = {(c.source_slug, c.locator) for c in ordered}
        actions = _open_actions_for_account(conn, account_slug)
        fields = _registry_fields(conn, account_slug)
        # Unknown ARR stays `None`, never $0: the prompt omits the ARR line
        # and no `account_arr` figure is licensed.
        arr = arr_by_note_slug.get(account_slug)

        # ARR and the open-action count are the computed figures. A claim's
        # numerals are licensed only in a segment that cites that claim;
        # action text has no citation, so its numerals are licensed anywhere.
        facts = NumberFacts()
        if arr is not None:
            facts.add("account_arr", arr, f"{arr:,.0f}")
        facts.add("open_actions_count", len(actions))
        for claim_text, source_slug, locator, _recorded in claim_rows:
            facts.harvest(claim_text, marker=(source_slug, locator))
        facts.harvest("\n".join(what for what, *_ in actions))

        result = llm.complete(
            system=ACCOUNT_SYNTHESIS_SYSTEM,
            user=_build_user_prompt(fields, claim_rows, actions, arr, facts),
            json_schema=_account_synthesis_json_schema(),
            model=prose_model,
            # Metered as report prose, so the report CLI's cost line counts it.
            surface="report",
        )
        parsed = _AccountSynthesis.model_validate(result.data)

        failure = _mechanical_check(parsed.synthesis, allowed_markers)
        if failure is not None:
            _queue_failed_synthesis(conn, account_slug, failure)
            outcomes.append(
                SynthesisOutcome(
                    account_slug=account_slug, status="kept_old_failed_check", reason=failure
                )
            )
            continue

        numeral_violations = [
            violation
            for segment in _cited_segments(parsed.synthesis)
            for violation in check_numerals(segment, facts)
        ]
        if numeral_violations:
            queue_numeral_rejection(
                conn,
                note_slug=account_slug,
                surface="account_synthesis",
                violations=numeral_violations,
                text=parsed.synthesis,
                facts=facts,
            )
            outcomes.append(
                SynthesisOutcome(
                    account_slug=account_slug,
                    status="kept_old_failed_check",
                    reason=(
                        f"numeral(s) not in the supplied figures: {', '.join(numeral_violations)}"
                    ),
                )
            )
            continue

        note.synthesis = parsed.synthesis
        # Skip the write (no version bump, no `updated` re-stamp) when the
        # rendered note is unchanged.
        if render_note(note, claims) == note.body_md:
            outcomes.append(SynthesisOutcome(account_slug=account_slug, status="kept_unchanged"))
            continue
        write_note(
            conn, note, claims, changed_by="report", reason="weekly synthesis", embedder=embedder
        )
        outcomes.append(SynthesisOutcome(account_slug=account_slug, status="rewritten"))

    return SynthesisReport(since=since, as_of=as_of, outcomes=outcomes)
