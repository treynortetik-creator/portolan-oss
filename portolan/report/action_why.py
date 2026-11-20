"""The action note's "Why" block: render what a confirmed research card
already proved, not new model prose.

- `persist_action_why`: at confirm time, freezes the card's evidence,
  inference, confidence and the `researcher_findings` rows it cites onto
  one `action_why` row per action (upsert).
- `build_why_context` / `why_summary_line`: the read side, for the note
  page and `/actions`.
- `find_duplicate_open_action` / `merge_duplicate_action`: a card whose
  evidence is a live claim on another open action's note names a
  duplicate; the older action takes the card's owner/due and the newer
  note is archived, never deleted.

Every line this module writes comes from stored tool fields, never the
model's prose; `finding_plain_text` has one branch per known tool.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

from portolan.notes.models import Claim, NoteStatus
from portolan.notes.persist import append_changelog, claims_by_ids, read_note, write_note
from portolan.util import clock as clock_module

__all__ = [
    "build_why_context",
    "find_duplicate_open_action",
    "finding_plain_text",
    "load_action_why",
    "merge_duplicate_action",
    "persist_action_why",
    "why_summary_line",
]


# ---------------------------------------------------------------------
# Persist (Confirm time).
# ---------------------------------------------------------------------


def persist_action_why(
    conn: psycopg.Connection,
    *,
    action_id: int,
    card: dict[str, Any],
    editor: str,
    review_row_id: int | None = None,
    confirmed_at: datetime | None = None,
    commit: bool = True,
) -> None:
    """Upsert `card`'s grounded content onto `action_why` for `action_id`.
    `proposed_at` is the `created` time of review row `review_row_id`, if
    any, and keeps its first non-null value across calls. Every other column
    is overwritten, including `confirmed_at` (default: now), so a repeat
    call is a no-op only with the same card, editor and `confirmed_at`.
    """
    proposed_at: datetime | None = None
    if review_row_id is not None:
        with conn.cursor() as cur:
            cur.execute("SELECT created FROM review_queue WHERE id = %s", (review_row_id,))
            row = cur.fetchone()
        if row is not None:
            proposed_at = row[0]

    finding_ids = [fid for fid in (card.get("findings") or []) if fid is not None]
    findings: list[dict[str, Any]] = []
    if finding_ids:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, tool, arguments, result FROM researcher_findings WHERE id = ANY(%s)",
                (finding_ids,),
            )
            by_id = {
                r[0]: {"id": r[0], "tool": r[1], "arguments": r[2], "result": r[3]}
                for r in cur.fetchall()
            }
        findings = [by_id[fid] for fid in finding_ids if fid in by_id]

    confirmed_at = confirmed_at or clock_module.now()

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO action_why
                (action_id, review_row_id, run_id, candidate_slug, verdict, evidence,
                 findings, inference, confidence, confidence_refs, proposed_at,
                 confirmed_by, confirmed_at)
            VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s, %s::jsonb, %s, %s, %s)
            ON CONFLICT (action_id) DO UPDATE SET
                review_row_id   = EXCLUDED.review_row_id,
                run_id          = EXCLUDED.run_id,
                candidate_slug  = EXCLUDED.candidate_slug,
                verdict         = EXCLUDED.verdict,
                evidence        = EXCLUDED.evidence,
                findings        = EXCLUDED.findings,
                inference       = EXCLUDED.inference,
                confidence      = EXCLUDED.confidence,
                confidence_refs = EXCLUDED.confidence_refs,
                proposed_at     = COALESCE(action_why.proposed_at, EXCLUDED.proposed_at),
                confirmed_by    = EXCLUDED.confirmed_by,
                confirmed_at    = EXCLUDED.confirmed_at
            """,
            (
                action_id,
                review_row_id,
                card.get("run_id"),
                card.get("candidate_slug"),
                card.get("verdict"),
                Jsonb(card.get("evidence") or []),
                Jsonb(findings),
                Jsonb(card.get("inference") or []),
                card.get("confidence"),
                Jsonb(card.get("confidence_refs") or {}),
                proposed_at,
                editor,
                confirmed_at,
            ),
        )
    if commit:
        conn.commit()


def load_action_why(conn: psycopg.Connection, action_id: int) -> dict[str, Any] | None:
    """The raw stored row, or `None` for an action with no card."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT evidence, findings, inference, confidence, confidence_refs, "
            "proposed_at, confirmed_by, confirmed_at FROM action_why WHERE action_id = %s",
            (action_id,),
        )
        row = cur.fetchone()
    if row is None:
        return None
    (
        evidence,
        findings,
        inference,
        confidence,
        confidence_refs,
        proposed_at,
        confirmed_by,
        confirmed_at,
    ) = row
    return {
        "evidence": evidence or [],
        "findings": findings or [],
        "inference": inference or [],
        "confidence": confidence,
        "confidence_refs": confidence_refs or {},
        "proposed_at": proposed_at,
        "confirmed_by": confirmed_by,
        "confirmed_at": confirmed_at,
    }


# ---------------------------------------------------------------------
# Read side: raw row -> template-ready context.
# ---------------------------------------------------------------------


def _note_titles(conn: psycopg.Connection, slugs: list[str]) -> dict[str, str]:
    real_slugs = [s for s in slugs if s]
    if not real_slugs:
        return {}
    with conn.cursor() as cur:
        cur.execute("SELECT slug, title FROM notes WHERE slug = ANY(%s)", (real_slugs,))
        return dict(cur.fetchall())


def _resolved_refs(
    refs: dict[str, Any], evidence: list[dict[str, Any]], findings: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ev = [evidence[i] for i in (refs.get("evidence_refs") or []) if 0 <= i < len(evidence)]
    fi = [findings[j] for j in (refs.get("finding_refs") or []) if 0 <= j < len(findings)]
    return ev, fi


def build_why_context(conn: psycopg.Connection, action_id: int) -> dict[str, Any] | None:
    """`load_action_why`'s row resolved for the note's "Why" block:
    inference refs resolved to items, each finding with a `plain_text`, and
    `claims`, the live `Claim` rows the evidence points at. `None` when the
    action has no card."""
    row = load_action_why(conn, action_id)
    if row is None:
        return None

    evidence = row["evidence"]
    findings = [{**f, "plain_text": finding_plain_text(conn, f)} for f in row["findings"]]
    inference = []
    for sentence in row["inference"]:
        ev, fi = _resolved_refs(sentence, evidence, findings)
        inference.append({"sentence": sentence.get("sentence"), "evidence": ev, "findings": fi})
    confidence_evidence, confidence_findings = _resolved_refs(
        row["confidence_refs"], evidence, findings
    )

    claim_ids = [e.get("claim_id") for e in evidence if e.get("claim_id") is not None]
    claims: list[Claim] = claims_by_ids(conn, claim_ids)

    return {
        "inference": inference,
        "evidence": evidence,
        "findings": findings,
        "confidence": row["confidence"],
        "confidence_evidence": confidence_evidence,
        "confidence_findings": confidence_findings,
        "proposed_at": row["proposed_at"],
        "confirmed_by": row["confirmed_by"],
        "confirmed_at": row["confirmed_at"],
        "claims": claims,
    }


def why_summary_line(conn: psycopg.Connection, action_id: int) -> str | None:
    """The `/actions` one-line "why": the first inference sentence, else the
    first evidence text, else `None` (also `None` with no card)."""
    row = load_action_why(conn, action_id)
    if row is None:
        return None
    inference = row["inference"]
    if inference:
        return inference[0].get("sentence")
    evidence = row["evidence"]
    if evidence:
        return evidence[0].get("text")
    return None


# ---------------------------------------------------------------------
# finding_plain_text -- one branch per known finding tool, numbers taken
# straight from the stored `result`, never composed from prose.
# ---------------------------------------------------------------------


def _join_or(items: list[str]) -> str:
    items = [i for i in items if i]
    if not items:
        return "claims"
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} or {items[1]}"
    return ", ".join(items[:-1]) + f", or {items[-1]}"


def _scope_label(conn: psycopg.Connection, arguments: dict[str, Any]) -> str:
    account_slug = arguments.get("account_slug")
    theme_slug = arguments.get("theme_slug")
    titles = _note_titles(conn, [s for s in (account_slug, theme_slug) if s])
    if account_slug and theme_slug:
        return f"{titles.get(account_slug, account_slug)} / {titles.get(theme_slug, theme_slug)}"
    if account_slug:
        return titles.get(account_slug, account_slug)
    if theme_slug:
        return titles.get(theme_slug, theme_slug)
    return "the graph"


def _window_clause(since: str | None, until: str | None) -> str:
    if since and until:
        return f" from {since} to {until}"
    if since:
        return f" since {since}"
    if until:
        return f" through {until}"
    return ""


def _absence_check_plain(
    conn: psycopg.Connection, arguments: dict[str, Any], result: dict[str, Any]
) -> str:
    label = f"{_join_or(arguments.get('fact_types') or [])} claims"
    scope = _scope_label(conn, arguments)
    after = arguments.get("after_date") or "an unknown date"
    count = result.get("count", 0)
    count_before = result.get("count_before", 0)
    latest = result.get("latest_date")
    if count == 0:
        head = f"No {label} recorded for {scope} after {after}"
    else:
        latest_clause = f" (latest {latest})" if latest else ""
        head = f"{count} {label} recorded for {scope} after {after}{latest_clause}"
    return f"{head}; {count_before} before."


def _count_by_plain(
    conn: psycopg.Connection, arguments: dict[str, Any], result: dict[str, Any]
) -> str:
    fact_type = arguments.get("fact_type")
    theme_slug = arguments.get("theme_slug")
    group_by = arguments.get("group_by") or "account"
    total = result.get("total", 0)
    groups = result.get("groups") or []
    if fact_type:
        label = f"{fact_type} claim(s)"
    elif theme_slug:
        titles = _note_titles(conn, [theme_slug])
        label = f"{titles.get(theme_slug, theme_slug)} theme claim(s)"
    else:
        label = "claim(s)"
    window = _window_clause(arguments.get("since"), arguments.get("until"))
    top = ", ".join(f"{g['group']}: {g['count']}" for g in groups[:5])
    more = f", +{len(groups) - 5} more" if len(groups) > 5 else ""
    breakdown = f": {top}{more}" if top else ""
    return f"{total} {label} by {group_by}{window}{breakdown}."


def _timeline_plain(
    conn: psycopg.Connection, arguments: dict[str, Any], result: dict[str, Any]
) -> str:
    scope = _scope_label(conn, arguments)
    count = result.get("count", 0)
    entries = result.get("entries") or []
    window = _window_clause(arguments.get("since"), arguments.get("until"))
    span = ""
    if entries:
        first, last = entries[0].get("date"), entries[-1].get("date")
        if first and last:
            span = f", {first} to {last}"
    return f"{count} claim(s) for {scope}{window}{span}."


def _renewal_plain(
    conn: psycopg.Connection, arguments: dict[str, Any], result: dict[str, Any]
) -> str:
    scope = _scope_label(conn, arguments)
    close_date = result.get("close_date") or "no close date on file"
    arr = result.get("arr")
    arr_str = f"${arr:,.0f}" if isinstance(arr, int | float) else "no ARR on file"
    stage = result.get("stage")
    stage_str = f", stage {stage}" if stage else ""
    n = len(result.get("renewal_claims") or [])
    return (
        f"Renewal on file for {scope}: close {close_date}{stage_str}, ARR {arr_str}; "
        f"{n} claim(s) mention a renewal date."
    )


_FINDING_RENDERERS = {
    "absence_check": _absence_check_plain,
    "count_by": _count_by_plain,
    "timeline": _timeline_plain,
    "renewal": _renewal_plain,
}


def finding_plain_text(conn: psycopg.Connection, finding: dict[str, Any]) -> str:
    """One stored finding (`{tool, arguments, result}`) as a plain sentence
    built only from its stored fields. An unknown tool falls back to
    `"<tool>: <result>"` rather than raising."""
    tool = finding.get("tool")
    arguments = finding.get("arguments") or {}
    result = finding.get("result") or {}
    renderer = _FINDING_RENDERERS.get(tool)
    if renderer is None:
        return f"{tool}: {result}"
    return renderer(conn, arguments, result)


# ---------------------------------------------------------------------
# No duplicates.
# ---------------------------------------------------------------------


def find_duplicate_open_action(
    conn: psycopg.Connection,
    *,
    account_note_slug: str,
    evidence_claim_ids: list[int],
    exclude_action_id: int | None = None,
) -> int | None:
    """The oldest open action body-linked from `account_note_slug` whose own
    note carries one of `evidence_claim_ids` as a live claim, or `None`.
    `exclude_action_id` skips one action (so an action never matches itself).
    """
    if not evidence_claim_ids:
        return None
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT a.id
            FROM actions a
            JOIN claims c ON c.note_slug = a.note_slug
            JOIN links l ON l.to_slug = a.note_slug AND l.kind = 'body'
            WHERE l.from_slug = %s
              AND a.status = 'open'
              AND c.id = ANY(%s)
              AND c.archived = false
              AND (%s::bigint IS NULL OR a.id != %s)
            ORDER BY a.id
            LIMIT 1
            """,
            (account_note_slug, evidence_claim_ids, exclude_action_id, exclude_action_id),
        )
        row = cur.fetchone()
    return row[0] if row is not None else None


def merge_duplicate_action(
    conn: psycopg.Connection,
    *,
    keep_action_id: int,
    drop_action_id: int,
    card: dict[str, Any] | None,
    editor: str,
    reason: str = "duplicate action merged",
    commit: bool = True,
) -> bool:
    """Merge `drop_action_id` into `keep_action_id`: mark the dropped action
    `dropped` (with an `edit_records` row) and archive its note (with a
    changelog row), never delete. `card`, when given, supplies the owner/due
    written onto the kept action on every call.

    Returns `True` only if the drop or archive step changed something, so a
    replay on an already-merged pair returns `False`.
    """
    with conn.cursor() as cur:
        cur.execute("SELECT note_slug, status FROM actions WHERE id = %s", (drop_action_id,))
        drop_row = cur.fetchone()
    if drop_row is None:
        return False
    drop_note_slug, drop_status = drop_row

    changed = False

    if card is not None:
        owner_crm_id = card.get("owner")
        raw_due = card.get("due")
        due_value = date.fromisoformat(raw_due) if isinstance(raw_due, str) else raw_due
        owner_dbid = None
        if owner_crm_id:
            with conn.cursor() as cur:
                cur.execute("SELECT id FROM registry_entities WHERE crm_id = %s", (owner_crm_id,))
                owner_row = cur.fetchone()
            owner_dbid = owner_row[0] if owner_row is not None else None
        if owner_dbid is not None:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE actions SET owner_entity_id = %s WHERE id = %s",
                    (owner_dbid, keep_action_id),
                )
        if due_value is not None:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE actions SET due = %s WHERE id = %s", (due_value, keep_action_id)
                )

    if drop_status != "dropped":
        with conn.cursor() as cur:
            cur.execute("UPDATE actions SET status = 'dropped' WHERE id = %s", (drop_action_id,))
            cur.execute(
                "INSERT INTO edit_records (note_slug, editor, before_text, after_text, reason) "
                "VALUES (%s, %s, %s, %s, %s)",
                (
                    drop_note_slug,
                    editor,
                    drop_status,
                    "dropped",
                    f"merged into action {keep_action_id} (duplicate)",
                ),
            )
        changed = True

    note, claims = read_note(conn, drop_note_slug)
    if note.status != NoteStatus.ARCHIVED:
        note.status = NoteStatus.ARCHIVED
        write_note(conn, note, claims, changed_by=editor, reason=reason, commit=commit)
        append_changelog(
            conn,
            "action_duplicate_merged",
            f"[[{drop_note_slug}]] merged into an existing action as a duplicate.",
            {"kept_action_id": keep_action_id, "dropped_action_id": drop_action_id},
            commit=commit,
        )
        changed = True

    if commit:
        conn.commit()
    return changed
