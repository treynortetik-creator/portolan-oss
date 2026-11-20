"""`/review`: open `review_queue` rows, with resolve/dismiss and, for
`unknown_entity`/`ambiguous_entity` rows, an "add as alias of" registry
picker (portolan-spec-v2.md sec 3: "A human maps a name once; it sticks").
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any
from urllib.parse import quote

import psycopg
from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import ValidationError

from portolan.activity import record_activity
from portolan.extract.schema import Coverage, ExtractionResult, Fact
from portolan.ingest.intake import SOURCE_COLUMNS, row_to_source_row
from portolan.ingest.models import SourceRow
from portolan.llm.base import LLMClient
from portolan.merge import merge_extraction
from portolan.registry import Registry, add_alias
from portolan.registry.owners import confirm_owner_proposal
from portolan.report.action_done import confirm_action_done
from portolan.report.action_why import finding_plain_text
from portolan.report.researcher import ConfirmOutcome, confirm_action_card_with_reason
from portolan.report.themes import (
    confirm_theme_resolution,
    reject_theme_resolution,
    resolution_proposal_context,
)
from portolan.search.embed import Embedder
from portolan.store import Store
from portolan.web.auth import Session, require_editor, require_session, require_session_api
from portolan.web.bounds import bounded_int, bounded_str
from portolan.web.deps import get_conn, get_embedder_dep, get_llm_dep, get_store_dep
from portolan.web.queries import (
    backfill_fact_evidence,
    note_summaries,
    registry_alias_conflict,
    registry_entities_by_crm_ids,
    registry_entities_search,
    review_queue_open_index,
    review_queue_rows_by_ids,
    sources_brief,
)
from portolan.web.review_presenter import category_label_for, present_rows
from portolan.web.routes.originals import context_snippet
from portolan.web.tracking import track_page_view

router = APIRouter()
_logger = logging.getLogger(__name__)

# The queue is paginated oldest first, and the registry picker is fetched on
# demand (`GET /review/registry?q=`) rather than shipped with every card.
REVIEW_PAGE_SIZE = 25
MAX_REVIEW_PAGE = 100_000
REGISTRY_PICKER_LIMIT = 20
MAX_REGISTRY_QUERY_CHARS = 200
MAX_REVIEW_KIND_CHARS = 100


# The `?notice=` keys the review routes redirect with, mapped to the
# plain-language sentence `review_page` shows. A fixed dict, so the URL
# carries a short token, never reflected free text.
_NOTICES = {
    "superseded": (
        "That resolution was already superseded (confirmed away, reopened some other "
        "way, or replaced by a newer resolution) — nothing was changed."
    ),
    "malformed": (
        "That row was missing the theme/claim it should have pointed at — nothing was changed."
    ),
    "confirmed": "Resolution confirmed — the theme now reads resolved.",
    "rejected": "Resolution rejected — the theme is back to whatever it would read otherwise.",
    # A second submit on a row that is no longer open.
    "already_handled": (
        "That row was already resolved (probably by an earlier submit of this same "
        "form) — nothing was changed."
    ),
    # The mention already maps to a different record; the template names
    # that record on the card itself.
    "alias_conflict": (
        "That name is already mapped to a different record, so nothing was changed "
        "— a name keeps its first mapping. See the note on the card below."
    ),
    # `quote_unsupported` "Keep as claim" (`_keep_quote_unsupported_fact`).
    "kept_as_claim": "Kept — this fact is now on the graph as a claim.",
    "kept_unresolved": (
        "Kept, but Portolan couldn't tell which account, contact, or product it belongs "
        "to, so it's now parked under a different review row instead of filed as a claim."
    ),
    "kept_matched_existing": (
        "Kept — the graph already has this fact as a claim, so no new claim was added."
    ),
    # `owner_proposal` (01-spec/actions-and-owners-2026-09-19.md).
    "owner_confirmed": "Confirmed — the account's owner is set.",
    "owner_confirm_failed": (
        "Nothing was changed — the proposal's candidate is no longer a valid vendor-side "
        "record, or the account no longer exists."
    ),
    # `action_card` (the action researcher's card).
    "action_confirmed": "Confirmed — the action is now open on the Actions page.",
    "action_confirm_failed": (
        "Nothing was changed — the owner isn't a valid vendor-side record, the account no "
        "longer exists, the evidence it cites has changed since it was queued, or the card "
        "has no evidence claim left to cite a source from."
    ),
    # A proposed due date can pass while the card waits; the reviewer can
    # set a new one on the card.
    "action_due_stale": (
        "Nothing was changed — the due date the researcher proposed has passed. Set a new "
        "due date on the card and confirm again."
    ),
    "action_card_concluded": (
        "Recorded — the researcher's conclusion stands and no action was created."
    ),
    "action_due_invalid": (
        "Nothing was changed — the due date has to be today or later, and no more than a "
        "year out. Pick a new date and confirm again."
    ),
    # Auto-close from later evidence (`portolan.report.action_done`).
    "action_done_confirmed": (
        "Confirmed — the action is marked done, and the claim that answered it is kept on "
        "the action."
    ),
    "action_done_already_closed": (
        "That action was already closed, so nothing was changed. The proposal is cleared."
    ),
    "action_done_evidence_changed": (
        "Nothing was changed — the claim this proposal cites was edited or retracted after "
        "it was proposed. Dismiss the proposal; Portolan proposes again if other evidence "
        "turns up."
    ),
}


def _enrich_theme_resolution_rows(conn: psycopg.Connection, rows: list[dict[str, Any]]) -> None:
    """Add a `proposal` key (`resolution_proposal_context`: theme name,
    claim, account, date) to every `kind='theme_resolution'` row in place;
    the payload carries only slugs and ids. `None` when the theme or claim
    no longer exists."""
    for row in rows:
        if row["kind"] != "theme_resolution":
            continue
        payload = row["payload"] or {}
        theme_slug = payload.get("theme_slug")
        resolution_claim_id = payload.get("resolution_claim_id")
        row["proposal"] = (
            resolution_proposal_context(
                conn, theme_slug=theme_slug, resolution_claim_id=resolution_claim_id
            )
            if theme_slug is not None and resolution_claim_id is not None
            else None
        )


def _snippet_source_slug(row: dict[str, Any]) -> str | None:
    """Which source (if any) a row's evidence points at. A
    `theme_resolution` row has no `source_slug`; its evidence is the
    resolving claim, reached through `row['proposal']`."""
    if row["kind"] == "theme_resolution":
        return (row.get("proposal") or {}).get("source_slug")
    return row.get("source_slug")


def _locator_for(row: dict[str, Any]) -> str | None:
    payload = row["payload"] or {}
    if row["kind"] == "unknown_entity":
        return (payload.get("context") or {}).get("locator")
    if row["kind"] == "ambiguous_entity":
        return payload.get("locator")
    if row["kind"] == "theme_resolution":
        return (row.get("proposal") or {}).get("locator")
    if row["kind"] == "quote_unsupported":
        return (payload.get("fact") or {}).get("locator")
    if row["kind"] == "action_done":
        return (payload.get("evidence") or {}).get("locator")
    return None


def _note_slug_for(row: dict[str, Any]) -> str | None:
    """`payload['note_slug']` for `failed_consolidation` and `other` rows
    (several `other` reasons carry one), so the card can show the note's
    title instead of its slug."""
    if row["kind"] in ("failed_consolidation", "other"):
        return (row["payload"] or {}).get("note_slug")
    return None


def _needs_evidence_backfill(row: dict[str, Any]) -> bool:
    """True for an `unknown_entity`, or document `ambiguous_entity`, row with
    no `claim` in its payload (older rows lack it; `Fact.claim` is required,
    so absence means not saved, not empty). A CRM-import `ambiguous_entity`
    row has no `source_slug` to search and is excluded."""
    if row["kind"] not in ("unknown_entity", "ambiguous_entity"):
        return False
    if row["kind"] == "ambiguous_entity" and not row.get("source_slug"):
        return False
    payload = row["payload"] or {}
    if row["kind"] == "unknown_entity":
        claim = (payload.get("context") or {}).get("claim")
    else:
        claim = payload.get("claim")
    return claim is None


def _mention_for(row: dict[str, Any]) -> str | None:
    return (row["payload"] or {}).get("mention")


def _enrich_action_card_row(conn: psycopg.Connection, row: dict[str, Any]) -> None:
    """Set `row["action_context"]` for the action card (01-spec/actions-and-
    owners-2026-09-19.md "Phase 2, revised contract", "Review page"): finding
    ids resolved to `researcher_findings` rows, each inference sentence's
    `evidence_refs`/`finding_refs` resolved to the items they index, the
    account/theme titles, and the active internal people for the owner
    picker."""
    payload = row.get("payload") or {}
    card = payload.get("card") or {}
    if not card:
        return

    evidence = card.get("evidence") or []
    finding_ids = card.get("findings") or []
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

    def _resolved_refs(refs: dict[str, Any]) -> dict[str, Any]:
        ev = [evidence[i] for i in (refs.get("evidence_refs") or []) if 0 <= i < len(evidence)]
        fi = [findings[j] for j in (refs.get("finding_refs") or []) if 0 <= j < len(findings)]
        return {"evidence": ev, "findings": fi}

    inference = [
        {**sentence, **_resolved_refs(sentence)} for sentence in (card.get("inference") or [])
    ]
    confidence_refs = _resolved_refs(card.get("confidence_refs") or {})

    account_slug = card.get("account")
    theme_slug = card.get("theme")
    titles = note_summaries(conn, [s for s in (account_slug, theme_slug) if s])

    owner_crm_id = card.get("owner")
    owner_name = owner_crm_id
    with conn.cursor() as cur:
        cur.execute(
            "SELECT crm_id, canonical_name FROM registry_entities "
            "WHERE kind = 'internal_person' AND active ORDER BY canonical_name"
        )
        vendor_reps = cur.fetchall()
    owner_choices = [{"crm_id": r[0], "name": r[1]} for r in vendor_reps]
    for crm_id, name in vendor_reps:
        if crm_id == owner_crm_id:
            owner_name = name
            break

    row["action_context"] = {
        "account_name": (titles.get(account_slug) or {}).get("title") if account_slug else None,
        "theme_name": (titles.get(theme_slug) or {}).get("title") if theme_slug else None,
        "owner_name": owner_name,
        "evidence": evidence,
        "findings": findings,
        "inference": inference,
        "confidence_refs": confidence_refs,
        "trail": card.get("trail") or {},
        "owner_choices": owner_choices,
    }


def _enrich_action_done_row(conn: psycopg.Connection, row: dict[str, Any]) -> None:
    """The names and current state `_present_action_done` shows next to the
    answering claim: the account's title, the owner's name, the action's
    status and due date now, and the finding in plain words."""
    payload = row.get("payload") or {}
    account_slug = payload.get("account")
    titles = note_summaries(conn, [account_slug] if account_slug else [])
    owner_name = None
    with conn.cursor() as cur:
        if payload.get("owner_crm_id"):
            cur.execute(
                "SELECT canonical_name FROM registry_entities WHERE crm_id = %s",
                (payload["owner_crm_id"],),
            )
            name_row = cur.fetchone()
            owner_name = name_row[0] if name_row else None
        cur.execute("SELECT status, due FROM actions WHERE id = %s", (payload.get("action_id"),))
        action_row = cur.fetchone()
        cur.execute(
            "SELECT tool, arguments, result FROM researcher_findings WHERE id = %s",
            (payload.get("finding_id"),),
        )
        finding_row = cur.fetchone()
    finding_text = None
    if finding_row is not None:
        tool, arguments, result = finding_row
        finding_text = finding_plain_text(
            conn, {"tool": tool, "arguments": arguments, "result": result}
        )
    row["done_context"] = {
        "account_name": (titles.get(account_slug) or {}).get("title") if account_slug else None,
        "owner_name": owner_name,
        "status": action_row[0] if action_row else None,
        "due": action_row[1] if action_row else None,
        "finding_text": finding_text,
    }


def _enrich_rows_for_presentation(
    conn: psycopg.Connection, store: Store, rows: list[dict[str, Any]]
) -> None:
    """Add what `review_presenter.present_row` needs but, being pure, can't
    fetch: source briefs, context snippets, note titles, evidence backfill,
    action and done context. Missing data (no stored original, an unmatched
    locator, a deleted note) leaves that row's evidence empty, and so does a
    source whose original view cannot be built (`context_snippet` raises):
    one bad source row must not take down the whole queue.
    """
    source_slugs = {slug for row in rows if (slug := _snippet_source_slug(row))}
    sources = sources_brief(conn, list(source_slugs))

    note_slugs = {slug for row in rows if (slug := _note_slug_for(row))}
    titles = note_summaries(conn, list(note_slugs))

    for row in rows:
        source_slug = _snippet_source_slug(row)
        if source_slug and source_slug in sources:
            row["source"] = sources[source_slug]
            locator = _locator_for(row)
            if locator:
                try:
                    row["context_snippet"] = context_snippet(conn, store, source_slug, locator)
                except Exception:  # noqa: BLE001 - logged; the card renders without a snippet
                    _logger.exception("review: no context snippet for source %s", source_slug)
        note_slug = _note_slug_for(row)
        if note_slug and note_slug in titles:
            row["note_title"] = titles[note_slug]["title"]

        if _needs_evidence_backfill(row) and source_slug:
            row["evidence_backfill"] = backfill_fact_evidence(
                conn,
                source_slug=source_slug,
                mention=_mention_for(row),
                locator=_locator_for(row),
            )
            backfill = row["evidence_backfill"]
            if backfill and backfill.get("locator") and "context_snippet" not in row:
                row["context_snippet"] = context_snippet(
                    conn, store, source_slug, backfill["locator"]
                )

        if row["kind"] == "action_card":
            _enrich_action_card_row(conn, row)
        elif row["kind"] == "action_done":
            _enrich_action_done_row(conn, row)


@router.get("/review", dependencies=[Depends(track_page_view)])
def review_page(
    request: Request,
    kind: str = "",
    page: int = 1,
    session: Session = Depends(require_session),
    conn: psycopg.Connection = Depends(get_conn),
    store: Store = Depends(get_store_dep),
) -> object:
    """One page of the open queue. Order, per-category counts and the
    `kind` filter (a category label, `?kind=...&page=...`) are computed over
    the whole queue from `review_queue_open_index`'s small columns; only the
    current page's rows are loaded in full, enriched and presented.
    """
    templates = request.app.state.templates
    notice_key = request.query_params.get("notice", "")
    notice = _NOTICES.get(notice_key)
    # Display text only: Jinja escapes both, and `conflict_row` is only
    # compared with each card's row id, never used in a query.
    conflict_row_id = (
        request.query_params.get("conflict_row", "") if notice_key == "alias_conflict" else ""
    )
    conflict_name = (
        request.query_params.get("conflict_name", "") if notice_key == "alias_conflict" else ""
    )
    kind = bounded_str(kind, name="kind", max_chars=MAX_REVIEW_KIND_CHARS).strip()
    page = bounded_int(page, name="page", minimum=1, maximum=MAX_REVIEW_PAGE)

    index_rows = review_queue_open_index(conn, status="open")
    labeled = [(row, category_label_for(row["kind"], row["reason"])) for row in index_rows]

    # First-seen order over the oldest-first queue, so the strip is stable.
    counts_by_label: dict[str, int] = {}
    for _row, label in labeled:
        counts_by_label[label] = counts_by_label.get(label, 0) + 1
    counts = list(counts_by_label.items())

    filtered = [row for row, label in labeled if not kind or label == kind]
    total = len(filtered)
    total_pages = max(1, -(-total // REVIEW_PAGE_SIZE))  # ceiling division
    start = (page - 1) * REVIEW_PAGE_SIZE
    page_ids = [row["id"] for row in filtered[start : start + REVIEW_PAGE_SIZE]]

    rows = review_queue_rows_by_ids(conn, page_ids)
    _enrich_theme_resolution_rows(conn, rows)
    _enrich_rows_for_presentation(conn, store, rows)

    candidate_ids = {
        crm_id
        for row in rows
        if row["kind"] == "ambiguous_entity"
        for crm_id in (row["payload"] or {}).get("candidates") or []
    }
    entities_by_crm_id = registry_entities_by_crm_ids(conn, list(candidate_ids))
    cards = present_rows(rows, entities_by_crm_id=entities_by_crm_id)

    return templates.TemplateResponse(
        request,
        "review.html",
        {
            "cards": cards,
            "counts": counts,
            "total_open": len(index_rows),
            "kind": kind,
            "page": page,
            "total_pages": total_pages,
            "total": total,
            "notice": notice,
            "conflict_row_id": conflict_row_id,
            "conflict_name": conflict_name,
        },
    )


@router.get("/review/registry")
def review_registry_picker(
    q: str = "",
    session: Session = Depends(require_session_api),
    conn: psycopg.Connection = Depends(get_conn),
) -> JSONResponse:
    """The alias picker's lookup: up to `REGISTRY_PICKER_LIMIT` registry
    entities matching `q`, as JSON `{crm_id, canonical_name, kind,
    description}`.
    """
    q = bounded_str(q, name="q", max_chars=MAX_REGISTRY_QUERY_CHARS)
    matches = registry_entities_search(conn, q, limit=REGISTRY_PICKER_LIMIT)
    return JSONResponse(content=matches)


def _get_row(conn: psycopg.Connection, row_id: int) -> dict[str, Any] | None:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, kind, payload, source_slug, status FROM review_queue WHERE id = %s",
            (row_id,),
        )
        row = cur.fetchone()
    if row is None:
        return None
    return {
        "id": row[0],
        "kind": row[1],
        "payload": row[2],
        "source_slug": row[3],
        "status": row[4],
    }


def _lock_if_open(conn: psycopg.Connection, row_id: int) -> bool:
    """Inside Confirm's/Dismiss's transaction: lock the row and re-read its
    status. The earlier unlocked check can race (a double click on two pool
    connections); the second request waits here, then sees it handled."""
    with conn.cursor() as cur:
        cur.execute("SELECT status FROM review_queue WHERE id = %s FOR UPDATE", (row_id,))
        row = cur.fetchone()
    return row is not None and row[0] == "open"


def _resolve_row(conn: psycopg.Connection, row_id: int, status: str, resolved_by: str) -> None:
    """Mark the row handled. Never commits: callers run it inside their
    `with conn.transaction():` (psycopg forbids `commit()` there), so the
    row update and its side effects land together. Outside a transaction
    (`review_add_alias`) the web app's autocommit connection commits the
    `UPDATE` by itself."""
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE review_queue SET status = %s, resolved_at = now(), resolved_by = %s "
            "WHERE id = %s",
            (status, resolved_by, row_id),
        )


def _fetch_source_row(conn: psycopg.Connection, slug: str) -> SourceRow | None:
    """The `sources` row `slug` names, as a `SourceRow`, read with
    `portolan.ingest.intake`'s column list and row mapper."""
    with conn.cursor() as cur:
        cur.execute(f"SELECT {', '.join(SOURCE_COLUMNS)} FROM sources WHERE slug = %s", (slug,))
        row = cur.fetchone()
    return row_to_source_row(row) if row is not None else None


def _keep_quote_unsupported_fact(
    conn: psycopg.Connection, llm: LLMClient, embedder: Embedder, row: dict[str, Any]
) -> str:
    """The "Keep as claim" choice: run the one fact the quote gate held back
    (`portolan.pipeline._gate_quotes`) through `merge_extraction`, the
    pipeline's own merge entry point, as a single-fact `ExtractionResult`.

    Returns the `_NOTICES` key: `"malformed"` when the payload or source no
    longer supports rebuilding the fact; `"kept_as_claim"` when a claim was
    inserted; `"kept_matched_existing"` when the fact matched a claim already
    on file (a `same` outcome); `"kept_unresolved"` otherwise (queued as
    unknown)."""
    payload = row.get("payload") or {}
    fact_payload = payload.get("fact") or {}
    source_slug = row.get("source_slug")
    if not fact_payload or not source_slug:
        return "malformed"
    try:
        fact = Fact.model_validate(fact_payload)
    except ValidationError:
        return "malformed"
    source_row = _fetch_source_row(conn, source_slug)
    if source_row is None:
        return "malformed"
    registry = Registry.load(conn)
    result = ExtractionResult(
        source_id=source_row.source_id or source_row.slug,
        summary="",
        facts=[fact],
        coverage=Coverage(segments_read=1, segments_total=1),
    )
    report = merge_extraction(conn, llm, embedder, registry, result, source_row)
    if report.claims_inserted > 0:
        return "kept_as_claim"
    if report.outcome_counts.get("same"):
        return "kept_matched_existing"
    return "kept_unresolved"


def _card_verdict(row: dict[str, Any]) -> str | None:
    return ((row.get("payload") or {}).get("card") or {}).get("verdict")


def _record_confirm_override(
    conn: psycopg.Connection,
    *,
    row_id: int,
    action_id: int,
    original_owner: str | None,
    original_due: str | None,
    final_owner: str | None,
    final_due: str | None,
    editor: str,
) -> None:
    """An owner or due override at confirm time: write an `edit_records` row
    on the new action's note (`note_slug` is `NOT NULL`, so only after a
    successful confirm) and store the confirmed owner/due in the row's
    payload."""
    with conn.cursor() as cur:
        cur.execute("SELECT note_slug FROM actions WHERE id = %s", (action_id,))
        note_row = cur.fetchone()
    note_slug = note_row[0] if note_row else None
    if note_slug is None:
        return
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO edit_records (note_slug, editor, before_text, after_text, reason)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (
                note_slug,
                editor,
                f"owner={original_owner}, due={original_due}",
                f"owner={final_owner}, due={final_due}",
                "owner/due overridden at confirm time",
            ),
        )
        cur.execute(
            """
            UPDATE review_queue
            SET payload = jsonb_set(
                jsonb_set(payload, '{card,owner}', to_jsonb(%s::text)),
                '{card,due}', to_jsonb(%s::text)
            )
            WHERE id = %s
            """,
            (final_owner, final_due, row_id),
        )


_DECISION_ROUTES = {
    "resolve": "/review/{row_id}/resolve",
    "dismiss": "/review/{row_id}/dismiss",
    "add_alias": "/review/{row_id}/add-alias",
}


def _record_decision(
    conn: psycopg.Connection,
    editor: str,
    row_id: int,
    decision: str,
    row_kind: str,
    notice: str | None,
) -> None:
    record_activity(
        conn,
        kind="review_decision",
        user_name=editor,
        route=_DECISION_ROUTES[decision],
        target=str(row_id),
        detail={"decision": decision, "row_kind": row_kind, "notice": notice},
    )


@router.post("/review/{row_id}/resolve")
def review_resolve(
    row_id: int,
    session: Session = Depends(require_editor),
    conn: psycopg.Connection = Depends(get_conn),
    llm: LLMClient = Depends(get_llm_dep),
    embedder: Embedder = Depends(get_embedder_dep),
    owner_crm_id: str | None = Form(None),
    due: str | None = Form(None),
) -> object:
    """Confirm a row. Per kind, inside one transaction with the row update:
    `theme_resolution` stamps the resolution confirmed (`superseded` notice
    when there was nothing left to confirm, `malformed` when the payload
    lacks its ids; the row is resolved either way); `quote_unsupported`
    re-merges the held-back fact; `owner_proposal` sets the top candidate
    as owner; `action_done` closes the action; `action_card` creates the
    action, or records a no-action verdict. Any other kind only marks the
    row resolved. A failed `action_card` confirm, and `action_done` with
    changed evidence, leave the row open."""
    row = _get_row(conn, row_id)
    if row is None:
        raise HTTPException(status_code=404, detail="no such review row")
    # A row that is no longer open is never acted on twice (double click, two
    # tabs, Confirm after Dismiss); `_lock_if_open` closes the race.
    if row["status"] != "open":
        return RedirectResponse(url="/review?notice=already_handled", status_code=303)
    notice: str | None = None
    confirm: tuple[str, int] | None = None
    with conn.transaction():
        if not _lock_if_open(conn, row_id):
            return RedirectResponse(url="/review?notice=already_handled", status_code=303)
        should_resolve = True
        if row["kind"] == "theme_resolution":
            payload = row["payload"] or {}
            theme_slug = payload.get("theme_slug")
            resolution_claim_id = payload.get("resolution_claim_id")
            if theme_slug is None or resolution_claim_id is None:
                notice = "malformed"
            elif not confirm_theme_resolution(
                conn,
                theme_slug=theme_slug,
                resolution_claim_id=resolution_claim_id,
                changed_by=session.editor,
            ):
                notice = "superseded"
            else:
                notice = "confirmed"
        elif row["kind"] == "quote_unsupported":
            notice = _keep_quote_unsupported_fact(conn, llm, embedder, row)
        elif row["kind"] == "owner_proposal":
            payload = row["payload"] or {}
            account_crm_id = payload.get("account_crm_id")
            candidates = payload.get("candidates") or []
            top_crm_id = candidates[0].get("crm_id") if candidates else None
            if not account_crm_id or not top_crm_id:
                notice = "malformed"
            elif confirm_owner_proposal(
                conn, account_crm_id=account_crm_id, owner_crm_id=top_crm_id, commit=False
            ):
                notice = "owner_confirmed"
            else:
                notice = "owner_confirm_failed"
        elif row["kind"] == "action_done":
            done = confirm_action_done(
                conn, row_id=row_id, payload=row["payload"] or {}, editor=session.editor
            )
            notice = {
                "done": "action_done_confirmed",
                "already_closed": "action_done_already_closed",
                "evidence_changed": "action_done_evidence_changed",
            }.get(done, "malformed")
            # Changed evidence leaves the row open for a person to dismiss;
            # a closed or malformed action has nothing left to decide.
            should_resolve = done != "evidence_changed"
            if done == "done":
                confirm = ("done_confirmed", row_id)
        elif row["kind"] == "action_card" and _card_verdict(row) in ("no_action", "trigger_stale"):
            # The card promises "Confirming records that conclusion": the row
            # resolves and nothing is created.
            notice = "action_card_concluded"
        elif row["kind"] == "action_card":
            payload = row["payload"] or {}
            card = payload.get("card") or {}
            original_owner = card.get("owner")
            original_due = card.get("due")
            due_override: date | None = None
            due_unparseable = False
            if due and due.strip():
                try:
                    due_override = date.fromisoformat(due.strip())
                except ValueError:
                    due_unparseable = True
            if due_unparseable:
                outcome = ConfirmOutcome(action_id=None, reason="due_invalid")
            else:
                outcome = confirm_action_card_with_reason(
                    conn,
                    card=card,
                    owner_crm_id=owner_crm_id or None,
                    due=due_override,
                    editor=session.editor,
                    commit=False,
                    review_row_id=row_id,
                )
            action_id = outcome.action_id
            if action_id is not None:
                notice = "action_confirmed"
                confirm = ("card_confirmed", action_id)
                # The confirmed action's id on the row, so the report can find
                # the action directly rather than by matching its text.
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE review_queue SET payload = "
                        "jsonb_set(payload, '{action_id}', to_jsonb(%s::bigint)) WHERE id = %s",
                        (action_id, row_id),
                    )
                final_owner = owner_crm_id or original_owner
                final_due = due_override.isoformat() if due_override else original_due
                if (owner_crm_id and owner_crm_id != original_owner) or (
                    due_override is not None and due_override.isoformat() != original_due
                ):
                    _record_confirm_override(
                        conn,
                        row_id=row_id,
                        action_id=action_id,
                        original_owner=original_owner,
                        original_due=original_due,
                        final_owner=final_owner,
                        final_due=final_due,
                        editor=session.editor,
                    )
            else:
                notice = {
                    "due_stale": "action_due_stale",
                    "due_invalid": "action_due_invalid",
                }.get(outcome.reason or "", "action_confirm_failed")
                # Nothing was created, so the row stays open to edit and retry.
                should_resolve = False
        if should_resolve:
            _resolve_row(conn, row_id, "resolved", session.editor)
    _record_decision(conn, session.editor, row_id, "resolve", row["kind"], notice)
    if confirm is not None:
        record_activity(
            conn,
            kind="action_confirm",
            user_name=session.editor,
            route="/review/{row_id}/resolve",
            target=str(confirm[1]),
            detail={"how": confirm[0], "review_row_id": row_id},
        )
    url = f"/review?notice={notice}" if notice else "/review"
    return RedirectResponse(url=url, status_code=303)


@router.post("/review/{row_id}/dismiss")
def review_dismiss(
    row_id: int,
    session: Session = Depends(require_editor),
    conn: psycopg.Connection = Depends(get_conn),
    reason: str | None = Form(None),
) -> object:
    """Dismiss a row, inside one transaction with the row update. For
    `theme_resolution` this rejects the resolution (`reject_theme_resolution`
    un-resolves the theme); a no-op reject gets the `superseded` notice and
    a payload missing its ids gets `malformed`, and the row is dismissed
    either way. Other kinds only mark the row dismissed. An optional
    `reason` is stored in the payload."""
    row = _get_row(conn, row_id)
    if row is None:
        raise HTTPException(status_code=404, detail="no such review row")
    if row["status"] != "open":
        return RedirectResponse(url="/review?notice=already_handled", status_code=303)
    notice: str | None = None
    with conn.transaction():
        if not _lock_if_open(conn, row_id):
            return RedirectResponse(url="/review?notice=already_handled", status_code=303)
        if row["kind"] == "theme_resolution":
            payload = row["payload"] or {}
            theme_slug = payload.get("theme_slug")
            resolution_claim_id = payload.get("resolution_claim_id")
            if theme_slug is None or resolution_claim_id is None:
                notice = "malformed"
            elif not reject_theme_resolution(
                conn,
                theme_slug=theme_slug,
                resolution_claim_id=resolution_claim_id,
                changed_by=session.editor,
            ):
                notice = "superseded"
            else:
                notice = "rejected"
        # The reason goes in the payload: `review_queue` has no reason
        # column, and a dismissed card has no note for an `edit_records` row.
        if reason and reason.strip():
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE review_queue SET payload = jsonb_set("
                    "payload, '{dismiss_reason}', to_jsonb(%s::text)) WHERE id = %s",
                    (reason.strip(), row_id),
                )
        _resolve_row(conn, row_id, "dismissed", session.editor)
    _record_decision(conn, session.editor, row_id, "dismiss", row["kind"], notice)
    url = f"/review?notice={notice}" if notice else "/review"
    return RedirectResponse(url=url, status_code=303)


@router.post("/review/{row_id}/add-alias")
def review_add_alias(
    row_id: int,
    entity_id: str = Form(...),
    session: Session = Depends(require_editor),
    conn: psycopg.Connection = Depends(get_conn),
) -> object:
    """Map the row's mention to `entity_id` (spec sec 3: "a human maps a
    name once; it sticks"). Refused, with no change, when the row is no
    longer open or the mention already maps to a different entity (the
    `registry_aliases` key does not prevent that)."""
    row = _get_row(conn, row_id)
    if row is None:
        raise HTTPException(status_code=404, detail="no such review row")
    if row["status"] != "open":
        return RedirectResponse(url="/review?notice=already_handled", status_code=303)
    if row["kind"] not in ("unknown_entity", "ambiguous_entity"):
        raise HTTPException(
            status_code=400, detail="only unknown_entity/ambiguous_entity rows can be aliased"
        )

    mention = (row["payload"] or {}).get("mention")
    if not mention:
        raise HTTPException(status_code=400, detail="review row has no mention to alias")

    conflict = registry_alias_conflict(conn, mention)
    if conflict is not None and conflict["crm_id"] != entity_id:
        url = (
            "/review?notice=alias_conflict"
            f"&conflict_row={row_id}"
            f"&conflict_name={quote(conflict['canonical_name'])}"
        )
        return RedirectResponse(url=url, status_code=303)

    # add_alias raises ValueError for an entity_id that isn't a registry
    # entity; a hand-crafted POST gets a 400, not a 500.
    try:
        add_alias(conn, entity_id, mention, source="human")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    _resolve_row(conn, row_id, "resolved", session.editor)
    _record_decision(conn, session.editor, row_id, "add_alias", row["kind"], None)
    return RedirectResponse(url="/review", status_code=303)


__all__ = ["router"]
