"""`/notes`, `/notes/{slug}`, `/notes/{slug}.md` and `/notes/{slug}/edit`.

Human edits are attributed to the session's editor (portolan-spec-v2.md
sec 4: an edit record plus a `^[edit:name date]` marker). This route reads
the note or claim only for `before_text`; `record_edit` is the single
writer: it applies `after_text` to the claim (with `claim_id`) or the
synthesis, appends the marker, and writes the note once (edit tests in
`tests/test_web.py`).
"""

from __future__ import annotations

import psycopg
from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import RedirectResponse, Response

from portolan.consolidate.edits import record_edit
from portolan.context import COMPANY_CONTEXT_BODY_CHAR_CAP, COMPANY_CONTEXT_SLUG
from portolan.notes.models import NoteType
from portolan.notes.persist import note_exists, read_note
from portolan.web.auth import (
    Session,
    require_editor,
    require_editor_page,
    require_session,
)
from portolan.web.bounds import bounded_int, bounded_str
from portolan.web.deps import get_conn
from portolan.web.markdown import render_download_markdown
from portolan.web.note_view import (
    action_done_for_note,
    action_for_note,
    action_why_for_note,
    render_note_detail,
)
from portolan.web.queries import NOTES_INDEX_PAGE_SIZE, list_notes
from portolan.web.tracking import track_page_view

router = APIRouter()

MAX_NOTES_INDEX_PAGE = 100_000
# Filters a title/slug, not a document.
MAX_NOTES_QUERY_CHARS = 200
# Generous for prose, but the text is written into the claim or note body.
MAX_EDIT_TEXT_CHARS = 20_000


@router.get("/notes", dependencies=[Depends(track_page_view)])
def notes_index(
    request: Request,
    type: str | None = None,
    q: str = "",
    page: int = 1,
    session: Session = Depends(require_session),
    conn: psycopg.Connection = Depends(get_conn),
) -> object:
    templates = request.app.state.templates
    page = bounded_int(page, name="page", minimum=1, maximum=MAX_NOTES_INDEX_PAGE)
    q = bounded_str(q, name="q", max_chars=MAX_NOTES_QUERY_CHARS)
    type_filter = type if type in {t.value for t in NoteType} else None
    rows, total = list_notes(conn, type_filter=type_filter, q=q.strip() or None, page=page)
    total_pages = max(1, -(-total // NOTES_INDEX_PAGE_SIZE))  # ceiling division

    # Shows how current the company-context note is; `None` (line hidden)
    # before it has been seeded.
    company_context_updated = None
    if note_exists(conn, COMPANY_CONTEXT_SLUG):
        company_context_note, _claims = read_note(conn, COMPANY_CONTEXT_SLUG)
        company_context_updated = company_context_note.updated

    return templates.TemplateResponse(
        request,
        "notes_index.html",
        {
            "rows": rows,
            "total": total,
            "page": page,
            "total_pages": total_pages,
            "type": type_filter or "",
            "q": q,
            "note_types": [t.value for t in NoteType],
            "company_context_updated": company_context_updated,
        },
    )


@router.get("/notes/{slug}.md")
def note_download(
    slug: str,
    request: Request,
    session: Session = Depends(require_session),
    conn: psycopg.Connection = Depends(get_conn),
) -> Response:
    """The note as a markdown download: synthesis plus its claims (a signal
    note's content is its claims). Registered before `/notes/{slug}`,
    which would otherwise match the `.md` path."""
    try:
        note, claims = read_note(conn, slug)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=f"no such note: {slug}") from exc

    why = action_why_for_note(conn, note)
    md = render_download_markdown(
        conn,
        note,
        claims,
        base_url=str(request.base_url),
        why=why,
        action=action_for_note(conn, slug) if why is not None else None,
        done=action_done_for_note(conn, note),
    )
    return Response(
        content=md,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{slug}.md"'},
    )


@router.get("/notes/{slug}", dependencies=[Depends(track_page_view)])
def note_detail(
    request: Request,
    slug: str,
    session: Session = Depends(require_session),
    conn: psycopg.Connection = Depends(get_conn),
) -> object:
    return render_note_detail(request, conn, slug, base_path="/notes")


@router.get("/notes/{slug}/edit")
def note_edit_form(
    request: Request,
    slug: str,
    claim_id: int | None = None,
    session: Session = Depends(require_editor_page),
    conn: psycopg.Connection = Depends(get_conn),
) -> object:
    templates = request.app.state.templates
    try:
        note, claims = read_note(conn, slug)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=f"no such note: {slug}") from exc

    target_claim = None
    if claim_id is not None:
        target_claim = next((c for c in claims if c.id == claim_id), None)
        if target_claim is None:
            raise HTTPException(status_code=404, detail="no such claim on this note")

    return templates.TemplateResponse(
        request, "edit_note.html", {"note": note, "claim": target_claim}
    )


@router.post("/notes/{slug}/edit")
def note_edit_submit(
    request: Request,
    slug: str,
    after_text: str = Form(...),
    reason: str = Form(""),
    claim_id: str = Form(""),
    session: Session = Depends(require_editor),
    conn: psycopg.Connection = Depends(get_conn),
) -> object:
    try:
        note, claims = read_note(conn, slug)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=f"no such note: {slug}") from exc

    after_text = bounded_str(after_text.strip(), name="after_text", max_chars=MAX_EDIT_TEXT_CHARS)
    editor = session.editor
    edit_reason = reason.strip() or "human edit"

    # A non-numeric `claim_id` is a 400, not a 500.
    try:
        target_id = int(claim_id) if claim_id.strip() else None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="claim_id must be a number") from exc
    if target_id is not None:
        target_claim = next((c for c in claims if c.id == target_id), None)
        if target_claim is None:
            raise HTTPException(status_code=404, detail="no such claim on this note")
        before_text = target_claim.claim_text
    else:
        before_text = note.synthesis or ""

    # The company-context body is the note's synthesis, capped at
    # `COMPANY_CONTEXT_BODY_CHAR_CAP`. This check blocks the save; the lint
    # (`check_bad_header`) only reports an over-length body afterwards.
    is_context_body_edit = note.type == NoteType.CONTEXT and target_id is None
    if is_context_body_edit and len(after_text) > COMPANY_CONTEXT_BODY_CHAR_CAP:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Company context is {len(after_text)} characters, over the "
                f"{COMPANY_CONTEXT_BODY_CHAR_CAP}-character limit. Shorten it and save again."
            ),
        )

    record_edit(
        conn,
        note_slug=slug,
        editor=editor,
        before=before_text,
        after=after_text,
        reason=edit_reason,
        claim_id=target_id,
    )

    return RedirectResponse(url=f"/notes/{slug}", status_code=303)


__all__ = ["router"]
