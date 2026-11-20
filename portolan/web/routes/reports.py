"""`/reports` (list), `/reports/{slug}` (detail) and `/reports/{slug}.md`.
A report is a note of type `report` (portolan-spec-v2.md sec 7); its body
renders through `render_note_body`, like `/notes/{slug}` and the graph
note panel.
"""

from __future__ import annotations

import psycopg
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response

from portolan.notes.persist import read_note
from portolan.report.researcher import refresh_action_card_lines
from portolan.report.util import week_bounds
from portolan.web.auth import Session, require_session
from portolan.web.deps import get_conn
from portolan.web.markdown import render_download_markdown, render_note_body
from portolan.web.queries import note_type, report_notes, report_week
from portolan.web.tracking import track_page_view

router = APIRouter()


def _live_body(conn: psycopg.Connection, slug: str, body: str) -> str:
    """The stored report body with each suggested action card's state read
    from the review queue now (`refresh_action_card_lines`): a card
    confirmed or dismissed after the report was written shows as such."""
    week = report_week(conn, slug)
    return refresh_action_card_lines(conn, body, week=week) if week else body


@router.get("/reports", dependencies=[Depends(track_page_view)])
def reports_list(
    request: Request,
    session: Session = Depends(require_session),
    conn: psycopg.Connection = Depends(get_conn),
) -> object:
    templates = request.app.state.templates
    reports = report_notes(conn)
    return templates.TemplateResponse(request, "reports_list.html", {"reports": reports})


@router.get("/reports/{slug}.md")
def report_download(
    slug: str,
    request: Request,
    session: Session = Depends(require_session),
    conn: psycopg.Connection = Depends(get_conn),
) -> Response:
    """The report as a markdown download, with front matter and absolute
    links (`render_download_markdown`). Registered before
    `/reports/{slug}`, which would otherwise match the `.md` path."""
    if note_type(conn, slug) != "report":
        raise HTTPException(status_code=404, detail=f"no such report: {slug}")
    try:
        note, claims = read_note(conn, slug)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=f"no such report: {slug}") from exc

    note = note.model_copy(update={"synthesis": _live_body(conn, slug, note.synthesis or "")})
    md = render_download_markdown(conn, note, claims, base_url=str(request.base_url))
    return Response(
        content=md,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{slug}.md"'},
    )


@router.get("/reports/{slug}", dependencies=[Depends(track_page_view)])
def report_detail(
    request: Request,
    slug: str,
    session: Session = Depends(require_session),
    conn: psycopg.Connection = Depends(get_conn),
) -> object:
    """A report in its own template, `report.html`: the rendered body and the
    week it covers, without the Related, Backlinks or Version history a
    note page shows.
    """
    if note_type(conn, slug) != "report":
        raise HTTPException(status_code=404, detail=f"no such report: {slug}")
    try:
        note, _claims = read_note(conn, slug)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=f"no such report: {slug}") from exc

    body = _live_body(conn, slug, note.synthesis or "")
    body_html = render_note_body(conn, body)

    week = report_week(conn, slug)
    covers = week_bounds(week) if week else None

    templates = request.app.state.templates
    return templates.TemplateResponse(
        request, "report.html", {"note": note, "covers": covers, "body_html": body_html}
    )


__all__ = ["router"]
