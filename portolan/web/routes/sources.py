"""`/sources/{slug}`: a source note's fields, extraction status, the
notes extracted from it, and an "open original" button
(provenance-chain.md link 3: source note -> original).
"""

from __future__ import annotations

import psycopg
from fastapi import APIRouter, Depends, HTTPException, Request

from portolan.config import Settings
from portolan.mcp_server.tools import tool_open_source
from portolan.store import Store
from portolan.web.auth import Session, require_session
from portolan.web.deps import get_conn, get_settings_dep, get_store_dep
from portolan.web.queries import claims_citing_source, note_summaries
from portolan.web.tracking import track_page_view

router = APIRouter()


@router.get("/sources/{slug}", dependencies=[Depends(track_page_view)])
def source_detail(
    request: Request,
    slug: str,
    session: Session = Depends(require_session),
    conn: psycopg.Connection = Depends(get_conn),
    store: Store = Depends(get_store_dep),
    settings: Settings = Depends(get_settings_dep),
) -> object:
    templates = request.app.state.templates
    # Reuses the MCP tool's query, so each page view also logs a
    # `usage_events` row under surface "mcp".
    source = tool_open_source(conn, store, tenant=settings.tenant, slug=slug)
    if source is None:
        raise HTTPException(status_code=404, detail=f"no such source: {slug}")
    summaries = note_summaries(conn, source["notes_extracted"])
    citing_claims = claims_citing_source(conn, slug)
    return templates.TemplateResponse(
        request,
        "source.html",
        {"source": source, "note_summaries": summaries, "citing_claims": citing_claims},
    )


__all__ = ["router"]
