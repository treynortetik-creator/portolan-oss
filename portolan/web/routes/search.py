"""`/search?q=`: a 302 redirect to `/atlas?q=`, which now hosts search,
kept so old bookmarks still land somewhere.
"""

from __future__ import annotations

from urllib.parse import urlencode

from fastapi import APIRouter
from fastapi.responses import RedirectResponse

from portolan.web.bounds import bounded_str

router = APIRouter()

# Shared with the MCP search tool. Atlas has its own `MAX_ATLAS_QUERY_CHARS`.
from portolan.search.query import MAX_SEARCH_QUERY_CHARS  # noqa: E402


@router.get("/search")
def search_redirect(q: str = "") -> RedirectResponse:
    q = bounded_str(q, name="q", max_chars=MAX_SEARCH_QUERY_CHARS)
    query = q.strip()
    target = f"/atlas?{urlencode({'q': query})}" if query else "/atlas"
    return RedirectResponse(url=target, status_code=302)


__all__ = ["router"]
