"""`/how-it-works`: the self-contained explainer at
`01-spec/how-portolan-works.html` (several MB, with narration audio),
served without a login.

Read from the repo path: `.dockerignore` keeps `01-spec/` in the image, so
there is no copy to drift. The file is read once at import; responses
carry a content `ETag` and `Cache-Control: public, max-age=3600`, and a
matching `If-None-Match` gets a bodiless 304, so the one public route
cannot be used to pull megabytes per request.
"""

from __future__ import annotations

import hashlib
from email.utils import formatdate

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

from portolan.web.templating import PROJECT_ROOT

router = APIRouter()

HOW_IT_WORKS_PATH = PROJECT_ROOT / "01-spec" / "how-portolan-works.html"

# `None` when the file is absent from this checkout or image; the route 404s.
try:
    _CONTENT: bytes | None = HOW_IT_WORKS_PATH.read_bytes()
except OSError:
    _CONTENT = None

_ETAG = f'"{hashlib.sha256(_CONTENT).hexdigest()}"' if _CONTENT is not None else None
_LAST_MODIFIED = (
    formatdate(HOW_IT_WORKS_PATH.stat().st_mtime, usegmt=True) if _CONTENT is not None else None
)
# The file only changes on a redeploy.
_CACHE_CONTROL = "public, max-age=3600"


def _not_modified(request: Request) -> bool:
    """True when `If-None-Match` names this file's ETag (weak or strong) or
    `*`; the header may list several tags (RFC 9110)."""
    if _ETAG is None:
        return False
    if_none_match = request.headers.get("if-none-match")
    if not if_none_match:
        return False
    candidates = {tag.strip().removeprefix("W/") for tag in if_none_match.split(",")}
    return "*" in candidates or _ETAG in candidates


@router.get("/how-it-works")
def how_it_works(request: Request) -> Response:
    if _CONTENT is None:
        raise HTTPException(
            status_code=404, detail="the explainer file is not present in this build"
        )

    headers = {
        "Cache-Control": _CACHE_CONTROL,
        "ETag": _ETAG,
        "Last-Modified": _LAST_MODIFIED,
    }

    if _not_modified(request):
        return Response(status_code=304, headers=headers)

    return Response(content=_CONTENT, media_type="text/html", headers=headers)


__all__ = ["router"]
