"""Page-view tracking for the pilot activity log (`portolan.activity`).

`track_page_view` is a route dependency for full-page GET routes
(`dependencies=[Depends(track_page_view)]`). It records the route template
(`/notes/{slug}`, never the query string) and the first path parameter as
the target. An HTMX fragment request is not a page view.
"""

from __future__ import annotations

import psycopg
from fastapi import Depends, Request

from portolan.activity import record_activity
from portolan.web.auth import Session, require_session
from portolan.web.deps import get_conn

__all__ = ["track_page_view"]


def track_page_view(
    request: Request,
    session: Session = Depends(require_session),  # noqa: B008 - FastAPI dependency
    conn: psycopg.Connection = Depends(get_conn),  # noqa: B008 - FastAPI dependency
) -> None:
    if request.headers.get("hx-request", "").lower() == "true":
        return
    route = request.scope.get("route")
    path = getattr(route, "path", None) or request.url.path
    params = request.path_params
    target = str(next(iter(params.values()))) if params else None
    record_activity(conn, kind="page_view", user_name=session.editor, route=path, target=target)
