"""Pilot usage: `/admin/usage` (the dashboard), thumbs feedback on chat
answers, and the email open pixel and click redirect
(`portolan.activity`, migration 0037; 01-spec/red-team-2026-09-24.md sec 6).

`/admin/usage` needs an admin (`portolan.web.auth.require_admin`): under
web_auth="workos" an email in `Settings.admin_emails`; under "none" there is
no verified identity, so any signed-in reader, as before. The email routes
need no login; the signed token is the credential
(`portolan.activity.email_links`).
"""

from __future__ import annotations

import psycopg
from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import RedirectResponse, Response

from portolan.activity import record_activity
from portolan.activity.dashboard import pilot_dashboard
from portolan.activity.email_links import read_email_token
from portolan.activity.feedback import save_feedback
from portolan.web.auth import Session, require_admin, require_session_api
from portolan.web.deps import get_conn
from portolan.web.tracking import track_page_view

router = APIRouter()

# The smallest transparent GIF: 1x1, one colour.
_PIXEL_GIF = (
    b"GIF89a\x01\x00\x01\x00\x80\x00\x00\x00\x00\x00\xff\xff\xff!\xf9\x04\x01\x00\x00\x00\x00"
    b",\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;"
)


@router.get("/admin/usage", dependencies=[Depends(track_page_view)])
def usage_dashboard(
    request: Request,
    session: Session = Depends(require_admin),
    conn: psycopg.Connection = Depends(get_conn),
) -> object:
    templates = request.app.state.templates
    return templates.TemplateResponse(request, "admin_usage.html", {"dash": pilot_dashboard(conn)})


@router.post("/messages/{message_id}/feedback")
def answer_feedback(
    message_id: int,
    rating: str = Form(...),
    reason: str = Form(""),
    session: Session = Depends(require_session_api),
    conn: psycopg.Connection = Depends(get_conn),
) -> object:
    if rating not in ("up", "down"):
        raise HTTPException(status_code=400, detail="rating must be up or down")
    saved = save_feedback(
        conn, message_id=message_id, user_name=session.editor, rating=rating, reason=reason
    )
    if saved is None:
        raise HTTPException(status_code=404, detail="no such answer")
    return RedirectResponse(url=f"/c/{saved.conversation_id}", status_code=303)


def _secret(request: Request) -> str:
    return request.app.state.settings.session_secret.get_secret_value()


@router.get("/t/o/{token}.gif", include_in_schema=False)
def email_open_pixel(
    request: Request, token: str, conn: psycopg.Connection = Depends(get_conn)
) -> Response:
    """Always the pixel, valid token or not, so a probe learns nothing."""
    data = read_email_token(token, secret=_secret(request), kind="open")
    if data is not None:
        record_activity(
            conn,
            kind="email_open",
            user_name=data["r"],
            route="/t/o/{token}.gif",
            target=data["e"],
            detail={"email_id": data["e"]},
        )
    return Response(content=_PIXEL_GIF, media_type="image/gif")


@router.get("/t/c/{token}", include_in_schema=False)
def email_click(
    request: Request, token: str, conn: psycopg.Connection = Depends(get_conn)
) -> RedirectResponse:
    """Redirects only to the URL inside a valid signature; anything else
    lands on `/`."""
    data = read_email_token(token, secret=_secret(request), kind="click")
    if data is None:
        return RedirectResponse(url="/", status_code=302)
    record_activity(
        conn,
        kind="email_click",
        user_name=data["r"],
        route="/t/c/{token}",
        target=data["e"],
        detail={"email_id": data["e"], "url": data["u"]},
    )
    return RedirectResponse(url=data["u"], status_code=302)


__all__ = ["router"]
