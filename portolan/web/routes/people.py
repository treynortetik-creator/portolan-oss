"""`/admin/people`: everyone who has signed in through WorkOS, and the
admin-only editor/viewer switch (`portolan.web.people`, migration 0038).
Exists only under web_auth="workos"; under "none" there are no verified people.
"""

from __future__ import annotations

import psycopg
from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import RedirectResponse

from portolan.web.auth import Session, require_admin, require_admin_api
from portolan.web.deps import get_conn
from portolan.web.people import ROLES, list_people, parse_email_list, set_role
from portolan.web.tracking import track_page_view

router = APIRouter()


def _require_workos(request: Request) -> None:
    """Called after the session check, so a signed-out caller still gets 302/401."""
    if request.app.state.settings.web_auth != "workos":
        raise HTTPException(status_code=404, detail="Not Found")


@router.get("/admin/people", dependencies=[Depends(track_page_view)])
def people_page(
    request: Request,
    session: Session = Depends(require_admin),
    conn: psycopg.Connection = Depends(get_conn),
) -> object:
    _require_workos(request)
    templates = request.app.state.templates
    admins = parse_email_list(request.app.state.settings.admin_emails)
    return templates.TemplateResponse(
        request,
        "admin_people.html",
        {"people": list_people(conn), "admins": admins, "roles": ROLES},
    )


@router.post("/admin/people/role")
def people_set_role(
    request: Request,
    email: str = Form(...),
    role: str = Form(...),
    session: Session = Depends(require_admin_api),
    conn: psycopg.Connection = Depends(get_conn),
) -> object:
    _require_workos(request)
    if role not in ROLES:
        raise HTTPException(status_code=400, detail="role must be editor or viewer")
    if not set_role(conn, email=email, role=role, changed_by=session.editor):
        raise HTTPException(status_code=404, detail="nobody with that email has signed in")
    return RedirectResponse(url="/admin/people", status_code=303)


__all__ = ["router"]
