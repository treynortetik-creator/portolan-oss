"""`/actions`: open actions grouped by owner, overdue judged by the world
clock, Done/Dropped controls (`01-spec/actions-and-owners-2026-09-19.md`).
"""

from __future__ import annotations

from typing import Any

import psycopg
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse

from portolan.activity import record_activity
from portolan.report.util import week_start
from portolan.util import clock
from portolan.web.auth import Session, require_editor, require_session
from portolan.web.deps import get_conn
from portolan.web.queries import actions_feed, record_action_status_change
from portolan.web.tracking import track_page_view

router = APIRouter()

UNASSIGNED_LABEL = "Unassigned"
_OPEN_STATUSES = ("open", "overdue")
_CLOSED_STATUSES = ("done", "dropped")


def _group_open_by_owner(rows: list[dict[str, Any]], today) -> list[dict[str, Any]]:
    """Open actions (status `open`, or `overdue`, which the schema allows),
    grouped by owner: unassigned last, others
    alphabetical, each group sorted by due date. Each row gets `overdue`
    computed against the caller's `today`."""
    groups: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for row in rows:
        if row["status"] not in _OPEN_STATUSES:
            continue
        key = row["owner_crm_id"] or ""
        if key not in groups:
            groups[key] = {
                "owner_crm_id": row["owner_crm_id"],
                "owner_name": row["owner_name"] or UNASSIGNED_LABEL,
                "owner_note_slug": row["owner_note_slug"],
                "actions": [],
            }
            order.append(key)
        overdue = row["due"] is not None and row["due"] < today
        groups[key]["actions"].append({**row, "overdue": overdue})

    def sort_key(key: str) -> tuple[bool, str]:
        name = groups[key]["owner_name"]
        return (name == UNASSIGNED_LABEL, name.lower())

    ordered_keys = sorted(order, key=sort_key)
    for key in ordered_keys:
        groups[key]["actions"].sort(key=lambda a: (a["due"] is None, a["due"]))
    return [groups[key] for key in ordered_keys]


def _closed_this_week(rows: list[dict[str, Any]], today) -> list[dict[str, Any]]:
    monday = week_start(today)
    closed = []
    for row in rows:
        if row["status"] not in _CLOSED_STATUSES or row["closed_at"] is None:
            continue
        closed_date = clock.world_date(row["closed_at"])
        if monday <= closed_date <= today:
            closed.append({**row, "closed_date": closed_date})
    closed.sort(key=lambda r: r["closed_date"], reverse=True)
    return closed


@router.get("/actions", dependencies=[Depends(track_page_view)])
def actions_page(
    request: Request,
    session: Session = Depends(require_session),
    conn: psycopg.Connection = Depends(get_conn),
) -> object:
    templates = request.app.state.templates
    today = clock.today()
    rows = actions_feed(conn)

    groups = _group_open_by_owner(rows, today)
    total_open = sum(len(g["actions"]) for g in groups)
    total_overdue = sum(1 for g in groups for a in g["actions"] if a["overdue"])
    closed_this_week = _closed_this_week(rows, today)

    return templates.TemplateResponse(
        request,
        "actions.html",
        {
            "groups": groups,
            "total_open": total_open,
            "total_overdue": total_overdue,
            "closed_this_week": closed_this_week,
            "today": today,
        },
    )


def _set_status(
    conn: psycopg.Connection, action_id: int, *, to_status: str, editor: str
) -> RedirectResponse:
    with conn.cursor() as cur:
        cur.execute("SELECT note_slug, status FROM actions WHERE id = %s", (action_id,))
        row = cur.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="no such action")
    note_slug, from_status = row
    if from_status not in _OPEN_STATUSES:
        # Already closed (a double submit, or two open tabs) -- a no-op,
        # not a second edit_records row for the same transition.
        return RedirectResponse(url="/actions", status_code=303)
    record_action_status_change(
        conn,
        action_id=action_id,
        note_slug=note_slug,
        from_status=from_status,
        to_status=to_status,
        editor=editor,
        reason=f"marked {to_status}",
    )
    record_activity(
        conn,
        kind="action_confirm",
        user_name=editor,
        route=f"/actions/{{action_id}}/{to_status}",
        target=str(action_id),
        detail={"how": f"marked_{to_status}"},
    )
    return RedirectResponse(url="/actions", status_code=303)


@router.post("/actions/{action_id}/done")
def action_mark_done(
    action_id: int,
    session: Session = Depends(require_editor),
    conn: psycopg.Connection = Depends(get_conn),
) -> object:
    return _set_status(conn, action_id, to_status="done", editor=session.editor)


@router.post("/actions/{action_id}/dropped")
def action_mark_dropped(
    action_id: int,
    session: Session = Depends(require_editor),
    conn: psycopg.Connection = Depends(get_conn),
) -> object:
    return _set_status(conn, action_id, to_status="dropped", editor=session.editor)


__all__ = ["router"]
