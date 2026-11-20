"""Note-detail data and rendering for `/notes/{slug}`.

`build_note_context` builds the data with no `Request` dependency;
`render_note_detail` wraps it in `note.html`, and `routes.atlas` wraps the
same data in its own pane. The body renders through
`portolan.web.markdown.render_note_body`, the renderer `/reports/{slug}`
and the graph note panel also use.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import psycopg
from fastapi import HTTPException, Request

from portolan.notes.models import NoteType
from portolan.notes.persist import read_note
from portolan.report.action_done import build_done_context
from portolan.report.action_why import build_why_context
from portolan.util import clock
from portolan.web.markdown import render_note_body
from portolan.web.queries import (
    account_owner,
    actions_feed,
    backlinks,
    note_summaries,
    note_versions,
)

__all__ = [
    "action_done_for_note",
    "action_for_note",
    "action_why_for_note",
    "build_note_context",
    "render_note_detail",
]


def _action_id_for_note(conn: psycopg.Connection, note: Any) -> int | None:
    if note.type != NoteType.ACTION:
        return None
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM actions WHERE note_slug = %s", (note.slug,))
        action_row = cur.fetchone()
    return action_row[0] if action_row is not None else None


def action_why_for_note(conn: psycopg.Connection, note: Any) -> dict[str, Any] | None:
    """The researched card's why for an action note, or `None` (a pipeline
    action with no card, or not an action note)."""
    action_id = _action_id_for_note(conn, note)
    return build_why_context(conn, action_id) if action_id is not None else None


def action_done_for_note(conn: psycopg.Connection, note: Any) -> dict[str, Any] | None:
    """The later claim a person confirmed as closing this action, or `None`
    (still open, closed by hand, or not an action note)."""
    action_id = _action_id_for_note(conn, note)
    return build_done_context(conn, action_id) if action_id is not None else None


def action_for_note(conn: psycopg.Connection, slug: str) -> dict[str, Any] | None:
    """The action's `actions_feed` row (owner, account, due, status)."""
    return next((a for a in actions_feed(conn) if a["note_slug"] == slug), None)


def build_note_context(conn: psycopg.Connection, slug: str) -> dict[str, Any]:
    """The note, its claims, and everything a detail page renders around
    them. Raises `ValueError` (from `read_note`) if no such note exists.
    """
    note, claims = read_note(conn, slug)

    related_summaries = note_summaries(conn, note.related)
    related = [related_summaries[s] for s in note.related if s in related_summaries]

    # Drives the fresh/stale colour on the "updated:" stamp; computed here
    # so the template never needs today's date.
    stale_after = note.stale_after
    if stale_after is not None and stale_after.tzinfo is None:
        stale_after = stale_after.replace(tzinfo=UTC)
    is_stale = stale_after is not None and stale_after < datetime.now(UTC)

    # Same fallback chain as `routes.graph.graph_note_json`.
    body_source = note.synthesis or note.description or note.title
    body_html = render_note_body(conn, body_source)

    # An account note shows its registry owner and its open actions; a
    # contact note shows the open actions that contact owns
    # (01-spec/actions-and-owners-2026-09-19.md).
    owner: dict[str, Any] | None = None
    note_actions: list[dict[str, Any]] = []
    if note.type == NoteType.ACCOUNT and note.crm_id:
        owner = account_owner(conn, note.crm_id)
        note_actions = [
            a
            for a in actions_feed(conn)
            if a["account_crm_id"] == note.crm_id and a["status"] in ("open", "overdue")
        ]
    elif note.type == NoteType.CONTACT and note.crm_id:
        note_actions = [
            a
            for a in actions_feed(conn)
            if a["owner_crm_id"] == note.crm_id and a["status"] in ("open", "overdue")
        ]
    # Overdue against the world clock's today, the rule `/actions` uses.
    today = clock.today()
    note_actions = [
        {**a, "overdue": a["due"] is not None and a["due"] < today} for a in note_actions
    ]

    # `why` is `None` for an action with no researched card. When set, the
    # page shows its inference and lists `why["claims"]`, then any claim of
    # the note's own: chat can cite one, and its link needs the anchor.
    why = action_why_for_note(conn, note) if note.type == NoteType.ACTION else None
    claims_to_show = claims
    if why is not None:
        card_ids = {c.id for c in why["claims"]}
        claims_to_show = [*why["claims"], *(c for c in claims if c.id not in card_ids)]
    done = action_done_for_note(conn, note) if note.type == NoteType.ACTION else None

    return {
        "note": note,
        "claims": claims,
        "claims_to_show": claims_to_show,
        "related": related,
        "backlinks": backlinks(conn, slug),
        "versions": note_versions(conn, slug),
        "is_stale": is_stale,
        "body_html": body_html,
        "owner": owner,
        "note_actions": note_actions,
        "why": why,
        "done": done,
    }


def render_note_detail(
    request: Request, conn: psycopg.Connection, slug: str, *, base_path: str
) -> object:
    templates = request.app.state.templates
    try:
        context = build_note_context(conn, slug)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=f"no such note: {slug}") from exc

    context["base_path"] = base_path
    return templates.TemplateResponse(request, "note.html", context)
