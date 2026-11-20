"""`/graph` (a page with 2D and 3D JavaScript islands), `/graph.json`
(their data) and `/graph/note/{slug}.json` (the side panel a node click
opens).

`/graph.json` returns the newest `GRAPH_NODE_CAP` notes (optional
`?type=`) and the `body`/`parent`/`supersession` links among them, plus
per-node `degree` and `total_notes`/`truncated` for the cap notice.
`mode` (`2d`/`3d`) only picks the island `/graph` shows first; `/graph.json`
accepts it but returns the same data.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import psycopg
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from portolan.notes.models import NoteRow
from portolan.notes.persist import read_note
from portolan.web.auth import Session, require_session, require_session_api
from portolan.web.deps import get_conn
from portolan.web.markdown import render_note_body
from portolan.web.queries import backlinks, note_summaries
from portolan.web.tracking import track_page_view

router = APIRouter()

GRAPH_NODE_CAP = 2000
_EDGE_KINDS = ("body", "parent", "supersession")
_MODES = ("2d", "3d")


@router.get("/graph", dependencies=[Depends(track_page_view)])
def graph_page(
    request: Request, mode: str = "3d", session: Session = Depends(require_session)
) -> object:
    templates = request.app.state.templates
    initial_mode = mode if mode in _MODES else "3d"
    return templates.TemplateResponse(request, "graph.html", {"initial_mode": initial_mode})


def _total_notes(conn: psycopg.Connection, type_filter: str | None) -> int:
    with conn.cursor() as cur:
        if type_filter:
            cur.execute("SELECT count(*) FROM notes WHERE type = %s", (type_filter,))
        else:
            cur.execute("SELECT count(*) FROM notes")
        (total,) = cur.fetchone()
    return total


@router.get("/graph.json")
def graph_json(
    request: Request,
    type: str | None = None,
    mode: str | None = None,
    session: Session = Depends(require_session_api),
    conn: psycopg.Connection = Depends(get_conn),
) -> JSONResponse:
    # Accepted but unused; an unknown value is a 400 rather than ignored.
    if mode is not None and mode not in _MODES:
        raise HTTPException(status_code=400, detail=f"mode must be one of {_MODES} (got {mode!r})")
    with conn.cursor() as cur:
        if type:
            cur.execute(
                "SELECT slug, title, type, updated FROM notes WHERE type = %s "
                "ORDER BY updated DESC LIMIT %s",
                (type, GRAPH_NODE_CAP),
            )
        else:
            cur.execute(
                "SELECT slug, title, type, updated FROM notes ORDER BY updated DESC LIMIT %s",
                (GRAPH_NODE_CAP,),
            )
        rows = cur.fetchall()

    nodes = [{"id": r[0], "title": r[1], "type": r[2], "updated": r[3].isoformat()} for r in rows]
    slug_list = [n["id"] for n in nodes]

    # Both endpoints bounded to the returned nodes, so the edge query never
    # outgrows GRAPH_NODE_CAP.
    with conn.cursor() as cur:
        cur.execute(
            "SELECT from_slug, to_slug, kind FROM links "
            "WHERE kind = ANY(%s) AND from_slug = ANY(%s) AND to_slug = ANY(%s)",
            (list(_EDGE_KINDS), slug_list, slug_list),
        )
        edge_rows = cur.fetchall()

    edges = [{"source": r[0], "target": r[1], "kind": r[2]} for r in edge_rows]

    degree: Counter[str] = Counter()
    for e in edges:
        degree[e["source"]] += 1
        degree[e["target"]] += 1
    for n in nodes:
        n["degree"] = degree.get(n["id"], 0)

    total_notes = _total_notes(conn, type)
    truncated = total_notes > len(nodes)

    return JSONResponse(
        {
            "nodes": nodes,
            "edges": edges,
            "total_notes": total_notes,
            "truncated": truncated,
            "node_cap": GRAPH_NODE_CAP,
        }
    )


def _summary_text(note: NoteRow) -> str:
    """A short lede for the graph note panel: the `description`, else the
    first paragraph of `synthesis`, else the title."""
    if note.description:
        return note.description
    synthesis = (note.synthesis or "").strip()
    if synthesis:
        first_para = synthesis.split("\n\n", 1)[0].strip()
        if first_para:
            return first_para
    return note.title


@router.get("/graph/note/{slug}.json")
def graph_note_json(
    slug: str,
    session: Session = Depends(require_session_api),
    conn: psycopg.Connection = Depends(get_conn),
) -> JSONResponse:
    """The graph side panel's data: title, type, updated, a summary, the
    body as safe HTML, and linked notes.

    `body_html` renders synthesis, else description, else title through
    `render_note_body`, the renderer `/notes/{slug}` and `/reports/{slug}`
    use; it resolves the body's own `[[slug]]` links.

    `links` is the union of `note.related` (the slugs listed in the note's
    "## Related" section, not every outgoing link) and `backlinks`
    (incoming), deduplicated by slug and sorted by title.
    """
    try:
        note, _claims = read_note(conn, slug)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=f"no such note: {slug}") from exc

    incoming = backlinks(conn, slug)
    outgoing_slugs = [s for s in note.related if s != slug]
    outgoing = note_summaries(conn, outgoing_slugs)

    links_by_slug: dict[str, dict[str, Any]] = {}
    for row in incoming:
        if row["slug"] == slug:
            continue
        links_by_slug[row["slug"]] = {
            "slug": row["slug"],
            "title": row["title"],
            "type": row["type"],
        }
    for link_slug, info in outgoing.items():
        links_by_slug[link_slug] = {
            "slug": link_slug,
            "title": info["title"],
            "type": info["type"],
        }

    body_source = note.synthesis or note.description or note.title
    body_html = str(render_note_body(conn, body_source))

    return JSONResponse(
        {
            "slug": note.slug,
            "title": note.title,
            "type": note.type.value,
            "updated": note.updated.isoformat(),
            "summary": _summary_text(note),
            "body_html": body_html,
            "links": sorted(links_by_slug.values(), key=lambda item: item["title"]),
        }
    )


__all__ = ["router"]
