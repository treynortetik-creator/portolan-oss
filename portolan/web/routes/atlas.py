"""`/atlas`, `/atlas/{slug}`, `/atlas/panel`: the note browser (folder tree
plus search) that `/search` redirects to.

- `GET /atlas` (optional `?q=`): no note open; an empty state with
  per-folder counts, or the fused-search results when `q` is set.
- `GET /atlas/{slug}` (optional `?q=`, kept so the search box stays
  filled): a note's Properties/body/Edges pane, with the tree open to its
  folder.
- `GET /atlas/panel`: the left-pane fragment (`_atlas_panel.html`) the
  tree's search box swaps in as the reader types. Registered before
  `/atlas/{slug}`, which would otherwise match `panel`.

The full pages include the same `_atlas_panel.html`, built by the same
`_panel_context`, as the fragment route.
"""

from __future__ import annotations

import psycopg
from fastapi import APIRouter, Depends, HTTPException, Request

from portolan.search.embed import Embedder
from portolan.search.query import search as run_search
from portolan.web.atlas_data import (
    ATLAS_BEST_MATCHES_LIMIT,
    active_tree_location,
    atlas_folder_counts,
    atlas_matching_slugs,
    atlas_matching_type_counts,
    atlas_tree_groups,
    linked_accounts,
    note_edges,
    note_properties,
    note_type_label,
)
from portolan.web.auth import Session, require_session
from portolan.web.bounds import bounded_str
from portolan.web.deps import get_conn, get_embedder_dep
from portolan.web.note_view import build_note_context
from portolan.web.queries import note_summaries
from portolan.web.tracking import track_page_view

router = APIRouter()

# Bounds what reaches the embedder and ILIKE.
MAX_ATLAS_QUERY_CHARS = 500
MAX_ACTIVE_SLUG_CHARS = 200


def _panel_context(
    conn: psycopg.Connection, embedder: Embedder, *, q: str, active_slug: str | None
) -> dict[str, object]:
    """Everything `_atlas_panel.html` needs: the folder tree (filtered to
    keyword matches when `q` is set), the fused "Best matches" group (only
    when `q` is set), and which folder/sub-folder holds the active note.
    """
    best_matches: list[dict[str, object]] = []
    if q:
        result = run_search(conn, embedder, q, k=ATLAS_BEST_MATCHES_LIMIT)
        summaries = note_summaries(conn, [hit.note_slug for hit in result.hits])
        best_matches = [
            {
                "slug": hit.note_slug,
                "title": hit.title,
                "updated": hit.updated,
                "best_chunk": hit.best_chunk,
                "description": (summaries.get(hit.note_slug) or {}).get("description"),
            }
            for hit in result.hits
        ]
        # `atlas_matching_slugs` is capped; folder totals come from the
        # uncapped counts.
        groups = atlas_tree_groups(
            conn,
            only_slugs=atlas_matching_slugs(conn, q),
            true_totals=atlas_matching_type_counts(conn, q),
        )
    else:
        groups = atlas_tree_groups(conn)

    active_type, active_subgroup_key = (None, None)
    if active_slug:
        active_type, active_subgroup_key = active_tree_location(conn, active_slug)

    return {
        "q": q,
        "groups": groups,
        "best_matches": best_matches,
        "active_slug": active_slug or "",
        "active_type": active_type,
        "active_subgroup_key": active_subgroup_key,
    }


@router.get("/atlas/panel")
def atlas_panel(
    request: Request,
    q: str = "",
    active: str = "",
    session: Session = Depends(require_session),
    conn: psycopg.Connection = Depends(get_conn),
    embedder: Embedder = Depends(get_embedder_dep),
) -> object:
    templates = request.app.state.templates
    q = bounded_str(q, name="q", max_chars=MAX_ATLAS_QUERY_CHARS).strip()
    active = bounded_str(active, name="active", max_chars=MAX_ACTIVE_SLUG_CHARS).strip()
    context = _panel_context(conn, embedder, q=q, active_slug=active or None)
    return templates.TemplateResponse(request, "_atlas_panel.html", context)


def _atlas_response(
    request: Request,
    conn: psycopg.Connection,
    embedder: Embedder,
    *,
    slug: str | None,
    q: str,
) -> object:
    templates = request.app.state.templates
    q = bounded_str(q, name="q", max_chars=MAX_ATLAS_QUERY_CHARS)
    query = q.strip()

    note_ctx: dict[str, object] | None = None
    if slug:
        try:
            note_ctx = build_note_context(conn, slug)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=f"no such note: {slug}") from exc
        note = note_ctx["note"]
        account = linked_accounts(conn, [slug]).get(slug)
        note_ctx["properties"] = note_properties(
            conn, note, account=account, is_stale=note_ctx["is_stale"]
        )
        note_ctx["edges"] = note_edges(conn, slug)
        note_ctx["account"] = account
        note_ctx["folder_label"] = note_type_label(note.type.value)

    search_ctx: dict[str, object] | None = None
    if query and not slug:
        result = run_search(conn, embedder, query, k=12, expand=1)
        summaries = note_summaries(conn, [hit.note_slug for hit in result.hits])
        search_ctx = {"result": result, "note_summaries": summaries}

    panel_ctx = _panel_context(conn, embedder, q=query, active_slug=slug)
    counts = atlas_folder_counts(conn)

    return templates.TemplateResponse(
        request,
        "atlas.html",
        {
            "q": q,
            "note": note_ctx,
            "search": search_ctx,
            "counts": counts,
            **panel_ctx,
        },
    )


@router.get("/atlas", dependencies=[Depends(track_page_view)])
def atlas_home(
    request: Request,
    q: str = "",
    session: Session = Depends(require_session),
    conn: psycopg.Connection = Depends(get_conn),
    embedder: Embedder = Depends(get_embedder_dep),
) -> object:
    return _atlas_response(request, conn, embedder, slug=None, q=q)


@router.get("/atlas/{slug}", dependencies=[Depends(track_page_view)])
def atlas_note(
    request: Request,
    slug: str,
    q: str = "",
    session: Session = Depends(require_session),
    conn: psycopg.Connection = Depends(get_conn),
    embedder: Embedder = Depends(get_embedder_dep),
) -> object:
    return _atlas_response(request, conn, embedder, slug=slug, q=q)


__all__ = ["router"]
