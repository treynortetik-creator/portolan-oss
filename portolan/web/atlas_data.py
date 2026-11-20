"""Query helpers for `/atlas` (`routes.atlas`): the folder tree, the
keyword filter behind the tree's search box, and the Properties/Edges
data for a note's Atlas pane.

The tree must not load note bodies: the tree queries select only `slug,
title, type, updated`, and the keyword filters (`atlas_matching_slugs`,
`atlas_matching_type_counts`) read `body_md` only inside their WHERE
clause. The single-note pane helpers use `note_summaries`, which does
return `description`.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import psycopg

from portolan.notes.models import NoteType
from portolan.report.priorities import format_priority_stats, theme_priorities_for_atlas
from portolan.web.markdown import WIKILINK_RE
from portolan.web.queries import account_owner, escape_ilike, note_summaries

__all__ = [
    "ATLAS_BEST_MATCHES_LIMIT",
    "ATLAS_FOLDER_CAP",
    "ATLAS_TYPE_LABELS",
    "ATLAS_TYPE_ORDER",
    "active_tree_location",
    "atlas_folder_counts",
    "atlas_matching_slugs",
    "atlas_matching_type_counts",
    "atlas_tree_groups",
    "linked_accounts",
    "note_edges",
    "note_properties",
    "note_type_label",
]

# Plain-word folder names; a type missing here falls back to
# `note_type_label`'s title-case default.
ATLAS_TYPE_LABELS: dict[str, str] = {
    NoteType.ACCOUNT.value: "Accounts",
    NoteType.CONTACT.value: "People",
    NoteType.PRODUCT.value: "Products",
    NoteType.COMPETITOR.value: "Competitors",
    NoteType.SIGNAL.value: "Signals",
    NoteType.THEME.value: "Themes",
    NoteType.DECISION.value: "Decisions",
    NoteType.ACTION.value: "Actions",
    NoteType.SOURCE.value: "Sources",
    NoteType.REPORT.value: "Reports",
    NoteType.CONTEXT.value: "Context",
}

# Folder display order is `NoteType`'s declaration order.
ATLAS_TYPE_ORDER: tuple[str, ...] = tuple(t.value for t in NoteType)

# A folder shows its newest 200 entries; `/notes?type=<type>` is "show all".
ATLAS_FOLDER_CAP = 200

# How many fused-search hits the tree's own "Best matches" group shows.
ATLAS_BEST_MATCHES_LIMIT = 8

_EDGE_KIND_LABELS: dict[str, str] = {
    "body": "Linked notes",
    "parent": "Parent account",
    "supersession": "Supersedes",
    "provenance": "Sources cited",
}
_EDGE_KIND_ORDER = ("body", "parent", "supersession", "provenance")


def note_type_label(note_type: str) -> str:
    """The plain-word folder name for `note_type`: `ATLAS_TYPE_LABELS`, else
    the type title-cased."""
    return ATLAS_TYPE_LABELS.get(note_type, note_type.replace("_", " ").title())


def atlas_folder_counts(conn: psycopg.Connection) -> list[dict[str, Any]]:
    """`[{type, label, total}]`, one row per note type present, in
    `ATLAS_TYPE_ORDER`: the counts on Atlas's empty state."""
    with conn.cursor() as cur:
        cur.execute("SELECT type, count(*) FROM notes GROUP BY type")
        totals = dict(cur.fetchall())
    order = {t: i for i, t in enumerate(ATLAS_TYPE_ORDER)}
    present = sorted(totals, key=lambda t: (order.get(t, len(order)), t))
    return [
        {"type": t, "label": note_type_label(t), "total": totals[t]} for t in present if totals[t]
    ]


def linked_accounts(conn: psycopg.Connection, slugs: list[str]) -> dict[str, tuple[str, str]]:
    """`{slug: (account_slug, account_title)}` for every slug in `slugs`
    with an outgoing `body` link to an `account` note, used for signal
    sub-folders and the Properties "Account" row. One account per note:
    the first by title."""
    if not slugs:
        return {}
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT ON (l.from_slug) l.from_slug, n2.slug, n2.title
            FROM links l
            JOIN notes n2 ON n2.slug = l.to_slug AND n2.type = 'account'
            WHERE l.kind = 'body' AND l.from_slug = ANY(%s)
            ORDER BY l.from_slug, n2.title
            """,
            (slugs,),
        )
        return {row[0]: (row[1], row[2]) for row in cur.fetchall()}


def atlas_matching_slugs(conn: psycopg.Connection, q: str, *, limit: int = 500) -> set[str]:
    """Slugs of notes whose title or body contains `q` (case-insensitive
    substring via ILIKE, so a half-typed word still matches), at most
    `limit`. Never derive a folder total from this capped set; use
    `atlas_matching_type_counts`.

    `q` is escaped first, so `%` and `_` match literally.
    """
    like = f"%{escape_ilike(q)}%"
    with conn.cursor() as cur:
        cur.execute(
            "SELECT slug FROM notes WHERE title ILIKE %s ESCAPE '\\' "
            "OR body_md ILIKE %s ESCAPE '\\' LIMIT %s",
            (like, like, limit),
        )
        return {row[0] for row in cur.fetchall()}


def atlas_matching_type_counts(conn: psycopg.Connection, q: str) -> dict[str, int]:
    """`{type: count}` of notes matching `q`, uncapped: the folder totals
    for a keyword-filtered tree (`atlas_matching_slugs` is capped)."""
    like = f"%{escape_ilike(q)}%"
    with conn.cursor() as cur:
        cur.execute(
            "SELECT type, count(*) FROM notes "
            "WHERE title ILIKE %s ESCAPE '\\' OR body_md ILIKE %s ESCAPE '\\' "
            "GROUP BY type",
            (like, like),
        )
        return dict(cur.fetchall())


def atlas_tree_groups(
    conn: psycopg.Connection,
    *,
    only_slugs: set[str] | None = None,
    cap: int = ATLAS_FOLDER_CAP,
    true_totals: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """The folder tree: one group per note type present (in
    `ATLAS_TYPE_ORDER`), each `{type, label, total, notes, truncated,
    subgroups}`. `notes` holds at most `cap` entries, newest-updated first
    (themes reordered by `_theme_priority_order`); `truncated` is true when
    `total` exceeds what is shown. Signals get `subgroups` by linked
    account and `notes = None`; every other type has `subgroups = None`.

    `only_slugs` restricts the tree to that slug set (`None` means every
    note). The per-type cap is applied in SQL. `true_totals` (uncapped
    counts from `atlas_matching_type_counts`) replaces the totals counted
    from `only_slugs`, which is itself capped; a type in `true_totals`
    with no rows in `only_slugs` still gets an empty, truncated group.
    """
    # An empty filter result has no rows to query, but `true_totals` may
    # still name types that need an empty, truncated folder.
    if only_slugs is not None and not only_slugs and not true_totals:
        return []

    where = ""
    params: list[Any] = []
    skip_row_query = only_slugs is not None and not only_slugs
    if only_slugs:
        where = "WHERE slug = ANY(%s)"
        params.append(list(only_slugs))

    totals: dict[str, int] = {}
    rows: list[tuple[Any, ...]] = []
    if not skip_row_query:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT type, count(*) FROM notes {where} GROUP BY type",  # noqa: S608 - where is a fixed literal, params bound
                params,
            )
            totals = dict(cur.fetchall())

            cur.execute(
                f"""
                SELECT slug, title, type, updated FROM (
                    SELECT slug, title, type, updated,
                           row_number() OVER (PARTITION BY type ORDER BY updated DESC) AS rn
                    FROM notes {where}
                ) ranked
                WHERE rn <= %s
                ORDER BY type, updated DESC
                """,  # noqa: S608
                (*params, cap),
            )
            rows = cur.fetchall()

    if true_totals is not None:
        totals = dict(true_totals)

    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for slug, title, ntype, updated in rows:
        by_type[ntype].append({"slug": slug, "title": title, "updated": updated})

    order = {t: i for i, t in enumerate(ATLAS_TYPE_ORDER)}
    present_types = sorted(set(by_type) | set(totals), key=lambda t: (order.get(t, len(order)), t))

    groups: list[dict[str, Any]] = []
    for note_type in present_types:
        total = totals.get(note_type, 0)
        if total <= 0:
            continue
        entries = by_type.get(note_type, [])
        group: dict[str, Any] = {
            "type": note_type,
            "label": note_type_label(note_type),
            "total": total,
            "notes": entries,
            "truncated": total > len(entries),
            "subgroups": None,
        }
        if note_type == NoteType.SIGNAL.value:
            group["subgroups"] = _signal_subgroups(conn, entries)
            group["notes"] = None
        if note_type == NoteType.THEME.value:
            group["notes"] = _theme_priority_order(conn, entries)
        groups.append(group)
    return groups


def _theme_priority_order(
    conn: psycopg.Connection, entries: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """The theme folder's entries with the themes `theme_priorities_for_atlas`
    ranks moved first, in rank order, each carrying a `priority` line
    (`format_priority_stats`). Unranked themes follow in their original
    order, with no `priority` key (01-spec/actions-and-owners-2026-09-19.md,
    Phase 3)."""
    ranked = theme_priorities_for_atlas(conn)
    if not ranked:
        return entries
    stats_by_slug = {p.theme_slug: format_priority_stats(p) for p in ranked}
    rank_by_slug = {p.theme_slug: i for i, p in enumerate(ranked)}

    by_slug = {e["slug"]: e for e in entries}
    ordered = [by_slug[slug] for slug in rank_by_slug if slug in by_slug]
    ordered.extend(e for e in entries if e["slug"] not in rank_by_slug)
    for entry in ordered:
        stats = stats_by_slug.get(entry["slug"])
        if stats is not None:
            entry["priority"] = stats
    return ordered


def _signal_subgroups(
    conn: psycopg.Connection, entries: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    acct_map = linked_accounts(conn, [e["slug"] for e in entries])
    buckets: dict[str | None, dict[str, Any]] = {}
    for entry in entries:
        account = acct_map.get(entry["slug"])
        key = account[0] if account else None
        label = account[1] if account else "Unlinked"
        bucket = buckets.setdefault(key, {"key": key, "label": label, "notes": []})
        bucket["notes"].append(entry)
    # Unlinked (key is None) sorts last; everything else alphabetical by
    # its account's title.
    return sorted(buckets.values(), key=lambda b: (b["key"] is None, b["label"].lower()))


def active_tree_location(conn: psycopg.Connection, slug: str) -> tuple[str | None, str | None]:
    """`(type, subgroup_key)` for `slug`: which folder (and, for a signal,
    which account sub-folder) to expand when this note is open.
    `(None, None)` if the slug doesn't exist."""
    with conn.cursor() as cur:
        cur.execute("SELECT type FROM notes WHERE slug = %s", (slug,))
        row = cur.fetchone()
    if row is None:
        return (None, None)
    note_type = row[0]
    if note_type != NoteType.SIGNAL.value:
        return (note_type, None)
    account = linked_accounts(conn, [slug]).get(slug)
    return (note_type, account[0] if account else None)


def note_properties(
    conn: psycopg.Connection,
    note: Any,
    *,
    account: tuple[str, str] | None,
    is_stale: bool = False,
) -> list[dict[str, Any]]:
    """The note's frontmatter fields as `{label, kind, value, href}` rows for
    the Properties block. `kind` tells the template how to render `value`:
    `badge`, `date`, `mono`, `wrap` (long text, existing `[[slug]]` targets
    linkified), `link` (one `href`), or `source_links` (a list of
    `{slug, exists}`, linked only when `exists`).

    Parent, superseded-by and source targets have no foreign key, so each
    is looked up: a missing target renders as plain text, not a link to a
    404; a found one shows its title. `NoteRow` has no tags; type and
    status are the badges. `account` is the caller's
    `linked_accounts(...)[note.slug]`; `is_stale` colours the Updated row.
    """
    rows: list[dict[str, Any]] = [
        {
            "label": "Type",
            "kind": "badge",
            "value": note.type.value,
            "badge_class": f"type-{note.type.value}",
        },
        {"label": "Status", "kind": "badge", "value": note.status.value, "badge_class": ""},
    ]
    if note.description:
        # Only `[[slug]]`s that exist are linked (`wiki_targets`, passed to
        # `linkify_wikilinks` in atlas.html); a dangling one stays plain text.
        wiki_slugs = WIKILINK_RE.findall(note.description)
        wiki_targets = set(note_summaries(conn, wiki_slugs)) if wiki_slugs else set()
        rows.append(
            {
                "label": "Description",
                "kind": "wrap",
                "value": note.description,
                "wiki_targets": wiki_targets,
            }
        )
    rows.append({"label": "Created", "kind": "date", "value": note.created})
    rows.append(
        {
            "label": "Updated",
            "kind": "date",
            "value": note.updated,
            "extra_class": "stale" if is_stale else "fresh",
        }
    )
    if note.stale_after:
        rows.append({"label": "Stale after", "kind": "date", "value": note.stale_after})
    if note.crm_id:
        rows.append({"label": "CRM ID", "kind": "mono", "value": note.crm_id})

    # Owner is a registry fact, read live from `registry_entities`, not note
    # text (01-spec/actions-and-owners-2026-09-19.md).
    if note.type == NoteType.ACCOUNT and note.crm_id:
        owner = account_owner(conn, note.crm_id)
        owner_slug = owner.get("note_slug") if owner else None
        rows.append(
            {
                "label": "Owner",
                "kind": "link" if owner_slug else "wrap",
                "value": owner["name"] if owner else "unassigned",
                "href": f"/atlas/{owner_slug}" if owner_slug else None,
            }
        )

    related_slugs = [s for s in (note.parent_slug, note.superseded_by) if s]
    related = note_summaries(conn, related_slugs) if related_slugs else {}

    if note.parent_slug:
        target = related.get(note.parent_slug)
        rows.append(
            {
                "label": "Parent account",
                "kind": "link" if target else "wrap",
                "value": target["title"] if target else note.parent_slug,
                "href": f"/atlas/{note.parent_slug}" if target else None,
            }
        )
    if account:
        account_slug, account_title = account
        rows.append(
            {
                "label": "Account",
                "kind": "link",
                "value": account_title,
                "href": f"/atlas/{account_slug}",
            }
        )
    if note.superseded_by:
        target = related.get(note.superseded_by)
        rows.append(
            {
                "label": "Superseded by",
                "kind": "link" if target else "wrap",
                "value": target["title"] if target else note.superseded_by,
                "href": f"/atlas/{note.superseded_by}" if target else None,
            }
        )
    source_slugs = _cited_source_slugs(conn, note.slug)
    if source_slugs:
        existing_sources = _source_meta(conn, source_slugs)
        rows.append(
            {
                "label": "Source",
                "kind": "source_links",
                "value": [
                    {"slug": slug, "exists": slug in existing_sources} for slug in source_slugs
                ],
            }
        )
    rows.append({"label": "Version", "kind": "mono", "value": note.version})
    if note.schema_version:
        rows.append({"label": "Schema version", "kind": "mono", "value": note.schema_version})
    return rows


def _cited_source_slugs(conn: psycopg.Connection, note_slug: str) -> list[str]:
    """The sources this note cites: its `provenance` rows in `links`."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT to_slug FROM links WHERE from_slug = %s AND kind = 'provenance' "
            "ORDER BY to_slug",
            (note_slug,),
        )
        return [row[0] for row in cur.fetchall()]


def note_edges(conn: psycopg.Connection, note_slug: str) -> list[dict[str, Any]]:
    """Outgoing edges from `note_slug`, grouped by `links.kind`: `{kind,
    label, entries}`. The key is `entries`, not `items`, because Jinja's
    `group.items` would resolve to the dict method. Note targets
    (`body`/`parent`/`supersession`) are `{slug, title, href, type,
    updated}` linked to `/atlas/<slug>`; `provenance` targets are sources,
    `{slug, title, href}` with the title as "<slug> · <source_type>",
    linked to `/originals/<slug>`. A missing target has `href = None`.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT to_slug, kind FROM links WHERE from_slug = %s ORDER BY kind, to_slug",
            (note_slug,),
        )
        rows = cur.fetchall()
    if not rows:
        return []

    note_targets = [to_slug for to_slug, kind in rows if kind != "provenance"]
    source_targets = [to_slug for to_slug, kind in rows if kind == "provenance"]

    note_meta = note_summaries(conn, note_targets)
    source_meta = _source_meta(conn, source_targets)

    # `links.to_slug` has no foreign key, so a target may not exist; `href`
    # is set only when it does, and the template renders plain text otherwise.
    by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for to_slug, kind in rows:
        if kind == "provenance":
            meta = source_meta.get(to_slug)
            label = to_slug if meta is None else f"{to_slug} · {meta['source_type']}"
            href = f"/originals/{to_slug}" if meta is not None else None
            by_kind[kind].append({"slug": to_slug, "title": label, "href": href})
        else:
            meta = note_meta.get(to_slug)
            title = meta["title"] if meta else to_slug
            href = f"/atlas/{to_slug}" if meta is not None else None
            by_kind[kind].append(
                {
                    "slug": to_slug,
                    "title": title,
                    "href": href,
                    "type": meta["type"] if meta else None,
                    "updated": meta["updated"] if meta else None,
                }
            )

    return [
        {"kind": kind, "label": _EDGE_KIND_LABELS[kind], "entries": by_kind[kind]}
        for kind in _EDGE_KIND_ORDER
        if by_kind.get(kind)
    ]


def _source_meta(conn: psycopg.Connection, slugs: list[str]) -> dict[str, dict[str, Any]]:
    if not slugs:
        return {}
    with conn.cursor() as cur:
        cur.execute(
            "SELECT slug, source_type, source_date FROM sources WHERE slug = ANY(%s)",
            (list(dict.fromkeys(slugs)),),
        )
        rows = cur.fetchall()
    return {r[0]: {"source_type": r[1], "source_date": r[2]} for r in rows}
