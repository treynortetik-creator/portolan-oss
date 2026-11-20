"""The MCP tools' actual logic, decoupled from MCP registration.

**Raw, on purpose.** Everything here returns stored text exactly as
stored. Sanitizing for an outside client happens only at the door
(`server.py`, `sanitize.protect_payload`), because the action researcher
calls these functions directly and its evidence gate compares copied
quotes byte-for-byte with the stored claims
(`tests/test_mcp_door.py::test_the_raw_tools_the_researcher_uses_return_the_stored_bytes`).
Do not sanitize here.

Kept separate from `server.py` (which wires these onto `MCPServer` via
`.tool()`) so tests can call them directly against a migrated Postgres
connection, with no MCP protocol round-trip.

Every `tool_*` function takes an already-open `psycopg.Connection`
(`server.py` opens one per tool call) and ends by writing one
`usage_events` row with `surface='mcp'` (`_log_usage`, migration 0003):
the tool name, argument *sizes* and a hit count -- never a note body,
claim text, or query string.

Return values are plain JSON-shaped `dict`/`list` (dates and datetimes as
ISO 8601 strings); `server.py`'s tool docstrings describe the shape a
client model reads.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timedelta
from typing import Any
from urllib.parse import quote as urlquote

import psycopg
from psycopg.types.json import Jsonb

from portolan.ingest.ledger import summary as coverage_summary
from portolan.report.accounts import home_account_slug
from portolan.search.embed import Embedder
from portolan.search.query import MAX_SEARCH_QUERY_CHARS
from portolan.search.query import search as run_search
from portolan.store import Store
from portolan.util import clock as clock_module
from portolan.util.redact import redact_dsn

MCP_SURFACE = "mcp"

# Numeric arguments clamp silently to the nearest bound: an MCP client is a model guessing
# at values, and a smaller `k`, `depth`, `since_days` or `expand` still answers the question.
MAX_SEARCH_K = 100
MAX_GRAPH_DEPTH = 5
MAX_COVERAGE_SINCE_DAYS = 3_650

# Each `expand` hop is one links query per frontier note. Standing-question text raises past
# its bound instead of truncating: a cut-short question would silently change what is asked.
MAX_EXPAND = 3
MAX_NEAR_SLUGS = 20
MAX_STANDING_QUESTION_READER_CHARS = 200
MAX_STANDING_QUESTION_CHARS = 2_000


def _log_usage(
    conn: psycopg.Connection,
    *,
    tenant: str,
    tool: str,
    arg_sizes: dict[str, Any],
    hit_count: int,
) -> None:
    """Write one `usage_events` row for an MCP tool call.

    `model_key` carries the tool name (prefixed `mcp:`) because the column
    is `NOT NULL` (0001_init.sql); the token and cost columns are zero.
    This row records the call only; model spend inside it (the query
    embedding in `tool_search`) is not metered here.
    """
    payload: dict[str, Any] = {"tool": tool, "args": arg_sizes, "hit_count": hit_count}
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO usage_events
                (tenant, surface, model_key, input_tokens, output_tokens,
                 cached_tokens, cost_estimate, request_id, extra)
            VALUES (%s, %s, %s, 0, 0, 0, 0, %s, %s)
            """,
            (tenant, MCP_SURFACE, f"mcp:{tool}", f"req_{uuid.uuid4().hex}", Jsonb(payload)),
        )


def _load_claims(conn: psycopg.Connection, note_slug: str) -> list[dict[str, Any]]:
    """Every claim under a note, source slug + locator + quote + both
    clocks, per provenance-chain.md and portolan-spec-v2.md sec 2."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, fact_type, claim_text, source_slug, locator, quote, speaker,
                   valid_from, valid_to, recorded, superseded_at, severity, sentiment,
                   confidence, archived
            FROM claims
            WHERE note_slug = %s
            ORDER BY recorded
            """,
            (note_slug,),
        )
        rows = cur.fetchall()
    return _claim_rows_to_dicts(rows)


def _claim_rows_to_dicts(rows: list[tuple]) -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    for (
        claim_id,
        fact_type,
        claim_text,
        source_slug,
        locator,
        quote,
        speaker,
        valid_from,
        valid_to,
        recorded,
        superseded_at,
        severity,
        sentiment,
        confidence,
        archived,
    ) in rows:
        claims.append(
            {
                "id": claim_id,
                "fact_type": fact_type,
                "claim_text": claim_text,
                "source_slug": source_slug,
                "locator": locator,
                "quote": quote,
                "speaker": speaker,
                # World clock: when the fact was true.
                "valid_from": valid_from.isoformat() if valid_from else None,
                "valid_to": valid_to.isoformat() if valid_to else None,
                # Record clock: when Portolan learned it / when it was superseded.
                "recorded": recorded.isoformat(),
                "superseded_at": superseded_at.isoformat() if superseded_at else None,
                "severity": severity,
                "sentiment": sentiment,
                "confidence": float(confidence) if confidence is not None else None,
                "archived": archived,
            }
        )
    return claims


def _load_links(conn: psycopg.Connection, from_slug: str) -> list[dict[str, str]]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT to_slug, kind FROM links WHERE from_slug = %s ORDER BY kind, to_slug",
            (from_slug,),
        )
        return [{"to_slug": row[0], "kind": row[1]} for row in cur.fetchall()]


def _link_details(
    conn: psycopg.Connection, slug: str, links: list[dict[str, str]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """`links` with each target's `title` and `type` (a source's `source_type`
    for a provenance link; both `None` for a dangling link), and the notes
    linking to `slug` (`body`/`parent`/`supersession`, as `graph_neighbors`
    follows them), each with its title and type: the page's way out in both
    directions, so an MCP client can hop without a `graph_neighbors` call."""
    targets = [link["to_slug"] for link in links]
    with conn.cursor() as cur:
        cur.execute("SELECT slug, title, type FROM notes WHERE slug = ANY(%s)", (targets,))
        notes = {row[0]: (row[1], row[2]) for row in cur.fetchall()}
        cur.execute("SELECT slug, source_type FROM sources WHERE slug = ANY(%s)", (targets,))
        sources = {row[0]: row[1] for row in cur.fetchall()}
        cur.execute(
            """
            SELECT l.from_slug, l.kind, n.title, n.type
            FROM links l JOIN notes n ON n.slug = l.from_slug
            WHERE l.to_slug = %s AND l.kind = ANY(%s)
            ORDER BY l.kind, l.from_slug
            """,
            (slug, list(_NOTE_LINK_KINDS)),
        )
        incoming = cur.fetchall()
    detailed = []
    for link in links:
        title, note_type = notes.get(link["to_slug"], (None, None))
        if link["kind"] == "provenance":
            title, note_type = None, sources.get(link["to_slug"])
        detailed.append({**link, "title": title, "type": note_type})
    linked_from = [
        {"from_slug": from_slug, "kind": kind, "title": title, "type": note_type}
        for from_slug, kind, title, note_type in incoming
    ]
    return detailed, linked_from


# ---------------------------------------------------------------------
# Keyset paging over a note's claims, frozen at a snapshot. The door's
# `open_note` (first page) and `list_claims` (later pages) use it;
# `tool_open_note` with no `claims_limit` returns every claim, unpaged.
# ---------------------------------------------------------------------

DEFAULT_CLAIMS_PAGE = 50
MAX_CLAIMS_PAGE = 200


def take_claims_snapshot(conn: psycopg.Connection) -> tuple[int, datetime]:
    """`(max_claim_id, taken_at)`: the snapshot boundary for a new
    listing. `claims.id` is an identity column, so every claim inserted
    after this moment has a larger id and is excluded from every page of
    the listing, whatever its `recorded` value says (`recorded` is set by
    the writer and can be backdated, so it cannot be the boundary).
    `taken_at` is the database's `now()`, the clock `superseded_at` is
    written with, so a claim superseded after the snapshot still reads as
    current in it (`tests/test_mcp_door.py`)."""
    with conn.cursor() as cur:
        cur.execute("SELECT coalesce(max(id), 0), now() FROM claims")
        max_id, taken_at = cur.fetchone()
    return int(max_id), taken_at


def claims_page(
    conn: psycopg.Connection,
    *,
    slug: str,
    limit: int,
    snapshot_max_id: int,
    snapshot_taken_at: datetime,
    after: tuple[datetime, int] | None = None,
) -> dict[str, Any]:
    """One page of `slug`'s claims inside a snapshot, oldest first.

    Ordering key `(recorded, id)`: `recorded` is the unpaged order
    (`_load_claims`), and `id` breaks its ties, so the key is unique and
    total -- no row can sit on a page boundary twice or fall between two
    pages. `after` is the key of the last row of the previous page; this
    page is everything strictly after it. `superseded_at` is reported as of
    the snapshot (`NULL` if the supersession happened after `taken_at`).
    `archived` has no timestamp and is read live.

    Returns `{"claims", "total", "has_more", "next_after"}`; `next_after`
    is `(recorded, id)` of this page's last row when `has_more`, else
    `None`. The caller turns it into an opaque cursor.
    """
    limit = max(1, min(int(limit), MAX_CLAIMS_PAGE))
    after_recorded, after_id = after if after is not None else (None, None)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, fact_type, claim_text, source_slug, locator, quote, speaker,
                   valid_from, valid_to, recorded,
                   CASE WHEN superseded_at <= %(taken_at)s THEN superseded_at END,
                   severity, sentiment, confidence, archived
            FROM claims
            WHERE note_slug = %(slug)s
              AND id <= %(max_id)s
              AND (%(after_recorded)s::timestamptz IS NULL
                   OR (recorded, id) > (%(after_recorded)s::timestamptz, %(after_id)s::bigint))
            ORDER BY recorded, id
            LIMIT %(fetch)s
            """,
            {
                "slug": slug,
                "max_id": snapshot_max_id,
                "taken_at": snapshot_taken_at,
                "after_recorded": after_recorded,
                "after_id": after_id,
                "fetch": limit + 1,
            },
        )
        rows = cur.fetchall()
        cur.execute(
            "SELECT count(*) FROM claims WHERE note_slug = %s AND id <= %s",
            (slug, snapshot_max_id),
        )
        (total,) = cur.fetchone()
    has_more = len(rows) > limit
    rows = rows[:limit]
    next_after = (rows[-1][9], int(rows[-1][0])) if has_more and rows else None
    return {
        "claims": _claim_rows_to_dicts(rows),
        "total": int(total),
        "has_more": has_more,
        "next_after": next_after,
    }


def _note_exists(conn: psycopg.Connection, slug: str) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM notes WHERE slug = %s", (slug,))
        return cur.fetchone() is not None


def tool_list_claims(
    conn: psycopg.Connection,
    *,
    tenant: str,
    slug: str,
    limit: int = DEFAULT_CLAIMS_PAGE,
    snapshot: tuple[int, datetime] | None = None,
    after: tuple[datetime, int] | None = None,
) -> dict[str, Any] | None:
    """One page of a note's claims (`claims_page`), inside `snapshot`
    (`(max_claim_id, taken_at)`, from a cursor) or a fresh one when
    `snapshot` is `None` (a new listing). `None` if the note doesn't
    exist. Returns `{"slug", "claims", "total", "has_more", "next_after",
    "snapshot_max_claim_id", "snapshot_taken_at"}`; the door encodes the
    last three into the opaque cursor (and shows the snapshot boundary in
    `claims_page.snapshot`)."""
    result: dict[str, Any] | None = None
    if _note_exists(conn, slug):
        max_id, taken_at = snapshot if snapshot is not None else take_claims_snapshot(conn)
        page = claims_page(
            conn,
            slug=slug,
            limit=limit,
            snapshot_max_id=max_id,
            snapshot_taken_at=taken_at,
            after=after,
        )
        result = {
            "slug": slug,
            **page,
            "snapshot_max_claim_id": max_id,
            "snapshot_taken_at": taken_at,
        }
    _log_usage(
        conn,
        tenant=tenant,
        tool="list_claims",
        arg_sizes={
            "slug_chars": len(slug),
            "limit": max(1, min(int(limit), MAX_CLAIMS_PAGE)),
            "continuation": snapshot is not None,
        },
        hit_count=len(result["claims"]) if result is not None else 0,
    )
    return result


def tool_search(
    conn: psycopg.Connection,
    embedder: Embedder,
    *,
    tenant: str,
    query: str,
    k: int = 12,
    expand: int = 0,
    types: list[str] | None = None,
    since: date | None = None,
    near: list[str] | None = None,
) -> dict[str, Any]:
    """Fused search (`portolan.search.query.search`). See `server.py`'s
    `search` tool docstring for the returned shape.

    A query over `MAX_SEARCH_QUERY_CHARS` raises rather than being cut:
    it would otherwise go whole into the embedder and the keyword leg, and
    an overlong query is the caller's mistake, not a "zero results" answer.
    `k` and `expand` clamp. `near` narrows the search to those notes and
    the notes one `body`/`parent` link away (`search(entity_slugs=...)`).
    """
    if near is not None and len(near) > MAX_NEAR_SLUGS:
        raise ValueError(
            f"near names {len(near)} notes; the limit is {MAX_NEAR_SLUGS}. Search near fewer."
        )
    if near:
        # A scope of notes that do not exist would read as a healthy "nothing recorded".
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM notes WHERE slug = ANY(%s) LIMIT 1", (list(near),))
            if cur.fetchone() is None:
                raise ValueError(
                    "None of the `near` notes exist. Use slugs exactly as search hits or "
                    "open_note return them."
                )
    if len(query) > MAX_SEARCH_QUERY_CHARS:
        raise ValueError(
            f"query is {len(query):,} characters long; the limit is "
            f"{MAX_SEARCH_QUERY_CHARS:,}. Ask something shorter, or split it into two "
            "searches."
        )
    k = max(1, min(k, MAX_SEARCH_K))
    expand = max(0, min(expand, MAX_EXPAND))
    result = run_search(
        conn,
        embedder,
        query,
        k=k,
        expand=expand,
        types=types,
        since=since,
        entity_slugs=near or None,
    )
    hits = [
        {
            "note_slug": hit.note_slug,
            "title": hit.title,
            "score": hit.score,
            "via": hit.via,
            "best_chunk": hit.best_chunk,
            "heading": hit.heading,
            "updated": hit.updated.isoformat(),
            "content_date": hit.content_date.isoformat() if hit.content_date else None,
            "content_date_source": hit.content_date_source,
            "source_slugs": hit.source_slugs,
            "expanded_from": hit.expanded_from,
        }
        for hit in result.hits
    ]
    _log_usage(
        conn,
        tenant=tenant,
        tool="search",
        arg_sizes={
            "query_chars": len(query),
            "k": k,
            "expand": expand,
            "types_count": len(types) if types else 0,
            "since_given": since is not None,
            "near_count": len(near) if near else 0,
        },
        hit_count=len(hits),
    )
    return {
        "hits": hits,
        "control_ok": result.control_ok,
        "query_terms_unmatched": result.query_terms_unmatched,
        # Each hit's `content_date` only means something against a today.
        "today": clock_module.today().isoformat(),
    }


def tool_open_note(
    conn: psycopg.Connection,
    *,
    tenant: str,
    slug: str,
    claims_limit: int | None = None,
    link_details: bool = False,
) -> dict[str, Any] | None:
    """A note's frontmatter, body, claims and links. `None` if `slug`
    doesn't exist. See `server.py`'s `open_note` tool docstring.

    `link_details=True` (the door) adds each link's target `title`/`type`
    and `linked_from`, the notes linking here (`_link_details`).

    `claims_limit=None` (the default; the action researcher's evidence gate
    needs every claim id, `tests/test_mcp_door.py`) returns every claim,
    unpaged. An int returns only the first page of a fresh snapshot
    (`claims_page`) plus `claims_page_raw` (`total`, `has_more`,
    `next_after`, and the snapshot boundary), which the door turns into an
    opaque cursor for `list_claims`."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT slug, type, title, description, status, created, updated,
                   stale_after, schema_version, crm_id, parent_slug, superseded_by,
                   body_md, version
            FROM notes
            WHERE slug = %s
            """,
            (slug,),
        )
        row = cur.fetchone()

    result: dict[str, Any] | None = None
    if row is not None:
        (
            note_slug,
            note_type,
            title,
            description,
            status,
            created,
            updated,
            stale_after,
            schema_version,
            crm_id,
            parent_slug,
            superseded_by,
            body_md,
            version,
        ) = row
        result = {
            "slug": note_slug,
            "type": note_type,
            "title": title,
            "description": description,
            "status": status,
            "created": created.isoformat(),
            "updated": updated.isoformat(),
            "stale_after": stale_after.isoformat() if stale_after else None,
            "schema_version": schema_version,
            "crm_id": crm_id,
            "parent_slug": parent_slug,
            "superseded_by": superseded_by,
            "version": version,
            "body_md": body_md,
            "claims": [],
            "links": _load_links(conn, slug),
        }
        if link_details:
            result["links"], result["linked_from"] = _link_details(conn, slug, result["links"])
        if claims_limit is None:
            result["claims"] = _load_claims(conn, slug)
        else:
            max_id, taken_at = take_claims_snapshot(conn)
            page = claims_page(
                conn,
                slug=slug,
                limit=claims_limit,
                snapshot_max_id=max_id,
                snapshot_taken_at=taken_at,
            )
            result["claims"] = page["claims"]
            result["claims_page_raw"] = {
                "total": page["total"],
                "has_more": page["has_more"],
                "next_after": page["next_after"],
                "snapshot_max_claim_id": max_id,
                "snapshot_taken_at": taken_at,
            }

    _log_usage(
        conn,
        tenant=tenant,
        tool="open_note",
        arg_sizes={"slug_chars": len(slug)},
        hit_count=1 if result is not None else 0,
    )
    return result


def tool_open_source(
    conn: psycopg.Connection,
    store: Store,
    *,
    tenant: str,
    slug: str,
    hosted: bool = False,
    public_base_url: str | None = None,
) -> dict[str, Any] | None:
    """A source note's fields, extraction status, the notes extracted from
    it, and `original_url`. `None` if `slug` doesn't exist. See
    `server.py`'s `open_source` tool docstring.

    `hosted=False` (stdio) returns `Store.url(s3_key)`, useful to a client
    on the same machine. `hosted=True` (the streamable-HTTP transport,
    `create_server(..., hosted=True)`) returns the web app's
    `/originals/{slug}` viewer route instead, since a store path means
    nothing to a remote client: absolute under `public_base_url`
    (`PORTOLAN_PUBLIC_URL`) when set, else relative; `s3_key` is then `None`.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT slug, source_type, s3_key, content_hash, source_date, ingested_at,
                   extraction_status, filter_reason, schema_version, attempts,
                   last_error, summary, participants
            FROM sources
            WHERE slug = %s
            """,
            (slug,),
        )
        row = cur.fetchone()

    result: dict[str, Any] | None = None
    if row is not None:
        (
            source_slug,
            source_type,
            s3_key,
            content_hash,
            source_date,
            ingested_at,
            extraction_status,
            filter_reason,
            schema_version,
            attempts,
            last_error,
            summary,
            participants,
        ) = row
        with conn.cursor() as cur:
            cur.execute(
                "SELECT from_slug FROM links WHERE to_slug = %s AND kind = 'provenance' "
                "ORDER BY from_slug",
                (slug,),
            )
            notes_extracted = [r[0] for r in cur.fetchall()]

        original_url: str | None = None
        if s3_key:
            if not hosted:
                original_url = store.url(s3_key)
            else:
                # The slug is stored data, not ours to trust; escape everything, `/` included,
                # so `?`, `#` or `/` in it cannot change which source the URL points at.
                encoded_slug = urlquote(source_slug, safe="")
                if public_base_url:
                    original_url = f"{public_base_url.rstrip('/')}/originals/{encoded_slug}"
                else:
                    original_url = f"/originals/{encoded_slug}"

        # `last_error` is raw exception text, where a DSN or credentialed URL can surface;
        # redacted on every transport, since a tool result is at least as public as a log line.
        redacted_last_error = redact_dsn(last_error) if last_error else last_error

        result = {
            "slug": source_slug,
            "source_type": source_type,
            # Store layout is withheld from a remote client (`None` when hosted).
            "s3_key": s3_key if not hosted else None,
            "content_hash": content_hash,
            "source_date": source_date.isoformat() if source_date else None,
            "ingested_at": ingested_at.isoformat(),
            "extraction_status": extraction_status,
            "filter_reason": filter_reason,
            "schema_version": schema_version,
            "attempts": attempts,
            "last_error": redacted_last_error,
            "summary": summary,
            "participants": participants,
            "notes_extracted": notes_extracted,
            "original_url": original_url,
        }

    _log_usage(
        conn,
        tenant=tenant,
        tool="open_source",
        arg_sizes={"slug_chars": len(slug)},
        hit_count=1 if result is not None else 0,
    )
    return result


def tool_add_standing_question(
    conn: psycopg.Connection, *, tenant: str, reader: str, question: str
) -> dict[str, Any]:
    """Insert a standing question (migration 0003), return its row.

    `reader` and `question` raise past their bounds rather than truncating
    (see the `MAX_EXPAND` comment).
    """
    if len(reader) > MAX_STANDING_QUESTION_READER_CHARS:
        raise ValueError(
            f"reader is {len(reader):,} characters long; the limit is "
            f"{MAX_STANDING_QUESTION_READER_CHARS:,}."
        )
    if len(question) > MAX_STANDING_QUESTION_CHARS:
        raise ValueError(
            f"question is {len(question):,} characters long; the limit is "
            f"{MAX_STANDING_QUESTION_CHARS:,}. Ask something shorter."
        )
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO standing_questions (reader, question)
            VALUES (%s, %s)
            RETURNING id, reader, question, created, active
            """,
            (reader, question),
        )
        qid, row_reader, row_question, created, active = cur.fetchone()

    _log_usage(
        conn,
        tenant=tenant,
        tool="add_standing_question",
        arg_sizes={"reader_chars": len(reader), "question_chars": len(question)},
        hit_count=1,
    )
    return {
        "id": qid,
        "reader": row_reader,
        "question": row_question,
        "created": created.isoformat(),
        "active": active,
    }


def tool_list_standing_questions(
    conn: psycopg.Connection, *, tenant: str, reader: str | None = None
) -> list[dict[str, Any]]:
    """Standing questions, optionally filtered to one reader. Includes
    retired (`active = false`) questions."""
    with conn.cursor() as cur:
        if reader is not None:
            cur.execute(
                "SELECT id, reader, question, created, active FROM standing_questions "
                "WHERE reader = %s ORDER BY created",
                (reader,),
            )
        else:
            cur.execute(
                "SELECT id, reader, question, created, active FROM standing_questions "
                "ORDER BY created"
            )
        rows = cur.fetchall()

    result = [
        {
            "id": row[0],
            "reader": row[1],
            "question": row[2],
            "created": row[3].isoformat(),
            "active": row[4],
        }
        for row in rows
    ]
    _log_usage(
        conn,
        tenant=tenant,
        tool="list_standing_questions",
        arg_sizes={"reader_given": reader is not None},
        hit_count=len(result),
    )
    return result


def tool_coverage(
    conn: psycopg.Connection, *, tenant: str, since_days: int = 7
) -> list[dict[str, Any]]:
    """The coverage ledger summary (`portolan.ingest.ledger.summary`) for
    the last `since_days` days."""
    since_days = max(1, min(since_days, MAX_COVERAGE_SINCE_DAYS))
    since = date.today() - timedelta(days=since_days)
    rows = coverage_summary(conn, since)
    result = [{**row, "day": row["day"].isoformat()} for row in rows]
    _log_usage(
        conn,
        tenant=tenant,
        tool="coverage",
        arg_sizes={"since_days": since_days},
        hit_count=len(result),
    )
    return result


_NOTE_LINK_KINDS = ("body", "parent", "supersession")


def tool_graph_neighbors(
    conn: psycopg.Connection, *, tenant: str, slug: str, depth: int = 1
) -> list[dict[str, Any]]:
    """Notes linked to or from `slug` via `body`/`parent`/`supersession`
    links (not `provenance`, which points at a source, not a note --
    `open_note`/`open_source` already surface that), out to `depth` hops,
    breadth-first. Each result names the note it was reached from
    (`via`), the link `kind`, and its `direction` (`out`: `slug` links to
    it; `in`: it links to `slug`)."""
    depth = max(1, min(depth, MAX_GRAPH_DEPTH))
    seen = {slug}
    frontier = [slug]
    results: list[dict[str, Any]] = []

    for hop in range(1, depth + 1):
        next_frontier: list[str] = []
        for current in frontier:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT to_slug AS neighbor, kind, 'out' AS direction
                    FROM links
                    WHERE from_slug = %s AND kind = ANY(%s)
                    UNION ALL
                    SELECT from_slug AS neighbor, kind, 'in' AS direction
                    FROM links
                    WHERE to_slug = %s AND kind = ANY(%s)
                    """,
                    (current, list(_NOTE_LINK_KINDS), current, list(_NOTE_LINK_KINDS)),
                )
                edges = cur.fetchall()
            for neighbor_slug, kind, direction in edges:
                if neighbor_slug in seen:
                    continue
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT title, type, updated FROM notes WHERE slug = %s", (neighbor_slug,)
                    )
                    note_row = cur.fetchone()
                if note_row is None:
                    # A dangling link: skip it; link integrity is the lint's job, not this tool's.
                    continue
                seen.add(neighbor_slug)
                title, note_type, updated = note_row
                results.append(
                    {
                        "slug": neighbor_slug,
                        "title": title,
                        "type": note_type,
                        "updated": updated.isoformat(),
                        "depth": hop,
                        "kind": kind,
                        "direction": direction,
                        "via": current,
                    }
                )
                next_frontier.append(neighbor_slug)
        frontier = next_frontier
        if not frontier:
            break

    _log_usage(
        conn,
        tenant=tenant,
        tool="graph_neighbors",
        arg_sizes={"slug_chars": len(slug), "depth": depth},
        hit_count=len(results),
    )
    return results


# ---------------------------------------------------------------------
# precedents (01-spec/actions-and-owners-2026-09-19.md Phase 2): the
# commitments and decisions that followed a theme before, and how long
# until it resolved. Also an MCP tool (`server.precedents`).
# ---------------------------------------------------------------------


def tool_precedents(
    conn: psycopg.Connection,
    *,
    tenant: str,
    theme_slug: str,
    account_slug: str | None = None,
) -> dict[str, Any]:
    """Every unarchived `commitment`/`decision` claim that has ever joined
    `theme_slug` (via `theme_members`, oldest first), each with who said
    it, when (world date), which account it belongs to
    (`portolan.report.accounts.home_account_slug`), and -- when a later
    confirmed resolution exists for this theme (`theme_resolutions`,
    `status = 'confirmed'`) -- how many days elapsed between that claim
    and the resolution week that followed it. `None` for a precedent
    with no resolution yet (the theme is still open, or resolved before
    this claim was ever made).

    `account_slug` narrows nothing (another account's precedent under the
    same theme is still useful history); it is echoed back, and each
    precedent carries a `same_account` flag. Returns `{"theme_slug",
    "account_slug", "precedents": [...]}`; `precedents` is `[]` for a theme
    with no unarchived commitment/decision members.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT resolved_week, confirmed_at FROM theme_resolutions "
            "WHERE theme_slug = %s AND status = 'confirmed' "
            "ORDER BY resolved_week, id",
            (theme_slug,),
        )
        resolutions = cur.fetchall()

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT c.id, c.fact_type, c.claim_text, c.speaker, c.recorded,
                   c.source_slug, c.locator, c.note_slug
            FROM theme_members tm
            JOIN claims c ON c.id = tm.claim_id
            WHERE tm.theme_slug = %s
              AND c.fact_type IN ('commitment', 'decision')
              AND c.archived = false
            ORDER BY c.recorded
            """,
            (theme_slug,),
        )
        rows = cur.fetchall()

    precedents: list[dict[str, Any]] = []
    for claim_id, fact_type, claim_text, speaker, recorded, source_slug, locator, note_slug in rows:
        claim_world_date = clock_module.world_date(recorded)
        this_account = home_account_slug(
            conn, note_slug=note_slug, fact_type=fact_type, claim_text=claim_text
        )
        resolved_after_days: int | None = None
        resolved_week_reached: str | None = None
        for resolved_week, _confirmed_at in resolutions:
            if resolved_week >= claim_world_date:
                resolved_after_days = (resolved_week - claim_world_date).days
                resolved_week_reached = resolved_week.isoformat()
                break
        precedents.append(
            {
                "claim_id": claim_id,
                "fact_type": fact_type,
                "claim_text": claim_text,
                "who": speaker,
                "when": claim_world_date.isoformat(),
                "account_slug": this_account,
                "same_account": (
                    this_account == account_slug if account_slug is not None else None
                ),
                "source_slug": source_slug,
                "locator": locator,
                "resolved_after_days": resolved_after_days,
                "resolved_week": resolved_week_reached,
            }
        )

    _log_usage(
        conn,
        tenant=tenant,
        tool="precedents",
        arg_sizes={
            "theme_slug_chars": len(theme_slug),
            "account_slug_given": account_slug is not None,
        },
        hit_count=len(precedents),
    )
    return {
        "theme_slug": theme_slug,
        "account_slug": account_slug,
        "precedents": precedents,
    }
