"""Chunk + embed a note's body into `chunks`, keyed off a content hash.

Spec: 01-spec/portolan-spec-v2.md sec 9. "What needs re-embedding" keys off
`note_index_state.body_hash`, the sha256 of `body_md`
(0002_index_state.sql), never a timestamp. `write_note` renders the title
and description into `body_md`'s frontmatter, so a change to either
re-indexes the title chunk too (`tests/test_search_index.py`).

`reindex_all` is a full pass behind a `pg_try_advisory_lock` single-flight
guard. `index_pending` is the incremental path: it finds changed-or-new
notes in SQL, so it never pulls every body into Python.
"""

from __future__ import annotations

import hashlib

import psycopg
from pydantic import BaseModel

from portolan.search.chunk import chunk_note
from portolan.search.embed import Embedder

# Arbitrary fixed key for reindex_all's single-flight advisory lock.
REINDEX_ALL_ADVISORY_LOCK_KEY = 785_412_309


class IndexReport(BaseModel):
    notes_considered: int
    notes_reindexed: int
    notes_skipped: int
    chunks_written: int
    skipped_lock: bool = False


def _body_hash(body_md: str) -> str:
    return hashlib.sha256(body_md.encode("utf-8")).hexdigest()


def _stored_hash(conn: psycopg.Connection, note_slug: str) -> str | None:
    with conn.cursor() as cur:
        cur.execute("SELECT body_hash FROM note_index_state WHERE note_slug = %s", (note_slug,))
        row = cur.fetchone()
    return row[0] if row else None


def index_note(conn: psycopg.Connection, embedder: Embedder, note_slug: str) -> int:
    """Re-chunk and re-embed `note_slug` if (and only if) its `body_md`
    hash has changed since the last successful index. Returns the number
    of chunks written -- 0 when the body is unchanged, so calling this
    twice in a row with no edit in between is a no-op the second time.

    Deletes and rewrites the note's `chunks` rows in one transaction, so a
    reader never sees a note with half its old chunks and half its new
    ones.
    """
    with conn.cursor() as cur:
        cur.execute("SELECT title, description, body_md FROM notes WHERE slug = %s", (note_slug,))
        row = cur.fetchone()
    if row is None:
        raise ValueError(f"No such note: {note_slug!r}")
    title, description, body_md = row
    body_md = body_md or ""
    new_hash = _body_hash(body_md)

    if _stored_hash(conn, note_slug) == new_hash:
        return 0

    chunks = chunk_note(body_md, title=title, description=description)
    texts = [c.text for c in chunks]
    embeddings = embedder.embed(texts) if texts else []

    with conn.transaction(), conn.cursor() as cur:
        cur.execute("DELETE FROM chunks WHERE note_slug = %s", (note_slug,))
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            cur.execute(
                """
                    INSERT INTO chunks (note_slug, heading, text, token_count, embedding)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                (note_slug, chunk.heading, chunk.text, chunk.token_estimate, embedding),
            )
        cur.execute(
            """
                INSERT INTO note_index_state (note_slug, body_hash, indexed_at)
                VALUES (%s, %s, now())
                ON CONFLICT (note_slug) DO UPDATE
                    SET body_hash = EXCLUDED.body_hash, indexed_at = EXCLUDED.indexed_at
                """,
            (note_slug, new_hash),
        )

    return len(chunks)


def reindex_all(conn: psycopg.Connection, embedder: Embedder) -> IndexReport:
    """Walk every note and call `index_note` on it (which still skips a
    note whose hash hasn't changed). Single-flight via `pg_try_advisory_lock`:
    if another reindex is already running against this database, returns
    immediately with `skipped_lock=True` instead of racing it.
    """
    with conn.cursor() as cur:
        cur.execute("SELECT pg_try_advisory_lock(%s)", (REINDEX_ALL_ADVISORY_LOCK_KEY,))
        got_lock = cur.fetchone()[0]

    if not got_lock:
        return IndexReport(
            notes_considered=0,
            notes_reindexed=0,
            notes_skipped=0,
            chunks_written=0,
            skipped_lock=True,
        )

    try:
        with conn.cursor() as cur:
            cur.execute("SELECT slug FROM notes ORDER BY slug")
            slugs = [row[0] for row in cur.fetchall()]

        reindexed = 0
        chunks_written = 0
        for slug in slugs:
            written = index_note(conn, embedder, slug)
            if written:
                reindexed += 1
                chunks_written += written

        return IndexReport(
            notes_considered=len(slugs),
            notes_reindexed=reindexed,
            notes_skipped=len(slugs) - reindexed,
            chunks_written=chunks_written,
        )
    finally:
        with conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_unlock(%s)", (REINDEX_ALL_ADVISORY_LOCK_KEY,))


def index_pending(conn: psycopg.Connection, embedder: Embedder) -> IndexReport:
    """Index only notes whose `body_md` hash differs from `note_index_state`
    (or that have no row there yet) -- the incremental path a scheduled
    re-crawl uses. The changed/new filter runs in SQL (built-in `sha256()`,
    no pgcrypto). No advisory lock: this is meant to be cheap and callable
    often; `reindex_all` is the exclusive, whole-graph operation.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT n.slug
            FROM notes n
            LEFT JOIN note_index_state s ON s.note_slug = n.slug
            WHERE s.note_slug IS NULL
               OR s.body_hash IS DISTINCT FROM encode(
                      sha256(convert_to(coalesce(n.body_md, ''), 'UTF8')), 'hex'
                  )
            ORDER BY n.slug
            """
        )
        slugs = [row[0] for row in cur.fetchall()]

    reindexed = 0
    chunks_written = 0
    for slug in slugs:
        written = index_note(conn, embedder, slug)
        if written:
            reindexed += 1
            chunks_written += written

    return IndexReport(
        notes_considered=len(slugs),
        notes_reindexed=reindexed,
        notes_skipped=len(slugs) - reindexed,
        chunks_written=chunks_written,
    )
