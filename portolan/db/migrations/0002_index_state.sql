-- Portolan schema, migration 0002: index state
--
-- Owned by wave A4 (portolan/search/**). Spec references:
-- 01-spec/portolan-spec-v2.md sec 9 (search/index); spec-review-2026-09-12.md
-- item 3 ("Recency and incremental indexing key on file mtime, which git
-- destroys" -- fix: diff on content hash, not mtime).
--
-- note_index_state: one row per note, tracking the sha256 of the body_md
-- that was last chunked+embedded into `chunks`. portolan.search.index_note
-- re-chunks and re-embeds a note only when its current body_md hash
-- differs from what's stored here (or there's no row yet) -- this is the
-- "diff on content hash" fix, and it is what makes index_note idempotent:
-- calling it twice in a row with no body change writes zero chunks the
-- second time.
--
-- IDEMPOTENT, like 0001_init.sql: IF NOT EXISTS throughout.

CREATE TABLE IF NOT EXISTS note_index_state (
    note_slug   text PRIMARY KEY REFERENCES notes (slug) ON DELETE RESTRICT,
    body_hash   text NOT NULL,
    indexed_at  timestamptz NOT NULL DEFAULT now()
);

COMMENT ON TABLE note_index_state IS
    'sha256(body_md) as of the last successful index_note() run, per note '
    '(spec-review-2026-09-12 item 3: diff on content hash, never mtime). '
    'A note absent from this table has never been indexed.';
