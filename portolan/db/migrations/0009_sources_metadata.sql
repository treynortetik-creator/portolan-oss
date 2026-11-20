-- Portolan schema, migration 0009: sources.metadata gains the connector's
-- own RawSource.metadata (polish wave 2026-09-12).
--
-- 0006_review_fixes.sql added `sources.metadata jsonb` for exactly one
-- purpose: `supersedes_source`, a versioned re-ingest's prior slug. Its own
-- comment said as much ("Not a place for extraction output; that lives in
-- claims/notes"), and as a side effect said nothing about the connector's
-- OWN metadata -- `RawSource.metadata` (a ticket's structured comments, an
-- email's Message-ID/Subject/From, a CRM row's object/id, an RSS item's
-- link). `portolan.ingest.intake.intake` never stored it, so
-- `portolan.audit.second_pass._reconstruct_raw_source` had to re-derive it
-- from the stored original bytes on every audit re-read: exactly right for
-- call/doc (the stored bytes ARE `raw.text`), silently WRONG for
-- ticket/crm/rss, whose `RawSource.text` is a *rendering* the connector
-- builds from structured data it no longer had once intake threw the
-- metadata away (documented as a known limitation in second_pass.py,
-- flagged there for "whichever wave next touches portolan/ingest" -- this
-- one).
--
-- No column to add -- `sources.metadata` is already a general-purpose
-- jsonb. This migration is the schema *decision*, recorded where the
-- ledger expects it (01-spec/build-contracts.md): `sources.metadata` now
-- holds up to two top-level keys --
--   - `supersedes_source` (unchanged): the prior slug a same-source_id/
--     changed-content re-ingest versioned past.
--   - `raw`: the connector's own `RawSource.metadata` dict, stored
--     verbatim by `portolan.ingest.intake.intake` -- nested under its own
--     key, never merged flat, so a connector's own metadata keys can never
--     collide with `supersedes_source` or a future bookkeeping key.
--
-- IDEMPOTENT: a bare COMMENT, safe to re-apply (tests/test_db_schema.py
-- applies every migration file twice).

COMMENT ON COLUMN sources.metadata IS
    'Bookkeeping the sources row itself needs, up to two top-level keys: '
    'supersedes_source (the prior slug a same-source_id/changed-content '
    're-ingest versioned past -- portolan.ingest.intake) and raw (the '
    'connector''s own RawSource.metadata dict, stored verbatim since '
    'migration 0009 so portolan.audit.second_pass can reconstruct a '
    'RawSource losslessly instead of re-deriving one from the stored '
    'original bytes). Still not a place for extraction output; that lives '
    'in claims/notes.';
