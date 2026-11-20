-- Portolan schema, migration 0022: idempotency keys on (source_id,
-- content_hash), not content_hash alone.
--
-- Bug hunt 2026-09-18, finding 1 (P0): `portolan.ingest.intake._find_by_
-- hash` looked up an existing `sources` row by `content_hash` alone, and
-- `sources_content_hash_uq` (migration 0006) enforced that same
-- single-column uniqueness in the database. Two DIFFERENT sources whose
-- bytes happen to collide -- two Zendesk tickets auto-created from the
-- same alert text, the same RSS item mirrored into two feeds, a CRM row
-- exported under two ids -- collapsed into one `sources` row: the second
-- source never got its own row or stored original, and its facts were
-- filed under the FIRST source's slug and date.
--
-- `source_id` (new column) is the connector's own raw identity
-- (`RawSource.source_id`, portolan.ingest.intake) -- the un-versioned
-- base of `slug`, which `intake()` may suffix `--v<n>` on a same-
-- source_id, changed-content re-crawl (provenance-chain.md "Source
-- versioning"). Idempotency now keys on `(source_id, content_hash)`:
-- re-ingesting the SAME source with unchanged bytes is still a no-op
-- (same source_id, same hash -> existing row found); two DIFFERENT
-- sources with identical bytes now each get their own row, their own
-- stored original (`portolan.ingest.intake.original_key` already keys
-- off the resolved `slug`, which stays distinct per source_id), and
-- their own facts.
--
-- Backfill: every existing row's `source_id` is recovered from its own
-- `slug` by stripping a trailing `--v<n>` version suffix -- exactly the
-- inverse of how `intake()` minted that slug in the first place, so a
-- row written before this migration existed backfills to the same
-- `source_id` a fresh `intake()` call would already compute for it.
--
-- IDEMPOTENT: IF NOT EXISTS / IF EXISTS throughout, safe to re-apply (see
-- tests/test_db_schema.py, which applies every migration file twice).

ALTER TABLE sources ADD COLUMN IF NOT EXISTS source_id text;

UPDATE sources
   SET source_id = regexp_replace(slug, '--v[0-9]+$', '')
 WHERE source_id IS NULL;

DROP INDEX IF EXISTS sources_content_hash_uq;

CREATE UNIQUE INDEX IF NOT EXISTS sources_source_id_content_hash_uq
    ON sources (source_id, content_hash)
    WHERE content_hash IS NOT NULL AND source_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_sources_source_id
    ON sources (source_id);

COMMENT ON COLUMN sources.source_id IS
    'The connector''s own raw source identity (RawSource.source_id) -- '
    'the un-versioned base of slug. Idempotency (portolan.ingest.intake) '
    'keys on (source_id, content_hash), not content_hash alone, so two '
    'different sources with identical bytes never collapse into one row.';
