-- Portolan schema, migration 0006: review wave A fixes (F5, F6).
--
-- 01-spec/reviews/review-wave-a-2026-09-12.md F6: "sources.content_hash has
-- no unique index (nor any index -- _find_by_hash is a seq scan on every
-- intake)." The unique index is what portolan.ingest.intake's `INSERT ...
-- ON CONFLICT (content_hash) DO NOTHING RETURNING` idempotency actually
-- rests on -- without it, two concurrent first-writes for the same content
-- both insert. Partial (`WHERE content_hash IS NOT NULL`) because a source
-- with no computed hash (there is no such path today, but nothing in the
-- `sources` schema forbids a NULL) must not collide with every other NULL
-- row under a unique constraint.
--
-- `metadata` is new too: F6's other half is "a source whose content changed
-- but whose source_id did not" (a ticket that gained a comment, an edited
-- doc) -- `intake` now mints that source a new slug, `<slug>--v<n>`, and
-- records what it superseded at `metadata.supersedes_source`
-- (provenance-chain.md documents the versioning scheme). `sources` had no
-- general-purpose jsonb column to hold that until now.
--
-- IDEMPOTENT: IF NOT EXISTS / ADD COLUMN IF NOT EXISTS throughout, safe to
-- re-apply -- and see the test that applies every migration file twice
-- (tests/test_db_schema.py).

CREATE UNIQUE INDEX IF NOT EXISTS sources_content_hash_uq
    ON sources (content_hash)
    WHERE content_hash IS NOT NULL;

ALTER TABLE sources ADD COLUMN IF NOT EXISTS metadata jsonb NOT NULL DEFAULT '{}'::jsonb;

COMMENT ON COLUMN sources.metadata IS
    'Small free-form bookkeeping the sources row itself needs (today: '
    'supersedes_source, the prior slug a same-source_id/changed-content '
    're-ingest versioned past -- see portolan.ingest.intake). Not a place '
    'for extraction output; that lives in claims/notes.';
