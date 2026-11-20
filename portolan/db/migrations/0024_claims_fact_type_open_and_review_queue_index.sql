-- Portolan schema, migration 0022: open up claims.fact_type for customer
-- extensions; index review_queue for the dedup lookup.
--
-- Bug hunt 2026-09-18 (01-spec/reviews/bughunt-2026-09-18-merge.md),
-- finding 2: `claims.fact_type` (migration 0001) carried a CHECK against
-- the closed 12-name base list, but extraction-schema-v0.md's customer
-- extension (`CustomerExtension.extra_types`,
-- `portolan.extract.schema.ExtraFactType`) is a real, paid onboarding
-- feature that adds per-customer, open-ended type names (e.g.
-- `clinical_incident`) -- `extra_types` is not a fixed set this
-- migration, or any single migration, could enumerate once for every
-- future customer. `Fact._validate_against_effective_schema`
-- (portolan/extract/schema.py) already validates a fact's `type` in
-- Python against the effective schema (base types plus whichever
-- extension is in play for that source's customer) before a claim is
-- ever built -- so the DB-level CHECK was never the only guard, just a
-- second, incompatible one that could only ever agree with Python for
-- the base 12. Dropped in favour of the Python-side validation that
-- already does the real job; `fact_type` stays NOT NULL.
--
-- Finding 9: `review_queue` (migration 0001) indexed only `status`, but
-- `registry.queue_unknown`/`merge.resolve._queue_ambiguous` each run a
-- `WHERE kind = ... AND source_slug = ... AND payload ->> 'mention' = ...`
-- dedup lookup once per unresolved/ambiguous mention per fact -- a
-- sequential scan on every call. `(kind, source_slug)` narrows that scan
-- to the handful of rows for one source before the JSON payload
-- comparison runs.
--
-- Both idempotent: ALTER TABLE ... DROP CONSTRAINT IF EXISTS and
-- CREATE INDEX IF NOT EXISTS are safe to re-run.

ALTER TABLE claims DROP CONSTRAINT IF EXISTS claims_fact_type_check;

COMMENT ON COLUMN claims.fact_type IS
    'The signal type a fact was extracted as: the closed base list from '
    'extraction-schema-v0.md, or a customer extension''s extra_types name. '
    'No DB-level CHECK (migration 0022) -- validated in Python against the '
    'effective schema at extraction time (portolan.extract.schema.Fact).';

CREATE INDEX IF NOT EXISTS idx_review_queue_kind_source
    ON review_queue (kind, source_slug);
