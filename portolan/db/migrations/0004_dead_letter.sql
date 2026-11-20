-- Portolan schema, migration 0004: pipeline dead-letter queue
--
-- Owned by wave B2 (portolan/pipeline.py). Spec references:
-- 01-spec/portolan-spec-v2.md sec 6.2 ("Failed steps retry three times,
-- then land in a dead-letter queue a daily job re-drives");
-- 01-spec/build-contracts.md "Pipeline runner" / migration ledger.
--
-- dead_letter: one row per (source, stage) that failed all three retries
-- during a pipeline run. `attempt` is the number of attempts made before
-- giving up (always 3 under portolan.pipeline's current retry policy,
-- kept as a real column rather than a hardcoded assumption so a future
-- change to the retry count doesn't silently invalidate old rows).
-- `payload` is jsonb, not a fixed column set, so a future stage can carry
-- whatever extra context it needs to redrive without another migration.
--
-- NOTHING IS HARD-DELETED (spec sec 8, and 0001_init.sql's own framing):
-- `portolan.pipeline.redrive()` sets `redriven_at`, it does not delete the
-- row -- a successfully-redriven row still answers "what used to fail and
-- when did it get fixed." The pending queue a redrive job walks is simply
-- `WHERE redriven_at IS NULL`.
--
-- No foreign key to `sources`: a source can fail at the very first stage
-- (fetch/intake), before any `sources` row exists to reference, and this
-- table still needs to record that failure.
--
-- IDEMPOTENT, like every other migration here: IF NOT EXISTS throughout.

CREATE TABLE IF NOT EXISTS dead_letter (
    id           bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    source_id    text NOT NULL,
    stage        text NOT NULL,
    attempt      integer NOT NULL,
    error        text NOT NULL,
    payload      jsonb NOT NULL DEFAULT '{}'::jsonb,
    created      timestamptz NOT NULL DEFAULT now(),
    redriven_at  timestamptz
);

COMMENT ON TABLE dead_letter IS
    'A source that failed a pipeline stage after three retries (spec sec '
    '6.2). redriven_at is set, not the row deleted, when '
    'portolan.pipeline.redrive() successfully reprocesses the source.';

CREATE INDEX IF NOT EXISTS idx_dead_letter_source_id ON dead_letter (source_id);
CREATE INDEX IF NOT EXISTS idx_dead_letter_pending
    ON dead_letter (created) WHERE redriven_at IS NULL;
