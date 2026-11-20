-- Portolan schema, migration 0013: theme_resolutions becomes a history.
--
-- Owned by the theme review fix wave (2026-09-13, portolan/report/
-- themes.py). Review reference: 01-spec/reviews/review-theme-fix-wave.md,
-- should-fix 5 ("the merge hard-deletes the superseded resolution row")
-- and should-fix 6 ("a theme can reopen exactly once, and a reopened
-- theme can never be claim-resolved again").
--
-- IDEMPOTENT, like every prior migration file: IF NOT EXISTS throughout
-- (and a pg_constraint guard for the one ADD CONSTRAINT, which has no
-- native IF NOT EXISTS form -- see 0001_init.sql's own note on this),
-- safe to re-apply (tests/test_db_schema.py applies every migration file
-- twice).

-- ---------------------------------------------------------------------
-- Before this migration, theme_slug was theme_resolutions' PRIMARY KEY:
-- one row per theme, EVER. That made two things true that shouldn't be:
--
-- 1. A merge that had to reconcile two themes' resolution rows under one
--    merged slug had no choice but to DELETE the superseded side outright
--    (portolan/report/themes.py's old `_carry_theme_resolution`) --
--    against spec sec 8's "nothing is hard-deleted", and the one path in
--    that wave that actually destroyed history rather than annotating it.
-- 2. Once a theme reopened (reopened_week set), no LATER resolution claim
--    could ever resolve it again -- `_check_theme_resolutions` skips any
--    theme with an existing row, forever, and there was only ever one row
--    to have. "Late deliveries came back, and then we fixed them again"
--    was unrepresentable.
--
-- The fix: theme_resolutions is now a HISTORY of resolution events, one
-- row per (theme, resolved_week) rather than one row per theme for its
-- whole life. A synthetic `id` becomes the primary key; a PARTIAL unique
-- index enforces the one invariant that still needs enforcing -- at most
-- one ACTIVE (not yet reopened) resolution per theme at a time, which is
-- what `theme_slug` being a bare PRIMARY KEY used to guarantee as a side
-- effect. `_get_theme_resolution` (the "is this theme currently
-- resolved" question every other function asks) now means "the active
-- row, if any"; a full history read (`_theme_resolutions_history`) is
-- what `theme_trend` walks to classify a past week that fell inside a
-- since-reopened resolution interval.
-- ---------------------------------------------------------------------

-- A manually-built identity sequence rather than `bigserial`/`GENERATED
-- ALWAYS AS IDENTITY` in a bare ADD COLUMN: this is the idiom that is
-- guaranteed correct whether the table already holds rows (a live
-- deployment) or none (a fresh test database) -- the column's DEFAULT
-- backfills every existing row from the sequence when it's added.
--
-- Nit 18 (`01-spec/reviews/review-theme-fix-wave-2.md` finding 18, said
-- plainly here since deploy safety is this project's stated top risk):
-- `nextval()` is VOLATILE, so Postgres cannot take the fast path an
-- added column with a constant default gets (populate the catalog only,
-- no per-row rewrite) -- this ADD COLUMN rewrites the WHOLE table under
-- an ACCESS EXCLUSIVE lock, blocking every reader and writer of
-- `theme_resolutions` for the duration. Correct (verified: the backfill
-- produces distinct ids on a populated table) and harmless at this
-- table's size (one row per resolution EVENT, not per claim), but on a
-- live deployment with a large `theme_resolutions` table this migration
-- should run in a maintenance window, not assumed instant like every
-- other `ADD COLUMN ... DEFAULT <constant>` in this codebase's other
-- migrations.
CREATE SEQUENCE IF NOT EXISTS theme_resolutions_id_seq;
ALTER TABLE theme_resolutions
    ADD COLUMN IF NOT EXISTS id bigint NOT NULL DEFAULT nextval('theme_resolutions_id_seq');
ALTER SEQUENCE theme_resolutions_id_seq OWNED BY theme_resolutions.id;

DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'theme_resolutions_pkey'
          AND conrelid = 'theme_resolutions'::regclass
    ) THEN
        ALTER TABLE theme_resolutions DROP CONSTRAINT theme_resolutions_pkey;
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'theme_resolutions_id_pkey' AND conrelid = 'theme_resolutions'::regclass
    ) THEN
        ALTER TABLE theme_resolutions ADD CONSTRAINT theme_resolutions_id_pkey PRIMARY KEY (id);
    END IF;
END $$;

-- At most one ACTIVE (reopened_week IS NULL) resolution per theme_slug --
-- the one invariant `theme_slug PRIMARY KEY` used to enforce as a side
-- effect, now stated directly so multiple HISTORICAL (reopened) rows can
-- coexist under the same theme_slug.
CREATE UNIQUE INDEX IF NOT EXISTS uq_theme_resolutions_active_theme
    ON theme_resolutions (theme_slug) WHERE reopened_week IS NULL;

CREATE INDEX IF NOT EXISTS idx_theme_resolutions_theme_slug
    ON theme_resolutions (theme_slug);

COMMENT ON TABLE theme_resolutions IS
    'History of resolution events, one row per (theme, resolved_week) --'
    ' NOT one row per theme for its whole life (that was migration 0011''s'
    ' shape; migration 0013 widened it). uq_theme_resolutions_active_theme'
    ' keeps at most one ACTIVE (reopened_week IS NULL) row per theme_slug'
    ' at a time; a theme that reopens and is claim-resolved again gets a'
    ' SECOND row, not an overwrite of the first. Never hard-deleted: a'
    ' merge that must reconcile two active rows under one slug closes the'
    ' later one out (sets its reopened_week) rather than dropping it.';

-- ---------------------------------------------------------------------
-- Should-fix 4 (review finding 4, "give the row a real action"):
-- `review_queue.kind` widened to accept `'theme_resolution'` -- the
-- pending-confirmation row a first-ever claim-based resolution queues
-- (`portolan.report.themes._queue_resolution_for_confirmation`, task
-- item 2(a) recommendation 5) used to be inserted as `kind='other'`,
-- indistinguishable from every other advisory row and reachable only by
-- `/review`'s generic resolve/dismiss, which changed the QUEUE row's
-- status and nothing about the theme itself. A `theme_resolution` row's
-- own dismiss action (`portolan.report.themes.reject_theme_resolution`,
-- wired through `portolan.web.routes.review`) now un-resolves the theme
-- for real, the same mechanism `_mark_theme_reopened` already uses for
-- an organic reopening.
-- ---------------------------------------------------------------------
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'review_queue_kind_check' AND conrelid = 'review_queue'::regclass
    ) THEN
        ALTER TABLE review_queue DROP CONSTRAINT review_queue_kind_check;
    END IF;
    ALTER TABLE review_queue ADD CONSTRAINT review_queue_kind_check CHECK (
        kind IN ('unknown_entity', 'ambiguous_entity', 'failed_consolidation',
                 'theme_resolution', 'other')
    );
END $$;
