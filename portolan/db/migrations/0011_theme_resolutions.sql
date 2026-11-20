-- Portolan schema, migration 0011: theme resolution tracking.
--
-- Owned by the theme identity/resolution wave (2026-09-13,
-- portolan/report/themes.py, build task item 2). Spec references:
-- 01-spec/surfacing-and-weekly-report.md "Theme notes: the rollup layer",
-- 01-spec/weekly-report-eval-2026-09-13.md's top defect ("no theme ever
-- reaches 'resolved'").
--
-- IDEMPOTENT, like every prior migration file: IF NOT EXISTS throughout,
-- safe to re-apply (tests/test_db_schema.py applies every migration file
-- twice).

-- ---------------------------------------------------------------------
-- theme_resolutions: the one durable fact "this theme is resolved, as of
-- this week, for this reason" -- written once per theme (sticky: a claim-
-- based resolution is never overwritten by a later run, ON CONFLICT DO
-- NOTHING below and in portolan/report/themes.py's own insert).
--
-- Only the CLAIM-based resolution path (task item 2(a): "a positive or
-- resolution-type claim ... matches the theme's sub-topic") writes a row
-- here -- it is the one path that names a specific claim as evidence, so
-- it is the one path with something to persist and change-log. The
-- QUIET-weeks path (item 2(b): "no new member for THEME_RESOLVED_AFTER_
-- WEEKS weeks") has no single claim to point at; it is recomputed live,
-- every call, straight off theme_members' own weekly counts
-- (`portolan.report.themes.theme_trend`) -- nothing to store here.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS theme_resolutions (
    theme_slug           text PRIMARY KEY REFERENCES notes (slug) ON DELETE RESTRICT,
    resolution_claim_id  bigint NOT NULL REFERENCES claims (id) ON DELETE RESTRICT,
    resolved_reason      text NOT NULL,
    resolved_week        date NOT NULL,
    created              timestamptz NOT NULL DEFAULT now()
);

COMMENT ON TABLE theme_resolutions IS
    'One row per theme resolved by an explicit resolution claim (task item '
    '2(a)) -- resolved_week is the Monday of the ISO week that claim''s own '
    'world date falls in; theme_trend() treats every week from there '
    'onward as "resolved". Sticky: never overwritten once written.';

CREATE INDEX IF NOT EXISTS idx_theme_resolutions_resolution_claim_id
    ON theme_resolutions (resolution_claim_id);
