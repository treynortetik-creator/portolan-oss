-- Portolan schema, migration 0008: theme_members' second clock.
--
-- Owned by the fix wave 2026-09-12 (portolan/report/**). Review finding G8
-- (01-spec/reviews/review-wave-bc-graph-2026-09-12.md):
-- `theme_members.added` is documented (0005_themes_reports.sql) and used
-- as the claim's own WORLD date (valid_from, or else recorded's date) --
-- right for "first seen"/"last seen" (when did this story start/end out
-- in the world), wrong for TREND ("is this theme actively gaining
-- members right now"). A claim attached to a theme today but dated six
-- weeks ago landed in a six-week-old weekly bucket, so a theme actively
-- gaining members this week could read as quiet.
--
-- `attached_on` is the second clock: the wall-clock day the refresh pass
-- actually attached this claim (`portolan.report.themes.refresh_themes`'s
-- own `as_of`, not `date.today()` literally, so a backfill run over a
-- historical window still dates its attachments to the window it's
-- replaying rather than the day the backfill happened to execute).
-- `theme_weekly_counts`/`theme_trend` now read this column for trend;
-- `added` keeps its original meaning and is still what `MIN`/`MAX` reads
-- for first-seen/last-seen.
--
-- Backfilled from `added` for any row written before this migration
-- (there is no better information available for those rows, and this
-- makes the column non-nullable without a separate "unknown" state).
--
-- IDEMPOTENT: IF NOT EXISTS / ADD COLUMN IF NOT EXISTS throughout, safe to
-- re-apply (tests/test_db_schema.py applies every migration file twice).

ALTER TABLE theme_members ADD COLUMN IF NOT EXISTS attached_on date;

UPDATE theme_members SET attached_on = added WHERE attached_on IS NULL;

ALTER TABLE theme_members ALTER COLUMN attached_on SET NOT NULL;
ALTER TABLE theme_members ALTER COLUMN attached_on SET DEFAULT CURRENT_DATE;

COMMENT ON COLUMN theme_members.attached_on IS
    'The wall-clock day this pass actually attached the claim (review '
    'finding G8) -- what theme_trend buckets by. added (unchanged) stays '
    'the claim''s own world date, for first_seen/last_seen.';

CREATE INDEX IF NOT EXISTS idx_theme_members_attached_on ON theme_members (attached_on);
