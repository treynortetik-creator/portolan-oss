-- Portolan schema, migration 0012: theme resolution reopening.
--
-- Owned by the theme identity/resolution wave code review fix
-- (2026-09-13, portolan/report/themes.py). Review reference:
-- 01-spec/reviews/review-theme-identity-wave.md, blocker 1 / "Resolution
-- rule" section item 3 ("Make it reopen ... have `theme_trend` reopen a
-- theme when >= THEME_REOPEN_MIN_MEMBERS new members arrive in weeks
-- after `resolved_week`, and write a `theme_reopened` changelog row").
--
-- IDEMPOTENT, like every prior migration file: IF NOT EXISTS throughout,
-- safe to re-apply (tests/test_db_schema.py applies every migration file
-- twice).

-- ---------------------------------------------------------------------
-- theme_resolutions.reopened_week: NULL means "never reopened" (every
-- existing row, and every newly-resolved theme, starts this way) -- a
-- date means "the ISO week Monday in which renewed activity crossed
-- THEME_REOPEN_MIN_MEMBERS new members after resolved_week", from which
-- point `theme_trend()` stops reading the theme as `resolved` and resumes
-- ordinary rising/peak/falling classification, exactly as if a fresh arc
-- had started there. `theme_resolutions` itself is never deleted or
-- overwritten on reopen (task instruction: "keep theme_resolutions as
-- history") -- reopening is this one extra column, not a new row, so the
-- original resolution claim/reason/week stay queryable indefinitely
-- alongside whether (and when) the theme went back to being a live story.
-- ---------------------------------------------------------------------
ALTER TABLE theme_resolutions ADD COLUMN IF NOT EXISTS reopened_week date;

COMMENT ON COLUMN theme_resolutions.reopened_week IS
    'NULL until renewed activity (>= THEME_REOPEN_MIN_MEMBERS new members '
    'after resolved_week) reopens the theme; then the ISO week Monday that '
    'crossed the threshold. theme_trend() treats every week from there on '
    'as ordinary (non-resolved) movement again -- a theme can still reach '
    '"resolved" a second time via the unpersisted quiet-weeks path (no '
    'second claim-based resolution row: theme_slug stays PRIMARY KEY, one '
    'row per theme for its whole history, per task instruction "keep '
    'theme_resolutions as history"). Never cleared once set.';
