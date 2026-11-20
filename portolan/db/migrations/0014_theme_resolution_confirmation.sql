-- Portolan schema, migration 0014: a human confirmation leaves a mark.
--
-- Owned by the second theme fix wave (2026-09-13, portolan/report/
-- themes.py). Review reference: 01-spec/reviews/review-theme-fix-wave-2.md,
-- nit 19 ("confirming a resolution leaves no mark on the resolution
-- itself"; nit 13 of the third wave's own review caught this citation
-- pointing at wave 1 instead of wave 2). `/review`'s confirm action used
-- to be a bare review_queue
-- status flip -- right for every OTHER queue kind, but it left theme_
-- resolutions unable to distinguish a human-confirmed resolution from one
-- nobody has looked at yet, even though the join back to review_queue
-- (payload->>'theme_slug') already exists.
--
-- IDEMPOTENT, like every prior migration file: IF NOT EXISTS throughout,
-- safe to re-apply (tests/test_db_schema.py applies every migration file
-- twice).

ALTER TABLE theme_resolutions
    ADD COLUMN IF NOT EXISTS confirmed_by text;

ALTER TABLE theme_resolutions
    ADD COLUMN IF NOT EXISTS confirmed_at timestamptz;

COMMENT ON COLUMN theme_resolutions.confirmed_by IS
    'The editor who confirmed this resolution via /review (portolan.report.'
    'themes.confirm_theme_resolution), or NULL if nobody has confirmed it'
    ' (or acted on it at all -- confirmation is advisory, not required for'
    ' the resolution to take effect).';

COMMENT ON COLUMN theme_resolutions.confirmed_at IS
    'When confirmed_by confirmed this resolution. NULL until then.';
