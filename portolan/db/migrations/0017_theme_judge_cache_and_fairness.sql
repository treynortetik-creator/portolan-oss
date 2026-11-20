-- Portolan schema, migration 0017: cache a `false` judge verdict, and
-- give the per-run cap a fairness cursor.
--
-- Owned by the fix wave following review-theme-wave-5b.md (should-fix
-- 5). Two independent, small tables:
--
-- 1. `theme_judge_verdicts`. `_check_theme_resolutions` (portolan/
-- report/themes.py) already never re-judges a (theme, claim) pair that
-- produced a `true` verdict -- that pair has a `theme_resolutions` row,
-- and `already_judged` reads it. A `false` verdict wrote nothing at all,
-- so the SAME pair was re-judged, and re-paid, on every single run for
-- as long as the candidate survived `THEME_RESOLVED_LOOKBACK_WEEKS`.
-- This table remembers a `false` verdict was reached, and the THEME
-- STATE it was reached against (member count) plus the CANDIDATE TEXT
-- it was reached against (a hash) -- so a re-run skips the judge call
-- entirely for an unchanged pair, and re-judges only when the theme
-- gained members since (its own story may have changed) or the
-- candidate's text changed (rare, but a claim is not literally
-- immutable at the schema level).
--
-- 2. `theme_judge_cursor`. A single row (the `id boolean PRIMARY KEY
-- DEFAULT true` trick guarantees at most one) naming the theme_slug that
-- led the judge queue LAST run, so the next run's own queue rotates to
-- start with a DIFFERENT theme -- see `_check_theme_resolutions`'s own
-- docstring for why strict recency-first ordering, unchanged, starved
-- the same themes every week once the cap actually bound.
--
-- IDEMPOTENT, like every prior migration file: IF NOT EXISTS throughout,
-- safe to re-apply (tests/test_db_schema.py applies every migration file
-- twice).
--
-- Nit (review-theme-wave-5c.md finding 6): migration 0018 later replaces
-- `theme_member_count`/`candidate_text_hash` below with one
-- `prompt_state_hash` column (a member-count PROXY for "has the theme's
-- story changed" that a rename, an in-place member edit, or an
-- equal-count member swap could all defeat) -- left as originally
-- written here, a historical record of this migration's own shape, not
-- rewritten to describe schema a LATER file actually changes; the live
-- `COMMENT ON TABLE` (0018's own) is what a running database shows.
--
-- Nit (review-theme-wave-5c.md finding 10): `candidate_claim_id`'s
-- `ON DELETE CASCADE` below is the first claim FK in this schema that
-- ISN'T `ON DELETE RESTRICT` (every other one -- 0001:196, 0001:401,
-- 0005:42, 0007:28, 0011:30 -- treats a claim as provenance nothing
-- deletes out from under a reference). Deliberate, not an oversight:
-- this table is a cache, explicitly never read as evidence of anything
-- (see its own `COMMENT ON TABLE` below) -- when the claim it cached a
-- verdict about is gone, the cached verdict is worthless, not history
-- to preserve, so CASCADE is the correct behaviour rather than a
-- RESTRICT that would block a legitimate claim deletion over a row that
-- exists purely to save a judge call.
--
-- Nit (review-theme-wave-5c.md finding 11, "orphan cache rows"): no FK
-- ties a row here to `theme_slug`, and nothing deletes a merged-away
-- theme's own cached verdicts -- harmless (a merged slug's notes are
-- ARCHIVED, never deleted or reused, per spec sec 8) but worth naming so
-- a future reader auditing this table's growth does not mistake it for
-- a bug.

CREATE TABLE IF NOT EXISTS theme_judge_verdicts (
    theme_slug          text NOT NULL,
    candidate_claim_id  bigint NOT NULL REFERENCES claims (id) ON DELETE CASCADE,
    theme_member_count  integer NOT NULL,
    candidate_text_hash text NOT NULL,
    judged_at           timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (theme_slug, candidate_claim_id)
);

COMMENT ON TABLE theme_judge_verdicts IS
    'A cached FALSE resolution-judge verdict for one (theme, candidate '
    'claim) pair -- a TRUE verdict has its own durable record in '
    'theme_resolutions and needs no cache. theme_member_count/'
    'candidate_text_hash pin the theme/candidate STATE the verdict was '
    'reached against; _check_theme_resolutions skips a re-judge only '
    'while both are unchanged. Never read as evidence of anything by '
    'itself -- purely a cost/fairness optimization over re-asking the '
    'same question the answer has not changed.';

CREATE INDEX IF NOT EXISTS idx_theme_judge_verdicts_theme
    ON theme_judge_verdicts (theme_slug);

CREATE TABLE IF NOT EXISTS theme_judge_cursor (
    id               boolean PRIMARY KEY DEFAULT true,
    last_started_slug text,
    updated_at       timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT theme_judge_cursor_singleton CHECK (id)
);

COMMENT ON TABLE theme_judge_cursor IS
    'Singleton row (id boolean PRIMARY KEY, only ever true) naming the '
    'theme_slug that led the resolution-judge queue in the most recent '
    'refresh_themes run -- the round-robin fairness cursor '
    '_check_theme_resolutions advances by one eligible theme each run '
    'so THEME_JUDGE_MAX_CALLS truncating the queue never starves the '
    'same theme(s) indefinitely.';
