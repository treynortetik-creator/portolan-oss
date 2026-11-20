-- Portolan schema, migration 0016: the (theme, claim) uniqueness only
-- applies to OPEN proposals.
--
-- Owned by the fix wave following review-theme-wave-5-judge.md (blocker
-- 2). Migration 0015's `uq_theme_resolutions_theme_claim` was unique on
-- (theme_slug, resolution_claim_id) for EVERY row, open or closed --
-- which collides with this design's own blessed shape ("one claim CAN
-- propose more than one theme in the same call", themes.py:2969-2975,
-- tested by test_resolution_gate_one_claim_can_propose_several_themes):
-- when two such themes merge, `_carry_theme_resolution` closes out the
-- later-resolved side's row (reopened_week set, status='superseded' if
-- it was still 'proposed' -- the standing "close out, don't delete"
-- pattern) and then moves EVERY row from the absorbed slug onto the
-- surviving one, landing two rows with the SAME (theme_slug,
-- resolution_claim_id) under one slug: the still-open winner and the
-- now-closed-out loser. The old index did not distinguish them and
-- raised UniqueViolation out of a module with no exception handling
-- anywhere, taking the whole weekly report run down with it.
--
-- Fix: scope the uniqueness to OPEN rows only (add `reopened_week IS
-- NULL`, the same column `uq_theme_resolutions_active_theme` (migration
-- 0013) already keys its own partial index on). A given (theme, claim)
-- pair is still judged at most once EVER in practice -- that invariant
-- is, and always was, enforced at the application layer
-- (portolan.report.themes._check_theme_resolutions's `already_judged`
-- full-history check, themes.py:3012-3016), independent of any database
-- index; this index was only ever "enforcement a second way, belt and
-- suspenders" (0015's own comment) on top of that, never the sole
-- guard. Scoping it to open rows keeps that belt-and-suspenders role for
-- the case that actually matters (two OPEN proposals for the same
-- theme/claim colliding) while no longer forbidding the one case this
-- design deliberately allows: a closed-out historical row sharing its
-- key with a still-open survivor after a merge.
--
-- Nit 8 (review-theme-wave-5b.md finding 8), added without a new
-- migration (nothing here needs one -- a comment-only correction):
-- verified against `uq_theme_resolutions_active_theme` (migration 0013,
-- `(theme_slug) WHERE reopened_week IS NULL`), this index enforces
-- NOTHING that index did not already -- at most one OPEN row per
-- `theme_slug` already means two open rows can never share a
-- `(theme_slug, resolution_claim_id)` key, open-rows-only or not. Kept
-- rather than dropped (a `DROP INDEX` migration for a harmless, no-op
-- index is not worth the churn) as documentation of intent: the shape a
-- reader would otherwise have to re-derive from `_carry_theme_
-- resolution`'s own comments.
--
-- IDEMPOTENT, like every prior migration file: IF NOT EXISTS/IF EXISTS
-- throughout, safe to re-apply (tests/test_db_schema.py applies every
-- migration file twice).

DROP INDEX IF EXISTS uq_theme_resolutions_theme_claim;

CREATE UNIQUE INDEX IF NOT EXISTS uq_theme_resolutions_theme_claim
    ON theme_resolutions (theme_slug, resolution_claim_id)
    WHERE resolution_claim_id IS NOT NULL AND reopened_week IS NULL;

COMMENT ON TABLE theme_resolutions IS
    'History of resolution events, one row per (theme, resolved_week) -- '
    'proposed, confirmed, rejected, or superseded (status). '
    'uq_theme_resolutions_active_theme keeps at most one OPEN '
    '(reopened_week IS NULL) row per theme_slug at a time; '
    'uq_theme_resolutions_theme_claim (migration 0016: OPEN rows only) '
    'is subsumed by uq_theme_resolutions_active_theme (at most one OPEN '
    'row per theme_slug already rules out two sharing a claim), kept as '
    'documentation of intent -- a merge that carries over a closed-out '
    'row sharing its key with the surviving open one is expected and '
    'allowed. Never hard-'
    'deleted: a merge that must reconcile two open rows under one slug '
    'closes the later one out (sets its reopened_week, and '
    'status=''superseded'' if it was still ''proposed'') rather than '
    'dropping it.';
