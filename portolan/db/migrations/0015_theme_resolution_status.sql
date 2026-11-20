-- Portolan schema, migration 0015: theme resolutions become proposals.
--
-- Owned by the fifth theme fix wave (2026-09-14, portolan/report/
-- themes.py). Decisions log reference: 2026-09-14, "withdraw content-
-- based automatic resolution" -- the stem-based resolution gate
-- (distinctiveness, matching vocabulary, per-stem corroboration) is
-- deleted, not fixed a fifth time (`01-spec/reviews/review-theme-fix-
-- wave-4.md`). A cheap-model judge now decides resolution; a `true`
-- verdict is a PROPOSAL a human confirms or rejects, never an immediate
-- `resolved` state.
--
-- IDEMPOTENT, like every prior migration file: IF NOT EXISTS throughout
-- (and a pg_constraint guard for the CHECK constraint, which has no
-- native IF NOT EXISTS form -- see 0001_init.sql's own note on this),
-- safe to re-apply (tests/test_db_schema.py applies every migration file
-- twice).

-- ---------------------------------------------------------------------
-- status: `proposed` (a judge verdict, awaiting a human), `confirmed`
-- (a human agreed, OR the mechanical THEME_RESOLVED_AFTER_WEEKS-of-
-- silence path -- which needs no human), `rejected` (a human
-- disagreed), or `superseded` (a still-open proposal made moot by
-- renewed theme activity before a human acted on it). Every row written
-- before this migration was written by the old stem-based gate, which
-- had no "proposed" state at all -- it marked a theme resolved the
-- moment the gate passed, exactly what `status='confirmed'` means now,
-- so existing rows are backfilled that way (the column DEFAULT below;
-- every INSERT this codebase issues from here on always states `status`
-- explicitly, so the default is only ever read for the backfill).
-- ---------------------------------------------------------------------
ALTER TABLE theme_resolutions
    ADD COLUMN IF NOT EXISTS status text NOT NULL DEFAULT 'confirmed';

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'theme_resolutions_status_check'
          AND conrelid = 'theme_resolutions'::regclass
    ) THEN
        ALTER TABLE theme_resolutions ADD CONSTRAINT theme_resolutions_status_check
            CHECK (status IN ('proposed', 'confirmed', 'rejected', 'superseded'));
    END IF;
END $$;

COMMENT ON COLUMN theme_resolutions.status IS
    'proposed (a resolution judge verdict, awaiting a human via /review), '
    'confirmed (a human agreed, or the mechanical silence path -- '
    'resolution_claim_id IS NULL, needs no human), rejected (a human '
    'disagreed), or superseded (a still-open proposal made moot by '
    'renewed theme activity before a human acted on it). Only a '
    'CONFIRMED row makes theme_trend() read the theme "resolved"; a '
    'PROPOSED one reads "resolution proposed" instead.';

-- ---------------------------------------------------------------------
-- evidence: the exact sentence the judge quoted from the resolving
-- claim (decisions log task item 3) -- NULL for the mechanical silence
-- path (no claim, no quote) and for a resolution written by a caller
-- that supplies none.
-- ---------------------------------------------------------------------
ALTER TABLE theme_resolutions
    ADD COLUMN IF NOT EXISTS evidence text;

COMMENT ON COLUMN theme_resolutions.evidence IS
    'The exact sentence the resolution judge quoted from the resolving '
    'claim as its evidence (task item 3: must be a substring of the '
    'candidate claim''s own plain text). NULL for the mechanical silence '
    'path, which names no claim and so has nothing to quote.';

-- ---------------------------------------------------------------------
-- resolution_claim_id becomes nullable: the mechanical
-- THEME_RESOLVED_AFTER_WEEKS-of-silence path (task item 4, "recorded as
-- a confirmed row with reason silence") names no specific claim -- there
-- is nothing to cite for "nothing happened for three weeks." Every
-- claim-based row (proposed or confirmed) still names one; the FK to
-- claims(id) is untouched and simply does not apply to a NULL value.
-- ---------------------------------------------------------------------
ALTER TABLE theme_resolutions
    ALTER COLUMN resolution_claim_id DROP NOT NULL;

-- ---------------------------------------------------------------------
-- A given (theme, claim) pair is judged at most once, ever -- task item
-- 4: "a proposal is not re-proposed on later runs while it is open or
-- after rejection." `uq_theme_resolutions_active_theme` (migration 0013,
-- `(theme_slug) WHERE reopened_week IS NULL`) already keeps at most one
-- OPEN row per theme regardless of claim; this is the second, narrower
-- rule -- application code (portolan.report.themes._check_theme_
-- resolutions) already checks a theme's full resolution history before
-- ever calling the judge again for a claim it has already judged, so
-- this index is enforcement a second way, at the database, the same
-- "belt and suspenders" discipline this schema already applies
-- elsewhere. Partial (excludes NULL claim ids): Postgres already treats
-- NULL as distinct from every other NULL under a plain UNIQUE index, so
-- the WHERE clause is documentation of that, not a behaviour change --
-- multiple silence rows (resolution_claim_id IS NULL) for the same
-- theme, across different silent arcs, are expected and fine.
-- ---------------------------------------------------------------------
CREATE UNIQUE INDEX IF NOT EXISTS uq_theme_resolutions_theme_claim
    ON theme_resolutions (theme_slug, resolution_claim_id)
    WHERE resolution_claim_id IS NOT NULL;

COMMENT ON TABLE theme_resolutions IS
    'History of resolution events, one row per (theme, resolved_week) -- '
    'proposed, confirmed, rejected, or superseded (status). '
    'uq_theme_resolutions_active_theme keeps at most one OPEN '
    '(reopened_week IS NULL) row per theme_slug at a time; '
    'uq_theme_resolutions_theme_claim keeps a given (theme, claim) pair '
    'from ever being judged twice. Never hard-deleted: a merge that must '
    'reconcile two open rows under one slug closes the later one out '
    '(sets its reopened_week, and status=''superseded'' if it was still '
    '''proposed'') rather than dropping it.';
