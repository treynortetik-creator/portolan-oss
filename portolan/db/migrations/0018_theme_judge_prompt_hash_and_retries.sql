-- Portolan schema, migration 0018: hash the whole judge prompt for the
-- false-verdict cache, and give an unverifiable `true` verdict a bounded
-- number of retries instead of either caching it `false` or re-judging
-- it forever.
--
-- Owned by review-theme-wave-5c.md findings 3 and 4.
--
-- 1. `theme_judge_verdicts` (migration 0017): the false-verdict cache's
-- own staleness key used to be `theme_member_count` (an integer proxy
-- for "has the theme's own story changed") plus `candidate_text_hash`.
-- Measured, three ways, to leave a stale `false` verdict frozen in
-- place: a theme RENAME (member count unchanged), a member claim EDITED
-- IN PLACE (`portolan/consolidate/edits.py`'s `UPDATE claims SET
-- claim_text = ...` -- a real production path, count unchanged, text
-- changed but the count-only key never looks at member text), and a
-- member SWAP at equal count (one member removed, a different one
-- added -- the count is identical before and after). `theme_member_
-- count`/`candidate_text_hash` are replaced with a single `prompt_state_
-- hash`: a hash over exactly what `_build_resolution_judge_prompt`
-- actually shows the judge -- the theme's own name, every member claim
-- id paired with its own text (sorted, so a swap changes the hash even
-- when the member COUNT does not), and the candidate's own text. ANY
-- change to anything the prompt would show changes this hash and forces
-- a re-judge. Existing rows backfill to an empty string, which never
-- matches a freshly computed hash -- the one-time cost is each
-- previously-cached pair getting re-judged once more than strictly
-- necessary, never a correctness problem.
--
-- 2. `theme_judge_unverified` (new): a `resolves: true` verdict whose
-- `evidence` is not an exact substring of the candidate's own text is
-- neither a genuine `false` (the judge believes the theme IS resolved)
-- nor a usable `true` (it cannot back its own quote up) -- caching it as
-- `false` (finding 4's own repro) buries a real resolution for good the
-- first time a model fumbles a quote. This table counts how many
-- CONSECUTIVE times a (theme, candidate) pair has come back this way
-- against an UNCHANGED `prompt_state_hash`; `THEME_JUDGE_RETRY_LIMIT`
-- (`portolan/report/themes.py`) caps the retries, after which the pair
-- is finally written to `theme_judge_verdicts` as an ordinary cached
-- `false` and this row is cleared.
--
-- IDEMPOTENT, like every prior migration file: IF NOT EXISTS / IF
-- EXISTS throughout, safe to re-apply (tests/test_db_schema.py applies
-- every migration file twice).

ALTER TABLE theme_judge_verdicts
    ADD COLUMN IF NOT EXISTS prompt_state_hash text NOT NULL DEFAULT '';

ALTER TABLE theme_judge_verdicts
    DROP COLUMN IF EXISTS theme_member_count;

ALTER TABLE theme_judge_verdicts
    DROP COLUMN IF EXISTS candidate_text_hash;

ALTER TABLE theme_judge_verdicts
    ALTER COLUMN prompt_state_hash DROP DEFAULT;

COMMENT ON TABLE theme_judge_verdicts IS
    'A cached FALSE resolution-judge verdict for one (theme, candidate '
    'claim) pair -- a TRUE verdict has its own durable record in '
    'theme_resolutions and needs no cache. prompt_state_hash (migration '
    '0018) pins the exact theme name + sorted member (id, text) pairs + '
    'candidate text the judge prompt was built from; _check_theme_'
    'resolutions skips a re-judge only while it is unchanged. Never read '
    'as evidence of anything by itself -- purely a cost/fairness '
    'optimization over re-asking the same question the answer has not '
    'changed.';

-- `candidate_claim_id ON DELETE CASCADE` (not this schema's usual
-- claim-FK `RESTRICT` -- see 0017's own comment on `theme_judge_
-- verdicts` for the full reasoning): this table is a retry COUNTER, not
-- history: if the claim it is counting attempts against is gone, the
-- count is meaningless, and CASCADE is correct rather than blocking a
-- legitimate claim deletion over it.
CREATE TABLE IF NOT EXISTS theme_judge_unverified (
    theme_slug          text NOT NULL,
    candidate_claim_id  bigint NOT NULL REFERENCES claims (id) ON DELETE CASCADE,
    prompt_state_hash   text NOT NULL,
    retry_count         integer NOT NULL DEFAULT 0,
    updated_at          timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (theme_slug, candidate_claim_id)
);

COMMENT ON TABLE theme_judge_unverified IS
    'How many CONSECUTIVE runs, against an unchanged prompt_state_hash, a '
    '(theme, candidate claim) pair has come back resolves:true with '
    'evidence that failed the substring check. Reset to 1 the moment the '
    'hash changes (a new prompt is a fresh question); cleared once '
    'THEME_JUDGE_RETRY_LIMIT is reached and the pair is written to '
    'theme_judge_verdicts as an ordinary cached false verdict instead. '
    'Never read as evidence of anything by itself.';
