-- Portolan schema, migration 0005: theme membership, report rows,
-- standing-question answers.
--
-- Owned by wave C1 (portolan/report/**). Spec references:
-- 01-spec/surfacing-and-weekly-report.md ("Theme notes: the rollup layer",
-- "Standing questions", "The report is a node") and
-- 01-spec/portolan-spec-v2.md sec 7 ("Surfacing"). Contract:
-- 01-spec/build-contracts.md "Migration ledger" row for this file, and
-- "Report, audit, consolidation" ("the C agents propose signatures in
-- their reports").
--
-- IDEMPOTENT, like 0002/0003/0004: IF NOT EXISTS throughout, so this file
-- can be re-applied safely (the build instructions require it to apply
-- twice without error).

-- ---------------------------------------------------------------------
-- theme_members: which claims a theme note has pulled in, and when.
--
-- A `theme` is a note like any other (notes.type = 'theme', already in
-- the closed list, portolan/notes/models.py NoteType); this table is the
-- one thing a note's own body can't carry on its own -- the *dated*
-- membership list a trend computation can group by week. (The theme
-- note's own `## Related` links to every account/product it touches; the
-- claims it cites appear as `^[source @ locator]` markers in its
-- synthesis. `theme_members` is the machine-readable index behind both,
-- and the only place `added` -- the date a claim actually joined the
-- theme, used for the week-by-week trend, portolan/report/themes.py --
-- is recorded.)
--
-- `added` is a plain `date`, not `timestamptz`: trend is computed by ISO
-- week (`portolan/report/themes.py`'s `theme_weekly_counts`), and this
-- build backdates `added` to the claim's own world date (`valid_from`, or
-- else `recorded`'s date -- see that module) rather than the wall-clock
-- moment `refresh_themes` happened to run, so a single backfill run over
-- a historical window still buckets correctly week by week. `score` is
-- the cosine similarity (or NULL for a shared-product-and-fact-type
-- match, which isn't a similarity score) that won it the attachment, kept
-- for auditability.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS theme_members (
    theme_slug  text NOT NULL REFERENCES notes (slug) ON DELETE RESTRICT,
    claim_id    bigint NOT NULL REFERENCES claims (id) ON DELETE RESTRICT,
    added       date NOT NULL,
    score       numeric(5, 4),
    PRIMARY KEY (theme_slug, claim_id)
);

COMMENT ON TABLE theme_members IS
    'Which claims a theme note has pulled in and when (spec sec 7, '
    'surfacing-and-weekly-report.md "Theme notes"). added is the claim''s '
    'own world date, not the run date, so week-by-week trend backfills '
    'correctly from a single pass.';

CREATE INDEX IF NOT EXISTS idx_theme_members_claim_id ON theme_members (claim_id);
CREATE INDEX IF NOT EXISTS idx_theme_members_added ON theme_members (added);

-- ---------------------------------------------------------------------
-- reports: one row per rendered weekly report note (spec sec 7: "The
-- weekly report writes itself ... The report is a node"). note_slug is
-- `report-<ISO week>` (portolan/report/weekly.py); this table is the
-- lookup index (list past reports, find a week's report by name) the
-- note's own frontmatter doesn't give you without scanning `notes`.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS reports (
    id         bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    week       text NOT NULL,
    note_slug  text NOT NULL REFERENCES notes (slug) ON DELETE RESTRICT,
    created    timestamptz NOT NULL DEFAULT now(),
    UNIQUE (week)
);

COMMENT ON TABLE reports IS
    'One row per weekly report note (surfacing-and-weekly-report.md "The '
    'report is a node"). week is an ISO week string, e.g. "2026-W37".';

CREATE INDEX IF NOT EXISTS idx_reports_week ON reports (week);

-- ---------------------------------------------------------------------
-- standing_question_answers: the weekly answer to one reader's standing
-- question (0003_standing_questions.sql's standing_questions table,
-- surfacing-and-weekly-report.md "Standing questions": "written with
-- citations into that reader's section"). citations is a JSON array of
-- the note slugs the answer actually drew on -- every sentence's cited
-- slug is validated against the retrieved set before being kept
-- (portolan/report/questions.py), so this column is always a subset of
-- what was retrieved for that question that week.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS standing_question_answers (
    id           bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    question_id  bigint NOT NULL REFERENCES standing_questions (id) ON DELETE RESTRICT,
    week         text NOT NULL,
    answer_md    text NOT NULL,
    citations    jsonb NOT NULL DEFAULT '[]'::jsonb,
    created      timestamptz NOT NULL DEFAULT now(),
    UNIQUE (question_id, week)
);

COMMENT ON TABLE standing_question_answers IS
    'One reader''s standing question, answered for one ISO week, with the '
    'note slugs the answer actually cites (surfacing-and-weekly-report.md '
    '"Standing questions").';

CREATE INDEX IF NOT EXISTS idx_standing_question_answers_question_id
    ON standing_question_answers (question_id);
CREATE INDEX IF NOT EXISTS idx_standing_question_answers_week
    ON standing_question_answers (week);
