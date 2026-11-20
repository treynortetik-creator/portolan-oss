-- Portolan schema, migration 0030: the action note "why" (follow-up to
-- 01-spec/actions-and-owners-2026-09-19.md's Phase 2, 2026-09-20: "the
-- reason already exists and is fully grounded ... None of it reaches the
-- note. The fix is plumbing, not more model prose: render what the card
-- already proved.").
--
-- action_why: one row per action whose action came from a researched,
-- gate-passed `ActionCard` (portolan.report.researcher.confirm_action_
-- card) -- the card's own grounded content, persisted at Confirm time so
-- the action note and the Actions page can render WHY the action exists
-- (inference sentences, evidence copies, finding results, confidence,
-- who/when proposed and confirmed it) without re-deriving or re-wording
-- any of it. An action created by the extraction pipeline (`merge_
-- action`, no card at all) has no row here -- the note/page fall back to
-- their current rendering, unchanged, whenever this table has nothing
-- for a given action.
--
-- `evidence`/`findings`/`inference`/`confidence_refs` are frozen JSON
-- snapshots of the confirmed card's own fields (evidence copied verbatim
-- from claims already, findings resolved from `researcher_findings` at
-- confirm time) -- deliberately NOT re-read live from `review_queue`,
-- which may later be edited, superseded, or (per app retention) pruned;
-- the note's "why" must keep reading the same after that.
--
-- IDEMPOTENT (CREATE TABLE IF NOT EXISTS), same pattern as 0029.

CREATE TABLE IF NOT EXISTS action_why (
    action_id        bigint PRIMARY KEY REFERENCES actions (id) ON DELETE CASCADE,
    review_row_id    bigint REFERENCES review_queue (id) ON DELETE SET NULL,
    run_id           text,
    candidate_slug   text,
    verdict          text,
    evidence         jsonb NOT NULL DEFAULT '[]'::jsonb,
    findings         jsonb NOT NULL DEFAULT '[]'::jsonb,
    inference        jsonb NOT NULL DEFAULT '[]'::jsonb,
    confidence       text,
    confidence_refs  jsonb NOT NULL DEFAULT '{}'::jsonb,
    proposed_at      timestamptz,
    confirmed_by     text NOT NULL,
    confirmed_at     timestamptz NOT NULL DEFAULT now()
);

COMMENT ON TABLE action_why IS
    'The researched ActionCard''s own grounded content, persisted at '
    'Confirm time: evidence claim copies, resolved finding results, '
    'inference sentences with refs, confidence, and who/when proposed '
    'and confirmed it. Rendered on the action note and the Actions page '
    'as the "Why" this action exists, instead of the boilerplate '
    '"is an open action." synthesis.';

CREATE INDEX IF NOT EXISTS idx_action_why_review_row ON action_why (review_row_id);
