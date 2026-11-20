-- Portolan schema, migration 0033: auto-close actions from later evidence
-- (01-spec/app-checklist-2026-09-18.md item 24, the maintainer 2026-09-20).
--
-- review_queue.kind widens for 'action_done': one row proposing "Done" on an
-- open action, citing a later claim on the same account by someone on the
-- vendor's own side that a judge read as answering it
-- (portolan.report.action_done). A person confirms or dismisses; nothing
-- closes on its own. Same widening pattern as 0013/0026/0027/0028.
--
-- idx_review_queue_open_action_done: at most one OPEN proposal per action,
-- so a re-run can never stack a second card on the same action.
--
-- action_done_evidence: the claim that closed an action, frozen at Confirm
-- time with who and when, so the action note keeps showing why it is done
-- after the queue row is gone. One row per action, like action_why.
--
-- IDEMPOTENT: DROP-then-ADD for the CHECK, IF NOT EXISTS for the rest.
-- destructive-ok: the only DROP is the kind CHECK constraint, re-added wider in the same block; no row is touched.

DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'review_queue_kind_check' AND conrelid = 'review_queue'::regclass
    ) THEN
        ALTER TABLE review_queue DROP CONSTRAINT review_queue_kind_check;
    END IF;
    ALTER TABLE review_queue ADD CONSTRAINT review_queue_kind_check CHECK (
        kind IN ('unknown_entity', 'ambiguous_entity', 'failed_consolidation',
                 'theme_resolution', 'quote_unsupported', 'owner_proposal',
                 'action_card', 'action_done', 'other')
    );
END $$;

CREATE UNIQUE INDEX IF NOT EXISTS idx_review_queue_open_action_done
    ON review_queue ((payload ->> 'action_id'))
    WHERE kind = 'action_done' AND status = 'open';

CREATE TABLE IF NOT EXISTS action_done_evidence (
    action_id      bigint PRIMARY KEY REFERENCES actions (id) ON DELETE CASCADE,
    review_row_id  bigint REFERENCES review_queue (id) ON DELETE SET NULL,
    claim_id       bigint REFERENCES claims (id) ON DELETE RESTRICT,
    evidence       jsonb NOT NULL,
    finding        jsonb,
    judge_version  text,
    proposed_at    timestamptz,
    confirmed_by   text NOT NULL,
    confirmed_at   timestamptz NOT NULL DEFAULT now()
);

COMMENT ON TABLE action_done_evidence IS
    'The later claim a person confirmed as answering an action (review kind '
    'action_done): a verbatim copy of the claim, the finding that found it, '
    'and who confirmed it when. Rendered on the action note as "Done".';
