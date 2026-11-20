-- Portolan schema, migration 0026: widen review_queue.kind to accept
-- 'quote_unsupported' -- the mechanical quote gate (01-spec/
-- extraction-final-2026-09-18.md's production pick: "Nova Lite, five
-- samples, v6 prompt, mechanical quote gate, human review").
--
-- `portolan.pipeline._finish_one` now drops any quote-requiring fact
-- whose quote isn't actually found (verbatim or near-verbatim) in the
-- source text it cites, before that fact ever reaches merge, and parks
-- it here instead -- the same "facts are never dropped, only flagged"
-- rule migration 0001's own table comment states, and the same widening
-- pattern migration 0013 already used for 'theme_resolution'.
--
-- IDEMPOTENT: safe to re-apply (drops and re-adds the same constraint
-- either way).

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
                 'theme_resolution', 'quote_unsupported', 'other')
    );
END $$;
