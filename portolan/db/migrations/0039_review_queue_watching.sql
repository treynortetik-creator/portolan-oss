-- Portolan schema, migration 0039: review_queue.status gains 'watching'.
--
-- An unknown or ambiguous name on a fact that was filed anyway is not worth
-- a person's answer the first time it appears. Such a row is written as
-- 'watching' and becomes 'open' when the name blocks a fact from being
-- filed, or when it has appeared in enough distinct sources
-- (`portolan.registry.registry.park_mention`). A watching row changes
-- nothing in the graph; it only delays the question.
--
-- IDEMPOTENT: drops and re-adds the same constraint either way.
-- destructive-ok: the constraint is dropped only to be re-added one value wider; no data changes

DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'review_queue_status_check' AND conrelid = 'review_queue'::regclass
    ) THEN
        ALTER TABLE review_queue DROP CONSTRAINT review_queue_status_check;
    END IF;
    ALTER TABLE review_queue ADD CONSTRAINT review_queue_status_check CHECK (
        status IN ('open', 'watching', 'resolved', 'dismissed')
    );
END $$;
