-- Portolan schema, migration 0032: review-queue volume fixes (2026-09-22).
--
-- registry_entities.inactive_since: the date the CRM says a contact left or
-- an account churned (`left_2026-07` -> 2026-07-01), set by the CRM import
-- alongside `active = false`. `Registry.resolve` refuses an inactive entity
-- only for a fact said after that date; before this column every mention of
-- a departed contact became an unknown-entity row, including the CRM's own
-- row recording the departure. Null for an active entity, and for an
-- inactive one whose status carries no date (refused for every fact, as
-- before).
--
-- idx_review_queue_open_mention: `registry.queue_unknown` and
-- `merge.resolve._queue_ambiguous` now keep one open row per mention and
-- add each further sighting to that row; this is their lookup.
--
-- IDEMPOTENT: `IF NOT EXISTS` on both.

ALTER TABLE registry_entities ADD COLUMN IF NOT EXISTS inactive_since date;

COMMENT ON COLUMN registry_entities.inactive_since IS
    'Date the CRM status says this contact left or this account churned. '
    'Mentions in facts said on or before it still resolve.';

CREATE INDEX IF NOT EXISTS idx_review_queue_open_mention
    ON review_queue (kind, lower(btrim(payload ->> 'mention')))
    WHERE status = 'open';
