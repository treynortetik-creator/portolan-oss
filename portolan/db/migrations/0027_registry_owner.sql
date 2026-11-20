-- Portolan schema, migration 0027: owners (01-spec/actions-and-owners-
-- 2026-09-19.md, Phase 1). "Owner is a registry fact, not note text" --
-- `registry_entities.owner_entity_id` is nullable and points at another
-- `registry_entities` row (a vendor-side `internal_person`, enforced in
-- application code -- see `portolan.registry.owners.is_vendor_side` --
-- not a DB CHECK, since Postgres cannot cheaply constrain one row's
-- value against another row's column without a trigger, and every other
-- cross-entity rule in this schema so far (e.g. "an owner is always
-- vendor-side") is enforced the same way, in code, mirroring
-- `registry.resolve`'s own `active`-only filtering).
--
-- `review_queue`'s kind check widens for `owner_proposal`
-- (`python -m portolan.registry propose-owners`, build item 3): one row
-- per account, candidates + evidence, Confirm sets the field.
--
-- IDEMPOTENT: `ADD COLUMN IF NOT EXISTS` is natively idempotent; the
-- constraint is dropped and re-added either way, same pattern as 0026.

ALTER TABLE registry_entities
    ADD COLUMN IF NOT EXISTS owner_entity_id bigint
        REFERENCES registry_entities (id) ON DELETE RESTRICT;

COMMENT ON COLUMN registry_entities.owner_entity_id IS
    'The vendor-side (internal_person) registry entity responsible for '
    'this account (01-spec/actions-and-owners-2026-09-19.md Phase 1). '
    'Null until assigned. Lives on the registry row, not note text, so a '
    're-ingest or merge never clobbers it.';

CREATE INDEX IF NOT EXISTS idx_registry_entities_owner_entity_id
    ON registry_entities (owner_entity_id);

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
                 'theme_resolution', 'quote_unsupported', 'owner_proposal', 'other')
    );
END $$;
