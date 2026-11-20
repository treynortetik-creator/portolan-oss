-- Portolan schema, migration 0020: add `context` to the note-type closed list.
--
-- Owner decision 2026-09-17 ("Company context page"): a company context
-- page is a wiki page inside the graph, editable like any note, pulled
-- into every extraction and judge call so the model knows who the tenant
-- is, what it sells, who it competes with, and why a fact matters. It is
-- one note per tenant, at slug `company-context`, of this new type.
--
-- The CHECK constraint on `notes.type` (migration 0001) has no explicit
-- name, so Postgres auto-named it `notes_type_check` -- verified against
-- the running schema before writing this migration. Dropped and
-- re-added rather than altered in place (Postgres has no `ALTER
-- CONSTRAINT ... ADD VALUE`, unlike an enum type), which keeps this file
-- idempotent (IF EXISTS on the drop) the same way every other migration
-- in this ledger is.
--
-- `notes_crm_id_required_for_account_contact` and
-- `notes_parent_only_on_account` are untouched: a `context` note carries
-- neither a `crm_id` nor a `parent_slug`, exactly like `signal`/`theme`/
-- `decision`/`action`/`report` already don't.

ALTER TABLE notes DROP CONSTRAINT IF EXISTS notes_type_check;
ALTER TABLE notes ADD CONSTRAINT notes_type_check CHECK (
    type IN ('account', 'contact', 'product', 'competitor',
             'signal', 'theme', 'decision', 'action',
             'source', 'report', 'context')
);

COMMENT ON CONSTRAINT notes_type_check ON notes IS
    'The closed note-type list, portolan-spec-v2.md sec 2 (now eleven '
    'types: the original ten plus `context`, added 2026-09-17).';
