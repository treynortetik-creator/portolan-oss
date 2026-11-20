-- Portolan schema, migration 0036: CRM snapshots and the changes between them
-- (2026-09-25, 01-spec/monday-email-2026-09-25.md).
--
-- The red-team review (01-spec/red-team-2026-09-24.md sec 1) asks for CRM field
-- changes to become facts through code, not a model. A CRM row was already kept
-- as a source (one original per exported row), but nothing said which rows were
-- the same record on different export dates, so no code could compare them.
--
-- crm_snapshots: one row per exported CRM record per export date. `fields` is
-- the row as exported; `as_of` is the export date (the row's export_date
-- column, else the date in the file name, else the connector's source date for
-- an API connector). `source_slug`/`locator` point at the stored original, the
-- same citation a claim uses.
--
-- crm_changes: one row per field that differs between a record's snapshot and
-- the most recent earlier snapshot carrying that field, plus records added to
-- or dropped from an export. Written by portolan.crm.changes, which also writes
-- the claim (`claim_id`) whose text states the change; old/new values and the
-- computed deltas live here so the email never parses claim text for numbers.
--
-- IDEMPOTENT: CREATE ... IF NOT EXISTS throughout.

CREATE TABLE IF NOT EXISTS crm_snapshots (
    id              bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    object          text NOT NULL CHECK (object IN ('account', 'contact', 'opportunity')),
    crm_id          text NOT NULL,
    account_crm_id  text,
    as_of           date NOT NULL,
    export_key      text NOT NULL,
    fields          jsonb NOT NULL,
    source_slug     text NOT NULL REFERENCES sources (slug) ON DELETE RESTRICT,
    locator         text NOT NULL,
    recorded        timestamptz NOT NULL DEFAULT now(),
    UNIQUE (object, crm_id, as_of, source_slug)
);

COMMENT ON TABLE crm_snapshots IS
    'One exported CRM record on one export date (migration 0036). export_key '
    'names the export it came from (the file path), so "absent from this '
    'export" is answerable. Compared by portolan.crm.changes.';

CREATE INDEX IF NOT EXISTS idx_crm_snapshots_record ON crm_snapshots (object, crm_id, as_of);
CREATE INDEX IF NOT EXISTS idx_crm_snapshots_account ON crm_snapshots (account_crm_id, as_of);

CREATE TABLE IF NOT EXISTS crm_changes (
    id              bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    object          text NOT NULL,
    crm_id          text NOT NULL,
    account_crm_id  text NOT NULL,
    field           text NOT NULL,
    kind            text NOT NULL CHECK (kind IN ('changed', 'added', 'removed')),
    old_value       text,
    new_value       text,
    as_of           date NOT NULL,
    prev_as_of      date,
    snapshot_id     bigint NOT NULL REFERENCES crm_snapshots (id) ON DELETE RESTRICT,
    prev_snapshot_id bigint REFERENCES crm_snapshots (id) ON DELETE RESTRICT,
    direction       smallint NOT NULL DEFAULT 0 CHECK (direction BETWEEN -1 AND 1),
    points          integer NOT NULL DEFAULT 0,
    details         jsonb NOT NULL DEFAULT '{}'::jsonb,
    claim_id        bigint REFERENCES claims (id) ON DELETE RESTRICT,
    recorded        timestamptz NOT NULL DEFAULT now()
);

COMMENT ON TABLE crm_changes IS
    'A CRM field change, or a record added to or dropped from an export, found '
    'by code (migration 0036). direction: +1 toward a close or better health, '
    '-1 away, 0 neutral. points: the rule-based size used to rank accounts.';

-- One change per record, field and newer snapshot: a re-run is a no-op, and an
-- export that arrives late (dated between two others) does not repeat a change.
CREATE UNIQUE INDEX IF NOT EXISTS crm_changes_once
    ON crm_changes (object, crm_id, field, kind, snapshot_id);
CREATE INDEX IF NOT EXISTS idx_crm_changes_account_as_of ON crm_changes (account_crm_id, as_of);
