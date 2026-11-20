-- Portolan schema, migration 0031: the per-tenant AI-access switch for
-- the MCP door (01-spec/app-checklist-2026-09-18.md item 26(b); idea from
-- 01-spec/research/openworkgraph-2026-09-22.md sec 4, borrowable #3,
-- reimplemented, nothing copied).
--
-- ai_access: one row per tenant name (`Settings.tenant`, the same name
-- `usage_events.tenant` carries), saying whether that tenant's MCP door
-- (`portolan.mcp_server.create_server`, both the stdio and the hosted
-- `/mcp` transports) may answer at all. The door reads this row before
-- every tool call, tools/list, resources/list and resources/read -- no
-- cache -- so flipping it off closes the door on the very next request,
-- without a redeploy and without rotating PORTOLAN_MCP_ORG_KEY.
--
-- Lookup order (`portolan.mcp_server.access.get_ai_access`): the tenant's
-- own row wins; with no row of its own, the database-default row
-- `tenant = '*'` applies; with neither, the door is CLOSED. "Off until
-- turned on" is therefore the rule for every brand-new database.
--
-- GRANDFATHERING (the live demo's door must not close on deploy). This
-- migration inserts the database-default row `'*'` = ON, but only when
-- the database already existed before this migration ran -- i.e. it is
-- an existing tenant's database being upgraded, not a new one being
-- built from scratch. "Already existed" is either signal below:
--
--   1. `portolan.migrating_existing_database` = 'true'. The migration
--      runner (`portolan.db.apply_migrations`) sets this transaction-local
--      setting before every file it applies: 'true' when
--      `schema_migrations` already had rows before this run started,
--      'false' when the run began on an empty `schema_migrations` (a new
--      database). Exact, and independent of whether the database holds
--      any data yet.
--   2. The database already holds anything a client could have been
--      reading: a note, a source, a claim, a standing question, or a
--      metered usage row. The fallback for a file applied by hand (psql),
--      where the runner's setting is absent.
--
-- A brand-new database satisfies neither (the runner says 'false' and
-- every table was created moments ago, empty), so it gets no '*' row and
-- its door stays closed until someone runs
-- `python -m portolan.mcp_server access on`.
--
-- ai_access_log: append-only history of every change to ai_access,
-- including this migration's own seed row (who, when, on/off, why).
-- Nothing is hard-deleted (0001_init.sql header).
--
-- IDEMPOTENT: CREATE TABLE IF NOT EXISTS, and the seed is ON CONFLICT
-- DO NOTHING -- re-running never overwrites a switch someone has since
-- turned off, and never writes a second log row.

CREATE TABLE IF NOT EXISTS ai_access (
    tenant      text PRIMARY KEY CHECK (length(tenant) BETWEEN 1 AND 200),
    enabled     boolean NOT NULL,
    changed_at  timestamptz NOT NULL DEFAULT now(),
    changed_by  text NOT NULL CHECK (length(changed_by) BETWEEN 1 AND 200),
    reason      text CHECK (reason IS NULL OR length(reason) <= 2000)
);

COMMENT ON TABLE ai_access IS
    'Per-tenant AI-access switch for the MCP door (checklist item 26b). '
    'The tenant''s own row wins; tenant = ''*'' is the database default; '
    'no row at all means the door is closed.';

CREATE TABLE IF NOT EXISTS ai_access_log (
    id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    tenant      text NOT NULL,
    enabled     boolean NOT NULL,
    changed_at  timestamptz NOT NULL DEFAULT now(),
    changed_by  text NOT NULL,
    reason      text
);

COMMENT ON TABLE ai_access_log IS
    'Append-only history of every ai_access change: who, when, on/off, why.';

CREATE INDEX IF NOT EXISTS idx_ai_access_log_tenant_changed
    ON ai_access_log (tenant, changed_at);

WITH existing AS (
    SELECT (
        coalesce(current_setting('portolan.migrating_existing_database', true), '') = 'true'
        OR EXISTS (SELECT 1 FROM notes)
        OR EXISTS (SELECT 1 FROM sources)
        OR EXISTS (SELECT 1 FROM claims)
        OR EXISTS (SELECT 1 FROM standing_questions)
        OR EXISTS (SELECT 1 FROM usage_events)
    ) AS yes
),
seeded AS (
    INSERT INTO ai_access (tenant, enabled, changed_by, reason)
    SELECT '*', true, 'migration:0031_ai_access',
           'Database existed before the AI-access switch was introduced; '
           'grandfathered ON so an already-connected MCP client keeps working.'
    FROM existing
    WHERE existing.yes
    ON CONFLICT (tenant) DO NOTHING
    RETURNING tenant, enabled, changed_at, changed_by, reason
)
INSERT INTO ai_access_log (tenant, enabled, changed_at, changed_by, reason)
SELECT tenant, enabled, changed_at, changed_by, reason FROM seeded;
