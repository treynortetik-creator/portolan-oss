-- Portolan schema, migration 0003: standing questions + the `mcp` surface
--
-- Owned by wave C4 (portolan/mcp_server/**). Spec references:
-- 01-spec/portolan-spec-v2.md sec 10 ("MCP, second. Search, expand,
-- open-note, open-source, add-standing-question, as tools. Runs on the
-- customer's own AI; costs them nothing from us.") and sec 11 (metering:
-- "Every model call goes through one wrapper writing tenant, surface,
-- model, tokens, cost."). Contract: 01-spec/build-contracts.md "Migration
-- ledger" and "Web and MCP".
--
-- IDEMPOTENT, like 0001_init.sql and 0002_index_state.sql: IF NOT EXISTS /
-- DROP-then-ADD throughout, so this file can be re-applied safely.

-- ---------------------------------------------------------------------
-- standing_questions: a reader's saved question, answered on a cadence by
-- the weekly report (spec sec 7, sec 10) and readable/writable over MCP
-- (`add_standing_question`, `list_standing_questions`). Not itself an
-- answer -- just the durable ask. `active = false` retires a question
-- without deleting it (spec sec 8: nothing is hard-deleted).
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS standing_questions (
    id        bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    reader    text NOT NULL,
    question  text NOT NULL,
    created   timestamptz NOT NULL DEFAULT now(),
    active    boolean NOT NULL DEFAULT true
);

COMMENT ON TABLE standing_questions IS
    'A reader''s standing question, added via the MCP tool or the weekly '
    'report surface (spec sec 7, sec 10). active=false retires it without '
    'deleting the row.';

CREATE INDEX IF NOT EXISTS idx_standing_questions_reader ON standing_questions (reader);
CREATE INDEX IF NOT EXISTS idx_standing_questions_active ON standing_questions (active);

-- ---------------------------------------------------------------------
-- usage_events.surface: add 'mcp' (every MCP tool call writes a usage_events
-- row with this surface -- build instructions for wave C4). The existing
-- constraint is Postgres's default auto-generated name for an unnamed,
-- inline `CHECK` on a single column: `<table>_<column>_check`. Confirmed
-- against a freshly-migrated 0001_init.sql database (2026-09-12):
--
--   conname                    | pg_get_constraintdef
--   usage_events_surface_check | CHECK ((surface = ANY (ARRAY['extract', 'merge',
--                                 'consolidate', 'audit', 'chat', 'report', 'embed'])))
--
-- DROP + re-ADD under that same name (rather than a new constraint name)
-- so there is exactly one surface constraint on the table, matching how
-- 0001_init.sql declared it in the first place.
-- ---------------------------------------------------------------------
ALTER TABLE usage_events DROP CONSTRAINT IF EXISTS usage_events_surface_check;
ALTER TABLE usage_events ADD CONSTRAINT usage_events_surface_check CHECK (
    surface IN ('extract', 'merge', 'consolidate', 'audit', 'chat', 'report', 'embed', 'mcp')
);
