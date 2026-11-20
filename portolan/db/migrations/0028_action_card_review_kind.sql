-- Portolan schema, migration 0028: the action researcher (01-spec/
-- actions-and-owners-2026-09-19.md Phase 2).
--
-- `review_queue.kind` widens for 'action_card' -- one row per grounded
-- `ActionCard` the researcher returns (`portolan.report.researcher.
-- ActionCard`), payload = the card itself plus its candidate. Confirm
-- creates the action (owner, due, status open, source/locator from the
-- first evidence claim); Edit lets a human change owner or due before
-- confirming; Dismiss records why. Same widening pattern as 0013/0026/
-- 0027.
--
-- `usage_events.surface` widens for 'researcher' -- one row per
-- `Researcher.run()` call (tool-call count, wall time, cost in `extra`),
-- written directly the same way `portolan.mcp_server.tools._log_usage`
-- already writes `surface='mcp'` rows: a raw INSERT, not through
-- `portolan.metering.metered()` (whose `Surface` Literal is a closed,
-- token-count-shaped contract for a single model completion -- a multi-
-- turn agent run with tool calls doesn't fit that shape, the same reason
-- `tools.py` never routed MCP calls through it either).
--
-- IDEMPOTENT, like every prior migration file: DROP-then-ADD for both
-- CHECK constraints, safe to re-apply (tests/test_db_schema.py applies
-- every migration file twice).

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
                 'action_card', 'other')
    );
END $$;

DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'usage_events_surface_check' AND conrelid = 'usage_events'::regclass
    ) THEN
        ALTER TABLE usage_events DROP CONSTRAINT usage_events_surface_check;
    END IF;
    ALTER TABLE usage_events ADD CONSTRAINT usage_events_surface_check CHECK (
        surface IN ('extract', 'merge', 'consolidate', 'audit', 'chat', 'report',
                    'embed', 'mcp', 'researcher')
    );
END $$;
