"""Tests for portolan/db/migrations/0003_standing_questions.sql.

Covers: the migration applies (via the shared `db` fixture, which runs
every `*.sql` file in the migrations directory), `standing_questions`
exists with the right shape, and the `usage_events.surface` check
constraint accepts `'mcp'` (and still rejects garbage).
"""

from __future__ import annotations

import psycopg
import pytest

from portolan import db as db_module


def test_migration_0003_is_applied(db):
    applied = db_module.applied_migrations(db)
    assert "0003_standing_questions.sql" in applied


def test_migrations_are_idempotent_including_0003(db):
    # The `db` fixture already applied every migration once. Re-applying
    # (including 0003's DROP CONSTRAINT / ADD CONSTRAINT) must be a no-op.
    assert db_module.apply_migrations(db) == []


def test_standing_questions_table_exists_with_expected_columns(db):
    with db.cursor() as cur:
        cur.execute(
            """
            SELECT column_name FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = 'standing_questions'
            """
        )
        columns = {row[0] for row in cur.fetchall()}
    assert {"id", "reader", "question", "created", "active"} <= columns


def test_usage_events_surface_accepts_mcp(db):
    with db.cursor() as cur:
        cur.execute(
            """
            INSERT INTO usage_events (tenant, surface, model_key)
            VALUES ('dev', 'mcp', 'mcp:search')
            """
        )
    with db.cursor() as cur:
        cur.execute("SELECT surface FROM usage_events WHERE model_key = 'mcp:search'")
        assert cur.fetchone() == ("mcp",)


def test_usage_events_surface_still_rejects_unknown_values(db):
    with pytest.raises(psycopg.errors.CheckViolation), db.cursor() as cur:
        cur.execute(
            """
                INSERT INTO usage_events (tenant, surface, model_key)
                VALUES ('dev', 'not-a-real-surface', 'mcp:search')
                """
        )


def test_usage_events_surface_still_accepts_pre_existing_values(db):
    # 0003 drops and re-adds the constraint; make sure the original seven
    # values (0001_init.sql) survived the round trip.
    for surface in ("extract", "merge", "consolidate", "audit", "chat", "report", "embed"):
        with db.cursor() as cur:
            cur.execute(
                "INSERT INTO usage_events (tenant, surface, model_key) VALUES ('dev', %s, 'x')",
                (surface,),
            )
