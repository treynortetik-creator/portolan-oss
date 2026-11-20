"""`portolan/db/migrations/0005_themes_reports.sql` applies twice
(`01-spec/build-contracts.md` "Ground rules"; every migration in this
ledger is idempotent -- 0002/0003/0004 all say so in their own headers).

Two different senses of "applies twice", both checked: `apply_migrations`
tracks filenames in `schema_migrations` and simply skips a file it has
already recorded (so that path alone would pass trivially even if the SQL
itself were not idempotent) -- so this also executes the raw file content
a second time, directly, against a live schema that already has it
applied, the way `0002`/`0003`'s own `IF NOT EXISTS` / `DROP + ADD`
idioms are meant to survive.

Runs in its own throwaway Postgres *schema* (not `public`) on the shared
`portolan_test_report` database, specifically so it never touches the
`public` schema `tests/test_report_helpers.py`'s session-scoped
`full_graph` fixture builds its one expensive graph in -- other
`test_report_*.py` modules depend on that graph surviving the whole test
session.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import psycopg
import pytest

from portolan import db as db_module

MIGRATION_PATH = (
    Path(__file__).resolve().parent.parent
    / "portolan"
    / "db"
    / "migrations"
    / "0005_themes_reports.sql"
)


@pytest.fixture()
def scratch_conn():
    import os

    database_url = os.environ.get("DATABASE_URL", "postgresql://localhost/portolan_test_report")
    try:
        conn = db_module.connect(database_url)
    except psycopg.OperationalError as exc:
        pytest.skip(f"Postgres unreachable at {database_url!r}: {exc}")
        return

    schema = f"migration_test_{uuid.uuid4().hex[:12]}"
    with conn.cursor() as cur:
        cur.execute(f'CREATE SCHEMA "{schema}"')
        # `public` stays on the search path (after the scratch schema) so
        # the `vector` type -- an extension already created in `public` by
        # the shared database's own `public`-schema migration history --
        # stays visible; new objects still land in the scratch schema
        # first since it comes first.
        cur.execute(f'SET search_path TO "{schema}", public')
    conn.commit()
    conn.autocommit = True
    try:
        yield conn, schema
    finally:
        with conn.cursor() as cur:
            cur.execute(f'DROP SCHEMA "{schema}" CASCADE')
        conn.close()


def test_apply_migrations_is_a_no_op_the_second_time(scratch_conn) -> None:
    conn, _schema = scratch_conn
    applied_first = db_module.apply_migrations(conn)
    assert "0005_themes_reports.sql" in applied_first
    applied_second = db_module.apply_migrations(conn)
    assert applied_second == []


def test_0005_sql_is_idempotent_executed_raw_twice(scratch_conn) -> None:
    conn, schema = scratch_conn
    db_module.apply_migrations(conn)  # get 0001-0004 in place first
    sql = MIGRATION_PATH.read_text()
    with conn.cursor() as cur:
        cur.execute(sql)
        cur.execute(sql)  # must not raise

    with conn.cursor() as cur:
        cur.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = %s AND table_name = ANY(%s)",
            (schema, ["theme_members", "reports", "standing_question_answers"]),
        )
        tables = {r[0] for r in cur.fetchall()}
    assert tables == {"theme_members", "reports", "standing_question_answers"}
