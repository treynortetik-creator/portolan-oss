"""Shared pytest fixtures for the Portolan test suite.

DB-backed tests need a real Postgres with pgvector. They use
``DATABASE_URL`` if set in the environment, else fall back to a
per-process ``portolan_test_<pid>`` database (see README "Dev setup"). If
Postgres is unreachable, or the vector extension cannot be created, the
``db`` fixture skips with a clear reason instead of failing -- this repo
has no Docker on the dev machine, so local Postgres is the only option
until CI runs the containerised path.
"""

from __future__ import annotations

import logging
import os
from urllib.parse import urlsplit, urlunsplit

import psycopg
import pytest

from portolan import db as db_module
from portolan.util.redact import redact_dsn
from tests import guards

DEFAULT_TEST_DATABASE_URL = "postgresql://localhost/portolan_test"

logger = logging.getLogger(__name__)


def pytest_configure(config: pytest.Config) -> None:
    """The guardrails in `tests/guards.py` (test database, network, paid-model keys, repo
    `var/`) normally arrive through `pyproject.toml`'s `-p tests.guards`. Register them
    here too, for a run that bypassed the ini file (`-c /dev/null`), and refuse a run that
    switched them off by name."""
    if config.pluginmanager.is_blocked("tests.guards"):
        pytest.exit(
            "Refusing to run the test suite: its guardrails were disabled with "
            "`-p no:tests.guards`. They keep tests off non-test databases, the network, "
            "paid-model keys and the repo's var/. Guard: tests/guards.py.",
            returncode=4,
        )
    if not config.pluginmanager.has_plugin("tests.guards"):
        config.pluginmanager.register(guards, "tests.guards")


def _drop_database_if_exists(url: str) -> None:
    """`DROP DATABASE` for `url`'s dbname, connecting to the server's
    `postgres` maintenance database to do it (mirrors `ensure_database_
    exists`'s own connect-to-`postgres`-first shape -- you can't drop the
    database you're connected to either). Best-effort session-end cleanup
    for the per-process database `db_url` creates below: swallows every
    `psycopg.Error` (a maintenance-connection failure, another session
    still holding an open connection to the target database, etc.) rather
    than failing the suite on the way out the door over a database that
    is, worst case, an orphan for a human to `dropdb` later -- not lost
    test data, since nothing about this database is meant to outlive the
    session that created it."""
    parts = urlsplit(url)
    dbname = parts.path.lstrip("/")
    maintenance_url = urlunsplit((parts.scheme, parts.netloc, "/postgres", "", ""))

    try:
        conn = psycopg.connect(maintenance_url, autocommit=True)
    except psycopg.OperationalError:
        return
    try:
        with conn.cursor() as cur:
            # Drop any other session's connections to this database first
            # (a leaked `db`/`db_conn` connection some test forgot to
            # close) -- `DROP DATABASE` fails outright while any other
            # backend is connected.
            cur.execute(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                "WHERE datname = %s AND pid <> pg_backend_pid()",
                (dbname,),
            )
            cur.execute(f'DROP DATABASE IF EXISTS "{dbname}"')
    except psycopg.Error as exc:
        logger.warning("could not drop per-process test database %r: %s", dbname, exc)
    finally:
        conn.close()


@pytest.fixture(scope="session")
def db_url():
    """The test database URL for this pytest process.

    **Per-process isolation (2026-09-18).** Parallel agents running the
    suite against the same fallback database collide on shared DDL: one
    process's `db` fixture runs `DROP SCHEMA public CASCADE` while
    another's migrations are still applying, `CREATE EXTENSION vector`
    races to create the extension twice, or a table (`dead_letter`) a
    third process's fixture just dropped is missing when the next test
    reaches for it -- all symptoms of two processes sharing one physical
    database, not a bug in the migrations or the tests themselves.

    When `DATABASE_URL` is explicitly set -- CI's own
    `postgresql://portolan:portolan@localhost:5432/portolan_test`, or a
    developer following `01-spec/build-contracts.md`'s "own test
    database" convention (`DATABASE_URL=postgresql://localhost/
    portolan_test_<yourmodule>`) -- it is used verbatim, exactly as
    before this fix: CI stays a single deterministic name, and a
    developer who already picked their own database name is not
    overridden. Only the *unconfigured* fallback (`DATABASE_URL` unset)
    gets a per-process suffix, so two agents that both forgot to set it
    no longer fight over the same `portolan_test` database -- they each
    get their own, created here and dropped again at session end.
    """
    configured = os.environ.get("DATABASE_URL")
    if configured:
        yield configured
        return

    url = with_database_name(DEFAULT_TEST_DATABASE_URL, f"portolan_test_{os.getpid()}")
    logger.info("DATABASE_URL not set; using per-process test database %r", _database_name(url))
    ensure_database_exists(url)
    # Export it for the rest of this process too: CLIs exercised in-process
    # (`portolan.lint.cli.main`, report/ingest entry points) open their own
    # connection through `portolan.db.connect()`, which reads DATABASE_URL
    # and otherwise falls back to `.env` -- the DEV database. Without this
    # the fixture and the code under test silently disagree about which
    # database "the graph" is (seen 2026-09-18: test_lint_cli found the dev
    # graph's company-context note instead of its own clean tables).
    os.environ["DATABASE_URL"] = url
    try:
        yield url
    finally:
        os.environ.pop("DATABASE_URL", None)
        _drop_database_if_exists(url)


def _database_name(url: str) -> str:
    """The database name out of a `postgresql://.../<name>` URL, ignoring
    any `?query` string."""
    path = urlsplit(url).path
    return path.rsplit("/", 1)[-1].split("?", 1)[0]


def with_database_name(url: str, dbname: str) -> str:
    """`url` with its path replaced by `/<dbname>`, preserving every other
    DSN component -- scheme, user, password, host, and port (``netloc``),
    plus any query string or fragment.

    A handful of fixtures need a second, differently-named database on
    the same server (`tests/test_pipeline.py`'s own
    `portolan_test_pipeline`, kept separate so the function-scoped `db`
    fixture's schema reset can't run underneath a module-scoped setup).
    Those fixtures used to build that second URL by hand
    (``f"postgresql://localhost/{dbname}"``), which silently dropped
    ``DATABASE_URL``'s user and password. That went unnoticed on a dev
    machine with trust-auth Postgres, but fails in CI, where Postgres
    requires them (``fe_sendauth: no password supplied``). Route every
    "same server, different database" derivation through this function
    instead so credentials always survive the swap.
    """
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, f"/{dbname}", parts.query, parts.fragment))


def ensure_database_exists(url: str) -> None:
    """`CREATE DATABASE` for `url`'s dbname if it doesn't exist yet,
    connecting to the server's `postgres` maintenance database to do it
    (you cannot run `CREATE DATABASE` against the database you're
    creating). Mirrors `tests/test_report_helpers.py`'s own
    `_ensure_database_exists`, which this shares no import with but is the
    same idea: a database derived by `with_database_name` (as opposed to
    the primary one Postgres already created for the run) is not
    guaranteed to exist yet."""
    parts = urlsplit(url)
    dbname = parts.path.lstrip("/")
    maintenance_url = urlunsplit((parts.scheme, parts.netloc, "/postgres", "", ""))

    conn = psycopg.connect(maintenance_url, autocommit=True)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (dbname,))
            if cur.fetchone() is None:
                cur.execute(f'CREATE DATABASE "{dbname}"')
    finally:
        conn.close()


@pytest.fixture()
def db(db_url: str):
    """A connection to a freshly-rebuilt schema, one per test.

    Drops and recreates the public schema (so every test starts from a
    clean, migrated database), applies all migrations, and yields an open
    connection. Skips the test with a clear reason if Postgres cannot be
    reached, or if the vector extension is not installed on the server.

    Review finding F18: `DROP SCHEMA public CASCADE` used to run
    unconditionally against whatever `DATABASE_URL` pointed at. Someone
    running the suite with a populated, non-test `DATABASE_URL` exported
    (a habit, a copy-pasted shell, a forgotten `export`) loses it. This
    fixture now refuses outright unless the database name contains
    "test" -- a real customer or dev database is never going to be named
    that by accident, so this is a cheap, effective guard, not a false
    sense of safety.
    """
    db_name = _database_name(db_url)
    if "test" not in db_name.lower():
        pytest.fail(
            f"refusing to run the `db` fixture against database {db_name!r} "
            f"(from DATABASE_URL={redact_dsn(db_url)!r}): its name does not contain "
            '"test". This fixture drops and recreates the public schema -- '
            "point DATABASE_URL at a database whose name contains "
            '"test" (e.g. `createdb portolan_test_<yourmodule>`) before '
            "running DB-backed tests."
        )

    try:
        conn = db_module.connect(db_url)
    except psycopg.OperationalError as exc:
        pytest.skip(f"Postgres unreachable at {redact_dsn(db_url)!r}: {exc}")
        return

    try:
        with conn.cursor() as cur:
            cur.execute("DROP SCHEMA public CASCADE")
            cur.execute("CREATE SCHEMA public")
        conn.commit()

        db_module.apply_migrations(conn)
    except psycopg.errors.UndefinedFile as exc:  # pragma: no cover - defensive
        conn.close()
        pytest.skip(f"pgvector extension file not found on this Postgres server: {exc}")
        return
    except psycopg.errors.FeatureNotSupported as exc:
        conn.close()
        pytest.skip(f"pgvector extension not installed on this Postgres server: {exc}")
        return

    # Autocommit so a constraint-violation test doesn't poison the
    # transaction for whatever the test does next -- each statement
    # commits or fails on its own.
    conn.autocommit = True
    yield conn
    conn.close()


@pytest.fixture()
def db_conn(db, db_url: str):
    """A SECOND connection to the exact schema `db` just migrated, opened
    the same way `portolan.report.weekly.main` opens its own -- `db_
    module.connect()` followed by `conn.autocommit = True`.

    **Should-fix 2 update** (`01-spec/reviews/review-theme-fix-wave-2.md`
    finding 2): the CLI's own connection used to be left at `db.connect()`
    's bare default (`autocommit=False`), and `_ensure_real_transaction`
    existed solely to paper over the SAVEPOINT trap that shape fell into.
    That helper is gone; the CLI now sets `conn.autocommit = True` right
    after connecting (the same convention every test fixture already
    used), so THIS fixture is updated to match -- a durability test that
    runs against a connection shape production no longer opens would be
    proving a shape nobody ships, not the thing that matters.

    **What "durable" still means, and how these tests still catch a
    regression** even though an autocommit connection is trivially
    `IDLE` between ordinary statements (unlike the pre-fix shape, which
    could sit `INTRANS` from an uncommitted `SELECT`): the atomicity
    tests (`test_merge_failure_midway_leaves_both_themes_intact_on_the_
    cli_shaped_connection`, `test_merge_is_atomic_when_a_step_fails_
    partway`) still fail if a future edit removes the `with conn.
    transaction():` wrapping from `_merge_themes`/`_mark_theme_resolved`/
    `_mark_theme_reopened` -- under autocommit, EACH bare statement
    commits on its own the moment it runs, so without that wrapping a
    mid-function failure would leave a genuinely partial write, not a
    rolled-back one; the wrapping is what turns a run of autocommitted
    statements into one atomic unit. The durability tests
    (`test_merge_commits_for_real_on_the_cli_shaped_connection_and_
    survives_reconnect`, `test_mark_theme_resolved_commits_for_real_on_
    the_cli_shaped_connection`) still fail if a future edit reintroduces
    a helper that swallows or defers the commit, or if `write_note`/
    `append_changelog`'s own `commit=False` plumbing inside these
    functions' transaction blocks regresses to calling `conn.commit()`
    while a `Transaction` context is open (psycopg3 raises `Programming
    Error` in that case, which these tests would surface immediately
    rather than silently losing data the way the pre-fix shape could).

    Depends on `db` only to get the schema migrated and to guarantee
    `DATABASE_URL` was already validated as a test database (`db`'s own
    guard) -- this fixture opens its OWN, independent connection rather
    than reusing `db`'s, so the two calling styles (module-scoped setup
    on `db`, the function-under-test on `db_conn`) can never share one
    connection's state by accident."""
    conn = db_module.connect(db_url)
    conn.autocommit = True
    yield conn
    conn.close()
