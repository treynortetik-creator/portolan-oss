"""Tests for the ops fixes to `portolan.web.app`/`portolan.web.deps`
(bug hunt 2026-09-18):

- P0 finding 1: one shared Postgres connection, never reconnected -- a
  dropped backend connection permanently 500'd every request. Fixed with
  a `psycopg_pool.ConnectionPool`; `test_pool_heals_after_the_backend_
  connection_is_terminated` reproduces the bug hunt's own repro
  ("terminated the app's backend pid") end to end against a real running
  app and a real Postgres, and proves the *next* request succeeds
  instead of joining every request after it in a permanent outage.
- P1 finding 7: no health endpoint. `/healthz` runs `SELECT 1` through
  the same pool every real request uses and returns 503 on failure.

A dedicated app per test (not the shared `client` fixture in
`test_web.py`, which points `get_conn` at one demo connection) --
`/healthz` reads `request.app.state.pool` directly, so these tests need
the app's own real lifespan-built pool, not an overridden connection.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import psycopg
import pytest
from fastapi.testclient import TestClient

from portolan.config import Settings
from portolan.store import StoreRootError
from portolan.web.app import create_app

ORG_KEY = "test-org-key-for-web-ops-tests"
SESSION_SECRET = "test-session-secret-for-web-ops-tests-not-a-real-secret"


def _settings(db_url: str, store_root: Path) -> Settings:
    return Settings(
        database_url=db_url,
        tenant="test",
        store_root=store_root,
        llm_provider="fake",
        embed_provider="fake",
        org_key=ORG_KEY,
        session_secret=SESSION_SECRET,
        cookie_secure=False,
        db_pool_min_size=1,
        db_pool_max_size=2,
    )


@pytest.fixture()
def store_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("web-ops-store")


def test_healthz_returns_ok_when_the_database_is_reachable(db_url: str, store_root: Path) -> None:
    app = create_app(_settings(db_url, store_root))
    with TestClient(app) as client:
        r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_healthz_is_unauthenticated(db_url: str, store_root: Path) -> None:
    """No session cookie is sent -- must not redirect to /login the way
    every other route does."""
    app = create_app(_settings(db_url, store_root))
    with TestClient(app) as client:
        r = client.get("/healthz", follow_redirects=False)
    assert r.status_code == 200


def test_healthz_returns_503_when_the_pool_cannot_reach_the_database(
    db_url: str, store_root: Path
) -> None:
    class _BrokenPool:
        @contextmanager
        def connection(self):
            raise psycopg.OperationalError("simulated: connection refused")
            yield  # pragma: no cover -- unreachable, keeps this a generator

    app = create_app(_settings(db_url, store_root))
    with TestClient(app) as client:
        assert client.get("/healthz").status_code == 200
        app.state.pool = _BrokenPool()
        r = client.get("/healthz")
    assert r.status_code == 503
    assert r.json() == {"status": "error"}


def test_pool_heals_after_the_backend_connection_is_terminated(
    db_url: str, store_root: Path
) -> None:
    """Bug hunt 2026-09-18, P0 finding 1, reproduced end to end: the doc's
    own repro was `terminated the app's backend pid` against a running
    app, then watched every subsequent request 500 forever. Here, every
    Postgres backend belonging to this test's own database is terminated
    the same way (`pg_terminate_backend`, scoped to `current_database()`
    so this never touches another test process's database on a shared
    local Postgres server) -- the connection pool must detect the dead
    connection on the next checkout and hand back a fresh, working one
    instead of the same broken object forever.
    """
    app = create_app(_settings(db_url, store_root))
    with TestClient(app) as client:
        first = client.get("/healthz")
        assert first.status_code == 200

        admin = psycopg.connect(db_url, autocommit=True)
        try:
            with admin.cursor() as cur:
                cur.execute(
                    "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                    "WHERE datname = current_database() AND pid <> pg_backend_pid()"
                )
                terminated = cur.fetchall()
        finally:
            admin.close()
        assert terminated, "expected at least one pooled backend connection to terminate"

        healed = client.get("/healthz")
    assert healed.status_code == 200, (
        "a request after the backend connection was terminated should heal via the pool, "
        "not join every subsequent request in a permanent outage"
    )


def test_startup_refuses_before_any_backup_or_migration_when_the_volume_did_not_mount(
    db: psycopg.Connection,
    db_url: str,
    store_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Guardrails review 2026-09-22, row 9a: the pre-migration dump is written under
    store_root, so the volume check must run first. Before the fix the dump went to the
    container's own disk, the migrations applied, and only then did startup refuse."""
    with db.cursor() as cur:
        cur.execute("DROP SCHEMA public CASCADE")
        cur.execute("CREATE SCHEMA public")
    db.commit()
    monkeypatch.setenv("RAILWAY_VOLUME_MOUNT_PATH", str(tmp_path / "volume-never-mounted"))

    app = create_app(_settings(db_url, store_root))
    with pytest.raises(StoreRootError, match="did not mount"), TestClient(app):
        pass

    with db.cursor() as cur:
        cur.execute("SELECT to_regclass('public.schema_migrations')")
        assert cur.fetchone()[0] is None, "a migration ran before the volume check"
    assert not (store_root / "backups").exists(), "a backup was written before the check"
