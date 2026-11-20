"""Tests for `python -m portolan.db` (`portolan/db/__main__.py`), the
operational CLI the bug hunt 2026-09-18 P0 finding 2 fix added:
`migrate [--dry-run]` and `restore <file>`. `docker/entrypoint.sh`'s
`migrate` role execs `python -m portolan.db migrate` -- these tests call
`main()` in-process rather than shelling out, the same convention
`tests/test_lint_cli.py`-style CLI tests in this repo already use.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from portolan import db as db_module
from portolan.db.__main__ import main


def _reset_to_empty_schema(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("DROP SCHEMA public CASCADE")
        cur.execute("CREATE SCHEMA public")
    conn.commit()


@pytest.fixture()
def cli_env(db_url: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Points the CLI's own `get_settings()` (env-var driven, same as
    every other entry point in this repo) at the test database and a
    scratch STORE_ROOT."""
    monkeypatch.setenv("DATABASE_URL", db_url)
    monkeypatch.setenv("STORE_ROOT", str(tmp_path / "store"))
    return tmp_path


def test_migrate_dry_run_lists_pending_and_applies_nothing(
    db, db_url: str, cli_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _reset_to_empty_schema(db)
    rc = main(["migrate", "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "pending migration(s)" in out
    assert "0001_init.sql" in out
    assert db_module.applied_migrations(db) == set()


def test_migrate_dry_run_reports_none_pending_on_a_migrated_database(
    db, cli_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = main(["migrate", "--dry-run"])
    assert rc == 0
    assert "no pending migrations" in capsys.readouterr().out


def test_migrate_applies_pending_migrations_and_takes_a_backup(
    db, db_url: str, cli_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _reset_to_empty_schema(db)
    rc = main(["migrate"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "applied" in out
    assert db_module.pending_migrations(db) == []
    assert list((cli_env / "store" / "backups").glob("*.dump"))


def test_migrate_refuses_before_any_backup_when_the_volume_did_not_mount(
    db, cli_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Guardrails review 2026-09-22, row 9a: the backup goes under STORE_ROOT, so a
    volume that failed to mount must stop the migrate role before pg_dump runs."""
    _reset_to_empty_schema(db)
    monkeypatch.setenv("RAILWAY_VOLUME_MOUNT_PATH", str(cli_env / "volume-never-mounted"))
    rc = main(["migrate"])
    assert rc == 1
    assert not (cli_env / "store" / "backups").exists()
    assert len(db_module.pending_migrations(db)) == len(
        list(db_module.MIGRATIONS_DIR.glob("*.sql"))
    )


def test_restore_round_trips_through_the_cli(
    db, db_url: str, cli_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with db.cursor() as cur:
        cur.execute("CREATE TABLE _cli_restore_probe (id int PRIMARY KEY, note text)")
        cur.execute("INSERT INTO _cli_restore_probe (id, note) VALUES (1, 'before')")

    dump_path = db_module.backup_database(db_url, cli_env / "store")

    with db.cursor() as cur:
        cur.execute("UPDATE _cli_restore_probe SET note = 'after' WHERE id = 1")

    rc = main(["restore", str(dump_path)])
    assert rc == 0
    assert "restored" in capsys.readouterr().out

    conn2 = db_module.connect(db_url)
    try:
        with conn2.cursor() as cur:
            cur.execute("SELECT note FROM _cli_restore_probe WHERE id = 1")
            row = cur.fetchone()
    finally:
        conn2.close()
    assert row == ("before",)


def test_restore_of_a_missing_file_exits_non_zero_and_never_prints_the_dsn(
    db_url: str, cli_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = main(["restore", str(cli_env / "no-such-file.dump")])
    assert rc == 1
    captured = capsys.readouterr()
    assert "restored" not in captured.out


def test_restore_of_a_dump_without_a_checksum_file_needs_the_explicit_flag(
    db, db_url: str, cli_env: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with db.cursor() as cur:
        cur.execute("CREATE TABLE _cli_restore_probe (id int PRIMARY KEY, note text)")
        cur.execute("INSERT INTO _cli_restore_probe (id, note) VALUES (1, 'before')")
    dump_path = db_module.backup_database(db_url, cli_env / "store")
    db_module.checksum_path(dump_path).unlink()
    with db.cursor() as cur:
        cur.execute("UPDATE _cli_restore_probe SET note = 'after' WHERE id = 1")

    with caplog.at_level("WARNING"):
        assert main(["restore", str(dump_path)]) == 1
    assert "has no checksum file" in caplog.text
    with db.cursor() as cur:
        cur.execute("SELECT note FROM _cli_restore_probe WHERE id = 1")
        assert cur.fetchone() == ("after",)

    caplog.clear()
    with caplog.at_level("WARNING"):
        assert main(["restore", str(dump_path), "--no-verify-checksum"]) == 0
    assert "--no-verify-checksum" in caplog.text
    assert "WITHOUT checking" in caplog.text
    conn = db_module.connect(db_url)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT note FROM _cli_restore_probe WHERE id = 1")
            assert cur.fetchone() == ("before",)
    finally:
        conn.close()
