"""`python -m portolan.db.rehearse` and `portolan.db.fingerprint` (checklist item 15).

The rehearsal restores a backup into a second database on the same server; these tests run it
against the session's test database, so its target is `<session db>_restore_rehearsal`. The
current database must come out of every rehearsal exactly as it went in.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from portolan import db as db_module
from portolan.db import rehearse as rehearse_module
from portolan.db.fingerprint import compare, fingerprint

REPO_ROOT = Path(__file__).resolve().parent.parent


def _database_exists(db_url: str, name: str) -> bool:
    conn = db_module.connect(db_url)
    try:
        row = conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (name,)).fetchone()
    finally:
        conn.close()
    return row is not None


def _probe(db, note: str = "before") -> None:
    with db.cursor() as cur:
        cur.execute("CREATE TABLE _rehearsal_probe (id int PRIMARY KEY, note text)")
        cur.execute("INSERT INTO _rehearsal_probe VALUES (1, %s)", (note,))


def test_rehearsal_restores_a_fresh_backup_beside_the_database_and_drops_it(
    db, db_url, tmp_path: Path
):
    _probe(db)
    before = fingerprint(db_url)

    result = rehearse_module.rehearse(db_url, tmp_path)

    assert result["ok"], result["comparison"]
    assert result["comparison"]["equal"]
    assert result["restored_rows"] == sum(t["rows"] for t in before["tables"].values()) > 0
    assert result["pending_migrations_at_cutover"] == []
    # Its own backup directory: the deploy backups' retention is untouched.
    assert Path(result["dump"]).parent == tmp_path / "restore-rehearsal" / "backups"
    assert not (tmp_path / "backups").exists()
    assert not _database_exists(db_url, result["target"])
    assert compare(before, fingerprint(db_url))["equal"]


def test_rehearsal_of_an_older_dump_keeps_the_target_for_a_cutover(db, db_url, tmp_path: Path):
    _probe(db, "at backup time")
    dump = db_module.backup_database(db_url, tmp_path)
    with db.cursor() as cur:
        cur.execute("UPDATE _rehearsal_probe SET note = 'after backup'")
    target = f"{rehearse_module.database_name(db_url)}_cutover"

    try:
        result = rehearse_module.rehearse(db_url, tmp_path, dump=dump, target=target, keep=True)
        assert result["ok"] and result["kept"] and result["comparison"] is None
        conn = db_module.connect(rehearse_module.with_database(db_url, target))
        try:
            restored = conn.execute("SELECT note FROM _rehearsal_probe").fetchone()
        finally:
            conn.close()
        assert restored == ("at backup time",)
        with db.cursor() as cur:
            cur.execute("SELECT note FROM _rehearsal_probe")
            assert cur.fetchone() == ("after backup",)
    finally:
        rehearse_module._drop_target(db_url, target)


def test_rehearsal_refuses_a_dump_without_a_checksum_unless_told_not_to_check(
    db, db_url, tmp_path: Path
):
    _probe(db)
    dump = db_module.backup_database(db_url, tmp_path)
    db_module.checksum_path(dump).unlink()
    target = f"{rehearse_module.database_name(db_url)}{rehearse_module.REHEARSAL_SUFFIX}"

    with pytest.raises(db_module.BackupError, match="has no checksum file"):
        rehearse_module.rehearse(db_url, tmp_path, dump=dump)
    assert not _database_exists(db_url, target)

    result = rehearse_module.rehearse(db_url, tmp_path, dump=dump, check_checksum=False)
    assert result["ok"] and result["checksum_verified"] is False
    assert "NOT CHECKED" in rehearse_module.report(result)
    assert not _database_exists(db_url, target)


def test_rehearsal_refuses_the_current_database_and_an_existing_one(db, db_url, tmp_path: Path):
    current = rehearse_module.database_name(db_url)
    with pytest.raises(rehearse_module.RehearsalError, match="current database"):
        rehearse_module.rehearse(db_url, tmp_path, target=current)

    # `postgres` exists on every server and is not a leftover rehearsal: never dropped.
    with pytest.raises(rehearse_module.RehearsalError, match="already exists"):
        rehearse_module.rehearse(db_url, tmp_path, target="postgres")
    assert _database_exists(db_url, "postgres")


def test_rehearsal_cli_runs_in_a_child_process(db, db_url, tmp_path: Path):
    """The command the runbook runs in the web container, with its environment."""
    _probe(db)
    env = {k: v for k, v in os.environ.items() if not k.endswith("_API_KEY")}
    env.update(DATABASE_URL=db_url, STORE_ROOT=str(tmp_path))

    def run(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "portolan.db.rehearse", *args],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )

    proc = run()
    assert proc.returncode == 0, proc.stderr
    assert "PASS restore rehearsal" in proc.stdout
    assert "| row checksums (primary-key order) |" in proc.stdout
    assert "| NO:" not in proc.stdout

    backup = run("--backup-only")
    assert backup.returncode == 0, backup.stderr
    assert Path(backup.stdout.split()[0]).is_file()

    listed = run("--list")
    assert listed.returncode == 0, listed.stderr
    assert str(tmp_path / "restore-rehearsal" / "backups") in listed.stdout
    assert listed.stdout.strip().endswith("2 backup(s)")
    assert listed.stdout.count("checksum file present") == 2


def test_fingerprint_is_blind_to_nothing_it_claims_to_check(db, db_url):
    _probe(db)
    first = fingerprint(db_url)
    with db.cursor() as cur:
        cur.execute("UPDATE _rehearsal_probe SET note = 'changed'")
    rows = {row["check"]: row for row in compare(first, fingerprint(db_url))["rows"]}
    assert rows["row checksums (primary-key order)"]["differs"] == ["_rehearsal_probe"]
    assert rows["row counts"]["equal"]

    with db.cursor() as cur:
        cur.execute("CREATE INDEX _rehearsal_probe_note ON _rehearsal_probe (note)")
    second = fingerprint(db_url)
    with db.cursor() as cur:
        cur.execute("DROP INDEX _rehearsal_probe_note")
    rows = {row["check"]: row for row in compare(second, fingerprint(db_url))["rows"]}
    assert rows["schema objects"]["differs"] == ["indexes"]
