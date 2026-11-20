"""Tests for the pre-migration backup / advisory-lock path
(`portolan.db.backup_database`, `pending_migrations`, `migrate_with_backup`,
`restore_database`) added by the 2026-09-18 ops bug hunt:

- P0 finding 2: migrations used to auto-apply with no snapshot and no
  restore path -- a destructive migration ran unattended with nothing to
  roll back to.
- P1 finding 3: two replicas starting at once could both see the same
  migration file as unapplied and both run it (no advisory lock).

Uses the `db`/`db_url` fixtures from `tests/conftest.py` -- a real,
freshly-migrated-then-reset Postgres. Several tests here reset the schema
back to empty (`DROP SCHEMA public CASCADE`) so there is something
pending to back up and apply; that mirrors what `db` itself already does
before yielding a connection.
"""

from __future__ import annotations

import hashlib
import threading
from pathlib import Path

import psycopg
import pytest

from portolan import db as db_module


def _reset_to_empty_schema(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute("DROP SCHEMA public CASCADE")
        cur.execute("CREATE SCHEMA public")
    conn.commit()


# --------------------------------------------------------------------
# pending_migrations
# --------------------------------------------------------------------


def test_pending_migrations_lists_every_file_against_an_empty_schema(db, db_url):
    _reset_to_empty_schema(db)
    pending = db_module.pending_migrations(db)
    on_disk = sorted(p.name for p in db_module.MIGRATIONS_DIR.glob("*.sql"))
    assert pending == on_disk
    assert pending, "expected at least one migration file on disk"
    # Read-only: nothing was actually applied.
    assert db_module.applied_migrations(db) == set()


def test_pending_migrations_is_empty_once_everything_is_applied(db):
    # The `db` fixture already migrated this schema.
    assert db_module.pending_migrations(db) == []


# --------------------------------------------------------------------
# backup_database
# --------------------------------------------------------------------


def test_backup_database_writes_a_restorable_custom_format_dump(db_url, tmp_path: Path):
    dump_path = db_module.backup_database(db_url, tmp_path)
    assert dump_path.is_file()
    assert dump_path.parent == tmp_path / "backups"
    assert dump_path.suffix == ".dump"
    assert dump_path.stat().st_size > 0
    # pg_restore's own custom-format magic bytes -- proves this is a real
    # dump pg_restore/restore_database could load, not just a non-empty
    # file pg_dump happened to leave behind.
    assert dump_path.read_bytes()[:5] == b"PGDMP"


def test_backup_database_prunes_to_the_retention_count(db_url, tmp_path: Path):
    for _ in range(4):
        db_module.backup_database(db_url, tmp_path, retention=3)
    dumps = sorted((tmp_path / "backups").glob("*.dump"))
    assert len(dumps) == 3


def test_backup_database_raises_backup_error_when_pg_dump_is_missing(
    db_url, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Bug hunt 2026-09-18, P0 finding 2's own Dockerfile requirement:
    # "if pg_dump is not in the image today, add it." This proves the
    # code path that would have caught its absence -- fails loudly with
    # BackupError, never silently proceeds to apply a migration anyway.
    #
    # Follow-up (pg_dump/pg_restore server-version mismatch fix):
    # `_resolve_pg_binary` now also checks well-known Homebrew/PGDG
    # locations before falling back to PATH, so emptying PATH alone no
    # longer simulates "pg_dump missing" on a dev machine that has one of
    # those installed (this repo's own dev Mac does). Blanking
    # `_pg_bin_search_dirs` too makes this "missing everywhere", the
    # scenario the test is actually about.
    monkeypatch.setattr(db_module, "_pg_bin_search_dirs", lambda server_major: [])
    monkeypatch.setenv("PATH", str(tmp_path / "empty-bin"))
    (tmp_path / "empty-bin").mkdir()
    with pytest.raises(db_module.BackupError, match="pg_dump"):
        db_module.backup_database(db_url, tmp_path / "store")


def test_backup_database_never_leaks_the_password_in_its_error_message(
    tmp_path: Path,
) -> None:
    # Nothing listens on port 1; the name keeps tests/guards.py's test-database rule.
    bad_url = "postgresql://portolan:super-secret-password@localhost:1/portolan_test_no_such_db"
    with pytest.raises(db_module.BackupError) as excinfo:
        db_module.backup_database(bad_url, tmp_path)
    assert "super-secret-password" not in str(excinfo.value)


# --------------------------------------------------------------------
# migrate_with_backup
# --------------------------------------------------------------------


def test_migrate_with_backup_backs_up_then_applies_when_migrations_are_pending(
    db, db_url, tmp_path: Path
):
    _reset_to_empty_schema(db)
    applied = db_module.migrate_with_backup(db, database_url=db_url, store_root=tmp_path)
    on_disk = sorted(p.name for p in db_module.MIGRATIONS_DIR.glob("*.sql"))
    assert applied == on_disk
    dumps = list((tmp_path / "backups").glob("*.dump"))
    assert len(dumps) == 1, "expected exactly one backup taken before applying"


def test_migrate_with_backup_logs_which_migrations_it_applied(
    db, db_url, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Bug hunt 2026-09-18, P2 finding 13: migrations used to apply
    silently -- the deploy log showed nothing naming which schema change
    ran on which deploy. `migrate_with_backup` must log the applied
    filenames at INFO."""
    _reset_to_empty_schema(db)
    with caplog.at_level("INFO", logger="portolan.db"):
        applied = db_module.migrate_with_backup(db, database_url=db_url, store_root=tmp_path)
    assert applied
    joined = "\n".join(record.message for record in caplog.records)
    assert "applied" in joined
    assert applied[0] in joined
    assert applied[-1] in joined


def test_migrate_with_backup_is_idempotent_and_skips_the_backup_when_nothing_is_pending(
    db, db_url, tmp_path: Path
):
    # `db` is already fully migrated -- nothing pending, so no backup.
    applied = db_module.migrate_with_backup(db, database_url=db_url, store_root=tmp_path)
    assert applied == []
    assert not (tmp_path / "backups").exists() or not list((tmp_path / "backups").glob("*.dump"))


def test_migrate_with_backup_leaves_the_connection_with_no_open_transaction(
    db, db_url, tmp_path: Path
):
    _reset_to_empty_schema(db)
    was_autocommit = db.autocommit
    db.autocommit = False
    try:
        db_module.migrate_with_backup(db, database_url=db_url, store_root=tmp_path)
        # psycopg3 raises if you flip autocommit while a transaction is
        # still open -- this is the same guard app.py's old comment
        # described ("psycopg3 refuses mid-transaction").
        db.autocommit = True
    finally:
        db.autocommit = was_autocommit


def test_two_replicas_migrating_at_once_do_not_double_apply(db_url, tmp_path: Path):
    """Bug hunt 2026-09-18, P1 finding 3, reproduced exactly: two
    processes (here, two threads each with their own connection --
    psycopg releases the GIL during the network round trip, so this
    genuinely races the advisory lock at the database, not just in
    Python) released from a barrier at the same instant. Before the
    advisory lock existed, both could see the same file as unapplied and
    both run it -- the bug hunt's own reproduction crashed one replica on
    a duplicate-key error from the second one racing ahead. With the
    lock, one serializes behind the other; both finish without error, and
    the migration set is applied exactly once.
    """
    conn_a = db_module.connect(db_url)
    conn_b = db_module.connect(db_url)
    _reset_to_empty_schema(conn_a)

    barrier = threading.Barrier(2)
    results: dict[str, tuple[list[str] | None, Exception | None]] = {}

    def _run(name: str, conn: psycopg.Connection) -> None:
        barrier.wait()
        try:
            applied = db_module.migrate_with_backup(
                conn, database_url=db_url, store_root=tmp_path / name
            )
            results[name] = (applied, None)
        except Exception as exc:  # noqa: BLE001 -- captured for the assertion below
            results[name] = (None, exc)

    t_a = threading.Thread(target=_run, args=("replica-a", conn_a))
    t_b = threading.Thread(target=_run, args=("replica-b", conn_b))
    t_a.start()
    t_b.start()
    t_a.join(timeout=60)
    t_b.join(timeout=60)

    try:
        for name, (applied, exc) in results.items():
            assert exc is None, f"{name} raised: {exc!r}"
            assert applied is not None

        on_disk = sorted(p.name for p in db_module.MIGRATIONS_DIR.glob("*.sql"))
        combined = sorted(results["replica-a"][0] + results["replica-b"][0])
        # Applied exactly once, total, across both replicas -- not once
        # each (which is the crash/double-apply the bug hunt reproduced).
        assert combined == on_disk
        assert db_module.pending_migrations(conn_a) == []
    finally:
        conn_a.close()
        conn_b.close()


# --------------------------------------------------------------------
# restore_database
# --------------------------------------------------------------------


def test_restore_database_round_trips_a_backup(db, db_url, tmp_path: Path):
    # A scratch table the test fully controls, rather than depending on
    # any application table already having rows -- `db` is a freshly
    # migrated, otherwise-empty schema. `backup_database` dumps the whole
    # database regardless, so this proves the same round trip a real
    # `notes`/`claims` row would take.
    with db.cursor() as cur:
        cur.execute("CREATE TABLE _restore_probe (id int PRIMARY KEY, note text)")
        cur.execute("INSERT INTO _restore_probe (id, note) VALUES (1, 'before restore')")

    dump_path = db_module.backup_database(db_url, tmp_path)

    with db.cursor() as cur:
        cur.execute("UPDATE _restore_probe SET note = 'mutated after backup' WHERE id = 1")

    db_module.restore_database(dump_path, db_url)

    conn2 = db_module.connect(db_url)
    try:
        with conn2.cursor() as cur:
            cur.execute("SELECT note FROM _restore_probe WHERE id = 1")
            row = cur.fetchone()
    finally:
        conn2.close()

    assert row == ("before restore",)


def test_restore_database_raises_backup_error_for_a_missing_file(db_url, tmp_path: Path):
    with pytest.raises(db_module.BackupError, match="no such backup file"):
        db_module.restore_database(tmp_path / "does-not-exist.dump", db_url)


def _row_counts(conn: psycopg.Connection) -> dict[str, int]:
    with conn.cursor() as cur:
        cur.execute("SELECT tablename FROM pg_tables WHERE schemaname = 'public' ORDER BY 1")
        tables = [row[0] for row in cur.fetchall()]
        counts = {}
        for table in tables:
            cur.execute(f'SELECT count(*) FROM public."{table}"')
            counts[table] = cur.fetchone()[0]
    return counts


def test_restore_database_refuses_a_truncated_dump_and_leaves_the_target_unchanged(
    db, db_url, tmp_path: Path
):
    # Enough rows that the dump's second half is table data, not schema: cut there, a restore
    # without the pre-read drops and recreates the tables, then runs out of data to refill them.
    with db.cursor() as cur:
        cur.execute("CREATE TABLE _restore_probe (id int PRIMARY KEY, note text)")
        cur.execute(
            "INSERT INTO _restore_probe (id, note) "
            "SELECT g, repeat(md5(g::text), 20) FROM generate_series(1, 5000) g"
        )
    dump_path = db_module.backup_database(db_url, tmp_path)
    truncated = tmp_path / "truncated.dump"
    truncated.write_bytes(dump_path.read_bytes()[: dump_path.stat().st_size // 2])
    before = _row_counts(db)
    assert before["schema_migrations"] > 0

    # The read-through layer, which is what guards a dump older than checksums.
    with pytest.raises(db_module.BackupError, match="unreadable"):
        db_module.restore_database(truncated, db_url, check_checksum=False)

    assert _row_counts(db) == before


def test_restore_database_rolls_back_everything_when_a_statement_fails(db, db_url, tmp_path: Path):
    with db.cursor() as cur:
        cur.execute("CREATE TABLE _restore_probe (id int PRIMARY KEY, note text)")
        cur.execute("INSERT INTO _restore_probe (id, note) VALUES (1, 'before restore')")
    dump_path = db_module.backup_database(db_url, tmp_path)
    with db.cursor() as cur:
        cur.execute("UPDATE _restore_probe SET note = 'mutated after backup' WHERE id = 1")
        # Not in the dump and depends on `notes`, so the restore's DROP TABLE notes fails.
        cur.execute("CREATE VIEW _restore_blocker AS SELECT slug FROM notes")

    with pytest.raises(db_module.BackupError, match="rolled back"):
        db_module.restore_database(dump_path, db_url)

    # Outside one transaction pg_restore carries on past the failed DROP and puts the probe
    # back to 'before restore' while reporting failure.
    with db.cursor() as cur:
        cur.execute("SELECT note FROM _restore_probe WHERE id = 1")
        assert cur.fetchone() == ("mutated after backup",)


# --------------------------------------------------------------------
# checksum sidecar
# --------------------------------------------------------------------


def _refuse_any_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail the test if the restore reaches the target: checksum refusals come first."""

    def connected(database_url: str) -> int:
        raise AssertionError("restore_database connected to the target before refusing")

    monkeypatch.setattr(db_module, "_server_major_version", connected)


def test_backup_database_writes_a_sha256_sidecar_in_shasum_format(db_url, tmp_path: Path):
    dump_path = db_module.backup_database(db_url, tmp_path)
    sidecar = db_module.checksum_path(dump_path)
    assert sidecar == dump_path.with_name(dump_path.name + ".sha256")
    digest = hashlib.sha256(dump_path.read_bytes()).hexdigest()
    assert sidecar.read_text(encoding="utf-8") == f"{digest}  {dump_path.name}\n"
    assert not list((tmp_path / "backups").glob("*.tmp"))


def test_backup_database_prunes_each_sidecar_with_its_dump(db_url, tmp_path: Path):
    for _ in range(4):
        db_module.backup_database(db_url, tmp_path, retention=3)
    backups = tmp_path / "backups"
    dumps = sorted(p.name for p in backups.glob("*.dump"))
    sidecars = sorted(p.name for p in backups.glob("*.dump.sha256"))
    assert len(dumps) == 3
    assert sidecars == [f"{name}.sha256" for name in dumps]


def test_restore_database_refuses_a_dump_damaged_after_the_backup_before_connecting(
    db, db_url, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    dump_path = db_module.backup_database(db_url, tmp_path)
    data = bytearray(dump_path.read_bytes())
    middle = len(data) // 2
    data[middle : middle + 64] = bytes(b ^ 0xFF for b in data[middle : middle + 64])
    dump_path.write_bytes(bytes(data))
    before = _row_counts(db)
    _refuse_any_connection(monkeypatch)

    with pytest.raises(db_module.BackupError, match="does not match its checksum"):
        db_module.restore_database(dump_path, db_url)

    assert _row_counts(db) == before


def test_restore_database_refuses_a_dump_without_a_checksum_file_and_names_the_way_out(
    db_url, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    dump_path = db_module.backup_database(db_url, tmp_path)
    db_module.checksum_path(dump_path).unlink()
    _refuse_any_connection(monkeypatch)

    with pytest.raises(db_module.BackupError, match="has no checksum file") as excinfo:
        db_module.restore_database(dump_path, db_url)
    assert "--no-verify-checksum" in str(excinfo.value)


def test_restore_database_refuses_a_checksum_file_written_for_another_dump(
    db_url, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    first = db_module.backup_database(db_url, tmp_path)
    second = db_module.backup_database(db_url, tmp_path)
    db_module.checksum_path(second).write_text(
        db_module.checksum_path(first).read_text(encoding="utf-8"), encoding="utf-8"
    )
    _refuse_any_connection(monkeypatch)

    with pytest.raises(db_module.BackupError, match=f"is for {first.name}"):
        db_module.restore_database(second, db_url)


def test_restore_database_without_the_checksum_check_restores_and_logs_a_warning(
    db, db_url, tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    with db.cursor() as cur:
        cur.execute("CREATE TABLE _restore_probe (id int PRIMARY KEY, note text)")
        cur.execute("INSERT INTO _restore_probe (id, note) VALUES (1, 'before restore')")
    dump_path = db_module.backup_database(db_url, tmp_path)
    db_module.checksum_path(dump_path).unlink()
    with db.cursor() as cur:
        cur.execute("UPDATE _restore_probe SET note = 'mutated after backup' WHERE id = 1")

    with caplog.at_level("WARNING", logger="portolan.db"):
        db_module.restore_database(dump_path, db_url, check_checksum=False)

    assert any(
        r.levelname == "WARNING" and "WITHOUT checking it against a checksum" in r.getMessage()
        for r in caplog.records
    )
    conn = db_module.connect(db_url)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT note FROM _restore_probe WHERE id = 1")
            assert cur.fetchone() == ("before restore",)
    finally:
        conn.close()


# --------------------------------------------------------------------
# apply_migrations on an autocommit connection (the weekly CLIs open one)
# --------------------------------------------------------------------


def test_apply_migrations_on_autocommit_reaches_the_file_with_the_upgrade_setting(
    db, tmp_path: Path
):
    """`portolan.migrating_existing_database` is transaction-local, so it
    reaches the file only when the file runs in the same transaction."""
    assert db.autocommit
    (tmp_path / "9001_setting_probe.sql").write_text(
        "CREATE TABLE _setting_probe AS "
        "SELECT current_setting('portolan.migrating_existing_database', true) AS v;"
    )

    assert db_module.apply_migrations(db, tmp_path) == ["9001_setting_probe.sql"]

    with db.cursor() as cur:
        cur.execute("SELECT v FROM _setting_probe")
        assert cur.fetchone() == ("true",)


def test_apply_migrations_on_autocommit_rolls_back_a_file_whose_record_fails(db, tmp_path: Path):
    """A file and its `schema_migrations` row commit together: when the
    record cannot be written, the file's own changes are rolled back too,
    so a re-run does not trip over a half-applied, unrecorded migration."""
    assert db.autocommit
    (tmp_path / "9002_unrecordable.sql").write_text(
        """
        CREATE TABLE _atomic_probe (id int);
        CREATE FUNCTION _refuse_record() RETURNS trigger AS $$
        BEGIN RAISE EXCEPTION 'record refused'; END $$ LANGUAGE plpgsql;
        CREATE TRIGGER _refuse_record BEFORE INSERT ON schema_migrations
            FOR EACH ROW EXECUTE FUNCTION _refuse_record();
        """
    )

    with pytest.raises(psycopg.errors.RaiseException):
        db_module.apply_migrations(db, tmp_path)

    with db.cursor() as cur:
        cur.execute("SELECT to_regclass('_atomic_probe')")
        assert cur.fetchone() == (None,)
        cur.execute(
            "SELECT count(*) FROM schema_migrations WHERE filename = %s",
            ("9002_unrecordable.sql",),
        )
        assert cur.fetchone() == (0,)
