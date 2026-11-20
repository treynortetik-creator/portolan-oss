"""`python -m portolan.db.rehearse`: restore a backup into a second database on the same server
and show it matches, without writing to the database `DATABASE_URL` names.

    python -m portolan.db.rehearse              # fresh backup, restore beside, compare, drop
    python -m portolan.db.rehearse --dump FILE  # restore that backup beside, report, drop
    python -m portolan.db.rehearse --dump FILE --target NAME --keep   # keep it, for a cutover
    python -m portolan.db.rehearse --dump FILE --no-verify-checksum   # older than checksums
    python -m portolan.db.rehearse --list       # the backups on the volume
    python -m portolan.db.rehearse --backup-only    # a fresh backup, nothing restored

Run it in the web container (`railway ssh --service web -- python -m portolan.db.rehearse`), so
the volume, `DATABASE_URL` and the image's own `pg_dump`/`pg_restore` are the ones a real
restore would use. Runbook: `01-spec/restore-drill-2026-09-22.md`.

What it writes: a dump under `<STORE_ROOT>/restore-rehearsal/backups/` (its own retention, so
the deploy backups in `<STORE_ROOT>/backups/` are never pruned by it), and the target database,
dropped at the end unless `--keep`. The target is `<current name>_restore_rehearsal` by default;
it must differ from the current database, and only a leftover `*_restore_rehearsal` database is
ever dropped to make room.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from psycopg import sql

from portolan import db
from portolan.db.fingerprint import compare, equality_table, fingerprint

REHEARSAL_SUFFIX = "_restore_rehearsal"
REHEARSAL_RETENTION = 3


class RehearsalError(RuntimeError):
    """The rehearsal refused to start or could not finish; the current database is untouched."""


def database_name(url: str) -> str:
    return urlsplit(url).path.lstrip("/").split("?", 1)[0]


def with_database(url: str, name: str) -> str:
    """`url` naming database `name` instead. URL form only: `redact_dsn` masks the password
    in a URL, not in a key=value string, and error messages carry the target DSN."""
    parts = urlsplit(url)
    if parts.scheme not in ("postgres", "postgresql"):
        raise RehearsalError("DATABASE_URL must be a postgresql:// URL for a rehearsal")
    return urlunsplit((parts.scheme, parts.netloc, f"/{name}", parts.query, parts.fragment))


def _target_exists(database_url: str, target: str) -> bool:
    """True for a leftover rehearsal database (dropped by `_prepare_target`); raises for any
    other existing database, which a rehearsal never drops."""
    conn = db.connect(database_url, autocommit=True)
    try:
        exists = conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (target,)).fetchone()
    finally:
        conn.close()
    if exists and not target.endswith(REHEARSAL_SUFFIX):
        raise RehearsalError(
            f"database {target!r} already exists; choose another --target "
            "(only a leftover *_restore_rehearsal database is dropped automatically)"
        )
    return exists is not None


def _prepare_target(database_url: str, target: str) -> None:
    leftover = _target_exists(database_url, target)
    conn = db.connect(database_url, autocommit=True)
    try:
        if leftover:
            conn.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(target)))
        conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(target)))
    finally:
        conn.close()


def _drop_target(database_url: str, target: str) -> None:
    conn = db.connect(database_url, autocommit=True)
    try:
        conn.execute(
            sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(target))
        )
    finally:
        conn.close()


def rehearse(
    database_url: str,
    store_root: Path | str,
    *,
    dump: Path | str | None = None,
    target: str | None = None,
    keep: bool = False,
    check_checksum: bool = True,
) -> dict[str, Any]:
    """Restore `dump` (default: a fresh backup of `database_url`) into `target` on the same
    server. With a fresh backup, the restore is compared with the current database as it was
    just before and just after the dump; with a given dump, it is reported, not compared (the
    current database has moved on since)."""
    current = database_name(database_url)
    target = target or f"{current}{REHEARSAL_SUFFIX}"
    if target == current:
        raise RehearsalError("--target names the current database; a rehearsal never writes to it")
    target_url = with_database(database_url, target)
    _target_exists(database_url, target)
    timings: dict[str, float] = {}
    result: dict[str, Any] = {"current": current, "target": target, "fresh_backup": dump is None}

    def timed(name: str, fn, *args, **kwargs):
        started = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            timings[name] = round(time.perf_counter() - started, 3)

    live_before = live_after = None
    if dump is None:
        live_before = timed("fingerprint current (before backup)", fingerprint, database_url)
        dump = timed(
            "backup (portolan.db.backup_database)",
            db.backup_database,
            database_url,
            Path(store_root) / "restore-rehearsal",
            retention=REHEARSAL_RETENTION,
        )
        live_after = timed("fingerprint current (after backup)", fingerprint, database_url)
    dump = Path(dump)
    if not dump.is_file():
        raise RehearsalError(f"no such backup file: {dump}")
    if check_checksum:
        db.verify_checksum(dump)
    result["checksum_verified"] = check_checksum
    result["dump"] = str(dump)
    result["dump_bytes"] = dump.stat().st_size

    timed("create target database", _prepare_target, database_url, target)
    try:
        timed(
            "restore (portolan.db.restore_database)",
            db.restore_database,
            dump,
            target_url,
            check_checksum=check_checksum,
        )
        restored = timed("fingerprint target", fingerprint, target_url)
        conn = db.connect(target_url)
        try:
            result["pending_migrations_at_cutover"] = db.pending_migrations(conn)
        finally:
            conn.close()
    finally:
        if not keep:
            timed("drop target database", _drop_target, database_url, target)
    result["kept"] = keep
    result["restored_tables"] = len(restored["tables"])
    result["restored_rows"] = sum(t["rows"] for t in restored["tables"].values())
    result["restored_migrations"] = len(restored["migrations"])

    if live_before is not None:
        window = compare(live_before, live_after)
        result["written_during_backup"] = sorted(
            {name for row in window["rows"] for name in row["differs"]}
        )
        # The dump is a snapshot from inside the window: it matches the database as it was
        # just before or just after, and exactly, when nothing wrote in between.
        before = compare(live_before, restored)
        after = compare(live_after, restored)
        result["comparison"] = before if before["equal"] or not after["equal"] else after
        result["ok"] = result["comparison"]["equal"]
    else:
        result["comparison"] = None
        result["ok"] = True
    result["timings_seconds"] = timings
    return result


def report(result: dict[str, Any]) -> str:
    lines = [
        f"dump: {result['dump']} ({result['dump_bytes']} bytes)",
        "checksum: "
        + (
            "matches the .sha256 written at backup time"
            if result["checksum_verified"]
            else "NOT CHECKED (--no-verify-checksum)"
        ),
        f"restored into {result['target']}: {result['restored_tables']} tables, "
        f"{result['restored_rows']} rows, {result['restored_migrations']} migrations",
    ]
    if result["comparison"] is not None:
        lines += ["", equality_table(result["comparison"], result["current"], result["target"])]
        if result["written_during_backup"]:
            lines.append(
                "written while the backup ran: " + ", ".join(result["written_during_backup"])
            )
    pending = result["pending_migrations_at_cutover"]
    lines.append(
        "migrations this release would apply if the app started on it: "
        + (", ".join(pending) if pending else "none")
    )
    lines.append("")
    lines += [f"{name}: {seconds} s" for name, seconds in result["timings_seconds"].items()]
    kept = "kept" if result["kept"] else "dropped"
    verdict = "PASS" if result["ok"] else "FAIL"
    lines.append(f"{verdict} restore rehearsal; target {result['target']} {kept}")
    return "\n".join(lines)


def list_backups(store_root: Path | str) -> list[tuple[Path, int, bool]]:
    """Deploy backups, then rehearsal backups, oldest first (names are UTC timestamps)."""
    root = Path(store_root)
    found = []
    for folder in (root / "backups", root / "restore-rehearsal" / "backups"):
        if folder.is_dir():
            found += [
                (path, path.stat().st_size, db.checksum_path(path).is_file())
                for path in sorted(folder.glob("*.dump"))
            ]
    return found


def _settings_database_url(settings) -> str:
    # `Settings.database_url` is a SecretStr; a plain str is accepted too.
    value = settings.database_url
    return value.get_secret_value() if hasattr(value, "get_secret_value") else value


def main(argv: list[str] | None = None) -> int:
    from portolan.config import get_settings

    parser = argparse.ArgumentParser(prog="python -m portolan.db.rehearse")
    parser.add_argument("--dump", type=Path, default=None, help="restore this backup file")
    parser.add_argument("--target", default=None, help="database to restore into")
    parser.add_argument("--keep", action="store_true", help="keep the target database")
    parser.add_argument("--list", action="store_true", help="list the backups; restore nothing")
    parser.add_argument(
        "--no-verify-checksum",
        action="store_true",
        help="with --dump: restore a dump that has no .sha256 file (logged loudly)",
    )
    parser.add_argument(
        "--backup-only", action="store_true", help="take a fresh backup; restore nothing"
    )
    args = parser.parse_args(argv)
    if args.no_verify_checksum and args.dump is None:
        parser.error("--no-verify-checksum only applies to --dump (a fresh backup has a checksum)")
    if args.no_verify_checksum:
        print(
            f"WARNING --no-verify-checksum: {args.dump} is restored WITHOUT checking it against "
            "a checksum. Use this only for a dump taken before checksums existed.",
            file=sys.stderr,
        )

    settings = get_settings()
    if args.backup_only:
        try:
            path = db.backup_database(
                _settings_database_url(settings),
                Path(settings.store_root) / "restore-rehearsal",
                retention=REHEARSAL_RETENTION,
            )
        except db.BackupError as exc:
            print(f"FAIL backup: {exc}", file=sys.stderr)
            return 1
        print(f"{path}  {path.stat().st_size} bytes  checksum {db.checksum_path(path).name}")
        return 0
    if args.list:
        backups = list_backups(settings.store_root)
        for path, size, has_checksum in backups:
            checksum = "checksum file present" if has_checksum else "NO checksum file"
            print(f"{path}  {size} bytes  {checksum}")
        print(f"{len(backups)} backup(s)")
        return 0
    try:
        result = rehearse(
            _settings_database_url(settings),
            settings.store_root,
            dump=args.dump,
            target=args.target,
            keep=args.keep,
            check_checksum=not args.no_verify_checksum,
        )
    except (RehearsalError, db.BackupError) as exc:
        print(f"FAIL restore rehearsal: {exc}", file=sys.stderr)
        return 1
    print(report(result))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
