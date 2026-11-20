"""Database connection and migration runner for Portolan.

One Postgres database per customer (spec sec 8). Domain tables carry no
tenant column; the exceptions are shared-code tables (``usage_events``,
``ai_access``, ``ai_access_log``). See ``portolan/db/migrations/0001_init.sql``
for the schema, the two clocks on claims, and the no-hard-deletes rule.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import psycopg

from portolan.util.redact import redact_dsn

MIGRATIONS_DIR = Path(__file__).parent / "migrations"

logger = logging.getLogger(__name__)

# Session-level advisory lock id serializing migration runs across replicas.
# Must differ from every other advisory lock key (grep `ADVISORY_LOCK_KEY`) and
# never change: an old and a new release must contend for the same lock.
MIGRATION_ADVISORY_LOCK_KEY = 847_209_351

DEFAULT_BACKUP_RETENTION = 10


class BackupError(RuntimeError):
    """Raised when a pre-migration `pg_dump` (or a `pg_restore`) cannot
    be trusted to have worked -- the binary is missing from the image, or
    the process exited non-zero. Callers must treat this as fatal: never
    apply a migration when the backup that was supposed to precede it
    didn't happen."""


def connect(url: str | None = None, *, autocommit: bool = False) -> psycopg.Connection:
    """Open a connection to the Postgres database.

    Reads ``DATABASE_URL`` from the environment when ``url`` is not given,
    falling back to ``.env``-backed settings (``Settings.database_url``) so
    CLIs work without exporting the variable.

    ``autocommit`` is a parameter so a caller cannot forget it: a
    ``with conn.transaction():`` block is a real ``BEGIN``/``COMMIT`` only on
    a connection that is idle between statements.
    """
    dsn = url or os.environ.get("DATABASE_URL")
    if not dsn:
        from portolan.config import get_settings

        dsn = get_settings().database_url.get_secret_value()
    conn = psycopg.connect(dsn)
    if autocommit:
        conn.autocommit = True
    return conn


def _ensure_migrations_table(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                filename    text PRIMARY KEY,
                applied_at  timestamptz NOT NULL DEFAULT now()
            )
            """
        )
    conn.commit()


def applied_migrations(conn: psycopg.Connection) -> set[str]:
    """Return the set of migration filenames already recorded as applied."""
    with conn.cursor() as cur:
        cur.execute("SELECT filename FROM schema_migrations")
        return {row[0] for row in cur.fetchall()}


def apply_migrations(conn: psycopg.Connection, migrations_dir: Path | None = None) -> list[str]:
    """Apply every ``*.sql`` file in ``migrations_dir``, in filename order,
    that is not already recorded in ``schema_migrations``.

    Idempotent: a second call with nothing new on disk applies nothing and
    returns an empty list. Each file and its `schema_migrations` row run in
    one transaction, on autocommit connections too, so a partially-applied
    file is never left unrecorded (`tests/test_db_migrate_backup.py`).
    Commits whatever the caller had pending first, and leaves no transaction
    open.

    Before each file runs, the transaction-local setting
    `portolan.migrating_existing_database` is `'true'` when
    `schema_migrations` already had rows before this call (an upgrade) and
    `'false'` otherwise (a new database); `0031_ai_access.sql` reads it.
    """
    directory = migrations_dir or MIGRATIONS_DIR
    _ensure_migrations_table(conn)
    already = applied_migrations(conn)
    existing_database = "true" if already else "false"
    # End the read, so each `conn.transaction()` below is a real BEGIN/COMMIT
    # rather than a savepoint inside it.
    conn.commit()

    newly_applied: list[str] = []
    for path in sorted(directory.glob("*.sql")):
        if path.name in already:
            continue
        sql = path.read_text()
        with conn.transaction(), conn.cursor() as cur:
            cur.execute(
                "SELECT set_config('portolan.migrating_existing_database', %s, true)",
                (existing_database,),
            )
            cur.execute(sql)
            cur.execute(
                "INSERT INTO schema_migrations (filename) VALUES (%s)",
                (path.name,),
            )
        newly_applied.append(path.name)
    return newly_applied


def pending_migrations(conn: psycopg.Connection, migrations_dir: Path | None = None) -> list[str]:
    """The `*.sql` filenames in `migrations_dir` not yet recorded in
    `schema_migrations`, in the same filename order `apply_migrations`
    would apply them. Applies nothing; backs `python -m portolan.db migrate
    --dry-run`."""
    directory = migrations_dir or MIGRATIONS_DIR
    _ensure_migrations_table(conn)
    already = applied_migrations(conn)
    return [path.name for path in sorted(directory.glob("*.sql")) if path.name not in already]


_VERSION_RE = re.compile(r"\)\s*(\d+(?:\.\d+)*)")

# Searched (after `PG_BIN_DIR`, before PATH) for a `pg_dump`/`pg_restore` of the
# server's major version: several client majors can be installed side by side,
# and `pg_dump` refuses a server newer than itself.
_PG_BIN_DIR_TEMPLATES = (
    "/opt/homebrew/opt/postgresql@{major}/bin",
    "/usr/lib/postgresql/{major}/bin",
    "/usr/local/opt/postgresql@{major}/bin",
)


def _server_major_version(database_url: str) -> int:
    """The connected Postgres server's major version (`server_version_num //
    10000`, the Postgres 10+ numbering).

    Raises `BackupError` with the DSN redacted if the connection fails.
    """
    try:
        conn = connect(database_url)
    except psycopg.Error as exc:
        raise BackupError(
            f"could not connect to {redact_dsn(database_url)} to check its Postgres "
            "server version before choosing a matching pg_dump/pg_restore"
        ) from exc
    try:
        with conn.cursor() as cur:
            cur.execute("SHOW server_version_num")
            (version_num,) = cur.fetchone()
    finally:
        conn.close()
    return int(version_num) // 10000


def _pg_bin_search_dirs(server_major: int) -> list[Path]:
    """Directories to search, in priority order, for a `pg_dump`/
    `pg_restore` whose own major version matches `server_major`, before
    falling back to bare PATH: the `PG_BIN_DIR` setting/env var (if set),
    then the well-known Homebrew (both chip architectures) and PGDG apt
    locations for that exact major version."""
    from portolan.config import get_settings

    dirs: list[Path] = []
    configured = get_settings().pg_bin_dir
    if configured:
        dirs.append(Path(configured))
    dirs.extend(Path(template.format(major=server_major)) for template in _PG_BIN_DIR_TEMPLATES)
    return dirs


def _binary_version(path: str) -> str | None:
    """Run `<path> --version` and return the version token (e.g.
    `"17.11"` out of `"pg_dump (PostgreSQL) 17.11 (Homebrew)"`), or
    `None` if the binary can't be run or its output doesn't parse."""
    try:
        result = subprocess.run([path, "--version"], capture_output=True, text=True, check=False)
    except OSError:
        return None
    if result.returncode != 0:
        return None
    match = _VERSION_RE.search(result.stdout)
    return match.group(1) if match else None


def _resolve_pg_binary(
    name: str, server_major: int, *, search_dirs: list[Path] | None = None
) -> str:
    """Find an executable named `name` (`"pg_dump"` or `"pg_restore"`)
    whose own major version matches `server_major`: check `search_dirs`
    (default `_pg_bin_search_dirs(server_major)`) in order, then fall
    back to bare PATH via `shutil.which`.

    Never returns a binary whose version doesn't match: raises `BackupError`
    naming the server's version, what was found on PATH, and `PG_BIN_DIR`.
    """
    if search_dirs is None:
        search_dirs = _pg_bin_search_dirs(server_major)

    for directory in search_dirs:
        candidate = directory / name
        if not candidate.is_file():
            continue
        version = _binary_version(str(candidate))
        if version is not None and int(version.split(".")[0]) == server_major:
            return str(candidate)

    on_path = shutil.which(name)
    path_version = _binary_version(on_path) if on_path else None
    if path_version is not None and int(path_version.split(".")[0]) == server_major:
        return on_path

    if path_version is not None:
        found = f"PATH has {name} {path_version}"
    elif on_path:
        found = f"{name} is on PATH but its version could not be determined"
    else:
        found = f"no {name} on PATH"

    raise BackupError(
        f"no {name} matching Postgres server major version {server_major} was found "
        f"in PG_BIN_DIR, the Homebrew/PGDG install locations, or on PATH ({found}). "
        f"Install postgresql-client-{server_major} (see docker/Dockerfile), or set "
        f"the PG_BIN_DIR setting to a directory containing a matching {name}."
    )


def backup_database(
    database_url: str, store_root: Path | str, *, retention: int = DEFAULT_BACKUP_RETENTION
) -> Path:
    """`pg_dump --format=custom` the whole database at `database_url` to
    `<store_root>/backups/<UTC timestamp>.dump` with a `.sha256` sidecar,
    prune to the newest `retention` dumps, and return the new dump's path.

    Raises `BackupError`, with the DSN redacted, when no `pg_dump` matching
    the server's major version is found or the process exits non-zero.
    Writes to a `.tmp` file and renames it, so a failed dump never leaves a
    truncated file at the real name.
    """
    backups_dir = Path(store_root) / "backups"
    backups_dir.mkdir(parents=True, exist_ok=True)
    # Microseconds, so two calls in one second never overwrite each other;
    # still sorts lexicographically, which `_prune_backups` relies on.
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%f")
    dump_path = backups_dir / f"{timestamp}.dump"
    tmp_path = backups_dir / f"{timestamp}.dump.tmp"

    server_major = _server_major_version(database_url)
    pg_dump = _resolve_pg_binary("pg_dump", server_major)

    try:
        result = subprocess.run(
            [pg_dump, "--format=custom", "--file", str(tmp_path), database_url],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise BackupError(
            f"{pg_dump} could not be run -- cannot take the pre-migration backup this "
            "startup path requires. Aborting rather than apply a migration with nothing "
            "to roll back to (see docker/Dockerfile)."
        ) from exc

    if result.returncode != 0:
        tmp_path.unlink(missing_ok=True)
        stderr = (result.stderr or "").strip()
        raise BackupError(
            f"pg_dump exited {result.returncode} backing up {redact_dsn(database_url)}"
            + (f": {stderr}" if stderr else "")
        )

    digest = _sha256_file(tmp_path)
    os.replace(tmp_path, dump_path)
    write_checksum(dump_path, digest)
    _prune_backups(backups_dir, retention)
    return dump_path


def checksum_path(dump_path: Path | str) -> Path:
    """The sidecar `backup_database` writes next to every dump: `<dump>.sha256`."""
    path = Path(dump_path)
    return path.with_name(path.name + ".sha256")


def _sha256_file(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write_checksum(dump_path: Path, digest: str) -> Path:
    """`<digest>  <dump name>`, the `shasum -a 256` format, so `shasum -c` checks it too.
    Written to a `.tmp` and renamed, so a half-written sidecar never exists."""
    sidecar = checksum_path(dump_path)
    tmp = sidecar.with_name(sidecar.name + ".tmp")
    tmp.write_text(f"{digest}  {dump_path.name}\n", encoding="utf-8")
    os.replace(tmp, sidecar)
    return sidecar


def verify_checksum(dump_path: Path | str) -> None:
    """Refuse a dump whose sidecar is missing, unreadable, names another file, or does not
    match the file's SHA-256. Every corruption or truncation after the backup was written
    fails here, before anything connects to the restore target."""
    path = Path(dump_path)
    sidecar = checksum_path(path)
    if not sidecar.is_file():
        raise BackupError(
            f"backup file {path} has no checksum file ({sidecar.name}); nothing was restored. "
            "Dumps taken before checksums existed can be restored with --no-verify-checksum."
        )
    parts = sidecar.read_text(encoding="utf-8").split()
    if len(parts) != 2 or not re.fullmatch(r"[0-9a-f]{64}", parts[0]):
        raise BackupError(f"checksum file {sidecar} is malformed; nothing was restored")
    expected, named = parts
    if named.lstrip("*") != path.name:
        raise BackupError(
            f"checksum file {sidecar} is for {named}, not {path.name}; nothing was restored"
        )
    if _sha256_file(path) != expected:
        raise BackupError(
            f"backup file {path} does not match its checksum (corrupted or truncated after "
            "the backup was taken); nothing was restored"
        )


def _prune_backups(backups_dir: Path, retention: int) -> None:
    """Delete all but the newest `retention` `*.dump` files in
    `backups_dir`, with their checksum sidecars. Filenames sort
    chronologically (`%Y%m%dT%H%M%S.%f`), so a plain name sort is enough."""
    if retention <= 0:
        return
    dumps = sorted(backups_dir.glob("*.dump"))
    for stale in dumps[:-retention]:
        stale.unlink(missing_ok=True)
        checksum_path(stale).unlink(missing_ok=True)


def verify_dump(dump_path: Path | str, pg_restore: str) -> None:
    """Read every block of a `--format=custom` dump without a database
    (`pg_restore --file=<devnull>`), so a truncated or corrupted file
    raises `BackupError` before anything touches the restore target."""
    path = Path(dump_path)
    try:
        result = subprocess.run(
            [pg_restore, "--file", os.devnull, str(path)],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise BackupError(f"{pg_restore} could not be run") from exc
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise BackupError(
            f"backup file {path} is unreadable (truncated or corrupted); nothing was restored"
            + (f": {stderr}" if stderr else "")
        )


def restore_database(
    dump_path: Path | str, database_url: str, *, check_checksum: bool = True
) -> None:
    """`pg_restore` a `--format=custom` dump (as `backup_database` writes)
    onto `database_url`, replacing any existing objects (`--clean
    --if-exists`) rather than erroring on ones the target database
    already has. Backs `python -m portolan.db restore <file>` and the
    restore rehearsal (`portolan/db/rehearse.py`).

    All or nothing, in three layers, each before the next: the dump must
    match the checksum written at backup time (`verify_checksum`; skipped
    only with `check_checksum=False`, for dumps older than checksums), the
    whole file must read (`verify_dump`), and the restore runs in one
    transaction, so a bad file or a failed statement leaves the target as
    it was. Pinned by the checksum, truncated-dump and failed-statement
    tests in `tests/test_db_migrate_backup.py`.
    """
    path = Path(dump_path)
    if not path.is_file():
        raise BackupError(f"no such backup file: {path}")
    if check_checksum:
        verify_checksum(path)
    else:
        logger.warning(
            "restoring %s WITHOUT checking it against a checksum (--no-verify-checksum): "
            "only the read-through and the single transaction protect the target",
            path,
        )

    server_major = _server_major_version(database_url)
    pg_restore = _resolve_pg_binary("pg_restore", server_major)
    verify_dump(path, pg_restore)

    try:
        result = subprocess.run(
            [
                pg_restore,
                "--clean",
                "--if-exists",
                "--no-owner",
                "--single-transaction",
                "--dbname",
                database_url,
                str(path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise BackupError(f"{pg_restore} could not be run") from exc

    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise BackupError(
            f"pg_restore exited {result.returncode} restoring onto {redact_dsn(database_url)}"
            " (single transaction: rolled back, target unchanged)"
            + (f": {stderr}" if stderr else "")
        )


def migrate_with_backup(
    conn: psycopg.Connection,
    *,
    database_url: str,
    store_root: Path | str,
    migrations_dir: Path | None = None,
) -> list[str]:
    """The production migration path: the web app's startup `lifespan` and
    `python -m portolan.db migrate` (the entrypoint's `migrate` role):

    1. Take a session-level Postgres advisory lock
       (`MIGRATION_ADVISORY_LOCK_KEY`) so two replicas booting at once
       serialize instead of both reading `schema_migrations`, both seeing
       the same file as unapplied, and both running it.
    2. Only when there is at least one pending migration, `pg_dump` a
       backup to `<store_root>/backups/` *before* applying anything. A
       backup failure raises `BackupError` and aborts -- nothing is
       applied.
    3. Apply pending migrations (`apply_migrations`).

    Idempotent end to end: a second call with nothing pending takes no
    backup, applies nothing, and returns `[]`. `conn` must not be mid
    transaction when this is called (fresh from `db.connect()` is fine);
    the connection is left committed, with no open transaction, either
    way.
    """
    with conn.cursor() as cur:
        cur.execute("SELECT pg_advisory_lock(%s)", (MIGRATION_ADVISORY_LOCK_KEY,))
    conn.commit()
    try:
        pending = pending_migrations(conn, migrations_dir)
        if pending:
            logger.info(
                "portolan.db: %d pending migration(s) (%s) -- taking a backup before applying",
                len(pending),
                ", ".join(pending),
            )
            backup_path = backup_database(database_url, store_root)
            logger.info("portolan.db: backup written to %s", backup_path)
        else:
            logger.info("portolan.db: no pending migrations")

        applied = apply_migrations(conn, migrations_dir)
        if applied:
            logger.info("portolan.db: applied %d migration(s): %s", len(applied), applied)
        return applied
    finally:
        with conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_unlock(%s)", (MIGRATION_ADVISORY_LOCK_KEY,))
        conn.commit()
