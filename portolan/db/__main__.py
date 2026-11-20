"""`python -m portolan.db`: operational CLI for migrations and disaster
recovery.

    python -m portolan.db migrate              # check STORE_ROOT is on the
                                                # volume, back up (if anything
                                                # is pending), then apply
                                                # pending migrations
    python -m portolan.db migrate --dry-run    # list pending migrations;
                                                # touch nothing
    python -m portolan.db restore <dump-file>  # pg_restore a
                                                # --format=custom dump
                                                # (from backup_database)
                                                # onto DATABASE_URL, after
                                                # checking its .sha256
    python -m portolan.db restore <dump-file> --no-verify-checksum
                                                # a dump older than
                                                # checksums; logged loudly

`docker/entrypoint.sh`'s `migrate` role execs `python -m portolan.db
migrate`, so the container and a person running it by hand run the same code.

`DATABASE_URL` comes from `Settings.database_url`. Every line that names a DSN
goes through `portolan.util.redact.redact_dsn`.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from portolan import db
from portolan.config import get_settings
from portolan.store import StoreRootError, assert_store_root_ready
from portolan.util.redact import redact_dsn

logger = logging.getLogger("portolan.db.cli")


def _cmd_migrate(args: argparse.Namespace) -> int:
    settings = get_settings()
    if not args.dry_run:
        # The pre-migration backup lands under store_root: refuse before connecting when
        # the volume did not mount, rather than dump to the container's own disk.
        try:
            assert_store_root_ready(
                settings.store_root,
                railway_volume_mount_path=os.environ.get("RAILWAY_VOLUME_MOUNT_PATH"),
            )
        except StoreRootError as exc:
            logger.error("aborting migration: %s", exc)
            return 1
    database_url = settings.database_url.get_secret_value()
    conn = db.connect(database_url)
    try:
        if args.dry_run:
            pending = db.pending_migrations(conn)
            if pending:
                print(f"{len(pending)} pending migration(s):")
                for name in pending:
                    print(f"  {name}")
            else:
                print("no pending migrations")
            return 0

        applied = db.migrate_with_backup(
            conn, database_url=database_url, store_root=settings.store_root
        )
        if applied:
            print(f"applied {len(applied)} migration(s): {applied}")
        else:
            print("no pending migrations")
        return 0
    except db.BackupError as exc:
        logger.error("aborting migration: %s", exc)
        return 1
    finally:
        conn.close()


def _cmd_restore(args: argparse.Namespace) -> int:
    settings = get_settings()
    database_url = settings.database_url.get_secret_value()
    logger.warning(
        "restoring %s onto %s -- this REPLACES the database's current contents",
        args.dump_file,
        redact_dsn(database_url),
    )
    if args.no_verify_checksum:
        logger.warning(
            "--no-verify-checksum: %s is restored WITHOUT checking it against the checksum "
            "written at backup time. Use this only for a dump taken before checksums existed.",
            args.dump_file,
        )
    try:
        db.restore_database(
            args.dump_file, database_url, check_checksum=not args.no_verify_checksum
        )
    except db.BackupError as exc:
        logger.error("restore failed: %s", exc)
        return 1
    print(f"restored {args.dump_file} onto {redact_dsn(database_url)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m portolan.db")
    sub = parser.add_subparsers(dest="command", required=True)

    p_migrate = sub.add_parser("migrate", help="apply pending migrations, with a pre-flight backup")
    p_migrate.add_argument(
        "--dry-run", action="store_true", help="list pending migrations; apply nothing"
    )
    p_migrate.set_defaults(func=_cmd_migrate)

    p_restore = sub.add_parser("restore", help="pg_restore a backup dump onto DATABASE_URL")
    p_restore.add_argument(
        "dump_file", help="path to a --format=custom dump (from backup_database)"
    )
    p_restore.add_argument(
        "--no-verify-checksum",
        action="store_true",
        help="restore a dump that has no .sha256 file (taken before checksums existed)",
    )
    p_restore.set_defaults(func=_cmd_restore)

    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
