"""`python -m portolan.activity ...`: retention and per-person deletion for
the pilot activity log.

    python -m portolan.activity purge [--days N]
    python -m portolan.activity delete-user "Display Name"

`purge` defaults to `PORTOLAN_ACTIVITY_RETENTION_DAYS`. Both print row
counts only.
"""

from __future__ import annotations

import argparse
import json
import sys

from portolan import db
from portolan.activity import delete_user, purge_expired
from portolan.config import get_settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m portolan.activity")
    sub = parser.add_subparsers(dest="command", required=True)
    purge = sub.add_parser("purge", help="delete activity older than the retention window")
    purge.add_argument("--days", type=int, default=None)
    remove = sub.add_parser("delete-user", help="delete every pilot row for one person")
    remove.add_argument("user_name")
    args = parser.parse_args(argv)

    conn = db.connect(autocommit=True)
    try:
        if args.command == "purge":
            days = args.days if args.days is not None else get_settings().activity_retention_days
            result = {"retention_days": days, **purge_expired(conn, retention_days=days)}
        else:
            name = args.user_name.strip()
            if not name:
                parser.error("user_name must not be empty")
            result = {"user_name": name, **delete_user(conn, name)}
    finally:
        conn.close()
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
