"""Set `registry_entities.owner_entity_id` on a live graph from an owner
assignment CSV (`01-spec/actions-and-owners-2026-09-19.md`, Phase 1,
build item 2: "set it on the live registry with a script").

The CSV is `corpus/out/crm/accounts.csv` after `python -m portolan.registry
assign-owners --write-csv` (or any CSV shaped the same way: a `crm_id`
column and an `owner_crm_id` column) -- one row per account. For every row
with a non-blank `owner_crm_id`, this script sets that account's
`registry_entities.owner_entity_id` to the owner's own registry row id.

**Idempotent.** An account already carrying the correct owner is a no-op
(counted separately, never re-written); re-running this script twice in a
row changes nothing the second time.

**Prints counts only.** Never a crm_id, a name, or the connection string
-- `01-spec/CLAUDE.md`'s "never print deployment variables" rule. Read
`--database-url`/`$DATABASE_URL` but never echo it.

**Vendor-side only.** A row whose `owner_crm_id` doesn't resolve to a
registry entity of kind `internal_person` is skipped and counted, never
written -- the same rule `portolan.registry.owners.is_vendor_side`
enforces everywhere else an owner gets set.

Usage:
    uv run python scripts/set_owners.py \\
        --accounts-csv corpus/out/crm/accounts.csv \\
        --database-url "$DATABASE_URL"

This script is never run against Railway directly from an agent session
(`01-spec/CLAUDE.md`: no Railway commands at all) -- the orchestrator runs
it there via `railway run` with the connection string assembled in-shell.
Prove it first on a scratch copy of the local database:

    createdb portolan_owners_scratch
    pg_dump portolan_reseed_20260919 | psql portolan_owners_scratch
    uv run python scripts/set_owners.py \\
        --accounts-csv corpus/out/crm/accounts.csv \\
        --database-url postgresql://localhost/portolan_owners_scratch
    dropdb portolan_owners_scratch
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import psycopg

__all__ = ["set_owners"]


def set_owners(conn: psycopg.Connection, accounts_csv: Path) -> dict[str, int]:
    """Idempotently set `owner_entity_id` for every row in `accounts_csv`
    with a non-blank `owner_crm_id`. Returns counts only -- never a crm_id
    or a name, so a caller can safely print the result."""
    with accounts_csv.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    counts = {
        "total_rows": len(rows),
        "updated": 0,
        "already_correct": 0,
        "skipped_no_owner_field": 0,
        "skipped_account_not_found": 0,
        "skipped_owner_not_vendor_side": 0,
    }

    with conn.cursor() as cur:
        for row in rows:
            account_crm_id = (row.get("crm_id") or "").strip()
            owner_crm_id = (row.get("owner_crm_id") or "").strip()
            if not owner_crm_id:
                counts["skipped_no_owner_field"] += 1
                continue

            cur.execute(
                "SELECT id, owner_entity_id FROM registry_entities "
                "WHERE crm_id = %s AND kind = 'account'",
                (account_crm_id,),
            )
            account_row = cur.fetchone()
            if account_row is None:
                counts["skipped_account_not_found"] += 1
                continue
            account_id, current_owner_id = account_row

            cur.execute("SELECT id, kind FROM registry_entities WHERE crm_id = %s", (owner_crm_id,))
            owner_row = cur.fetchone()
            if owner_row is None or owner_row[1] != "internal_person":
                counts["skipped_owner_not_vendor_side"] += 1
                continue
            owner_id = owner_row[0]

            if current_owner_id == owner_id:
                counts["already_correct"] += 1
                continue

            cur.execute(
                "UPDATE registry_entities SET owner_entity_id = %s WHERE id = %s",
                (owner_id, account_id),
            )
            counts["updated"] += 1

    conn.commit()
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Set registry_entities.owner_entity_id from an owner assignment CSV."
    )
    parser.add_argument(
        "--accounts-csv",
        type=Path,
        default=Path("corpus/out/crm/accounts.csv"),
        help="CSV with crm_id + owner_crm_id columns "
        "(python -m portolan.registry assign-owners --write-csv).",
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help="Postgres connection string. Defaults to $DATABASE_URL. Never printed.",
    )
    args = parser.parse_args()

    database_url = args.database_url or os.environ.get("DATABASE_URL")
    if not database_url:
        raise SystemExit("no database url given: pass --database-url or set DATABASE_URL")

    conn = psycopg.connect(database_url)
    try:
        counts = set_owners(conn, args.accounts_csv)
    finally:
        conn.close()

    for name, value in counts.items():
        print(f"{name}={value}")


if __name__ == "__main__":
    main()
