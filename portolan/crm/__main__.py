"""`python -m portolan.crm ingest <dir> [--embed none|fake|fastembed]` takes in
a `crm/*.csv` export directory as snapshots and writes the changes;
`python -m portolan.crm changes` writes any changes not yet written. Counts
only are printed.

`--embed none` (the default) leaves the touched notes for the next
`index_pending` (every pipeline run), so this command never writes vectors
from a different embedder than the graph's own."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from portolan import db
from portolan.config import get_settings
from portolan.connectors.file_connector import FileConnector
from portolan.crm.changes import refresh_crm_changes
from portolan.crm.ingest import ingest_crm_snapshots
from portolan.registry import Registry
from portolan.search.embed import Embedder
from portolan.store import LocalStore


def _embedder(kind: str) -> Embedder | None:
    if kind == "none":
        return None
    if kind == "fastembed":
        from portolan.search.embed import FastEmbedEmbedder

        return FastEmbedEmbedder()
    from portolan.search.embed import FakeEmbedder

    return FakeEmbedder()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m portolan.crm")
    sub = parser.add_subparsers(dest="command", required=True)
    ingest = sub.add_parser("ingest")
    ingest.add_argument("root", type=Path, help="a directory holding crm/*.csv exports")
    ingest.add_argument("--embed", choices=["none", "fake", "fastembed"], default="none")
    changes = sub.add_parser("changes")
    changes.add_argument("--embed", choices=["none", "fake", "fastembed"], default="none")
    args = parser.parse_args(argv)

    settings = get_settings()
    conn = db.connect(autocommit=True)
    try:
        db.apply_migrations(conn)
        embedder = _embedder(args.embed)
        if args.command == "ingest":
            report = ingest_crm_snapshots(
                conn, FileConnector(args.root), LocalStore(settings.store_root), embedder=embedder
            )
            print(
                f"rows {report.rows_seen}, snapshots {report.snapshots_recorded}, "
                f"not recorded {report.rows_not_recorded}"
            )
            refresh = report.changes
        else:
            refresh = refresh_crm_changes(conn, Registry.load(conn), embedder=embedder)
        print(
            f"changes written {refresh.changes_written}, already written "
            f"{refresh.already_written}, unknown accounts {len(refresh.unknown_accounts)}"
        )
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
