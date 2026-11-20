"""Post-deploy smoke check: control query passes, note/source counts held.

Spec: `01-spec/portolan-spec-v2.md` §11 — "Post-deploy smoke checks: control
query passes, note and source counts did not fall, coverage ledger
consistent. Automatic rollback on a failed check." (The coverage-ledger
consistency check belongs to whichever wave owns `coverage_ledger`'s
invariants — build-contracts.md files it under the ingest filter, wave A2 —
and is not duplicated here; this script owns the control query and the two
counts.)

Two-step usage, one call on each side of a deploy (`--database-url` and
`--dry-run` are top-level flags and come before the `baseline`/`check`
subcommand):

    uv run python infra/deploy/smoke.py --database-url "$DATABASE_URL" \
        baseline --out baseline.json
    # ... deploy happens here (canary.py, or CI directly) ...
    uv run python infra/deploy/smoke.py --database-url "$DATABASE_URL" \
        check --baseline baseline.json

`check` exits non-zero — the signal `canary.py` treats as "roll back" — if
the control query comes back broken, or if either count dropped.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Any term that will always resolve to a genuine hit once a single note
# exists — used only to distinguish "the index is broken" from "there is
# nothing recorded yet" (portolan.search.query.SearchResult.control_ok
# already does exactly this distinction for a query with zero hits; this
# is the query text fed to it).
CONTROL_QUERY = "portolan"


@dataclass
class Counts:
    control_ok: bool
    note_count: int
    source_count: int

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Counts:
        return cls(**data)


def measure(conn: Any, embedder: Any, query: str = CONTROL_QUERY) -> Counts:
    """Run the control query and count notes/sources.

    `conn` is a `psycopg.Connection`; `embedder` any
    `portolan.search.embed.Embedder`. Imports `portolan.search.query.search`
    lazily so this module is importable (for its pure `compare`/`Counts`
    logic) without a live database or the search module's dependencies.
    """
    from portolan.search.query import search

    result = search(conn, embedder, query, k=1)
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM notes")
        note_count = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM sources")
        source_count = cur.fetchone()[0]
    return Counts(control_ok=result.control_ok, note_count=note_count, source_count=source_count)


def compare(before: Counts, after: Counts) -> list[str]:
    """Return a list of problems found comparing `before` to `after`; an
    empty list means the smoke check passed."""
    problems = []
    if not after.control_ok:
        problems.append(
            "control query failed after deploy (the search index looks broken, "
            "not just genuinely empty)"
        )
    if after.note_count < before.note_count:
        problems.append(f"note count dropped: {before.note_count} -> {after.note_count}")
    if after.source_count < before.source_count:
        problems.append(f"source count dropped: {before.source_count} -> {after.source_count}")
    return problems


def _connect_and_embed(database_url: str) -> tuple[Any, Any]:
    from portolan import db as db_module
    from portolan.config import settings
    from portolan.search.embed import get_embedder

    conn = db_module.connect(database_url)
    embedder = get_embedder(settings)
    return conn, embedder


def _redact_url(url: str | None) -> str:
    """`url`, with everything but the scheme replaced -- review finding
    F14: this module used to print `--database-url <url>` (the Aurora
    password lives in it) straight into a dry-run's stdout, which is CI
    log output."""
    if not url:
        return repr(url)
    scheme = url.split("://", 1)[0] if "://" in url else "postgresql"
    return f"{scheme}://***REDACTED***"


def cmd_baseline(args: argparse.Namespace) -> int:
    if args.dry_run:
        print(
            f"[dry-run] would connect to {_redact_url(args.database_url)}, run the control "
            f"query, count notes/sources, and write the result to {args.out}"
        )
        # Review finding F14: this used to write a real placeholder
        # baseline ({control_ok: True, note_count: 0, source_count: 0})
        # to args.out even in --dry-run -- and a later, real (non-dry)
        # `check` reading that file back would compare against a
        # trivially-satisfied {0, 0} baseline, silently passing a smoke
        # check that should have failed. --dry-run now writes nothing at
        # all: `check` either reads a real baseline from a real
        # `baseline` run, or fails loudly on a missing file, which is
        # what it should do.
        print(f"[dry-run] would NOT actually write {args.out} in --dry-run")
        return 0

    conn, embedder = _connect_and_embed(args.database_url)
    try:
        counts = measure(conn, embedder)
    finally:
        conn.close()
    Path(args.out).write_text(json.dumps(counts.to_json(), indent=2))
    print(f"baseline written to {args.out}: {counts.to_json()}")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    if args.dry_run:
        print(
            f"[dry-run] would connect to {_redact_url(args.database_url)}, re-run the "
            f"control query and counts, and compare to {args.baseline}"
        )
        return 0

    before = Counts.from_json(json.loads(Path(args.baseline).read_text()))

    conn, embedder = _connect_and_embed(args.database_url)
    try:
        after = measure(conn, embedder)
    finally:
        conn.close()

    problems = compare(before, after)
    if problems:
        print("SMOKE CHECK FAILED:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    print(f"smoke check passed: {after.to_json()}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--database-url", default=None, help="Defaults to $DATABASE_URL")
    parser.add_argument("--dry-run", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    baseline_parser = sub.add_parser("baseline", help="Capture pre-deploy counts")
    baseline_parser.add_argument("--out", required=True)
    baseline_parser.set_defaults(func=cmd_baseline)

    check_parser = sub.add_parser("check", help="Compare post-deploy counts to a baseline")
    check_parser.add_argument("--baseline", required=True)
    check_parser.set_defaults(func=cmd_check)

    args = parser.parse_args(argv)
    if args.database_url is None:
        args.database_url = os.environ.get("DATABASE_URL")
    if not args.dry_run and not args.database_url:
        parser.error("--database-url or $DATABASE_URL is required outside --dry-run")

    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
