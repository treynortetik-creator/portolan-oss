"""CLI: `python -m portolan.lint --out var/lint/`.

Connects to `DATABASE_URL`, runs every check, writes a dated markdown
report under `--out`, appends one `changelog` row, and exits 1 if the
report contains any error-severity finding (0 otherwise) — so this can sit
directly in a CI step or a scheduled job per spec sec 6.1/6.2.

The store is `portolan.store.LocalStore` at `Settings.store_root` (`STORE_ROOT`),
the same root the rest of the app uses, so `missing_original` checks the real store.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from portolan import db as db_module
from portolan.config import get_settings
from portolan.lint.report import render_markdown, run_lint, write_changelog_row
from portolan.store import LocalStore
from portolan.util import clock


def _default_store() -> LocalStore:
    return LocalStore(get_settings().store_root)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m portolan.lint")
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Directory to write lint-YYYY-MM-DD.md into.",
    )
    args = parser.parse_args(argv)

    conn = db_module.connect()
    store = _default_store()
    try:
        # The deployment's zone (`PORTOLAN_TIMEZONE`), not the host's, so `stale`
        # does not flip a day depending on where the job runs.
        today = clock.today()
        report = run_lint(conn, store, today=today)
        markdown = render_markdown(report)

        args.out.mkdir(parents=True, exist_ok=True)
        out_path = args.out / f"lint-{today:%Y-%m-%d}.md"
        out_path.write_text(markdown)

        counts = report.counts()
        summary = f"{len(report.errors())} error(s), {len(report.warnings())} warning(s)"
        # write_changelog_row commits its own work.
        write_changelog_row(
            conn,
            kind="lint",
            summary=summary,
            details={check: {"errors": e, "warnings": w} for check, (e, w) in counts.items()},
        )

        print(f"wrote {out_path}")
        print(summary)

        return 1 if report.has_errors else 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
