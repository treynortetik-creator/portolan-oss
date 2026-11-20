"""The Portolan lint (portolan-spec-v2.md sec 6.1, build-contracts.md).

    from portolan.lint import run_lint, render_markdown, write_changelog_row

`run_lint(conn, store, *, today=None) -> LintReport` runs all eleven checks
(broken_link, orphan, thin_links, stale, bad_header, relative_time,
unresolved_citation, missing_original, ambiguity, human_edit_contradiction,
growth_guard) against an already-migrated Postgres connection and a `Store`
for checking stored originals. `render_markdown` turns the report into the
dated markdown file the CLI writes; `write_changelog_row` records that a
run happened.

    python -m portolan.lint --out var/lint/

writes `var/lint/lint-YYYY-MM-DD.md` and exits 1 if any error-severity
finding exists.
"""

from __future__ import annotations

from portolan.lint.models import CHECK_NAMES, LintFinding, LintReport, Severity
from portolan.lint.report import render_markdown, run_lint, write_changelog_row

__all__ = [
    "CHECK_NAMES",
    "LintFinding",
    "LintReport",
    "Severity",
    "render_markdown",
    "run_lint",
    "write_changelog_row",
]
