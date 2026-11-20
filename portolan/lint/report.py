"""`run_lint`, `render_markdown`, and `write_changelog_row` (spec sec 6.1).

`run_lint` runs all eleven checks and returns one `LintReport`. Nothing here
prints or exits; that is the CLI's job.
"""

from __future__ import annotations

from datetime import date

import psycopg
from psycopg.types.json import Jsonb

from portolan.lint import checks
from portolan.lint._store import Store
from portolan.lint.models import CHECK_NAMES, LintReport
from portolan.util import clock


def run_lint(conn: psycopg.Connection, store: Store, *, today: date | None = None) -> LintReport:
    """Run all eleven lint checks and return one report.

    `today` defaults to `clock.today()`, the deployment's zone
    (`PORTOLAN_TIMEZONE`) rather than the host's, so `stale` does not flip a
    day depending on where the job runs; pin it for a reproducible run.
    `checks.py` never reads the clock itself: every check takes `today` from here.
    """
    resolved_today = today if today is not None else clock.today()
    report = LintReport(today=resolved_today)

    report.extend("broken_link", checks.check_broken_link(conn))
    report.extend("orphan", checks.check_orphan(conn))
    report.extend("thin_links", checks.check_thin_links(conn))
    report.extend("stale", checks.check_stale(conn, today=resolved_today))
    report.extend("bad_header", checks.check_bad_header(conn))
    report.extend("relative_time", checks.check_relative_time(conn))
    report.extend("unresolved_citation", checks.check_unresolved_citation(conn))
    report.extend("missing_original", checks.check_missing_original(conn, store))
    report.extend("ambiguity", checks.check_ambiguity(conn))
    report.extend("human_edit_contradiction", checks.check_human_edit_contradiction(conn))
    report.extend("growth_guard", checks.check_growth_guard(conn))

    return report


def _identifier(finding) -> str:  # noqa: ANN001 - LintFinding
    if finding.note_slug and finding.claim_id is not None:
        return f"{finding.note_slug} (claim #{finding.claim_id})"
    if finding.claim_id is not None:
        return f"claim #{finding.claim_id}"
    if finding.note_slug:
        return finding.note_slug
    if finding.source_slug:
        return finding.source_slug
    return "-"


def render_markdown(report: LintReport) -> str:
    """Dated header, a counts table, then findings grouped by check."""
    lines: list[str] = []

    lines.append(f"# Lint report — {report.today:%Y-%m-%d}")
    lines.append("")

    total_errors = len(report.errors())
    total_warnings = len(report.warnings())
    lines.append(f"{total_errors} error(s), {total_warnings} warning(s).")
    lines.append("")

    lines.append("| Check | Errors | Warnings |")
    lines.append("|---|---:|---:|")
    counts = report.counts()
    for check in CHECK_NAMES:
        errs, warns = counts[check]
        lines.append(f"| {check} | {errs} | {warns} |")
    lines.append(f"| **Total** | **{total_errors}** | **{total_warnings}** |")
    lines.append("")

    for check in CHECK_NAMES:
        rows = report.for_check(check)
        lines.append(f"## {check}")
        lines.append("")
        if not rows:
            lines.append("Clean.")
            lines.append("")
            continue
        for finding in rows:
            lines.append(f"- **{finding.severity}** — {_identifier(finding)}: {finding.message}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_changelog_row(
    conn: psycopg.Connection,
    *,
    kind: str = "lint",
    summary: str,
    details: dict | None = None,
) -> None:
    """One append-only row in `changelog` (spec sec 6.1, sec 7) recording
    that a lint run happened and what it found.

    Commits its own work (build-contracts.md "Ground rules": "each stage
    function commits its own work").
    """
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO changelog (kind, summary, details) VALUES (%s, %s, %s)",
            (kind, summary, Jsonb(details or {})),
        )
    conn.commit()
