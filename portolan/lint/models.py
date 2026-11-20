"""Types shared by every lint check (portolan-spec-v2.md sec 6.1).

One ``LintFinding`` per problem instance; ``LintReport`` keeps one list per
check name. Severity is two tiers, not a score: errors fail the run, warnings
never do, because a single-tier report of hundreds of issues stops being read
(the "TWO TIERS" comment in `02-reference-implementation/wiki-tooling/wiki-lint.py`).
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

Severity = Literal["error", "warning"]

# The order checks run in (`run_lint`) and render in the markdown report.
CHECK_NAMES: tuple[str, ...] = (
    "broken_link",
    "orphan",
    "thin_links",
    "stale",
    "bad_header",
    "relative_time",
    "unresolved_citation",
    "missing_original",
    "ambiguity",
    "human_edit_contradiction",
    "growth_guard",
)


class LintFinding(BaseModel):
    """One problem instance: which check found it, how bad it is, whichever
    identifiers apply (a finding may set none, one, or two of note_slug /
    claim_id / source_slug), and a one-line message."""

    check: str
    severity: Severity
    note_slug: str | None = None
    claim_id: int | None = None
    source_slug: str | None = None
    message: str


class LintReport(BaseModel):
    """One finding list per check, plus the `today` the run was pinned to so a
    re-render is reproducible."""

    today: date
    findings: dict[str, list[LintFinding]] = Field(default_factory=dict)

    def add(self, finding: LintFinding) -> None:
        self.findings.setdefault(finding.check, []).append(finding)

    def extend(self, check: str, findings: list[LintFinding]) -> None:
        self.findings.setdefault(check, []).extend(findings)

    def for_check(self, check: str) -> list[LintFinding]:
        return self.findings.get(check, [])

    def all_findings(self) -> list[LintFinding]:
        """Every finding, in the canonical check order."""
        out: list[LintFinding] = []
        for check in CHECK_NAMES:
            out.extend(self.findings.get(check, []))
        return out

    def errors(self) -> list[LintFinding]:
        return [f for f in self.all_findings() if f.severity == "error"]

    def warnings(self) -> list[LintFinding]:
        return [f for f in self.all_findings() if f.severity == "warning"]

    @property
    def has_errors(self) -> bool:
        return any(f.severity == "error" for f in self.all_findings())

    def counts(self) -> dict[str, tuple[int, int]]:
        """check -> (error_count, warning_count), for every known check
        name, including checks with zero findings (so the counts table in
        the rendered report always shows the full check list)."""
        out: dict[str, tuple[int, int]] = {}
        for check in CHECK_NAMES:
            rows = self.findings.get(check, [])
            errs = sum(1 for r in rows if r.severity == "error")
            warns = sum(1 for r in rows if r.severity == "warning")
            out[check] = (errs, warns)
        return out
