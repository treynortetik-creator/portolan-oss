"""The production audit (`01-spec/quality-loop.md` "The production
audit", `01-spec/build-contracts.md` "Report, audit, consolidation", wave
C2). Public surface: `sample_sources`, `second_pass`, `diff_facts`,
`judge`, `run_audit`, `render_markdown`.

CLI: `python -m portolan.audit --since 2026-08-01 --rate 0.05 [--llm fake|agent_sdk]`.
"""

from __future__ import annotations

from portolan.audit.diff import CLAIM_SIMILARITY_THRESHOLD, diff_facts
from portolan.audit.judge import JudgeVerdict, judge
from portolan.audit.models import (
    AuditReport,
    DiffCandidate,
    FactDiff,
    Judgement,
    TypeRate,
    UsageTotals,
    WorstSource,
)
from portolan.audit.report import render_markdown, run_audit
from portolan.audit.sample import sample_sources
from portolan.audit.second_pass import AuditBackend, SecondPassResults, second_pass

__all__ = [
    "CLAIM_SIMILARITY_THRESHOLD",
    "AuditBackend",
    "AuditReport",
    "DiffCandidate",
    "FactDiff",
    "JudgeVerdict",
    "Judgement",
    "SecondPassResults",
    "TypeRate",
    "UsageTotals",
    "WorstSource",
    "diff_facts",
    "judge",
    "render_markdown",
    "run_audit",
    "sample_sources",
    "second_pass",
]
