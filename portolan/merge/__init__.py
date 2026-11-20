"""Merge extracted facts and actions into the graph (`01-spec/build-contracts.md`
"Merge"). Public surface: `__all__`.
"""

from __future__ import annotations

from portolan.merge.actions import merge_action
from portolan.merge.facts import merge_fact
from portolan.merge.models import (
    ActionOutcome,
    CompareVerdict,
    FactPlacement,
    MergeOutcome,
    MergeReport,
    compare_verdict_json_schema,
)
from portolan.merge.pipeline import merge_extraction

__all__ = [
    "ActionOutcome",
    "CompareVerdict",
    "FactPlacement",
    "MergeOutcome",
    "MergeReport",
    "compare_verdict_json_schema",
    "merge_action",
    "merge_extraction",
    "merge_fact",
]
