"""The consolidation pass and human edits (`01-spec/quality-loop.md`
"The consolidation pass", "Human edits"; `01-spec/build-contracts.md`
"Report, audit, consolidation", wave C2). Public surface:
`needs_consolidation`, `draft_synthesis`, `check`, `consolidate_note`,
`run_consolidation`, `record_edit`, `edits_for`.

CLI: `python -m portolan.consolidate [--limit 20] [--llm fake|agent_sdk]`.
"""

from __future__ import annotations

from portolan.consolidate.archive import (
    archive_note_slug,
    commit_archive,
    select_claims_to_archive,
)
from portolan.consolidate.draft import CLAIM_ID_TAG_RE, check, draft_synthesis
from portolan.consolidate.edits import StaleEditError, edit_citation, edits_for, record_edit
from portolan.consolidate.models import CheckResult, ConsolidationOutcome, EditRecord
from portolan.consolidate.run import consolidate_note, run_consolidation
from portolan.consolidate.trigger import (
    CONSOLIDATION_CLAIMS_THRESHOLD,
    CONSOLIDATION_UPDATES_THRESHOLD,
    needs_consolidation,
)

__all__ = [
    "CLAIM_ID_TAG_RE",
    "CONSOLIDATION_CLAIMS_THRESHOLD",
    "CONSOLIDATION_UPDATES_THRESHOLD",
    "CheckResult",
    "ConsolidationOutcome",
    "EditRecord",
    "StaleEditError",
    "archive_note_slug",
    "check",
    "commit_archive",
    "consolidate_note",
    "draft_synthesis",
    "edit_citation",
    "edits_for",
    "needs_consolidation",
    "record_edit",
    "run_consolidation",
    "select_claims_to_archive",
]
