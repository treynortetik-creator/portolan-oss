"""Pydantic models for `portolan.consolidate` (`01-spec/quality-loop.md`
"The consolidation pass", "Human edits"; `01-spec/build-contracts.md`
"Report, audit, consolidation").
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ConsolidationStatus = Literal["consolidated", "failed", "skipped_no_claims"]


class CheckResult(BaseModel):
    """`check()`'s mechanical verdict on a drafted synthesis.

    Review finding G7 / task item 4 (fix wave 2026-09-12): the original
    rule -- "every claim id must appear" -- is what made archiving invert
    itself after the first consolidation (a synthesis forced to cite
    EVERY remaining claim makes every one of them "load-bearing", so the
    next pass could only archive claims added since, never the genuinely
    old ones). The rule now is completeness in the other direction: every
    citation marker and claim-id tag the draft DOES use must be real (no
    invented ones -- `unknown_citations`/`unknown_claim_ids`, still hard
    failures), and it must cite at least `min_required_citations` claims
    (a floor, not the full list) -- the "five to ten claims that matter"
    the prompt now asks for, not full coverage. The draft must also stay
    under 40 lines and carry no relative-time language."""

    model_config = ConfigDict(extra="forbid")

    ok: bool
    line_count: int
    too_long: bool
    cited_claim_ids: list[int] = Field(default_factory=list)
    min_required_citations: int = 0
    too_few_citations: bool = False
    unknown_citations: list[str] = Field(default_factory=list)
    unknown_claim_ids: list[int] = Field(default_factory=list)
    relative_time_hits: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


class ConsolidationOutcome(BaseModel):
    """What `consolidate_note` did with one note."""

    model_config = ConfigDict(extra="forbid")

    note_slug: str
    status: ConsolidationStatus
    draft_attempts: int = 0
    used_strong: bool = False
    check: CheckResult | None = None
    new_version: int | None = None
    archived_claim_ids: list[int] = Field(default_factory=list)
    archive_note_slug: str | None = None
    review_queue_id: int | None = None


class EditRecord(BaseModel):
    """Mirror of the `edit_records` table (`0001_init.sql`)."""

    model_config = ConfigDict(extra="forbid")

    id: int
    note_slug: str
    claim_id: int | None
    editor: str
    edited_at: datetime
    before_text: str | None
    after_text: str | None
    reason: str | None
