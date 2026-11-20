"""Pydantic models for `portolan.merge` (`01-spec/build-contracts.md`
"Merge").

`MergeOutcome`/`ActionOutcome` carry per-call rollups (notes created and
updated, compare calls, tokens, queue rows) that `merge_extraction` sums
into one `MergeReport` per source.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

FactOutcomeStatus = Literal["new", "same", "extends", "contradicts", "queued_unknown"]
ActionOutcomeStatus = Literal["new", "duplicate", "queued_unknown"]


class CompareVerdict(BaseModel):
    """The compare call's structured reply (`build-contracts.md` Merge
    section: "asking for same | extends | contradicts and a one-line
    reason").

    **Case normalization (fix wave 2026-09-12, found profiling one real
    source through the pipeline against `amazon/nova-lite-v1` via
    OpenRouter).** A real compare call came back `{"verdict": "SAME", ...}`
    -- schema-valid JSON (the `response_format`/tool schema this project
    sends only constrains shape, not casing, on `enum`/`Literal` string
    values with most providers) but not one of this `Literal`'s exact
    lowercase members, so a strict `model_validate` raised on every source
    that reached even one gated candidate: `merge_fact` -> `merge_extraction`
    -> `_retry_stage` (three attempts, all doomed -- the model says the
    same thing every time) -> the whole source dead-lettered, `failed`,
    zero notes/claims, for a call that actually succeeded and answered the
    real question correctly. The `mode="before"` validator below
    lowercases (and strips) a string verdict before the `Literal` check
    runs, so "SAME"/"Same "/"same" all resolve the same way; a non-string
    input is passed through unchanged so the normal `Literal`/type error
    still fires for a genuinely malformed reply.
    """

    model_config = ConfigDict(extra="forbid")

    verdict: Literal["same", "extends", "contradicts"]
    reason: str

    @field_validator("verdict", mode="before")
    @classmethod
    def _normalize_verdict_case(cls, v: object) -> object:
        return v.strip().lower() if isinstance(v, str) else v


def compare_verdict_json_schema() -> dict:
    return CompareVerdict.model_json_schema()


class MergeOutcome(BaseModel):
    """What `merge_fact` did with one fact."""

    model_config = ConfigDict(extra="forbid")

    fact_id: str
    status: FactOutcomeStatus
    note_slug: str | None = None
    claim_id: int | None = None
    # A new `claims` row was inserted (new/extends/contradicts, never same).
    claim_inserted: bool = False
    # A `same` verdict wrote a new `claim_citations` row (false when that
    # citation was already on file).
    citation_inserted: bool = False
    # An earlier version of the same source already recorded this fact;
    # nothing was written (`merge.facts._claim_from_an_earlier_version`).
    already_recorded: bool = False
    reason: str | None = None

    compare_calls: int = 0
    compare_input_tokens: int = 0
    compare_output_tokens: int = 0
    notes_created: list[str] = Field(default_factory=list)
    notes_updated: list[str] = Field(default_factory=list)
    queue_rows: int = 0
    # Product/competitor slugs for the home note's `## Related` list, applied
    # in `merge_extraction`'s one end-of-source write ("one version per
    # source, not per claim").
    related_links: list[str] = Field(default_factory=list)


class ActionOutcome(BaseModel):
    """What `merge_action` did with one action."""

    model_config = ConfigDict(extra="forbid")

    status: ActionOutcomeStatus
    action_id: int | None = None
    note_slug: str | None = None
    notes_created: list[str] = Field(default_factory=list)
    notes_updated: list[str] = Field(default_factory=list)


class FactPlacement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fact_id: str
    claim_id: int | None
    note_slug: str | None


class MergeReport(BaseModel):
    """Everything `merge_extraction` did with one source's extraction
    result (`build-contracts.md` Merge section)."""

    model_config = ConfigDict(extra="forbid")

    source_slug: str
    outcome_counts: dict[str, int] = Field(default_factory=dict)
    notes_created: int = 0
    notes_updated: int = 0
    claims_inserted: int = 0
    citations_inserted: int = 0
    facts_already_recorded: int = 0
    queue_rows: int = 0
    compare_calls: int = 0
    compare_input_tokens: int = 0
    compare_output_tokens: int = 0
    placements: list[FactPlacement] = Field(default_factory=list)
    actions_new: int = 0
    actions_deduped: int = 0
