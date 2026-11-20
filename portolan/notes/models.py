"""Pydantic models for `portolan.notes` (`01-spec/build-contracts.md`
"Notes and claims").

`NoteRow` mirrors the `notes` table (`portolan/db/migrations/0001_init.sql`),
minus the generated `search` column, plus two fields that are not columns:
`synthesis` and `related`. Both live inside `body_md` (the dossier pattern's
rewritable top layer, and the `## Related` section, `render.py`), so
`read_note` parses them out of the body and `write_note` renders them back
in. A claims-only edit therefore round-trips them unchanged.

`Claim` is one `claims` row (without `archived`; `read_note` filters those
out) plus its extra citations.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "0.1"  # matches portolan.extract.schema.SCHEMA_VERSION


class NoteType(StrEnum):
    """The closed list, `portolan-spec-v2.md` sec 2: "Types are closed"."""

    ACCOUNT = "account"
    CONTACT = "contact"
    PRODUCT = "product"
    COMPETITOR = "competitor"
    SIGNAL = "signal"
    THEME = "theme"
    DECISION = "decision"
    ACTION = "action"
    SOURCE = "source"
    REPORT = "report"
    # The tenant's company-context note (`portolan.context`); an entry point
    # like `source`/`report`, so the lint's orphan check skips it.
    CONTEXT = "context"


class NoteStatus(StrEnum):
    LIVING = "living"
    SUPERSEDED = "superseded"
    ARCHIVED = "archived"


# Days a note may go unrefreshed before the lint's `stale` check flags it
# (spec sec 2: "stale_after: per-type default, overridable"; the numbers are
# this build's choice). Source/report are historical records, so effectively
# never.
STALE_AFTER_DAYS: dict[NoteType, int] = {
    NoteType.ACCOUNT: 180,
    NoteType.CONTACT: 180,
    NoteType.PRODUCT: 365,
    NoteType.COMPETITOR: 365,
    NoteType.SIGNAL: 120,
    NoteType.THEME: 90,
    NoteType.DECISION: 365,
    NoteType.ACTION: 60,
    NoteType.SOURCE: 3650,
    NoteType.REPORT: 3650,
    # A living profile of the tenant, so it drifts like an account note.
    NoteType.CONTEXT: 180,
}


class NoteRow(BaseModel):
    """One `notes` row plus `synthesis` and `related`, which are parsed out
    of and rendered back into `body_md` (module docstring).
    """

    model_config = ConfigDict(extra="forbid")

    slug: str
    type: NoteType
    title: str
    description: str | None = None
    status: NoteStatus = NoteStatus.LIVING
    created: datetime
    updated: datetime
    stale_after: datetime | None = None
    schema_version: str | None = SCHEMA_VERSION
    crm_id: str | None = None
    parent_slug: str | None = None
    superseded_by: str | None = None
    body_md: str = ""
    version: int = 1

    # --- Python-only, not persisted as their own columns (see docstring) ---
    synthesis: str | None = None
    related: list[str] = Field(default_factory=list)


class ClaimCitation(BaseModel):
    """One extra citation on a claim, from `claim_citations`: the same fact
    seen again (another source, or the same source at another locator)."""

    model_config = ConfigDict(extra="forbid")

    id: int | None = None
    source_slug: str
    locator: str
    quote: str | None = None
    recorded: datetime | None = None


class Claim(BaseModel):
    """One atomic, cited fact underneath a note's synthesis
    (`build-contracts.md` "Notes and claims"). Mirrors the `claims` table.
    """

    model_config = ConfigDict(extra="forbid")

    id: int | None = None
    note_slug: str
    fact_type: str
    claim_text: str
    source_slug: str
    locator: str
    quote: str | None = None
    speaker: str | None = None
    valid_from: date | None = None
    valid_to: date | None = None
    recorded: datetime
    superseded_at: datetime | None = None
    severity: Literal["low", "medium", "high"] | None = None
    sentiment: int | None = None
    confidence: float | None = None
    superseded_by: int | None = None
    # Extra citations beyond the primary `source_slug`/`locator`, oldest
    # first; rendered after the primary marker on the claim's line.
    citations: list[ClaimCitation] = Field(default_factory=list)
    # The company-context note's `updated` when this claim was extracted;
    # provenance only, not rendered.
    company_context_updated_at: datetime | None = None
