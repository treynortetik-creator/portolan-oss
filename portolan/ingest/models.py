"""Ingest data model (build-contracts.md "Source model and intake").

`RawSource` is what a connector (`portolan/connectors/`) hands to
`intake()`. `SourceRow` mirrors the `sources` table column for column.
`PreparedText` and `LocatorSpan` are `prepare.py`'s output: the source's
text with locators injected, plus each locator's character span.

`SourceType` is re-exported from `portolan.extract.schema`, one closed list
for both (`extraction-schema-v0.md` "What the extractor receives").
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from portolan.extract.schema import SourceType

__all__ = [
    "LocatorSpan",
    "Participant",
    "PreparedText",
    "RawSource",
    "SourceRow",
    "SourceType",
]


class Participant(BaseModel):
    """One entry in `RawSource.participants` (build-contracts.md: "name,
    email|None, role|None"). Distinct from `portolan.extract.schema.
    Participant`, which additionally carries a resolved `registry_id` and
    is the shape the *extractor* sees after resolution — this one is the
    raw shape a connector hands to intake, before any registry lookup.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    email: str | None = None
    role: str | None = None


class RawSource(BaseModel):
    """What a connector hands to `intake()` (build-contracts.md)."""

    model_config = ConfigDict(extra="forbid")

    source_id: str
    source_type: SourceType
    source_date: date
    participants: list[Participant] = Field(default_factory=list)
    # CRM ids or names the connector already knows point at a customer
    # (spec sec 5.2's third filter rule), e.g. ["A-1010", "Ashgrove Health"].
    account_hints: list[str] = Field(default_factory=list)
    # Raw text, already normalised to plain text by the connector. Bare of
    # locators until `prepare_text` runs, except for source types (calls)
    # whose natural transcript already carries them.
    text: str
    original_bytes: bytes | None = None
    original_content_type: str | None = None
    # Connector-specific extras with no named field: an email's Message-ID
    # and subject, a ticket's comments, a CRM row's object and id, an RSS
    # item's link. `prepare.py`'s module docstring lists the keys each
    # source type reads.
    metadata: dict[str, Any] = Field(default_factory=dict)


class LocatorSpan(BaseModel):
    """One entry in `PreparedText.locators`: a locator string (the same
    string that goes after the `@` in a claim citation, provenance-
    chain.md's locator table) and the half-open `[start, end)` character
    span of `PreparedText.text` it covers.
    """

    model_config = ConfigDict(extra="forbid")

    locator: str
    start: int
    end: int


class PreparedText(BaseModel):
    """`prepare_text`'s output: text with locators injected (quoted text
    and boilerplate already stripped for email), plus the locator index.
    """

    model_config = ConfigDict(extra="forbid")

    text: str
    locators: list[LocatorSpan] = Field(default_factory=list)
    # Character count removed by quote-stripping and boilerplate removal,
    # measured against the connector's raw text — build item 3: "record
    # what was removed".
    removed_chars: int = 0
    # Set when stripping would have emptied non-empty input, so the
    # unstripped original was kept. Only the email handler strips.
    stripping_skipped: bool = False


class SourceRow(BaseModel):
    """Mirror of the `sources` table, column for column."""

    model_config = ConfigDict(extra="forbid")

    slug: str
    # The connector's own id (`RawSource.source_id`), the un-versioned base
    # of `slug`. Nullable in the schema; `intake()` always sets it.
    source_id: str | None = None
    source_type: str
    s3_key: str | None = None
    content_hash: str | None = None
    source_date: date | None = None
    ingested_at: datetime
    extraction_status: str = "pending"
    filter_reason: str | None = None
    schema_version: str | None = None
    attempts: int = 0
    last_error: str | None = None
    summary: str | None = None
    participants: list[Participant] = Field(default_factory=list)
    # `supersedes_source` (prior slug of a versioned re-ingest) and `raw`
    # (the connector's `RawSource.metadata`, verbatim); see `intake`.
    metadata: dict[str, Any] = Field(default_factory=dict)
    # The extraction signature the last time this source reached
    # done/partial (NULL until then), so an unchanged source extracted the
    # same way can be skipped.
    extraction_model: str | None = None
    extraction_prompt_variant: str | None = None
    # Extraction passes unioned (`--samples N`). NOT NULL DEFAULT 1: older
    # sources really were extracted with one pass.
    extraction_samples: int = 1
