"""Pydantic models for the registry.

`EntityKind` is imported from `portolan.extract.schema`, not redefined: one
closed entity-kind list (matching `registry_entities.kind`'s CHECK), not
two copies that could drift.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from portolan.extract.schema import EntityKind

__all__ = ["Entity", "EntityKind", "ImportReport", "Match", "Resolution"]


class Entity(BaseModel):
    """One registry entity: an account, contact, product, competitor,
    internal person, or `other` (the vendor itself and its branches).

    `id` is `registry_entities.crm_id` (`registry.py`'s module docstring).
    `import_crm` never leaves that null; a row made elsewhere without one
    loads as `row-<db id>`.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    kind: EntityKind
    canonical_name: str
    crm_id: str | None = None
    parent_id: str | None = None
    account_kind: str | None = None
    email: str | None = None
    email_domain: str | None = None
    aliases: list[str] = Field(default_factory=list)
    # False for a churned account or departed contact (`importer._is_inactive`);
    # `Registry.resolve` filters on it.
    active: bool = True
    # The date the CRM status names for an inactive entity (migration
    # 0032); `Registry.resolve(as_of=...)` still resolves it for a fact
    # said on or before this date.
    inactive_since: date | None = None


class Resolution(BaseModel):
    """The result of `Registry.resolve()`.

    `status="ambiguous"` carries every candidate id in `candidates` and,
    when they share a parent (or one candidate is every other's ancestor),
    that parent's id in `attach_to` (spec sec 3: "attaches to their shared
    parent with a flag"). `status="unknown"` means no candidate at any
    tier — never a guess.
    """

    model_config = ConfigDict(extra="forbid")

    status: Literal["resolved", "ambiguous", "unknown"]
    entity_id: str | None = None
    candidates: list[str] = Field(default_factory=list)
    attach_to: str | None = None


class Match(BaseModel):
    """One hit from `Registry.scan()`: the entity, the text matched, and
    its span in the scanned text."""

    model_config = ConfigDict(extra="forbid")

    entity_id: str
    kind: EntityKind
    canonical_name: str
    matched_text: str
    start: int
    end: int


class ImportReport(BaseModel):
    """Returned by `Registry.import_crm()`: counts plus a short human-readable
    trail of every merge and every refused parent link."""

    model_config = ConfigDict(extra="forbid")

    created: int = 0
    updated: int = 0
    merged: int = 0
    queued: int = 0
    notes: list[str] = Field(default_factory=list)
