"""Typed models over the bench's two answer-key files.

`corpus/truth-table.yaml` is the ANSWER KEY the synthetic corpus is
generated from: every fact the generator planted, the entities it touches,
and the `expectations` block the graph score checks the database against.
`corpus/out/manifest.json` is the generator's placement record: which
source (and which locator inside it) each planted fact actually landed in.

`Truth` joins the two into one `planted(source_id) -> list[PlantedFact]`
index, plus an entity name/alias index used to score "did the extractor's
entity mentions overlap the planted entities" without a live registry
(portolan.registry is a different wave's module and is not imported here).
"""

from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

DEFAULT_TRUTH_TABLE_PATH = Path("corpus/truth-table.yaml")
DEFAULT_MANIFEST_PATH = Path("corpus/out/manifest.json")


# --------------------------------------------------------------------------
# corpus/truth-table.yaml
# --------------------------------------------------------------------------


class Account(BaseModel):
    """One row of `accounts:`."""

    model_config = ConfigDict(extra="allow")

    id: str
    crm_id: str | None = None
    name: str
    kind: str | None = None
    parent: str | None = None
    domain: str | None = None
    arr: float = 0.0
    region: str | None = None
    aliases: list[str] = Field(default_factory=list)
    status: str | None = None


class Contact(BaseModel):
    """One row of `contacts:`. `crm_id: null` marks a contact deliberately
    kept out of the CRM (`maria-chen-personal`), which exercises the
    unknown-entity queue."""

    model_config = ConfigDict(extra="allow")

    id: str
    crm_id: str | None = None
    name: str
    account: str | None = None
    title: str | None = None
    email: str | None = None
    status: str | None = None
    note: str | None = None


class FactSourceRef(BaseModel):
    """One entry in a fact's `sources:` list."""

    model_config = ConfigDict(extra="allow")

    id: str
    type: str
    speaker: str | None = None
    position: str | None = None
    internal_only: bool = False


class TruthFact(BaseModel):
    """One row of `facts:` — a planted fact (`type` is a real `FactType`) or
    a distractor (`type: distractor`, `id` in `D01`..`D04`)."""

    model_config = ConfigDict(extra="allow")

    id: str
    type: str
    claim: str | None = None
    entities: list[str] = Field(default_factory=list)
    date: dt.date | None = None
    valid_from: dt.date | None = None
    valid_to: dt.date | None = None
    supersedes: str | None = None
    sources: list[FactSourceRef] = Field(default_factory=list)
    phrasing: str | None = None
    tests: list[str] = Field(default_factory=list)

    @property
    def is_distractor(self) -> bool:
        return self.type == "distractor"


class TruthTable(BaseModel):
    """The full parsed `truth-table.yaml`. `sources` and `expectations` are
    kept as raw dicts/lists (mirroring the YAML) rather than fully typed —
    they're heterogeneous per source type / expectation kind, and
    `graph.py` reads `expectations` directly rather than through a second
    layer of models."""

    model_config = ConfigDict(extra="allow")

    meta: dict[str, Any] = Field(default_factory=dict)
    accounts: list[Account] = Field(default_factory=list)
    contacts: list[Contact] = Field(default_factory=list)
    facts: list[TruthFact] = Field(default_factory=list)
    sources: dict[str, Any] = Field(default_factory=dict)
    expectations: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def load(cls, path: Path | str = DEFAULT_TRUTH_TABLE_PATH) -> TruthTable:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(data)


# --------------------------------------------------------------------------
# corpus/out/manifest.json
# --------------------------------------------------------------------------


class FactPlacement(BaseModel):
    """One entry in a manifest source's `facts` list: where, inside that
    source, the generator planted the fact."""

    model_config = ConfigDict(extra="allow")

    fact_id: str
    locator: str
    line_or_comment: str
    excerpt: str


class ManifestSource(BaseModel):
    """One entry in manifest.json's `sources` list."""

    model_config = ConfigDict(extra="allow")

    id: str
    type: str
    path: str | None = None
    date: dt.date | None = None
    participants: list[str] = Field(default_factory=list)
    accounts: list[str] = Field(default_factory=list)
    token_estimate: int | None = None
    is_noise: bool = False
    is_distractor: bool = False
    distractor_id: str | None = None
    facts: list[FactPlacement] = Field(default_factory=list)


class Manifest(BaseModel):
    """The full parsed `manifest.json`."""

    model_config = ConfigDict(extra="allow")

    generator_version: str | None = None
    schema_version: str | None = None
    mode: str | None = None
    reduced: bool | None = None
    generated_at: str | None = None
    truth_table_hash: str | None = None
    sources: list[ManifestSource] = Field(default_factory=list)
    # fact_id -> list of placements, each carrying its own source_id (unlike
    # a ManifestSource's `facts`, where source_id is implied by the source
    # the list lives on). Kept as raw dicts: it duplicates `sources[].facts`
    # and nothing here needs it typed to cross-check the two.
    facts: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)

    @classmethod
    def load(cls, path: Path | str = DEFAULT_MANIFEST_PATH) -> Manifest:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(data)

    def source(self, source_id: str) -> ManifestSource:
        for src in self.sources:
            if src.id == source_id:
                return src
        raise KeyError(f"manifest has no source {source_id!r}")


# --------------------------------------------------------------------------
# The joined view: truth-table fact definitions x manifest placements.
# --------------------------------------------------------------------------


class PlantedFact(BaseModel):
    """One planted fact, as it is required to appear in one specific
    source. `planted()` is keyed by `(source_id, fact_id)`: the same
    `fact_id` produces a *separate* `PlantedFact` per source it is planted
    in (e.g. F02 in both a ticket and a call, for `cross_source_dedup` —
    each source must independently surface it), but when the manifest
    places the *same* fact_id more than once *within one source* (F04,
    `within_source_dedup`: two mentions twenty minutes apart in one call),
    those placements collapse into one `PlantedFact` here — matching any
    one of `locators`/`excerpts` counts, since the extractor is allowed to
    emit either one merged fact or two, per the truth table's own note."""

    model_config = ConfigDict(extra="forbid")

    fact_id: str
    source_id: str
    source_type: str
    type: str
    entities: list[str] = Field(default_factory=list)
    claim: str | None = None
    locator: str
    locators: list[str] = Field(default_factory=list)
    line_or_comment: str
    excerpt: str
    excerpts: list[str] = Field(default_factory=list)
    tests: list[str] = Field(default_factory=list)
    is_distractor: bool = False


def _norm(s: str) -> str:
    """Case- and whitespace-insensitive normalisation for name/alias
    matching. Deliberately does not touch punctuation (`St. Aldric` and
    `St Aldric` are both real, separately-listed aliases in the truth
    table)."""
    return re.sub(r"\s+", " ", s.strip().lower())


def _norm_loose(s: str) -> str:
    """`_norm`, plus dashes/underscores/colons treated as whitespace too.
    Diagnosis (`var/bench/diag/diagnosis.md`) found a produced fact's
    `entities` list carrying registry-id-shaped guesses the model invented
    for lack of a real registry (`northlake-health-system`,
    `account:northlake`) rather than the plain mention string
    `entity_names()` already covers (`Northlake Health System`) — `_norm`
    alone doesn't unify `-`/`:` with a space, so those guesses missed every
    variant by punctuation alone despite naming the right entity."""
    return _norm(re.sub(r"[-_:]+", " ", s))


def _contains_phrase(haystack: str, phrase: str) -> bool:
    """Whole-word-bounded substring search: does `phrase` (already `_norm`-
    or `_norm_loose`-normalized) appear as a contiguous run of words inside
    `haystack`? Lets a descriptive produced mention ("Northlake's Director
    of Clinical Engineering, Derek Hollis") or a longer registry-context
    style string still count as naming the entity, without a bare 2-3
    letter alias (`T2`, `Link`) matching inside an unrelated word."""
    if len(phrase) < 4:
        return haystack == phrase
    return re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", haystack) is not None


def _variants(entity_id: str, name: str, aliases: list[str] | tuple[str, ...] = ()) -> set[str]:
    out = {_norm(name), _norm(entity_id), _norm(entity_id.replace("-", " "))}
    for alias in aliases:
        out.add(_norm(alias))
    return out


def _build_entity_variants(table: TruthTable) -> dict[str, frozenset[str]]:
    """entity id -> the set of normalised strings (canonical name, id,
    id-with-spaces, aliases) that should be treated as naming it. Covers
    every closed list the truth table's `entities:` fields draw from:
    products, competitors, internal people, branches, accounts, contacts."""
    variants: dict[str, set[str]] = {}
    meta = table.meta or {}

    for product in meta.get("products", []) or []:
        variants[product["id"]] = _variants(
            product["id"], product["name"], product.get("aliases", [])
        )
    for competitor in meta.get("competitors", []) or []:
        variants[competitor["id"]] = _variants(
            competitor["id"], competitor["name"], competitor.get("aliases", [])
        )
    for person in meta.get("internal_people", []) or []:
        variants[person["id"]] = _variants(person["id"], person["name"])
    for branch in meta.get("branches", []) or []:
        variants[branch["id"]] = _variants(branch["id"], branch["name"])
    for account in table.accounts:
        variants[account.id] = _variants(account.id, account.name, account.aliases)
    for contact in table.contacts:
        variants.setdefault(contact.id, set()).update(_variants(contact.id, contact.name))

    return {entity_id: frozenset(names) for entity_id, names in variants.items()}


def _build_reverse_index(entity_variants: dict[str, frozenset[str]]) -> dict[str, set[str]]:
    reverse: dict[str, set[str]] = {}
    for entity_id, names in entity_variants.items():
        for name in names:
            reverse.setdefault(name, set()).add(entity_id)
    return reverse


def _build_planted_index(
    table: TruthTable, manifest: Manifest, fact_by_id: dict[str, TruthFact]
) -> dict[str, list[PlantedFact]]:
    index: dict[str, list[PlantedFact]] = {}
    for src in manifest.sources:
        # Group placements by fact_id first: a manifest source can place the
        # same fact_id more than once (F04's two same-call mentions) and
        # those collapse to one PlantedFact with multiple acceptable
        # locators, rather than two independent requirements.
        grouped: dict[str, list[FactPlacement]] = {}
        order: list[str] = []
        for placement in src.facts:
            if placement.fact_id not in grouped:
                order.append(placement.fact_id)
            grouped.setdefault(placement.fact_id, []).append(placement)

        entries: list[PlantedFact] = []
        for fact_id in order:
            placements = grouped[fact_id]
            fact_def = fact_by_id.get(fact_id)
            primary = placements[0]
            entries.append(
                PlantedFact(
                    fact_id=fact_id,
                    source_id=src.id,
                    source_type=src.type,
                    type=fact_def.type if fact_def else "unknown",
                    entities=list(fact_def.entities) if fact_def else [],
                    claim=fact_def.claim if fact_def else None,
                    locator=primary.locator,
                    locators=[p.locator for p in placements],
                    line_or_comment=primary.line_or_comment,
                    excerpt=primary.excerpt,
                    excerpts=[p.excerpt for p in placements],
                    tests=list(fact_def.tests) if fact_def else [],
                    is_distractor=bool(fact_def and fact_def.is_distractor),
                )
            )
        index[src.id] = entries
    return index


class Truth:
    """The bench's one answer-key object: a loaded `TruthTable` joined with
    a loaded `Manifest`."""

    def __init__(self, table: TruthTable, manifest: Manifest) -> None:
        self.table = table
        self.manifest = manifest
        self._fact_by_id: dict[str, TruthFact] = {fact.id: fact for fact in table.facts}
        self._entity_variants = _build_entity_variants(table)
        self._variant_to_ids = _build_reverse_index(self._entity_variants)
        self._planted_by_source = _build_planted_index(table, manifest, self._fact_by_id)

    @classmethod
    def load(
        cls,
        truth_table_path: Path | str = DEFAULT_TRUTH_TABLE_PATH,
        manifest_path: Path | str = DEFAULT_MANIFEST_PATH,
    ) -> Truth:
        return cls(TruthTable.load(truth_table_path), Manifest.load(manifest_path))

    @classmethod
    def for_manifest(
        cls, manifest: Manifest, truth_table_path: Path | str = DEFAULT_TRUTH_TABLE_PATH
    ) -> Truth:
        """Pair an already-loaded `Manifest` (e.g. one `run_extraction` was
        given directly) with the truth table, without re-reading the
        manifest file."""
        return cls(TruthTable.load(truth_table_path), manifest)

    # -- fact lookup ------------------------------------------------------

    def fact(self, fact_id: str) -> TruthFact:
        return self._fact_by_id[fact_id]

    def planted(self, source_id: str) -> list[PlantedFact]:
        """Every planted fact placed in `source_id`, in manifest order."""
        return list(self._planted_by_source.get(source_id, []))

    def all_planted(self) -> list[PlantedFact]:
        return [pf for entries in self._planted_by_source.values() for pf in entries]

    def planted_fact_ids(self) -> list[str]:
        return [f.id for f in self.table.facts if not f.is_distractor]

    def distractor_fact_ids(self) -> list[str]:
        return [f.id for f in self.table.facts if f.is_distractor]

    def distractor_source_ids(self) -> set[str]:
        return {src.id for src in self.manifest.sources if src.is_distractor}

    # -- entity name/alias matching ---------------------------------------

    def entity_names(self, entity_id: str) -> frozenset[str]:
        return self._entity_variants.get(entity_id, frozenset())

    def entities_overlap(self, planted_entity_ids: list[str], produced_mentions: list[str]) -> bool:
        """True when at least one of `planted_entity_ids` is named by at
        least one of `produced_mentions` -- registry id, canonical name,
        alias, or (for a person) the truth table's contact name, all
        interchangeably, in any of three progressively looser senses:

        1. exact match, case/whitespace-insensitive (`_norm`);
        2. exact match treating `-`/`_`/`:` as whitespace too (`_norm_loose`)
           -- a registry-id-shaped guess like `northlake-health-system` or
           `account:northlake` names the same entity as the space-separated
           canonical name without knowing its real id;
        3. whole-word-bounded containment (`_contains_phrase`) -- a longer,
           descriptive produced mention ("Northlake's Director of Clinical
           Engineering, Derek Hollis") that names the entity along with
           other words still counts.

        A planted fact with no entities listed is trivially satisfied —
        there is nothing to check.
        """
        if not planted_entity_ids:
            return True
        mentions = [m for m in produced_mentions if m and m.strip()]
        if not mentions:
            return False
        normalized_mentions = {_norm(m) for m in mentions}
        loose_mentions = {_norm_loose(m) for m in mentions}

        for entity_id in planted_entity_ids:
            names = self.entity_names(entity_id)
            if normalized_mentions & names:
                return True
            loose_names = {_norm_loose(n) for n in names}
            if loose_mentions & loose_names:
                return True
            for name in names:
                loose_name = _norm_loose(name)
                if any(_contains_phrase(m, loose_name) for m in loose_mentions):
                    return True
        return False

    def identify_entity(self, mention: str) -> str | None:
        """Reverse-lookup a mention string to a truth-table entity id, only
        when the name/alias is unambiguous (used for `entity_accuracy`:
        checking a resolved `registry_id` the extractor supplied against
        what the truth table says that mention actually is)."""
        candidates = self._variant_to_ids.get(_norm(mention))
        if candidates and len(candidates) == 1:
            return next(iter(candidates))
        return None

    # -- consistency --------------------------------------------------------

    def missing_source_refs(self) -> list[str]:
        """Fact `sources[].id` values that don't correspond to any manifest
        source — should be empty; used by the load/cross-check test."""
        manifest_ids = {src.id for src in self.manifest.sources}
        missing: list[str] = []
        for fact in self.table.facts:
            for ref in fact.sources:
                if ref.id not in manifest_ids:
                    missing.append(ref.id)
        return missing
