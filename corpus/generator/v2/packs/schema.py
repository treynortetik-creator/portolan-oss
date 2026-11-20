"""`VerticalPack` -- the per-vertical content bank `simulate_world()` draws
from (`01-spec/world-generator-v2-2026-09-22.md` section 4, "Diversity and
anti-memorization": "Each has a *pack* (`corpus/generator/v2/packs/
<vertical>.yaml`): product archetypes, buyer roles ... an event library,
arc weights, a topic tree, jargon and competitor archetypes").

This work package (WP1) defines the schema and one complete, validating
example (`packs/medical_devices.yaml`, the vertical v1 already covers).
WP2 fills the other seven packs (senior-living tech, pharmacy services,
EHR add-ons, healthcare staffing, medical supplies, telehealth, plus the
held-out eighth vertical, spec section 4) against this same schema -- so
every field below carries a docstring explaining what it is *for*, not
just its type, and every list-shaped field documents its minimum size.

Names (vendor, product, competitor, person, account) are deliberately
**not** part of a pack. Spec section 6: the name pools are generic and
split-partitioned by hash (`v2.names`), shared across every vertical --
a pack only says how this vertical *composes* names (which account-kind
suffixes it uses, what a product line looks like), never which literal
names to draw.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, RootModel, field_validator, model_validator

from corpus.generator.v2.models import ArcType, EventType
from portolan.extract.schema import FactType

# Every pack's fact_directives must cover every production fact type, and
# every type needs at least this many directives. Spec section 4,
# "Phrasing": "12 or more *directives*: hedged, numbers spelled out,
# colloquial, embedded mid-sentence, passive, through an alias, corrected
# mid-sentence, list item, and so on."
MIN_DIRECTIVES_PER_FACT_TYPE = 12

# Spec section 4, "Personas": "a bank of 40 or more" is the whole system's
# aggregate across every pack a world can draw personas from, not one
# pack's own minimum -- resolved ambiguity, see this work package's
# report. One pack's own bank just needs enough variety that a world's
# ~20-ish people (spec section 1: "10 to 14 vendor staff, 3 to 6 contacts
# per account") aren't all wearing the same persona.
MIN_PERSONAS_PER_PACK = 8

# Spec section 4, "Hard negatives": the decoy classes named there --
# hypotheticals, negations, hearsay about another company, generic
# industry talk, rhetorical questions, vendor-internal speculation, and
# an already-resolved issue told as history -- are seven distinct classes.
MIN_DECOY_CLASSES = 7

# Spec section 4, "Phrasing": "Each fact type has 15 or more event
# variants per vertical, replacing v1's one template per type." Coordinator
# review of 7f779a3 (2026-09-22): counted across a pack's `event_templates`,
# every fact type must reach this floor -- checked by `_claim_templates_
# meet_variant_floor` below, summing (and deduplicating, case/whitespace-
# insensitively) every `claim_templates[fact_type]` entry across every
# `EventTemplate` that names that fact type.
MIN_CLAIM_VARIANTS_PER_FACT_TYPE = 15


class ProductArchetype(BaseModel):
    """One shape of product this vertical sells. `simulate_world()` fills
    `name_template` in with a product-line name drawn from `v2.names.
    PRODUCT_LINE_NAMES` (split-partitioned) plus the vendor's own name --
    e.g. a `name_template` of `"{line} {tier}"` becomes `"Meraline T3"`.
    Mirrors v1's product-line shape (`corpus.generator.worlds.
    _pick_vendor_mythology`: family / T2 / T3 / Link / service contract),
    generalized so each vertical can have a differently-shaped line (a
    staffing vertical's "products" are service tiers, not devices)."""

    model_config = ConfigDict(extra="forbid")

    kind: str = Field(description="Short archetype label, e.g. 'infusion_pump', 'ehr_module'.")
    name_template: str = Field(
        description=(
            "Format string with `{line}` (a drawn product-line name) and/or "
            "`{tier}` placeholders, e.g. '{line} {tier}'."
        )
    )
    tiers: list[str] = Field(
        default_factory=list,
        description="Tier labels this archetype comes in, e.g. ['T2', 'T3', 'Link'].",
    )
    known_issue_pool: list[str] = Field(
        default_factory=list,
        description=(
            "Candidate `Product.known_issues` entries for this archetype -- plain-"
            "language descriptions a complaint/risk fact's claim can reference."
        ),
    )


class BuyerRole(BaseModel):
    """One customer-side title this vertical's accounts staff. Spec
    section 4 names the per-vertical examples verbatim: "director of
    nursing, pharmacy director, CIO, community executive director,
    staffing coordinator, materials manager, telehealth program lead"."""

    model_config = ConfigDict(extra="forbid")

    title: str
    seniority: str = Field(description="'director' | 'vp' | 'c_suite' | 'manager' | 'staff'.")
    is_champion_candidate: bool = Field(
        default=False,
        description=(
            "Whether this role can be picked as the account's internal champion "
            "for a `champion_transition` arc (`ArcType.CHAMPION_TRANSITION`)."
        ),
    )


class EventTemplate(BaseModel):
    """One entry in this vertical's event library (spec section 4: "an
    event library"). Ties one `EventType` to the fact types it produces
    and the claim-template text used to render each fact's `claim`
    (`Fact.claim` is always code-written per spec section 1: "No fact is
    ever model-invented.")."""

    model_config = ConfigDict(extra="forbid")

    event_type: EventType
    weight: float = Field(gt=0, description="Relative draw weight among this vertical's events.")
    claim_templates: dict[FactType, list[str]] = Field(
        description=(
            "Fact types this event type can emit, each with one or more claim "
            "format strings. Every event of this type emits one `Fact` per key "
            "here. Template placeholders: {account}, {vendor}, {product}, "
            "{competitor}, {person}, {date}, {amount}, {days} -- whichever apply; "
            "`simulate_world()` supplies whichever of these the account/event "
            "context has, and a template must not reference one that never "
            "applies to its own fact type (e.g. a `usage` template referencing "
            "{competitor})."
        )
    )

    @field_validator("claim_templates")
    @classmethod
    def _at_least_one_type(cls, v: dict[FactType, list[str]]) -> dict[FactType, list[str]]:
        if not v:
            raise ValueError("EventTemplate.claim_templates must name at least one fact type")
        for fact_type, templates in v.items():
            if not templates:
                raise ValueError(f"EventTemplate: fact type {fact_type} has no claim templates")
        return v


class ArcStep(BaseModel):
    """One event in an `ArcTemplate`'s chain."""

    model_config = ConfigDict(extra="forbid")

    event_type: EventType
    day_offset: int = Field(description="Days after the arc's own start date this step fires.")


class ArcTemplate(BaseModel):
    """One entry in this vertical's arc library (spec section 1: "Each
    account draws one or two **arcs** from a per-vertical library"). Spec
    names the seven archetypes verbatim in `ArcType`; a pack can offer more
    than one concrete `ArcTemplate` per `arc_type` (e.g. two different
    `incident_escalation_exec` shapes), each independently weighted."""

    model_config = ConfigDict(extra="forbid")

    id: str
    arc_type: ArcType
    weight: float = Field(gt=0, description="Relative draw weight among this vertical's arcs.")
    steps: list[ArcStep] = Field(
        min_length=1, description="Ordered event chain, first step's offset is usually 0."
    )

    @field_validator("steps")
    @classmethod
    def _offsets_nondecreasing(cls, v: list[ArcStep]) -> list[ArcStep]:
        offsets = [s.day_offset for s in v]
        if offsets != sorted(offsets):
            raise ValueError("ArcTemplate.steps day_offsets must be non-decreasing (chronological)")
        return v


class TopicNode(BaseModel):
    """One node of this vertical's topic tree (spec section 4, "Topic
    tree"): "Each pack's tree is proposed by a model once per vertical,
    reviewed and committed ... Each leaf holds at most 5 documents per
    world." WP1 only needs the tree to validate and be walkable; WP3's doc
    planner is the first real consumer."""

    model_config = ConfigDict(extra="forbid")

    name: str
    children: list[TopicNode] = Field(default_factory=list)

    def leaves(self) -> list[str]:
        if not self.children:
            return [self.name]
        out: list[str] = []
        for child in self.children:
            out.extend(child.leaves())
        return out


class FactDirectiveBank(RootModel[dict[FactType, list[str]]]):
    """`{fact_type: [directive, ...]}`, every `FactType` present with at
    least `MIN_DIRECTIVES_PER_FACT_TYPE` entries (spec section 4,
    "Phrasing"). A directive is a short phrasing-style label the writer
    (WP4) is told to use when stating one scripted fact -- WP1 stores the
    bank; rendering happens later. Reusing the identical generic directive
    list across every fact type (via a YAML anchor) is expected and fine
    -- the spec's own directive list ("hedged, numbers spelled out,
    colloquial, ...") is not fact-type-specific. A bare `RootModel`, so a
    pack's YAML writes `fact_directives:` as a plain `{fact_type: [...]}`
    mapping with no extra wrapper key."""

    @model_validator(mode="after")
    def _covers_every_type_with_minimum(self) -> FactDirectiveBank:
        missing = set(FactType) - set(self.root.keys())
        if missing:
            raise ValueError(f"fact_directives missing type(s): {sorted(missing)}")
        short = {
            ft.value: len(directives)
            for ft, directives in self.root.items()
            if len(directives) < MIN_DIRECTIVES_PER_FACT_TYPE
        }
        if short:
            raise ValueError(
                f"fact_directives below the {MIN_DIRECTIVES_PER_FACT_TYPE}-entry minimum: {short}"
            )
        return self

    def __getitem__(self, fact_type: FactType) -> list[str]:
        return self.root[fact_type]


# What a decoy of a class must never say, as closed-vocabulary ids; the WP4
# writer prompt and check 5 (`verify.DECOY_FORBIDDEN_PATTERNS`) turn each id
# into an instruction and a regular expression.
DecoyForbiddenPattern = Literal["competitor_evaluation"]


class DecoyTextSource(BaseModel):
    """Provenance of one decoy class's `description` and `example` when they
    were regenerated after the pack's `claim_templates_source` run: the
    open-weight model and the raw responses they were curated from."""

    model_config = ConfigDict(extra="forbid")

    model: str
    generated: str
    raw_response_sha256: list[str] = Field(min_length=1)


class DecoyClass(BaseModel):
    """One near-fact statement class (spec section 4, "Hard negatives"):
    "a statement that looks like a fact and is not one" (`01-spec/world-
    generator-v2-2026-09-22.md` section 1's `Decoy` row)."""

    model_config = ConfigDict(extra="forbid")

    id: str
    near_type: FactType = Field(description="The real fact type this decoy is easy to mistake for.")
    description: str = Field(description="Plain-language description of the decoy pattern.")
    example: str = Field(description="One worked example in this vertical's own voice.")
    forbidden_patterns: list[DecoyForbiddenPattern] = Field(
        default_factory=list,
        description=(
            "What a decoy of this class must never say. `competitor_evaluation`: a "
            "competitor, another vendor, or which product anyone uses, evaluates, "
            "switches to or rules out -- a negated competitor statement is a "
            "`competitor_mention` fact (decisions log, 2026-09-23), not a decoy."
        ),
    )
    text_source: DecoyTextSource | None = None


class PersonaTemplate(BaseModel):
    """One entry in this vertical's persona bank (spec section 4,
    "Personas": "register x role x quirks"). Shape mirrors `v2.models.
    Persona` minus the fields that are per-person, not per-template
    (`typo_rate` is a range here, a fixed draw there)."""

    model_config = ConfigDict(extra="forbid")

    id: str
    speech_register: str = Field(
        description="e.g. 'clinical', 'operations', 'executive', 'casual'."
    )
    formality: str = Field(description="'formal' | 'neutral' | 'informal'.")
    typo_rate_range: tuple[float, float] = Field(
        default=(0.0, 0.0), description="Inclusive [min, max] typo rate this persona draws from."
    )
    chat_style: str = ""
    signature_style: str = ""
    hedging: bool = False
    second_language_features: list[str] = Field(default_factory=list)

    @field_validator("typo_rate_range")
    @classmethod
    def _range_ordered(cls, v: tuple[float, float]) -> tuple[float, float]:
        lo, hi = v
        if not (0.0 <= lo <= hi <= 1.0):
            raise ValueError(f"typo_rate_range {v} must satisfy 0 <= min <= max <= 1")
        return v


class NamePoolRefs(BaseModel):
    """How this vertical composes names out of the shared, split-
    partitioned pools in `v2.names` -- never literal names itself (see
    module docstring). Spec section 1's `Account` row lists `kind (system
    / facility / standalone)`; these suffix lists are what turns a drawn
    `ACCOUNT_STEMS` entry (e.g. "Fernhollow") into a full account name
    for each kind in this vertical ("Fernhollow Health System" vs.
    "Fernhollow Home Health" for a senior-living pack)."""

    model_config = ConfigDict(extra="forbid")

    account_suffixes_system: list[str] = Field(min_length=1)
    account_suffixes_facility: list[str] = Field(min_length=1)
    account_suffixes_standalone: list[str] = Field(min_length=1)
    vendor_domain_tld: str = Field(
        default=".com", description="Appended to the slugged vendor short name."
    )


class ClaimTemplatesSource(BaseModel):
    """Provenance for this pack's `claim_templates` (every `EventTemplate.
    claim_templates` string) and every `DecoyClass.example` (a class with its
    own `text_source` records that instead). Coordinator
    review of 7f779a3 (2026-09-22): both become text that can reach
    training data (`Fact.claim` labels, or verbatim decoy spans a document
    might quote), and Anthropic's terms bar training on Claude's own
    output -- so that phrasing must come from an open-weight teacher
    model, never be Claude-written. This field records which one, and
    when; the curation step (filtering, deduplication, rejecting bad
    candidates -- never editing a kept sentence beyond whitespace
    trimming) doesn't change the provenance, only which of the model's
    own sentences were kept. `raw_response_sha256` points at the exact,
    unmodified model responses this pack's text was curated from --
    `scripts/data/v2_gen_pack_templates.py` writes each one under
    `var/data/v2_packs/raw/<pack_id>/` (gitignored) and records its
    sha256 here, so the pack's provenance is checkable against the raw
    response even though the raw JSON itself isn't committed."""

    model_config = ConfigDict(extra="forbid")

    model: str = Field(description="OpenRouter model id, e.g. 'qwen/qwen3-32b'.")
    generated: str = Field(description="ISO 8601 date this pack's claim templates were generated.")
    raw_response_sha256: list[str] = Field(
        min_length=1,
        description=(
            "sha256 hex digest of every raw model response (`var/data/v2_packs/raw/"
            "<pack_id>/*.json`, gitignored) that contributed text to this pack -- "
            "the primary generation call plus any top-up/regeneration calls."
        ),
    )


class VerticalPack(BaseModel):
    """One vertical's complete content bank. `simulate_world(pack, seed,
    months, split)` reads only from a `VerticalPack` plus the shared name
    pools (`v2.names`) plus its own `seed`/`split` arguments -- nothing
    vertical-specific is hardcoded in the simulator itself."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(
        description="Vertical slug, e.g. 'medical_devices'. Matches the pack's filename."
    )
    name: str = Field(description="Display name, e.g. 'Medical devices'.")
    vendor_what: str = Field(
        description="One-line description of what this vertical's vendors sell, for `Vendor.overview` seeding."
    )
    products: list[ProductArchetype] = Field(min_length=1)
    roles: list[BuyerRole] = Field(
        min_length=3, description="At least 3 distinct customer-side buyer roles."
    )
    event_templates: list[EventTemplate] = Field(
        min_length=1,
        description=(
            "Should cover every `EventType` at least once across the list so "
            "every event kind is reachable; `simulate_world`'s top-up pass "
            "needs, for every `FactType`, at least one event_template whose "
            "claim_templates name it, or that type's 25-per-world floor "
            "(spec section 1) cannot be met."
        ),
    )
    arcs: list[ArcTemplate] = Field(
        min_length=1,
        description="Should cover every `ArcType` at least once (spec's seven archetypes).",
    )
    topic_tree: list[TopicNode] = Field(min_length=1)
    fact_directives: FactDirectiveBank
    persona_bank: list[PersonaTemplate] = Field(min_length=MIN_PERSONAS_PER_PACK)
    decoy_classes: list[DecoyClass] = Field(min_length=MIN_DECOY_CLASSES)
    name_pool_refs: NamePoolRefs
    excluded_product_kinds: list[str] = Field(
        default_factory=list,
        description=(
            "Product archetype `kind`s this vertical's packs must never generate. "
            "Spec section 4's senior-living guard: 'excludes resident fall "
            "detection and monitoring, a category deliberately out of scope.' "
            "Empty for verticals with no such guard."
        ),
    )
    held_out: bool = Field(
        default=False,
        description=(
            "True only for the eighth, held-out vertical (spec section 4: "
            "'one uses an eighth held-out vertical ... to measure how well "
            "the student generalizes past its training verticals'). A "
            "held-out pack is drawn only for test-split worlds -- WP11's "
            "round-1 world plan never picks it for a train or validation "
            "world. Defaults False so every WP1-era pack (`medical_devices."
            "yaml`) still validates unchanged."
        ),
    )
    claim_templates_source: ClaimTemplatesSource = Field(
        description=(
            "Provenance of every claim-template sentence and decoy example in "
            "this pack -- see `ClaimTemplatesSource`. Required: a pack with no "
            "recorded source cannot prove its training-facing text wasn't "
            "Claude-written."
        )
    )

    @model_validator(mode="after")
    def _every_event_type_covered(self) -> VerticalPack:
        covered = {t.event_type for t in self.event_templates}
        missing = set(EventType) - covered
        if missing:
            raise ValueError(
                f"pack {self.id!r}: event_templates missing EventType(s): {sorted(m.value for m in missing)}"
            )
        return self

    @model_validator(mode="after")
    def _every_arc_type_covered(self) -> VerticalPack:
        covered = {a.arc_type for a in self.arcs}
        missing = set(ArcType) - covered
        if missing:
            raise ValueError(
                f"pack {self.id!r}: arcs missing ArcType(s): {sorted(m.value for m in missing)}"
            )
        return self

    @model_validator(mode="after")
    def _every_fact_type_reachable(self) -> VerticalPack:
        """Spec section 1: "Per-type floors (at least 25 facts of each of
        the 12 types per world) are met by adding events." That only works
        if every fact type is emitted by at least one event template in
        this pack -- check it here, at pack-load time, rather than letting
        the simulator discover a shortfall it can never fill."""
        reachable: set[FactType] = set()
        for template in self.event_templates:
            reachable.update(template.claim_templates.keys())
        missing = set(FactType) - reachable
        if missing:
            raise ValueError(
                f"pack {self.id!r}: no event_template emits fact type(s) "
                f"{sorted(m.value for m in missing)} -- the 25-per-world floor cannot be met"
            )
        return self

    @model_validator(mode="after")
    def _excluded_kinds_actually_absent(self) -> VerticalPack:
        present = {p.kind for p in self.products}
        overlap = present & set(self.excluded_product_kinds)
        if overlap:
            raise ValueError(
                f"pack {self.id!r}: excluded_product_kinds present in products: {sorted(overlap)}"
            )
        return self

    @model_validator(mode="after")
    def _claim_templates_meet_variant_floor(self) -> VerticalPack:
        """Spec section 4, "Phrasing": "Each fact type has 15 or more event
        variants per vertical." Sums (deduplicated, case/whitespace-
        insensitively) every `claim_templates[fact_type]` list across every
        `EventTemplate` that names that fact type -- coordinator review of
        7f779a3 found every pack (including `medical_devices.yaml`) at 1-5
        variants per type, one static list per event/fact-type pair.

        Declared LAST (after `_every_event_type_covered`/`_every_fact_type_
        reachable`): pydantic v2 runs `mode="after"` validators in
        declaration order and stops at the first one that raises, so a
        pack fixture missing whole event types (WP1's own `test_pack_
        rejects_event_template_not_covering_every_event_type`/`test_pack_
        rejects_unreachable_fact_type`) still surfaces THOSE checks' own,
        more specific error messages instead of this one masking them."""
        seen: dict[FactType, set[str]] = {ft: set() for ft in FactType}
        for template in self.event_templates:
            for fact_type, claims in template.claim_templates.items():
                for claim in claims:
                    seen[fact_type].add(re.sub(r"\s+", " ", claim.strip().lower()))
        short = {
            ft.value: len(variants)
            for ft, variants in seen.items()
            if len(variants) < MIN_CLAIM_VARIANTS_PER_FACT_TYPE
        }
        if short:
            raise ValueError(
                f"pack {self.id!r}: fact types below the "
                f"{MIN_CLAIM_VARIANTS_PER_FACT_TYPE}-variant floor: {short}"
            )
        return self


def load_pack(path: str | Path) -> VerticalPack:
    """Load and validate one `packs/<vertical>.yaml` file."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return VerticalPack.model_validate(data)
