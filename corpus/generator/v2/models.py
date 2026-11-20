"""The v2 world data model (pydantic v2).

`01-spec/world-generator-v2-2026-09-22.md` section 1, "World simulation":
a world is one vendor, one vertical and one dated window, built
deterministically from a seed. This module is the typed shape of that
world -- every entity in the spec's section-1 table (`Vendor`, `Product`,
`Competitor`, `Account`, `Person`, `Event`, `Fact`, plus `Arc` and
`StateSnapshot`, which the section-1 prose describes but doesn't table).

**The one hard invariant** (this work package's own scope line: "Every
future fact must trace to an event id; make that a model invariant"):
every `Fact.event_id` must name a real `Event.id` in the same `World`.
`Fact` alone can't check this -- a fact doesn't carry its own world's
event list -- so the check lives on `World`, as a `model_validator(mode=
"after")` that runs every time a `World` is constructed or re-validated.
Constructing a `World` with an orphan fact raises `ValidationError`
immediately; there is no code path that produces a `World` with an
unlabeled-provenance fact.

`Fact.fact_type` reuses `portolan.extract.schema.FactType` directly (the
12 production fact types) rather than redefining the list here -- CLAUDE.
md's "reuse" instruction, and the whole point of the compat export in
`World.to_truth_table()`.
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from portolan.extract.schema import FactType

# --------------------------------------------------------------------------
# Closed lists
# --------------------------------------------------------------------------


class Split(StrEnum):
    """Which name pool (`v2.names.pool_for_split`) and world-count bucket
    (spec section 6's table) a world belongs to."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class AccountKind(StrEnum):
    SYSTEM = "system"
    FACILITY = "facility"
    STANDALONE = "standalone"


class AccountTier(StrEnum):
    """Spec section 1, `Account` row: "tier (enterprise / mid_market /
    smb)"."""

    ENTERPRISE = "enterprise"
    MID_MARKET = "mid_market"
    SMB = "smb"


class AccountStatus(StrEnum):
    ACTIVE = "active"
    CHURNED = "churned"
    EXPANDED = "expanded"


class PersonSide(StrEnum):
    VENDOR = "vendor"
    CUSTOMER = "customer"


class EventType(StrEnum):
    """Spec section 1, `Event` row: the closed `type` list, verbatim."""

    GO_LIVE = "go_live"
    RELEASE = "release"
    INCIDENT = "incident"
    ESCALATION = "escalation"
    QBR = "qbr"
    RENEWAL_STEP = "renewal_step"
    PRICE_NEGOTIATION = "price_negotiation"
    COMPETITIVE_EVAL = "competitive_eval"
    CHAMPION_DEPARTURE = "champion_departure"
    EXPANSION = "expansion"
    COMMITMENT_MADE = "commitment_made"
    COMMITMENT_DUE = "commitment_due"
    DECISION = "decision"
    POLICY_CHANGE = "policy_change"


class ArcType(StrEnum):
    """Spec section 1, "Timeline" paragraph: the seven per-account arc
    archetypes, verbatim ("or quiet and steady" -> `QUIET_STEADY`)."""

    ROCKY_GO_LIVE = "rocky_go_live"
    RENEWAL_PRICE_PUSHBACK = "renewal_price_pushback"
    COMPETITIVE_EVAL_SAVE_OR_CHURN = "competitive_eval_save_or_churn"
    CHAMPION_TRANSITION = "champion_transition"
    EXPANSION = "expansion"
    INCIDENT_ESCALATION_EXEC = "incident_escalation_exec"
    QUIET_STEADY = "quiet_steady"


class ReferenceMode(StrEnum):
    """Spec section 1, `Fact` row: "reference_mode (explicit / alias /
    first_name / title / handle / pronoun)", verbatim."""

    EXPLICIT = "explicit"
    ALIAS = "alias"
    FIRST_NAME = "first_name"
    TITLE = "title"
    HANDLE = "handle"
    PRONOUN = "pronoun"


class RenewalStatus(StrEnum):
    """Used by `StateSnapshot.AccountState.renewal_status` -- not in the
    spec's entity table verbatim (the table just says "renewal status"),
    closed here so snapshots are comparable across dates and diffable by
    downstream work packages (WP3's doc planner: "sees only the state at
    D")."""

    NOT_DUE = "not_due"
    IN_PROGRESS = "in_progress"
    AT_RISK = "at_risk"
    RENEWED = "renewed"
    CHURNED = "churned"


# --------------------------------------------------------------------------
# Vendor side
# --------------------------------------------------------------------------


class Release(BaseModel):
    """One entry in `Product.releases`."""

    model_config = ConfigDict(extra="forbid")

    version: str
    date: date
    changes: list[str] = Field(default_factory=list)


class Product(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    aliases: list[str] = Field(default_factory=list)
    kind: str
    releases: list[Release] = Field(default_factory=list)
    known_issues: list[str] = Field(default_factory=list)


class Competitor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    aliases: list[str] = Field(default_factory=list)
    positioning: str = ""


class Vendor(BaseModel):
    """Spec section 1, `Vendor` row. `overview` and `initiatives` are the
    only model-written surface color this work package's data model
    carries a slot for -- WP1 itself never calls a model; a later work
    package (the writer, WP4) fills them in, or a caller leaves them
    blank/templated."""

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    vertical: str
    domain: str
    message_id_domain: str
    overview: str = ""
    initiatives: list[str] = Field(default_factory=list)
    fiscal_year_start_month: int = Field(default=1, ge=1, le=12)
    products: list[Product] = Field(default_factory=list)
    competitors: list[Competitor] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Customer side
# --------------------------------------------------------------------------


class HealthPoint(BaseModel):
    """One entry in `Account.health` -- spec: `health[{date, score}]`."""

    model_config = ConfigDict(extra="forbid")

    date: date
    score: int = Field(ge=0, le=100)


class Account(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    crm_id: str
    name: str
    kind: AccountKind
    parent: str | None = None
    tier: AccountTier
    arr: float = Field(ge=0)
    contract_start: date
    renewal_date: date
    region: str
    # Email domain shared by an account and its system's facilities, as a
    # real health system's staff share one domain (the Halvard exam's
    # Larkspur facilities all use larkspurhealth.org). Added in the WP3
    # review: contact emails used to be `<slug of the full account name>.
    # org`, which carried "general-hospital-1" into every address.
    domain: str = ""
    aliases: list[str] = Field(default_factory=list)
    health: list[HealthPoint] = Field(default_factory=list)
    status: AccountStatus = AccountStatus.ACTIVE


class Employment(BaseModel):
    """Spec: `Person` row, "tenure {start, end}". `end is None` means
    still employed as of the world's `window_end`."""

    model_config = ConfigDict(extra="forbid")

    start: date
    end: date | None = None

    @model_validator(mode="after")
    def _end_after_start(self) -> Employment:
        if self.end is not None and self.end < self.start:
            raise ValueError(f"Employment.end {self.end} precedes start {self.start}")
        return self


class Persona(BaseModel):
    """Spec: `Person` row, "persona {register, formality, typo_rate,
    chat_style, signature, hedging, second-language features}",
    verbatim field list."""

    model_config = ConfigDict(extra="forbid")

    # Named `speech_register`, not `register`: pydantic's `BaseModel`
    # already defines a `register` classmethod (ABC virtual-subclass
    # registration), and a same-named field shadows it with a UserWarning.
    speech_register: str
    formality: str
    typo_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    chat_style: str = ""
    signature: str = ""
    hedging: bool = False
    second_language_features: list[str] = Field(default_factory=list)


class Person(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    side: PersonSide
    account_id: str | None = None
    role: str
    title: str
    email: str
    chat_handle: str
    tenure: Employment
    replaced_by: str | None = None
    persona: Persona

    @model_validator(mode="after")
    def _customer_side_has_account(self) -> Person:
        if self.side == PersonSide.CUSTOMER and not self.account_id:
            raise ValueError("customer-side Person requires account_id")
        if self.side == PersonSide.VENDOR and self.account_id is not None:
            raise ValueError("vendor-side Person must not carry a customer account_id")
        return self


# --------------------------------------------------------------------------
# Timeline
# --------------------------------------------------------------------------


class StateChange(BaseModel):
    """One entry in `Event.state_changes` -- spec: `state_changes[]`
    `(field, old, new)`."""

    model_config = ConfigDict(extra="forbid")

    field: str
    old: Any = None
    new: Any = None


class Event(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    date: date
    type: EventType
    account_id: str
    product_ids: list[str] = Field(default_factory=list)
    people_ids: list[str] = Field(default_factory=list)
    causes: list[str] = Field(default_factory=list)
    state_changes: list[StateChange] = Field(default_factory=list)


class Fact(BaseModel):
    """Spec section 1, `Fact` row. `event_id` is required and non-blank
    here (the local half of the invariant); the referential half --
    `event_id` must name a real event in the same world -- is checked by
    `World._facts_trace_to_events` below, since a bare `Fact` has no view
    of its world's event list."""

    model_config = ConfigDict(extra="forbid")

    id: str
    fact_type: FactType
    claim: str
    entities: list[str] = Field(default_factory=list)
    speaker: str
    date: date
    account_id: str
    event_id: str = Field(min_length=1)
    valid_from: date | None = None
    valid_to: date | None = None
    supersedes: str | None = None
    anchors: list[str] = Field(default_factory=list)
    directive: str | None = None
    reference_mode: ReferenceMode = ReferenceMode.EXPLICIT


class Arc(BaseModel):
    """Spec section 1, "Timeline" paragraph: "Each account draws one or
    two arcs from a per-vertical library ... Arcs emit events." `event_ids`
    is the ordered chain of events this arc instantiated -- each later
    event's `Event.causes` also names the event immediately before it in
    this list, so the chain is recoverable from either side."""

    model_config = ConfigDict(extra="forbid")

    id: str
    arc_type: ArcType
    account_id: str
    event_ids: list[str] = Field(default_factory=list)


class AccountState(BaseModel):
    """One account's slice of a `StateSnapshot` -- spec: "who holds each
    role, current ARR, open incidents, renewal status"."""

    model_config = ConfigDict(extra="forbid")

    account_id: str
    role_holders: dict[str, str] = Field(default_factory=dict)  # role -> person_id
    arr: float = Field(ge=0)
    open_incident_event_ids: list[str] = Field(default_factory=list)
    renewal_status: RenewalStatus = RenewalStatus.NOT_DUE


class StateSnapshot(BaseModel):
    """Spec: "The world keeps a state snapshot per date ... A document
    dated D sees only the state at D, so facts agree across sources and
    later facts can supersede earlier ones." WP3's doc planner is the
    first consumer; WP1 only has to produce these correctly."""

    model_config = ConfigDict(extra="forbid")

    date: date
    account_states: list[AccountState] = Field(default_factory=list)


# --------------------------------------------------------------------------
# World
# --------------------------------------------------------------------------


class World(BaseModel):
    """One simulated vendor, one vertical, one dated window -- the output
    of `v2.simulator.simulate_world`."""

    model_config = ConfigDict(extra="forbid")

    id: str
    seed: int
    vertical: str
    split: Split
    window_start: date
    window_end: date
    vendor: Vendor
    accounts: list[Account]
    people: list[Person]
    events: list[Event]
    facts: list[Fact]
    arcs: list[Arc]
    snapshots: list[StateSnapshot] = Field(default_factory=list)

    @model_validator(mode="after")
    def _facts_trace_to_events(self) -> World:
        """The invariant this work package exists to enforce (spec
        item 1: "Every future fact must trace to an event id; make that a
        model invariant"). Every `Fact.event_id` must name a real
        `Event.id` in this same world -- an orphan fact means some code
        path invented a fact without going through the timeline, exactly
        the v1 defect (`01-spec/world-generator-v2-2026-09-22.md` section
        0: "no timeline, no cause and effect ... Nothing checks whether
        the writer stated facts it was not asked to")."""
        event_ids = {e.id for e in self.events}
        orphans = [f.id for f in self.facts if f.event_id not in event_ids]
        if orphans:
            raise ValueError(
                f"World {self.id!r}: {len(orphans)} fact(s) with no matching "
                f"event id (invariant violated): {orphans[:10]}"
                + (" ..." if len(orphans) > 10 else "")
            )
        return self

    @model_validator(mode="after")
    def _window_is_ordered(self) -> World:
        if self.window_end <= self.window_start:
            raise ValueError(
                f"World {self.id!r}: window_end {self.window_end} must be after "
                f"window_start {self.window_start}"
            )
        return self

    # -- derived views ------------------------------------------------

    def facts_by_type(self) -> dict[FactType, list[Fact]]:
        out: dict[FactType, list[Fact]] = {ft: [] for ft in FactType}
        for fact in self.facts:
            out[fact.fact_type].append(fact)
        return out

    def account_by_id(self) -> dict[str, Account]:
        return {a.id: a for a in self.accounts}

    def person_by_id(self) -> dict[str, Person]:
        return {p.id: p for p in self.people}

    # -- compat export --------------------------------------------------

    def to_truth_table(self) -> dict[str, Any]:
        """`portolan.bench.truth.TruthTable`-shaped dict (spec section 1,
        "Output per world": "`truth-table.yaml` (a compatibility export in
        today's shape so `portolan/bench/truth.py` scores it unchanged)").

        WP1 has no doc planner or writer yet (those are WP3/WP4), so there
        are no real source documents to place facts in -- this export
        carries `meta`/`accounts`/`contacts`/`facts` (everything
        `TruthTable` itself validates field-by-field) plus a placeholder
        one-entry `sources` list per fact, pointing at the fact's origin
        event rather than a document. `Truth.load()` (which additionally
        wants a `manifest.json` of document placements) is a WP9 concern;
        `TruthTable.model_validate(world.to_truth_table())` alone is what
        this work package's round-trip test exercises.
        """
        accounts_by_id = self.account_by_id()
        customer_people = [p for p in self.people if p.side == PersonSide.CUSTOMER]
        vendor_people = [p for p in self.people if p.side == PersonSide.VENDOR]

        return {
            "meta": {
                "version": "synth-v2-0.1",
                "schema_version": "0.1",
                "seed": self.seed,
                "vertical": self.vertical,
                "split": self.split.value,
                "window": {
                    "start": self.window_start.isoformat(),
                    "end": self.window_end.isoformat(),
                },
                "supplier": {
                    "name": self.vendor.name,
                    "domain": self.vendor.domain,
                    "message_id_domain": self.vendor.message_id_domain,
                },
                "products": [
                    {"id": p.id, "name": p.name, "aliases": p.aliases} for p in self.vendor.products
                ],
                "competitors": [
                    {"id": c.id, "name": c.name, "aliases": c.aliases}
                    for c in self.vendor.competitors
                ],
                "internal_people": [
                    {
                        "id": p.id,
                        "name": p.name,
                        "role": p.role,
                        "title": p.title,
                        "email": p.email,
                    }
                    for p in vendor_people
                ],
            },
            "accounts": [
                {
                    "id": a.id,
                    "crm_id": a.crm_id,
                    "name": a.name,
                    "kind": a.kind.value,
                    "parent": a.parent,
                    "domain": a.domain or self.vendor.domain,
                    "arr": a.arr,
                    "region": a.region,
                    "aliases": a.aliases,
                    "status": a.status.value,
                }
                for a in self.accounts
            ],
            "contacts": [
                {
                    "id": p.id,
                    "crm_id": p.id.upper(),
                    "name": p.name,
                    "account": p.account_id,
                    "title": p.title,
                    "email": p.email,
                    "status": "active" if p.tenure.end is None else "former",
                }
                for p in customer_people
            ],
            "facts": [
                {
                    "id": f.id,
                    "type": f.fact_type.value,
                    "claim": f.claim,
                    "entities": [
                        e if e not in accounts_by_id else accounts_by_id[e].id for e in f.entities
                    ],
                    "date": f.date.isoformat(),
                    "valid_from": f.valid_from.isoformat() if f.valid_from else None,
                    "valid_to": f.valid_to.isoformat() if f.valid_to else None,
                    "supersedes": f.supersedes,
                    "sources": [
                        {
                            "id": f"E-{f.event_id}",
                            "type": "event",
                            "speaker": f.speaker,
                            "internal_only": False,
                        }
                    ],
                    "phrasing": f.directive,
                }
                for f in self.facts
            ],
            "sources": {},
            "expectations": {},
        }
