"""Pydantic v2 models mirroring `01-spec/extraction-schema-v0.md` exactly.

This module is the contract. Everything else in `portolan.extract` (prompt
rendering, segmentation/merge, the backends) imports from here rather than
redefining a type list, a fact rule, or a field constraint — the build
instructions call this out explicitly for the signal-type checklist, and
the same "one source of truth" logic applies to the fact rules and the
entity-kind list.

Two closed lists live here as data, not scattered string literals:
`FACT_TYPES` (definition + example + counter-example per type, used both
for validation and to render the extractor's prompt) and `FACT_RULES` (the
numbered rules in the schema doc, rendered verbatim into the prompt).
"""

from __future__ import annotations

import re
from datetime import date
from enum import StrEnum
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

SCHEMA_VERSION = "0.1"


class SourceType(StrEnum):
    """Closed list, extraction-schema-v0.md "What the extractor receives"."""

    CALL = "call"
    EMAIL = "email"
    TICKET = "ticket"
    DOC = "doc"
    CRM = "crm"
    CHAT = "chat"
    SURVEY = "survey"
    RSS = "rss"


class EntityKind(StrEnum):
    """Closed list, extraction-schema-v0.md "Entity kinds (closed)"."""

    ACCOUNT = "account"
    CONTACT = "contact"
    PRODUCT = "product"
    COMPETITOR = "competitor"
    INTERNAL_PERSON = "internal_person"
    OTHER = "other"


class FactType(StrEnum):
    """Closed list, extraction-schema-v0.md "Signal types, base set"."""

    COMPLAINT = "complaint"
    REQUEST = "request"
    PRAISE = "praise"
    RISK = "risk"
    COMPETITOR_MENTION = "competitor_mention"
    COMMITMENT = "commitment"
    DECISION = "decision"
    CONTACT_CHANGE = "contact_change"
    PRICING = "pricing"
    USAGE = "usage"
    QUESTION = "question"
    OTHER = "other"


class FactTypeInfo(BaseModel):
    """One row of the signal-type checklist: definition, a real example, and
    a counter-example (what it is NOT). The schema doc says this checklist
    "is what makes the model look for each kind explicitly" — the
    counter-examples are not in the doc's table, so they're written here,
    each one a claim that could plausibly be mistaken for the type above it
    if the model were pattern-matching on surface features alone.
    """

    model_config = ConfigDict(extra="forbid")

    definition: str
    example: str
    counter_example: str
    counter_reason: str


# The base signal-type table. Definitions are copied verbatim from
# extraction-schema-v0.md. Examples name a fictional vendor's accounts that
# appear in no exam corpus (`tests/test_extract_prompt.py` pins that).
# counter_example/counter_reason satisfy rule 8's checklist requirement:
# "definition, example, and one counter-example (what it is not)".
FACT_TYPES: dict[FactType, FactTypeInfo] = {
    FactType.COMPLAINT: FactTypeInfo(
        definition="The customer reports something went wrong",
        example="Brackenholt received two late cartridge shipments in May 2026",
        counter_example="Brackenholt asked for a quiet-hours alarm setting on the R4 skid",
        counter_reason=(
            "This is a request for something new, not a report that something already went wrong."
        ),
    ),
    FactType.REQUEST: FactTypeInfo(
        definition="The customer asks for something that does not exist yet",
        example="Brackenholt asked for a quiet-hours alarm setting on the R4 skid",
        counter_example="Brackenholt received two late cartridge shipments in May 2026",
        counter_reason="This reports a past failure, not a request for a new capability.",
    ),
    FactType.PRAISE: FactTypeInfo(
        definition="The customer says something works well",
        example="Brackenholt's quality lead called the onboarding \"the smoothest we've had\"",
        counter_example='Brackenholt\'s CFO said the renewal terms were "acceptable"',
        counter_reason=(
            "Lukewarm acceptance is not a statement that something works well; "
            "do not file faint or neutral language as praise."
        ),
    ),
    FactType.RISK: FactTypeInfo(
        definition="Evidence the relationship or revenue is threatened",
        example="Brackenholt's CFO froze new equipment purchases until Q2 2028",
        counter_example=(
            "Brackenholt's quality lead called the onboarding \"the smoothest we've had\""
        ),
        counter_reason=(
            "This is positive sentiment with no threat to revenue or the "
            "relationship; it is praise, not risk."
        ),
    ),
    FactType.COMPETITOR_MENTION: FactTypeInfo(
        definition="A named competitor comes up",
        example="Brackenholt is evaluating an Ondrell skid for the Harrowfield canning line",
        counter_example="Brackenholt asked whether the skid supports Tarnwick MES integration",
        counter_reason=(
            "No competitor is named; Tarnwick MES is an integration target, not a competing vendor."
        ),
    ),
    FactType.COMMITMENT: FactTypeInfo(
        definition="Either side promises something",
        example="Our rep promised a written root-cause report by 2026-05-19",
        counter_example="Brackenholt chose to renew the service contract for two years",
        counter_reason="A decision already made is not a promise of future action.",
    ),
    FactType.DECISION: FactTypeInfo(
        definition="Something was decided",
        example="Brackenholt chose to renew the service contract for two years",
        counter_example="Our rep promised a written root-cause report by 2026-05-19",
        counter_reason=(
            "A promise about the future is a commitment, not a decision that "
            "has already been reached."
        ),
    ),
    FactType.CONTACT_CHANGE: FactTypeInfo(
        definition="A person's role, presence or contact info changed",
        example="Wendeline Sorvald moved from procurement to plant operations in 2026-04",
        counter_example="Ms. Sandqvist said the canning line wants Tarnwick MES integration",
        counter_reason=(
            "Ms. Sandqvist's role and presence are unchanged here; this is a "
            "request from a still-current contact."
        ),
    ),
    FactType.PRICING: FactTypeInfo(
        definition="Price, discount, budget or invoice discussion",
        example="Brackenholt expects an 8 percent volume discount on the 2028 renewal",
        counter_example="Harrowfield runs nine R4 skids across two canning lines",
        counter_reason="A count of deployed units is a usage fact; no money is discussed.",
    ),
    FactType.USAGE: FactTypeInfo(
        definition="How the product is used, how much, where",
        example="Harrowfield runs nine R4 skids across two canning lines",
        counter_example="Brackenholt expects an 8 percent volume discount on the 2028 renewal",
        counter_reason=(
            "This is a discount expectation, a pricing fact, not a statement "
            "of how the product is used."
        ),
    ),
    FactType.QUESTION: FactTypeInfo(
        definition="The customer asked something and did not get an answer",
        example="Brackenholt asked whether the skid supports Tarnwick MES integration; unanswered",
        counter_example=(
            "Brackenholt asked whether the skid supports Tarnwick MES integration; "
            "our rep answered yes"
        ),
        counter_reason=(
            "The question was answered in the same exchange, so the "
            "open-question type does not apply."
        ),
    ),
    FactType.OTHER: FactTypeInfo(
        definition="Durable fact that fits none of the above",
        example="counted; drives schema growth",
        counter_example=(
            "Brackenholt Harrowfield Brewery received two late cartridge shipments in May 2026"
        ),
        counter_reason=(
            "This is a clean complaint; do not default to `other` when a real "
            "type fits — `other` is for facts that genuinely fit nothing above."
        ),
    ),
}

# Types where rule 5 requires a verbatim quote, forty words or fewer:
# "These are the types where the exact words carry the evidence."
QUOTE_REQUIRED_TYPES: frozenset[FactType] = frozenset(
    {
        FactType.COMPLAINT,
        FactType.PRAISE,
        FactType.COMPETITOR_MENTION,
        FactType.COMMITMENT,
        FactType.PRICING,
    }
)

# The nine numbered "Fact rules" from extraction-schema-v0.md, verbatim
# (rule 7's "recorded time" clause and rule 9's role list are kept intact).
# Rendered into the system prompt by extract/prompt.py; enforced here where
# the rule is mechanically checkable (locator, quote, sentiment range,
# confidence range, closed type list — rules 4, 5, and part of 6).
FACT_RULES: list[str] = [
    'One claim per fact. No "and." Split compound statements.',
    (
        "Durable language. Absolute dates resolved from `source_date`. No "
        '"yesterday," "next quarter," "recently."'
    ),
    "No pronouns in claims. Name the entity every time.",
    (
        "Locator is required. A fact without one is rejected at merge and "
        "counted in the coverage report."
    ),
    (
        "Quote is required for `complaint`, `praise`, `competitor_mention`, "
        "`commitment`, `pricing`. Verbatim, forty words or fewer. These are "
        "the types where the exact words carry the evidence."
    ),
    (
        "Entities resolve to registry ids when confident. Otherwise the "
        "mention goes in `unknown_entities` for the review queue. The "
        "extractor never invents an id."
    ),
    (
        "`valid_from` / `valid_to` are the world-time window when known (a "
        "contract term, a role change). Null is fine. Recorded time comes "
        "from the source note, not the extractor."
    ),
    (
        "`type: other` is allowed and counted. If more than one fact in five "
        "is `other` for a customer, the customer extension needs a new type."
    ),
    "Speaker is a registry contact id, or a role (`customer`, `our-rep`, `unknown`).",
]

# --------------------------------------------------------------------------
# Prompt A/B variants (extraction accuracy campaign, 2026-09-18,
# `01-spec/extraction-accuracy-campaign-2026-09-18.md` wave 1B,
# `01-spec/prompt-ab-2026-09-18.md`). Additive only: the base FACT_TYPES /
# FACT_RULES / QUOTE_REQUIRED_TYPES a live prompt renders from today are
# never mutated, so `portolan.extract.prompt.build_system_prompt(variant=
# "baseline")` (the default, and the only variant every caller before this
# campaign ever saw) renders byte-for-byte what it always has --
# `tests/test_extract_prompt.py::test_baseline_variant_is_byte_identical_
# to_pre_campaign_default` pins that. `portolan.extract.prompt` is the
# only importer of the type/example content below; the rest of a
# variant's prompt text (process instructions, source-type reading
# guidance -- prose, not a type definition) lives in `prompt.py` itself,
# next to `_DURABILITY_BAR`/`_LOCATOR_INSTRUCTIONS`, which were already
# plain string constants there before this campaign.
# --------------------------------------------------------------------------

# v1-other-example: replaces FACT_TYPES[FactType.OTHER] in the rendered
# checklist. The placeholder original (`example="counted; drives schema
# growth"`) is the single largest documented miss cluster in
# `01-spec/extraction-miss-analysis-2026-09-18.md` (recommendation #1):
# Kimi K2 produced zero facts, in any of 3 runs, from all four bare CRM
# account rows carrying a planted `other` fact (F35, F36, F68, F69). The
# row below has the field layout `portolan.bench.corpus_loader.
# _prepare_crm` renders; its account, ids and figures are fictional.
FACT_TYPE_OTHER_V1 = FactTypeInfo(
    definition=(
        "Durable fact that fits none of the above -- including a structured "
        "record's own field values, not just something said aloud"
    ),
    example=(
        'A CRM account row "[account:A-7730] crm_id=A-7730, name=Brackenholt '
        "Beverage Group, kind=system, parent_crm_id=, domain=brackenholt.com, "
        'arr=1875000, region=High Plains, status=" is itself one or more '
        'facts, even though it has no sentence to quote: "Brackenholt '
        "Beverage Group's annual recurring revenue with Veltrona is "
        '1,875,000 dollars as of 2026-05-01" (type `other`, locator '
        "`account:A-7730`, no quote -- there is no prose here, the row's "
        "own field is the evidence). A "
        "populated `parent_crm_id` field is likewise an ownership fact "
        '("X is owned by Y", also `other`). Do not skip a source just '
        "because it has no sentences -- every populated field on a "
        "structured record is a candidate fact. (An `opportunity` row's "
        "`amount`/`close_date` is a live negotiation, not a stored record "
        "value -- that is `pricing`, not `other`; see the source-type "
        "reading guidance.)"
    ),
    counter_example=(
        "Brackenholt Harrowfield Brewery received two late cartridge shipments in May 2026"
    ),
    counter_reason=(
        "This is a clean complaint; do not default to `other` when a real "
        "type fits -- `other` is for facts that genuinely fit nothing "
        "above, including structured-record fields with no natural-"
        "language type of their own."
    ),
)


class TypePrecedenceRule(BaseModel):
    """One entry in `TYPE_PRECEDENCE_RULES_V6`: two types the extractor
    keeps confusing, which one wins and why, and one worked example per
    side. `positive_example` is a claim correctly typed `winning_type`;
    `negative_example` is a claim correctly typed `losing_type` -- i.e.
    the model should NOT apply `winning_type` to it, even though it looks
    similar on the surface. Examples name the same fictional vendor as the
    rest of the prompt, so a rule teaches the distinction, never the answer
    to an exam fact."""

    model_config = ConfigDict(extra="forbid")

    winning_type: FactType
    losing_type: FactType
    rule: str
    positive_example: str
    negative_example: str


# v6-type-precision (facttype-4a, 2026-09-18, `01-spec/facttype-4a-2026-09-
# 18.md`): four confused-pair rules, one per pair the confusion-matrix
# diagnosis found pooling `matched` facts' `fact_type` (planted) against
# `produced_type` across Sonnet 5, Haiku 4.5, Kimi K2, Kimi K2.5 and Nova
# Lite's `var/bench/2026-09-18-final/` runs on the v4-two-step prompt --
# 73 wrong-typed matches total, 71 of them (97%) inside these four pairs:
# decision/other (40, every one planted `decision` produced `other` -- a
# CRM-recorded or call-restated contract term/renewal/stage the model read
# as a raw stored value instead of the choice it represents), commitment/
# decision (15, every one planted `commitment` produced `decision` --
# already a near-match pair in `_NEAR_MATCH_TYPE_PAIRS`, half-credited
# there but still worth a precision rule since it is the single largest
# EXACT-accuracy miss after decision/other), request/question (10, both
# directions -- an ask phrased as a question that is really a capability
# request, or vice versa), other/risk (6, every one planted `other`
# produced `risk` -- an internal-only operations note using risk-adjacent
# words with no customer or account actually threatened). A fifth pair,
# praise/other, appeared twice pooled -- not given a rule here; two
# instances against a single run per model on this corpus is not enough
# to separate a real pattern from noise, and the doc says so rather than
# chasing it. commitment/decision and, in part, decision/other are also
# flagged in the doc as partly genuine truth-table ambiguity (the same
# sentence can honestly read as either type) -- the rule below narrows
# that ambiguity with a concrete tie-breaker, it does not claim to
# eliminate it.
TYPE_PRECEDENCE_RULES_V6: list[TypePrecedenceRule] = [
    TypePrecedenceRule(
        winning_type=FactType.DECISION,
        losing_type=FactType.OTHER,
        rule=(
            "a contract term, renewal date, or CRM opportunity stage is "
            "`decision` when it changed because a party CHOSE it -- even "
            "when you read it off a CRM row, or a call just restates the "
            "same fact -- because the choice is the fact, not the record "
            "of it. `other` is for a raw stored attribute nobody actively "
            "chose: an ARR figure, a region, a product SKU, a status "
            "field with no decision behind it."
        ),
        positive_example=(
            "Glenmoss Dairy's CRM opportunity O-6273 shows the renewal "
            "term changed from twenty-four months to twelve months after "
            "Glenmoss's finance lead pushed back on the multi-year price "
            "-- `decision`, because a party's choice changed the term."
        ),
        negative_example=(
            "Glenmoss Dairy's CRM account row lists arr=960,000 and "
            "region=Intermountain -- `other`, a raw stored attribute "
            "nobody chose."
        ),
    ),
    TypePrecedenceRule(
        winning_type=FactType.COMMITMENT,
        losing_type=FactType.DECISION,
        rule=(
            "when a party promises to DO something that has not happened "
            "yet -- a pilot start date, a delivery, a fix, a trial -- the "
            "emphasis is on the pending future action, so write "
            "`commitment`. `decision` is for a choice already settled "
            "with nothing further pending (ending an agreement, picking a "
            "renewal length) -- the emphasis is on what was chosen, not "
            "on an action still to come."
        ),
        positive_example=(
            "Vantrell Brewing Cooperative agreed to trial two Rillfold R5 "
            "skids in its canning hall starting 2026-10-05 -- "
            "`commitment`: the trial itself is a pending future action."
        ),
        negative_example=(
            "Vantrell Brewing Cooperative chose to end its Rillfold R4 "
            "rental agreement effective 2026-10-05 -- `decision`: the "
            "choice is settled, nothing further is pending."
        ),
    ),
    TypePrecedenceRule(
        winning_type=FactType.REQUEST,
        losing_type=FactType.QUESTION,
        rule=(
            "an ask for a capability, feature, or change that does not "
            "exist yet is `request`, even when it is phrased as a "
            'question ("could X do Y?", "does the product support Z?"). '
            "`question` is for an ask about STATUS or a fact that may "
            "already be true or false, left unanswered -- nothing new is "
            "being asked for."
        ),
        positive_example=(
            "Duncarron Distillery asked whether Rillfold Bridge could "
            "sync with its Pellworth SCADA instance -- `request`: a "
            "capability that doesn't exist yet, even phrased as "
            '"whether".'
        ),
        negative_example=(
            "Duncarron Distillery asked whether the October pricing "
            "change also applied to its own site, and the thread ended "
            "with no reply -- `question`: a status ask, left unanswered, "
            "asking for nothing new."
        ),
    ),
    TypePrecedenceRule(
        winning_type=FactType.RISK,
        losing_type=FactType.OTHER,
        rule=(
            "`risk` requires a customer or account facing the threat -- "
            "evidence the relationship or revenue itself is in danger. "
            "An internal operations note (staffing, a warehouse, a "
            "carrier change) with no customer or account named stays "
            '`other`, even when the text itself uses a word like "risk" '
            'or "delay".'
        ),
        positive_example=(
            "Glenmoss Dairy's supply lead said a continued packing delay "
            "would push them to dual-source cartridges from a competitor "
            "-- `risk`: a named customer ties the delay to a threat "
            "against the relationship."
        ),
        negative_example=(
            "Veltrona's own distribution manager noted the Yakima "
            "warehouse was short one packer from 2026-04-06 through "
            "2026-04-24 -- `other`: an internal-only staffing note, no "
            "customer or account is threatened."
        ),
    ),
]

Severity = Literal["low", "medium", "high"]
SpeakerRole = Literal["customer", "our-rep", "unknown"]


class Participant(BaseModel):
    """One entry in the envelope's `participants` list."""

    model_config = ConfigDict(extra="forbid")

    name: str
    role: str | None = None
    registry_id: str | None = None


class RegistryContextEntry(BaseModel):
    """One entry in the envelope's `registry_context`: an account, contact,
    product or competitor the connector already thinks is relevant."""

    model_config = ConfigDict(extra="forbid")

    name: str
    kind: EntityKind
    registry_id: str
    aliases: list[str] = Field(default_factory=list)


class Vocabulary(BaseModel):
    """The customer's own product names, jargon and known competitors —
    shared shape between the envelope's `vocabulary` field and the customer
    extension's `vocabulary` block, since both describe the same thing."""

    model_config = ConfigDict(extra="forbid")

    products: list[str] = Field(default_factory=list)
    competitors: list[str] = Field(default_factory=list)
    jargon: dict[str, str] = Field(default_factory=dict)


class SegmentRef(BaseModel):
    """`{index, total}` when a long source is split. Distinct from the
    `Segment` dataclass in `extract/segment.py`, which additionally carries
    the segment's text and locator range — this is just the small envelope
    field the extractor sees."""

    model_config = ConfigDict(extra="forbid")

    index: int = Field(ge=1)
    total: int = Field(ge=1)


class SourceEnvelope(BaseModel):
    """The envelope the connector hands the extractor: one per source, or
    one per segment of a long source. extraction-schema-v0.md "What the
    extractor receives"."""

    model_config = ConfigDict(extra="forbid")

    source_id: str
    source_type: SourceType
    source_date: date
    participants: list[Participant] = Field(default_factory=list)
    registry_context: list[RegistryContextEntry] = Field(default_factory=list)
    vocabulary: Vocabulary = Field(default_factory=Vocabulary)
    # Added 2026-09-17 (owner decision, "Company context page"):
    # {registry_id: text} from `portolan.context.build_account_context`,
    # one entry per ACCOUNT in `registry_context` -- the registry line
    # plus a capped first paragraph of that account's note synthesis.
    # Rendered by `portolan.extract.prompt.build_user_message` as its own
    # "Account context" section. Computed by the caller (this module
    # stays free of any DB access, per the module docstring) -- empty by
    # default, so every existing caller/test is unaffected.
    account_context: dict[str, str] = Field(default_factory=dict)
    text: str
    segment: SegmentRef | None = None

    @field_validator("source_id")
    @classmethod
    def _source_id_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("source_id must not be empty")
        return v


class EntityMention(BaseModel):
    """One entry in the output's `entities` list."""

    model_config = ConfigDict(extra="forbid")

    mention: str
    kind: EntityKind
    registry_id: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)


class _FactTypeValue(str):
    """The runtime value of `Fact.type` (review finding F13).

    Behaves exactly like the plain string it wraps -- equality, hashing,
    f-string interpolation, JSON serialization, membership in a
    `frozenset[FactType]` -- but also exposes `.value`, so code written
    against the old closed `FactType` `StrEnum` field (`fact.type.value`,
    in `portolan.merge` and `portolan.audit`, both other waves' modules
    this fix does not touch) keeps working unchanged. Opening `Fact.type`
    to a customer extension's type names must not silently break that call
    pattern in code this wave cannot edit.
    """

    @property
    def value(self) -> str:
        return str(self)


class Fact(BaseModel):
    """One extracted fact. Field-level and cross-field validators enforce
    everything in FACT_RULES that is mechanically checkable at this layer
    (rules 4, 5, and the closed-list/range parts of 6, 7, 9).

    `type` is a plain `str` (review finding F13), not the closed `FactType`
    enum: `extraction-schema-v0.md`'s customer extension advertises extra
    type names to the model via the prompt, and the schema must accept
    them back or every customer-extension fact fails validation, which is
    exactly the bug F13 reported. Validated in `_validate_against_effective_
    schema` below against the *effective* type list -- base `FactType`
    names when no customer extension is in play (the default, via
    `model_validate(data)` with no `context`), or base + extension names
    when a caller passes `context={"effective_schema": EffectiveSchema(...)}`
    (see `merge_customer_extension`).
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    type: str
    claim: str
    entities: list[str] = Field(default_factory=list)
    locator: str
    quote: str | None = None
    speaker: str | None = None
    valid_from: date | None = None
    valid_to: date | None = None
    severity: Severity | None = None
    sentiment: int = Field(ge=-2, le=2)
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("type")
    @classmethod
    def _type_non_empty(cls, v: str) -> _FactTypeValue:
        if not v or not v.strip():
            raise ValueError("type must not be empty")
        return _FactTypeValue(v)

    @field_validator("locator")
    @classmethod
    def _locator_required(cls, v: str) -> str:
        # Rule 4: "Locator is required. A fact without one is rejected at
        # merge and counted in the coverage report." Enforcing it here means
        # a malformed extraction never reaches the merge step at all.
        if not v or not v.strip():
            raise ValueError("locator is required and must not be empty (fact rule 4)")
        return v

    @field_validator("claim")
    @classmethod
    def _claim_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("claim must not be empty")
        return v

    @model_validator(mode="after")
    def _validate_against_effective_schema(self, info: ValidationInfo) -> Fact:
        """F13, in one place: (1) `type` must be in the effective type
        checklist -- base `FactType` names, plus a customer extension's
        `extra_types` when the caller supplies one via `context=
        {"effective_schema": EffectiveSchema(...)}`; (2) rule 5's quote
        requirement, honouring `quote_required` on an extension type, not
        just the five hardcoded base ones; (3) a customer extension's
        `severity_overrides` for `type`, applied here as the post-
        validation hook build-contracts.md calls for -- a keyword in
        `high_if` found in the claim or quote forces `severity="high"`
        (checked before `medium_if`, so a claim matching both reads high).
        """
        context = info.context or {}
        effective: EffectiveSchema | None = context.get("effective_schema")

        allowed_types = (
            set(effective.fact_type_checklist.keys())
            if effective is not None
            else {t.value for t in FactType}
        )
        if self.type not in allowed_types:
            raise ValueError(
                f"fact type {self.type!r} is not in the effective type checklist "
                "(base types plus any customer extension's extra_types)"
            )

        quote_required_types = (
            effective.quote_required_types
            if effective is not None
            else {t.value for t in QUOTE_REQUIRED_TYPES}
        )
        # Rule 5: quote required, verbatim, forty words or fewer.
        if self.type in quote_required_types:
            if not self.quote or not self.quote.strip():
                raise ValueError(f"quote is required for fact type {self.type!r} (fact rule 5)")
            word_count = len(self.quote.split())
            if word_count > 40:
                raise ValueError(
                    f"quote for fact type {self.type!r} must be forty words or "
                    f"fewer (fact rule 5); got {word_count}"
                )

        if effective is not None:
            override = effective.severity_overrides.get(self.type)
            if override is not None:
                haystack = f"{self.claim} {self.quote or ''}".lower()
                if any(keyword.lower() in haystack for keyword in override.high_if):
                    self.severity = "high"
                elif any(keyword.lower() in haystack for keyword in override.medium_if):
                    self.severity = "medium"

        return self


class Action(BaseModel):
    """`An action is not a fact.` extraction-schema-v0.md "Actions". Carries
    a locator like a fact, for the same citation discipline described in
    portolan-spec-v2.md section 2 ("A claim without a citation is not
    written") — the schema doc's own example action includes one."""

    model_config = ConfigDict(extra="forbid")

    what: str
    owner_mention: str | None = None
    due: date | None = None
    locator: str

    @field_validator("what")
    @classmethod
    def _what_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("what must not be empty")
        return v

    @field_validator("locator")
    @classmethod
    def _locator_required(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("locator is required and must not be empty")
        return v


class Coverage(BaseModel):
    """`{"segments_read": 1, "segments_total": 1}`."""

    model_config = ConfigDict(extra="forbid")

    segments_read: int = Field(ge=0)
    segments_total: int = Field(ge=0)


class RejectedFact(BaseModel):
    """One fact or action `ExtractionResult.parse_lenient` dropped rather
    than fail the whole result over (review finding, Group 3 of the fix
    wave: a real bench run crashed on a `commitment` fact with no quote
    and a `valid_from: "2025-08"` month-precision date -- one bad fact used
    to take the entire source's extraction down with it).

    `index` is the position in the raw `facts`/`actions` list `kind`
    identifies (`"fact"` or `"action"`); `raw` is exactly what the model
    sent for it, kept for audit/debugging."""

    model_config = ConfigDict(extra="forbid")

    index: int
    error: str
    raw: dict[str, Any]
    kind: Literal["fact", "action"] = "fact"


class EntityCoercion(BaseModel):
    """One default `ExtractionResult.parse_lenient` applied rather than
    reject the entity/fact/result it lives on over it -- fix instructions
    item 2 (the original Nova Lite/Micro bench losses: a model left
    `confidence` off an entity, or used a `kind` outside the closed
    `EntityKind` list, e.g. `"service"`), extended today by two more
    groups of the same shape:

    - A fact-level coercion (`field` one of `"entities"`, `"sentiment"`,
      `"severity"`, `"confidence"`): `entities` fixes a list item that came
      back as an object instead of a registry id/mention string (Nova
      Lite: `{"mention": "<a rep's name>", "kind": "internal_person",
      ...}`) or an empty string, `sentiment`/`confidence` default a `null`
      to `0`/`0.5`, `severity` drops a `null` to the field's own `None`
      default. `index` is the position in the raw `facts` list.
    - A result-level coercion (`field` one of `"coverage"`, `"summary"`,
      `"source_id"`): the "genuinely incomplete" top-level cases --
      `coverage` defaults to `{"segments_read": 1, "segments_total": 1}`,
      `summary` to `""`, `source_id` to the caller's `default_source_id`
      when parse_lenient was given one. `index` is `-1` (not a list
      position) and `mention` is `""`.

    None of these is a reason to drop the fact or the whole result, per
    this schema's own rule: drop the fact, never the source. This is the
    record of that substitution -- "recording the coercion" -- kept for
    audit/debugging the same way `RejectedFact` records a genuinely
    dropped fact or action.

    `original` is exactly what the model sent for the coerced field
    (`None` when the key was absent entirely, indistinguishable from an
    explicit `null` -- both mean "use the default"); `applied` is the
    default value substituted, or `None` when the offending value was
    dropped outright (an unusable `entities` item).
    """

    model_config = ConfigDict(extra="forbid")

    index: int
    mention: str
    field: Literal[
        "confidence",
        "kind",
        "entities",
        "sentiment",
        "severity",
        # A quote over the forty-word limit, cut to its first forty words
        # (still verbatim) instead of rejecting the fact it supports.
        "quote",
        "coverage",
        "summary",
        "source_id",
        # Added 2026-09-18 (infra-2b, non-fact bookkeeping leniency):
        # "entity" -- a top-level `entities` list item wasn't a dict at
        # all (Nemotron: a bare mention string); "unknown_entities" -- the
        # `unknown_entities` field itself, or one of its items, wasn't the
        # `list[str]` shape it's declared as.
        "entity",
        "unknown_entities",
    ]
    original: Any = None
    applied: Any


# Fact rule 5: "Verbatim, forty words or fewer."
_MAX_QUOTE_WORDS = 40

_YEAR_MONTH_RE = re.compile(r"^\d{4}-\d{2}$")
_YEAR_ONLY_RE = re.compile(r"^\d{4}$")


def _coerce_fact_entity_item(item: Any) -> tuple[str | None, bool]:
    """One item of a `Fact.entities` list, coerced to the plain string that
    field actually wants. Returns `(value, changed)`: `value` is `None`
    when the item is unusable and must be dropped; `changed` is `False`
    only for an already-good non-empty string (the common case, kept out
    of `coercions` entirely).

    Handles the real Nova Lite bench failure: a fact's `entities` item
    came back shaped like a top-level `EntityMention`
    (`{"mention": "Priya Natarajan", "kind": "internal_person",
    "registry_id": "internal:priya-natarajan"}`) instead of the registry
    id or mention string `Fact.entities: list[str]` expects -- the model
    conflating the two different `entities` shapes in the schema. Prefers
    `registry_id` (a resolved reference, rule 6) over `mention` (free
    text) when both are present and usable. An empty string is likewise
    coerced away -- dropped, not kept as a blank entity reference.
    """
    if isinstance(item, str):
        if item.strip():
            return item, False
        return None, True

    if isinstance(item, dict):
        registry_id = item.get("registry_id")
        if isinstance(registry_id, str) and registry_id.strip():
            return registry_id, True
        mention = item.get("mention")
        if isinstance(mention, str) and mention.strip():
            return mention, True
        return None, True

    # Anything else (a number, a list, None, ...) is unusable.
    return None, True


# Fix wave 2026-09-18 (Kimi K2.5 bench: 9 to 14 sources per run lost to
# schema validation because the model wrote `confidence` as a word, not a
# number -- see `_coerce_confidence` below).
_CONFIDENCE_WORDS: dict[str, float] = {
    "high": 0.9,
    "very high": 0.9,
    "medium": 0.6,
    "moderate": 0.6,
    "low": 0.3,
    # Not explicitly specified by the fix instructions, but the natural
    # counterpart to "very high" -> 0.9 above; anything else unmapped
    # still falls through to `default`, per those instructions.
    "very low": 0.3,
}


def _coerce_confidence(value: Any, default: float = 0.5) -> tuple[float, bool]:
    """`Fact.confidence`/`EntityMention.confidence` both want a float in
    [0.0, 1.0]; a model routinely sends something else. Kimi K2.5 bench,
    2026-09-18: an entity's `confidence` came back as "high"/"medium"/
    "low" -- not `None`, so the coercion loops' old `is None` check
    missed it entirely, and an entity (unlike a fact) is never validated
    item-by-item in this function -- it only ever fails at the final
    strict `cls.model_validate(working, ...)` below, outside any per-item
    try/except, so ONE bad entity confidence took the WHOLE source's
    extraction down with it (9 to 14 sources per run). This closes that
    gap for both the entity-level and fact-level confidence coercion
    below, and for `severity_overrides`-adjacent callers that reuse it.

    Returns `(value, changed)`: `value` is always a float in [0.0, 1.0];
    `changed` is `False` only for an already-good in-range int/float (the
    common case, kept out of `coercions` entirely, matching every other
    coercion helper in this module).

    - `None`, an empty/blank string, a bool, or anything else this
      function doesn't recognize -> `default`.
    - A number (int/float) -> clamped to [0.0, 1.0].
    - A numeric string ("0.8", "1", "  0.42 ") -> parsed then clamped.
    - A confidence word, case/whitespace-insensitive ("high", "very
      high", "medium", "moderate", "low", "very low") -> the fixed value
      in `_CONFIDENCE_WORDS`.
    """
    if isinstance(value, bool):
        # bool is an int subclass in Python -- True/False must be checked
        # before the int | float branch below would otherwise silently
        # accept them as 1.0 / 0.0.
        return default, True

    if isinstance(value, int | float):
        clamped = max(0.0, min(1.0, float(value)))
        return clamped, clamped != value

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return default, True
        try:
            parsed = float(stripped)
        except ValueError:
            word_value = _CONFIDENCE_WORDS.get(stripped.lower())
            return (word_value, True) if word_value is not None else (default, True)
        return max(0.0, min(1.0, parsed)), True

    return default, True


# Fix wave 2026-09-22 (`01-spec/bench-deepseek-v41-flash-2026-09-22.md`):
# DeepSeek V4.1 Flash routinely sends `sentiment` as a word
# ("negative"/"neutral"/"positive") instead of the required int scale --
# only an explicit `null` was coerced (to `0`) before this fix, so every
# string value failed `Fact.sentiment`'s `ge=-2, le=2` int validation and
# took the WHOLE FACT down with it: 0 facts produced across all 185
# non-call sources in that bench. The silver-labeling pipeline
# (`scripts/data/silver_label.py`) shares this parser, so some DeepSeek
# V3.2/V4.1 Flash labels were lost the same silent way.
_SENTIMENT_WORDS: dict[str, int] = {
    "very negative": -2,
    "negative": -1,
    "neg": -1,
    "neutral": 0,
    "positive": 1,
    "pos": 1,
    "very positive": 2,
}


def _coerce_sentiment(value: str, default: int = 0) -> int:
    """`Fact.sentiment` wants an int on the five-point scale `-2` (very
    negative) .. `2` (very positive), `0` neutral -- the same scale
    `extraction-schema-v0.md`'s own example and `_SENTIMENT_WORDS` above
    use. This handles the STRING shapes a model sends instead of the
    int: a sentiment word, case/whitespace-insensitive ("very negative",
    "negative", "neg", "neutral", "positive", "pos", "very positive"), or
    a numeric string, signed or not ("2", "-1", "+1", "  0 ", "1.0").
    Anything else this function doesn't recognize -> `default` -- the
    same value an explicit `null` already maps to, never a reason to
    reject the whole fact.

    Only called on `str` values; `None` is handled by the existing
    null-coercion check next to this function's call site, and an
    already-good in-range int is never routed through here at all.
    """
    stripped = value.strip()
    if not stripped:
        return default

    try:
        parsed = int(stripped)
    except ValueError:
        try:
            parsed = round(float(stripped))
        except ValueError:
            word_value = _SENTIMENT_WORDS.get(stripped.lower())
            return word_value if word_value is not None else default

    return max(-2, min(2, parsed))


def _coerce_partial_date(value: Any) -> Any:
    """`"2025-08"` -> `"2025-08-01"`, `"2025"` -> `"2025-01-01"`. Anything
    else (a full `YYYY-MM-DD` string, `None`, a non-string) passes through
    unchanged -- this only widens what already-lenient date parsing
    accepts, it never invents a date `pydantic` would otherwise reject for
    a different reason."""
    if isinstance(value, str):
        if _YEAR_MONTH_RE.match(value):
            return f"{value}-01"
        if _YEAR_ONLY_RE.match(value):
            return f"{value}-01-01"
    return value


# --------------------------------------------------------------------------
# Fix wave 2026-09-18 (`01-spec/extraction-accuracy-campaign-2026-09-18.md`,
# infra-2b): non-fact bookkeeping leniency. Two real, independently-
# diagnosed bench failures, both a bare Python exception escaping
# `parse_lenient` entirely (never a `pydantic.ValidationError`, so neither
# was ever caught by this function's own try/except blocks OR by
# `portolan.extract.backends.LLMClientBackend.extract`'s
# `except ValidationError:` one layer up -- both took the WHOLE source
# down, violating this module's own rule: "a source must only fail when
# its facts cannot be parsed"):
#
# 1. `openai/gpt-5.6-luna`, 207 of 209 sources (`var/bench/
#    2026-09-18-models/openai_gpt-5.6-luna/run1/`): the model's facts
#    parse fine, but it reshapes the bookkeeping `coverage` object into
#    whatever field names it feels like -- a different shape almost every
#    source (`{"source_segment_reviewed": true, "high_value_signals_
#    captured": true, "omitted_low_durability_content": true}` on one,
#    `{"source_lines_considered": 137, "facts_extracted": 30, "high_
#    value_signals": 30}` on the next), and on 17 of the 207 `coverage`
#    isn't even a dict -- a bare string like `"complete"` or a free-text
#    sentence. Before this fix, only a MISSING `coverage` key got a safe
#    default (see the block below); a PRESENT-but-malformed one hit
#    `Coverage`'s own strict validation inside the final `cls.
#    model_validate(working, ...)` and raised `pydantic.ValidationError`
#    for the entire result -- exactly the gap this module's own docstring
#    already flagged ("Still raises ... for a malformed, not merely
#    missing, coverage block"). See `_coerce_coverage` below.
# 2. `nvidia/nemotron-3-super-120b-a12b` / `nvidia/nemotron-3-nano-30b-a3b`
#    (`var/bench/2026-09-18-models/nvidia_nemotron-3-{super-120b-a12b,
#    nano-30b-a3b}/run1/`, 1 and 2 sources respectively): both models
#    occasionally emit a top-level `entities` list item as a bare mention
#    string (`"Ashgrove Health"`) instead of the `{"mention": ...,
#    "kind": ..., "confidence": ...}` shape -- exactly what
#    `unknown_entities` (a plain `list[str]`) already expects, just under
#    the wrong key. The old code called `dict(raw_entity)` unconditionally
#    on every list item; for a multi-character string that raises
#    `ValueError: dictionary update sequence element #0 has length 1; 2 is
#    required` (Python tries to build key/value pairs from the string's
#    own individual characters) -- a bare `ValueError`, not a
#    `ValidationError`, propagating straight out of this function. See the
#    `entities` loop below.
# --------------------------------------------------------------------------


def _as_item_list(value: Any) -> list[Any]:
    """Normalize a top-level `entities`/`facts`/`actions` field to a list
    of items to walk one at a time. `None`/missing -> `[]`; already a
    list -> unchanged; anything else (a lone dict, a lone scalar -- a
    model that collapsed what should have been a one-item list) -> a
    one-item list holding it, so the per-item leniency below still gets a
    chance rather than an `enumerate()` over a dict's keys or a string's
    characters silently doing the wrong thing."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


_DEFAULT_COVERAGE: dict[str, int] = {"segments_read": 1, "segments_total": 1}


def _coerce_coverage(value: Any) -> tuple[dict[str, Any], bool]:
    """`Coverage` is pure bookkeeping -- never a fact -- so a malformed
    value here must never take the whole result down with it (see the
    module-level comment above for the real Luna failure this closes).

    Returns `(value, changed)`: a dict that already validates against
    `Coverage` passes through unchanged (`changed=False`); anything else
    (a differently-shaped dict, a string, a list, `None`) is replaced by
    the same safe default a MISSING `coverage` key already got.
    """
    if isinstance(value, dict):
        try:
            Coverage.model_validate(value)
        except ValidationError:
            pass
        else:
            return value, False
    return dict(_DEFAULT_COVERAGE), True


class ExtractionResult(BaseModel):
    """The extractor's output. Strict JSON, enforced by the API's structured
    output — this model IS that strict schema; `model_json_schema()` is
    handed to the backends as the tool/output schema they force the model
    into."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    source_id: str
    summary: str
    entities: list[EntityMention] = Field(default_factory=list)
    facts: list[Fact] = Field(default_factory=list)
    actions: list[Action] = Field(default_factory=list)
    unknown_entities: list[str] = Field(default_factory=list)
    coverage: Coverage
    # Never produced by the model (excluded from `extraction_result_json_
    # schema()` below) -- filled in by `parse_lenient` when it had to drop
    # a fact or action to salvage the rest of the result. Empty for a
    # result built the normal, fully-strict way.
    rejected: list[RejectedFact] = Field(default_factory=list)
    # Same shape, different fate: `parse_lenient` fills this in with every
    # entity-level default it had to apply (a missing `confidence`, a
    # `kind` outside the closed list) -- the entity was KEPT, just
    # corrected, unlike `rejected`. Also excluded from the model-facing
    # schema; also empty for a result built the normal, fully-strict way.
    coercions: list[EntityCoercion] = Field(default_factory=list)

    @property
    def coerced_sentiment_count(self) -> int:
        """How many facts in `.coercions` had their `sentiment` field
        defaulted or coerced -- a `null` defaulted to `0`, or a string
        like `"negative"`/`"+1"` mapped onto the int scale (fix wave
        2026-09-22, `_coerce_sentiment`). Computed, not stored, so a
        result built any other way (a direct `model_validate`, an older
        saved report re-loaded as this type) still reports `0` correctly
        without a migration. Lets the bench and `scripts/data/
        silver_label.py --stats` surface how often this happens instead
        of it staying invisible in the scorecard the way the DeepSeek
        V4.1 Flash bench run found it.
        """
        return sum(1 for c in self.coercions if c.field == "sentiment")

    @classmethod
    def parse_lenient(
        cls,
        data: dict[str, Any],
        *,
        effective_schema: EffectiveSchema | None = None,
        default_source_id: str | None = None,
    ) -> tuple[ExtractionResult, list[RejectedFact]]:
        """Best-effort parse of one model's raw JSON reply.

        A strict `ExtractionResult.model_validate(data)` fails the WHOLE
        result the moment a single fact or action is invalid -- exactly
        what took down the first real bench run, on a `commitment` fact
        missing its rule-5 quote and a fact whose `valid_from` was
        `"2025-08"` (month precision, not a full date). This coerces the
        date shapes `_coerce_partial_date` can fix, then validates each
        fact and action ONE AT A TIME, dropping (into the returned
        `list[RejectedFact]`, never raising for) whichever ones still
        don't validate. The rest of the extraction is never held hostage
        by one bad fact.

        Two more tolerances (fix instructions item 2, the real Nova Lite/
        Micro bench losses -- 12 and 13 sources respectively): a model
        routinely echoes a plausible key the schema never asked for, at
        the TOP level (`source_type`, `source_date`, `rejected_facts: []`,
        `unknown_competitors: []` were all observed live) or on a fact
        (mirroring the additionalProperties leniency
        `portolan.llm.openrouter` already applies to its own fallback-path
        schema, for the identical reason) -- these are stripped before
        validation rather than left to fail `ExtractionResult`'s own
        `extra="forbid"` the same way an unknown fact/action key would.
        Before this fix, the top-level strip didn't happen: the FINAL
        `cls.model_validate(working, ...)` call below was just as strict
        as the two strict attempts `portolan.extract.backends.
        LLMClientBackend.extract` already made before ever reaching this
        function, so a model that repeated the same top-level extra on
        both attempts made `parse_lenient` itself raise the very
        `pydantic.ValidationError` it exists to absorb -- one level too
        late for that backend's `except ValidationError:` to help, since
        it was already inside the branch that calls `parse_lenient`
        precisely because the direct attempt had failed. An entity mention
        missing `confidence` or carrying a `kind` outside the closed
        `EntityKind` list is, similarly, defaulted (`0.5`, `other`) rather
        than left to fail validation -- each substitution recorded as an
        `EntityCoercion` on the returned result's `.coercions` (`.rejected`'s
        counterpart: kept and corrected, not dropped).

        Three more tolerances, added today, for the "genuinely incomplete"
        top-level cases a real bench run hit (a model that emitted little
        more than `{"entities": [], "facts": [], "actions": []}`, missing
        `source_id`/`summary`/`coverage` entirely -- previously this
        function's own final `cls.model_validate(working, ...)` raised
        `pydantic.ValidationError` right back out, undoing everything the
        per-fact/per-entity leniency above bought, since this schema's own
        rule is drop the fact, never the source): a missing `coverage`
        defaults to `{"segments_read": 1, "segments_total": 1}`, a missing
        `summary` defaults to `""`, and a missing `source_id` is filled
        from `default_source_id` when the caller supplies one (the
        envelope's own `source_id` -- this function has no envelope, only
        whatever `LLMClientBackend.extract` passes here). Each is recorded
        as an `EntityCoercion` the same as the per-fact/per-entity ones,
        with `index=-1` (there's no list position at the top level) and
        `mention=""`. A `source_id` that's missing with no
        `default_source_id` supplied is left missing -- that still raises
        below, same as before this fix, since there's nothing to fill it
        with.

        Fix wave 2026-09-18 (infra-2b, `01-spec/extraction-accuracy-
        campaign-2026-09-18.md`): a PRESENT-but-malformed `coverage`
        (any shape that doesn't validate against `Coverage`, including a
        bare string) is now ALSO defaulted, not just a missing one --
        `_coerce_coverage` closes the gap this docstring used to flag
        right here as a known limitation (the real Luna failure: 207 of
        209 sources, every one a differently-reshaped `coverage` object).
        `summary`/`source_id` present with the wrong type, `unknown_
        entities` present with the wrong shape, and a top-level `entities`/
        `facts`/`actions` list item that isn't a JSON object at all (the
        real Nemotron 3 Super/Nano failure: a bare mention string in
        `entities`) are handled the same way, each at its own call site
        above/below. Still raises `pydantic.ValidationError` for a problem
        outside all of the above (a missing `source_id` with no `default_
        source_id` given is the main one left) -- that is a genuinely
        different failure this function does not try to paper over.
        """
        context = {"effective_schema": effective_schema} if effective_schema is not None else None
        working = dict(data)
        rejected: list[RejectedFact] = []
        coercions: list[EntityCoercion] = []

        # Ignore unknown top-level keys rather than let the final strict
        # `model_validate` below choke on them (see docstring above).
        known_top_level_fields = set(cls.model_fields.keys())
        working = {key: value for key, value in working.items() if key in known_top_level_fields}

        # Entities: strip unknown per-entity keys, default a missing
        # `confidence` to 0.5, default a `kind` outside the closed list to
        # `other`. The entity is always kept -- only facts/actions are ever
        # dropped by this function.
        known_entity_fields = set(EntityMention.model_fields.keys())
        valid_entity_kinds = {kind.value for kind in EntityKind}
        kept_entities: list[dict[str, Any]] = []
        for index, raw_entity in enumerate(_as_item_list(working.get("entities"))):
            if isinstance(raw_entity, dict):
                candidate = {
                    key: value for key, value in raw_entity.items() if key in known_entity_fields
                }
            else:
                # Real Nemotron 3 Super/Nano failure (see the module-level
                # comment above `_as_item_list`): a bare mention string in
                # place of the `{"mention": ..., "kind": ..., "confidence":
                # ...}` shape. The entity is always KEPT (this function's
                # own rule) -- synthesized from whatever scalar the model
                # actually sent; `confidence`/`kind` get their usual
                # defaults from the unchanged code just below, exactly as
                # they would for a dict entity that merely omitted them.
                candidate = {"mention": "" if raw_entity is None else str(raw_entity)}
                coercions.append(
                    EntityCoercion(
                        index=index,
                        mention=candidate["mention"],
                        field="entity",
                        original=raw_entity,
                        applied=candidate,
                    )
                )
            mention = candidate.get("mention")
            mention_str = mention if isinstance(mention, str) else str(mention or "")

            original_confidence = candidate.get("confidence")
            confidence, confidence_changed = _coerce_confidence(original_confidence)
            candidate["confidence"] = confidence
            if confidence_changed:
                coercions.append(
                    EntityCoercion(
                        index=index,
                        mention=mention_str,
                        field="confidence",
                        original=original_confidence,
                        applied=confidence,
                    )
                )

            if candidate.get("kind") not in valid_entity_kinds:
                original = candidate.get("kind")
                candidate["kind"] = EntityKind.OTHER.value
                coercions.append(
                    EntityCoercion(
                        index=index,
                        mention=mention_str,
                        field="kind",
                        original=original,
                        applied=EntityKind.OTHER.value,
                    )
                )

            kept_entities.append(candidate)
        working["entities"] = kept_entities

        known_fact_fields = set(Fact.model_fields.keys())
        kept_facts: list[dict[str, Any]] = []
        for index, raw_fact in enumerate(_as_item_list(working.get("facts"))):
            if not isinstance(raw_fact, dict):
                # Same non-dict-item shape the `entities` loop above
                # closes, on `facts` instead -- not independently observed
                # in a real run as of this fix, but the old `dict(raw_fact)`
                # call would crash the same way on a bare string/number/
                # list item. Unlike an entity (always kept), a fact in an
                # unusable shape genuinely CANNOT be parsed -- rejected,
                # not defaulted, per this module's own rule: drop the one
                # fact, never the whole source.
                rejected.append(
                    RejectedFact(
                        index=index,
                        error=f"fact item is not a JSON object (got {type(raw_fact).__name__})",
                        raw={"value": raw_fact},
                        kind="fact",
                    )
                )
                continue
            candidate = {key: value for key, value in raw_fact.items() if key in known_fact_fields}
            candidate["valid_from"] = _coerce_partial_date(candidate.get("valid_from"))
            candidate["valid_to"] = _coerce_partial_date(candidate.get("valid_to"))
            fact_label = str(candidate.get("id") or "")

            # A fact's own `entities` list wants plain registry id/mention
            # strings (`Fact.entities: list[str]`) -- a model that instead
            # nests a top-level-`EntityMention`-shaped object (or an empty
            # string) there is fixed up item by item, never left to fail
            # `Fact.model_validate` and take the whole fact down with it.
            if isinstance(candidate.get("entities"), list):
                coerced_entities: list[str] = []
                for raw_item in candidate["entities"]:
                    value, changed = _coerce_fact_entity_item(raw_item)
                    if changed:
                        coercions.append(
                            EntityCoercion(
                                index=index,
                                mention=fact_label,
                                field="entities",
                                original=raw_item,
                                applied=value,
                            )
                        )
                    if value is not None:
                        coerced_entities.append(value)
                candidate["entities"] = coerced_entities

            # `sentiment`/`confidence` are required, non-nullable fields;
            # a `null` (or absent) value is defaulted rather than left to
            # reject the whole fact. `severity` is already optional and
            # defaults to `None` on its own -- an explicit `null` needs no
            # value change, just recording (it's otherwise silent).
            #
            # A string sentiment ("negative", "+1", ...) gets the same
            # treatment (fix wave 2026-09-22, see `_coerce_sentiment`
            # above): mapped onto the int scale when recognized, defaulted
            # to `0` -- same as `null` -- when not. Either way this is
            # recorded as a `field="sentiment"` coercion, same as the null
            # case, so `ExtractionResult.coerced_sentiment_count` counts
            # both.
            raw_sentiment = candidate.get("sentiment")
            if raw_sentiment is None:
                candidate["sentiment"] = 0
                coercions.append(
                    EntityCoercion(
                        index=index,
                        mention=fact_label,
                        field="sentiment",
                        original=raw_sentiment,
                        applied=0,
                    )
                )
            elif isinstance(raw_sentiment, str):
                coerced_sentiment = _coerce_sentiment(raw_sentiment)
                candidate["sentiment"] = coerced_sentiment
                coercions.append(
                    EntityCoercion(
                        index=index,
                        mention=fact_label,
                        field="sentiment",
                        original=raw_sentiment,
                        applied=coerced_sentiment,
                    )
                )

            original_confidence = candidate.get("confidence")
            confidence, confidence_changed = _coerce_confidence(original_confidence)
            candidate["confidence"] = confidence
            if confidence_changed:
                coercions.append(
                    EntityCoercion(
                        index=index,
                        mention=fact_label,
                        field="confidence",
                        original=original_confidence,
                        applied=confidence,
                    )
                )

            raw_quote = candidate.get("quote")
            if isinstance(raw_quote, str) and len(raw_quote.split()) > _MAX_QUOTE_WORDS:
                trimmed = " ".join(raw_quote.split()[:_MAX_QUOTE_WORDS])
                candidate["quote"] = trimmed
                coercions.append(
                    EntityCoercion(
                        index=index,
                        mention=fact_label,
                        field="quote",
                        original=raw_quote,
                        applied=trimmed,
                    )
                )

            if "severity" in candidate and candidate["severity"] is None:
                coercions.append(
                    EntityCoercion(
                        index=index,
                        mention=fact_label,
                        field="severity",
                        original=None,
                        applied=None,
                    )
                )

            try:
                Fact.model_validate(candidate, context=context)
            except ValidationError as exc:
                rejected.append(
                    RejectedFact(index=index, error=str(exc), raw=raw_fact, kind="fact")
                )
                continue
            kept_facts.append(candidate)

        known_action_fields = set(Action.model_fields.keys())
        kept_actions: list[dict[str, Any]] = []
        for index, raw_action in enumerate(_as_item_list(working.get("actions"))):
            if not isinstance(raw_action, dict):
                # Same non-dict-item guard as `facts` above. An action is
                # "not a fact" (its own class docstring), but the same
                # rule applies: reject the one item, never crash the
                # source over it.
                rejected.append(
                    RejectedFact(
                        index=index,
                        error=f"action item is not a JSON object (got {type(raw_action).__name__})",
                        raw={"value": raw_action},
                        kind="action",
                    )
                )
                continue
            candidate = {
                key: value for key, value in raw_action.items() if key in known_action_fields
            }
            candidate["due"] = _coerce_partial_date(candidate.get("due"))
            try:
                Action.model_validate(candidate)
            except ValidationError as exc:
                rejected.append(
                    RejectedFact(index=index, error=str(exc), raw=raw_action, kind="action")
                )
                continue
            kept_actions.append(candidate)

        working["facts"] = kept_facts
        working["actions"] = kept_actions

        # The "genuinely incomplete" (or, as of this fix, genuinely
        # MALFORMED) top-level cases: `coverage`/`summary` get a safe
        # default rather than failing the whole result; a missing
        # `source_id` is filled from the caller's `default_source_id` when
        # one was given (see docstring) and otherwise left alone -- the
        # final `model_validate` below still raises for that, same as
        # always.
        coverage_value, coverage_changed = _coerce_coverage(working.get("coverage"))
        if coverage_changed:
            coercions.append(
                EntityCoercion(
                    index=-1,
                    mention="",
                    field="coverage",
                    original=working.get("coverage"),
                    applied=coverage_value,
                )
            )
        working["coverage"] = coverage_value

        if "summary" not in working or working["summary"] is None:
            original_summary = working.get("summary")
            working["summary"] = ""
            coercions.append(
                EntityCoercion(
                    index=-1, mention="", field="summary", original=original_summary, applied=""
                )
            )
        elif not isinstance(working["summary"], str):
            # A model that hands back a structured object or a number
            # where a plain summary sentence was expected -- `str()`'d
            # rather than left to fail `summary: str`'s type check and
            # take the whole result down over a field nothing downstream
            # scores on.
            original_summary = working["summary"]
            working["summary"] = str(original_summary)
            coercions.append(
                EntityCoercion(
                    index=-1,
                    mention="",
                    field="summary",
                    original=original_summary,
                    applied=working["summary"],
                )
            )

        if "source_id" not in working and default_source_id is not None:
            working["source_id"] = default_source_id
            coercions.append(
                EntityCoercion(
                    index=-1,
                    mention="",
                    field="source_id",
                    original=None,
                    applied=default_source_id,
                )
            )
        elif isinstance(working.get("source_id"), int | float):
            # A bare numeric id (e.g. a CRM row number) instead of a
            # string -- `str()`'d rather than left to fail `source_id:
            # str`'s type check.
            original_source_id = working["source_id"]
            working["source_id"] = str(original_source_id)
            coercions.append(
                EntityCoercion(
                    index=-1,
                    mention="",
                    field="source_id",
                    original=original_source_id,
                    applied=working["source_id"],
                )
            )

        # `unknown_entities` is a plain `list[str]` -- the review queue for
        # a mention the extractor couldn't resolve to a registry id (rule
        # 6). Not independently observed failing in a real bench run as of
        # this fix, but it is exactly the same class of non-fact
        # bookkeeping field as `coverage`/`summary`/`source_id` above, and
        # a model that hands back a non-list, or a list with a non-string
        # item, would otherwise fail this field's type check in the final
        # strict `model_validate` below and take the whole result down
        # over a field that exists purely as a review-queue hint.
        raw_unknown_entities = working.get("unknown_entities")
        if raw_unknown_entities is not None and not isinstance(raw_unknown_entities, list):
            working["unknown_entities"] = []
            coercions.append(
                EntityCoercion(
                    index=-1,
                    mention="",
                    field="unknown_entities",
                    original=raw_unknown_entities,
                    applied=[],
                )
            )
        elif isinstance(raw_unknown_entities, list):
            kept_unknown: list[str] = []
            dropped_any = False
            for item in raw_unknown_entities:
                if isinstance(item, str) and item.strip():
                    kept_unknown.append(item)
                else:
                    dropped_any = True
            if dropped_any:
                coercions.append(
                    EntityCoercion(
                        index=-1,
                        mention="",
                        field="unknown_entities",
                        original=raw_unknown_entities,
                        applied=kept_unknown,
                    )
                )
                working["unknown_entities"] = kept_unknown

        result = cls.model_validate(working, context=context)
        result.rejected = rejected
        result.coercions = coercions
        return result, rejected


def extraction_result_json_schema() -> dict[str, Any]:
    """Strict JSON schema for ExtractionResult, suitable for a Bedrock
    Converse tool `inputSchema.json` or an Anthropic tool `input_schema`.

    Pydantic v2's `model_json_schema()` already sets `additionalProperties:
    false` on every model here because every one of them declares
    `model_config = ConfigDict(extra="forbid")`. This wrapper exists so
    callers don't have to know that pydantic detail — they just get "the
    strict schema" — and so the enum-as-$ref shape pydantic emits stays an
    implementation detail of this one function.

    `rejected` and `coercions` are both dropped from the schema handed to
    the model: they're wrapper-only bookkeeping fields
    `ExtractionResult.parse_lenient` fills in after the fact, never
    something the model itself should produce.
    """
    schema = ExtractionResult.model_json_schema()
    schema.get("properties", {}).pop("rejected", None)
    schema.get("properties", {}).pop("coercions", None)
    if "required" in schema:
        schema["required"] = [
            name for name in schema["required"] if name not in ("rejected", "coercions")
        ]
    return schema


# --------------------------------------------------------------------------
# Customer extension (extraction-schema-v0.md "The customer extension")
# --------------------------------------------------------------------------


class ExtraFactType(BaseModel):
    """One entry in a customer extension's `extra_types` list."""

    model_config = ConfigDict(extra="forbid")

    name: str
    definition: str
    quote_required: bool = False
    # Extensions are written at onboarding without a worked example or
    # counter-example on hand yet (the doc's sample extension omits both).
    # Both are optional so the extension format doesn't force onboarding to
    # invent one before the schema can be tested against real sources.
    example: str | None = None
    counter_example: str | None = None
    counter_reason: str | None = None


class SkipRule(BaseModel):
    """One entry in a customer extension's `skip_rules` list. `when` is a
    short free-text condition (e.g. "subject matches /.../"); the doc does
    not define a formal grammar for it, so it is kept as an opaque string
    here and interpreted by the filter step (spec section 5.2), not by this
    schema module."""

    model_config = ConfigDict(extra="forbid")

    source_type: SourceType
    when: str


class SeverityOverride(BaseModel):
    """One entry in a customer extension's `severity_overrides` map, e.g.
    `complaint: {high_if: [delivery, patient, safety, outage]}`."""

    model_config = ConfigDict(extra="forbid")

    high_if: list[str] = Field(default_factory=list)
    medium_if: list[str] = Field(default_factory=list)


class CustomerExtension(BaseModel):
    """The per-customer YAML written at onboarding, extraction-schema-v0.md
    "The customer extension (produced at onboarding)"."""

    model_config = ConfigDict(extra="forbid")

    customer: str
    schema_version: str = SCHEMA_VERSION
    extra_types: list[ExtraFactType] = Field(default_factory=list)
    vocabulary: Vocabulary = Field(default_factory=Vocabulary)
    aliases: dict[str, list[str]] = Field(default_factory=dict)
    skip_rules: list[SkipRule] = Field(default_factory=list)
    severity_overrides: dict[str, SeverityOverride] = Field(default_factory=dict)


class EffectiveSchema(BaseModel):
    """The result of merging a CustomerExtension into the base schema: the
    full type checklist (base + extra) and the merged vocabulary. This is
    what `extract/prompt.py` actually renders — it never re-merges the
    extension itself, keeping the merge logic in one place.

    `quote_required_types` and `severity_overrides` (review finding F13)
    are what let `Fact`'s validator (`_validate_against_effective_schema`,
    passed this object via `model_validate(data, context={"effective_
    schema": effective})`) honour an extension type's own `quote_required`
    flag and a customer's `severity_overrides`, instead of only ever
    knowing about the five hardcoded base evidentiary types.
    """

    model_config = ConfigDict(extra="forbid")

    fact_type_checklist: dict[str, FactTypeInfo]
    extra_type_names: list[str]
    vocabulary: Vocabulary
    aliases: dict[str, list[str]]
    quote_required_types: frozenset[str] = frozenset()
    severity_overrides: dict[str, SeverityOverride] = Field(default_factory=dict)


def merge_customer_extension(extension: CustomerExtension | None) -> EffectiveSchema:
    """Merge a customer extension into the base type list and vocabulary.

    Base types always come first (in FACT_TYPES order), extra types are
    appended in the order the extension lists them. Vocabulary lists are
    unioned and de-duplicated while preserving first-seen order (via
    `dict.fromkeys`, not a hand-rolled `_union` helper -- every call site
    here only ever passes one list, so that's all `_union` ever was);
    jargon dicts are merged with the extension's definitions winning on key
    collision (onboarding's correction of a base term should stick).
    """
    checklist: dict[str, FactTypeInfo] = {
        fact_type.value: info for fact_type, info in FACT_TYPES.items()
    }
    extra_names: list[str] = []
    quote_required_types: set[str] = {t.value for t in QUOTE_REQUIRED_TYPES}

    if extension is None:
        return EffectiveSchema(
            fact_type_checklist=checklist,
            extra_type_names=extra_names,
            vocabulary=Vocabulary(),
            aliases={},
            quote_required_types=frozenset(quote_required_types),
            severity_overrides={},
        )

    for extra in extension.extra_types:
        checklist[extra.name] = FactTypeInfo(
            definition=extra.definition,
            example=extra.example or "(no example provided at onboarding)",
            counter_example=extra.counter_example or "(no counter-example provided at onboarding)",
            counter_reason=extra.counter_reason or "(not specified)",
        )
        extra_names.append(extra.name)
        if extra.quote_required:
            quote_required_types.add(extra.name)

    vocabulary = Vocabulary(
        products=list(dict.fromkeys(extension.vocabulary.products)),
        competitors=list(dict.fromkeys(extension.vocabulary.competitors)),
        jargon=dict(extension.vocabulary.jargon),
    )
    aliases = dict(extension.aliases)

    return EffectiveSchema(
        fact_type_checklist=checklist,
        extra_type_names=extra_names,
        vocabulary=vocabulary,
        aliases=aliases,
        quote_required_types=frozenset(quote_required_types),
        severity_overrides=dict(extension.severity_overrides),
    )
