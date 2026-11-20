"""Pass-2 reviewer: verify pass 1's facts and sweep for what it missed.

Wave 1D of `01-spec/extraction-accuracy-campaign-2026-09-18.md`, the maintainer's
two-pass idea: "First pass is normal extraction, second pass checks the
first pass's work and looks for missed facts and errors." Pass 1 is
FROZEN (the saved Nova Lite run 1 at
`var/bench/2026-09-18-corrected/amazon_nova-lite-v1/run1/`) -- this module
only builds and parses the SECOND call: given one source's text/context
(reusing the renderers already in `portolan.extract.prompt`) and pass 1's
own produced facts for that source, ask a reviewer model for:

(a) a keep/fix/reject verdict on every pass-1 fact for this source, and
(b) a per-type sweep for facts pass 1 missed, one targeted question per
    fact type ("any pricing statements not already listed?").

Both come back in ONE request's JSON, parsed leniently -- the same
drop-the-bad-item-not-the-whole-result philosophy
`portolan.extract.schema.ExtractionResult.parse_lenient` applies to a
fact/action, reimplemented here (not imported: `ReviewResult` is a
different shape than `ExtractionResult`, and this module owns it).

Ownership: this file, not `portolan/extract/prompt.py` or
`portolan/extract/schema.py` -- both are imported from here, never
edited (`01-spec/extraction-accuracy-campaign-2026-09-18.md` wave 1D's
ownership rule).
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from portolan.extract.prompt import build_user_message
from portolan.extract.schema import FACT_TYPES, FactType, SourceEnvelope
from portolan.metering import Surface
from portolan.util.prompt_safety import clean

if TYPE_CHECKING:
    from portolan.bench.extraction import ProducedFactRecord

_FACT_TYPE_VALUES: list[str] = [t.value for t in FactType]

RejectReason = Literal["not_in_source", "wrong_account", "duplicate"]
Verdict = Literal["keep", "fix", "reject"]


# --------------------------------------------------------------------------
# Reviewer output schema
# --------------------------------------------------------------------------


class FactVerdict(BaseModel):
    """One verdict on one pass-1 fact, keyed by that fact's own `id`
    (e.g. `f3`) from the pass-1 `ExtractionResult` -- these ids are unique
    within one source, which is the scope of one reviewer call."""

    model_config = ConfigDict(extra="forbid")

    fact_id: str
    verdict: Verdict
    # `fix` only: the corrected field(s). A verdict of `fix` that changes
    # nothing (every fixed_* left null) is treated as `keep`.
    fixed_type: str | None = None
    fixed_claim: str | None = None
    fixed_quote: str | None = None
    # `reject` only: why. Free-text `reason` is kept alongside the closed
    # `reject_reason` for the report's "five examples" writeup even when
    # the model's free text doesn't fit one of the three closed buckets.
    reject_reason: RejectReason | None = None
    reason: str | None = None


class ReviewedMissedFact(BaseModel):
    """One fact the sweep found that pass 1's produced-facts list for
    this source didn't already cover. Same shape as an extractor's own
    fact (type/claim/quote/locator), no `id` -- combine assigns one.

    `entities` is one addition beyond the DESIGN brief's literal
    "type, claim, quote, locator" list, required for the same reason pass
    1's own `Fact.entities` exists: `portolan.bench.extraction`'s scorer
    (`Truth.entities_overlap`) rejects a match whose produced fact names
    no entity at all, so a sweep fact with no entities can essentially
    never be credited against a planted fact that has any -- which is
    nearly all of them. Optional (defaults to empty, same as a genuinely
    entity-less internal-only fact), populated by instructing the
    reviewer to use the same `registry_context` ids/mentions pass 1's own
    prompt already teaches it (`build_review_user_message` reuses
    `build_user_message`, which renders that same registry context)."""

    model_config = ConfigDict(extra="forbid")

    type: str
    claim: str
    quote: str | None = None
    locator: str
    entities: list[str] = Field(default_factory=list)


class ReviewResult(BaseModel):
    """The reviewer's whole reply for one source."""

    model_config = ConfigDict(extra="forbid")

    verdicts: list[FactVerdict] = Field(default_factory=list)
    missed: list[ReviewedMissedFact] = Field(default_factory=list)


class DroppedReviewItem(BaseModel):
    """One verdict or missed-fact item the lenient parser could not use --
    kept for diagnosis, mirroring `ExtractionResult.parse_lenient`'s
    `RejectedFact`: record it, never let one bad item sink the rest of
    the reviewer's reply."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["verdict", "missed"]
    index: int
    error: str
    raw: dict[str, Any] = Field(default_factory=dict)


def review_result_json_schema() -> dict[str, Any]:
    """Strict JSON schema for `ReviewResult` -- same contract as
    `portolan.extract.schema.extraction_result_json_schema`, handed to
    `OpenRouterClient.complete(json_schema=...)`."""
    return ReviewResult.model_json_schema()


def parse_review_lenient(
    data: dict[str, Any], *, valid_fact_ids: set[str] | None = None
) -> tuple[ReviewResult, list[DroppedReviewItem]]:
    """Best-effort parse of one reviewer reply: validate each `verdicts`
    and `missed` item ONE AT A TIME, dropping (recorded in the returned
    list, never raising) whichever ones don't validate. A verdict naming
    a `fact_id` pass 1 never produced for this source (`valid_fact_ids`,
    when given) is dropped too -- nothing downstream can apply a verdict
    to a fact that doesn't exist. A `missed` item whose `type` isn't one
    of the twelve base fact types is dropped the same way.
    """
    dropped: list[DroppedReviewItem] = []
    verdicts: list[FactVerdict] = []
    for i, raw in enumerate(data.get("verdicts") or []):
        if not isinstance(raw, dict):
            dropped.append(DroppedReviewItem(kind="verdict", index=i, error="not an object"))
            continue
        try:
            fv = FactVerdict.model_validate(raw)
        except ValidationError as exc:
            dropped.append(DroppedReviewItem(kind="verdict", index=i, error=str(exc), raw=raw))
            continue
        if valid_fact_ids is not None and fv.fact_id not in valid_fact_ids:
            dropped.append(
                DroppedReviewItem(
                    kind="verdict",
                    index=i,
                    error=f"unknown fact_id {fv.fact_id!r}",
                    raw=raw,
                )
            )
            continue
        verdicts.append(fv)

    missed: list[ReviewedMissedFact] = []
    for i, raw in enumerate(data.get("missed") or []):
        if not isinstance(raw, dict):
            dropped.append(DroppedReviewItem(kind="missed", index=i, error="not an object"))
            continue
        try:
            mf = ReviewedMissedFact.model_validate(raw)
        except ValidationError as exc:
            dropped.append(DroppedReviewItem(kind="missed", index=i, error=str(exc), raw=raw))
            continue
        if mf.type not in _FACT_TYPE_VALUES:
            dropped.append(
                DroppedReviewItem(
                    kind="missed", index=i, error=f"unknown type {mf.type!r}", raw=raw
                )
            )
            continue
        missed.append(mf)

    return ReviewResult(verdicts=verdicts, missed=missed), dropped


# --------------------------------------------------------------------------
# Prompt building -- reuses portolan.extract.prompt's renderers rather
# than re-implementing "source text + registry context + account context
# + vocabulary" a second time.
# --------------------------------------------------------------------------

_ROLE = (
    "You are Portolan's second-pass reviewer. You have already been given "
    "one source's text and a first extraction pass's produced facts for "
    "it. Do two jobs on this one source, in one reply:\n\n"
    "1. VERIFY: for every pass-1 fact listed below, decide keep, fix, or "
    "reject.\n"
    "   - keep: the fact is correct as written -- right type, the claim "
    "is true to the source, the quote (if any) really appears in the "
    "text.\n"
    "   - fix: the fact is basically right but has a wrong type, an "
    "inaccurate claim, or a quote that doesn't match the source verbatim "
    "-- give fixed_type/fixed_claim/fixed_quote for whichever field(s) "
    "are wrong (leave the others null).\n"
    "   - reject: the fact should not exist. Give reject_reason: "
    "not_in_source (nothing in the text supports this, even loosely), "
    "wrong_account (the fact is about a different account/entity than "
    "this source is actually about), or duplicate (this is the same fact "
    "as another pass-1 fact already listed, just phrased differently).\n"
    "   Every pass-1 fact_id listed below needs exactly one verdict.\n\n"
    "2. SWEEP: read the source again for facts pass 1's list is missing "
    "entirely. Go type by type -- the questions below name every base "
    "fact type in turn. Only report a genuinely new fact: don't repeat "
    "anything already in the pass-1 list, even reworded. Every missed "
    "fact needs type, claim, locator, entities, and a quote when this "
    "type requires one (see the type checklist below). `entities`: use "
    "the same registry context ids/mentions pass 1 was given below -- a "
    "fact naming no entity at all almost never scores as a match against "
    "a planted fact, so leaving this empty wastes a genuine find."
)

_VERDICT_RULES_REMINDER = (
    "## Fact rules, unchanged from pass 1\n"
    "The nine fact rules from pass 1 still apply to anything you fix or "
    "add: one claim per fact, durable language, no pronouns, a locator "
    "copied from a line you actually read, a verbatim quote (forty words "
    "or fewer) for complaint/praise/competitor_mention/commitment/"
    "pricing, and never invent an entity id."
)


def _type_checklist_block() -> str:
    lines = ["## Fact types (for the sweep and for fixing a wrong type)"]
    for type_name, info in FACT_TYPES.items():
        lines.append(f"- `{type_name.value}`: {info.definition} (e.g. {info.example})")
    return "\n".join(lines)


def _sweep_questions_block() -> str:
    lines = [
        "## Sweep questions -- answer each by re-reading the source, not "
        "by re-reading the pass-1 list"
    ]
    for type_name, info in FACT_TYPES.items():
        if type_name is FactType.OTHER:
            continue
        lines.append(
            f"- Any `{type_name.value}` facts ({info.definition.lower()}) not already listed?"
        )
    return "\n".join(lines)


def build_review_system_prompt(*, company_context: str | None = None) -> str:
    """The reviewer's system prompt: role/instructions, the company
    context block (reused from `portolan.context.render_company_context_
    block`, same renderer pass 1's own system prompt uses -- see
    `portolan.extract.prompt.build_system_prompt`), the fact-rule
    reminder, and the full type checklist. Stable across calls (no
    per-source data), same cacheability intent as
    `portolan.extract.prompt.build_system_prompt`.
    """
    sections: list[str] = [_ROLE]

    if company_context:
        from portolan.context import render_company_context_block

        sections.append(render_company_context_block(company_context))

    sections.append(_VERDICT_RULES_REMINDER)
    sections.append(_type_checklist_block())

    return "\n\n".join(sections)


class _Pass1FactLike(Protocol):
    """Structural shape this module needs from a pass-1 produced fact --
    matches `portolan.bench.extraction.ProducedFactRecord` (imported only
    under `TYPE_CHECKING` above, so this module never has to import that
    off-limits-to-edit module's sibling at runtime just to type-hint
    against it)."""

    id: str
    type: str
    claim: str
    locator: str
    quote: str | None


def _render_pass1_facts(facts: list[_Pass1FactLike] | list[ProducedFactRecord]) -> str:
    if not facts:
        return "(pass 1 produced no facts for this source)"
    lines = []
    for f in facts:
        quote_part = f' Quote: "{clean(f.quote)}"' if f.quote else " Quote: (none)"
        lines.append(
            f"- fact_id={f.id} type={f.type} locator={f.locator}: {clean(f.claim)}{quote_part}"
        )
    return "\n".join(lines)


def build_review_user_message(
    envelope: SourceEnvelope, pass1_facts: list[_Pass1FactLike] | list[ProducedFactRecord]
) -> str:
    """Per-call user message: the same source blob `build_user_message`
    renders for pass 1 (source metadata, participants, registry context,
    account context, vocabulary, locator-annotated text), plus the pass-1
    facts to verify and the sweep questions.
    """
    base = build_user_message(envelope)
    facts_block = _render_pass1_facts(pass1_facts)
    return (
        f"{base}\n\n"
        "## Pass 1 facts for this source (give exactly one verdict per fact_id)\n"
        f"{facts_block}\n\n"
        f"{_sweep_questions_block()}"
    )


# --------------------------------------------------------------------------
# One reviewer call
# --------------------------------------------------------------------------


class ReviewClient(Protocol):
    """Structurally `portolan.llm.LLMClient.complete` -- any object with a
    matching method works, including `OpenRouterClient` or a test fake."""

    def complete(
        self,
        *,
        system: str,
        user: str,
        json_schema: dict[str, Any] | None = None,
        model: str | None = None,
        surface: Surface,
        max_tokens: int = 4096,
    ) -> Any: ...


def review_source(
    client: ReviewClient,
    *,
    envelope: SourceEnvelope,
    pass1_facts: list[_Pass1FactLike] | list[ProducedFactRecord],
    system_prompt: str,
    model: str | None,
    max_tokens: int = 4096,
) -> tuple[ReviewResult, list[DroppedReviewItem], Any]:
    """Run the reviewer call for one source and parse its reply leniently.
    Returns `(review_result, dropped_items, raw_llm_result)` -- the raw
    result is kept so a caller can log tokens/cost/text for the run
    artifact without a second call.
    """
    user_message = build_review_user_message(envelope, pass1_facts)
    schema = review_result_json_schema()
    kwargs: dict[str, Any] = {
        "system": system_prompt,
        "user": user_message,
        "json_schema": schema,
        "model": model,
        "surface": "audit",
        "max_tokens": max_tokens,
    }
    # Same leniency plumbing as `portolan.extract.backends.
    # LLMClientBackend._complete`: only `OpenRouterClient` (of the two
    # clients this bench actually drives -- OpenRouter forks and the
    # local `AgentSDKClient` Haiku fork) declares `validate_json`, and
    # only that client's local jsonschema gate needs turning off, since
    # `parse_review_lenient` above is where this module's own leniency
    # decision belongs -- not a gate one layer earlier pre-empting it and
    # losing the whole source's review.
    if "validate_json" in inspect.signature(client.complete).parameters:
        kwargs["validate_json"] = False
    llm_result = client.complete(**kwargs)
    data = llm_result.data if llm_result.data is not None else {}
    valid_ids = {f.id for f in pass1_facts}
    review, dropped = parse_review_lenient(data, valid_fact_ids=valid_ids)
    return review, dropped, llm_result
