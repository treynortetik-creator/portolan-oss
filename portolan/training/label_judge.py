"""The label judge: an open-weight model re-reads a source and the silver facts
a teacher wrote for it, with the production prompt's own rules for what a fact
is (`portolan.extract.prompt.render_labeling_rules`), and answers five typed
questions per fact with a one-sentence reason; code turns the answers into
keep, drop or retype (`verdict_from_answers`).

Code applies the verdicts. A drop removes the fact; a retype changes only its
`fact_type`; a claim or quote is never edited (CLAUDE.md, training-data
provenance). A fact the judge gave no usable verdict for stays as it was and is
counted unresolved, so the record or document it belongs to is not trained on
as though it had been checked.

The judge's model must be on the training-set teacher allowlist
(`scripts/data/build_training_set.py`), since its verdicts decide what the
student learns; the builder checks `LabelAudit.judge_model`.

Calibration and the model choice: `01-spec/label-judge-2026-09-22.md`.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from portolan.extract.prompt import DEFAULT_PROMPT_VARIANT, render_labeling_rules
from portolan.extract.schema import QUOTE_REQUIRED_TYPES, FactType
from portolan.training.records import LabelAudit, TrainingFact, TrainingRecord

__all__ = [
    "DEFAULT_LABEL_JUDGE_MODEL",
    "JUDGE_PROMPT_VERSION",
    "JUDGE_SCHEMA",
    "LABEL_JUDGE_SPEND_KEY",
    "FactJudgement",
    "JudgeCall",
    "JudgeFact",
    "JudgeLogEntry",
    "JudgeResult",
    "judge_facts",
    "judge_key",
    "judge_model_allowed",
    "judge_record",
    "judge_system_prompt",
    "judge_user_message",
    "openrouter_judge_call",
    "parse_judge_reply",
    "retype_allowed",
    "verdict_from_answers",
]

LABEL_JUDGE_SPEND_KEY = "label_judge"
DEFAULT_LABEL_JUDGE_MODEL = "deepseek/deepseek-v4.1-flash"
JUDGE_PROMPT_VERSION = "label-judge-2026-09-22.3"

FACT_TYPES: tuple[str, ...] = tuple(t.value for t in FactType)
QUOTE_REQUIRED: frozenset[str] = frozenset(t.value for t in QUOTE_REQUIRED_TYPES)
MAX_QUOTE_WORDS = 40  # fact rule 5, as `portolan.training.repair` enforces it
JUDGE_MAX_TOKENS = 3000

Verdict = Literal["keep", "drop", "retype"]

JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "reason": {"type": "string"},
                    "stated_as_true": {"type": "boolean"},
                    "durable": {"type": "boolean"},
                    "claim_supported": {"type": "boolean"},
                    "repeats": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                    "best_type": {"type": "string", "enum": list(FACT_TYPES)},
                },
                "required": [
                    "id",
                    "reason",
                    "stated_as_true",
                    "durable",
                    "claim_supported",
                    "repeats",
                    "best_type",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["facts"],
    "additionalProperties": False,
}

_JUDGE_INSTRUCTIONS = "\n".join(
    [
        "You check facts that an extraction model wrote from one source. You get the "
        "source text and a numbered list of facts, each with its type, its claim and "
        "the quote it rests on. For every fact, answer five questions:",
        "",
        '- "stated_as_true": does a party to this source state it as true, about its '
        "own organization, the other party, or their dealings? false for a rumor, "
        "something heard second-hand about other companies or other customers, industry "
        "chatter, a hypothetical, a rhetorical question, or speculation.",
        '- "durable": does it clear the durability bar below (would someone still want '
        "to know it a month from now)? false for scheduling or logistics, small talk, a "
        "pleasantry, a courtesy, thanks, or a reassurance that adds no new information "
        '(a vendor saying "we value you" or "we are on it" is neither a decision nor a '
        "commitment), and for an acknowledgment with no deliverable or date.",
        '- "claim_supported": does the source support every part of the claim: who '
        "did or said it, what, every name and number? false when the claim names the "
        "wrong person or company, turns a request into a promise, or adds a detail. A "
        'date resolved from a relative one ("yesterday", "the 13th") against the '
        "source date counts as supported. Judge meaning, not quote length or wording.",
        '- "repeats": the id of an EARLIER fact in this list that states the same '
        "thing (same party, same point), or null. Two different findings, errors or "
        "requests are two facts, even when related.",
        '- "best_type": the type from the checklist below that fits the claim best, '
        "with the precedence rules applied. Check the definition: a statement that no "
        "harm occurred is not a complaint, and work already done is not a commitment.",
        "",
        'Write "reason" first: one short sentence on what the source says. Facts about '
        "the relationship's own parties count whether the customer or the vendor says "
        "them. You never rewrite a claim.",
    ]
)


def judge_system_prompt(
    company_context: str | None, *, variant: str = DEFAULT_PROMPT_VARIANT
) -> str:
    """Stable per (company context, variant): the judge's instructions, who the
    vendor is, then the production prompt's rules for what a fact is."""
    parts = [_JUDGE_INSTRUCTIONS]
    if company_context and company_context.strip():
        parts.append("## Company context\n" + company_context.strip())
    parts.append(
        "# The extraction rules the facts must follow\n\n" + render_labeling_rules(variant)
    )
    return "\n\n".join(parts)


@dataclass(frozen=True)
class JudgeFact:
    fact_type: str
    claim: str
    quote: str


def judge_user_message(
    text: str, facts: Sequence[JudgeFact], source_date: str | None = None
) -> str:
    """`source_date` (ISO) is what the extractor resolved relative dates
    against; without it a correctly resolved date reads as invented."""
    listed = [
        {"id": i, "fact_type": f.fact_type, "claim": f.claim, "quote": f.quote}
        for i, f in enumerate(facts)
    ]
    dated = f"Source date: {source_date}\n\n" if source_date else ""
    return (
        dated + "## Source text\n"
        f"{text}\n\n"
        "## Facts to check\n"
        f"{json.dumps(listed, ensure_ascii=False, indent=1)}"
    )


def judge_key(model: str, system: str, user: str) -> str:
    material = "\x1f".join([JUDGE_PROMPT_VERSION, model, system, user])
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]


@dataclass(frozen=True)
class FactJudgement:
    """`fact_type` is the type the fact carries after the verdict: its own
    for keep and drop, the new one for retype."""

    verdict: Verdict
    fact_type: str
    reason: str


def verdict_from_answers(row: dict[str, Any], own_type: str) -> tuple[Verdict, str] | None:
    """The verdict code derives from the judge's typed answers: drop when the
    fact is not stated as true, does not clear the durability bar, is not
    supported, or repeats an earlier fact; otherwise retype when the best type
    differs; otherwise keep. `None` when an answer is missing or malformed."""
    flags = [row.get(k) for k in ("stated_as_true", "durable", "claim_supported")]
    if not all(isinstance(v, bool) for v in flags):
        return None
    repeats = row.get("repeats")
    if repeats is not None and (isinstance(repeats, bool) or not isinstance(repeats, int)):
        return None
    best = row.get("best_type")
    if best not in FACT_TYPES:
        return None
    if not all(flags) or repeats is not None:
        return "drop", own_type
    if best != own_type:
        return "retype", best
    return "keep", own_type


def parse_judge_reply(
    reply: dict[str, Any] | None, facts: Sequence[JudgeFact]
) -> list[FactJudgement | None]:
    """One judgement per fact, in order; `None` where the reply has no usable
    answer for it (missing or repeated id, a malformed answer). A `repeats`
    that does not point at an earlier fact in the list is ignored."""
    out: list[FactJudgement | None] = [None] * len(facts)
    rows = reply.get("facts") if isinstance(reply, dict) else None
    if not isinstance(rows, list):
        return out
    seen: set[int] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        fid = row.get("id")
        if isinstance(fid, bool) or not isinstance(fid, int) or not 0 <= fid < len(facts):
            continue
        if fid in seen:
            out[fid] = None
            continue
        seen.add(fid)
        repeats = row.get("repeats")
        if isinstance(repeats, int) and not isinstance(repeats, bool) and not 0 <= repeats < fid:
            row = {**row, "repeats": None}
        derived = verdict_from_answers(row, facts[fid].fact_type)
        if derived is not None:
            reason = str(row.get("reason") or "").strip()
            out[fid] = FactJudgement(derived[0], derived[1], reason)
    return out


def judge_model_allowed(model: str) -> bool:
    """On the training-set builder's allowlist (`scripts/data/
    build_training_set.py` `teacher_allowed`): the judge decides which labels
    ship, so it must be a model labels may come from."""
    import importlib.util
    import sys
    from pathlib import Path

    name = "build_training_set_allowlist"
    module = sys.modules.get(name)
    if module is None:
        path = Path(__file__).resolve().parents[2] / "scripts" / "data" / "build_training_set.py"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            # A failed load must not leave an empty module cached for the next call.
            del sys.modules[name]
            raise
    return bool(module.teacher_allowed(model))


def retype_allowed(new_type: str, quote: str | None) -> bool:
    """A retype must still pass fact rule 5: a quote-required type needs a
    quote of forty words or fewer."""
    if new_type not in QUOTE_REQUIRED:
        return True
    return bool(quote) and len(quote.split()) <= MAX_QUOTE_WORDS


# `(model, system, user) -> (parsed reply or None, cost in USD)`
JudgeCall = Callable[[str, str, str], tuple[dict[str, Any] | None, float]]


@dataclass
class JudgeResult:
    model: str
    key: str
    judgements: list[FactJudgement | None] = field(default_factory=list)
    cost_usd: float = 0.0
    error: str | None = None

    @property
    def complete(self) -> bool:
        return self.error is None and all(j is not None for j in self.judgements)


def judge_facts(
    text: str,
    facts: Sequence[JudgeFact],
    *,
    model: str,
    call: JudgeCall | None,
    company_context: str | None,
    source_date: str | None = None,
    variant: str = DEFAULT_PROMPT_VARIANT,
    cache: dict[str, JudgeLogEntry] | None = None,
    budget_left: Callable[[], bool] = lambda: True,
) -> tuple[JudgeResult, str]:
    """`(result, how)` for every fact of one source: `how` is "cached" (a
    logged reply for the same model, prompt and input), "called", "skipped"
    (no call allowed: no client, or the budget is spent) or "empty". A failed
    call returns no judgements and the error, never raises."""
    system = judge_system_prompt(company_context, variant=variant)
    user = judge_user_message(text, facts, source_date)
    key = judge_key(model, system, user)
    if not facts:
        return JudgeResult(model=model, key=key), "empty"
    hit = (cache or {}).get(key)
    if hit is not None and hit.error is None:
        return hit.to_result(), "cached"
    none_yet: list[FactJudgement | None] = [None] * len(facts)
    if call is None or not budget_left():
        return JudgeResult(model, key, none_yet, error="not_attempted"), "skipped"
    try:
        reply, cost = call(model, system, user)
    except Exception as exc:  # noqa: BLE001 -- logged; the facts stay without a verdict
        error = f"{type(exc).__name__}: {exc}"[:300]
        return JudgeResult(model, key, none_yet, error=error), "called"
    judgements = parse_judge_reply(reply, facts)
    error = None if reply is not None else "no parsed reply"
    return JudgeResult(model, key, judgements, cost_usd=cost, error=error), "called"


def judge_record(record: TrainingRecord, result: JudgeResult) -> TrainingRecord:
    """The record after its judge verdicts: dropped facts removed, retyped
    facts with only `fact_type` changed (the teacher's type kept as
    `teacher_fact_type`). A fact with no usable verdict, or a retype fact
    rule 5 refuses, stays as it was and counts as `judge_unresolved`, which
    makes the record incomplete."""
    facts: list[TrainingFact] = []
    dropped = retyped = open_facts = 0
    judgements = result.judgements or [None] * len(record.facts)
    for fact, judgement in zip(record.facts, judgements, strict=True):
        if judgement is None:
            open_facts += 1
            facts.append(fact)
        elif judgement.verdict == "drop":
            dropped += 1
        elif judgement.verdict == "retype" and retype_allowed(judgement.fact_type, fact.quote):
            retyped += 1
            facts.append(
                fact.model_copy(
                    update={"fact_type": judgement.fact_type, "teacher_fact_type": fact.fact_type}
                )
            )
        elif judgement.verdict == "retype":
            open_facts += 1
            facts.append(fact)
        else:
            facts.append(fact)
    audit = record.label_audit or LabelAudit(
        status="complete",
        teacher_facts=len(record.facts),
        kept=len(record.facts),
        regated=0,
        repaired=0,
        withdrawn=0,
        unresolved=0,
        unlogged=0,
        audited="",
    )
    audit = audit.model_copy(
        update={
            "judge_model": result.model,
            "judge_dropped": dropped,
            "judge_retyped": retyped,
            "judge_unresolved": open_facts,
            "status": "incomplete" if open_facts else audit.status,
        }
    )
    return record.model_copy(update={"facts": facts, "label_audit": audit})


class JudgeLogEntry(BaseModel):
    """One judge call and its verdicts; also the cache a rerun reads instead
    of paying again (keyed by model, prompt version, system and user text)."""

    model_config = ConfigDict(extra="forbid")

    key: str
    model: str
    prompt_version: str
    subject: str  # a record id, or a v2 plan id and source id
    facts: list[dict[str, Any]]
    verdicts: list[dict[str, Any] | None]
    cost_usd: float = 0.0
    error: str | None = None
    created: str

    @classmethod
    def from_result(
        cls, result: JudgeResult, *, subject: str, facts: Sequence[JudgeFact], created: str
    ) -> JudgeLogEntry:
        return cls(
            key=result.key,
            model=result.model,
            prompt_version=JUDGE_PROMPT_VERSION,
            subject=subject,
            facts=[{"fact_type": f.fact_type, "claim": f.claim, "quote": f.quote} for f in facts],
            verdicts=[
                None
                if j is None
                else {"verdict": j.verdict, "fact_type": j.fact_type, "reason": j.reason}
                for j in result.judgements
            ],
            cost_usd=result.cost_usd,
            error=result.error,
            created=created,
        )

    def to_result(self) -> JudgeResult:
        return JudgeResult(
            model=self.model,
            key=self.key,
            judgements=[
                None if v is None else FactJudgement(v["verdict"], v["fact_type"], v["reason"])
                for v in self.verdicts
            ],
            cost_usd=0.0,
            error=self.error,
        )


def openrouter_judge_call(
    api_key: str, spend: Any, *, max_tokens: int = JUDGE_MAX_TOKENS
) -> JudgeCall:
    """One `OpenRouterClient` per thread, metered in `spend` (a
    `portolan.training.spend.SpendMeter`) under `label_judge`; the call
    returns the sink's cost for the log."""
    from portolan.llm.openrouter import OpenRouterClient
    from portolan.metering import InMemorySink

    local = threading.local()

    def call(model: str, system: str, user: str) -> tuple[dict[str, Any] | None, float]:
        client = getattr(local, "client", None)
        if client is None:
            client = OpenRouterClient(
                spend_key=LABEL_JUDGE_SPEND_KEY,
                spend_meter=spend,
                api_key=api_key,
                default_model=model,
                sink=InMemorySink(),
                max_retries=3,
            )
            local.client = client
        before = len(client.sink.events)
        result = client.complete(
            system=system,
            user=user,
            json_schema=JUDGE_SCHEMA,
            model=model,
            surface="audit",
            max_tokens=max_tokens,
        )
        cost = sum((e.cost_estimate or 0.0) for e in client.sink.events[before:])
        return result.data, cost

    return call
