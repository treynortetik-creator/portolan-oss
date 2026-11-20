"""Repair pass for silver-labeled facts the keep gate rejected.

Every fact a teacher emitted for a record ends as one of:

- kept: kept at label time and still passes the gate;
- regated: dropped at label time, passes the gate as it is now;
- repaired: one call to the same teacher, given the record text, the
  rejected fact and the gate's reason, returned a corrected fact that
  passes the full gate;
- withdrawn: that call withdrew the fact;
- unresolved: the corrected fact failed again, or the call failed or was
  not made (budget).

A drop that was never logged (records labeled before `dropped.jsonl`
existed, or a run stopped mid-record) cannot be repaired and counts as
`unlogged`. A record's `LabelAudit` is `complete` only with no unresolved
and no unlogged facts; `scripts/data/build_training_set.py` trains only on
complete records.

The teacher's words only: a repaired claim is exactly what the teacher
returned, and its quote goes through the same exact-span resolution as a
live run (`resolve_quote_span`). Code never rewrites a claim.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, get_args

from pydantic import BaseModel, ConfigDict

from portolan.extract.quote_gate import quote_is_supported
from portolan.extract.schema import QUOTE_REQUIRED_TYPES, FactType, Severity
from portolan.training.overlap_gate import (
    overlap_gate_reason,
    unsupported_names,
    unsupported_numerals,
)
from portolan.training.quote_span import resolve_quote_span
from portolan.training.records import (
    TEACHER_FACT_FIELDS,
    DroppedFact,
    LabelAudit,
    TrainingFact,
    TrainingRecord,
)

__all__ = [
    "GATE_THRESHOLD",
    "REPAIR_SCHEMA",
    "GateVerdict",
    "gate_fact",
    "DropMapping",
    "map_drops_to_records",
    "EnvelopeLookup",
    "recover_unwritten_records",
    "FailedFact",
    "RecordPlan",
    "plan_record",
    "build_repair_messages",
    "RepairLogEntry",
    "repair_key",
    "FINAL_OUTCOMES",
    "judge_reply",
    "RepairCall",
    "repair_fact",
    "assemble_record",
]

# The content-overlap rule is off in live labeling (`scripts/data/
# silver_label.py` DEFAULT_OVERLAP_THRESHOLD); the repair pass gates the same way.
GATE_THRESHOLD = 0.0

BACKFILL_REASON = "quote_not_exact_backfill"

FACT_TYPES: tuple[str, ...] = tuple(t.value for t in FactType)
SEVERITIES: tuple[str, ...] = get_args(Severity)
QUOTE_REQUIRED: frozenset[str] = frozenset(t.value for t in QUOTE_REQUIRED_TYPES)
MAX_QUOTE_WORDS = 40  # fact rule 5, `portolan.extract.schema.Fact`

_NULLABLE_STRING = {"anyOf": [{"type": "string"}, {"type": "null"}]}

# The corrected fact carries the same fields a labeled fact does, so the SFT
# formatter (`portolan.training.sft_format`) gets the teacher's own values.
REPAIR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["correct", "withdraw"]},
        "reason": {"type": "string"},
        "fact_type": {"type": "string", "enum": list(FACT_TYPES)},
        "claim": {"type": "string"},
        "quote": {"type": "string"},
        "locator": {"type": "string"},
        "entities": {"type": "array", "items": {"type": "string"}},
        "sentiment": {"type": "integer", "minimum": -2, "maximum": 2},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "severity": {"anyOf": [{"type": "string", "enum": list(SEVERITIES)}, {"type": "null"}]},
        "speaker": _NULLABLE_STRING,
        "valid_from": _NULLABLE_STRING,
        "valid_to": _NULLABLE_STRING,
    },
    "required": [
        "decision",
        "reason",
        "fact_type",
        "claim",
        "quote",
        "locator",
        "entities",
        "sentiment",
        "confidence",
        "severity",
        "speaker",
        "valid_from",
        "valid_to",
    ],
    "additionalProperties": False,
}


# --------------------------------------------------------------------------
# The gate, with the detail a repair prompt needs
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GateVerdict:
    """`reason` is None when the fact passes; then `quote`/`span` are the
    exact span of the text the quote resolved to."""

    reason: str | None
    detail: tuple[str, ...] = ()
    quote: str | None = None
    span: tuple[int, int] | None = None


def gate_fact(
    claim: str, quote: str | None, text: str, fact_type: str | None = None
) -> GateVerdict:
    """The live keep gate (`run_corpus_single`): overlap gate, then an
    exact quote span; then, given `fact_type`, fact rule 5's forty-word
    limit, which the SFT formatter's strict `ExtractionResult` enforces."""
    reason = overlap_gate_reason(claim=claim, quote=quote, doc_text=text, threshold=GATE_THRESHOLD)
    if reason == "overlap_gate_numeral":
        return GateVerdict(reason, tuple(unsupported_numerals(claim, text)))
    if reason == "overlap_gate_name":
        return GateVerdict(reason, tuple(unsupported_names(claim, text)))
    if reason is not None:
        return GateVerdict(reason)
    resolved = resolve_quote_span(quote, text)
    if resolved is None:
        return GateVerdict("quote_not_exact")
    words = len(resolved.quote.split())
    if fact_type in QUOTE_REQUIRED and words > MAX_QUOTE_WORDS:
        return GateVerdict("quote_too_long", (str(words),))
    return GateVerdict(None, (), resolved.quote, resolved.span)


# --------------------------------------------------------------------------
# Which record each logged drop belongs to
# --------------------------------------------------------------------------


@dataclass
class DropMapping:
    by_record: dict[str, list[DroppedFact]] = field(default_factory=dict)
    # drops whose record was never written (a negative capped at label
    # time) or is not in this snapshot yet
    orphans: list[DroppedFact] = field(default_factory=list)


def _single_teacher(record: TrainingRecord) -> str | None:
    return record.teachers[0] if len(record.teachers) == 1 else None


def map_drops_to_records(
    records: Sequence[TrainingRecord], drops: Iterable[DroppedFact]
) -> DropMapping:
    """A live run logs an envelope's drops just before it writes that
    envelope's record, so a label-time drop belongs to the first record of
    the same source and teacher written at or after it, provided that
    record's text supports the drop's quote (a quote-gate drop has no
    supported quote to check). A backfill drop was removed from an existing
    record later, so it belongs to the record whose text holds its quote."""
    groups: dict[tuple[str, str], list[TrainingRecord]] = defaultdict(list)
    for record in records:
        teacher = _single_teacher(record)
        if teacher is not None:
            groups[(record.source_id, teacher)].append(record)
    for group in groups.values():
        group.sort(key=lambda r: datetime.fromisoformat(r.created))

    mapping = DropMapping()
    for drop in drops:
        candidates = groups.get((drop.source_id, drop.teacher), [])
        quote = drop.fact.get("quote")
        match: TrainingRecord | None = None
        if drop.reason == BACKFILL_REASON:
            match = next((r for r in candidates if quote_is_supported(quote, r.text)), None)
        else:
            dropped_at = datetime.fromisoformat(drop.created)
            match = next(
                (r for r in candidates if datetime.fromisoformat(r.created) >= dropped_at), None
            )
            if (
                match is not None
                and drop.reason != "quote_gate"
                and not quote_is_supported(quote, match.text)
            ):
                match = None
        if match is None:
            mapping.orphans.append(drop)
        else:
            mapping.by_record.setdefault(match.record_id, []).append(drop)
    return mapping


# (record_id, text) for each envelope of a source, as the live run built them
EnvelopeLookup = Callable[[str], list[tuple[str, str]]]


def recover_unwritten_records(
    orphans: Sequence[DroppedFact],
    *,
    processed_sources: set[str],
    existing_record_ids: set[str],
    envelopes_for: EnvelopeLookup,
    corpus: str,
    kind: str,
    license_text: str,
) -> tuple[list[tuple[TrainingRecord, list[DroppedFact]]], list[DroppedFact]]:
    """`(records, still_orphaned)`. An envelope of a finished document with
    no record was a negative dropped by the label-time negative cap; when
    the gate made it negative, every teacher fact it had is in the drop log.
    Such an envelope becomes a record with no kept facts and those drops.
    A source is recovered only when each of its drops lands on exactly one
    unwritten envelope (by quote; a quote-gate drop needs a single
    candidate)."""
    by_source: dict[str, list[DroppedFact]] = defaultdict(list)
    left: list[DroppedFact] = []
    for drop in orphans:
        if drop.reason == BACKFILL_REASON or drop.source_id not in processed_sources:
            left.append(drop)
        else:
            by_source[drop.source_id].append(drop)

    recovered: list[tuple[TrainingRecord, list[DroppedFact]]] = []
    for source_id, drops in by_source.items():
        candidates = [
            (rid, text) for rid, text in envelopes_for(source_id) if rid not in existing_record_ids
        ]
        placed: dict[str, list[DroppedFact]] = defaultdict(list)
        ok = bool(candidates)
        for drop in drops:
            if drop.reason == "quote_gate":
                hits = candidates if len(candidates) == 1 else []
            else:
                hits = [c for c in candidates if quote_is_supported(drop.fact.get("quote"), c[1])]
            if len({d.teacher for d in drops}) != 1 or not hits:
                ok = False
                break
            placed[hits[0][0]].append(drop)
        if not ok:
            left.extend(drops)
            continue
        texts = dict(candidates)
        for record_id, envelope_drops in placed.items():
            record = TrainingRecord(
                record_id=record_id,
                corpus=corpus,
                source_id=source_id,
                kind=kind,
                license=license_text,
                text=texts[record_id],
                facts=[],
                teachers=[envelope_drops[0].teacher],
                agreement=0.0,
                created=max(d.created for d in envelope_drops),
                mode="single",
            )
            recovered.append((record, envelope_drops))
    return recovered, left


# --------------------------------------------------------------------------
# One record's plan: what passes now, what needs a repair call
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FailedFact:
    """A teacher fact that fails the gate: a label-time drop (`origin`
    "dropped") or a kept fact the fixed gate now rejects ("kept")."""

    record_id: str
    source_id: str
    teacher: str
    origin: str
    fact: dict[str, Any]
    verdict: GateVerdict


@dataclass
class RecordPlan:
    record: TrainingRecord
    passing: list[TrainingFact]
    kept: int
    regated: int
    failed: list[FailedFact]
    unlogged: int
    teacher_facts: int | None


def _raw_fact(fact: TrainingFact) -> dict[str, Any]:
    return {
        "fact_type": fact.fact_type,
        "claim": fact.claim,
        "quote": fact.quote,
        "locator": fact.locator,
        "entities": list(fact.entities),
        **{k: getattr(fact, k) for k in TEACHER_FACT_FIELDS},
    }


def _expected_label_time_drops(record: TrainingRecord, backfill_drops: int) -> int | None:
    """Facts the gate dropped while labeling, from `agreement` (kept /
    emitted then). None when agreement is 0: the count is unknown."""
    if record.agreement >= 1.0:
        return 0
    if record.agreement <= 0.0:
        return None
    kept_then = sum(1 for f in record.facts if not f.regated) + backfill_drops
    return max(0, round(kept_then / record.agreement) - kept_then)


def plan_record(record: TrainingRecord, drops: Sequence[DroppedFact]) -> RecordPlan:
    teacher = _single_teacher(record) or ",".join(record.teachers)
    passing: list[TrainingFact] = []
    failed: list[FailedFact] = []
    kept = regated = 0

    for fact in record.facts:
        verdict = gate_fact(fact.claim, fact.quote, record.text, fact.fact_type)
        if verdict.reason is None:
            span_ok = fact.quote_span is not None and (
                record.text[fact.quote_span[0] : fact.quote_span[1]] == fact.quote
            )
            passing.append(
                fact
                if span_ok
                else fact.model_copy(update={"quote": verdict.quote, "quote_span": verdict.span})
            )
            if fact.regated:
                regated += 1
            else:
                kept += 1
        else:
            failed.append(
                FailedFact(
                    record.record_id, record.source_id, teacher, "kept", _raw_fact(fact), verdict
                )
            )

    readmitted_claims = {f.claim for f in record.facts if f.regated}
    backfill = [d for d in drops if d.reason == BACKFILL_REASON]
    live = [d for d in drops if d.reason != BACKFILL_REASON]
    # a drop an earlier regate already put back into this record
    open_live = [d for d in live if d.fact.get("claim") not in readmitted_claims]

    for drop in [*backfill, *open_live]:
        raw = {
            "fact_type": drop.fact.get("fact_type", "other"),
            "claim": drop.fact.get("claim", ""),
            "quote": drop.fact.get("quote"),
            "locator": drop.fact.get("locator", ""),
            "entities": list(drop.fact.get("entities") or []),
            **{k: drop.fact[k] for k in TEACHER_FACT_FIELDS if k in drop.fact},
        }
        verdict = gate_fact(raw["claim"], raw["quote"], record.text, raw["fact_type"])
        if verdict.reason is None:
            passing.append(
                TrainingFact(
                    fact_type=raw["fact_type"],
                    claim=raw["claim"],
                    quote=verdict.quote,
                    quote_span=verdict.span,
                    locator=raw["locator"],
                    entities=raw["entities"],
                    regated=True,
                    **{k: raw.get(k) for k in TEACHER_FACT_FIELDS},
                )
            )
            regated += 1
        else:
            failed.append(
                FailedFact(
                    record.record_id, record.source_id, drop.teacher, "dropped", raw, verdict
                )
            )

    expected = _expected_label_time_drops(record, len(backfill))
    unlogged = (0 if live else 1) if expected is None else max(0, expected - len(live))
    known = len(record.facts) + len(backfill) + len(open_live)
    teacher_facts = None if expected is None and unlogged else known + unlogged
    return RecordPlan(
        record=record,
        passing=passing,
        kept=kept,
        regated=regated,
        failed=failed,
        unlogged=unlogged,
        teacher_facts=teacher_facts,
    )


# --------------------------------------------------------------------------
# The repair call
# --------------------------------------------------------------------------

_REPAIR_SYSTEM = "\n".join(
    [
        "You are Portolan's extraction engine. Earlier you extracted a fact from the source "
        "text below, and a mechanical check rejected it. Decide whether the fact can be "
        "stated faithfully from the source text alone.",
        "",
        "Return one JSON object:",
        '- "decision": "correct" when you can restate the fact so that it passes the check, '
        'or "withdraw" when the source text does not support it.',
        '- "reason": one short sentence explaining the decision.',
        '- "fact_type", "claim", "quote", "locator", "entities": the corrected fact. When '
        "you withdraw, return the original fact's values unchanged.",
        "",
        "Rules for a corrected fact:",
        '1. "quote" is copied character for character from the source text: one contiguous '
        'span, forty words or fewer, no "..." joining two places, no changed, added or '
        "dropped words.",
        '2. Every number, date and proper name in "claim" appears in the source text. '
        "Anything you knew from outside the text (the date a message was sent, a "
        "participant list, a full name the text never uses, general knowledge) cannot be "
        "used: leave it out, or withdraw if the fact depends on it.",
        "3. One claim per fact. No pronouns: name the entity every time.",
        '4. "fact_type" is one of: {fact_types}. Keep the original type unless it was wrong.',
        '5. "locator" is the locator in square brackets at the start of the line or '
        "paragraph the quote comes from, without the brackets.",
        '6. "entities" lists names exactly as the source text writes them.',
        '7. "sentiment" (-2 to 2), "confidence" (0 to 1), "severity" ("low", "medium", '
        '"high" or null), "speaker", "valid_from" and "valid_to" (ISO dates or null) are '
        "your own values for the corrected fact.{company_context}",
    ]
)

_REASON_TEXT = {
    "quote_gate": "The quote does not appear in the source text.",
    "quote_not_exact": (
        "The quote is close to the source text but is not an exact copy of one contiguous span."
    ),
    BACKFILL_REASON: (
        "The quote is close to the source text but is not an exact copy of one contiguous span."
    ),
    "overlap_gate_content": "The claim's wording is not supported by the source text.",
}


def _reason_text(verdict: GateVerdict) -> str:
    if verdict.reason == "overlap_gate_numeral":
        return (
            "These numbers or dates in the claim do not appear in the source text: "
            + ", ".join(f'"{d}"' for d in verdict.detail)
            + "."
        )
    if verdict.reason == "overlap_gate_name":
        return (
            "These names in the claim do not appear in the source text: "
            + ", ".join(f'"{d}"' for d in verdict.detail)
            + "."
        )
    if verdict.reason == "quote_too_long":
        words = verdict.detail[0] if verdict.detail else "more than forty"
        return f"The quote is {words} words long; this fact type allows forty words or fewer."
    return _REASON_TEXT.get(verdict.reason or "", "The fact failed the check.")


def build_repair_messages(
    *, text: str, fact: dict[str, Any], verdict: GateVerdict, company_context: str | None
) -> tuple[str, str]:
    """`(system, user)` for one repair call."""
    context = f"\n\n## Company context\n{company_context.strip()}" if company_context else ""
    system = _REPAIR_SYSTEM.format(
        fact_types=", ".join(f'"{t}"' for t in FACT_TYPES), company_context=context
    )
    keys = ("fact_type", "claim", "quote", "locator", "entities", *sorted(TEACHER_FACT_FIELDS))
    rejected = {k: fact.get(k) for k in keys if k in fact}
    user = (
        "## Source text\n"
        f"{text}\n\n"
        "## Rejected fact\n"
        f"{json.dumps(rejected, ensure_ascii=False, indent=2)}\n\n"
        "## Why it was rejected\n"
        f"{_reason_text(verdict)}"
    )
    return system, user


class RepairLogEntry(BaseModel):
    """One line of `repairs.jsonl`: a repair call and what became of it.
    Also the cache that keeps a rerun from paying twice."""

    model_config = ConfigDict(extra="forbid")

    key: str
    corpus: str
    record_id: str
    source_id: str
    teacher: str
    origin: str
    original: dict[str, Any]
    gate_reason: str
    gate_detail: list[str]
    decision: str | None = None
    teacher_reason: str | None = None
    returned: dict[str, Any] | None = None
    # repaired | withdrawn | still_failing | invalid_reply | call_failed | not_attempted
    outcome: str
    outcome_reason: str | None = None
    cost_usd: float = 0.0
    created: str


# Outcomes whose teacher reply is final; the others are retried on a rerun.
FINAL_OUTCOMES = frozenset({"repaired", "withdrawn", "still_failing", "invalid_reply"})


def repair_key(record_id: str, fact: dict[str, Any]) -> str:
    material = "\x1f".join([record_id, fact.get("claim") or "", fact.get("quote") or ""])
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]


def judge_reply(
    reply: dict[str, Any] | None, text: str
) -> tuple[str, str | None, TrainingFact | None]:
    """`(outcome, outcome_reason, fact)` for a teacher's reply: the
    corrected fact runs through the full gate; nothing in it is edited
    except that the quote becomes the exact span it resolved to."""
    if not isinstance(reply, dict) or reply.get("decision") not in ("correct", "withdraw"):
        return "invalid_reply", "no decision", None
    if reply["decision"] == "withdraw":
        return "withdrawn", None, None
    claim = reply.get("claim")
    quote = reply.get("quote")
    fact_type = reply.get("fact_type")
    if not isinstance(claim, str) or not claim.strip():
        return "invalid_reply", "empty claim", None
    if fact_type not in FACT_TYPES:
        return "invalid_reply", f"fact_type {fact_type!r}", None
    values = _teacher_values(reply)
    if isinstance(values, str):
        return "invalid_reply", values, None
    verdict = gate_fact(claim, quote if isinstance(quote, str) else None, text, fact_type)
    if verdict.reason is not None:
        return "still_failing", verdict.reason, None
    locator = reply.get("locator")
    # the SFT formatter needs the locator as it appears in the text
    if not isinstance(locator, str) or f"[{locator}]" not in text:
        return "still_failing", "locator_not_in_text", None
    entities = reply.get("entities")
    return (
        "repaired",
        None,
        TrainingFact(
            fact_type=fact_type,
            claim=claim,
            quote=verdict.quote,
            quote_span=verdict.span,
            locator=locator,
            entities=[e for e in entities if isinstance(e, str)]
            if isinstance(entities, list)
            else [],
            repaired=True,
            **values,
        ),
    )


def _teacher_values(reply: dict[str, Any]) -> dict[str, Any] | str:
    """The reply's `TEACHER_FACT_FIELDS`, or why they are unusable. An old
    reply without them yields `None`s, which the SFT formatter refuses."""
    sentiment = reply.get("sentiment")
    confidence = reply.get("confidence")
    if sentiment is not None and (
        isinstance(sentiment, bool) or not isinstance(sentiment, int) or not -2 <= sentiment <= 2
    ):
        return f"sentiment {sentiment!r}"
    if confidence is not None and (
        isinstance(confidence, bool)
        or not isinstance(confidence, int | float)
        or not 0 <= confidence <= 1
    ):
        return f"confidence {confidence!r}"
    severity = reply.get("severity")
    if severity is not None and severity not in SEVERITIES:
        return f"severity {severity!r}"
    values: dict[str, Any] = {
        "sentiment": sentiment,
        "confidence": None if confidence is None else float(confidence),
        "severity": severity,
    }
    for key in ("speaker", "valid_from", "valid_to"):
        value = reply.get(key)
        values[key] = value if isinstance(value, str) and value.strip() else None
    for key in ("valid_from", "valid_to"):
        if values[key] is not None:
            try:
                date.fromisoformat(values[key])
            except ValueError:
                return f"{key} {values[key]!r}"
    return values


# `(model, system, user) -> (parsed reply or None, cost in USD)`
RepairCall = Callable[[str, str, str], tuple[dict[str, Any] | None, float]]


def repair_fact(
    failed: FailedFact,
    *,
    corpus: str,
    text: str,
    company_context: str | None,
    call: RepairCall | None,
    now: str,
) -> RepairLogEntry:
    """One repair call for `failed`, or a `not_attempted` entry when `call`
    is None (budget spent, or a dry run)."""
    base = {
        "key": repair_key(failed.record_id, failed.fact),
        "corpus": corpus,
        "record_id": failed.record_id,
        "source_id": failed.source_id,
        "teacher": failed.teacher,
        "origin": failed.origin,
        "original": failed.fact,
        "gate_reason": failed.verdict.reason or "",
        "gate_detail": list(failed.verdict.detail),
        "created": now,
    }
    if call is None:
        return RepairLogEntry(**base, outcome="not_attempted")
    system, user = build_repair_messages(
        text=text, fact=failed.fact, verdict=failed.verdict, company_context=company_context
    )
    try:
        reply, cost = call(failed.teacher, system, user)
    except Exception as exc:  # noqa: BLE001 - logged as call_failed and retried on a rerun
        return RepairLogEntry(
            **base, outcome="call_failed", outcome_reason=f"{type(exc).__name__}: {exc}"[:300]
        )
    outcome, outcome_reason, _fact = judge_reply(reply, text)
    return RepairLogEntry(
        **base,
        decision=reply.get("decision") if isinstance(reply, dict) else None,
        teacher_reason=reply.get("reason") if isinstance(reply, dict) else None,
        returned=reply if isinstance(reply, dict) else None,
        outcome=outcome,
        outcome_reason=outcome_reason,
        cost_usd=cost,
    )


def assemble_record(
    plan: RecordPlan, entries: Sequence[RepairLogEntry], *, audited: str
) -> TrainingRecord:
    """The record with every passing and repaired fact, and its audit.
    `entries` holds one log entry per failed fact; every corrected fact is
    judged again here against the record text, so a cached reply follows
    the gate as it is now. A fact identical to one already in the record
    (same claim, same quote span) is counted but not added twice."""
    facts: list[TrainingFact] = []
    seen: set[tuple[str, tuple[int, int] | None]] = set()

    def add(fact: TrainingFact) -> None:
        key = (" ".join(fact.claim.lower().split()), fact.quote_span)
        if key not in seen:
            seen.add(key)
            facts.append(fact)

    for fact in plan.passing:
        add(fact)
    repaired = withdrawn = unresolved = 0
    for entry in entries:
        if entry.outcome == "withdrawn":
            withdrawn += 1
            continue
        if entry.decision == "correct":
            outcome, _reason, fact = judge_reply(entry.returned, plan.record.text)
            if outcome == "repaired" and fact is not None:
                add(fact)
                repaired += 1
                continue
        unresolved += 1
    unresolved += max(0, len(plan.failed) - len(entries))
    complete = unresolved == 0 and plan.unlogged == 0
    audit = LabelAudit(
        status="complete" if complete else "incomplete",
        teacher_facts=plan.teacher_facts,
        kept=plan.kept,
        regated=plan.regated,
        repaired=repaired,
        withdrawn=withdrawn,
        unresolved=unresolved,
        unlogged=plan.unlogged,
        audited=audited,
    )
    return plan.record.model_copy(update={"facts": facts, "label_audit": audit})
