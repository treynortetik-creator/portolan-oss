"""The one SFT formatter for the extraction student: turns a silver
`TrainingRecord` into the `[system, user, assistant]` chat messages the
student is trained on.

- **system**: `build_system_prompt(extension, company_context=...,
  variant=...)`, called exactly as `portolan.pipeline.run()` calls it. When
  the record carries the sha256 of the prompt its teacher saw, the rebuilt
  prompt must hash to the same value or the call raises.
- **user**: `build_user_message(envelope)`, the production function, on
  the envelope stored with the record (`user_message_inputs` plus `text`).
- **assistant**: the teacher's own answer in the exact JSON shape the
  model-facing schema defines (`extraction_result_json_schema()`), strict-
  valid, so it parses through `LLMClientBackend.extract` on the first
  attempt with no retry and no `parse_lenient` coercion
  (`tests/test_sft_format.py` pins that).

Fields a record does not carry are never filled with placeholders: a
record without the teacher's summary, sentiment and confidence raises
`LegacyRecordError` instead. Records written before 2026-09-22 are all in
that state and need relabeling.

Entity values follow the production prompt's rule ("the `registry_id`
from `registry_context` ... or the exact mention string ... never invent a
registry id"): a value is kept only when it is a registry id the prompt
shows, or a name that appears in the prompt's participants or source text.
Anything else is dropped from the answer, never rewritten.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from portolan.extract.prompt import (
    build_system_prompt,
    build_user_message,
    system_prompt_sha256,
)
from portolan.extract.schema import (
    SCHEMA_VERSION,
    CustomerExtension,
    ExtractionResult,
    SourceEnvelope,
)
from portolan.training.records import TeacherAnswer, TrainingRecord

__all__ = [
    "LegacyRecordError",
    "SystemPromptMismatchError",
    "envelope_for_record",
    "training_answer",
    "answer_json",
    "production_sft_messages",
    "production_sft_messages_for_record",
]

# Fact rule 9: a speaker is a registry contact id or one of these roles.
_SPEAKER_ROLES = frozenset({"customer", "our-rep", "unknown"})


class LegacyRecordError(ValueError):
    """The record lacks the prompt stamp or the teacher's own answer
    fields, so it cannot be formatted without inventing values."""


class SystemPromptMismatchError(ValueError):
    """The rebuilt system prompt does not hash to the one the record's
    teacher saw: wrong company context, variant, or extension."""


def envelope_for_record(record: TrainingRecord) -> SourceEnvelope:
    """The exact envelope the teacher was given: `user_message_inputs`
    plus the record's `text`."""
    if record.user_message_inputs is None:
        raise LegacyRecordError(f"{record.record_id}: record has no user_message_inputs")
    return SourceEnvelope.model_validate({**record.user_message_inputs, "text": record.text})


def _allowed_ids(envelope: SourceEnvelope) -> set[str]:
    ids = {entry.registry_id for entry in envelope.registry_context}
    ids.update(p.registry_id for p in envelope.participants if p.registry_id)
    return ids


def _mention_in_prompt(value: str, envelope: SourceEnvelope) -> bool:
    if not value.strip():
        return False
    if any(value == p.name for p in envelope.participants):
        return True
    return value in envelope.text


def _require_teacher_values(record: TrainingRecord) -> TeacherAnswer:
    missing = []
    if record.teacher_answer is None:
        missing.append("teacher_answer")
    for index, fact in enumerate(record.facts):
        if fact.sentiment is None or fact.confidence is None:
            missing.append(f"facts[{index}].sentiment/confidence")
    if missing or record.teacher_answer is None:
        raise LegacyRecordError(
            f"{record.record_id}: record lacks the teacher's {', '.join(missing)}; "
            "relabel it rather than fill placeholders"
        )
    return record.teacher_answer


def training_answer(record: TrainingRecord, envelope: SourceEnvelope) -> ExtractionResult:
    """The assistant turn as a strict-valid `ExtractionResult`: the record's
    kept facts with the teacher's own field values, and the teacher's
    summary, entities, actions, unknown entities and coverage. Every fact's
    `[locator]` and quote must appear in `envelope.text` (callers gate facts
    first; a fact is never silently dropped here, since that would turn a
    labeled fact into a false negative)."""
    teacher = _require_teacher_values(record)
    ids = _allowed_ids(envelope)

    facts: list[dict[str, Any]] = []
    for index, fact in enumerate(record.facts, start=1):
        if f"[{fact.locator}]" not in envelope.text:
            raise ValueError(
                f"{record.record_id}: fact {index} locator {fact.locator!r} not in text"
            )
        if fact.quote is not None and fact.quote not in envelope.text:
            raise ValueError(f"{record.record_id}: fact {index} quote not in text")
        speaker = fact.speaker if fact.speaker in ids or fact.speaker in _SPEAKER_ROLES else None
        facts.append(
            {
                "id": f"f{index}",
                "type": fact.fact_type,
                "claim": fact.claim,
                "entities": [
                    e for e in fact.entities if e in ids or _mention_in_prompt(e, envelope)
                ],
                "locator": fact.locator,
                "quote": fact.quote,
                "speaker": speaker,
                "valid_from": fact.valid_from,
                "valid_to": fact.valid_to,
                "severity": fact.severity,
                "sentiment": fact.sentiment,
                "confidence": fact.confidence,
            }
        )

    entities = [
        e
        for e in teacher.entities
        if _mention_in_prompt(str(e.get("mention", "")), envelope)
        and (e.get("registry_id") is None or e.get("registry_id") in ids)
    ]
    unknown = [
        u for u in teacher.unknown_entities if u not in ids and _mention_in_prompt(u, envelope)
    ]
    actions = [a for a in teacher.actions if f"[{a.get('locator')}]" in envelope.text]

    return ExtractionResult.model_validate(
        {
            "schema_version": SCHEMA_VERSION,
            "source_id": envelope.source_id,
            "summary": teacher.summary,
            "entities": entities,
            "facts": facts,
            "actions": actions,
            "unknown_entities": unknown,
            "coverage": teacher.coverage,
        }
    )


def answer_json(result: ExtractionResult) -> str:
    """`result` as the model is asked to write it: the fields of the
    model-facing schema (`extraction_result_json_schema()` drops
    `rejected` and `coercions`), nothing else. Optional fields that are
    `None` are left out: the schema does not require them, and
    `parse_lenient` records an explicit `"severity": null` as a
    coercion."""
    payload = result.model_dump(mode="json", exclude={"rejected", "coercions"}, exclude_none=True)
    return json.dumps(payload, ensure_ascii=False)


def production_sft_messages(
    envelope: SourceEnvelope,
    answer: ExtractionResult,
    *,
    company_context: str | None,
    variant: str,
    extension: CustomerExtension | None = None,
    expected_system_sha256: str | None = None,
    serve_system: Callable[[str], str] | None = None,
) -> list[dict[str, str]]:
    """`[system, user, assistant]` for one example. `serve_system` is the
    serving client's own rewrite of the system prompt (e.g. `portolan.llm.
    mlx_local.effective_system_prompt` appending its JSON-schema block);
    the sha check applies to the prompt before that rewrite."""
    system = build_system_prompt(extension, company_context=company_context, variant=variant)
    if expected_system_sha256 is not None:
        actual = system_prompt_sha256(system)
        if actual != expected_system_sha256:
            raise SystemPromptMismatchError(
                f"rebuilt system prompt sha256 {actual[:16]} != teacher's "
                f"{expected_system_sha256[:16]} (variant={variant!r})"
            )
    if serve_system is not None:
        system = serve_system(system)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": build_user_message(envelope)},
        {"role": "assistant", "content": answer_json(answer)},
    ]


def production_sft_messages_for_record(
    record: TrainingRecord,
    *,
    company_context: str | None,
    envelope: SourceEnvelope | None = None,
    extension: CustomerExtension | None = None,
    serve_system: Callable[[str], str] | None = None,
) -> list[dict[str, str]]:
    """The record's training messages. `company_context` is the vendor note
    the teacher was given (checked against the record's sha); `envelope`
    defaults to the one stored with the record."""
    if record.prompt_variant is None or record.system_prompt_sha256 is None:
        raise LegacyRecordError(f"{record.record_id}: record has no prompt stamp")
    env = envelope if envelope is not None else envelope_for_record(record)
    return production_sft_messages(
        env,
        training_answer(record, env),
        company_context=company_context,
        variant=record.prompt_variant,
        extension=extension,
        expected_system_sha256=record.system_prompt_sha256,
        serve_system=serve_system,
    )
