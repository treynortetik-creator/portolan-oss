"""Tests for `portolan.training.sft_format`: a record written by the silver
pipeline formats to the exact production prompts, and its answer parses
through the production extractor path on the first attempt."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from portolan.extract.backends import FakeBackend, LLMClientBackend
from portolan.extract.prompt import (
    DEFAULT_PROMPT_VARIANT,
    build_system_prompt,
    build_user_message,
    system_prompt_sha256,
)
from portolan.extract.schema import (
    Action,
    Coverage,
    EntityMention,
    ExtractionResult,
    Fact,
    SourceType,
    extraction_result_json_schema,
)
from portolan.ingest.models import Participant, RawSource
from portolan.llm.fake import FakeLLMClient
from portolan.metering import InMemorySink
from portolan.training.pipeline import Document, document_envelopes, run_corpus_single
from portolan.training.records import (
    dropped_path,
    failures_path,
    progress_path,
    read_records,
    records_path,
)
from portolan.training.sft_format import (
    LegacyRecordError,
    SystemPromptMismatchError,
    production_sft_messages_for_record,
    training_answer,
)
from portolan.training.spend import SpendMeter, default_spend_path

CONTEXT = "Veltrona Water Systems sells filtration skids and cartridges to breweries."
TEXT = (
    "Ana Ruiz: our cartridge order arrived three days late again\n"
    "Sam Cole: sorry, I will send a root-cause note on what happened\n"
    "Ana Ruiz: if it happens again we will look at other suppliers"
)


def _document() -> Document:
    raw = RawSource(
        source_id="S-chat-1",
        source_type=SourceType.CHAT,
        source_date=date(2026, 3, 4),
        participants=[Participant(name="Ana Ruiz", role="customer"), Participant(name="Sam Cole")],
        text=TEXT,
        metadata={},
    )
    return Document(source_id="S-chat-1", raw=raw, kind="chat")


def _teacher_result() -> ExtractionResult:
    facts = [
        Fact(
            id="a",
            type="complaint",
            claim="Ana Ruiz reported that a cartridge order arrived three days late.",
            quote="our cartridge order arrived three days late again",
            locator="L1",
            entities=["Ana Ruiz", "acme-foods-1"],
            speaker="customer",
            severity="medium",
            sentiment=-1,
            confidence=0.9,
        ),
        Fact(
            id="b",
            type="commitment",
            claim="Sam Cole promised a root-cause note on the late cartridge order.",
            quote="I will send a root-cause note on what happened",
            locator="L2",
            entities=["Sam Cole", "internal:sam-cole"],
            speaker="Sam Cole",
            sentiment=0,
            confidence=0.8,
        ),
    ]
    return ExtractionResult(
        source_id="S-chat-1",
        summary="Ana Ruiz reported a late cartridge order; Sam Cole promised a root-cause note.",
        entities=[
            EntityMention(mention="Ana Ruiz", kind="contact", confidence=0.9),
            EntityMention(
                mention="Acme Foods", kind="account", registry_id="acme-foods-1", confidence=0.7
            ),
        ],
        facts=facts,
        actions=[
            Action(what="Send the root-cause note", owner_mention="Sam Cole", locator="L2"),
            Action(what="Call the depot", locator="L9"),
        ],
        unknown_entities=["Ana Ruiz", "Acme Foods"],
        coverage=Coverage(segments_read=1, segments_total=1),
    )


def _label(tmp_path: Path, system_prompt: str) -> list:
    corpus = "ubuntu_irc"
    run_corpus_single(
        corpus=corpus,
        kind="chat",
        license_text="test",
        documents=[_document()],
        system_prompt=system_prompt,
        prompt_variant=DEFAULT_PROMPT_VARIANT,
        teacher_model="qwen/qwen3-32b",
        overlap_threshold=0.0,
        backend_factory=lambda model: FakeBackend({"S-chat-1": _teacher_result()}),
        limiter=type("L", (), {"acquire": lambda self: None})(),
        sink=InMemorySink(),
        tenant="test",
        records_file=records_path(tmp_path, corpus),
        failures_file=failures_path(tmp_path, corpus),
        progress_file=progress_path(tmp_path, corpus),
        dropped_file=dropped_path(tmp_path, corpus),
        spend=SpendMeter(path=default_spend_path(tmp_path)),
        budget_usd=100.0,
        sleep=lambda _s: None,
        now=lambda: "2026-09-22T00:00:00+00:00",
    )
    return list(read_records(records_path(tmp_path, corpus)))


@pytest.fixture
def labeled(tmp_path: Path):
    system_prompt = build_system_prompt(company_context=CONTEXT, variant=DEFAULT_PROMPT_VARIANT)
    (record,) = _label(tmp_path, system_prompt)
    return record, system_prompt


def test_pipeline_record_carries_the_prompt_stamp_and_teacher_answer(labeled) -> None:
    record, system_prompt = labeled
    assert record.prompt_variant == DEFAULT_PROMPT_VARIANT
    assert record.system_prompt_sha256 == system_prompt_sha256(system_prompt)
    assert record.user_message_inputs["source_date"] == "2026-03-04"
    assert "text" not in record.user_message_inputs
    assert record.teacher_answer.summary.startswith("Ana Ruiz reported")
    assert [(f.sentiment, f.confidence, f.severity) for f in record.facts] == [
        (-1, 0.9, "medium"),
        (0, 0.8, None),
    ]


def test_messages_are_the_production_prompts(labeled) -> None:
    record, system_prompt = labeled
    system, user, assistant = production_sft_messages_for_record(record, company_context=CONTEXT)
    (envelope,) = document_envelopes(_document())
    assert system == {"role": "system", "content": system_prompt}
    assert user == {"role": "user", "content": build_user_message(envelope)}
    assert assistant["role"] == "assistant"
    assert set(json.loads(assistant["content"])) == set(
        extraction_result_json_schema()["properties"]
    )


def test_answer_round_trips_through_the_production_parser_on_the_first_call(labeled) -> None:
    record, system_prompt = labeled
    _, _, assistant = production_sft_messages_for_record(record, company_context=CONTEXT)
    (envelope,) = document_envelopes(_document())
    # One canned reply: a strict-validation failure would make the backend
    # retry and this client would raise on the second call.
    client = FakeLLMClient([assistant["content"]])
    parsed = LLMClientBackend(client).extract(envelope, system_prompt, InMemorySink(), "test")
    assert parsed.rejected == [] and parsed.coercions == []
    lenient, rejected = ExtractionResult.parse_lenient(json.loads(assistant["content"]))
    assert rejected == [] and lenient.coercions == []
    assert [(f.type, f.quote, f.locator, f.sentiment, f.confidence) for f in parsed.facts] == [
        ("complaint", "our cartridge order arrived three days late again", "L1", -1, 0.9),
        ("commitment", "I will send a root-cause note on what happened", "L2", 0, 0.8),
    ]
    assert parsed.summary == record.teacher_answer.summary


def test_answer_keeps_only_entities_and_speakers_the_prompt_allows(labeled) -> None:
    record, _ = labeled
    answer = training_answer(record, document_envelopes(_document())[0])
    assert [f.entities for f in answer.facts] == [["Ana Ruiz"], ["Sam Cole"]]
    assert [f.speaker for f in answer.facts] == ["customer", None]
    assert [e.mention for e in answer.entities] == ["Ana Ruiz"]
    assert answer.unknown_entities == ["Ana Ruiz"]
    assert [a.locator for a in answer.actions] == ["L2"]


def test_wrong_company_context_is_refused(labeled) -> None:
    record, _ = labeled
    with pytest.raises(SystemPromptMismatchError):
        production_sft_messages_for_record(record, company_context=None)


def test_record_without_teacher_values_is_refused_not_filled(labeled) -> None:
    record, _ = labeled
    with pytest.raises(LegacyRecordError, match="teacher_answer"):
        production_sft_messages_for_record(
            record.model_copy(update={"teacher_answer": None}), company_context=CONTEXT
        )
    no_sentiment = record.facts[0].model_copy(update={"sentiment": None})
    with pytest.raises(LegacyRecordError, match="sentiment"):
        production_sft_messages_for_record(
            record.model_copy(update={"facts": [no_sentiment]}), company_context=CONTEXT
        )
    with pytest.raises(LegacyRecordError, match="prompt stamp"):
        production_sft_messages_for_record(
            record.model_copy(update={"prompt_variant": None}), company_context=CONTEXT
        )


def test_a_fact_whose_quote_is_not_in_the_prompt_raises_rather_than_vanishes(labeled) -> None:
    record, _ = labeled
    moved = record.facts[0].model_copy(update={"quote": "a sentence nobody wrote"})
    with pytest.raises(ValueError, match="quote not in text"):
        training_answer(
            record.model_copy(update={"facts": [moved]}), document_envelopes(_document())[0]
        )
