"""Tests for portolan.extract.backends.

No network access: the two real backends (BedrockConverseBackend,
AnthropicBedrockBackend) are only exercised through their pure
request-building helpers and through construction (which must not require
credentials). Their `extract()` methods are never called here.
"""

from __future__ import annotations

from portolan.extract import segment as seg_module
from portolan.extract.backends import (
    TOOL_NAME,
    AnthropicBedrockBackend,
    BedrockConverseBackend,
    FakeBackend,
    build_anthropic_bedrock_request,
    build_converse_request,
    infer_model_key,
)
from portolan.extract.prompt import build_system_prompt, build_user_message
from portolan.extract.schema import ExtractionResult, SourceEnvelope, extraction_result_json_schema
from portolan.extract.segment import merge_segment_results
from portolan.metering import InMemorySink


def _envelope(source_id: str = "ticket-1") -> SourceEnvelope:
    return SourceEnvelope.model_validate(
        {
            "source_id": source_id,
            "source_type": "ticket",
            "source_date": "2026-08-14",
            "text": "[L1] customer reports a late delivery",
        }
    )


def _canned_result(source_id: str) -> ExtractionResult:
    return ExtractionResult.model_validate(
        {
            "schema_version": "0.1",
            "source_id": source_id,
            "summary": "Customer reported a late delivery.",
            "entities": [],
            "facts": [
                {
                    "id": "f1",
                    "type": "complaint",
                    "claim": "The customer reported a late delivery.",
                    "entities": [],
                    "locator": "[L1]",
                    "quote": "customer reports a late delivery",
                    "sentiment": -1,
                    "confidence": 0.7,
                }
            ],
            "actions": [],
            "unknown_entities": [],
            "coverage": {"segments_read": 1, "segments_total": 1},
        }
    )


# --------------------------------------------------------------------------
# infer_model_key
# --------------------------------------------------------------------------


def test_infer_model_key_matches_known_bedrock_ids() -> None:
    assert infer_model_key("amazon.nova-lite-v1:0") == "nova-lite"
    assert infer_model_key("amazon.nova-micro-v1:0") == "nova-micro"
    assert infer_model_key("us.openai.gpt-5.6-luna") == "gpt-5.6-luna"
    assert infer_model_key("anthropic.claude-haiku-4-5") == "claude-haiku-4-5"
    assert infer_model_key("anthropic.claude-sonnet-5") == "claude-sonnet-5"
    assert infer_model_key("anthropic.claude-opus-5") == "claude-opus-5"


def test_infer_model_key_falls_back_to_raw_id_when_unknown() -> None:
    assert infer_model_key("some.unrecognized.model-id") == "some.unrecognized.model-id"


# --------------------------------------------------------------------------
# FakeBackend
# --------------------------------------------------------------------------


def test_fake_backend_dict_returns_canned_result_and_meters_once() -> None:
    envelope = _envelope("ticket-1")
    canned = {"ticket-1": _canned_result("ticket-1")}
    sink = InMemorySink()
    backend = FakeBackend(canned)

    result = backend.extract(envelope, "system prompt", sink, tenant="acme")

    assert result.source_id == "ticket-1"
    assert len(sink.events) == 1
    event = sink.events[0]
    assert event.surface == "extract"
    assert event.model_key == "nova-lite"
    assert event.input_tokens > 0
    assert event.output_tokens > 0


def test_fake_backend_callable_returns_canned_result() -> None:
    sink = InMemorySink()
    backend = FakeBackend(lambda envelope: _canned_result(envelope.source_id))
    result = backend.extract(_envelope("ticket-9"), "system prompt", sink, tenant="acme")
    assert result.source_id == "ticket-9"
    assert len(sink.events) == 1


def test_fake_backend_missing_source_id_raises_key_error() -> None:
    sink = InMemorySink()
    backend = FakeBackend({})
    try:
        backend.extract(_envelope("missing"), "system prompt", sink, tenant="acme")
    except KeyError as exc:
        assert "missing" in str(exc)
    else:
        raise AssertionError("expected KeyError for an unmapped source_id")


# --------------------------------------------------------------------------
# Request-building pure functions (no network, no client)
# --------------------------------------------------------------------------


def test_build_converse_request_shape() -> None:
    envelope = _envelope()
    system_prompt = build_system_prompt(None)
    request = build_converse_request(envelope, system_prompt)

    assert request["system"] == [{"text": system_prompt}]
    assert request["messages"][0]["role"] == "user"
    assert build_user_message(envelope) in request["messages"][0]["content"][0]["text"]

    tool_config = request["toolConfig"]
    tool_spec = tool_config["tools"][0]["toolSpec"]
    assert tool_spec["name"] == TOOL_NAME
    assert tool_spec["inputSchema"]["json"] == extraction_result_json_schema()
    assert tool_config["toolChoice"] == {"tool": {"name": TOOL_NAME}}
    assert request["inferenceConfig"]["maxTokens"] > 0


def test_build_anthropic_bedrock_request_shape() -> None:
    envelope = _envelope()
    system_prompt = build_system_prompt(None)
    request = build_anthropic_bedrock_request(envelope, system_prompt)

    assert request["system"] == system_prompt
    assert request["messages"] == [{"role": "user", "content": build_user_message(envelope)}]
    assert request["tool_choice"] == {"type": "tool", "name": TOOL_NAME}

    tool = request["tools"][0]
    assert tool["name"] == TOOL_NAME
    assert tool["strict"] is True
    assert tool["input_schema"] == extraction_result_json_schema()
    # No top-level "model" key -- the caller supplies model= separately.
    assert "model" not in request


def test_build_converse_request_respects_max_tokens_override() -> None:
    request = build_converse_request(_envelope(), "sys", max_tokens=1234)
    assert request["inferenceConfig"]["maxTokens"] == 1234


# --------------------------------------------------------------------------
# Real backends: constructible without credentials, never actually called.
# --------------------------------------------------------------------------


def test_bedrock_converse_backend_constructs_without_credentials() -> None:
    backend = BedrockConverseBackend("amazon.nova-lite-v1:0", region="us-east-1")
    assert backend.model_id == "amazon.nova-lite-v1:0"
    assert backend.model_key == "nova-lite"
    assert backend._client is None  # lazy: no client created at construction


def test_bedrock_converse_backend_accepts_explicit_model_key() -> None:
    backend = BedrockConverseBackend(
        "some-future-model-id", region="us-east-1", model_key="gpt-oss-120b"
    )
    assert backend.model_key == "gpt-oss-120b"


def test_anthropic_bedrock_backend_constructs_without_credentials() -> None:
    backend = AnthropicBedrockBackend("anthropic.claude-haiku-4-5", region="us-east-1")
    assert backend.model_id == "anthropic.claude-haiku-4-5"
    assert backend.model_key == "claude-haiku-4-5"
    assert backend._client is None  # lazy: no client created at construction


# --------------------------------------------------------------------------
# End-to-end: envelope -> segment -> FakeBackend per segment ->
# merge_segment_results, with one metering event per segment.
# --------------------------------------------------------------------------


def test_end_to_end_segment_extract_merge_and_metering() -> None:
    lines = [f"[00:00:{i:02d}] segment line {i}" for i in range(12)]
    text = "\n".join(lines)

    segments = seg_module.segment(text, max_tokens=5, overlap_tokens=0, count_tokens=lambda line: 1)
    assert len(segments) == 3  # 5, 5, 2 lines per the deterministic token counter

    envelopes = [
        SourceEnvelope.model_validate(
            {
                "source_id": "call-42",
                "source_type": "call",
                "source_date": "2026-08-14",
                "text": seg.text,
                "segment": {"index": seg.index, "total": seg.total},
            }
        )
        for seg in segments
    ]

    def canned(envelope: SourceEnvelope) -> ExtractionResult:
        assert envelope.segment is not None
        idx = envelope.segment.index
        if idx in (1, 2):
            # The same complaint is picked up independently in the first
            # two segments -- this is what the merge step must dedup.
            facts = [
                {
                    "id": "f1",
                    "type": "complaint",
                    "claim": (
                        "Ashgrove Medical Center received three late deliveries in August 2026."
                    ),
                    "entities": ["ashgrove-medical"],
                    "locator": "00:00:01" if idx == 1 else "00:00:06",
                    "quote": "we had three late deliveries last month",
                    "sentiment": -2,
                    "confidence": 0.9,
                }
            ]
        else:
            facts = [
                {
                    "id": "f1",
                    "type": "usage",
                    "claim": "Springfield runs eleven X-series pumps across two ICUs.",
                    "entities": ["ashgrove-medical"],
                    "locator": "00:00:10",
                    "sentiment": 0,
                    "confidence": 0.8,
                }
            ]
        return ExtractionResult.model_validate(
            {
                "schema_version": "0.1",
                "source_id": "call-42",
                "summary": f"segment {idx} summary",
                "entities": [],
                "facts": facts,
                "actions": [],
                "unknown_entities": [],
                "coverage": {"segments_read": 1, "segments_total": envelope.segment.total},
            }
        )

    sink = InMemorySink()
    backend = FakeBackend(canned)
    system_prompt = build_system_prompt(None)

    results = [backend.extract(env, system_prompt, sink, tenant="acme") for env in envelopes]

    # One metering event per segment.
    assert len(sink.events) == 3
    assert all(event.surface == "extract" for event in sink.events)
    assert all(event.model_key == "nova-lite" for event in sink.events)
    assert len({event.request_id for event in sink.events}) == 3  # all unique

    merged = merge_segment_results(results)
    assert merged.source_id == "call-42"
    assert merged.coverage.segments_read == 3
    assert merged.coverage.segments_total == 3
    # The complaint from segments 1 and 2 collapsed into one fact; the usage
    # fact from segment 3 stayed separate.
    assert len(merged.facts) == 2
    complaint = next(f for f in merged.facts if f.type.value == "complaint")
    assert complaint.locator == "00:00:01"  # earliest of the two duplicate sightings
    usage_fact = next(f for f in merged.facts if f.type.value == "usage")
    assert usage_fact.locator == "00:00:10"
