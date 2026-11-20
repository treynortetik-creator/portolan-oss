"""End-to-end test for portolan.extract.backends.LLMClientBackend using
FakeLLMClient -- build-contracts.md "LLM access": `LLMClientBackend(client)`
implementing the existing `Backend` protocol so extraction runs through
the same `LLMClient` every other model call goes through.
"""

from __future__ import annotations

import json
from contextlib import contextmanager

import httpx
import pytest

from portolan.extract.backends import LLMClientBackend
from portolan.extract.schema import ExtractionResult, SourceEnvelope
from portolan.llm import openrouter as openrouter_module
from portolan.llm.base import LLMResult
from portolan.llm.fake import FakeLLMClient
from portolan.llm.openrouter import OpenRouterClient
from portolan.metering import InMemorySink, metered


def _envelope(source_id: str = "ticket-1") -> SourceEnvelope:
    return SourceEnvelope.model_validate(
        {
            "source_id": source_id,
            "source_type": "ticket",
            "source_date": "2026-08-14",
            "text": "[L1] customer reports a late delivery",
        }
    )


def _canned_extraction_dict(source_id: str) -> dict:
    return {
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


def test_llm_client_backend_extracts_via_fake_client() -> None:
    envelope = _envelope("ticket-1")
    client_sink = InMemorySink()
    client = FakeLLMClient([_canned_extraction_dict("ticket-1")], sink=client_sink, tenant="acme")
    backend = LLMClientBackend(client)

    # sink/tenant here satisfy the Backend protocol but aren't used --
    # metering happens inside the client, against the sink/tenant it was
    # constructed with (see LLMClientBackend's docstring).
    unused_sink = InMemorySink()
    result = backend.extract(envelope, "system prompt", unused_sink, tenant="ignored")

    assert isinstance(result, ExtractionResult)
    assert result.source_id == "ticket-1"
    assert len(result.facts) == 1
    assert result.facts[0].type.value == "complaint"

    # Metering happened exactly once, through the client's own sink.
    assert len(client_sink.events) == 1
    assert client_sink.events[0].surface == "extract"
    assert client_sink.events[0].tenant == "acme"
    assert len(unused_sink.events) == 0


def test_llm_client_backend_passes_system_prompt_and_user_message() -> None:
    envelope = _envelope("ticket-2")
    seen: dict = {}

    def responder(*, system, user, json_schema, model, surface):
        seen.update(system=system, user=user, json_schema=json_schema, model=model, surface=surface)
        return _canned_extraction_dict("ticket-2")

    client = FakeLLMClient(responder)
    backend = LLMClientBackend(client)

    backend.extract(envelope, "the system prompt", InMemorySink(), tenant="acme")

    assert seen["system"] == "the system prompt"
    assert envelope.text in seen["user"]
    assert seen["surface"] == "extract"
    assert seen["json_schema"] is not None
    assert "source_id" in seen["json_schema"]["properties"]
    assert seen["model"] is None


def test_llm_client_backend_passes_model_through_to_complete() -> None:
    # Review finding F17: LLMClientBackend couldn't select a model at all;
    # the bench worked around it with a bare `except TypeError` retry.
    envelope = _envelope("ticket-model")
    seen: dict = {}

    def responder(*, system, user, json_schema, model, surface):
        del system, user, json_schema, surface
        seen["model"] = model
        return _canned_extraction_dict("ticket-model")

    client = FakeLLMClient(responder)
    backend = LLMClientBackend(client, model="claude-sonnet-5")

    backend.extract(envelope, "sys", InMemorySink(), tenant="acme")

    assert seen["model"] == "claude-sonnet-5"


class _CapturingClient:
    def __init__(self) -> None:
        self.kwargs: list[dict] = []

    def complete(self, **kwargs) -> LLMResult:
        self.kwargs.append(kwargs)
        data = _canned_extraction_dict("ticket-cap")
        return LLMResult(
            text=json.dumps(data),
            data=data,
            input_tokens=1,
            output_tokens=1,
            cached_tokens=0,
            model="m",
        )


def test_llm_client_backend_sends_max_tokens_only_when_set() -> None:
    default_client, capped_client = _CapturingClient(), _CapturingClient()
    LLMClientBackend(default_client).extract(
        _envelope("ticket-cap"), "sys", InMemorySink(), tenant="acme"
    )
    LLMClientBackend(capped_client, max_tokens=32000).extract(
        _envelope("ticket-cap"), "sys", InMemorySink(), tenant="acme"
    )
    assert "max_tokens" not in default_client.kwargs[0]
    assert capped_client.kwargs[0]["max_tokens"] == 32000


# --------------------------------------------------------------------
# Fix wave 2026-09-12 (bench telemetry bug): LLMClientBackend.extract()
# forwards the sink/tenant it receives into `client.using_sink(...)` when
# the client provides that optional extension (`portolan.llm.base`'s
# module docstring). `FakeLLMClient` (portolan/llm/fake.py) has no
# `using_sink` -- not owned by this fix -- so this tiny local double
# stands in, implementing just enough of `LLMClient` to prove the
# contract without touching that file.
# --------------------------------------------------------------------


class _ReroutableFakeClient:
    """`complete()` meters into `self.sink`/`self.tenant` (like every real
    `LLMClient`); `using_sink()` temporarily swaps them, exactly the
    contract `OpenRouterClient.using_sink` implements for real."""

    def __init__(self, canned: list[dict], *, sink: InMemorySink, tenant: str) -> None:
        self._canned = list(canned)
        self._call_count = 0
        self.sink = sink
        self.tenant = tenant

    def complete(self, *, system, user, json_schema, model, surface, max_tokens=4096):
        del system, user, json_schema, max_tokens
        data = self._canned[self._call_count]
        self._call_count += 1
        with metered(self.sink, self.tenant, surface, "fake-model") as usage:
            usage.input_tokens = 1
            usage.output_tokens = 1
        return LLMResult(
            text=None,
            data=data,
            input_tokens=1,
            output_tokens=1,
            cached_tokens=0,
            model=model or "",
        )

    @contextmanager
    def using_sink(self, sink, tenant):
        previous_sink, previous_tenant = self.sink, self.tenant
        self.sink, self.tenant = sink, tenant
        try:
            yield
        finally:
            self.sink, self.tenant = previous_sink, previous_tenant


def test_llm_client_backend_forwards_sink_when_client_supports_using_sink() -> None:
    envelope = _envelope("ticket-sink")
    client_own_sink = InMemorySink()
    client = _ReroutableFakeClient(
        [_canned_extraction_dict("ticket-sink")], sink=client_own_sink, tenant="client-tenant"
    )
    backend = LLMClientBackend(client)

    bench_sink = InMemorySink()
    result = backend.extract(envelope, "sys", bench_sink, tenant="bench-tenant")

    assert result.source_id == "ticket-sink"
    # The bench's own sink/tenant -- not the client's default -- saw the
    # usage event: this is the whole point of the fix (bench tokens/cost
    # used to always read 0/$0.00 because this never happened).
    assert len(bench_sink.events) == 1
    assert bench_sink.events[0].tenant == "bench-tenant"
    assert len(client_own_sink.events) == 0
    # Restored on exit, so a later call through this same client (e.g. the
    # judge, or another surface) goes back to its own sink/tenant.
    assert client.sink is client_own_sink
    assert client.tenant == "client-tenant"


def test_llm_client_backend_forwards_sink_across_the_validation_retry() -> None:
    # The retry-then-parse_lenient path (Group 3) makes a SECOND
    # `complete()` call inside the same `extract()` -- both must land in
    # the bench's sink, not just the first.
    envelope = _envelope("ticket-sink-retry")
    client_own_sink = InMemorySink()
    client = _ReroutableFakeClient(
        [
            _invalid_commitment_dict("ticket-sink-retry"),
            _canned_extraction_dict("ticket-sink-retry"),
        ],
        sink=client_own_sink,
        tenant="client-tenant",
    )
    backend = LLMClientBackend(client)

    bench_sink = InMemorySink()
    result = backend.extract(envelope, "sys", bench_sink, tenant="bench-tenant")

    assert len(result.facts) == 1
    assert len(bench_sink.events) == 2
    assert all(e.tenant == "bench-tenant" for e in bench_sink.events)
    assert len(client_own_sink.events) == 0


def test_llm_client_backend_without_using_sink_is_unaffected() -> None:
    # FakeLLMClient has no using_sink -- the sink/tenant passed to
    # extract() are still not forwarded, exactly today's pre-fix
    # behavior, so every existing FakeLLMClient-based test above keeps
    # passing unchanged.
    envelope = _envelope("ticket-no-using-sink")
    client_sink = InMemorySink()
    client = FakeLLMClient([_canned_extraction_dict("ticket-no-using-sink")], sink=client_sink)
    backend = LLMClientBackend(client)

    bench_sink = InMemorySink()
    backend.extract(envelope, "sys", bench_sink, tenant="bench-tenant")

    assert len(client_sink.events) == 1
    assert len(bench_sink.events) == 0


def test_llm_client_backend_raises_when_client_returns_no_data() -> None:
    # A plain string with no JSON object in it -- FakeLLMClient tries to
    # extract JSON (LLMClientBackend always passes a json_schema) and
    # comes up empty, so `result.data` is None.
    client = FakeLLMClient(["sorry, I can't help with that"])
    backend = LLMClientBackend(client)

    with pytest.raises(ValueError, match="no JSON data returned"):
        backend.extract(_envelope("ticket-3"), "sys", InMemorySink(), tenant="acme")


# --------------------------------------------------------------------
# Group 3 of the review fix wave: retry-then-parse_lenient
# --------------------------------------------------------------------


def _invalid_commitment_dict(source_id: str) -> dict:
    # Rule 5: commitment requires a quote. The real failure from the
    # first bench run against the corpus.
    return {
        "schema_version": "0.1",
        "source_id": source_id,
        "summary": "s",
        "entities": [],
        "facts": [
            {
                "id": "f1",
                "type": "commitment",
                "claim": "Our rep promised a summary by Friday.",
                "entities": [],
                "locator": "[L1]",
                "quote": None,
                "sentiment": 0,
                "confidence": 0.8,
            }
        ],
        "actions": [],
        "unknown_entities": [],
        "coverage": {"segments_read": 1, "segments_total": 1},
    }


def test_llm_client_backend_retries_once_on_validation_failure() -> None:
    envelope = _envelope("ticket-retry")
    client = FakeLLMClient(
        [_invalid_commitment_dict("ticket-retry"), _canned_extraction_dict("ticket-retry")]
    )
    backend = LLMClientBackend(client)

    result = backend.extract(envelope, "sys", InMemorySink(), tenant="acme")

    assert result.source_id == "ticket-retry"
    assert len(result.facts) == 1
    assert result.facts[0].type == "complaint"
    assert result.rejected == []
    assert client._call_count == 2  # the retry actually happened


def test_llm_client_backend_falls_back_to_parse_lenient_after_a_failed_retry() -> None:
    envelope = _envelope("ticket-lenient")
    # Both attempts are invalid the same way; parse_lenient must still
    # salvage a result rather than raise, and record what it dropped.
    client = FakeLLMClient(
        [
            _invalid_commitment_dict("ticket-lenient"),
            _invalid_commitment_dict("ticket-lenient"),
        ]
    )
    backend = LLMClientBackend(client)

    result = backend.extract(envelope, "sys", InMemorySink(), tenant="acme")

    assert result.source_id == "ticket-lenient"
    assert result.facts == []
    assert len(result.rejected) == 1
    assert result.rejected[0].kind == "fact"
    assert client._call_count == 2  # strict attempt + one retry, no more


# --------------------------------------------------------------------
# Fix instructions item 2: the real Nova Lite/Micro bench losses. Both
# strict attempts kept failing on a top-level extra the model repeated
# (source_type/source_date, echoed on both calls) -- pre-fix, this made
# `ExtractionResult.parse_lenient` itself raise `pydantic.ValidationError`
# (it didn't strip unknown top-level keys before its own final strict
# validate), one level too late for the `except ValidationError:` branch
# below to help, since that branch's whole job is to call parse_lenient
# BECAUSE the direct attempt already failed. These tests prove
# `LLMClientBackend.extract` really reaches a salvaged `ExtractionResult`
# instead of letting that second `ValidationError` escape.
# --------------------------------------------------------------------


def _dict_with_top_level_extras(source_id: str) -> dict:
    data = _canned_extraction_dict(source_id)
    # Real extras observed live: source_type/source_date echoed verbatim,
    # plus an empty rejected_facts list -- none are ExtractionResult
    # fields.
    data["source_type"] = "ticket"
    data["source_date"] = "2026-08-14"
    data["rejected_facts"] = []
    return data


def test_llm_client_backend_reaches_parse_lenient_for_repeated_top_level_extras() -> None:
    envelope = _envelope("ticket-extras")
    client = FakeLLMClient(
        [
            _dict_with_top_level_extras("ticket-extras"),
            _dict_with_top_level_extras("ticket-extras"),
        ]
    )
    backend = LLMClientBackend(client)

    # Pre-fix, this raised pydantic.ValidationError straight out of
    # extract() -- the whole source lost, not just the top-level extras.
    result = backend.extract(envelope, "sys", InMemorySink(), tenant="acme")

    assert result.source_id == "ticket-extras"
    assert len(result.facts) == 1
    assert result.facts[0].type == "complaint"
    assert result.rejected == []
    assert client._call_count == 2  # strict attempt + one retry, no more


def _dict_with_entity_needing_coercion(source_id: str) -> dict:
    data = _canned_extraction_dict(source_id)
    data["entities"] = [{"mention": "Support Plan", "kind": "service"}]
    return data


def test_llm_client_backend_reaches_parse_lenient_and_coerces_entity_defaults() -> None:
    envelope = _envelope("ticket-entity-coercion")
    client = FakeLLMClient(
        [
            _dict_with_entity_needing_coercion("ticket-entity-coercion"),
            _dict_with_entity_needing_coercion("ticket-entity-coercion"),
        ]
    )
    backend = LLMClientBackend(client)

    result = backend.extract(envelope, "sys", InMemorySink(), tenant="acme")

    assert result.source_id == "ticket-entity-coercion"
    assert len(result.entities) == 1
    assert result.entities[0].confidence == 0.5
    assert result.entities[0].kind.value == "other"
    fields_coerced = {c.field for c in result.coercions}
    assert fields_coerced == {"confidence", "kind"}


def test_llm_client_backend_records_coercion_count_on_the_metering_sink() -> None:
    envelope = _envelope("ticket-coercion-metering")
    client = FakeLLMClient(
        [
            _dict_with_entity_needing_coercion("ticket-coercion-metering"),
            _dict_with_entity_needing_coercion("ticket-coercion-metering"),
        ]
    )
    backend = LLMClientBackend(client)
    metering_sink = InMemorySink()

    result = backend.extract(envelope, "sys", metering_sink, tenant="acme")

    assert len(result.coercions) == 2  # confidence + kind, same fixture as the test above
    # client.complete()'s own two events (FakeLLMClient has no using_sink,
    # so those land in `client.sink`, not here) are separate from this
    # backend-recorded, zero-token bookkeeping event for the coercion
    # count -- the only way that count reaches a metering sink at all,
    # since the events for the two real calls above are already recorded
    # (and immutable) by the time extract() gets control back.
    coercion_events = [e for e in metering_sink.events if "coercions" in e.extra]
    assert len(coercion_events) == 1
    assert coercion_events[0].extra["coercions"] == 2
    assert coercion_events[0].tenant == "acme"
    assert coercion_events[0].surface == "extract"


def _dict_with_a_rejected_fact(source_id: str) -> dict:
    data = _canned_extraction_dict(source_id)
    # A second fact with no locator: Fact's own validator rejects it
    # outright (rule 4, "Locator is required. A fact without one is
    # rejected at merge") -- parse_lenient drops it into `.rejected`
    # rather than failing the whole result.
    data["facts"].append(
        {
            "id": "f2",
            "type": "complaint",
            "claim": "a second, unlocatable complaint",
            "entities": [],
            "locator": "",
            "quote": None,
            "sentiment": -1,
            "confidence": 0.5,
        }
    )
    return data


def test_llm_client_backend_records_rejected_facts_on_the_metering_sink() -> None:
    # Bug hunt 2026-09-18, finding 6: LLMClientBackend recorded a
    # coercion count on its own synthetic metering event (see the test
    # above) but never `rejected_facts` -- only the Bedrock backends did,
    # so a rejected fact was visible on `result.rejected` but invisible to
    # anything reading the usage ledger, breaking fact rule 4 ("a rejected
    # fact is counted in the coverage report") for the backend this
    # project actually runs extraction through.
    envelope = _envelope("ticket-rejected-fact")
    client = FakeLLMClient(
        [
            _dict_with_a_rejected_fact("ticket-rejected-fact"),
            _dict_with_a_rejected_fact("ticket-rejected-fact"),
        ]
    )
    backend = LLMClientBackend(client)
    metering_sink = InMemorySink()

    result = backend.extract(envelope, "sys", metering_sink, tenant="acme")

    assert len(result.rejected) == 1
    assert len(result.facts) == 1  # the good fact survives; the bad one is dropped, not crashing

    rejected_events = [e for e in metering_sink.events if "rejected_facts" in e.extra]
    assert len(rejected_events) == 1
    assert len(rejected_events[0].extra["rejected_facts"]) == 1
    assert rejected_events[0].extra["rejected_facts"][0]["kind"] == "fact"
    assert rejected_events[0].tenant == "acme"
    assert rejected_events[0].surface == "extract"


def _dict_missing_source_id_and_needing_entity_coercion(source_id: str) -> dict:
    data = _dict_with_entity_needing_coercion(source_id)
    del data["source_id"]
    return data


def test_llm_client_backend_fills_missing_source_id_from_envelope_via_parse_lenient() -> None:
    # Real "genuinely incomplete" case parse_lenient's own docstring
    # describes: a model that leaves source_id off entirely. Without
    # default_source_id, parse_lenient leaves it missing and the final
    # strict model_validate raises -- losing the whole source.
    envelope = _envelope("ticket-missing-source-id")
    client = FakeLLMClient(
        [
            _dict_missing_source_id_and_needing_entity_coercion("ticket-missing-source-id"),
            _dict_missing_source_id_and_needing_entity_coercion("ticket-missing-source-id"),
        ]
    )
    backend = LLMClientBackend(client)

    result = backend.extract(envelope, "sys", InMemorySink(), tenant="acme")

    assert result.source_id == "ticket-missing-source-id"
    fields_coerced = {c.field for c in result.coercions}
    assert "source_id" in fields_coerced


# --------------------------------------------------------------------
# `validate_json` (fix wave 2026-09-12): LLMClientBackend must opt a
# supporting client OUT of its own client-side jsonschema gate (only
# `portolan.llm.openrouter.OpenRouterClient` has one, but detected via
# `inspect.signature` rather than an isinstance check, per the class
# docstring), while a client with no such parameter -- FakeLLMClient and
# every `_ReroutableFakeClient`-style double above -- must never see an
# unexpected kwarg.
# --------------------------------------------------------------------


class _ValidateJsonAwareFakeClient:
    """Minimal `LLMClient` stand-in whose `complete()` accepts
    `validate_json`, used only to prove `LLMClientBackend._complete()`
    detects and forwards it. `OpenRouterClient` is the real
    implementation (covered end-to-end below, via a mocked transport);
    this avoids depending on httpx mocking just to observe one kwarg."""

    def __init__(self, canned: dict) -> None:
        self._canned = canned
        self.received_validate_json: list[bool] = []

    def complete(
        self,
        *,
        system: str,
        user: str,
        json_schema: dict | None = None,
        model: str | None = None,
        surface: str,
        max_tokens: int = 4096,
        validate_json: bool = True,
    ) -> LLMResult:
        del system, user, json_schema, max_tokens
        self.received_validate_json.append(validate_json)
        return LLMResult(
            text=json.dumps(self._canned),
            data=self._canned,
            input_tokens=1,
            output_tokens=1,
            cached_tokens=0,
            model=model or "nova-lite",
        )


def test_llm_client_backend_opts_out_of_client_side_validation_when_supported() -> None:
    envelope = _envelope("ticket-validate-json")
    client = _ValidateJsonAwareFakeClient(_canned_extraction_dict("ticket-validate-json"))
    backend = LLMClientBackend(client)

    result = backend.extract(envelope, "sys", InMemorySink(), tenant="acme")

    assert result.source_id == "ticket-validate-json"
    # The strict attempt validated immediately -- no retry needed -- so
    # only one call happened, and it opted out of the client's own gate.
    assert client.received_validate_json == [False]


def test_llm_client_backend_does_not_pass_validate_json_to_a_client_without_it() -> None:
    # FakeLLMClient's complete() has no validate_json parameter at all --
    # if LLMClientBackend passed it unconditionally this would raise
    # TypeError instead of extracting normally.
    envelope = _envelope("ticket-no-validate-json-param")
    client = FakeLLMClient([_canned_extraction_dict("ticket-no-validate-json-param")])
    backend = LLMClientBackend(client)

    result = backend.extract(envelope, "sys", InMemorySink(), tenant="acme")

    assert result.source_id == "ticket-no-validate-json-param"


# --------------------------------------------------------------------
# End-to-end reproduction of the real bug this fix wave addresses: the 18
# amazon/nova-lite-v1 failures in
# var/bench/compare3/amazon_nova-lite-v1/bench-2026-09-12-openrouter.json,
# every one an `entities` item inside a fact shaped like a top-level
# EntityMention object (`{"mention": ..., "kind": ..., "registry_id":
# ...}`) where `Fact.entities` wants a plain string -- rejected by
# OpenRouterClient's own local jsonschema gate in its fallback path
# before ExtractionResult.parse_lenient (which already knows how to
# coerce this exact shape) ever got a chance to run. Uses a real
# `OpenRouterClient` over a mocked `httpx` transport (no network) rather
# than `FakeLLMClient`, since `FakeLLMClient` never had a client-side gate
# to begin with -- this is the only test in this file that actually
# exercises the fix.
# --------------------------------------------------------------------


def test_llm_client_backend_survives_entity_object_via_openrouter_client_mocked_transport() -> None:
    envelope = _envelope("S-call-002")
    # The real observed Nova Lite reply shape (var/bench/compare3/
    # amazon_nova-lite-v1/bench-2026-09-12-openrouter.json, S-call-002),
    # trimmed to one fact: entities[1] is a nested object, and sentiment
    # is left null.
    raw_reply = {
        "schema_version": "0.1",
        "source_id": "S-call-002",
        "summary": "Call recap.",
        "entities": [],
        "facts": [
            {
                "id": "f1",
                "type": "complaint",
                "claim": "Customer reported a late delivery.",
                "entities": [
                    {
                        "mention": "Priya Natarajan",
                        "kind": "internal_person",
                        "registry_id": "internal:priya-natarajan",
                    }
                ],
                "locator": "[L1]",
                "quote": "customer reports a late delivery",
                "sentiment": None,
                "confidence": 0.7,
            }
        ],
        "actions": [],
        "unknown_entities": [],
        "coverage": {"segments_read": 1, "segments_total": 1},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        # amazon/nova-lite-v1 on OpenRouter advertises no response_format
        # support at all in its supported_parameters -- STRATEGY_NONE,
        # straight to the manual system-instruction fallback.
        assert "response_format" not in body
        return httpx.Response(
            200,
            json={
                "id": "gen-test",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": json.dumps(raw_reply)},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 50, "completion_tokens": 40},
            },
        )

    model_key = "amazon/nova-lite-v1"
    openrouter_module._json_strategy_cache[model_key] = openrouter_module.STRATEGY_NONE
    try:
        client = OpenRouterClient(
            spend_key="test",
            api_key="or-test-key",
            sink=InMemorySink(),
            transport=httpx.MockTransport(handler),
            sleep=lambda _seconds: None,
        )
        backend = LLMClientBackend(client, model=model_key)

        metering_sink = InMemorySink()
        result = backend.extract(envelope, "system prompt", metering_sink, tenant="acme")
    finally:
        openrouter_module.reset_json_strategy_cache()

    assert isinstance(result, ExtractionResult)
    assert result.source_id == "S-call-002"
    assert len(result.facts) == 1
    # The nested entity object was coerced to its registry id, not
    # dropped; the null sentiment was defaulted to 0 -- neither lost the
    # fact, unlike before this fix (LLMJsonError, whole source lost).
    assert result.facts[0].entities == ["internal:priya-natarajan"]
    assert result.facts[0].sentiment == 0
    assert result.rejected == []

    fields_coerced = {c.field for c in result.coercions}
    assert "entities" in fields_coerced
    assert "sentiment" in fields_coerced

    coercion_events = [e for e in metering_sink.events if "coercions" in e.extra]
    assert len(coercion_events) == 1
    assert coercion_events[0].extra["coercions"] == len(result.coercions)
