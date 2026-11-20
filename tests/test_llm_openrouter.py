"""Tests for portolan.llm.openrouter.OpenRouterClient.

No network anywhere in this file: every test injects an `httpx.MockTransport`
so the request never leaves the process. Response shapes (the `usage`
block, `response_format`, headers) come from
https://openrouter.ai/docs/api-reference/overview,
https://openrouter.ai/docs/features/structured-outputs, and
https://openrouter.ai/docs/use-cases/usage-accounting (fetched 2026-09-12
-- see portolan/llm/openrouter.py's module docstring for the per-field
citation).
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from portolan.llm import openrouter as openrouter_module
from portolan.llm.base import LLMError, LLMJsonError
from portolan.llm.openrouter import (
    CHAT_COMPLETIONS_URL,
    MODEL_ENDPOINTS_URL_TEMPLATE,
    STRATEGY_NONE,
    STRATEGY_OBJECT,
    STRATEGY_SCHEMA,
    OpenRouterClient,
    _choose_json_strategy,
    _deep_flatten_document_trees,
    _flatten_document_tree_node,
    _is_document_tree_node,
    _is_reasoning_mandatory_rejection,
    fetch_supported_parameters,
    prime_json_strategy,
    reset_json_strategy_cache,
    reset_reasoning_locked_cache,
)
from portolan.metering import InMemorySink

SCHEMA = {
    "type": "object",
    "properties": {"greeting": {"type": "string"}},
    "required": ["greeting"],
    "additionalProperties": False,
}


@pytest.fixture(autouse=True)
def _isolated_json_strategy_cache():
    """`portolan.llm.openrouter`'s JSON-strategy cache is deliberately
    module-level/process-wide in production (see its own docstring: worker
    threads must share one learned answer per model id) -- but that means
    it persists across tests in this same pytest process unless reset.
    Without this, a test that triggers a 400 rejection for the (shared)
    default model id would downgrade the cache, and a LATER test for that
    same model id would silently run a different code path than the one
    it's actually testing. Every test in this file starts and ends with
    an empty cache."""
    reset_json_strategy_cache()
    yield
    reset_json_strategy_cache()


@pytest.fixture(autouse=True)
def _isolated_reasoning_locked_cache(monkeypatch: pytest.MonkeyPatch):
    """Same isolation concern as `_isolated_json_strategy_cache`, for the
    reasoning-locked-model cache (fix instructions item 1): module-level/
    process-wide in production, reset around every test here so one test's
    learned-from-a-400 model id can't leak into another. Also clears the
    `PORTOLAN_OPENROUTER_REASONING_OFF_MODELS` env var so a test that sets
    it doesn't leak into a later one."""
    monkeypatch.delenv("PORTOLAN_OPENROUTER_REASONING_OFF_MODELS", raising=False)
    monkeypatch.delenv("PORTOLAN_OPENROUTER_REASONING_EFFORT", raising=False)
    reset_reasoning_locked_cache()
    yield
    reset_reasoning_locked_cache()


@pytest.fixture(autouse=True)
def _isolated_temperature_env(monkeypatch: pytest.MonkeyPatch):
    """`PORTOLAN_TEMPERATURE` (infra-2b wave, 2026-09-18) is read fresh on
    every call via `portolan.llm.base.resolve_temperature` -- clear it
    around every test in this file so a test that sets it can't leak into
    a later one that doesn't expect it."""
    monkeypatch.delenv("PORTOLAN_TEMPERATURE", raising=False)
    yield
    monkeypatch.delenv("PORTOLAN_TEMPERATURE", raising=False)


def _chat_response(
    text: str,
    *,
    status_code: int = 200,
    usage: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    body = {
        "id": "gen-test",
        "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "model": "anthropic/claude-haiku-4.5",
        "usage": usage if usage is not None else {"prompt_tokens": 10, "completion_tokens": 5},
    }
    return httpx.Response(status_code, json=body, headers=headers or {})


def _error_response(
    status_code: int, message: str, headers: dict[str, str] | None = None
) -> httpx.Response:
    body = {"error": {"code": status_code, "message": message}}
    return httpx.Response(status_code, json=body, headers=headers or {})


def _client(handler, **kwargs: Any) -> OpenRouterClient:
    transport = httpx.MockTransport(handler)
    return OpenRouterClient(
        spend_key=kwargs.pop("spend_key", "test"),
        api_key=kwargs.pop("api_key", "or-test-key"),
        sink=kwargs.pop("sink", InMemorySink()),
        transport=transport,
        sleep=kwargs.pop("sleep", lambda _seconds: None),
        **kwargs,
    )


def test_request_headers_and_url() -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return _chat_response("hi")

    client = _client(handler, api_key="or-secret-abc")
    client.complete(system="be terse", user="hello", surface="chat")

    assert str(captured[0].url) == CHAT_COMPLETIONS_URL
    assert captured[0].headers["authorization"] == "Bearer or-secret-abc"
    assert captured[0].headers["http-referer"] == "Portolan"
    assert captured[0].headers["x-title"] == "Portolan"


def test_request_body_shape() -> None:
    captured: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return _chat_response("hi")

    client = _client(handler)
    client.complete(system="be terse", user="hello", surface="chat", model="openai/gpt-5.6-luna")

    body = captured[0]
    assert body["model"] == "openai/gpt-5.6-luna"
    assert body["messages"] == [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "hello"},
    ]
    assert body["usage"] == {"include": True}
    assert "response_format" not in body
    assert "provider" not in body


def test_reasoning_disabled_on_every_request() -> None:
    # Fix instructions item 1 / live diagnosis: DeepSeek's real "no JSON
    # object found" failures traced to reasoning silently consuming the
    # whole max_tokens budget before the model ever wrote the answer
    # (confirmed live: finish_reason "length", 4096/4096 reasoning
    # tokens, empty content). Disabling it is safe even for a model with
    # no `reasoning` support at all (OpenRouter ignores it rather than
    # rejecting the request), so it's sent unconditionally.
    captured: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return _chat_response("hi")

    client = _client(handler)
    client.complete(system="s", user="u", surface="chat")

    assert captured[0]["reasoning"] == {"enabled": False}


# --------------------------------------------------------------------------
# temperature (infra-2b wave, 2026-09-18): `01-spec/harness-review-
# 2026-09-18.md` found "temperature is not settable anywhere in this
# checkout's LLM client stack" and left `--temperature` recorded-only.
# --------------------------------------------------------------------------


def test_temperature_defaults_to_zero_on_extract_surface() -> None:
    captured: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return _chat_response("hi")

    client = _client(handler)
    client.complete(system="s", user="u", surface="extract")

    assert captured[0]["temperature"] == 0.0


def test_temperature_defaults_to_zero_on_audit_surface() -> None:
    captured: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return _chat_response("hi")

    client = _client(handler)
    client.complete(system="s", user="u", surface="audit")

    assert captured[0]["temperature"] == 0.0


def test_temperature_omitted_entirely_for_report_and_chat_surfaces() -> None:
    captured: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return _chat_response("hi")

    client = _client(handler)
    client.complete(system="s", user="u", surface="chat")
    client.complete(system="s", user="u", surface="report")

    assert "temperature" not in captured[0]
    assert "temperature" not in captured[1]


def test_explicit_temperature_argument_wins_over_the_surface_default() -> None:
    captured: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return _chat_response("hi")

    client = _client(handler)
    client.complete(system="s", user="u", surface="extract", temperature=0.7)

    assert captured[0]["temperature"] == 0.7


def test_explicit_temperature_argument_applies_even_to_report_surface() -> None:
    captured: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return _chat_response("hi")

    client = _client(handler)
    client.complete(system="s", user="u", surface="report", temperature=0.9)

    assert captured[0]["temperature"] == 0.9


def test_temperature_env_var_overrides_the_zero_default_for_extract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This is the actual seam `python -m portolan.bench extract
    # --temperature <n>` uses to reach the model call (bench/__main__.py's
    # flag plumbing, outside this file's scope) -- verified directly here.
    monkeypatch.setenv("PORTOLAN_TEMPERATURE", "0.4")
    captured: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return _chat_response("hi")

    client = _client(handler)
    client.complete(system="s", user="u", surface="extract")

    assert captured[0]["temperature"] == 0.4


def test_temperature_env_var_has_no_effect_on_report_or_chat_surfaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PORTOLAN_TEMPERATURE", "0.4")
    captured: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return _chat_response("hi")

    client = _client(handler)
    client.complete(system="s", user="u", surface="chat")

    assert "temperature" not in captured[0]


def test_temperature_carried_into_the_json_manual_fallback_body() -> None:
    # STRATEGY_NONE (no response_format at all) uses `_complete_json_
    # fallback`, a different call path from the plain-text one above --
    # confirm `temperature` reaches the request there too.
    from portolan.llm.openrouter import STRATEGY_NONE, _json_strategy_cache

    _json_strategy_cache["amazon/nova-lite-v1"] = STRATEGY_NONE
    captured: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return _chat_response('{"greeting": "hi"}')

    client = _client(handler)
    client.complete(
        system="s", user="u", json_schema=SCHEMA, surface="extract", model="amazon/nova-lite-v1"
    )

    assert captured[0]["temperature"] == 0.0


def test_provider_order_passed_through() -> None:
    captured: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return _chat_response("hi")

    client = _client(handler, provider_order=["anthropic", "together"])
    client.complete(system="s", user="u", surface="chat")

    assert captured[0]["provider"] == {"order": ["anthropic", "together"]}


def test_response_format_sent_when_json_schema_given() -> None:
    captured: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return _chat_response('{"greeting": "hi"}')

    client = _client(handler)
    result = client.complete(system="s", user="u", surface="extract", json_schema=SCHEMA)

    assert captured[0]["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "portolan_response", "strict": True, "schema": SCHEMA},
    }
    assert result.data == {"greeting": "hi"}


def test_cost_recorded_from_response_usage() -> None:
    sink = InMemorySink()

    def handler(request: httpx.Request) -> httpx.Response:
        return _chat_response(
            "hi",
            usage={
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "prompt_tokens_details": {"cached_tokens": 30},
                "cost": 0.00123,
            },
        )

    client = _client(handler, sink=sink)
    result = client.complete(
        system="s", user="u", surface="chat", model="anthropic/claude-haiku-4.5"
    )

    assert result.input_tokens == 100
    assert result.output_tokens == 20
    assert result.cached_tokens == 30
    assert len(sink.events) == 1
    event = sink.events[0]
    assert event.cost_estimate == pytest.approx(0.00123)
    assert event.extra["cost_source"] == "openrouter"


def test_client_cannot_be_built_without_a_spend_key() -> None:
    # CLAUDE.md: every paid call is metered under a named key. Guardrails review
    # 2026-09-22, row 33: three scripts built this client with no meter at all.
    with pytest.raises(TypeError, match="spend_key"):
        OpenRouterClient(api_key="k", sink=InMemorySink())  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="spend_key"):
        OpenRouterClient(spend_key="  ", api_key="k", sink=InMemorySink())


def test_spend_meter_records_every_calls_cost_under_the_spend_key(tmp_path) -> None:
    from portolan.training.spend import SpendMeter

    def handler(request: httpx.Request) -> httpx.Response:
        return _chat_response(
            "hi", usage={"prompt_tokens": 10, "completion_tokens": 2, "cost": 0.004}
        )

    meter = SpendMeter(tmp_path / "spend.json")
    client = _client(handler, spend_key="bench_two_pass", spend_meter=meter)
    client.complete(system="s", user="u", surface="chat", model="anthropic/claude-haiku-4.5")
    client.complete(system="s", user="u", surface="chat", model="anthropic/claude-haiku-4.5")

    snapshot = meter.snapshot()
    assert snapshot["total_usd"] == pytest.approx(0.008)
    assert snapshot["per_corpus_usd"] == {"bench_two_pass": pytest.approx(0.008)}
    assert snapshot["per_model_usd"] == {"anthropic/claude-haiku-4.5": pytest.approx(0.008)}


def test_without_a_spend_meter_the_client_writes_no_meter(tmp_path, monkeypatch) -> None:
    # Callers that meter themselves (silver labeling, the synthetic generator) pass none,
    # so a call is never counted twice.
    monkeypatch.chdir(tmp_path)

    def handler(request: httpx.Request) -> httpx.Response:
        return _chat_response(
            "hi", usage={"prompt_tokens": 10, "completion_tokens": 2, "cost": 0.004}
        )

    client = _client(handler, spend_key="enron")
    client.complete(system="s", user="u", surface="chat", model="anthropic/claude-haiku-4.5")
    assert client.spend_meter is None
    assert client.sink.events[0].cost_estimate == pytest.approx(0.004)
    assert list(tmp_path.iterdir()) == []


def test_cost_falls_back_to_pricing_table_when_response_has_no_cost() -> None:
    sink = InMemorySink()

    def handler(request: httpx.Request) -> httpx.Response:
        usage = {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000}
        return _chat_response("hi", usage=usage)

    client = _client(handler, sink=sink)
    client.complete(system="s", user="u", surface="chat", model="amazon/nova-lite-v1")

    event = sink.events[0]
    assert "cost_source" not in event.extra
    # amazon/nova-lite-v1 pricing row: 0.06 / 0.24 per 1M tokens.
    assert event.cost_estimate == pytest.approx(0.06 + 0.24)


def test_cost_none_with_pricing_error_when_model_unknown_and_no_response_cost() -> None:
    sink = InMemorySink()

    def handler(request: httpx.Request) -> httpx.Response:
        return _chat_response("hi", usage={"prompt_tokens": 10, "completion_tokens": 5})

    client = _client(handler, sink=sink)
    client.complete(system="s", user="u", surface="chat", model="some-vendor/unknown-model")

    event = sink.events[0]
    assert event.cost_estimate is None
    assert "pricing_error" in event.extra


def test_retries_429_then_succeeds() -> None:
    calls = {"n": 0}
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return _error_response(429, "rate limited", headers={"Retry-After": "0"})
        return _chat_response("hi")

    client = _client(handler, sleep=sleeps.append)
    result = client.complete(system="s", user="u", surface="chat")

    assert calls["n"] == 2
    assert result.text == "hi"
    assert sleeps == [0.0]


def test_gives_up_after_max_retries() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _error_response(503, "upstream overloaded")

    client = _client(handler)
    with pytest.raises(LLMError):
        client.complete(system="s", user="u", surface="chat")


def test_falls_back_to_json_util_when_response_format_rejected() -> None:
    calls: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body)
        if "response_format" in body:
            return _error_response(400, "response_format is not supported for this model")
        return _chat_response('{"greeting": "fallback"}')

    client = _client(handler)
    result = client.complete(system="s", user="u", surface="extract", json_schema=SCHEMA)

    assert len(calls) == 2
    assert "response_format" not in calls[1]
    assert result.data == {"greeting": "fallback"}


def test_fallback_retries_once_on_invalid_json() -> None:
    calls: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body)
        if "response_format" in body:
            return _error_response(400, "response_format is not supported for this model")
        if len(calls) == 2:
            return _chat_response("not json")
        return _chat_response('{"greeting": "second try"}')

    client = _client(handler)
    result = client.complete(system="s", user="u", surface="extract", json_schema=SCHEMA)

    assert len(calls) == 3
    assert result.data == {"greeting": "second try"}


def test_fallback_raises_llm_json_error_after_exhausting_retry() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if "response_format" in body:
            return _error_response(400, "response_format is not supported for this model")
        return _chat_response("still not json")

    client = _client(handler)
    with pytest.raises(LLMJsonError):
        client.complete(system="s", user="u", surface="extract", json_schema=SCHEMA)


# --------------------------------------------------------------------------
# Bug hunt 2026-09-18, findings 4/5: `complete()` can make up to three real
# (billed) HTTP requests -- the native attempt, the fallback attempt, and
# the fallback's own retry -- but used to record exactly one UsageEvent for
# the whole call, carrying only the LAST request's tokens/cost. A call that
# fell back or retried lost the earlier requests' real spend; a call that
# failed on the last request recorded $0.00 despite every request having
# been billed.
# --------------------------------------------------------------------------


def test_every_billed_http_request_records_its_own_usage_event() -> None:
    sink = InMemorySink()
    calls: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body)
        if "response_format" in body:
            return _error_response(400, "response_format is not supported for this model")
        if len(calls) == 2:
            # First real (billed) fallback attempt: unparseable.
            return _chat_response(
                "not json", usage={"prompt_tokens": 1000, "completion_tokens": 100, "cost": 0.01}
            )
        # Second fallback attempt (the retry): succeeds.
        return _chat_response(
            '{"greeting": "second try"}',
            usage={"prompt_tokens": 2000, "completion_tokens": 50, "cost": 0.02},
        )

    client = _client(handler, sink=sink)
    result = client.complete(system="s", user="u", surface="extract", json_schema=SCHEMA)

    assert len(calls) == 3  # native rejection + fallback attempt + fallback retry
    assert result.data == {"greeting": "second try"}

    # The native rejection was never billed by OpenRouter -- no event for
    # it. Each of the two real fallback requests gets its OWN event, with
    # its own tokens/cost -- not one event carrying only the last one's.
    assert len(sink.events) == 2
    first, second = sink.events
    assert (first.input_tokens, first.output_tokens) == (1000, 100)
    assert first.cost_estimate == pytest.approx(0.01)
    assert (second.input_tokens, second.output_tokens) == (2000, 50)
    assert second.cost_estimate == pytest.approx(0.02)


def test_a_failing_extraction_still_records_the_real_cost_of_its_billed_retries() -> None:
    sink = InMemorySink()

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if "response_format" in body:
            return _error_response(400, "response_format is not supported for this model")
        return _chat_response(
            "still not json",
            usage={"prompt_tokens": 500, "completion_tokens": 40, "cost": 0.005},
        )

    client = _client(handler, sink=sink)
    with pytest.raises(LLMJsonError):
        client.complete(system="s", user="u", surface="extract", json_schema=SCHEMA)

    # Two real, billed fallback requests (the attempt and its retry), both
    # unparseable -- the call still fails, but neither request's real,
    # billed cost is lost as $0.00.
    assert len(sink.events) == 2
    assert all(event.input_tokens == 500 for event in sink.events)
    assert all(event.cost_estimate == pytest.approx(0.005) for event in sink.events)


def test_exception_message_never_includes_api_key() -> None:
    secret = "or-super-secret-key-99999"

    def handler(request: httpx.Request) -> httpx.Response:
        return _error_response(401, "invalid credentials")

    client = _client(handler, api_key=secret)
    with pytest.raises(LLMError) as exc_info:
        client.complete(system="s", user="u", surface="chat")

    assert secret not in str(exc_info.value)


def test_missing_api_key_raises_clear_runtime_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - never reached
        return _chat_response("hi")

    client = OpenRouterClient(
        spend_key="test",
        api_key=None,
        transport=httpx.MockTransport(handler),
        sink=InMemorySink(),
    )
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        client.complete(system="s", user="u", surface="chat")


def test_default_model() -> None:
    client = OpenRouterClient(spend_key="test", api_key="k", sink=InMemorySink())
    assert client.default_model == "anthropic/claude-haiku-4.5"


def test_default_read_timeout_is_120_seconds() -> None:
    # Fix instructions item 2: "use a 120s read timeout for long
    # transcripts". Constructor default, not just a magic number buried in
    # `_post_with_retry`.
    client = OpenRouterClient(spend_key="test", api_key="k", sink=InMemorySink())
    assert client._timeout.read == 120.0


# --------------------------------------------------------------------------
# Fix item 1: the real openai/gpt-5.6-luna rejection (live-checked
# 2026-09-12) -- OpenRouter's top-level `error.message` is the generic
# "Provider returned error"; the actual "your schema is invalid for
# response_format" detail is nested in `error.metadata.raw`. The pre-fix
# `_is_response_format_rejection` only ever looked at the top-level
# message, so this exact shape used to fall straight through to
# `_raise_for_status` and abort the whole source.
# --------------------------------------------------------------------------


def _luna_style_rejection() -> httpx.Response:
    body = {
        "error": {
            "message": "Provider returned error",
            "code": 400,
            "metadata": {
                "raw": (
                    '{"error": {"message": "Invalid schema for response_format '
                    "'portolan_response': In context=(), 'required' is required to be "
                    "supplied and to be an array including every key in properties. "
                    'Missing \'owner_mention\'.", "type": "invalid_request_error", '
                    '"param": "text.format.schema", "code": "invalid_json_schema"}}'
                ),
                "provider_name": "Azure",
                "previous_errors": [
                    {"code": 400, "message": "Provider returned error", "provider_name": "OpenAI"}
                ],
            },
        }
    }
    return httpx.Response(400, json=body)


def test_falls_back_when_rejection_detail_is_nested_in_metadata_raw() -> None:
    calls: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body)
        if "response_format" in body:
            return _luna_style_rejection()
        return _chat_response('{"greeting": "fallback worked"}')

    client = _client(handler)
    result = client.complete(
        system="s", user="u", surface="extract", json_schema=SCHEMA, model="openai/gpt-5.6-luna"
    )

    assert len(calls) == 2
    assert "response_format" not in calls[1]
    assert result.data == {"greeting": "fallback worked"}


def test_metadata_raw_rejection_downgrades_cache_for_next_call() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if "response_format" in body and body["response_format"].get("type") == "json_schema":
            return _luna_style_rejection()
        return _chat_response('{"greeting": "hi"}')

    client = _client(handler)
    client.complete(
        system="s", user="u", surface="extract", json_schema=SCHEMA, model="openai/gpt-5.6-luna"
    )

    assert openrouter_module._json_strategy_cache["openai/gpt-5.6-luna"] == STRATEGY_OBJECT

    # A second call for the same model id no longer wastes a doomed
    # json_schema attempt -- it goes straight to json_object, which this
    # handler accepts on the first try.
    captured: list[dict[str, Any]] = []

    def handler2(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured.append(body)
        return _chat_response('{"greeting": "second call"}')

    client2 = _client(handler2)
    result = client2.complete(
        system="s", user="u", surface="extract", json_schema=SCHEMA, model="openai/gpt-5.6-luna"
    )

    assert len(captured) == 1
    assert captured[0]["response_format"] == {"type": "json_object"}
    assert result.data == {"greeting": "second call"}


def test_non_rejection_400_still_raises_with_full_provider_detail() -> None:
    # A genuine, unrelated 400 (bad api key format, say) must NOT be
    # swallowed into the fallback path just because it also happens to be
    # wrapped in the same generic "Provider returned error" + metadata.raw
    # shape -- only a rejection whose detail actually names the feature
    # counts.
    def handler(request: httpx.Request) -> httpx.Response:
        body = {
            "error": {
                "message": "Provider returned error",
                "metadata": {"raw": '{"error": {"message": "insufficient quota"}}'},
            }
        }
        return httpx.Response(400, json=body)

    client = _client(handler)
    with pytest.raises(LLMError) as exc_info:
        client.complete(system="s", user="u", surface="chat")

    # Fix instructions item 2: the surfaced error is diagnosable -- status
    # plus the real provider detail, not just "Provider returned error".
    assert "400" in str(exc_info.value)
    assert "insufficient quota" in str(exc_info.value)


# --------------------------------------------------------------------------
# Fix item 1: the fallback path's local schema check must not be stricter
# than parse_lenient downstream -- additionalProperties: false is dropped
# for the manual system-instruction + json_util path (never for a native
# response_format: json_schema request).
# --------------------------------------------------------------------------


def test_native_schema_attempt_keeps_the_strict_schema_as_is() -> None:
    captured: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return _chat_response('{"greeting": "hi"}')

    client = _client(handler)
    client.complete(system="s", user="u", surface="extract", json_schema=SCHEMA)

    sent_schema = captured[0]["response_format"]["json_schema"]["schema"]
    assert sent_schema == SCHEMA
    assert sent_schema["additionalProperties"] is False


def test_fallback_schema_drops_additional_properties_false() -> None:
    calls: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body)
        if "response_format" in body:
            return _error_response(400, "response_format is not supported for this model")
        return _chat_response('{"greeting": "hi", "extra_field": "slop"}')

    client = _client(handler)
    result = client.complete(system="s", user="u", surface="extract", json_schema=SCHEMA)

    # A strict local check (SCHEMA's own additionalProperties: false) would
    # have rejected `extra_field` and raised LLMJsonError after exhausting
    # the retry -- this must succeed instead, since the fallback path uses
    # the lenient schema; the caller (LLMClientBackend) has its own,
    # stricter pydantic validate + retry + parse_lenient chain for facts
    # that really are malformed.
    assert result.data == {"greeting": "hi", "extra_field": "slop"}

    fallback_user_message = calls[1]["messages"][0]["content"]
    assert '"additionalProperties": false' not in fallback_user_message


# --------------------------------------------------------------------------
# Fix item 4: JSON-strategy selection from supported_parameters, and the
# process-wide cache.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("supported_parameters", "expected"),
    [
        (["max_tokens", "structured_outputs", "response_format", "tools"], STRATEGY_SCHEMA),
        (["max_tokens", "response_format", "tools"], STRATEGY_OBJECT),
        (["max_tokens", "temperature", "top_p", "top_k", "stop", "tools"], STRATEGY_NONE),
        (None, STRATEGY_SCHEMA),
        ([], STRATEGY_SCHEMA),
    ],
)
def test_choose_json_strategy(supported_parameters, expected) -> None:
    assert _choose_json_strategy(supported_parameters) == expected


def test_fetch_supported_parameters_unions_every_endpoint() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == MODEL_ENDPOINTS_URL_TEMPLATE.format(model="deepseek/x")
        return httpx.Response(
            200,
            json={
                "data": {
                    "id": "deepseek/x",
                    "endpoints": [
                        {"provider_name": "A", "supported_parameters": ["response_format"]},
                        {
                            "provider_name": "B",
                            "supported_parameters": ["structured_outputs", "response_format"],
                        },
                    ],
                }
            },
        )

    params = fetch_supported_parameters(
        "deepseek/x", api_key="k", transport=httpx.MockTransport(handler)
    )
    assert params == ["response_format", "structured_outputs"]


@pytest.mark.parametrize(
    "handler",
    [
        lambda request: httpx.Response(404, json={"error": "not found"}),
        lambda request: httpx.Response(200, text="not json"),
        lambda request: httpx.Response(200, json={"data": {"endpoints": []}}),
    ],
)
def test_fetch_supported_parameters_returns_none_on_failure(handler) -> None:
    params = fetch_supported_parameters(
        "some/model", api_key="k", transport=httpx.MockTransport(handler)
    )
    assert params is None


def test_fetch_supported_parameters_never_raises_on_transport_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no network", request=request)

    assert (
        fetch_supported_parameters(
            "some/model", api_key="k", transport=httpx.MockTransport(handler)
        )
        is None
    )


def test_prime_json_strategy_caches_and_is_idempotent() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(
            200,
            json={"data": {"endpoints": [{"supported_parameters": ["response_format"]}]}},
        )

    transport = httpx.MockTransport(handler)
    first = prime_json_strategy("amazon/nova-lite-v1", api_key="k", transport=transport)
    second = prime_json_strategy("amazon/nova-lite-v1", api_key="k", transport=transport)

    assert first == second == STRATEGY_OBJECT
    assert calls["n"] == 1  # second call was a cache hit, no network


def test_prime_json_strategy_force_refetches() -> None:
    responses = iter(
        [
            {"data": {"endpoints": [{"supported_parameters": ["response_format"]}]}},
            {"data": {"endpoints": [{"supported_parameters": ["structured_outputs"]}]}},
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=next(responses))

    transport = httpx.MockTransport(handler)
    first = prime_json_strategy("some/model", api_key="k", transport=transport)
    second = prime_json_strategy("some/model", api_key="k", transport=transport, force=True)

    assert first == STRATEGY_OBJECT
    assert second == STRATEGY_SCHEMA


def test_object_strategy_sends_json_object_without_a_schema() -> None:
    openrouter_module._json_strategy_cache["some/object-model"] = STRATEGY_OBJECT
    captured: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return _chat_response('{"greeting": "hi"}')

    client = _client(handler)
    result = client.complete(
        system="s", user="u", surface="extract", json_schema=SCHEMA, model="some/object-model"
    )

    assert len(captured) == 1
    assert captured[0]["response_format"] == {"type": "json_object"}
    assert result.data == {"greeting": "hi"}


def test_object_strategy_puts_the_schema_in_the_system_prompt() -> None:
    openrouter_module._json_strategy_cache["some/object-model"] = STRATEGY_OBJECT
    captured: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return _chat_response('{"greeting": "hi"}')

    _client(handler).complete(
        system="s", user="u", surface="extract", json_schema=SCHEMA, model="some/object-model"
    )
    system = captured[0]["messages"][0]["content"]
    assert system.startswith("s")
    assert "JSON Schema:" in system
    assert '"greeting"' in system


def test_none_strategy_skips_the_native_attempt_entirely() -> None:
    openrouter_module._json_strategy_cache["some/no-json-model"] = STRATEGY_NONE
    captured: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return _chat_response('{"greeting": "hi"}')

    client = _client(handler)
    result = client.complete(
        system="s", user="u", surface="extract", json_schema=SCHEMA, model="some/no-json-model"
    )

    assert len(captured) == 1  # no wasted first attempt
    assert "response_format" not in captured[0]
    assert result.data == {"greeting": "hi"}


def test_warm_json_strategy_uses_clients_own_transport_and_key() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer or-warm-key"
        return httpx.Response(
            200, json={"data": {"endpoints": [{"supported_parameters": ["structured_outputs"]}]}}
        )

    client = _client(handler, api_key="or-warm-key")
    strategy = client.warm_json_strategy("some/warm-model")

    assert strategy == STRATEGY_SCHEMA
    assert openrouter_module._json_strategy_cache["some/warm-model"] == STRATEGY_SCHEMA


# --------------------------------------------------------------------------
# Fix item 2: retry budget, jitter, 502/503/504, and read timeouts.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("status", [429, 502, 503, 504])
def test_retries_up_to_five_times_on_retryable_statuses(status: int) -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] <= 5:
            return _error_response(status, "upstream trouble")
        return _chat_response("hi")

    client = _client(handler)
    result = client.complete(system="s", user="u", surface="chat")

    assert calls["n"] == 6  # 5 failures + 1 success, all within budget
    assert result.text == "hi"


def test_gives_up_after_five_retries_not_three() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return _error_response(503, "still overloaded")

    client = _client(handler)
    with pytest.raises(LLMError):
        client.complete(system="s", user="u", surface="chat")

    assert calls["n"] == 6  # 1 initial attempt + 5 retries


def test_jitter_function_is_applied_to_computed_backoff() -> None:
    sleeps: list[float] = []
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return _error_response(503, "overloaded")  # no Retry-After header
        return _chat_response("hi")

    client = _client(handler, sleep=sleeps.append, jitter=lambda base: base * 10)
    client.complete(system="s", user="u", surface="chat")

    # attempt 0 backoff is _BASE_BACKOFF_SECONDS * 2**0 == 1.0, times the
    # injected jitter multiplier.
    assert sleeps == [10.0]


def test_read_timeout_is_retried_then_succeeds() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ReadTimeout("timed out", request=request)
        return _chat_response("hi")

    client = _client(handler)
    result = client.complete(system="s", user="u", surface="chat")

    assert calls["n"] == 2
    assert result.text == "hi"


def test_read_timeout_exhausts_retry_budget_and_raises_clear_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    client = _client(handler)
    with pytest.raises(LLMError, match="timed out"):
        client.complete(system="s", user="u", surface="chat")


def test_connection_reset_is_retried_then_succeeds() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ReadError("[Errno 54] Connection reset by peer", request=request)
        return _chat_response("hi")

    result = _client(handler).complete(system="s", user="u", surface="chat")
    assert calls["n"] == 2
    assert result.text == "hi"


def test_connection_reset_exhausts_retry_budget_and_raises_llm_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadError("[Errno 54] Connection reset by peer", request=request)

    with pytest.raises(LLMError, match="request failed after 6 attempt"):
        _client(handler).complete(system="s", user="u", surface="chat")


# --------------------------------------------------------------------------
# Fix instructions item 1: gpt-oss-20b/120b real bench failure -- every one
# of 78 sources 400'd in 3 seconds on the unconditional `reasoning:
# {enabled: false}` this client sends on every request. "Reasoning is
# mandatory for this endpoint and cannot be disabled."
# --------------------------------------------------------------------------


def _reasoning_mandatory_rejection() -> httpx.Response:
    return _error_response(400, "Reasoning is mandatory for this endpoint and cannot be disabled.")


def test_reasoning_mandatory_rejection_is_retried_without_the_reasoning_field() -> None:
    calls: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body)
        if "reasoning" in body:
            return _reasoning_mandatory_rejection()
        return _chat_response('{"greeting": "hi"}')

    client = _client(handler)
    result = client.complete(
        system="s", user="u", surface="extract", json_schema=SCHEMA, model="openai/gpt-oss-20b"
    )

    assert len(calls) == 2
    assert "reasoning" in calls[0]
    assert "reasoning" not in calls[1]
    assert result.data == {"greeting": "hi"}


def test_reasoning_mandatory_rejection_does_not_consume_the_retry_budget() -> None:
    # The reasoning-shape fix must not eat into the 429/5xx retry budget --
    # it's a request-shape correction, not congestion.
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls["n"] += 1
        if "reasoning" in body:
            return _reasoning_mandatory_rejection()
        return _error_response(503, "upstream overloaded")

    client = _client(handler)
    with pytest.raises(LLMError):
        client.complete(system="s", user="u", surface="chat", model="openai/gpt-oss-20b")

    # 1 reasoning-shape retry (free) + 1 initial 503 attempt + 5 more
    # retries (the full _MAX_RETRIES budget) = 7.
    assert calls["n"] == 7


def test_reasoning_locked_learned_from_one_call_skips_the_field_on_the_next() -> None:
    def first_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if "reasoning" in body:
            return _reasoning_mandatory_rejection()
        return _chat_response("hi")

    client = _client(first_handler)
    client.complete(system="s", user="u", surface="chat", model="openai/gpt-oss-120b")

    from portolan.llm import openrouter as _openrouter_module

    assert "openai/gpt-oss-120b" in _openrouter_module._reasoning_locked_models

    # A second call for the same model id, via a brand-new client (mirrors
    # `run_extraction`'s one-client-per-worker-thread pattern), never even
    # attempts the doomed field -- no 400 needed.
    captured: list[dict[str, Any]] = []

    def second_handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return _chat_response("hi again")

    client2 = _client(second_handler)
    result = client2.complete(system="s", user="u", surface="chat", model="openai/gpt-oss-120b")

    assert len(captured) == 1  # no wasted first attempt
    assert "reasoning" not in captured[0]
    assert result.text == "hi again"


def test_reasoning_off_models_env_var_preempts_the_first_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "PORTOLAN_OPENROUTER_REASONING_OFF_MODELS", "other/model, openai/gpt-oss-20b"
    )
    captured: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return _chat_response("hi")

    client = _client(handler)
    client.complete(system="s", user="u", surface="chat", model="openai/gpt-oss-20b")

    assert len(captured) == 1  # never even attempted the disable
    assert "reasoning" not in captured[0]


def test_reasoning_off_models_env_var_does_not_affect_other_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PORTOLAN_OPENROUTER_REASONING_OFF_MODELS", "openai/gpt-oss-20b")
    captured: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return _chat_response("hi")

    client = _client(handler)
    client.complete(system="s", user="u", surface="chat", model="anthropic/claude-haiku-4.5")

    assert captured[0]["reasoning"] == {"enabled": False}


def test_reasoning_effort_env_var_sends_effort_instead_of_disable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Model-sweep wave, 2026-09-18: a reasoning-only model
    # (moonshotai/kimi-k2-thinking) declared via
    # PORTOLAN_OPENROUTER_REASONING_EFFORT gets OpenRouter's unified
    # `reasoning: {"effort": ...}` instead of this client's default
    # `enabled: false` -- and the disable is never even attempted.
    monkeypatch.setenv("PORTOLAN_OPENROUTER_REASONING_EFFORT", "moonshotai/kimi-k2-thinking=low")
    captured: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return _chat_response("hi")

    client = _client(handler)
    client.complete(system="s", user="u", surface="chat", model="moonshotai/kimi-k2-thinking")

    assert len(captured) == 1
    assert captured[0]["reasoning"] == {"effort": "low"}


def test_reasoning_effort_env_var_does_not_affect_other_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PORTOLAN_OPENROUTER_REASONING_EFFORT", "moonshotai/kimi-k2-thinking=low")
    captured: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return _chat_response("hi")

    client = _client(handler)
    client.complete(system="s", user="u", surface="chat", model="anthropic/claude-haiku-4.5")

    assert captured[0]["reasoning"] == {"enabled": False}


def test_reasoning_effort_env_var_parses_multiple_pairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from portolan.llm.openrouter import _env_reasoning_effort_overrides

    monkeypatch.setenv(
        "PORTOLAN_OPENROUTER_REASONING_EFFORT",
        " moonshotai/kimi-k2-thinking=low , other/model=medium",
    )
    assert _env_reasoning_effort_overrides() == {
        "moonshotai/kimi-k2-thinking": "low",
        "other/model": "medium",
    }


def test_non_reasoning_400_is_unaffected_by_the_reasoning_fallback() -> None:
    # A genuine, unrelated 400 that happens to still carry `reasoning` in
    # the request body must not be misread as the mandatory-reasoning
    # rejection just because both are 400s.
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return _error_response(400, "invalid model id")

    client = _client(handler)
    with pytest.raises(LLMError, match="invalid model id"):
        client.complete(system="s", user="u", surface="chat", model="openai/gpt-oss-20b")

    assert calls["n"] == 1  # no wasted reasoning-fallback retry


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("Reasoning is mandatory for this endpoint and cannot be disabled.", True),
        ("insufficient quota", False),
        ("reasoning tokens exceeded the budget", False),  # "reasoning" alone isn't enough
        ("this field cannot be disabled: temperature", False),  # keyword alone isn't enough
    ],
)
def test_is_reasoning_mandatory_rejection(message: str, expected: bool) -> None:
    assert _is_reasoning_mandatory_rejection(_error_response(400, message)) is expected


# --------------------------------------------------------------------------
# Fix instructions item 2 (openrouter.py's half): the real Nova Lite/Micro
# bench losses that showed up as LLMJsonError, not ValidationError -- the
# model omitted `confidence` on an entity, or used a `kind` outside the
# closed list, and THIS client's own local `extract_and_validate` check
# (not portolan.extract.schema.parse_lenient, which never got the chance)
# failed the whole source over it. `lenient_schema` must be relaxed enough
# that this client's own gate doesn't pre-empt what parse_lenient can
# actually fix downstream.
# --------------------------------------------------------------------------


# A miniature stand-in for the real `entities` shape in
# `portolan.extract.schema.extraction_result_json_schema()` -- `required`
# naming `confidence` and a closed `enum` on `kind`, exactly the two gates
# that failed the real Nova Lite/Micro bench sources locally in this
# client, before the data ever reached `parse_lenient`.
ENTITY_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "mention": {"type": "string"},
                    "kind": {
                        "type": "string",
                        "enum": [
                            "account",
                            "contact",
                            "product",
                            "competitor",
                            "internal_person",
                            "other",
                        ],
                    },
                    "confidence": {"type": "number"},
                },
                "required": ["mention", "kind", "confidence"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["entities"],
    "additionalProperties": False,
}


def test_fallback_tolerates_entity_missing_confidence() -> None:
    # Real observed failure (amazon/nova-micro-v1, S-call-n01/S-email-n012):
    # `'confidence' is a required property`, failing
    # `schema['properties']['entities']['items']['required']`.
    openrouter_module._json_strategy_cache["some/no-json-model"] = STRATEGY_NONE

    def handler(request: httpx.Request) -> httpx.Response:
        return _chat_response('{"entities": [{"mention": "x", "kind": "product"}]}')

    client = _client(handler)
    result = client.complete(
        system="s",
        user="u",
        surface="extract",
        json_schema=ENTITY_SCHEMA,
        model="some/no-json-model",
    )

    assert result.data["entities"] == [{"mention": "x", "kind": "product"}]


def test_fallback_tolerates_entity_kind_outside_closed_list() -> None:
    # Real observed failure (amazon/nova-micro-v1, S-ticket-019):
    # `'service' is not one of [...]`, failing
    # `schema['properties']['entities']['items']['properties']['kind']['enum']`.
    openrouter_module._json_strategy_cache["some/no-json-model"] = STRATEGY_NONE

    def handler(request: httpx.Request) -> httpx.Response:
        return _chat_response(
            '{"entities": [{"mention": "x", "kind": "service", "confidence": 0.9}]}'
        )

    client = _client(handler)
    result = client.complete(
        system="s",
        user="u",
        surface="extract",
        json_schema=ENTITY_SCHEMA,
        model="some/no-json-model",
    )

    assert result.data["entities"][0]["kind"] == "service"


# --------------------------------------------------------------------------
# `validate_json` (fix wave 2026-09-12): leniency belongs in ONE place --
# `portolan.extract.schema.ExtractionResult.parse_lenient` -- not doubled up
# in this client's own local jsonschema gate. Real bench evidence: all 18
# amazon/nova-lite-v1 failures in var/bench/compare3/amazon_nova-lite-v1/
# bench-2026-09-12-openrouter.json were this exact shape -- a fact's
# `entities` item came back as a nested top-level-EntityMention object
# (`{"mention": ..., "kind": ..., "registry_id": ...}`) where the schema
# (mirroring `Fact.entities: list[str]`) wants a plain string, raising
# `LLMJsonError` from inside this client -- one layer before
# `parse_lenient`, which already knows how to coerce this exact shape,
# ever got a chance to run.
# --------------------------------------------------------------------------

# A miniature stand-in for the real `facts[].entities` shape (`Fact.
# entities: list[str]`) -- the actual gate that rejected all 18 Nova Lite
# failures, distinct from ENTITY_SCHEMA above (which mirrors the top-level
# `entities` list, `list[EntityMention]`).
FACT_ENTITY_STRING_SCHEMA = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "entities": {"type": "array", "items": {"type": "string"}},
                    "sentiment": {"type": "integer"},
                },
                "required": ["id", "entities", "sentiment"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["facts"],
    "additionalProperties": False,
}

# Real observed shape (S-call-002): entities[0] nested instead of a plain
# string, sentiment left null.
_SHAPE_MISMATCHED_REPLY = json.dumps(
    {
        "facts": [
            {
                "id": "f1",
                "entities": [
                    {
                        "mention": "Priya Natarajan",
                        "kind": "internal_person",
                        "registry_id": "internal:priya-natarajan",
                    }
                ],
                "sentiment": None,
            }
        ]
    }
)


def test_validate_json_false_returns_shape_mismatched_reply_without_raising() -> None:
    openrouter_module._json_strategy_cache["amazon/nova-lite-v1"] = STRATEGY_NONE

    def handler(request: httpx.Request) -> httpx.Response:
        return _chat_response(_SHAPE_MISMATCHED_REPLY)

    client = _client(handler)
    result = client.complete(
        system="s",
        user="u",
        surface="extract",
        json_schema=FACT_ENTITY_STRING_SCHEMA,
        model="amazon/nova-lite-v1",
        validate_json=False,
    )

    # Returned as-is -- this client's job stopped at "is this a JSON
    # object", not "does it conform"; coercing the nested entity object
    # and the null sentiment is parse_lenient's job, untouched by this
    # client.
    assert result.data["facts"][0]["entities"] == [
        {
            "mention": "Priya Natarajan",
            "kind": "internal_person",
            "registry_id": "internal:priya-natarajan",
        }
    ]
    assert result.data["facts"][0]["sentiment"] is None


def test_validate_json_true_default_still_raises_on_the_same_shape_mismatch() -> None:
    # The exact real failure this fix was diagnosed from, reproduced
    # against the default (unchanged) validate_json=True behavior: a
    # caller that doesn't opt out (every existing caller of
    # complete(json_schema=...) -- merge/facts.py, audit/judge.py,
    # web/answer.py, report/*.py) keeps getting today's client-side
    # validation gate exactly as before.
    openrouter_module._json_strategy_cache["amazon/nova-lite-v1"] = STRATEGY_NONE

    def handler(request: httpx.Request) -> httpx.Response:
        return _chat_response(_SHAPE_MISMATCHED_REPLY)

    client = _client(handler)
    with pytest.raises(LLMJsonError):
        client.complete(
            system="s",
            user="u",
            surface="extract",
            json_schema=FACT_ENTITY_STRING_SCHEMA,
            model="amazon/nova-lite-v1",
        )


def test_validate_json_false_still_retries_once_when_nothing_extracts_at_all() -> None:
    # validate_json=False only relaxes "does it conform to the schema" --
    # "is there a JSON object here at all" still gets the one retry.
    openrouter_module._json_strategy_cache["amazon/nova-lite-v1"] = STRATEGY_NONE
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append("x")
        if len(calls) == 1:
            return _chat_response("not json at all")
        return _chat_response('{"facts": []}')

    client = _client(handler)
    result = client.complete(
        system="s",
        user="u",
        surface="extract",
        json_schema=FACT_ENTITY_STRING_SCHEMA,
        model="amazon/nova-lite-v1",
        validate_json=False,
    )

    assert len(calls) == 2
    assert result.data == {"facts": []}


def test_validate_json_false_raises_after_exhausting_retry_with_no_object_at_all() -> None:
    openrouter_module._json_strategy_cache["amazon/nova-lite-v1"] = STRATEGY_NONE

    def handler(request: httpx.Request) -> httpx.Response:
        return _chat_response("still not json")

    client = _client(handler)
    with pytest.raises(LLMJsonError):
        client.complete(
            system="s",
            user="u",
            surface="extract",
            json_schema=FACT_ENTITY_STRING_SCHEMA,
            model="amazon/nova-lite-v1",
            validate_json=False,
        )


def test_validate_json_false_skips_fallback_when_native_schema_attempt_extracts_something() -> None:
    # STRATEGY_SCHEMA's native response_format request is untouched by
    # validate_json (still sent with strict: True) -- only the LOCAL
    # post-response gate this client applies to the reply is affected.
    # With validate_json=False, a 2xx reply that merely extracts (even if
    # it wouldn't have validated) is accepted directly, with no second
    # round-trip to the manual-instruction fallback.
    calls: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        return _chat_response(_SHAPE_MISMATCHED_REPLY)

    client = _client(handler)
    result = client.complete(
        system="s",
        user="u",
        surface="extract",
        json_schema=FACT_ENTITY_STRING_SCHEMA,
        validate_json=False,
    )

    assert len(calls) == 1
    assert "response_format" in calls[0]
    assert result.data["facts"][0]["sentiment"] is None


# --------------------------------------------------------------------------
# Fix item 3: `using_sink` reroutes metering for the duration of a `with`
# block, restoring the original sink/tenant afterward (even on error).
# --------------------------------------------------------------------------


def test_using_sink_reroutes_metering_for_the_duration_of_the_block() -> None:
    default_sink = InMemorySink()
    client = _client(lambda request: _chat_response("hi"), sink=default_sink)
    client.tenant = "default-tenant"

    bench_sink = InMemorySink()
    with client.using_sink(bench_sink, "bench-tenant"):
        client.complete(system="s", user="u", surface="extract")

    assert len(bench_sink.events) == 1
    assert bench_sink.events[0].tenant == "bench-tenant"
    assert len(default_sink.events) == 0

    # Restored afterward -- a call outside the `with` block goes back to
    # the client's own sink/tenant.
    client.complete(system="s", user="u", surface="chat")
    assert len(default_sink.events) == 1
    assert default_sink.events[0].tenant == "default-tenant"


def test_using_sink_restores_original_sink_even_if_the_block_raises() -> None:
    default_sink = InMemorySink()

    def handler(request: httpx.Request) -> httpx.Response:
        return _error_response(401, "bad key")

    client = _client(handler, sink=default_sink)
    bench_sink = InMemorySink()

    with pytest.raises(LLMError), client.using_sink(bench_sink, "bench-tenant"):
        client.complete(system="s", user="u", surface="chat")

    assert client.sink is default_sink
    assert len(bench_sink.events) == 1  # the failed call was still metered, into bench_sink


# --------------------------------------------------------------------------
# Corpus-fix wave, 2026-09-19 (`01-spec/corpus-fix-2026-09-19.md`): the
# production pipeline extracted 0 facts against `amazon/nova-lite-v1` while
# the bench, same model/prompt/sample count, extracted hundreds. The real
# cause: `amazon/nova-lite-v1` has no `response_format` in its OpenRouter
# `supported_parameters` at all, so a `response_format: json_schema`
# request against it gets routed through Amazon Bedrock's native tool-
# calling shim -- which, for this project's deeply nested schema (an array
# of multi-field fact objects), intermittently failed to collapse
# Bedrock's internal typed "Document" representation before handing the
# reply back, leaking it as a JSON-encoded STRING inside what the schema
# expects to be a plain nested object. The two samples below are real
# responses saved from that leak (`portolan_reseed_20260919`'s
# `usage_events.extra.rejected_facts`, `raw={"value": <one of these
# strings>}` -- exactly `ExtractionResult.parse_lenient` rejecting a
# `facts` item that arrived as a string instead of a dict).
# --------------------------------------------------------------------------

# Real leaked response, saved verbatim (S-call-002, "f1" -- the training-
# session-request fact).
NOVA_TREE_SAMPLE_1 = (
    '{"completionState":"complete","entries":[["id",{"completionState":"complete",'
    '"type":"String","value":"f1"}],["claim",{"completionState":"complete","type":'
    '"String","value":"Nina Vale from Vireo Surgical Group requested to schedule a '
    'training session for new OR staff."}],["type",{"completionState":"complete",'
    '"type":"String","value":"commitment"}],["sentiment",{"completionState":'
    '"complete","type":"String","value":"neutral"}],["confidence",'
    '{"completionState":"complete","type":"String","value":"high"}],["entities",'
    '{"completionState":"complete","items":[{"completionState":"complete","type":'
    '"String","value":"Nina Vale"},{"completionState":"complete","type":"String",'
    '"value":"Vireo Surgical Group"},{"completionState":"complete","type":"String",'
    '"value":"Priya Natarajan"},{"completionState":"complete","type":"String",'
    '"value":"Halvard Medical"}],"type":"Array"}],["locator",{"completionState":'
    '"complete","type":"String","value":"09:10:29"}],["quote",{"completionState":'
    '"complete","type":"String","value":"I wanted to touch base about scheduling '
    'the training session for our new OR staff."}]],"type":"Object"}'
)

# Real leaked response, saved verbatim (S-call-002, "f1" -- the follow-up
# commitment fact; also exercises a null leaf, `valid_to`, which -- unlike
# every other node in these samples -- has no `completionState` key at
# all, proving the flattener doesn't require it at every recursion level,
# only at the outermost node it's asked to classify).
NOVA_TREE_SAMPLE_2 = (
    '{"completionState":"complete","entries":[["id",{"completionState":"complete",'
    '"type":"String","value":"f1"}],["type",{"completionState":"complete","type":'
    '"String","value":"commitment"}],["claim",{"completionState":"complete","type":'
    '"String","value":"Priya Natarajan will send over the available training '
    'session slots after the call."}],["sentiment",{"completionState":"complete",'
    '"type":"String","value":"neutral"}],["confidence",{"completionState":'
    '"complete","type":"String","value":"high"}],["source_date",{"completionState":'
    '"complete","type":"String","value":"2026-03-19"}],["locator",'
    '{"completionState":"complete","type":"String","value":"09:11:35"}],["quote",'
    '{"completionState":"complete","type":"String","value":"I can send over the '
    'open slots after we finish, and you can circulate them internally."}],'
    '["entities",{"completionState":"complete","items":[{"completionState":'
    '"complete","type":"String","value":"Nina Vale"},{"completionState":"complete",'
    '"type":"String","value":"Priya Natarajan"}],"type":"Array"}],["valid_from",'
    '{"completionState":"complete","type":"String","value":"2026-03-19"}],'
    '["valid_to",{"type":"Null"}]],"type":"Object"}'
)


def test_flatten_document_tree_node_converts_saved_sample_1() -> None:
    node = json.loads(NOVA_TREE_SAMPLE_1)
    assert _is_document_tree_node(node)

    flattened = _flatten_document_tree_node(node)

    assert flattened == {
        "id": "f1",
        "claim": (
            "Nina Vale from Vireo Surgical Group requested to schedule a "
            "training session for new OR staff."
        ),
        "type": "commitment",
        "sentiment": "neutral",
        "confidence": "high",
        "entities": ["Nina Vale", "Vireo Surgical Group", "Priya Natarajan", "Halvard Medical"],
        "locator": "09:10:29",
        "quote": (
            "I wanted to touch base about scheduling the training session for our new OR staff."
        ),
    }


def test_flatten_document_tree_node_converts_saved_sample_2_including_null_leaf() -> None:
    node = json.loads(NOVA_TREE_SAMPLE_2)
    assert _is_document_tree_node(node)

    flattened = _flatten_document_tree_node(node)

    assert flattened == {
        "id": "f1",
        "type": "commitment",
        "claim": (
            "Priya Natarajan will send over the available training session slots after the call."
        ),
        "sentiment": "neutral",
        "confidence": "high",
        "source_date": "2026-03-19",
        "locator": "09:11:35",
        "quote": (
            "I can send over the open slots after we finish, and you can circulate them internally."
        ),
        "entities": ["Nina Vale", "Priya Natarajan"],
        "valid_from": "2026-03-19",
        "valid_to": None,
    }


def test_is_document_tree_node_requires_completion_state_not_just_a_type_key() -> None:
    # This project's own schema uses `type` as a real field name (`Fact.
    # type`, the fact-type string) -- a plain, correctly-shaped fact dict
    # must never be misread as a leaked tree node just because it has a
    # `type` key.
    real_fact = {"id": "f1", "type": "commitment", "claim": "...", "entities": ["A"]}
    assert not _is_document_tree_node(real_fact)


def test_deep_flatten_document_trees_converts_stringified_trees_inside_a_facts_array() -> None:
    # The actual production shape: the top-level reply IS the correct
    # `{"facts": [...], ...}` object -- but each array item is a
    # JSON-encoded STRING of the tree instead of a plain object (see
    # `ExtractionResult.parse_lenient`'s `raw={"value": raw_fact}` for a
    # non-dict `facts` item, which is exactly what a string there produces
    # today, one call before this recovery has a chance to run).
    raw_data = {
        "schema_version": "0.2",
        "source_id": "S-call-002",
        "facts": [NOVA_TREE_SAMPLE_1, NOVA_TREE_SAMPLE_2],
        "entities": [],
    }

    converted, changed = _deep_flatten_document_trees(raw_data)

    assert changed is True
    assert converted["schema_version"] == "0.2"  # untouched plain fields pass through
    assert converted["source_id"] == "S-call-002"
    assert isinstance(converted["facts"][0], dict)
    assert isinstance(converted["facts"][1], dict)
    assert converted["facts"][0]["id"] == "f1"
    assert converted["facts"][0]["claim"].startswith("Nina Vale from Vireo Surgical Group")
    assert converted["facts"][1]["valid_to"] is None


def test_deep_flatten_document_trees_leaves_ordinary_data_untouched() -> None:
    ordinary = {
        "facts": [{"id": "f1", "type": "commitment", "claim": "A plain claim.", "entities": ["A"]}],
        "note": "a string that starts with a brace but isn't JSON: {not json",
    }

    converted, changed = _deep_flatten_document_trees(ordinary)

    assert changed is False
    assert converted == ordinary


def test_client_recovers_facts_from_a_tree_shaped_response() -> None:
    # End to end through OpenRouterClient.complete(): the native
    # response_format: json_schema attempt (STRATEGY_SCHEMA, today's
    # optimistic default for a model never primed via warm_json_strategy
    # -- exactly production's own gap this wave also fixed in
    # portolan/pipeline.py) comes back 200 with a leaked tree instead of
    # plain JSON. The client must recover the fact rather than storing an
    # empty result for it.
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.dumps({"schema_version": "0.2", "facts": [NOVA_TREE_SAMPLE_1]})
        return _chat_response(body)

    client = _client(handler)
    result = client.complete(
        system="s",
        user="u",
        surface="extract",
        json_schema=FACT_ENTITY_STRING_SCHEMA,
        model="amazon/nova-lite-v1",
        validate_json=False,
    )

    assert isinstance(result.data["facts"][0], dict)
    assert result.data["facts"][0]["id"] == "f1"
    assert result.data["facts"][0]["claim"].startswith("Nina Vale from Vireo Surgical Group")
    assert result.data["facts"][0]["entities"] == [
        "Nina Vale",
        "Vireo Surgical Group",
        "Priya Natarajan",
        "Halvard Medical",
    ]


def test_tree_shaped_response_downgrades_json_strategy_for_the_next_call() -> None:
    # Fix instructions item 2(b): receiving a tree is treated as a
    # structured-output failure for this model, exactly like a real 400
    # rejection -- `_downgrade_json_strategy`'s own one-rung-down rule.
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.dumps({"schema_version": "0.2", "facts": [NOVA_TREE_SAMPLE_1]})
        return _chat_response(body)

    client = _client(handler)
    client.complete(
        system="s",
        user="u",
        surface="extract",
        json_schema=FACT_ENTITY_STRING_SCHEMA,
        model="amazon/nova-lite-v1",
        validate_json=False,
    )

    assert openrouter_module._json_strategy_cache["amazon/nova-lite-v1"] == STRATEGY_OBJECT

    # A second call for the same model id no longer attempts the doomed
    # native json_schema request at all.
    captured: list[dict[str, Any]] = []

    def handler2(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured.append(body)
        return _chat_response('{"facts": []}')

    client2 = _client(handler2)
    client2.complete(
        system="s",
        user="u",
        surface="extract",
        json_schema=FACT_ENTITY_STRING_SCHEMA,
        model="amazon/nova-lite-v1",
        validate_json=False,
    )

    assert len(captured) == 1
    assert captured[0]["response_format"] == {"type": "json_object"}


def test_tree_detected_under_validate_json_true_recovers_and_validates() -> None:
    # validate_json=True (every caller besides LLMClientBackend) must also
    # benefit: the flatten pass runs BEFORE schema validation, so a
    # tree-contaminated fact no longer fails validation before this
    # recovery gets a chance to run.
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.dumps({"facts": [NOVA_TREE_SAMPLE_1]})
        return _chat_response(body)

    schema = {
        "type": "object",
        "properties": {
            "facts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "claim": {"type": "string"},
                        "type": {"type": "string"},
                        "sentiment": {"type": "string"},
                        "confidence": {"type": "string"},
                        "entities": {"type": "array", "items": {"type": "string"}},
                        "locator": {"type": "string"},
                        "quote": {"type": "string"},
                    },
                    "required": ["id", "claim"],
                },
            },
        },
        "required": ["facts"],
    }

    client = _client(handler)
    result = client.complete(
        system="s",
        user="u",
        surface="extract",
        json_schema=schema,
        model="amazon/nova-lite-v1",
    )

    assert result.data["facts"][0]["id"] == "f1"
    assert isinstance(result.data["facts"][0], dict)


# --------------------------------------------------------------------------
# Per-client reasoning and service tier (GPT-6 Luna eval, 2026-09-23)
# --------------------------------------------------------------------------


def test_client_reasoning_replaces_the_default_disable() -> None:
    captured: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return _chat_response("hi")

    client = _client(handler, reasoning={"effort": "max"})
    client.complete(system="s", user="u", surface="chat", model="openai/gpt-6-luna-pro")
    assert captured[0]["reasoning"] == {"effort": "max"}
    assert "service_tier" not in captured[0]


def test_client_service_tier_is_sent_top_level() -> None:
    captured: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return _chat_response("hi")

    _client(handler, service_tier="flex").complete(system="s", user="u", surface="chat")
    assert captured[0]["service_tier"] == "flex"
    assert captured[0]["reasoning"] == {"enabled": False}


def test_reasoning_tokens_provider_and_tier_are_recorded() -> None:
    sink = InMemorySink()

    def handler(request: httpx.Request) -> httpx.Response:
        body = {
            "id": "gen-test",
            "provider": "OpenAI",
            "service_tier": "flex",
            "choices": [{"message": {"role": "assistant", "content": "hi"}}],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 900,
                "completion_tokens_details": {"reasoning_tokens": 850},
                "cost": 0.0005,
            },
        }
        return httpx.Response(200, json=body)

    _client(handler, sink=sink).complete(system="s", user="u", surface="chat")
    extra = sink.events[0].extra
    assert extra["reasoning_tokens"] == 850
    assert extra["provider"] == "OpenAI"
    assert extra["service_tier"] == "flex"
