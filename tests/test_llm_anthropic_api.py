"""Tests for portolan.llm.anthropic_api.AnthropicAPIClient.

No real `anthropic` SDK client and no network anywhere in this file: every
test injects a fake object with the same `.messages.stream(**kwargs) ->
contextmanager yielding something with .get_final_message()` shape that
`anthropic.Anthropic()` has (per the claude-api skill's README, "Extended
Thinking" and "Prompt Caching" examples this module is built against).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from portolan.llm.anthropic_api import AnthropicAPIClient, LLMRefusal
from portolan.llm.base import LLMJsonError
from portolan.metering import InMemorySink

SCHEMA = {
    "type": "object",
    "properties": {"greeting": {"type": "string"}},
    "required": ["greeting"],
    "additionalProperties": False,
}


class _FakeStream:
    def __init__(self, message: Any) -> None:
        self._message = message

    def __enter__(self) -> _FakeStream:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        return False

    def get_final_message(self) -> Any:
        return self._message


class _FakeMessages:
    def __init__(self, message: Any) -> None:
        self._message = message
        self.calls: list[dict[str, Any]] = []

    def stream(self, **kwargs: Any) -> _FakeStream:
        self.calls.append(kwargs)
        message = self._message(**kwargs) if callable(self._message) else self._message
        return _FakeStream(message)


class FakeAnthropicSDKClient:
    """Stands in for `anthropic.Anthropic()`. `message` is either a fixed
    response object or a `(**request_kwargs) -> response` callable, so a
    test can make the canned response depend on what was actually sent."""

    def __init__(self, message: Any) -> None:
        self.messages = _FakeMessages(message)


def _message(
    *,
    text: str = "hello",
    stop_reason: str = "end_turn",
    input_tokens: int = 100,
    output_tokens: int = 20,
    cache_read_input_tokens: int = 0,
    cache_creation_input_tokens: int = 0,
    stop_details: Any = None,
    request_id: str = "req_test123",
) -> SimpleNamespace:
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
            cache_creation_input_tokens=cache_creation_input_tokens,
        ),
        stop_reason=stop_reason,
        stop_details=stop_details,
        _request_id=request_id,
    )


def _client(fake_sdk_client: Any, **kwargs: Any) -> AnthropicAPIClient:
    return AnthropicAPIClient(client=fake_sdk_client, sink=InMemorySink(), **kwargs)


@pytest.fixture(autouse=True)
def _isolated_temperature_env(monkeypatch: pytest.MonkeyPatch):
    """`PORTOLAN_TEMPERATURE` (infra-2b wave, 2026-09-18) is read fresh on
    every call via `portolan.llm.base.resolve_temperature` -- clear it
    around every test in this file, same as `test_llm_openrouter.py`."""
    monkeypatch.delenv("PORTOLAN_TEMPERATURE", raising=False)
    yield
    monkeypatch.delenv("PORTOLAN_TEMPERATURE", raising=False)


def test_system_prompt_has_cache_control_breakpoint() -> None:
    fake_sdk = FakeAnthropicSDKClient(_message())
    client = _client(fake_sdk)

    client.complete(system="be terse", user="hi", surface="extract", model="claude-haiku-4-5")

    request = fake_sdk.messages.calls[0]
    assert request["system"] == [
        {"type": "text", "text": "be terse", "cache_control": {"type": "ephemeral"}}
    ]
    assert request["messages"] == [{"role": "user", "content": "hi"}]


def test_thinking_adaptive_for_sonnet() -> None:
    fake_sdk = FakeAnthropicSDKClient(_message())
    client = _client(fake_sdk)

    client.complete(system="s", user="u", surface="chat", model="claude-sonnet-5")

    request = fake_sdk.messages.calls[0]
    assert request["thinking"] == {"type": "adaptive"}


def test_no_thinking_for_haiku() -> None:
    fake_sdk = FakeAnthropicSDKClient(_message())
    client = _client(fake_sdk)

    client.complete(system="s", user="u", surface="extract", model="claude-haiku-4-5")

    request = fake_sdk.messages.calls[0]
    assert "thinking" not in request


# --------------------------------------------------------------------------
# temperature (infra-2b wave, 2026-09-18): the Messages API's own top-
# level `temperature` field -- but NEVER alongside extended thinking
# (Anthropic requires temperature=1/omitted whenever thinking is on), so
# this only actually reaches the request for Haiku (no thinking) through
# this client; see `complete()`'s own comment.
# --------------------------------------------------------------------------


def test_temperature_defaults_to_zero_for_extract_on_haiku_no_thinking() -> None:
    fake_sdk = FakeAnthropicSDKClient(_message())
    client = _client(fake_sdk)

    client.complete(system="s", user="u", surface="extract", model="claude-haiku-4-5")

    request = fake_sdk.messages.calls[0]
    assert request["temperature"] == 0.0
    assert "thinking" not in request


def test_temperature_defaults_to_zero_for_audit_on_haiku() -> None:
    fake_sdk = FakeAnthropicSDKClient(_message())
    client = _client(fake_sdk)

    client.complete(system="s", user="u", surface="audit", model="claude-haiku-4-5")

    assert fake_sdk.messages.calls[0]["temperature"] == 0.0


def test_temperature_omitted_for_chat_surface_on_haiku() -> None:
    fake_sdk = FakeAnthropicSDKClient(_message())
    client = _client(fake_sdk)

    client.complete(system="s", user="u", surface="chat", model="claude-haiku-4-5")

    assert "temperature" not in fake_sdk.messages.calls[0]


def test_temperature_not_sent_when_extended_thinking_is_active() -> None:
    # claude-sonnet-5 gets adaptive thinking by default (see
    # test_thinking_adaptive_for_sonnet above) -- forcing temperature=0.0
    # here would conflict with Anthropic's own thinking requirement
    # (temperature must be 1 or omitted while thinking is enabled), so
    # this client must NOT send it even for the extract surface.
    fake_sdk = FakeAnthropicSDKClient(_message())
    client = _client(fake_sdk)

    client.complete(system="s", user="u", surface="extract", model="claude-sonnet-5")

    request = fake_sdk.messages.calls[0]
    assert request["thinking"] == {"type": "adaptive"}
    assert "temperature" not in request


def test_explicit_temperature_argument_wins_even_for_report_surface() -> None:
    fake_sdk = FakeAnthropicSDKClient(_message())
    client = _client(fake_sdk)

    client.complete(
        system="s", user="u", surface="report", model="claude-haiku-4-5", temperature=0.9
    )

    assert fake_sdk.messages.calls[0]["temperature"] == 0.9


def test_temperature_env_var_overrides_the_zero_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PORTOLAN_TEMPERATURE", "0.4")
    fake_sdk = FakeAnthropicSDKClient(_message())
    client = _client(fake_sdk)

    client.complete(system="s", user="u", surface="extract", model="claude-haiku-4-5")

    assert fake_sdk.messages.calls[0]["temperature"] == 0.4


def test_no_output_config_without_json_schema() -> None:
    fake_sdk = FakeAnthropicSDKClient(_message())
    client = _client(fake_sdk)

    client.complete(system="s", user="u", surface="extract", model="claude-haiku-4-5")

    assert "output_config" not in fake_sdk.messages.calls[0]


def test_structured_output_requested_when_json_schema_given() -> None:
    fake_sdk = FakeAnthropicSDKClient(_message(text='{"greeting": "hi"}'))
    client = _client(fake_sdk)

    result = client.complete(
        system="s", user="u", surface="extract", model="claude-haiku-4-5", json_schema=SCHEMA
    )

    request = fake_sdk.messages.calls[0]
    assert request["output_config"] == {"format": {"type": "json_schema", "schema": SCHEMA}}
    assert result.data == {"greeting": "hi"}
    assert result.text == '{"greeting": "hi"}'


def test_invalid_structured_output_raises_llm_json_error() -> None:
    fake_sdk = FakeAnthropicSDKClient(_message(text="not json at all"))
    client = _client(fake_sdk)

    with pytest.raises(LLMJsonError):
        client.complete(
            system="s", user="u", surface="extract", model="claude-haiku-4-5", json_schema=SCHEMA
        )


def test_usage_recorded_with_cached_tokens() -> None:
    sink = InMemorySink()
    fake_sdk = FakeAnthropicSDKClient(
        _message(input_tokens=1200, output_tokens=40, cache_read_input_tokens=40000)
    )
    client = AnthropicAPIClient(client=fake_sdk, sink=sink, tenant="acme")

    result = client.complete(system="s", user="u", surface="extract", model="claude-sonnet-5")

    assert result.input_tokens == 1200
    assert result.output_tokens == 40
    assert result.cached_tokens == 40000

    assert len(sink.events) == 1
    event = sink.events[0]
    assert event.tenant == "acme"
    assert event.surface == "extract"
    assert event.model_key == "claude-sonnet-5"
    assert event.input_tokens == 1200
    assert event.cached_tokens == 40000
    # 1200*3.00/1e6 + 40000*3.00*0.1/1e6 + 40*15.00/1e6 -- worked example
    # from portolan.pricing's own docstring, plus the output-token cost.
    assert event.cost_estimate == pytest.approx(0.0156 + 40 * 15.00 / 1_000_000)


def test_cache_write_tokens_are_priced_at_the_premium_tier_and_counted_in_input_tokens() -> None:
    # Bug hunt 2026-09-18, finding 9: `cache_creation_input_tokens` (the
    # first call of every 5-minute cache window, since the system prompt
    # carries a `cache_control` breakpoint) used to land in `extra` only
    # -- never priced, and never counted in `input_tokens`.
    sink = InMemorySink()
    fake_sdk = FakeAnthropicSDKClient(
        _message(input_tokens=1000, output_tokens=40, cache_creation_input_tokens=20000)
    )
    client = AnthropicAPIClient(client=fake_sdk, sink=sink, tenant="acme")

    result = client.complete(system="s", user="u", surface="extract", model="claude-sonnet-5")

    # Cache-WRITE tokens are now counted in the input mass this event
    # reports (cache-READ accounting is unchanged -- see the test above).
    assert result.input_tokens == 1000 + 20000
    assert result.output_tokens == 40
    assert result.cached_tokens == 0  # no cache READ here, only a cache WRITE

    assert len(sink.events) == 1
    event = sink.events[0]
    assert event.extra["cache_creation_input_tokens"] == 20000
    # 1000*3.00/1e6 (fresh input) + 20000*3.00*1.25/1e6 (cache WRITE,
    # CACHE_WRITE_MULTIPLIER_5M's premium over the base input rate) +
    # 40*15.00/1e6 (output) -- never $0.00 or the fresh-input-only figure
    # the old code recorded for a call that was, in fact, mostly a cache
    # write.
    expected = (1000 * 3.00 + 20000 * 3.00 * 1.25) / 1_000_000 + 40 * 15.00 / 1_000_000
    assert event.cost_estimate == pytest.approx(expected)


def test_refusal_raises_llm_refusal_and_records_category() -> None:
    sink = InMemorySink()
    fake_sdk = FakeAnthropicSDKClient(
        _message(stop_reason="refusal", stop_details=SimpleNamespace(category="cyber"))
    )
    client = AnthropicAPIClient(client=fake_sdk, sink=sink)

    with pytest.raises(LLMRefusal):
        client.complete(system="s", user="u", surface="chat", model="claude-sonnet-5")

    assert len(sink.events) == 1
    assert sink.events[0].extra["refusal_category"] == "cyber"
    assert "error" in sink.events[0].extra


def test_missing_api_key_raises_clear_runtime_error() -> None:
    client = AnthropicAPIClient(api_key=None, sink=InMemorySink())

    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        client.complete(system="s", user="u", surface="extract")


def test_default_models() -> None:
    fake_sdk = FakeAnthropicSDKClient(_message())
    client = AnthropicAPIClient(client=fake_sdk, sink=InMemorySink())
    assert client.default_model == "claude-haiku-5-5"

    strong_client = AnthropicAPIClient(
        client=fake_sdk, sink=InMemorySink(), default_model="claude-sonnet-5"
    )
    assert strong_client.default_model == "claude-sonnet-5"


# ---------------------------------------------------------------------------
# Current-generation request shapes (Haiku 5.5, Sonnet 5.5).
# ---------------------------------------------------------------------------


def test_haiku_5_5_thinks_adaptively_and_is_sent_no_temperature() -> None:
    """Haiku 5.5 answers a non-default sampling parameter with a 400; the
    old "any haiku: no thinking, temperature 0" rule would have failed every
    extraction call on it."""
    fake_sdk = FakeAnthropicSDKClient(_message())
    _client(fake_sdk).complete(system="s", user="u", surface="extract", model="claude-haiku-5-5")
    request = fake_sdk.messages.calls[0]
    assert request["thinking"] == {"type": "adaptive"}
    assert "temperature" not in request


def test_a_thinking_model_gets_room_for_its_answer() -> None:
    fake_sdk = FakeAnthropicSDKClient(_message())
    client = _client(fake_sdk)
    client.complete(system="s", user="u", surface="extract", model="claude-haiku-5-5")
    client.complete(
        system="s", user="u", surface="extract", model="claude-haiku-5-5", max_tokens=32_000
    )
    client.complete(system="s", user="u", surface="extract", model="claude-haiku-4-5")
    assert [c["max_tokens"] for c in fake_sdk.messages.calls] == [16_000, 32_000, 4096]


def test_the_compare_verdict_runs_at_low_effort_alongside_the_schema() -> None:
    fake_sdk = FakeAnthropicSDKClient(_message(text='{"greeting": "x"}'))
    _client(fake_sdk).complete(
        system="s", user="u", surface="merge", model="claude-haiku-5-5", json_schema=SCHEMA
    )
    output_config = fake_sdk.messages.calls[0]["output_config"]
    assert output_config["effort"] == "low"
    assert output_config["format"]["type"] == "json_schema"


def test_a_reply_cut_off_at_max_tokens_is_not_parsed_as_whole() -> None:
    fake_sdk = FakeAnthropicSDKClient(_message(text='{"greeting": "x', stop_reason="max_tokens"))
    with pytest.raises(LLMJsonError, match="cut off at max_tokens"):
        _client(fake_sdk).complete(
            system="s", user="u", surface="extract", model="claude-haiku-5-5", json_schema=SCHEMA
        )
