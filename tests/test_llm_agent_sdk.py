"""Tests for portolan.llm.agent_sdk.AgentSDKClient.

No network, no `claude_agent_sdk` import touched: `_call` (the one method
that shells out) is monkeypatched on the class, so these tests exercise
everything around it -- the JSON instruction, the retry-with-error-appended
path, `LLMJsonError` after two failures, model defaulting -- without
needing the CLI installed or a live session.
"""

from __future__ import annotations

import pytest

from portolan.llm.agent_sdk import AgentSDKClient, _usage_from_result
from portolan.llm.base import LLMJsonError

SCHEMA = {
    "type": "object",
    "properties": {"greeting": {"type": "string"}, "count": {"type": "integer"}},
    "required": ["greeting", "count"],
    "additionalProperties": False,
}

CANNED_USAGE = {"input_tokens": 10, "output_tokens": 5, "cached_tokens": 0}
# `_call`'s own return dict (what these stubs return, distinct from
# `_usage_from_result`'s richer dict below) only ever carries these three
# keys -- `cache_creation_tokens` is folded into `input_tokens` before
# `_call` returns (infra-2b wave, 2026-09-18), never a separate key on
# this particular dict shape.


def test_complete_plain_text_no_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str, str, str]] = []

    def fake_call(self, system, user, model_key, surface, temperature=None):
        calls.append((system, user, model_key, surface))
        return "Hello there.", dict(CANNED_USAGE)

    monkeypatch.setattr(AgentSDKClient, "_call", fake_call)
    client = AgentSDKClient(default_model="claude-haiku-4-5")

    result = client.complete(system="be terse", user="say hi", surface="chat")

    assert result.text == "Hello there."
    assert result.data is None
    assert result.model == "claude-haiku-4-5"
    assert result.input_tokens == 10
    assert result.output_tokens == 5
    assert len(calls) == 1
    # No json_schema -- the system prompt is passed through unmodified.
    assert calls[0][0] == "be terse"


def test_complete_json_success_on_first_try(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str]] = []

    def fake_call(self, system, user, model_key, surface, temperature=None):
        calls.append((system, user))
        return '{"greeting": "hi", "count": 3}', dict(CANNED_USAGE)

    monkeypatch.setattr(AgentSDKClient, "_call", fake_call)
    client = AgentSDKClient()

    result = client.complete(system="sys", user="usr", json_schema=SCHEMA, surface="extract")

    assert result.data == {"greeting": "hi", "count": 3}
    assert len(calls) == 1
    # json_schema given -- the schema is folded into the system prompt.
    assert "sys" in calls[0][0]
    assert "JSON Schema" in calls[0][0]


def test_complete_json_retries_once_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    users_seen: list[str] = []
    responses = iter(["not json at all", '{"greeting": "hi", "count": 3}'])

    def fake_call(self, system, user, model_key, surface, temperature=None):
        users_seen.append(user)
        return next(responses), dict(CANNED_USAGE)

    monkeypatch.setattr(AgentSDKClient, "_call", fake_call)
    client = AgentSDKClient()

    result = client.complete(
        system="sys", user="original user message", json_schema=SCHEMA, surface="extract"
    )

    assert result.data == {"greeting": "hi", "count": 3}
    assert len(users_seen) == 2
    assert users_seen[0] == "original user message"
    assert "original user message" in users_seen[1]
    assert "Validation error" in users_seen[1]
    assert "no JSON object found" in users_seen[1]


def test_complete_json_raises_llm_json_error_after_one_failed_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_call(self, system, user, model_key, surface, temperature=None):
        return "still not json", dict(CANNED_USAGE)

    monkeypatch.setattr(AgentSDKClient, "_call", fake_call)
    client = AgentSDKClient()

    with pytest.raises(LLMJsonError):
        client.complete(system="sys", user="usr", json_schema=SCHEMA, surface="extract")


def test_complete_json_retries_exactly_once_not_more(monkeypatch: pytest.MonkeyPatch) -> None:
    call_count = 0

    def fake_call(self, system, user, model_key, surface, temperature=None):
        nonlocal call_count
        call_count += 1
        return "never valid json", dict(CANNED_USAGE)

    monkeypatch.setattr(AgentSDKClient, "_call", fake_call)
    client = AgentSDKClient()

    with pytest.raises(LLMJsonError):
        client.complete(system="sys", user="usr", json_schema=SCHEMA, surface="extract")

    assert call_count == 2  # one original attempt, one retry -- never a second retry


def test_complete_uses_default_model_when_none_given(monkeypatch: pytest.MonkeyPatch) -> None:
    seen_model_keys: list[str] = []

    def fake_call(self, system, user, model_key, surface, temperature=None):
        seen_model_keys.append(model_key)
        return "hi", dict(CANNED_USAGE)

    monkeypatch.setattr(AgentSDKClient, "_call", fake_call)
    client = AgentSDKClient(default_model="claude-sonnet-5")

    client.complete(system="s", user="u", surface="chat")

    assert seen_model_keys == ["claude-sonnet-5"]


def test_complete_model_override_wins_over_default(monkeypatch: pytest.MonkeyPatch) -> None:
    seen_model_keys: list[str] = []

    def fake_call(self, system, user, model_key, surface, temperature=None):
        seen_model_keys.append(model_key)
        return "hi", dict(CANNED_USAGE)

    monkeypatch.setattr(AgentSDKClient, "_call", fake_call)
    client = AgentSDKClient(default_model="claude-haiku-4-5")

    client.complete(system="s", user="u", model="claude-sonnet-5", surface="chat")

    assert seen_model_keys == ["claude-sonnet-5"]


def test_complete_records_usage_event(monkeypatch: pytest.MonkeyPatch) -> None:
    from portolan.metering import InMemorySink

    def fake_call(self, system, user, model_key, surface, temperature=None):
        # _call is normally the thing that meters -- since we're stubbing
        # it out entirely, meter here too so this test can assert the
        # sink actually gets used end to end through AgentSDKClient's own
        # bound sink/tenant.
        from portolan.metering import metered

        with metered(self.sink, self.tenant, surface, model_key) as usage:
            usage.input_tokens = 7
            usage.output_tokens = 3
        return "hi", {"input_tokens": 7, "output_tokens": 3, "cached_tokens": 0}

    monkeypatch.setattr(AgentSDKClient, "_call", fake_call)
    sink = InMemorySink()
    client = AgentSDKClient(sink=sink, tenant="acme")

    client.complete(system="s", user="u", surface="chat")

    assert len(sink.events) == 1
    assert sink.events[0].tenant == "acme"
    assert sink.events[0].surface == "chat"


# --------------------------------------------------------------------------
# _usage_from_result (pure helper, no SDK/network)
# --------------------------------------------------------------------------


def test_usage_from_result_uses_reported_usage() -> None:
    usage = _usage_from_result(
        {"input_tokens": 100, "output_tokens": 50, "cache_read_input_tokens": 10},
        text="reply text",
        system="sys",
        user="usr",
    )
    assert usage == {
        "input_tokens": 100,
        "output_tokens": 50,
        "cached_tokens": 10,
        "cache_creation_tokens": 0,
        "estimated": False,
    }


def test_usage_from_result_reads_cache_creation_tokens() -> None:
    # infra-2b wave, 2026-09-18: the real accounting bug this closes --
    # `cache_creation_input_tokens` used to be dropped entirely. Real
    # shape from a live smoke call (`01-spec/two-pass-2026-09-18.md`):
    # `{"input_tokens": 10, "cache_creation_input_tokens": 18672}`.
    usage = _usage_from_result(
        {"input_tokens": 10, "output_tokens": 100, "cache_creation_input_tokens": 18672},
        text="reply text",
        system="sys",
        user="usr",
    )
    assert usage["cache_creation_tokens"] == 18672
    assert usage["input_tokens"] == 10
    assert usage["cached_tokens"] == 0


def test_usage_from_result_falls_back_to_estimate_when_missing() -> None:
    usage = _usage_from_result(None, text="a" * 40, system="b" * 20, user="c" * 20)
    assert usage["estimated"] is True
    assert usage["input_tokens"] == max(1, 40 // 4)
    assert usage["output_tokens"] == max(1, 40 // 4)
    assert usage["cached_tokens"] == 0
    assert usage["cache_creation_tokens"] == 0


# --------------------------------------------------------------------------
# `_call` itself (infra-2b wave, 2026-09-18): unlike the `complete()` tests
# above, which monkeypatch `_call` entirely, these monkeypatch the lower-
# level async `_aquery` so `_call`'s own cache-accounting/cost logic runs
# for real -- this is what would have caught the cache-creation
# under-counting bug in the first place.
# --------------------------------------------------------------------------


def test_call_reports_input_tokens_as_the_full_cache_aware_sum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from portolan.metering import InMemorySink

    async def fake_aquery(self, system, user, model, temperature=None):
        return "reply", {
            "input_tokens": 10,
            "output_tokens": 100,
            "cached_tokens": 2_000,
            "cache_creation_tokens": 18_672,
            "estimated": False,
        }

    monkeypatch.setattr(AgentSDKClient, "_aquery", fake_aquery)
    sink = InMemorySink()
    client = AgentSDKClient(default_model="claude-haiku-4-5", sink=sink, tenant="acme")

    text, usage_info = client._call("sys", "usr", "claude-haiku-4-5", "extract")

    assert text == "reply"
    # Fix instructions: "input = input_tokens + cache_creation_input_tokens
    # + cache_read_input_tokens" -- the real total, not the fresh-input
    # floor the old code reported.
    assert usage_info["input_tokens"] == 10 + 2_000 + 18_672
    assert usage_info["cached_tokens"] == 2_000  # unchanged meaning: reads only
    assert len(sink.events) == 1
    event = sink.events[0]
    assert event.input_tokens == 10 + 2_000 + 18_672
    assert event.cached_tokens == 2_000
    assert event.extra["cache_creation_input_tokens"] == 18_672


def test_call_prices_cache_creation_at_the_write_premium_not_the_read_discount(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from portolan.metering import InMemorySink
    from portolan.pricing import estimate_cost_with_cache

    async def fake_aquery(self, system, user, model, temperature=None):
        return "reply", {
            "input_tokens": 10,
            "output_tokens": 100,
            "cached_tokens": 2_000,
            "cache_creation_tokens": 18_672,
            "estimated": False,
        }

    monkeypatch.setattr(AgentSDKClient, "_aquery", fake_aquery)
    sink = InMemorySink()
    client = AgentSDKClient(default_model="claude-haiku-4-5", sink=sink, tenant="acme")

    client._call("sys", "usr", "claude-haiku-4-5", "extract")

    expected_cost = estimate_cost_with_cache(
        "claude-haiku-4-5",
        10,
        100,
        cache_read_tokens=2_000,
        cache_creation_tokens=18_672,
    )
    assert sink.events[0].cost_estimate == pytest.approx(expected_cost)
    # A cache write costs strictly MORE per token than a normal input
    # token -- confirming this isn't accidentally using the cache-READ
    # discount (which would make the cost LOWER, not higher).
    from portolan.pricing import estimate_cost

    as_if_all_uncached = estimate_cost("claude-haiku-4-5", 10 + 2_000 + 18_672, 100)
    assert expected_cost > as_if_all_uncached


def test_call_records_no_cache_creation_extra_when_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    from portolan.metering import InMemorySink

    async def fake_aquery(self, system, user, model, temperature=None):
        return "reply", {
            "input_tokens": 500,
            "output_tokens": 100,
            "cached_tokens": 0,
            "cache_creation_tokens": 0,
            "estimated": False,
        }

    monkeypatch.setattr(AgentSDKClient, "_aquery", fake_aquery)
    sink = InMemorySink()
    client = AgentSDKClient(sink=sink, tenant="acme")

    client._call("sys", "usr", "claude-haiku-4-5", "extract")

    assert "cache_creation_input_tokens" not in sink.events[0].extra
    assert sink.events[0].input_tokens == 500


def test_call_records_temperature_requested_in_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    from portolan.metering import InMemorySink

    async def fake_aquery(self, system, user, model, temperature=None):
        return "reply", dict(CANNED_USAGE, cache_creation_tokens=0, estimated=False)

    monkeypatch.setattr(AgentSDKClient, "_aquery", fake_aquery)
    sink = InMemorySink()
    client = AgentSDKClient(sink=sink, tenant="acme")

    client._call("sys", "usr", "claude-haiku-4-5", "extract", temperature=0.0)

    assert sink.events[0].extra["temperature_requested"] == 0.0


# --------------------------------------------------------------------------
# using_sink (infra-2b wave, 2026-09-18) -- mirrors
# OpenRouterClient.using_sink's own tests exactly.
# --------------------------------------------------------------------------


def test_using_sink_reroutes_metering_for_the_duration_of_the_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_aquery(self, system, user, model, temperature=None):
        return "hi", dict(CANNED_USAGE, cache_creation_tokens=0, estimated=False)

    monkeypatch.setattr(AgentSDKClient, "_aquery", fake_aquery)
    from portolan.metering import InMemorySink

    default_sink = InMemorySink()
    client = AgentSDKClient(sink=default_sink, tenant="default-tenant")

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


def test_using_sink_actually_swaps_the_bound_attributes(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_call(self, system, user, model_key, surface, temperature=None):
        # Assert from inside the call that the swap is live at call time.
        assert self.tenant == "bench-tenant"
        return "hi", dict(CANNED_USAGE)

    monkeypatch.setattr(AgentSDKClient, "_call", fake_call)
    client = AgentSDKClient(tenant="default-tenant")

    from portolan.metering import InMemorySink

    bench_sink = InMemorySink()
    with client.using_sink(bench_sink, "bench-tenant"):
        assert client.sink is bench_sink
        assert client.tenant == "bench-tenant"
        client.complete(system="s", user="u", surface="extract")

    assert client.tenant == "default-tenant"


def test_using_sink_restores_original_sink_even_if_the_block_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_call(self, system, user, model_key, surface, temperature=None):
        raise RuntimeError("boom")

    monkeypatch.setattr(AgentSDKClient, "_call", fake_call)
    from portolan.metering import InMemorySink

    default_sink = InMemorySink()
    client = AgentSDKClient(sink=default_sink, tenant="default-tenant")
    bench_sink = InMemorySink()

    with pytest.raises(RuntimeError), client.using_sink(bench_sink, "bench-tenant"):
        client.complete(system="s", user="u", surface="chat")

    assert client.sink is default_sink
    assert client.tenant == "default-tenant"


@pytest.mark.parametrize("cli_path", [None, "/opt/claude/bin/claude"])
def test_cli_path_reaches_the_sdk_options(monkeypatch: pytest.MonkeyPatch, cli_path) -> None:
    import claude_agent_sdk
    from claude_agent_sdk import ResultMessage

    seen = []

    async def fake_query(*, prompt, options):
        seen.append(options)
        yield ResultMessage(
            subtype="success",
            duration_ms=1,
            duration_api_ms=1,
            is_error=False,
            num_turns=1,
            session_id="s",
            result="hi",
            usage={"input_tokens": 3, "output_tokens": 1},
        )

    monkeypatch.setattr(claude_agent_sdk, "query", fake_query)
    client = AgentSDKClient(default_model="claude-haiku-4-5", cli_path=cli_path)

    assert client.complete(system="s", user="u", surface="chat").text == "hi"
    assert seen[0].cli_path == cli_path
