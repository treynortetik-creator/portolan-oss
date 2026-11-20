"""Tests for portolan.metering."""

from __future__ import annotations

import json

import pytest

from portolan.metering import InMemorySink, JsonlSink, UsageEvent, metered


def test_metered_records_one_event_on_success() -> None:
    sink = InMemorySink()
    with metered(sink, tenant="acme", surface="extract", model_key="nova-lite") as usage:
        usage.input_tokens = 1000
        usage.output_tokens = 200

    assert len(sink.events) == 1
    event = sink.events[0]
    assert isinstance(event, UsageEvent)
    assert event.tenant == "acme"
    assert event.surface == "extract"
    assert event.model_key == "nova-lite"
    assert event.input_tokens == 1000
    assert event.output_tokens == 200
    assert event.cached_tokens == 0
    assert event.cost_estimate > 0
    assert event.request_id.startswith("req_")
    assert event.timestamp  # non-empty absolute timestamp
    assert event.extra == {}


def test_metered_cost_override_wins_over_the_plain_pricing_table_estimate() -> None:
    # Bug hunt 2026-09-18, finding 9: `cost_override` (new) lets a caller
    # that needs a pricier calculation than `estimate_cost`'s flat
    # `(input, output, cached)` shape -- `AnthropicAPIClient`'s cache-WRITE
    # premium via `estimate_cost_with_cache`, in particular -- substitute
    # its own number instead of the plain pricing-table estimate.
    sink = InMemorySink()
    with metered(sink, tenant="acme", surface="extract", model_key="nova-lite") as usage:
        usage.input_tokens = 1000
        usage.output_tokens = 200
        usage.cost_override = 1.2345

    event = sink.events[0]
    assert event.cost_estimate == pytest.approx(1.2345)


def test_metered_records_event_on_exception_with_error_in_extra() -> None:
    sink = InMemorySink()
    with (
        pytest.raises(ValueError, match="boom"),
        metered(sink, tenant="acme", surface="extract", model_key="nova-lite") as usage,
    ):
        usage.input_tokens = 500
        raise ValueError("boom")

    assert len(sink.events) == 1
    event = sink.events[0]
    assert event.input_tokens == 500
    assert event.output_tokens == 0
    assert event.extra.get("error") == "boom"


def test_metered_preserves_caller_supplied_extra_on_exception() -> None:
    sink = InMemorySink()
    with (
        pytest.raises(RuntimeError),
        metered(sink, tenant="acme", surface="extract", model_key="nova-lite") as usage,
    ):
        usage.extra["segment"] = 2
        raise RuntimeError("nope")

    event = sink.events[0]
    assert event.extra["segment"] == 2
    assert event.extra["error"] == "nope"


def test_metered_each_call_gets_a_unique_request_id() -> None:
    sink = InMemorySink()
    with metered(sink, "acme", "extract", "nova-lite"):
        pass
    with metered(sink, "acme", "extract", "nova-lite"):
        pass
    assert sink.events[0].request_id != sink.events[1].request_id


def test_metered_records_event_for_unknown_model_instead_of_swallowing_the_result() -> None:
    # Review finding F3: cost used to be computed unguarded in `finally`,
    # so `estimate_cost` raising `UnknownModelError` for a model_key with
    # no pricing row replaced the caller's own successful result with that
    # pricing error -- money spent, work discarded, nothing in the ledger.
    # A successful call now records an event with cost_estimate=None and
    # extra["pricing_error"] set, and does NOT raise.
    sink = InMemorySink()
    with metered(sink, "acme", "extract", "not-a-real-model") as usage:
        usage.input_tokens = 500_000
        usage.output_tokens = 1_000

    assert len(sink.events) == 1
    event = sink.events[0]
    assert event.input_tokens == 500_000
    assert event.output_tokens == 1_000
    assert event.cost_estimate is None
    assert "pricing_error" in event.extra
    assert "not-a-real-model" in event.extra["pricing_error"]


def test_metered_unknown_model_does_not_mask_a_real_exception() -> None:
    # The caller's real exception must win, not a pricing-config error two
    # __context__ levels away from it.
    sink = InMemorySink()
    with (
        pytest.raises(ValueError, match="boom"),
        metered(sink, "acme", "extract", "not-a-real-model") as usage,
    ):
        usage.input_tokens = 10
        raise ValueError("boom")

    assert len(sink.events) == 1
    event = sink.events[0]
    assert event.cost_estimate is None
    assert event.extra["error"] == "boom"
    assert "pricing_error" in event.extra


def test_jsonl_sink_appends_one_json_object_per_line(tmp_path) -> None:
    path = tmp_path / "usage.jsonl"
    sink = JsonlSink(path)

    with metered(sink, "acme", "extract", "nova-lite") as usage:
        usage.input_tokens = 10
        usage.output_tokens = 5

    with metered(sink, "acme", "chat", "claude-sonnet-5") as usage:
        usage.input_tokens = 20
        usage.output_tokens = 8

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2

    first = json.loads(lines[0])
    second = json.loads(lines[1])
    assert first["surface"] == "extract"
    assert first["model_key"] == "nova-lite"
    assert second["surface"] == "chat"
    assert second["model_key"] == "claude-sonnet-5"
