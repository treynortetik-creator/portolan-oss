"""Tests for portolan.pricing."""

from __future__ import annotations

import pytest

from portolan.pricing import (
    CACHE_READ_DISCOUNT,
    CACHE_WRITE_MULTIPLIER_1H,
    CACHE_WRITE_MULTIPLIER_5M,
    PRICING,
    UnknownModelError,
    estimate_cost,
    estimate_cost_with_cache,
)

REQUIRED_MODEL_KEYS = [
    "nova-lite",
    "nova-micro",
    "gpt-oss-20b",
    "gpt-oss-120b",
    "gpt-5.6-luna",
    "claude-haiku-4-5",
    "claude-sonnet-5",
    "claude-opus-5",
    "titan-embed-v2",
]


@pytest.mark.parametrize("model_key", REQUIRED_MODEL_KEYS)
def test_every_required_model_is_priced(model_key: str) -> None:
    assert model_key in PRICING
    row = PRICING[model_key]
    assert row.input_per_million >= 0
    assert row.output_per_million >= 0
    assert "re-verify before budgeting" in row.source


def test_estimate_cost_matches_hand_calculation() -> None:
    # Nova Lite: $0.06 / $0.24 per 1M. 1M in + 1M out => 0.06 + 0.24 = 0.30.
    cost = estimate_cost("nova-lite", input_tokens=1_000_000, output_tokens=1_000_000)
    assert cost == pytest.approx(0.30)


def test_estimate_cost_zero_tokens_is_zero() -> None:
    assert estimate_cost("nova-lite", 0, 0) == 0.0


def test_estimate_cost_titan_has_no_output_cost() -> None:
    # Titan Text Embeddings V2 only ever has input tokens; output pricing is 0.
    cost = estimate_cost("titan-embed-v2", input_tokens=1_000_000, output_tokens=500)
    assert cost == pytest.approx(0.02)


def test_estimate_cost_cached_tokens_are_discounted_relative_to_full_price() -> None:
    # F4: cached_tokens is ADDITIVE to input_tokens (Anthropic's own usage
    # shape: input_tokens excludes cache reads, cache_read_input_tokens is
    # separate), not a subset of it -- so adding cached tokens on top of a
    # request can only raise its cost, never lower it. What the discount
    # buys is that those added tokens cost less than the same count of
    # full-price uncached input would.
    baseline = estimate_cost("claude-sonnet-5", input_tokens=1_000_000, output_tokens=0)
    with_cache_read = estimate_cost(
        "claude-sonnet-5", input_tokens=1_000_000, output_tokens=0, cached_tokens=1_000_000
    )
    as_if_all_uncached = estimate_cost("claude-sonnet-5", input_tokens=2_000_000, output_tokens=0)
    assert with_cache_read > baseline
    assert with_cache_read < as_if_all_uncached


def test_estimate_cost_matches_review_worked_example() -> None:
    # F4's worked example: 1,200 uncached + 40,000 cached input tokens on
    # Claude Sonnet 5 ($3.00/M input). There used to be a
    # `cached_tokens = max(0, min(cached_tokens, input_tokens))` clamp here
    # that threw the cache read away entirely, computing $0.000360 instead
    # of the correct $0.015600. The clamp is gone; this pins the correct
    # figure so it can't come back.
    cost = estimate_cost(
        "claude-sonnet-5", input_tokens=1_200, output_tokens=0, cached_tokens=40_000
    )
    assert cost == pytest.approx(0.0156)


def test_estimate_cost_unknown_model_raises() -> None:
    with pytest.raises(UnknownModelError):
        estimate_cost("not-a-real-model", 100, 100)


def test_unknown_model_error_is_a_key_error() -> None:
    # So existing `except KeyError` call sites keep working.
    assert issubclass(UnknownModelError, KeyError)


def test_estimate_cost_with_cache_matches_estimate_cost_with_no_cache_tokens() -> None:
    plain = estimate_cost("claude-sonnet-5", input_tokens=1_000, output_tokens=500)
    with_cache = estimate_cost_with_cache("claude-sonnet-5", input_tokens=1_000, output_tokens=500)
    assert with_cache == pytest.approx(plain)


def test_estimate_cost_with_cache_read_matches_estimate_cost_cached_tokens() -> None:
    # cache_read_tokens should be billed exactly the same as
    # estimate_cost's own cached_tokens (same CACHE_READ_DISCOUNT).
    plain = estimate_cost(
        "claude-sonnet-5", input_tokens=1_200, output_tokens=0, cached_tokens=40_000
    )
    with_cache = estimate_cost_with_cache(
        "claude-sonnet-5", input_tokens=1_200, output_tokens=0, cache_read_tokens=40_000
    )
    assert with_cache == pytest.approx(plain)


def test_estimate_cost_with_cache_write_is_a_premium_not_a_discount() -> None:
    # Claude Sonnet 5: $3.00/M input. 18,672 cache-creation tokens (the
    # real number from the live agent_sdk smoke call this fix was
    # diagnosed from) at the 5-minute write multiplier (1.25x).
    cost = estimate_cost_with_cache(
        "claude-sonnet-5", input_tokens=10, output_tokens=0, cache_creation_tokens=18_672
    )
    expected = 10 * 3.00 / 1_000_000 + 18_672 * 3.00 * CACHE_WRITE_MULTIPLIER_5M / 1_000_000
    assert cost == pytest.approx(expected)
    # A cache write always costs strictly more per token than a plain
    # uncached input token at the same volume -- never a discount.
    as_if_uncached = estimate_cost("claude-sonnet-5", input_tokens=18_682, output_tokens=0)
    assert cost > as_if_uncached


def test_estimate_cost_with_cache_write_one_hour_tier_costs_more_than_five_minute() -> None:
    five_min = estimate_cost_with_cache(
        "claude-sonnet-5",
        input_tokens=0,
        output_tokens=0,
        cache_creation_tokens=100_000,
        cache_write_ttl="5m",
    )
    one_hour = estimate_cost_with_cache(
        "claude-sonnet-5",
        input_tokens=0,
        output_tokens=0,
        cache_creation_tokens=100_000,
        cache_write_ttl="1h",
    )
    assert one_hour > five_min
    assert CACHE_WRITE_MULTIPLIER_1H > CACHE_WRITE_MULTIPLIER_5M


def test_estimate_cost_with_cache_read_and_write_together() -> None:
    cost = estimate_cost_with_cache(
        "claude-haiku-4-5",
        input_tokens=10,
        output_tokens=15_562,
        cache_read_tokens=2_000,
        cache_creation_tokens=18_672,
    )
    price = PRICING["claude-haiku-4-5"]
    expected = (
        10 * price.input_per_million
        + 2_000 * price.input_per_million * CACHE_READ_DISCOUNT
        + 18_672 * price.input_per_million * CACHE_WRITE_MULTIPLIER_5M
    ) / 1_000_000 + 15_562 * price.output_per_million / 1_000_000
    assert cost == pytest.approx(expected)


def test_estimate_cost_with_cache_unknown_model_raises() -> None:
    with pytest.raises(UnknownModelError):
        estimate_cost_with_cache("not-a-real-model", 100, 100, cache_creation_tokens=50)
