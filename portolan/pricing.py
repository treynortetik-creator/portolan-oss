"""Per-model token pricing for the ingest pipeline and every other model call.

Every price below comes from
`01-spec/research/research-bedrock-models-2026-09-12.md` (cross-checked against
the summary table in `01-spec/ingest-cost-and-models.md`), both dated
2026-09-12. Bedrock prices move; re-verify against the live Bedrock pricing
page before these numbers go into a customer budget or contract — that
research doc says as much on nearly every line.

This module has no dependency on anything else in the ingest pipeline so it
can be imported from cost-estimation scripts, tests, or a future admin UI
without pulling in pydantic, boto3, or the anthropic SDK.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Repeated verbatim on every row per the build instructions, so a person
# reading a cost report knows exactly how stale the number might be without
# chasing a separate citations table.
SOURCE_NOTE = "from research-bedrock-models-2026-09-12.md, re-verify before budgeting"

# OpenRouter rows below are keyed by the exact OpenRouter model id (slash
# form, e.g. "anthropic/claude-haiku-4.5") rather than this file's usual
# short key -- portolan.llm.openrouter.OpenRouterClient metering uses the
# id the caller actually requested as its model_key, and the four numbers
# here were pulled from OpenRouter's own model-list endpoint, not
# hand-copied off a rendered pricing page (that page's numbers are loaded
# client-side and didn't appear in a plain fetch).
OPENROUTER_SOURCE_NOTE = (
    "https://openrouter.ai/api/v1/models, fetched 2026-09-12 -- OpenRouter's public "
    "model-list endpoint (the openrouter.ai/models pages render pricing client-side "
    "and returned no numbers to a plain fetch); re-verify before budgeting, same as "
    "every other row in this table."
)

# Rough, provider-agnostic discount applied to cached input tokens. Anthropic's
# own cache-read discount is documented as "~0.1x cost" in the claude-api
# skill (`claude-api/python/claude-api/README.md`, Prompt Caching section).
# Bedrock/Nova cache pricing is not itemized in the research doc, so this is
# used as the default for every model until a per-model number is confirmed.
CACHE_READ_DISCOUNT = 0.1

# Cache-WRITE tiers -- the other half of Anthropic's published cache
# pricing (claude-api skill, "Prompt Caching"), added 2026-09-18 (infra-2b
# wave, `portolan/llm/agent_sdk.py`'s cache-creation accounting fix,
# `01-spec/two-pass-2026-09-18.md`). Unlike a cache READ (discounted,
# `CACHE_READ_DISCOUNT` above), a cache WRITE costs MORE than a normal
# input token -- Anthropic bills the first write of a prompt prefix into
# the cache at a premium over the base input rate, tiered by how long the
# entry is kept: 1.25x for the default 5-minute ephemeral tier, 2x for the
# optional 1-hour tier. Every Anthropic `cache_control` block this project
# sends (`portolan.llm.anthropic_api.AnthropicAPIClient`,
# `portolan.llm.agent_sdk`'s Claude Agent SDK session cache) is a plain
# `{"type": "ephemeral"}` with no `ttl` field, which is Anthropic's
# 5-minute default -- `estimate_cost_with_cache`'s own `cache_write_ttl`
# default matches that.
CACHE_WRITE_MULTIPLIER_5M = 1.25
CACHE_WRITE_MULTIPLIER_1H = 2.0


@dataclass(frozen=True)
class ModelPrice:
    """One row of the pricing table.

    Prices are dollars per 1,000,000 tokens, on-demand, list price. `source`
    is carried on every row (not just the module docstring) so the row is
    self-describing if it's copied out of this file into a report.
    """

    model_key: str
    display_name: str
    input_per_million: float
    output_per_million: float
    source: str = field(default=SOURCE_NOTE)


# Keyed by the short model key every other module (metering, backends) uses
# to refer to a model without repeating its full Bedrock/API id.
PRICING: dict[str, ModelPrice] = {
    "nova-micro": ModelPrice("nova-micro", "Amazon Nova Micro", 0.035, 0.14),
    "nova-lite": ModelPrice("nova-lite", "Amazon Nova Lite", 0.06, 0.24),
    "gpt-oss-20b": ModelPrice("gpt-oss-20b", "gpt-oss-20b", 0.07, 0.20),
    "gpt-oss-120b": ModelPrice("gpt-oss-120b", "gpt-oss-120b", 0.15, 0.60),
    "gpt-5.6-luna": ModelPrice("gpt-5.6-luna", "GPT-5.6 Luna", 0.20, 1.20),
    "claude-haiku-4-5": ModelPrice("claude-haiku-4-5", "Claude Haiku 4.5", 1.00, 5.00),
    # Current generation, list prices per the Claude API model table checked
    # 2026-11-20. Haiku 5.5's row is its rate for prompts up to 100K tokens;
    # a longer prompt bills at 0.50 / 2.50, which this table does not model
    # (extraction segments are far smaller).
    "claude-haiku-5-5": ModelPrice("claude-haiku-5-5", "Claude Haiku 5.5", 0.10, 0.50),
    "claude-sonnet-5-5": ModelPrice("claude-sonnet-5-5", "Claude Sonnet 5.5", 2.00, 10.00),
    "claude-opus-5-5": ModelPrice("claude-opus-5-5", "Claude Opus 5.5", 4.00, 20.00),
    "claude-fable-5-2": ModelPrice("claude-fable-5-2", "Claude Fable 5.2", 8.00, 40.00),
    "claude-sonnet-5": ModelPrice("claude-sonnet-5", "Claude Sonnet 5", 3.00, 15.00),
    "claude-opus-5": ModelPrice("claude-opus-5", "Claude Opus 5", 5.00, 25.00),
    # Titan has no output tokens (it embeds, it doesn't generate), so
    # output_per_million is 0.0 rather than undefined.
    "titan-embed-v2": ModelPrice("titan-embed-v2", "Titan Text Embeddings V2", 0.02, 0.0),
    # OpenRouter model ids (see OPENROUTER_SOURCE_NOTE above). Same
    # dollar figures as this table's own "claude-haiku-4-5", "gpt-5.6-luna",
    # and "nova-lite" rows -- OpenRouter passes Anthropic/OpenAI/Amazon's
    # own list prices through unmarked up for these three -- confirming
    # this is genuinely the same per-token price, not a coincidence of
    # rounding.
    "anthropic/claude-haiku-4.5": ModelPrice(
        "anthropic/claude-haiku-4.5",
        "Claude Haiku 4.5 (via OpenRouter)",
        1.00,
        5.00,
        source=OPENROUTER_SOURCE_NOTE,
    ),
    # Chat's agent mode (`portolan.web.chat_agent`). OpenRouter lists Sonnet 5 below
    # the direct "claude-sonnet-5" row above; cache reads 0.1x, 5-minute writes 1.25x.
    "anthropic/claude-sonnet-5": ModelPrice(
        "anthropic/claude-sonnet-5",
        "Claude Sonnet 5 (via OpenRouter)",
        2.00,
        10.00,
        source="https://openrouter.ai/api/v1/models, fetched 2026-09-25; re-verify before "
        "budgeting.",
    ),
    "openai/gpt-5.6-luna": ModelPrice(
        "openai/gpt-5.6-luna",
        "GPT-5.6 Luna (via OpenRouter)",
        0.20,
        1.20,
        source=OPENROUTER_SOURCE_NOTE,
    ),
    "qwen/qwen3.8-27b": ModelPrice(
        "qwen/qwen3.8-27b",
        "Qwen3.8 27B (via OpenRouter)",
        0.214,
        2.55,
        source=OPENROUTER_SOURCE_NOTE,
    ),
    "amazon/nova-lite-v1": ModelPrice(
        "amazon/nova-lite-v1",
        "Amazon Nova Lite 1.0 (via OpenRouter)",
        0.06,
        0.24,
        source=OPENROUTER_SOURCE_NOTE,
    ),
    # Added 2026-09-12 (OpenRouter bench fix wave): both rows below were
    # missing entirely -- the first three OpenRouter bench runs against
    # `deepseek/deepseek-v4-flash-0731` still recorded a real cost (OpenRouter's
    # `usage.cost` on the response wins over this table whenever present,
    # `portolan.llm.openrouter._cost_from_usage`), but any call whose
    # response omitted `usage.cost` would have silently recorded
    # `cost_estimate=None` with `extra["pricing_error"]` set instead of a
    # real number -- this table is the only fallback. Same
    # `/api/v1/models` canonical per-model pricing row every other
    # OpenRouter entry above cites (not the often-cheaper/pricier
    # per-endpoint numbers `/api/v1/models/{id}/endpoints` lists -- see
    # `portolan.llm.openrouter`'s module docstring for that endpoint).
    "deepseek/deepseek-v4-flash-0731": ModelPrice(
        "deepseek/deepseek-v4-flash-0731",
        "DeepSeek V4 Flash 0731 (via OpenRouter)",
        0.04,
        0.08,
        source=OPENROUTER_SOURCE_NOTE,
    ),
    # deepseek/deepseek-v4.1-flash: the owner's preferred DeepSeek model
    # (2026-09-12) for the OpenRouter bench, added alongside the row above
    # for the same reason. OpenRouter's own listing carries time-of-day/
    # day-of-week price overrides for this model (roughly double during
    # weekday UTC 01:00-04:00 and 06:00-10:00); the numbers below are the
    # off-peak/weekend base rate `/api/v1/models` reports as this model's
    # headline price -- re-verify before budgeting, same as every other
    # row in this table, more so here given the overrides.
    "deepseek/deepseek-v4.1-flash": ModelPrice(
        "deepseek/deepseek-v4.1-flash",
        "DeepSeek V4.1 Flash (via OpenRouter)",
        0.15,
        0.60,
        source=OPENROUTER_SOURCE_NOTE,
    ),
}


class UnknownModelError(KeyError):
    """Raised when estimate_cost is asked about a model_key not in PRICING.

    A KeyError subclass so existing `except KeyError` callers keep working,
    but with a message that says what to do about it instead of just the
    missing key.
    """


def estimate_cost(
    model_key: str,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int = 0,
) -> float:
    """Estimate the dollar cost of one model call.

    `cached_tokens` is ADDITIVE to `input_tokens`, not a subset of it
    (review finding F4): Anthropic's `usage` block reports `input_tokens`
    *excluding* cache reads, with `cache_read_input_tokens` reported
    separately (`portolan/extract/backends.py`'s `AnthropicBedrockBackend`
    assigns them exactly that way), so clamping `cached_tokens` to
    `input_tokens` silently discarded the cache read entirely. There used
    to be a `cached_tokens = max(0, min(cached_tokens, input_tokens))`
    clamp here; it is gone. `cached_tokens` is billed at
    `CACHE_READ_DISCOUNT` of the normal input rate, `input_tokens` at full
    rate: on a realistic shape (1,200 uncached + 40,000 cached input
    tokens, Claude Sonnet 5) that is `1200*3.00/1e6 + 40000*3.00*0.1/1e6 =
    0.0156` -- the worked example from the review that pinned this fix.

    Raises UnknownModelError if `model_key` is not in PRICING — fails loud
    rather than silently returning 0.0, since a silent zero would hide a
    real cost from a budget rollup.
    """
    try:
        price = PRICING[model_key]
    except KeyError as exc:
        raise UnknownModelError(
            f"no pricing row for model_key={model_key!r}; add it to "
            "portolan.pricing.PRICING before metering calls against it"
        ) from exc

    input_cost = (
        input_tokens * price.input_per_million
        + cached_tokens * price.input_per_million * CACHE_READ_DISCOUNT
    ) / 1_000_000
    output_cost = output_tokens * price.output_per_million / 1_000_000
    return input_cost + output_cost


def estimate_cost_with_cache(
    model_key: str,
    input_tokens: int,
    output_tokens: int,
    *,
    cache_read_tokens: int = 0,
    cache_creation_tokens: int = 0,
    cache_write_ttl: str = "5m",
) -> float:
    """Same contract as `estimate_cost`, extended with the cache-WRITE
    side `estimate_cost` has no parameter for -- added 2026-09-18 (infra-
    2b wave) for `portolan.llm.agent_sdk.AgentSDKClient`, the first caller
    that reads `cache_creation_input_tokens` off a real response and needs
    a real per-model rate to multiply it by instead of dropping it (the
    bug `01-spec/two-pass-2026-09-18.md` found: every Claude Agent SDK
    call's true input mass landed almost entirely in this uncounted
    field).

    `cache_read_tokens` is billed exactly like `estimate_cost`'s own
    `cached_tokens` (`CACHE_READ_DISCOUNT` of the base input rate);
    `cache_creation_tokens` is billed at a PREMIUM over the base rate
    (`CACHE_WRITE_MULTIPLIER_5M`/`_1H`, Anthropic's own published cache-
    write tiers) -- never a discount, and never folded into
    `estimate_cost`'s single `cached_tokens` slot, which would have
    applied the wrong (discounted) rate to a write. `cache_write_ttl`
    selects which write tier applies (`"5m"` or `"1h"`); every Anthropic
    call in this codebase sends a plain `{"type": "ephemeral"}` cache-
    control block with no `ttl` field, which is Anthropic's 5-minute
    default, so `"5m"` is this function's own default too.

    Raises `UnknownModelError` exactly like `estimate_cost` (same
    `PRICING` lookup, same failure mode -- fail loud on an unpriced model
    rather than silently return a number that ignores the cache tokens).
    """
    try:
        price = PRICING[model_key]
    except KeyError as exc:
        raise UnknownModelError(
            f"no pricing row for model_key={model_key!r}; add it to "
            "portolan.pricing.PRICING before metering calls against it"
        ) from exc

    write_multiplier = (
        CACHE_WRITE_MULTIPLIER_1H if cache_write_ttl == "1h" else CACHE_WRITE_MULTIPLIER_5M
    )

    input_cost = input_tokens * price.input_per_million / 1_000_000
    cache_read_cost = cache_read_tokens * price.input_per_million * CACHE_READ_DISCOUNT / 1_000_000
    cache_write_cost = (
        cache_creation_tokens * price.input_per_million * write_multiplier / 1_000_000
    )
    output_cost = output_tokens * price.output_per_million / 1_000_000
    return input_cost + cache_read_cost + cache_write_cost + output_cost
