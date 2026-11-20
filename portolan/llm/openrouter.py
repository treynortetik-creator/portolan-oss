"""`LLMClient` backed by OpenRouter's OpenAI-compatible chat completions
endpoint, so the bench and the app can use any model id behind one API shape.

Fields used, per https://openrouter.ai/docs/api-reference/overview,
https://openrouter.ai/docs/features/structured-outputs,
https://openrouter.ai/docs/use-cases/usage-accounting,
https://openrouter.ai/docs/use-cases/reasoning-tokens and
https://openrouter.ai/docs/api-reference/list-endpoints-for-a-model:

- `POST /api/v1/chat/completions`: `model`, `messages`, `max_tokens`,
  `response_format`, `provider` (routing order), `usage`, `reasoning`,
  `temperature`; for `complete_with_tools`, `tools`, `tool_choice` and the
  reply's `message.tool_calls`.
- Structured outputs: `response_format` `json_schema` (strict) or
  `json_object`. Support is per endpoint, so the strategy per model id comes
  from `GET /api/v1/models/{model}/endpoints` (`supported_parameters`, unioned
  over endpoints) and is downgraded when a live call rejects it. OpenRouter's
  top-level `error.message` is often just "Provider returned error"; the
  reason is in `error.metadata.raw` and `error.metadata.previous_errors`,
  which `_error_detail_text()` reads. The last resort is a plain system
  instruction plus `json_util`, which needs no provider support.
- Reasoning: some models reason by default, and reasoning spends the same
  `max_tokens` budget as the answer, which can leave `content` empty. Every
  request sends `reasoning: {"enabled": false}` except for models known to
  reject it, configured with an effort level, or a client built with its own
  `reasoning` (see `_base_body`).
- Usage: the response `usage` block (`prompt_tokens`, `completion_tokens`,
  `prompt_tokens_details.cached_tokens`, `completion_tokens_details.
  reasoning_tokens`, `cost`) plus the top-level `provider` and `service_tier`.
  `cost` is recorded as `cost_estimate`, assuming 1 OpenRouter credit is US$1.
- `HTTP-Referer` and `X-Title` headers are both `"Portolan"`.
- 429, 5xx, timeouts and dropped connections are retried up to `max_retries`
  times, honouring `Retry-After` when present, else jittered exponential backoff.

Tests inject `transport=httpx.MockTransport(...)` so no request leaves the
process.
"""

from __future__ import annotations

import json
import os
import random
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import httpx

from portolan.llm.base import (
    LLMError,
    LLMJsonError,
    LLMResult,
    ToolCall,
    ToolTurn,
    resolve_temperature,
)
from portolan.llm.json_util import extract_json_object, validate_json_object
from portolan.metering import InMemorySink, Surface, UsageEvent, UsageSink
from portolan.pricing import UnknownModelError, estimate_cost

if TYPE_CHECKING:
    from portolan.training.spend import SpendMeter

CHAT_COMPLETIONS_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_ENDPOINTS_URL_TEMPLATE = "https://openrouter.ai/api/v1/models/{model}/endpoints"

# Model ids look like "anthropic/claude-haiku-4.5" or "amazon/nova-lite-v1";
# there is no allowlist, `model=` is sent verbatim.
DEFAULT_MODEL = "anthropic/claude-haiku-4.5"

_MAX_RETRIES = 5
_BASE_BACKOFF_SECONDS = 1.0
_JITTER_FRACTION = 0.25
_RETRYABLE_STATUS = {429, 502, 503, 504}
_READ_TIMEOUT_SECONDS = 120.0
_CONNECT_TIMEOUT_SECONDS = 30.0
_DEFAULT_TIMEOUT = httpx.Timeout(_CONNECT_TIMEOUT_SECONDS, read=_READ_TIMEOUT_SECONDS)

JSON_INSTRUCTION_TEMPLATE = """

Respond with ONLY a single JSON object that validates against the following JSON Schema. \
No prose, no markdown code fence, no explanation before or after -- the JSON object and \
nothing else.

JSON Schema:
{schema}
"""

RETRY_TEMPLATE = """{user}

Your previous reply did not contain a JSON object that validates against the schema. \
Validation error: {error}

Reply again with ONLY the corrected JSON object -- no prose, no markdown code fence."""


# Per-model JSON strategy: which `response_format` (if any) to attempt,
# primed from the model's endpoint listing and only ever downgraded after a
# live rejection. Process-wide, not per instance, because worker threads each
# build their own client and must share one learned answer per model id.

STRATEGY_SCHEMA = "json_schema"
STRATEGY_OBJECT = "json_object"
STRATEGY_NONE = "none"

_STRATEGY_RANK = {STRATEGY_NONE: 0, STRATEGY_OBJECT: 1, STRATEGY_SCHEMA: 2}

_json_strategy_cache: dict[str, str] = {}
_json_strategy_cache_lock = threading.Lock()


def _choose_json_strategy(supported_parameters: list[str] | None) -> str:
    """Strategy from the union of a model's endpoint `supported_parameters`.

    Empty or `None` (fetch failed) -> `STRATEGY_SCHEMA`, the optimistic
    default, so a failed lookup never disables JSON mode.
    `structured_outputs` -> strict `json_schema`. `response_format` alone ->
    `json_object`. Neither -> no `response_format`; go straight to the
    manual-instruction fallback.
    """
    if not supported_parameters:
        return STRATEGY_SCHEMA
    params = set(supported_parameters)
    if "structured_outputs" in params:
        return STRATEGY_SCHEMA
    if "response_format" in params:
        return STRATEGY_OBJECT
    return STRATEGY_NONE


def fetch_supported_parameters(
    model_key: str,
    *,
    api_key: str | None,
    transport: httpx.BaseTransport | None = None,
    timeout: float | httpx.Timeout = 30.0,
) -> list[str] | None:
    """`GET /api/v1/models/{model}/endpoints`: the union of every endpoint's
    `supported_parameters`. Best-effort: `None` on any failure or an empty
    union, never raises, so a metadata lookup never blocks extraction.
    Needs no client instance.
    """
    try:
        client = httpx.Client(transport=transport, timeout=timeout)
        try:
            headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
            response = client.get(
                MODEL_ENDPOINTS_URL_TEMPLATE.format(model=model_key), headers=headers
            )
        finally:
            client.close()
        if response.status_code >= 400:
            return None
        payload = response.json()
        endpoints = ((payload.get("data") or {}).get("endpoints")) or []
        params: set[str] = set()
        for endpoint in endpoints:
            if isinstance(endpoint, dict):
                params.update(endpoint.get("supported_parameters") or [])
        return sorted(params) if params else None
    except Exception:  # noqa: BLE001 - best-effort metadata lookup, never fatal
        return None


def prime_json_strategy(
    model_key: str,
    *,
    api_key: str | None,
    transport: httpx.BaseTransport | None = None,
    timeout: float | httpx.Timeout = 30.0,
    force: bool = False,
) -> str:
    """Populate the process-wide JSON-strategy cache for `model_key` from a
    live `fetch_supported_parameters()` call. A repeat call returns the
    cached value unless `force=True`. Thread-safe.

    Unprimed model ids default to `STRATEGY_SCHEMA` in `_complete_json`.
    """
    with _json_strategy_cache_lock:
        if not force and model_key in _json_strategy_cache:
            return _json_strategy_cache[model_key]
    params = fetch_supported_parameters(
        model_key, api_key=api_key, transport=transport, timeout=timeout
    )
    strategy = _choose_json_strategy(params)
    with _json_strategy_cache_lock:
        if force or model_key not in _json_strategy_cache:
            _json_strategy_cache[model_key] = strategy
        return _json_strategy_cache[model_key]


def _downgrade_json_strategy(model_key: str, to: str) -> None:
    """Record a live rejection: move `model_key`'s cached strategy down
    `_STRATEGY_RANK`, never up, so an observed failure is never un-learned.
    """
    with _json_strategy_cache_lock:
        current = _json_strategy_cache.get(model_key, STRATEGY_SCHEMA)
        if _STRATEGY_RANK[to] < _STRATEGY_RANK.get(current, _STRATEGY_RANK[STRATEGY_SCHEMA]):
            _json_strategy_cache[model_key] = to


def reset_json_strategy_cache() -> None:
    """Clear the process-wide cache, so one test's learned strategy does not
    leak into another test using the same model id."""
    with _json_strategy_cache_lock:
        _json_strategy_cache.clear()


# Reasoning-locked models: some endpoints (gpt-oss-20b/120b) answer the
# default `reasoning: {"enabled": false}` with a 400, "Reasoning is mandatory
# for this endpoint and cannot be disabled." Such model ids never get the
# disable field, either declared up front in
# `PORTOLAN_OPENROUTER_REASONING_OFF_MODELS` (comma-separated, read on every
# call) or learned from that 400 by `_post_with_retry`, which resends the
# request without `reasoning` and caches the model id for the process.

_reasoning_locked_models: set[str] = set()
_reasoning_locked_models_lock = threading.Lock()

_REASONING_OFF_MODELS_ENV_VAR = "PORTOLAN_OPENROUTER_REASONING_OFF_MODELS"

# For reasoning-only models (e.g. moonshotai/kimi-k2-thinking), where
# disabling reasoning is rejected or ignored: `model_id=effort` pairs send
# OpenRouter's `reasoning: {"effort": ...}` ("low", "medium", "high") instead
# of the disable, capping reasoning spend. Read on every call.
_REASONING_EFFORT_ENV_VAR = "PORTOLAN_OPENROUTER_REASONING_EFFORT"


def _env_reasoning_locked_models() -> set[str]:
    raw = os.environ.get(_REASONING_OFF_MODELS_ENV_VAR, "")
    return {model.strip() for model in raw.split(",") if model.strip()}


def _env_reasoning_effort_overrides() -> dict[str, str]:
    """`PORTOLAN_OPENROUTER_REASONING_EFFORT="model_id=low,other/model=medium"`
    parsed into `{model_id: effort}`; malformed pairs are skipped."""
    raw = os.environ.get(_REASONING_EFFORT_ENV_VAR, "")
    overrides: dict[str, str] = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair or "=" not in pair:
            continue
        model_id, _, effort = pair.partition("=")
        model_id = model_id.strip()
        effort = effort.strip()
        if model_id and effort:
            overrides[model_id] = effort
    return overrides


def _should_send_reasoning_disable(model_key: str) -> bool:
    """False when `model_key` is known (env list or a learned 400) to reject
    `reasoning: {enabled: false}`; True otherwise."""
    if model_key in _env_reasoning_locked_models():
        return False
    with _reasoning_locked_models_lock:
        return model_key not in _reasoning_locked_models


def _mark_reasoning_locked(model_key: str) -> None:
    with _reasoning_locked_models_lock:
        _reasoning_locked_models.add(model_key)


def reset_reasoning_locked_cache() -> None:
    """Test helper, like `reset_json_strategy_cache()`."""
    with _reasoning_locked_models_lock:
        _reasoning_locked_models.clear()


_REASONING_MANDATORY_KEYWORDS = ("mandatory", "cannot be disabled")


def _is_reasoning_mandatory_rejection(response: httpx.Response) -> bool:
    """A 400 whose detail says reasoning is mandatory / cannot be disabled.
    Needs both "reasoning" and a keyword, so an unrelated 400 that mentions
    reasoning is not misread."""
    if response.status_code != 400:
        return False
    text = _error_detail_text(response).lower()
    return "reasoning" in text and any(keyword in text for keyword in _REASONING_MANDATORY_KEYWORDS)


def _strip_additional_properties_false(schema: Any) -> Any:
    """Recursively drop every `"additionalProperties": false` (a schema or
    `true` value is left alone). Part of `lenient_schema` in `_complete_json`:
    models without enforced structured output often add a plausible extra
    key, and failing the local check on it loses the whole reply. The strict
    `json_schema` request, and validation of its reply, keep the original.
    """
    if isinstance(schema, dict):
        return {
            key: _strip_additional_properties_false(value)
            for key, value in schema.items()
            if not (key == "additionalProperties" and value is False)
        }
    if isinstance(schema, list):
        return [_strip_additional_properties_false(item) for item in schema]
    return schema


def _relax_required_and_enum(schema: Any) -> Any:
    """Recursively drop every `"required"` array and `"enum"` constraint.
    The other half of `lenient_schema`: a missing field or an out-of-list
    value is left for the caller's own lenient parse to handle rather than
    rejecting the whole reply here.

    `lenient_schema` is rendered into the fallback's system instruction and
    validates both fallback replies and `json_object`-strategy replies; it is
    never sent as a `json_schema` `response_format`.
    """
    if isinstance(schema, dict):
        return {
            key: _relax_required_and_enum(value)
            for key, value in schema.items()
            if key not in ("required", "enum")
        }
    if isinstance(schema, list):
        return [_relax_required_and_enum(item) for item in schema]
    return schema


# Leaked "Document tree" recovery (`01-spec/corpus-fix-2026-09-19.md`). When
# a `json_schema` request reaches a model whose endpoints lack
# `response_format` support (amazon/nova-lite-v1), OpenRouter's shim can
# return part of the reply as Bedrock's internal tagged-union tree, often as a
# JSON-encoded string where a plain object belongs:
# `{"completionState": ..., "type": "Object"|"Array"|"String"|...,
#   "entries": [[key, node], ...] | "items": [node, ...] | "value": scalar}`.
# The helpers below convert such trees back to plain JSON, and detecting one
# downgrades the model's JSON strategy. Priming the strategy
# (`warm_json_strategy`) avoids the shim in the first place.

_TREE_SCALAR_TYPES = frozenset(
    {"String", "Number", "Integer", "Long", "Double", "Float", "Boolean", "Null"}
)


def _is_document_tree_node(value: Any) -> bool:
    """True for one node of the leaked tree described above. Requires
    `completionState`, because `type` is also a real field (`Fact.type`) and
    an ordinary fact dict must never be misread as a tree node."""
    if not isinstance(value, dict) or "completionState" not in value:
        return False
    if not isinstance(value.get("type"), str):
        return False
    return "entries" in value or "items" in value or "value" in value or value["type"] == "Null"


def _flatten_document_tree_node(node: Any) -> Any:
    """Convert one tree node, recursively, to the plain JSON value it
    represents. Anything that is not a recognizable node is returned
    unchanged."""
    if not isinstance(node, dict) or not isinstance(node.get("type"), str):
        return node
    node_type = node["type"]
    if node_type == "Object":
        entries = node.get("entries")
        if not isinstance(entries, list):
            return node
        result: dict[str, Any] = {}
        for entry in entries:
            if isinstance(entry, list | tuple) and len(entry) == 2:
                key, value_node = entry
                if isinstance(key, str):
                    result[key] = _flatten_document_tree_node(value_node)
        return result
    if node_type == "Array":
        items = node.get("items")
        if not isinstance(items, list):
            return node
        return [_flatten_document_tree_node(item) for item in items]
    if node_type == "Null":
        return None
    if node_type in _TREE_SCALAR_TYPES and "value" in node:
        return node["value"]
    return node


def _deep_flatten_document_trees(value: Any) -> tuple[Any, bool]:
    """Walk parsed JSON and convert every tree node found, whether a nested
    dict or a JSON-encoded string holding one. Returns `(value, changed)`;
    `changed` tells the caller to downgrade the model's JSON strategy.
    A string that does not parse to a tree node is left as it was.
    """
    if isinstance(value, dict):
        if _is_document_tree_node(value):
            return _flatten_document_tree_node(value), True
        changed = False
        result = {}
        for key, sub_value in value.items():
            new_value, sub_changed = _deep_flatten_document_trees(sub_value)
            result[key] = new_value
            changed = changed or sub_changed
        return result, changed
    if isinstance(value, list):
        changed = False
        result = []
        for item in value:
            new_item, sub_changed = _deep_flatten_document_trees(item)
            result.append(new_item)
            changed = changed or sub_changed
        return result, changed
    if isinstance(value, str):
        stripped = value.strip()
        if stripped[:1] == "{" and stripped[-1:] == "}":
            try:
                parsed = json.loads(stripped)
            except (json.JSONDecodeError, ValueError):
                return value, False
            if _is_document_tree_node(parsed):
                return _flatten_document_tree_node(parsed), True
        return value, False
    return value, False


@dataclass
class _OpenRouterUsageHandle:
    """Local copy of `portolan.metering._MeteringHandle` (including
    `cost_override`). `_metered_openrouter()` keeps `metered()`'s recording
    contract (one `UsageEvent`, `extra["error"]` and still recorded on an
    exception) and adds `on_cost`, which feeds the `SpendMeter`.
    """

    tenant: str
    surface: Surface
    model_key: str
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    request_id: str = ""
    extra: dict[str, Any] = field(default_factory=dict)
    cost_override: float | None = None


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


class _MeteredOpenRouterContext:
    def __init__(
        self,
        sink: UsageSink,
        tenant: str,
        surface: Surface,
        model_key: str,
        on_cost: Callable[[str, float], None] | None = None,
    ) -> None:
        self._sink = sink
        self._on_cost = on_cost
        # The dollars recorded for this call, set on exit (`None`: unknown).
        self.cost: float | None = None
        self.handle = _OpenRouterUsageHandle(
            tenant=tenant,
            surface=surface,
            model_key=model_key,
            request_id=f"req_{uuid.uuid4().hex}",
        )

    def __enter__(self) -> _OpenRouterUsageHandle:
        return self.handle

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        handle = self.handle
        if exc is not None:
            handle.extra = {**handle.extra, "error": str(exc)}

        cost = handle.cost_override
        if cost is None:
            try:
                cost = estimate_cost(
                    handle.model_key,
                    handle.input_tokens,
                    handle.output_tokens,
                    handle.cached_tokens,
                )
            except UnknownModelError as pricing_exc:
                cost = None
                handle.extra = {**handle.extra, "pricing_error": str(pricing_exc)}

        event = UsageEvent(
            tenant=handle.tenant,
            surface=handle.surface,
            model_key=handle.model_key,
            input_tokens=handle.input_tokens,
            output_tokens=handle.output_tokens,
            cached_tokens=handle.cached_tokens,
            cost_estimate=cost,
            request_id=handle.request_id,
            timestamp=_utc_now_iso(),
            extra=handle.extra,
        )
        self.cost = cost
        self._sink.record(event)
        if self._on_cost is not None and cost:
            self._on_cost(handle.model_key, cost)
        return False  # never swallow the exception


def _metered_openrouter(
    sink: UsageSink,
    tenant: str,
    surface: Surface,
    model_key: str,
    on_cost: Callable[[str, float], None] | None = None,
) -> _MeteredOpenRouterContext:
    return _MeteredOpenRouterContext(sink, tenant, surface, model_key, on_cost)


def _extract_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Some providers/routes return OpenAI-style content-block lists
        # instead of a plain string.
        return "".join(
            part.get("text", "") for part in content if isinstance(part, dict) and "text" in part
        )
    return ""


def _cost_from_usage(usage_block: dict[str, Any]) -> float | None:
    """`usage.cost`, "the total amount charged to your account" (usage
    accounting docs), or None when the response carries no numeric cost."""
    cost = usage_block.get("cost")
    if isinstance(cost, int | float):
        return float(cost)
    return None


def _cached_tokens_from_usage(usage_block: dict[str, Any]) -> int:
    details = usage_block.get("prompt_tokens_details") or {}
    return int(details.get("cached_tokens", 0) or 0)


def _response_extras(usage_block: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """Reasoning tokens (billed as output, docs/guides/best-practices/reasoning-tokens),
    and the provider and service tier that served the request (billing follows the tier)."""
    extras: dict[str, Any] = {}
    details = usage_block.get("completion_tokens_details") or {}
    reasoning = details.get("reasoning_tokens") if isinstance(details, dict) else None
    if isinstance(reasoning, int | float):
        extras["reasoning_tokens"] = int(reasoning)
    for key in ("provider", "service_tier"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            extras[key] = value
    return extras


def _retry_after_seconds(response: httpx.Response) -> float | None:
    value = response.headers.get("Retry-After")
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _default_jitter(base_seconds: float) -> float:
    """+/- `_JITTER_FRACTION` of `base_seconds`, so worker threads that were
    rate-limited together do not all retry on the same tick."""
    spread = base_seconds * _JITTER_FRACTION
    return base_seconds + random.uniform(-spread, spread)


def _error_payload(response: httpx.Response) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError:
        return {}
    error = payload.get("error")
    return error if isinstance(error, dict) else {}


def _error_detail_text(response: httpx.Response) -> str:
    """All provider-error text in this response, joined: `error.message`
    (often just "Provider returned error"), `error.metadata.raw` (the
    upstream provider's own error body), `provider_name` and
    `metadata.previous_errors`. Built from the response only, never the
    request, so the API key is never interpolated.
    """
    error = _error_payload(response)
    if not error:
        return response.text
    parts = [str(error.get("message", "")).strip()]
    metadata = error.get("metadata")
    if isinstance(metadata, dict):
        raw = metadata.get("raw")
        if raw:
            parts.append(str(raw).strip())
        provider_name = metadata.get("provider_name")
        if provider_name:
            parts.append(f"(provider: {provider_name})")
        for previous in metadata.get("previous_errors") or []:
            if isinstance(previous, dict):
                prev_text = previous.get("raw") or previous.get("message")
                if prev_text:
                    parts.append(str(prev_text).strip())
    return " | ".join(p for p in parts if p)


_STRUCTURED_OUTPUT_REJECTION_KEYWORDS = (
    "response_format",
    "json_schema",
    "structured",
    "not supported",
)


def _is_response_format_rejection(response: httpx.Response) -> bool:
    """Best-effort sniff of "this endpoint rejects the `response_format` just
    sent" (the docs do not fix the error shape). Only a 4xx whose
    `_error_detail_text()` mentions one of
    `_STRUCTURED_OUTPUT_REJECTION_KEYWORDS` counts, so an auth, quota or
    model-not-found error is not swallowed into the fallback path.
    """
    if response.status_code < 400 or response.status_code >= 500:
        return False
    text = _error_detail_text(response).lower()
    return any(keyword in text for keyword in _STRUCTURED_OUTPUT_REJECTION_KEYWORDS)


def _raise_for_status(response: httpx.Response, *, model_key: str) -> None:
    """Raise `LLMError` for a 4xx/5xx, with the response's status and headers
    attached. The message interpolates only response text, never the API key.
    """
    if response.status_code < 400:
        return
    message = _error_detail_text(response)
    raise LLMError(
        f"OpenRouterClient: request failed with status {response.status_code} "
        f"(model={model_key!r}): {message}",
        status_code=response.status_code,
        headers=dict(response.headers),
    )


class OpenRouterClient:
    """`LLMClient` implementation over OpenRouter's chat completions API.

    `transport` goes straight into the internal `httpx.Client` (tests pass an
    `httpx.MockTransport`). `sleep` and `jitter` (a `float -> float` applied
    to each computed backoff) are injectable so retry tests neither wait nor
    guess the delay.

    `spend_key` is required (CLAUDE.md: every paid call is metered under a
    named key; `tests/test_paid_calls_metered.py`). With `spend_meter`, each
    request's non-zero cost is also recorded there under `spend_key`;
    without one, recording spend is the caller's job.
    """

    def __init__(
        self,
        *,
        spend_key: str,
        spend_meter: SpendMeter | None = None,
        api_key: str | None = None,
        default_model: str = DEFAULT_MODEL,
        provider_order: list[str] | None = None,
        sink: UsageSink | None = None,
        tenant: str = "dev",
        transport: httpx.BaseTransport | None = None,
        timeout: float | httpx.Timeout = _DEFAULT_TIMEOUT,
        sleep: Callable[[float], None] = time.sleep,
        jitter: Callable[[float], float] = _default_jitter,
        max_retries: int = _MAX_RETRIES,
        reasoning: dict[str, Any] | None = None,
        service_tier: str | None = None,
    ) -> None:
        if not spend_key or not spend_key.strip():
            raise ValueError("OpenRouterClient needs a spend_key naming what its calls are for")
        self.spend_key = spend_key
        # `reasoning` replaces the default disable on every request (e.g. {"effort": "max"});
        # `service_tier` ("flex") is OpenRouter's top-level tier field, docs/guides/features/
        # service-tiers: flex never falls back to a default-tier endpoint.
        self.reasoning = dict(reasoning) if reasoning is not None else None
        self.service_tier = service_tier
        self.spend_meter = spend_meter
        self._on_cost = self._record_spend if spend_meter is not None else None
        self._api_key = api_key
        self.default_model = default_model
        self.provider_order = provider_order
        self.sink = sink or InMemorySink()
        self.tenant = tenant
        self._transport = transport
        self._timeout = timeout
        self._sleep = sleep
        self._jitter = jitter
        # A caller with its own model rotation passes a smaller budget so a
        # congested model fails over quickly instead of backing off for minutes.
        self._max_retries = max_retries
        self._client: httpx.Client | None = None

    def _record_spend(self, model_key: str, cost: float) -> None:
        self.spend_meter.record(model=model_key, corpus=self.spend_key, amount_usd=cost)

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(transport=self._transport, timeout=self._timeout)
        return self._client

    def _headers(self) -> dict[str, str]:
        if not self._api_key:
            raise RuntimeError(
                "OpenRouterClient has no API key: set OPENROUTER_API_KEY in the "
                "environment or .env (portolan.config.Settings.openrouter_api_key)."
            )
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "Portolan",
            "X-Title": "Portolan",
        }

    def _base_body(
        self,
        model_key: str,
        system: str,
        user: str,
        max_tokens: int,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": model_key,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            # Deprecated per the docs (usage is always included now) but harmless.
            "usage": {"include": True},
        }
        # Reasoning spends the same `max_tokens` budget as the answer, so a
        # reasoning-by-default model can return empty content (a 16-token judge
        # verdict especially). Disable it unless the model is configured with an
        # effort level or is known to reject the disable (see module comments).
        effort_overrides = _env_reasoning_effort_overrides()
        if self.reasoning is not None:
            body["reasoning"] = dict(self.reasoning)
        elif model_key in effort_overrides:
            body["reasoning"] = {"effort": effort_overrides[model_key]}
        elif _should_send_reasoning_disable(model_key):
            body["reasoning"] = {"enabled": False}
        if self.provider_order:
            body["provider"] = {"order": list(self.provider_order)}
        if self.service_tier:
            body["service_tier"] = self.service_tier
        # Omitted when `None`, leaving the upstream provider's default.
        if temperature is not None:
            body["temperature"] = temperature
        return body

    def _backoff_seconds(self, attempt: int) -> float:
        return self._jitter(_BASE_BACKOFF_SECONDS * (2**attempt))

    def _post_with_retry(
        self, body: dict[str, Any], *, timeout: float | None = None
    ) -> httpx.Response:
        """`timeout` (seconds) replaces the client's own for this request."""
        client = self._get_client()
        headers = self._headers()
        attempt = 0
        current_body = body
        reasoning_fallback_tried = False
        extra: dict[str, Any] = {"timeout": timeout} if timeout is not None else {}
        while True:
            try:
                response = client.post(
                    CHAT_COMPLETIONS_URL, headers=headers, json=current_body, **extra
                )
            except httpx.TransportError as exc:
                # Timeouts and dropped connections ("connection reset by peer") are retried
                # on the same budget as a 429/5xx, then raised as LLMError.
                if attempt >= self._max_retries:
                    kind = "timed out" if isinstance(exc, httpx.TimeoutException) else "failed"
                    raise LLMError(
                        f"OpenRouterClient: request {kind} after {attempt + 1} "
                        f"attempt(s) (model={current_body.get('model')!r}): {exc}"
                    ) from exc
                self._sleep(self._backoff_seconds(attempt))
                attempt += 1
                continue

            # "Reasoning is mandatory" 400: resend once without `reasoning`,
            # immediately and outside the retry budget (a request-shape fix, not
            # congestion), and remember the model id for the process.
            if (
                response.status_code == 400
                and not reasoning_fallback_tried
                and "reasoning" in current_body
                and _is_reasoning_mandatory_rejection(response)
            ):
                model_key = current_body.get("model")
                if isinstance(model_key, str) and model_key:
                    _mark_reasoning_locked(model_key)
                current_body = {k: v for k, v in current_body.items() if k != "reasoning"}
                reasoning_fallback_tried = True
                continue

            retryable = response.status_code in _RETRYABLE_STATUS or response.status_code >= 500
            if not retryable or attempt >= self._max_retries:
                return response
            delay = _retry_after_seconds(response)
            if delay is None:
                delay = self._backoff_seconds(attempt)
            self._sleep(delay)
            attempt += 1

    def _record_response_usage(
        self,
        usage: _OpenRouterUsageHandle,
        usage_block: dict[str, Any],
        cost: float | None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Fill an open `_metered_openrouter` handle from one response's `usage`
        block, so every billed request is metered the same way."""
        usage.input_tokens = int(usage_block.get("prompt_tokens", 0) or 0)
        usage.output_tokens = int(usage_block.get("completion_tokens", 0) or 0)
        usage.cached_tokens = _cached_tokens_from_usage(usage_block)
        if cost is not None:
            usage.cost_override = cost
            usage.extra["cost_source"] = "openrouter"
        usage.extra.update(_response_extras(usage_block, payload or {}))

    def _complete_text(
        self,
        model_key: str,
        system: str,
        user: str,
        max_tokens: int,
        temperature: float | None = None,
        *,
        surface: Surface,
    ) -> tuple[str, int, int, int]:
        # Metered per attempt, so a failed attempt is still recorded
        # (0 tokens, `extra["error"]`).
        with _metered_openrouter(
            self.sink, self.tenant, surface, model_key, self._on_cost
        ) as usage:
            response = self._post_with_retry(
                self._base_body(model_key, system, user, max_tokens, temperature)
            )
            _raise_for_status(response, model_key=model_key)
            payload = response.json()
            usage_block = payload.get("usage") or {}
            self._record_response_usage(usage, usage_block, _cost_from_usage(usage_block), payload)
        return _extract_text(payload), usage.input_tokens, usage.output_tokens, usage.cached_tokens

    def _extract_reply(
        self, text: str, json_schema: dict[str, Any], *, validate_json: bool
    ) -> tuple[dict[str, Any] | None, str | None, bool]:
        """Extract the reply's JSON object and, when `validate_json`, check it
        against `json_schema`. Returns `(data, error, tree_detected)`.

        The Document-tree flatten runs first, always, so a leaked tree is
        converted before validation could reject it. `validate_json=False`
        returns any JSON object as-is, leaving leniency to the caller.
        """
        obj = extract_json_object(text)
        if obj is None:
            return None, "no JSON object found in the response", False
        data, tree_detected = _deep_flatten_document_trees(obj)
        if not validate_json:
            return data, None, tree_detected
        error = validate_json_object(data, json_schema)
        if error is not None:
            return None, error, tree_detected
        return data, None, tree_detected

    def _complete_json_fallback(
        self,
        model_key: str,
        system: str,
        user: str,
        json_schema: dict[str, Any],
        max_tokens: int,
        *,
        surface: Surface,
        validate_json: bool,
        temperature: float | None = None,
    ) -> tuple[str, dict[str, Any], int, int, int]:
        """`response_format` was rejected, skipped by the cached strategy, or
        accepted with unusable content: send a plain system instruction plus
        `json_util` extraction, with one retry that appends the error.
        `json_schema` here is always the lenient schema from `_complete_json`.

        Each of the two possible requests is metered as its own `UsageEvent`
        when its response arrives, so a retry that ends in `LLMJsonError` is
        still recorded.
        """
        fallback_system = system + JSON_INSTRUCTION_TEMPLATE.format(schema=json.dumps(json_schema))

        with _metered_openrouter(
            self.sink, self.tenant, surface, model_key, self._on_cost
        ) as usage:
            response = self._post_with_retry(
                self._base_body(model_key, fallback_system, user, max_tokens, temperature)
            )
            _raise_for_status(response, model_key=model_key)
            payload = response.json()
            usage_block = payload.get("usage") or {}
            self._record_response_usage(usage, usage_block, _cost_from_usage(usage_block), payload)
        text = _extract_text(payload)
        data, error, tree_detected = self._extract_reply(
            text, json_schema, validate_json=validate_json
        )
        if tree_detected:
            # A tree even without `response_format`: pin the model to the fallback.
            _downgrade_json_strategy(model_key, STRATEGY_NONE)
        if data is not None:
            return text, data, usage.input_tokens, usage.output_tokens, usage.cached_tokens

        retry_user = RETRY_TEMPLATE.format(user=user, error=error)
        with _metered_openrouter(
            self.sink, self.tenant, surface, model_key, self._on_cost
        ) as usage:
            response = self._post_with_retry(
                self._base_body(model_key, fallback_system, retry_user, max_tokens, temperature)
            )
            _raise_for_status(response, model_key=model_key)
            payload = response.json()
            usage_block = payload.get("usage") or {}
            self._record_response_usage(usage, usage_block, _cost_from_usage(usage_block), payload)
        text = _extract_text(payload)
        data, error, tree_detected = self._extract_reply(
            text, json_schema, validate_json=validate_json
        )
        if tree_detected:
            _downgrade_json_strategy(model_key, STRATEGY_NONE)
        if data is None:
            raise LLMJsonError(
                f"OpenRouterClient: model did not return schema-valid JSON after the "
                f"json_util fallback retry (model={model_key!r}): {error}"
            )
        return text, data, usage.input_tokens, usage.output_tokens, usage.cached_tokens

    def _complete_json(
        self,
        model_key: str,
        system: str,
        user: str,
        json_schema: dict[str, Any],
        max_tokens: int,
        *,
        surface: Surface,
        validate_json: bool,
        temperature: float | None = None,
    ) -> tuple[str, dict[str, Any], int, int, int]:
        lenient_schema = _relax_required_and_enum(_strip_additional_properties_false(json_schema))
        strategy = _json_strategy_cache.get(model_key, STRATEGY_SCHEMA)

        if strategy == STRATEGY_NONE:
            return self._complete_json_fallback(
                model_key,
                system,
                user,
                lenient_schema,
                max_tokens,
                surface=surface,
                validate_json=validate_json,
                temperature=temperature,
            )

        if strategy == STRATEGY_SCHEMA:
            body = self._base_body(model_key, system, user, max_tokens, temperature)
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "portolan_response", "strict": True, "schema": json_schema},
            }
        else:
            # STRATEGY_OBJECT: `json_object` carries no schema, so the schema goes in the
            # system prompt, as in the fallback. Without it a model whose endpoint rejects our
            # strict schema (OpenAI strict mode wants every property required) guesses fields.
            object_system = system + JSON_INSTRUCTION_TEMPLATE.format(
                schema=json.dumps(lenient_schema)
            )
            body = self._base_body(model_key, object_system, user, max_tokens, temperature)
            body["response_format"] = {"type": "json_object"}

        response = self._post_with_retry(body)

        if response.status_code >= 400:
            if _is_response_format_rejection(response):
                # Downgrade one rung for the rest of the process. The rejected
                # request is not metered; the fallback's requests are.
                downgrade_to = STRATEGY_OBJECT if strategy == STRATEGY_SCHEMA else STRATEGY_NONE
                _downgrade_json_strategy(model_key, downgrade_to)
                return self._complete_json_fallback(
                    model_key,
                    system,
                    user,
                    lenient_schema,
                    max_tokens,
                    surface=surface,
                    validate_json=validate_json,
                    temperature=temperature,
                )
            # Any other error is metered as a failed request, then raised.
            with _metered_openrouter(self.sink, self.tenant, surface, model_key, self._on_cost):
                _raise_for_status(response, model_key=model_key)

        payload = response.json()
        usage_block = payload.get("usage") or {}
        cost = _cost_from_usage(usage_block)
        with _metered_openrouter(
            self.sink, self.tenant, surface, model_key, self._on_cost
        ) as usage:
            self._record_response_usage(usage, usage_block, cost, payload)
        text = _extract_text(payload)
        validate_against = json_schema if strategy == STRATEGY_SCHEMA else lenient_schema
        data, _error, tree_detected = self._extract_reply(
            text, validate_against, validate_json=validate_json
        )
        if tree_detected:
            # A leaked tree means structured output here can't be trusted:
            # downgrade one rung, as for a rejection.
            downgrade_to = STRATEGY_OBJECT if strategy == STRATEGY_SCHEMA else STRATEGY_NONE
            _downgrade_json_strategy(model_key, downgrade_to)
        if data is not None:
            return text, data, usage.input_tokens, usage.output_tokens, usage.cached_tokens
        # Accepted (2xx) but unusable content: try the fallback, which meters
        # its own requests on top of this one.
        return self._complete_json_fallback(
            model_key,
            system,
            user,
            lenient_schema,
            max_tokens,
            surface=surface,
            validate_json=validate_json,
            temperature=temperature,
        )

    def complete(
        self,
        *,
        system: str,
        user: str,
        json_schema: dict[str, Any] | None = None,
        model: str | None = None,
        surface: Surface,
        max_tokens: int = 4096,
        validate_json: bool = True,
        temperature: float | None = None,
    ) -> LLMResult:
        """`temperature` goes through `resolve_temperature(temperature, surface)`.

        `validate_json` (default `True`): with a `json_schema`, `True` means
        this client's local jsonschema check decides whether a reply is good
        enough (a mismatch is retried, then raises `LLMJsonError`). `False`
        returns any parsed JSON object and leaves leniency to the caller.
        Ignored without a `json_schema`.

        This method meters nothing itself: each attempt below (native,
        fallback, fallback retry) records its own `UsageEvent`.
        """
        model_key = model or self.default_model
        effective_temperature = resolve_temperature(temperature, surface)

        if json_schema is not None:
            text, data, input_tokens, output_tokens, cached_tokens = self._complete_json(
                model_key,
                system,
                user,
                json_schema,
                max_tokens,
                surface=surface,
                validate_json=validate_json,
                temperature=effective_temperature,
            )
        else:
            text, input_tokens, output_tokens, cached_tokens = self._complete_text(
                model_key, system, user, max_tokens, effective_temperature, surface=surface
            )
            data = None

        return LLMResult(
            text=text,
            data=data,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            model=model_key,
        )

    def complete_with_tools(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        model: str | None = None,
        surface: Surface,
        tool_choice: str | dict[str, Any] = "auto",
        max_tokens: int = 4096,
        temperature: float | None = None,
        timeout: float | None = None,
    ) -> ToolTurn:
        """One turn of a tool loop (`portolan.llm.base.ToolChatClient`):
        `messages` and `tools` are sent as given, in chat-completions form.
        Metered like every other request here, retries included."""
        model_key = model or self.default_model
        body = self._base_body(
            model_key, "", "", max_tokens, resolve_temperature(temperature, surface)
        )
        body["messages"] = messages
        body["tools"] = tools
        body["tool_choice"] = tool_choice
        metering = _metered_openrouter(self.sink, self.tenant, surface, model_key, self._on_cost)
        with metering as usage:
            response = self._post_with_retry(body, timeout=timeout)
            _raise_for_status(response, model_key=model_key)
            payload = response.json()
            usage_block = payload.get("usage") or {}
            self._record_response_usage(usage, usage_block, _cost_from_usage(usage_block), payload)
        choices = payload.get("choices") or []
        message = (choices[0].get("message") if choices else None) or {}
        calls: list[ToolCall] = []
        for raw in message.get("tool_calls") or []:
            function = raw.get("function") or {}
            raw_arguments = function.get("arguments") or ""
            arguments = raw_arguments if isinstance(raw_arguments, dict) else None
            if isinstance(raw_arguments, str):
                try:
                    parsed = json.loads(raw_arguments) if raw_arguments.strip() else {}
                except json.JSONDecodeError:
                    parsed = None
                arguments = parsed if isinstance(parsed, dict) else None
            calls.append(
                ToolCall(
                    id=str(raw.get("id") or f"call_{uuid.uuid4().hex[:12]}"),
                    name=str(function.get("name") or ""),
                    arguments=arguments,
                    raw_arguments=raw_arguments
                    if isinstance(raw_arguments, str)
                    else json.dumps(raw_arguments),
                )
            )
        return ToolTurn(
            text=_extract_text(payload),
            tool_calls=calls,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cached_tokens=usage.cached_tokens,
            cost_usd=metering.cost,
            model=model_key,
            finish_reason=choices[0].get("finish_reason") if choices else None,
        )

    @contextmanager
    def using_sink(self, sink: UsageSink, tenant: str) -> Iterator[None]:
        """Meter every `complete()` call inside this block into `sink`/`tenant`,
        restoring the originals on exit, even if the block raises (see
        `portolan.llm.base`). Mutates the instance with no lock: give each
        worker thread its own client.
        """
        previous_sink, previous_tenant = self.sink, self.tenant
        self.sink, self.tenant = sink, tenant
        try:
            yield
        finally:
            self.sink, self.tenant = previous_sink, previous_tenant

    def warm_json_strategy(self, model: str | None = None) -> str:
        """`prime_json_strategy()` with this client's `api_key`, `transport`
        and `timeout`. Call it before a run so every worker's client starts
        from the primed, process-wide answer.
        """
        return prime_json_strategy(
            model or self.default_model,
            api_key=self._api_key,
            transport=self._transport,
            timeout=self._timeout,
        )
