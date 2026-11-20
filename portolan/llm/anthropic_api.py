"""`LLMClient` backed directly by the first-party Anthropic Messages API
through the `anthropic` SDK (written against `anthropic==1.5.0`). Needs only
an API key, so a hosted deployment has Claude without a logged-in CLI.

Request shapes follow the `claude-api` skill docs:
- `system` carries an ephemeral `cache_control` breakpoint (prompt caching).
- `thinking={"type": "adaptive"}` for every model except Haiku 4.5, which
  has no adaptive mode, so no `thinking` field is sent for it.
- `output_config={"format": {"type": "json_schema", ...}}` for a raw JSON
  Schema (the caller passes a dict, not a Pydantic model).
- `stop_reason == "refusal"` raises `LLMRefusal`; no fallback is attempted.

The `anthropic.Anthropic` client is built lazily on first `.complete()`, or
never when a test injects `client=`.
"""

from __future__ import annotations

from typing import Any

from portolan.llm.base import LLMError, LLMJsonError, LLMResult, resolve_temperature
from portolan.llm.json_util import extract_and_validate
from portolan.metering import InMemorySink, Surface, UsageSink, metered
from portolan.pricing import UnknownModelError, estimate_cost_with_cache

DEFAULT_EXTRACTION_MODEL = "claude-haiku-4-5"
DEFAULT_STRONG_MODEL = "claude-sonnet-5"


class LLMRefusal(LLMError):
    """Raised when the model's `stop_reason` is `"refusal"`: a safety
    classifier declined the request rather than the call failing.
    """


def _is_haiku(model_key: str) -> bool:
    return "haiku" in model_key.lower()


def _thinking_config(model_key: str) -> dict[str, Any] | None:
    """`None` means "send no `thinking` field". Haiku 4.5 has no adaptive
    mode (only a fixed `budget_tokens`), so thinking stays off for it; every
    other model gets adaptive thinking.
    """
    if _is_haiku(model_key):
        return None
    return {"type": "adaptive"}


class AnthropicAPIClient:
    """`LLMClient` implementation using `anthropic.Anthropic().messages.stream()`.

    `client` is normally `None` and a real `anthropic.Anthropic` is built
    from `api_key` on first use. Tests pass a fake with the same
    `.messages.stream(**kwargs)` context-manager shape.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        default_model: str = DEFAULT_EXTRACTION_MODEL,
        sink: UsageSink | None = None,
        tenant: str = "dev",
        client: Any | None = None,
    ) -> None:
        self._api_key = api_key
        self.default_model = default_model
        self.sink = sink or InMemorySink()
        self.tenant = tenant
        self._client = client

    def _get_client(self) -> Any:
        if self._client is None:
            if not self._api_key:
                raise RuntimeError(
                    "AnthropicAPIClient has no API key: set ANTHROPIC_API_KEY in the "
                    "environment or .env (portolan.config.Settings.anthropic_api_key)."
                )
            import anthropic

            # Never log or print self._api_key -- passed straight to the
            # SDK constructor and nowhere else in this module.
            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    def complete(
        self,
        *,
        system: str,
        user: str,
        json_schema: dict[str, Any] | None = None,
        model: str | None = None,
        surface: Surface,
        max_tokens: int = 4096,
        temperature: float | None = None,
    ) -> LLMResult:
        model_key = model or self.default_model
        client = self._get_client()

        request: dict[str, Any] = {
            "model": model_key,
            "max_tokens": max_tokens,
            # Cache breakpoint on the stable system prompt; the per-call user
            # message stays out of the cached prefix.
            "system": [
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": [{"role": "user", "content": user}],
        }

        thinking = _thinking_config(model_key)
        if thinking is not None:
            request["thinking"] = thinking

        # Temperature is omitted whenever thinking is on: the API requires it to
        # be 1 or absent then, so only Haiku calls here ever carry one.
        effective_temperature = resolve_temperature(temperature, surface)
        if effective_temperature is not None and thinking is None:
            request["temperature"] = effective_temperature

        if json_schema is not None:
            request["output_config"] = {"format": {"type": "json_schema", "schema": json_schema}}

        with metered(self.sink, self.tenant, surface, model_key) as usage:
            # Streaming, because long outputs time out on a plain messages.create().
            with client.messages.stream(**request) as stream:
                response = stream.get_final_message()

            response_usage = response.usage
            # Cache writes are counted in `input_tokens` and priced at the write
            # premium via `cost_override`; cache reads go in `cached_tokens` only.
            base_input_tokens = getattr(response_usage, "input_tokens", 0) or 0
            cache_read = getattr(response_usage, "cache_read_input_tokens", 0) or 0
            cache_creation = getattr(response_usage, "cache_creation_input_tokens", 0) or 0

            usage.output_tokens = getattr(response_usage, "output_tokens", 0) or 0
            usage.cached_tokens = cache_read
            usage.input_tokens = base_input_tokens + cache_creation
            usage.request_id = getattr(response, "_request_id", "") or ""

            if cache_creation:
                usage.extra["cache_creation_input_tokens"] = cache_creation
            try:
                usage.cost_override = estimate_cost_with_cache(
                    model_key,
                    base_input_tokens,
                    usage.output_tokens,
                    cache_read_tokens=cache_read,
                    cache_creation_tokens=cache_creation,
                )
            except UnknownModelError as pricing_exc:
                usage.extra["pricing_error"] = str(pricing_exc)

            if response.stop_reason == "refusal":
                stop_details = getattr(response, "stop_details", None)
                category = getattr(stop_details, "category", None) if stop_details else None
                usage.extra["refusal_category"] = category
                raise LLMRefusal(
                    f"AnthropicAPIClient: model refused the request "
                    f"(surface={surface!r}, model={model_key!r}, category={category!r})"
                )

            text = "".join(
                block.text for block in response.content if getattr(block, "type", None) == "text"
            )

            data = None
            if json_schema is not None:
                data, error = extract_and_validate(text, json_schema)
                if data is None:
                    raise LLMJsonError(
                        f"AnthropicAPIClient: structured output did not validate against "
                        f"the requested schema (surface={surface!r}, model={model_key!r}): "
                        f"{error}"
                    )

        return LLMResult(
            text=text,
            data=data,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cached_tokens=usage.cached_tokens,
            model=model_key,
        )
