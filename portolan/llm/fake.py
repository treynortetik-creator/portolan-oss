"""Test double for `LLMClient`: canned output from a list (in call order) or a
callable, still metered on every call so metering tests need no real model.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from portolan.llm.base import LLMResult
from portolan.llm.json_util import extract_json_object
from portolan.metering import InMemorySink, Surface, UsageSink, metered

# A canned response: plain text, a dict (used as `data` directly), or a full
# LLMResult for a test that controls every field.
FakeResponse = str | dict[str, Any] | LLMResult
ResponsesArg = list[FakeResponse] | Callable[..., FakeResponse]


class FakeLLMClient:
    """`responses` is either:

    - a list of canned responses, consumed in call order (one per
      `.complete()` call; a call past the end of the list raises
      `IndexError` with a clear message rather than a bare index error), or
    - a callable `(*, system, user, json_schema, model, surface) ->
      FakeResponse`, invoked fresh on every call.

    Every call records exactly one `UsageEvent` via `portolan.metering`,
    with deterministic token counts from text length (chars / 4) unless the
    canned response is itself an `LLMResult`, whose counts are used as-is.
    """

    def __init__(
        self,
        responses: ResponsesArg,
        *,
        # a real portolan.pricing key; an unknown one records cost_estimate=None
        model_key: str = "nova-lite",
        sink: UsageSink | None = None,
        tenant: str = "dev",
    ) -> None:
        self._responses = responses
        self._call_count = 0
        self.model_key = model_key
        self.sink = sink or InMemorySink()
        self.tenant = tenant

    def _next_raw(
        self,
        *,
        system: str,
        user: str,
        json_schema: dict[str, Any] | None,
        model: str | None,
        surface: Surface,
    ) -> FakeResponse:
        if callable(self._responses):
            return self._responses(
                system=system, user=user, json_schema=json_schema, model=model, surface=surface
            )
        try:
            raw = self._responses[self._call_count]
        except IndexError as exc:
            raise IndexError(
                f"FakeLLMClient has no canned response for call #{self._call_count + 1} "
                f"(only {len(self._responses)} provided)"
            ) from exc
        finally:
            self._call_count += 1
        return raw

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
        # `temperature` is only recorded in usage.extra; there is no request to carry it.
        model_key = model or self.model_key
        raw = self._next_raw(
            system=system, user=user, json_schema=json_schema, model=model, surface=surface
        )

        if isinstance(raw, LLMResult):
            result = raw
        else:
            if isinstance(raw, dict):
                data = raw
                text = json.dumps(raw)
            else:
                text = str(raw)
                data = extract_json_object(text) if json_schema is not None else None
            result = LLMResult(
                text=text,
                data=data,
                input_tokens=max(1, len(system) // 4) + max(1, len(user) // 4),
                output_tokens=max(1, len(text) // 4),
                cached_tokens=0,
                model=model_key,
            )

        with metered(self.sink, self.tenant, surface, model_key) as usage:
            usage.input_tokens = result.input_tokens
            usage.output_tokens = result.output_tokens
            usage.cached_tokens = result.cached_tokens
            usage.extra["backend"] = "fake"
            if temperature is not None:
                usage.extra["temperature_requested"] = temperature

        return result
