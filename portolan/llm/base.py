"""The contract every `LLMClient` honours: `complete()` returns an
`LLMResult`, and each implementation meters its own model calls into its
`sink` (via `portolan.metering.metered` or a local equivalent).

Optional extension, deliberately not on the `Protocol`:
`using_sink(sink, tenant)`, a context manager that meters `complete()` calls
into another sink for its duration and restores the original on exit. Check
for it with `getattr(client, "using_sink", None)`; a bench run uses it to
read one run's tokens and cost back from its own `InMemorySink`.
"""

from __future__ import annotations

import os
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict

from portolan.metering import Surface


class LLMResult(BaseModel):
    """What every `LLMClient.complete()` call returns.

    `text` is the model's raw text reply, kept even when a `json_schema` was
    given. `data` is the parsed JSON object; whether it was also validated
    against the schema depends on the client (`validate_json=False` on
    `OpenRouterClient`/`MLXClient`, and `FakeLLMClient`, skip validation).
    """

    model_config = ConfigDict(extra="forbid")

    text: str | None
    data: dict[str, Any] | None
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    model: str


class ToolCall(BaseModel):
    """One function call a model asked for. `arguments` is the parsed JSON
    object, or `None` when `raw_arguments` did not parse as one."""

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    arguments: dict[str, Any] | None
    raw_arguments: str = ""


class ToolTurn(BaseModel):
    """One model turn of a tool loop (`ToolChatClient.complete_with_tools`).
    `cost_usd` is what the call cost, the provider's figure when it sent one;
    `None` when neither the provider nor `portolan.pricing` knows it."""

    model_config = ConfigDict(extra="forbid")

    text: str
    tool_calls: list[ToolCall]
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    cost_usd: float | None
    model: str
    finish_reason: str | None = None


class ToolChatClient(Protocol):
    """A client that can run one turn of an OpenAI-shaped tool loop:
    `messages` and `tools` in chat-completions form, `tool_choice` "auto" or
    `{"type": "function", "function": {"name": ...}}`. `timeout` bounds this
    one request, in seconds. Each call meters itself, like `complete()`."""

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
    ) -> ToolTurn: ...


class LLMClient(Protocol):
    """Anything that can complete a system/user prompt pair, optionally
    enforcing a JSON schema on the reply."""

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
    ) -> LLMResult: ...


# One temperature rule for every client, decided from `surface`, so the
# clients cannot drift apart.

#: Surfaces that default to a deterministic temperature (extraction and the
#: judge). Every other surface resolves to `None` unless the caller asks.
DETERMINISTIC_TEMPERATURE_SURFACES: frozenset[Surface] = frozenset({"extract", "audit"})

#: Read fresh on every call; the bench CLI's `--temperature` flag sets it so
#: the value reaches the client without a new call-site parameter.
TEMPERATURE_ENV_VAR = "PORTOLAN_TEMPERATURE"

DEFAULT_DETERMINISTIC_TEMPERATURE = 0.0


def resolve_temperature(explicit: float | None, surface: Surface) -> float | None:
    """The temperature a client should send, or `None` to send none (the
    provider's default).

    An explicit argument always wins. Otherwise, for a surface in
    `DETERMINISTIC_TEMPERATURE_SURFACES`: `TEMPERATURE_ENV_VAR` when it
    parses as a float, else `DEFAULT_DETERMINISTIC_TEMPERATURE`. Every other
    surface gets `None`.
    """
    if explicit is not None:
        return explicit
    if surface not in DETERMINISTIC_TEMPERATURE_SURFACES:
        return None
    raw = os.environ.get(TEMPERATURE_ENV_VAR)
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    return DEFAULT_DETERMINISTIC_TEMPERATURE


class LLMError(Exception):
    """Base class for `portolan.llm` failures that aren't just "the SDK/
    boto3 client raised" -- those propagate unchanged so callers see the
    real exception type.

    `status_code`/`headers`: the HTTP status and response headers of the
    request that failed, when the raising client has them
    (`OpenRouterClient._raise_for_status` sets both); otherwise `None` and
    `{}`. Lets a caller react to a 429/5xx and `Retry-After` without
    parsing the message."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.headers = headers or {}


class LLMJsonError(LLMError):
    """Raised when a `json_schema` was requested and the reply did not yield
    valid JSON. Whether a retry came first depends on the client."""
