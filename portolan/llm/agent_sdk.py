"""`LLMClient` backed by the Claude Agent SDK (`claude-agent-sdk`), which
talks to Claude through the local Claude Code CLI login -- no API key
(adr-001-language-and-stack.md "Local model access"). Written against
`claude-agent-sdk==0.2.152`.

Isolation: every call passes `tools=[]`, `setting_sources=[]`, `skills=[]`,
`strict_mcp_config=True`, `mcp_servers={}`. Without them the CLI loads the
machine's `~/.claude` settings, hooks and every installed skill description
into the prompt, which costs tokens and makes output depend on the machine.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from portolan.llm.base import LLMError, LLMJsonError, LLMResult, resolve_temperature
from portolan.llm.json_util import extract_and_validate
from portolan.metering import InMemorySink, Surface, UsageEvent, UsageSink
from portolan.pricing import UnknownModelError, estimate_cost, estimate_cost_with_cache

DEFAULT_MODEL = "claude-haiku-5-5"

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


def _usage_from_result(
    result_usage: dict[str, Any] | None, text: str, system: str, user: str
) -> dict[str, Any]:
    """Token counts from `ResultMessage.usage`, a plain dict of Anthropic
    usage keys in SDK 0.2.152. Falls back to a length-based estimate,
    flagged `estimated`, when the CLI reports no usage.

    Returns `input_tokens` (uncached), `cached_tokens` (cache reads) and
    `cache_creation_tokens` (cache writes) separately. Each call is a new
    session, so most input arrives as a cache write; the full input is the
    sum of all three (what `AgentSDKClient._call` records).
    """
    if result_usage:
        return {
            "input_tokens": result_usage.get("input_tokens", 0),
            "output_tokens": result_usage.get("output_tokens", 0),
            "cached_tokens": result_usage.get("cache_read_input_tokens", 0) or 0,
            "cache_creation_tokens": result_usage.get("cache_creation_input_tokens", 0) or 0,
            "estimated": False,
        }
    return {
        "input_tokens": max(1, len(system + user) // 4),
        "output_tokens": max(1, len(text) // 4),
        "cached_tokens": 0,
        "cache_creation_tokens": 0,
        "estimated": True,
    }


@dataclass
class _AgentSDKUsageHandle:
    """Local copy of `portolan.metering._MeteringHandle`, including
    `cost_override` (which `metered()` also honours), so `_call` can record
    the cache-write-aware cost from `estimate_cost_with_cache`.
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


class _MeteredAgentSDKContext:
    def __init__(self, sink: UsageSink, tenant: str, surface: Surface, model_key: str) -> None:
        self._sink = sink
        self.handle = _AgentSDKUsageHandle(
            tenant=tenant,
            surface=surface,
            model_key=model_key,
            request_id=f"req_{uuid.uuid4().hex}",
        )

    def __enter__(self) -> _AgentSDKUsageHandle:
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
        self._sink.record(event)
        return False  # never swallow the exception


class AgentSDKClient:
    """`LLMClient` implementation using `claude_agent_sdk.query()` in its
    single-shot form: one turn, no tools, isolated from local machine
    settings/skills/MCP config.
    """

    def __init__(
        self,
        *,
        default_model: str = DEFAULT_MODEL,
        sink: UsageSink | None = None,
        tenant: str = "dev",
        cli_path: str | None = None,
    ) -> None:
        self.default_model = default_model
        self.sink = sink or InMemorySink()
        self.tenant = tenant
        # The SDK's bundled CLI lags new models (0.2.152 bundles 2.1.259, which refuses
        # claude-opus-5-5); a newer installed `claude` can be named here.
        self.cli_path = cli_path

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
        # `ClaudeAgentOptions` (0.2.152) has no output-token cap and no
        # temperature field, so `max_tokens` is unused and `temperature` is only
        # recorded in usage.extra; calls run at the CLI's default temperature.
        model_key = model or self.default_model
        effective_temperature = resolve_temperature(temperature, surface)

        effective_system = system
        if json_schema is not None:
            effective_system = system + JSON_INSTRUCTION_TEMPLATE.format(
                schema=json.dumps(json_schema)
            )

        text, usage_info = self._call(
            effective_system, user, model_key, surface, effective_temperature
        )

        data = None
        if json_schema is not None:
            data, error = extract_and_validate(text, json_schema)
            if data is None:
                retry_user = RETRY_TEMPLATE.format(user=user, error=error)
                text, usage_info = self._call(
                    effective_system, retry_user, model_key, surface, effective_temperature
                )
                data, error = extract_and_validate(text, json_schema)
                if data is None:
                    raise LLMJsonError(
                        f"AgentSDKClient: model did not return schema-valid JSON after one "
                        f"retry (surface={surface!r}, model={model_key!r}): {error}"
                    )

        return LLMResult(
            text=text,
            data=data,
            input_tokens=usage_info["input_tokens"],
            output_tokens=usage_info["output_tokens"],
            cached_tokens=usage_info["cached_tokens"],
            model=model_key,
        )

    def _call(
        self,
        system: str,
        user: str,
        model_key: str,
        surface: Surface,
        temperature: float | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """One metered model call; a JSON retry is a second call and a second
        `UsageEvent`. `input_tokens` is the full input (uncached + cache
        read + cache write); `cached_tokens` is cache reads only.
        """
        with _MeteredAgentSDKContext(self.sink, self.tenant, surface, model_key) as usage:
            text, usage_dict = asyncio.run(self._aquery(system, user, model_key, temperature))
            cache_read = usage_dict["cached_tokens"]
            cache_creation = usage_dict["cache_creation_tokens"]
            usage.input_tokens = usage_dict["input_tokens"] + cache_read + cache_creation
            usage.output_tokens = usage_dict["output_tokens"]
            usage.cached_tokens = cache_read
            if cache_creation:
                usage.extra["cache_creation_input_tokens"] = cache_creation
            if temperature is not None:
                usage.extra["temperature_requested"] = temperature
            if usage_dict.get("estimated"):
                usage.extra["usage"] = "estimated"
            else:
                try:
                    usage.cost_override = estimate_cost_with_cache(
                        model_key,
                        usage_dict["input_tokens"],
                        usage_dict["output_tokens"],
                        cache_read_tokens=cache_read,
                        cache_creation_tokens=cache_creation,
                    )
                except UnknownModelError as pricing_exc:
                    usage.extra["pricing_error"] = str(pricing_exc)
        return text, {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cached_tokens": usage.cached_tokens,
        }

    @contextmanager
    def using_sink(self, sink: UsageSink, tenant: str) -> Iterator[None]:
        """Meter every `complete()` call inside this block into `sink`/`tenant`,
        restoring the originals on exit (see `portolan.llm.base`). Mutates the
        instance with no lock: give each worker thread its own client.
        """
        previous_sink, previous_tenant = self.sink, self.tenant
        self.sink, self.tenant = sink, tenant
        try:
            yield
        finally:
            self.sink, self.tenant = previous_sink, previous_tenant

    async def _aquery(
        self, system: str, user: str, model: str, temperature: float | None = None
    ) -> tuple[str, dict[str, Any]]:
        # Lazy import: only calling .complete() needs the SDK and the `claude` CLI.
        from claude_agent_sdk import (
            AssistantMessage,
            ClaudeAgentOptions,
            ResultMessage,
            TextBlock,
            query,
        )

        # `ClaudeAgentOptions` has no temperature field (see `complete()`).
        del temperature
        options = ClaudeAgentOptions(
            system_prompt=system,
            max_turns=1,
            tools=[],  # disables every built-in tool (Read/Write/Bash/...)
            model=model,
            setting_sources=[],  # don't load ~/.claude or project settings/hooks/CLAUDE.md
            skills=[],  # don't list any installed skill in the prompt
            strict_mcp_config=True,
            mcp_servers={},  # ...and don't load any locally-configured MCP server either
            cli_path=self.cli_path,
        )

        text_parts: list[str] = []
        result_text: str | None = None
        result_usage: dict[str, Any] | None = None
        is_error = False
        error_detail = ""

        async for message in query(prompt=user, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        text_parts.append(block.text)
            elif isinstance(message, ResultMessage):
                result_text = message.result
                result_usage = message.usage
                is_error = message.is_error
                if message.errors:
                    error_detail = "; ".join(message.errors)

        text = result_text if result_text is not None else "\n".join(text_parts)

        if is_error and not text:
            raise LLMError(
                "Claude Agent SDK query failed with no output"
                + (f": {error_detail}" if error_detail else "")
            )

        usage_dict = _usage_from_result(result_usage, text, system, user)
        return text, usage_dict
