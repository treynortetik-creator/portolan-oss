"""Extraction backends: FakeBackend for tests, and the two real Bedrock
backends for non-Anthropic and Anthropic models.

Both real backends are constructible without AWS credentials — the SDK
client object itself is only created the first time `extract()` runs, not
in `__init__` — and neither is ever exercised in tests (no network access
during tests, per the build instructions). What IS tested against these two
is the pure request-building helpers (`build_converse_request`,
`build_anthropic_bedrock_request`): given an envelope and a system prompt,
what dict would be sent, with no client and no network involved.

Model-id note (read before wiring up a real customer): `BedrockConverseBackend`
and `AnthropicBedrockBackend` take a `model_id` (the exact string the AWS/
Anthropic SDK call needs) separately from a `model_key` (the short slug
`portolan.pricing.PRICING` is keyed by, e.g. "nova-lite"). `model_key`
defaults to `infer_model_key(model_id)`, a small substring match against the
model ids this file already knows about; pass `model_key=` explicitly for
any model id that guess won't catch, so metering always resolves a real
price instead of raising UnknownModelError only when someone finally reads
the bill.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from typing import TYPE_CHECKING, Any, Protocol

from pydantic import ValidationError

from portolan.extract.prompt import build_user_message
from portolan.extract.schema import (
    EffectiveSchema,
    ExtractionResult,
    SourceEnvelope,
    extraction_result_json_schema,
)
from portolan.metering import UsageSink, metered

if TYPE_CHECKING:
    # Only for the type checker: importing portolan.llm for real here would
    # be circular, since portolan.llm.bedrock imports from this module (see
    # LLMClientBackend's docstring below). `from __future__ import
    # annotations` above already means annotations are never evaluated at
    # runtime, so this import never actually runs outside a type checker.
    from portolan.llm.base import LLMClient

TOOL_NAME = "emit_extraction"
TOOL_DESCRIPTION = "Emit the ExtractionResult JSON for this source. Call this tool exactly once."

DEFAULT_MAX_TOKENS = 4096


def _build_retry_message(user_message: str, error: ValidationError | str) -> str:
    """Group 3 of the review fix wave: "one strict attempt, on validation
    failure one retry with the errors appended." Every backend below that
    talks to a real model uses this to build that one retry's prompt."""
    return (
        f"{user_message}\n\n"
        "---\n"
        "Your previous response did not validate against the required JSON "
        "schema. Fix every error below and re-emit the COMPLETE "
        "ExtractionResult JSON (not a diff, not an explanation):\n"
        f"{error}"
    )


# --------------------------------------------------------------------------
# Bedrock model id references, verified 2026-09-12 (today, per the run this
# module was written in) directly against AWS's model-card pages via
# WebFetch, since the build instructions require citing a live source rather
# than guessing:
#
# - Nova Lite: base on-demand id "amazon.nova-lite-v1:0" works directly with
#   bedrock-runtime Converse, no cross-region inference profile required.
#   https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-amazon-nova-lite.html
#   (Programmatic Access table + the page's own Converse API sample code.)
# - GPT-5.6 Luna: on bedrock-runtime, the base id "openai.gpt-5.6-luna" is
#   NOT invokable on demand — AWS requires a cross-region inference profile
#   id. Use "us.openai.gpt-5.6-luna" (or "global.openai.gpt-5.6-luna").
#   https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-openai-gpt-56-luna.html
#   ("On bedrock-runtime, ... requests must name the geographic cross-Region
#   inference ID us.openai.gpt-5.6-luna as the model.") Converse IS
#   supported for this model on bedrock-runtime (see that page's API table).
NOVA_LITE_MODEL_ID = "amazon.nova-lite-v1:0"
GPT_5_6_LUNA_MODEL_ID = "us.openai.gpt-5.6-luna"

_MODEL_KEY_HINTS: list[tuple[str, str]] = [
    ("nova-micro", "nova-micro"),
    ("nova-lite", "nova-lite"),
    ("gpt-oss-120b", "gpt-oss-120b"),
    ("gpt-oss-20b", "gpt-oss-20b"),
    ("5.6-luna", "gpt-5.6-luna"),
    ("5-6-luna", "gpt-5.6-luna"),
    # Most specific first: "claude-sonnet-5" is a prefix of "claude-sonnet-5-5".
    ("claude-haiku-5-5", "claude-haiku-5-5"),
    ("claude-sonnet-5-5", "claude-sonnet-5-5"),
    ("claude-opus-5-5", "claude-opus-5-5"),
    ("claude-fable-5-2", "claude-fable-5-2"),
    ("claude-haiku-4-5", "claude-haiku-4-5"),
    ("claude-sonnet-5", "claude-sonnet-5"),
    ("claude-opus-5", "claude-opus-5"),
    ("titan-embed", "titan-embed-v2"),
]


def infer_model_key(model_id: str) -> str:
    """Best-effort map from a full Bedrock/Anthropic model id to the short
    pricing key in `portolan.pricing.PRICING`. Falls back to returning
    `model_id` unchanged, which will raise `UnknownModelError` at metering
    time if nobody overrides it with the real key — a loud failure when the
    bill is computed, not a silent wrong number."""
    lowered = model_id.lower()
    for hint, key in _MODEL_KEY_HINTS:
        if hint in lowered:
            return key
    return model_id


class Backend(Protocol):
    """Anything that can extract structured facts from one envelope."""

    def extract(
        self,
        envelope: SourceEnvelope,
        system_prompt: str,
        sink: UsageSink,
        tenant: str,
    ) -> ExtractionResult: ...


class FakeBackend:
    """Test double. `canned` is either a `{source_id: ExtractionResult}` map
    or a callable `envelope -> ExtractionResult`. Still records a metering
    event with fake-but-deterministic token counts on every call, so tests
    of the metering integration don't need a real backend."""

    def __init__(
        self,
        canned: dict[str, ExtractionResult] | Callable[[SourceEnvelope], ExtractionResult],
        model_key: str = "nova-lite",
    ) -> None:
        self._canned = canned
        self.model_key = model_key

    def _lookup(self, envelope: SourceEnvelope) -> ExtractionResult:
        if callable(self._canned):
            return self._canned(envelope)
        try:
            return self._canned[envelope.source_id]
        except KeyError as exc:
            raise KeyError(
                f"FakeBackend has no canned ExtractionResult for source_id={envelope.source_id!r}"
            ) from exc

    def extract(
        self,
        envelope: SourceEnvelope,
        system_prompt: str,
        sink: UsageSink,
        tenant: str,
    ) -> ExtractionResult:
        with metered(sink, tenant, "extract", self.model_key) as usage:
            result = self._lookup(envelope)
            # Fake but deterministic: stands in for a real usage block so
            # tests can assert metering fired without needing one.
            usage.input_tokens = max(1, len(envelope.text) // 4) + max(1, len(system_prompt) // 4)
            usage.output_tokens = max(1, len(result.model_dump_json()) // 4)
            usage.extra["backend"] = "fake"
        return result


# --------------------------------------------------------------------------
# BedrockConverseBackend — Nova, gpt-oss, GPT-5.6 Luna, or any other Bedrock
# model reachable through bedrock-runtime's Converse API.
# --------------------------------------------------------------------------


def build_converse_request(
    envelope: SourceEnvelope,
    system_prompt: str,
    tool_name: str = TOOL_NAME,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> dict[str, Any]:
    """Pure function: envelope + system prompt -> the keyword arguments for
    `boto3_client.converse(modelId=..., **this)`. No network, no client,
    testable on its own."""
    return {
        "messages": [
            {
                "role": "user",
                "content": [{"text": build_user_message(envelope)}],
            }
        ],
        "system": [{"text": system_prompt}],
        "toolConfig": {
            "tools": [
                {
                    "toolSpec": {
                        "name": tool_name,
                        "description": TOOL_DESCRIPTION,
                        "inputSchema": {"json": extraction_result_json_schema()},
                    }
                }
            ],
            "toolChoice": {"tool": {"name": tool_name}},
        },
        "inferenceConfig": {"maxTokens": max_tokens},
    }


def _extract_converse_tool_input(response: dict[str, Any], tool_name: str) -> dict[str, Any]:
    message = response.get("output", {}).get("message", {})
    for block in message.get("content", []):
        tool_use = block.get("toolUse")
        if tool_use and tool_use.get("name") == tool_name:
            return tool_use.get("input", {})
    raise ValueError(f"Bedrock Converse response had no toolUse block for tool {tool_name!r}")


class BedrockConverseBackend:
    """Non-Anthropic Bedrock models (Nova, gpt-oss, GPT-5.6 Luna, ...) via
    boto3 `bedrock-runtime` `converse`, with a single forced tool whose
    input schema is the ExtractionResult JSON schema — this is how a
    Converse-API model is made to emit schema-shaped JSON; there is no
    separate "structured output" mode on this API.
    """

    def __init__(
        self,
        model_id: str,
        region: str,
        *,
        model_key: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        # Lazy import: constructing this backend must not require boto3 to
        # be usable (credentials, network) — only importable. The client
        # itself is created later, in _get_client(), on first extract().
        import boto3

        self._boto3 = boto3
        self.model_id = model_id
        self.region = region
        self.model_key = model_key or infer_model_key(model_id)
        self.max_tokens = max_tokens
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            self._client = self._boto3.client("bedrock-runtime", region_name=self.region)
        return self._client

    def extract(
        self,
        envelope: SourceEnvelope,
        system_prompt: str,
        sink: UsageSink,
        tenant: str,
    ) -> ExtractionResult:
        # Group 3 of the review fix wave: one strict attempt; on validation
        # failure, one retry with the errors appended; only if THAT still
        # doesn't validate strictly does `parse_lenient` salvage what it
        # can, dropping individual bad facts/actions rather than losing
        # the whole source. Each real model call is its own `metered()`
        # event -- `usage.extra` is only ever set before its own `with`
        # block exits, so it lands on the event for the call it describes.
        client = self._get_client()
        request = build_converse_request(envelope, system_prompt, max_tokens=self.max_tokens)

        with metered(sink, tenant, "extract", self.model_key) as usage:
            response = client.converse(modelId=self.model_id, **request)
            usage.request_id = response.get("ResponseMetadata", {}).get("RequestId", "")
            usage_block = response.get("usage", {})
            usage.input_tokens = usage_block.get("inputTokens", 0)
            usage.output_tokens = usage_block.get("outputTokens", 0)
            data = _extract_converse_tool_input(response, TOOL_NAME)
            try:
                result: ExtractionResult | None = ExtractionResult.model_validate(data)
            except ValidationError as exc:
                result = None
                usage.extra["validation_error"] = str(exc)
        if result is not None:
            return result

        retry_request = build_converse_request(envelope, system_prompt, max_tokens=self.max_tokens)
        retry_request["messages"][0]["content"][0]["text"] = _build_retry_message(
            request["messages"][0]["content"][0]["text"], usage.extra["validation_error"]
        )

        with metered(sink, tenant, "extract", self.model_key) as usage:
            response = client.converse(modelId=self.model_id, **retry_request)
            usage.request_id = response.get("ResponseMetadata", {}).get("RequestId", "")
            usage_block = response.get("usage", {})
            usage.input_tokens = usage_block.get("inputTokens", 0)
            usage.output_tokens = usage_block.get("outputTokens", 0)
            data = _extract_converse_tool_input(response, TOOL_NAME)
            try:
                result = ExtractionResult.model_validate(data)
            except ValidationError:
                result, rejected = ExtractionResult.parse_lenient(data)
                usage.extra["rejected_facts"] = [r.model_dump(mode="json") for r in rejected]
        return result


# --------------------------------------------------------------------------
# AnthropicBedrockBackend — Claude models via the Anthropic SDK's Bedrock
# Mantle client.
# --------------------------------------------------------------------------


def build_anthropic_bedrock_request(
    envelope: SourceEnvelope,
    system_prompt: str,
    tool_name: str = TOOL_NAME,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    force_tool: bool = True,
) -> dict[str, Any]:
    """Pure function: envelope + system prompt -> the keyword arguments for
    `client.messages.create(model=..., **this)` on `AnthropicBedrockMantle`.

    `force_tool=False` for a model that rejects a forced `tool_choice`
    (`portolan.llm.base.claude_takes_forced_tool_choice`): the tool stays
    strict, the choice is `auto`, and the system prompt says to call it.

    Uses strict tool use with a forced `tool_choice`, not
    `output_config={"format": ...}`. That is a deliberate deviation from a
    first-party-API pattern: fetching the live "Claude in Amazon Bedrock"
    doc (platform.claude.com/docs/en/build-with-claude/claude-in-amazon-bedrock,
    checked 2026-09-12) lists "Structured outputs" under Features NOT
    supported on the Mantle client, while "Tool use" is listed as supported.
    Forcing a single strict tool is the Bedrock-compatible way to get
    schema-shaped JSON out of this client.
    """
    return {
        "max_tokens": max_tokens,
        "system": system_prompt
        if force_tool
        else system_prompt + _call_tool_instruction(tool_name),
        "messages": [{"role": "user", "content": build_user_message(envelope)}],
        "tools": [
            {
                "name": tool_name,
                "description": TOOL_DESCRIPTION,
                "input_schema": extraction_result_json_schema(),
                "strict": True,
            }
        ],
        "tool_choice": ({"type": "tool", "name": tool_name} if force_tool else {"type": "auto"}),
    }


def _call_tool_instruction(tool_name: str) -> str:
    return (
        f"\n\nAnswer by calling the `{tool_name}` tool exactly once. Write no text "
        "outside the tool call."
    )


def _extract_anthropic_tool_input(response: Any, tool_name: str) -> dict[str, Any]:
    for block in response.content:
        if getattr(block, "type", None) == "tool_use" and getattr(block, "name", None) == tool_name:
            return block.input
    raise ValueError(f"Anthropic Bedrock response had no tool_use block for tool {tool_name!r}")


class LLMClientBackend:
    """Extraction backend that runs through a `portolan.llm.LLMClient`
    rather than talking to boto3/anthropic directly (build-contracts.md
    "LLM access": "`portolan/extract/backends.py` gains an
    `LLMClientBackend(client)` so extraction runs through the same
    client."). This is how extraction picks up whichever provider
    `portolan.llm.get_llm()` resolved to (`AgentSDKClient`,
    `BedrockClient`, or a test's `FakeLLMClient`) without this module
    needing to know which one.

    `model` (review finding F17) is passed straight through to every
    `client.complete()` call. Without it, `LLMClientBackend` could not
    select a model at all -- `settings.extraction_model` was whatever the
    client happened to be constructed with -- and `bench/__main__.py`
    worked around the missing parameter with a bare `except TypeError`
    retry that would just as happily have swallowed a genuine constructor
    bug. That workaround is gone now that this takes `model` directly.

    `surface` (polish wave 2026-09-12) is likewise a constructor argument,
    defaulting to `"extract"`: what used to require a whole subclass
    (`portolan.audit.second_pass.AuditBackend`, differing from this class
    by one string, `surface="audit"`) is now `LLMClientBackend(client,
    surface="audit")`. `AuditBackend` is kept as a one-line alias rather
    than deleted, since `portolan.audit.__main__`/`__init__` (outside this
    fix's owned paths) still import it by name.

    Metering happens inside `client.complete()` itself, against whichever
    sink/tenant the client was constructed with (`portolan.llm.get_llm()`,
    typically a durable one -- Postgres in production) -- UNLESS the
    client provides `using_sink()` (`portolan.llm.base`'s optional
    extension; see its module docstring), in which case `extract()` below
    wraps its whole call (initial attempt plus the one validation retry)
    in `client.using_sink(sink, tenant)`, so the `sink`/`tenant` THIS
    backend received (to satisfy the `Backend` protocol) really is where
    the call's usage lands. Fix wave 2026-09-12 (bench telemetry bug):
    `portolan.bench.extraction.run_extraction` always builds its own
    `InMemorySink` and passes it here on every `extract()` call, precisely
    so `score_extraction` can read real tokens/cost back off it afterward
    -- before `using_sink` existed, that sink was silently ignored (`del
    sink, tenant` below) and every bench report against an
    `LLMClientBackend`-based backend showed 0 tokens / $0.00 cost even
    though the calls happened and were, in fact, metered -- just into
    whatever sink `get_llm()` built the client with instead. A client
    with no `using_sink` (every `LLMClient` implementation besides
    `OpenRouterClient`, as of this wave) keeps exactly the old behavior:
    `sink`/`tenant` are accepted (to satisfy `Backend`) but not forwarded.

    One remaining consequence for such a client (Group 3 of the review fix
    wave, noted rather than worked around: `portolan.llm` is a different
    wave's module, not touched by this fix): `usage.extra["rejected_
    facts"]` can be set on a `metered()` event only by the code holding
    that event's handle, which is `client.complete()` itself, not this
    backend -- so a rejected fact from `parse_lenient` below is visible on
    `result.rejected`, but not inside the metering event's `extra` for
    this backend specifically (unlike `BedrockConverseBackend` and
    `AnthropicBedrockBackend`, which own their `metered()` block directly
    and do set it there).

    Leniency lives in one place (fix wave 2026-09-12, the 18 Nova Lite
    `S-call-*` bench failures in `var/bench/compare3/amazon_nova-lite-v1/`
    -- every one an `entities` item shaped like a top-level `EntityMention`
    object where `Fact.entities` wants a plain string): `_complete()`
    below passes `validate_json=False` to any client whose `complete()`
    accepts it (`portolan.llm.openrouter.OpenRouterClient`, checked via
    `inspect.signature` rather than assumed, since `FakeLLMClient` and
    every other `LLMClient` implementation don't have this parameter and
    a bare kwarg would raise `TypeError` against them). That turns off
    OpenRouterClient's own local jsonschema gate in its manual-instruction
    fallback path, which used to reject a shape mismatch (an entity
    object instead of a string, `null` for an integer) before this
    backend's `data` ever arrived -- losing the whole source to
    `LLMJsonError` one layer before `parse_lenient` ever got a chance to
    coerce it. With the gate off, this backend's own strict-attempt/retry/
    `parse_lenient` chain (right below) is the only place a lenient
    decision gets made, exactly as it already was for `FakeLLMClient` and
    every other client, which never had a local gate to begin with.
    """

    def __init__(
        self,
        client: LLMClient,
        model: str | None = None,
        surface: str = "extract",
        max_tokens: int | None = None,
        effective_schema: EffectiveSchema | None = None,
    ) -> None:
        self.client = client
        self.model = model
        # The customer extension merged into the base schema
        # (`merge_customer_extension`). Without it a fact typed with one of
        # the customer's own `extra_types` (which the prompt advertises)
        # fails validation, costs a retry and is then dropped.
        self.effective_schema = effective_schema
        self.surface = surface
        # None keeps the client's own default; a reasoning model at high effort spends
        # reasoning tokens from the same cap, so it needs room for the answer too.
        self.max_tokens = max_tokens

    def _sink_scope(self, sink: UsageSink, tenant: str) -> AbstractContextManager[None]:
        """`client.using_sink(sink, tenant)` when the client provides it
        (see class docstring), else a no-op context -- today's unchanged
        behavior for any `LLMClient` without that extension."""
        using_sink = getattr(self.client, "using_sink", None)
        if using_sink is None:
            return nullcontext()
        return using_sink(sink, tenant)

    def extract(
        self,
        envelope: SourceEnvelope,
        system_prompt: str,
        sink: UsageSink,
        tenant: str,
    ) -> ExtractionResult:
        schema = extraction_result_json_schema()
        user_message = build_user_message(envelope)

        with self._sink_scope(sink, tenant):
            result = self._complete(system_prompt, user_message, schema)
            data = result.data
            if data is None:
                raise ValueError(
                    f"LLMClientBackend: no JSON data returned for source_id={envelope.source_id!r}"
                )

            # One strict attempt. When it fails, salvage first: a reply
            # whose only faults are ones `parse_lenient` can correct without
            # losing a fact (a null sentiment, a confidence word, an extra
            # key) is accepted as is, with no second paid call. Only a reply
            # that loses a fact or action gets the one retry, with the
            # errors appended; then whichever attempt kept more is used.
            context = (
                {"effective_schema": self.effective_schema}
                if self.effective_schema is not None
                else None
            )
            try:
                return ExtractionResult.model_validate(data, context=context)
            except ValidationError as exc:
                first_error: ValidationError = exc
            first = self._salvage(data, envelope)
            if first is not None and not first.rejected:
                self._meter_salvage(sink, tenant, result.model, first)
                return first

            retry_message = _build_retry_message(user_message, first_error)
            retry_result = self._complete(system_prompt, retry_message, schema)
            retry_data = retry_result.data
            second: ExtractionResult | None = None
            if retry_data is not None:
                try:
                    return ExtractionResult.model_validate(retry_data, context=context)
                except ValidationError:
                    second = self._salvage(retry_data, envelope)

            candidates = [c for c in (second, first) if c is not None]
            if not candidates:
                raise first_error
            # The retry wins a tie: it was written with the errors in view.
            best = max(candidates, key=lambda c: len(c.facts) + len(c.actions))
            self._meter_salvage(sink, tenant, retry_result.model, best)
            return best

    def _salvage(self, data: dict[str, Any], envelope: SourceEnvelope) -> ExtractionResult | None:
        """`parse_lenient` on one reply; `None` when even that cannot build
        a result. `default_source_id` fills a missing top-level `source_id`
        from the envelope, the one place that id is known."""
        try:
            salvaged, _rejected = ExtractionResult.parse_lenient(
                data,
                effective_schema=self.effective_schema,
                default_source_id=envelope.source_id,
            )
        except ValidationError:
            return None
        return salvaged

    def _meter_salvage(
        self, sink: UsageSink, tenant: str, model: str, salvaged: ExtractionResult
    ) -> None:
        """A zero-token metering event carrying the coercion count and the
        rejected facts (fact rule 4: "a rejected fact is counted"). The
        model calls themselves are already metered by the client."""
        if not (salvaged.coercions or salvaged.rejected):
            return
        with metered(sink, tenant, self.surface, model) as usage:
            usage.extra["coercions"] = len(salvaged.coercions)
            usage.extra["rejected_facts"] = [r.model_dump(mode="json") for r in salvaged.rejected]

    def _complete(self, system_prompt: str, user_message: str, schema: dict[str, Any]):
        kwargs: dict[str, Any] = {
            "system": system_prompt,
            "user": user_message,
            "json_schema": schema,
            "model": self.model,
            "surface": self.surface,
        }
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        # Opt-in only, and only for a client that actually declares the
        # parameter (see class docstring) -- FakeLLMClient and every other
        # LLMClient implementation as of this wave have no such parameter,
        # and calling complete(validate_json=...) against one would raise
        # TypeError rather than silently ignore it.
        complete_params = inspect.signature(self.client.complete).parameters
        if "validate_json" in complete_params:
            kwargs["validate_json"] = False
        return self.client.complete(**kwargs)


class AnthropicBedrockBackend:
    """Claude models via `AnthropicBedrockMantle`
    (platform.claude.com/docs/en/build-with-claude/claude-in-amazon-bedrock,
    Python tab: `from anthropic import AnthropicBedrockMantle`;
    `AnthropicBedrockMantle(aws_region=...)`), reached through
    `pip install "anthropic[bedrock]"`. Model ids on this client carry the
    `anthropic.` provider prefix, e.g. `anthropic.claude-haiku-4-5`.
    """

    def __init__(
        self,
        model_id: str,
        region: str,
        *,
        model_key: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        # Lazy import: constructing this backend must not require the
        # anthropic[bedrock] extra to be configured with credentials, only
        # importable. The client is created later, in _get_client().
        from anthropic import AnthropicBedrockMantle

        self._client_cls = AnthropicBedrockMantle
        self.model_id = model_id
        self.region = region
        self.model_key = model_key or infer_model_key(model_id)
        self.max_tokens = max_tokens
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            self._client = self._client_cls(aws_region=self.region)
        return self._client

    def extract(
        self,
        envelope: SourceEnvelope,
        system_prompt: str,
        sink: UsageSink,
        tenant: str,
    ) -> ExtractionResult:
        # See BedrockConverseBackend.extract for the retry/lenient-parse
        # shape (Group 3 of the review fix wave) -- identical here, just
        # over the Anthropic Bedrock Mantle client's request/response shape.
        client = self._get_client()
        # Imported here: `portolan.llm` imports this module (see `LLMClientBackend`).
        from portolan.llm.base import claude_takes_forced_tool_choice

        force_tool = claude_takes_forced_tool_choice(self.model_id)
        request = build_anthropic_bedrock_request(
            envelope, system_prompt, max_tokens=self.max_tokens, force_tool=force_tool
        )

        with metered(sink, tenant, "extract", self.model_key) as usage:
            response = client.messages.create(model=self.model_id, **request)
            usage.request_id = getattr(response, "_request_id", "") or ""
            response_usage = response.usage
            usage.input_tokens = getattr(response_usage, "input_tokens", 0)
            usage.output_tokens = getattr(response_usage, "output_tokens", 0)
            usage.cached_tokens = getattr(response_usage, "cache_read_input_tokens", 0) or 0
            data = _extract_anthropic_tool_input(response, TOOL_NAME)
            try:
                result: ExtractionResult | None = ExtractionResult.model_validate(data)
            except ValidationError as exc:
                result = None
                usage.extra["validation_error"] = str(exc)
        if result is not None:
            return result

        retry_request = build_anthropic_bedrock_request(
            envelope, system_prompt, max_tokens=self.max_tokens, force_tool=force_tool
        )
        retry_request["messages"][0]["content"] = _build_retry_message(
            request["messages"][0]["content"], usage.extra["validation_error"]
        )

        with metered(sink, tenant, "extract", self.model_key) as usage:
            response = client.messages.create(model=self.model_id, **retry_request)
            usage.request_id = getattr(response, "_request_id", "") or ""
            response_usage = response.usage
            usage.input_tokens = getattr(response_usage, "input_tokens", 0)
            usage.output_tokens = getattr(response_usage, "output_tokens", 0)
            usage.cached_tokens = getattr(response_usage, "cache_read_input_tokens", 0) or 0
            data = _extract_anthropic_tool_input(response, TOOL_NAME)
            try:
                result = ExtractionResult.model_validate(data)
            except ValidationError:
                result, rejected = ExtractionResult.parse_lenient(data)
                usage.extra["rejected_facts"] = [r.model_dump(mode="json") for r in rejected]
        return result
