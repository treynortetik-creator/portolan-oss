"""`LLMClient` backed by AWS Bedrock (portolan-spec-v2.md section 11). The
same two request shapes `portolan.extract.backends` builds for extraction
(forced Converse tool for non-Anthropic models, forced strict tool on the
Anthropic Bedrock Mantle client for Claude), generalized to any
`json_schema`, or a plain text completion when no schema is given.

Neither SDK client is constructed until `.complete()` first runs; tests
cover request building and construction only, never the network.
"""

from __future__ import annotations

import json
from typing import Any

from portolan.extract.backends import (
    DEFAULT_MAX_TOKENS,
    _extract_anthropic_tool_input,
    _extract_converse_tool_input,
    infer_model_key,
)
from portolan.llm.base import LLMResult, resolve_temperature
from portolan.metering import InMemorySink, Surface, UsageSink, metered

GENERIC_TOOL_NAME = "emit_json"
GENERIC_TOOL_DESCRIPTION = "Emit the JSON object for this response. Call this tool exactly once."


def is_anthropic_model_id(model_id: str) -> bool:
    """True for a Bedrock model id that should go through the Anthropic
    Bedrock Mantle client rather than plain Converse (any id containing
    "claude")."""
    return "claude" in model_id.lower()


def build_converse_json_request(
    user_message: str,
    system_prompt: str,
    json_schema: dict[str, Any],
    *,
    tool_name: str = GENERIC_TOOL_NAME,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float | None = None,
) -> dict[str, Any]:
    """Same shape as `portolan.extract.backends.build_converse_request`, which
    hardcodes the `ExtractionResult` schema, for an arbitrary schema.
    `temperature` goes in `inferenceConfig`; omitted when `None`.
    """
    inference_config: dict[str, Any] = {"maxTokens": max_tokens}
    if temperature is not None:
        inference_config["temperature"] = temperature
    return {
        "messages": [{"role": "user", "content": [{"text": user_message}]}],
        "system": [{"text": system_prompt}],
        "toolConfig": {
            "tools": [
                {
                    "toolSpec": {
                        "name": tool_name,
                        "description": GENERIC_TOOL_DESCRIPTION,
                        "inputSchema": {"json": json_schema},
                    }
                }
            ],
            "toolChoice": {"tool": {"name": tool_name}},
        },
        "inferenceConfig": inference_config,
    }


def build_converse_text_request(
    user_message: str,
    system_prompt: str,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float | None = None,
) -> dict[str, Any]:
    """Plain (no forced tool) Converse request, used when no `json_schema` is
    given. `temperature` as in `build_converse_json_request`."""
    inference_config: dict[str, Any] = {"maxTokens": max_tokens}
    if temperature is not None:
        inference_config["temperature"] = temperature
    return {
        "messages": [{"role": "user", "content": [{"text": user_message}]}],
        "system": [{"text": system_prompt}],
        "inferenceConfig": inference_config,
    }


def build_anthropic_json_request(
    user_message: str,
    system_prompt: str,
    json_schema: dict[str, Any],
    *,
    tool_name: str = GENERIC_TOOL_NAME,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float | None = None,
) -> dict[str, Any]:
    """Same shape as `portolan.extract.backends.build_anthropic_bedrock_request`
    for an arbitrary schema; that function's docstring says why a strict
    forced tool, not `output_config`, gets JSON on Bedrock. `temperature` is
    a top-level field in this (Messages API) shape; omitted when `None`.
    """
    request: dict[str, Any] = {
        "max_tokens": max_tokens,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_message}],
        "tools": [
            {
                "name": tool_name,
                "description": GENERIC_TOOL_DESCRIPTION,
                "input_schema": json_schema,
                "strict": True,
            }
        ],
        "tool_choice": {"type": "tool", "name": tool_name},
    }
    if temperature is not None:
        request["temperature"] = temperature
    return request


def build_anthropic_text_request(
    user_message: str,
    system_prompt: str,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float | None = None,
) -> dict[str, Any]:
    """Plain (no forced tool) Anthropic Bedrock request. `temperature` as in
    `build_anthropic_json_request`."""
    request: dict[str, Any] = {
        "max_tokens": max_tokens,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_message}],
    }
    if temperature is not None:
        request["temperature"] = temperature
    return request


class BedrockClient:
    """`LLMClient` implementation for AWS Bedrock. `model_id` is the exact
    Bedrock/Anthropic model id (e.g. `"amazon.nova-lite-v1:0"`,
    `"anthropic.claude-haiku-4-5"`); `model_key` defaults to
    `infer_model_key(model_id)` for metering.
    """

    def __init__(
        self,
        model_id: str,
        region: str,
        *,
        model_key: str | None = None,
        sink: UsageSink | None = None,
        tenant: str = "dev",
    ) -> None:
        self.model_id = model_id
        self.region = region
        self.model_key = model_key or infer_model_key(model_id)
        self.sink = sink or InMemorySink()
        self.tenant = tenant
        self._boto3_client: Any = None
        self._anthropic_client: Any = None

    def _get_boto3_client(self) -> Any:
        if self._boto3_client is None:
            import boto3

            self._boto3_client = boto3.client("bedrock-runtime", region_name=self.region)
        return self._boto3_client

    def _get_anthropic_client(self) -> Any:
        if self._anthropic_client is None:
            from anthropic import AnthropicBedrockMantle

            self._anthropic_client = AnthropicBedrockMantle(aws_region=self.region)
        return self._anthropic_client

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
        model_id = model or self.model_id
        model_key = infer_model_key(model_id) if model else self.model_key
        effective_temperature = resolve_temperature(temperature, surface)

        with metered(self.sink, self.tenant, surface, model_key) as usage:
            if is_anthropic_model_id(model_id):
                text, data, usage_dict = self._complete_anthropic(
                    model_id, system, user, json_schema, max_tokens, effective_temperature
                )
            else:
                text, data, usage_dict = self._complete_converse(
                    model_id, system, user, json_schema, max_tokens, effective_temperature
                )
            usage.input_tokens = usage_dict.get("input_tokens", 0)
            usage.output_tokens = usage_dict.get("output_tokens", 0)
            usage.cached_tokens = usage_dict.get("cached_tokens", 0)
            usage.request_id = usage_dict.get("request_id", "")

        return LLMResult(
            text=text,
            data=data,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cached_tokens=usage.cached_tokens,
            model=model_key,
        )

    def _complete_converse(
        self,
        model_id: str,
        system: str,
        user: str,
        json_schema: dict[str, Any] | None,
        max_tokens: int,
        temperature: float | None = None,
    ) -> tuple[str, dict[str, Any] | None, dict[str, Any]]:
        client = self._get_boto3_client()
        if json_schema is not None:
            request = build_converse_json_request(
                user, system, json_schema, max_tokens=max_tokens, temperature=temperature
            )
        else:
            request = build_converse_text_request(
                user, system, max_tokens=max_tokens, temperature=temperature
            )

        response = client.converse(modelId=model_id, **request)
        usage_block = response.get("usage", {})
        usage_dict = {
            "input_tokens": usage_block.get("inputTokens", 0),
            "output_tokens": usage_block.get("outputTokens", 0),
            # This path reads no cache-read count from Converse's usage block.
            "cached_tokens": 0,
            "request_id": response.get("ResponseMetadata", {}).get("RequestId", ""),
        }

        if json_schema is not None:
            data = _extract_converse_tool_input(response, GENERIC_TOOL_NAME)
            text = json.dumps(data)
        else:
            data = None
            message = response.get("output", {}).get("message", {})
            text = "".join(block.get("text", "") for block in message.get("content", []))
        return text, data, usage_dict

    def _complete_anthropic(
        self,
        model_id: str,
        system: str,
        user: str,
        json_schema: dict[str, Any] | None,
        max_tokens: int,
        temperature: float | None = None,
    ) -> tuple[str, dict[str, Any] | None, dict[str, Any]]:
        client = self._get_anthropic_client()
        if json_schema is not None:
            request = build_anthropic_json_request(
                user, system, json_schema, max_tokens=max_tokens, temperature=temperature
            )
        else:
            request = build_anthropic_text_request(
                user, system, max_tokens=max_tokens, temperature=temperature
            )

        response = client.messages.create(model=model_id, **request)
        response_usage = response.usage
        usage_dict = {
            "input_tokens": getattr(response_usage, "input_tokens", 0),
            "output_tokens": getattr(response_usage, "output_tokens", 0),
            "cached_tokens": getattr(response_usage, "cache_read_input_tokens", 0) or 0,
            "request_id": getattr(response, "_request_id", "") or "",
        }

        if json_schema is not None:
            data = _extract_anthropic_tool_input(response, GENERIC_TOOL_NAME)
            text = json.dumps(data)
        else:
            data = None
            text = "".join(
                block.text for block in response.content if getattr(block, "type", None) == "text"
            )
        return text, data, usage_dict
