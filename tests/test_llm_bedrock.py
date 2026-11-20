"""Tests for portolan.llm.bedrock.BedrockClient.

Same discipline as tests/test_backends.py: only the pure request-building
helpers and lazy construction are exercised. `.complete()` (which would
create a boto3/anthropic client and hit the network) is never called.
"""

from __future__ import annotations

from portolan.llm.bedrock import (
    GENERIC_TOOL_NAME,
    BedrockClient,
    build_anthropic_json_request,
    build_anthropic_text_request,
    build_converse_json_request,
    build_converse_text_request,
    is_anthropic_model_id,
)

SCHEMA = {
    "type": "object",
    "properties": {"greeting": {"type": "string"}},
    "required": ["greeting"],
    "additionalProperties": False,
}


def test_is_anthropic_model_id() -> None:
    assert is_anthropic_model_id("anthropic.claude-haiku-4-5") is True
    assert is_anthropic_model_id("anthropic.claude-sonnet-5") is True
    assert is_anthropic_model_id("amazon.nova-lite-v1:0") is False
    assert is_anthropic_model_id("us.openai.gpt-5.6-luna") is False


def test_build_converse_json_request_shape() -> None:
    request = build_converse_json_request("hello", "system prompt", SCHEMA)
    assert request["system"] == [{"text": "system prompt"}]
    assert request["messages"] == [{"role": "user", "content": [{"text": "hello"}]}]
    tool_spec = request["toolConfig"]["tools"][0]["toolSpec"]
    assert tool_spec["name"] == GENERIC_TOOL_NAME
    assert tool_spec["inputSchema"]["json"] == SCHEMA
    assert request["toolConfig"]["toolChoice"] == {"tool": {"name": GENERIC_TOOL_NAME}}
    assert request["inferenceConfig"]["maxTokens"] > 0


def test_build_converse_text_request_shape_has_no_tool_config() -> None:
    request = build_converse_text_request("hello", "system prompt")
    assert request["system"] == [{"text": "system prompt"}]
    assert request["messages"] == [{"role": "user", "content": [{"text": "hello"}]}]
    assert "toolConfig" not in request


# --------------------------------------------------------------------------
# temperature (infra-2b wave, 2026-09-18): the Converse API's own
# `inferenceConfig.temperature` field, right next to `maxTokens`.
# --------------------------------------------------------------------------


def test_build_converse_json_request_carries_temperature() -> None:
    request = build_converse_json_request("hello", "system prompt", SCHEMA, temperature=0.0)
    assert request["inferenceConfig"]["temperature"] == 0.0
    assert request["inferenceConfig"]["maxTokens"] > 0


def test_build_converse_json_request_omits_temperature_when_none() -> None:
    request = build_converse_json_request("hello", "system prompt", SCHEMA)
    assert "temperature" not in request["inferenceConfig"]


def test_build_converse_text_request_carries_temperature() -> None:
    request = build_converse_text_request("hello", "system prompt", temperature=0.4)
    assert request["inferenceConfig"]["temperature"] == 0.4


def test_build_converse_text_request_omits_temperature_when_none() -> None:
    request = build_converse_text_request("hello", "system prompt")
    assert "temperature" not in request["inferenceConfig"]


def test_build_anthropic_json_request_shape() -> None:
    request = build_anthropic_json_request("hello", "system prompt", SCHEMA)
    assert request["system"] == "system prompt"
    assert request["messages"] == [{"role": "user", "content": "hello"}]
    assert request["tool_choice"] == {"type": "tool", "name": GENERIC_TOOL_NAME}
    tool = request["tools"][0]
    assert tool["name"] == GENERIC_TOOL_NAME
    assert tool["strict"] is True
    assert tool["input_schema"] == SCHEMA
    assert "model" not in request


def test_build_anthropic_text_request_shape_has_no_tools() -> None:
    request = build_anthropic_text_request("hello", "system prompt")
    assert request["system"] == "system prompt"
    assert request["messages"] == [{"role": "user", "content": "hello"}]
    assert "tools" not in request
    assert "tool_choice" not in request


# --------------------------------------------------------------------------
# temperature (infra-2b wave, 2026-09-18): the Anthropic Messages/Mantle
# shape wants a TOP-LEVEL `temperature`, not nested under
# `inferenceConfig` the way Bedrock's plain Converse API does (see the
# Converse tests above) -- confirming the two builder families use the
# right shape for their own API.
# --------------------------------------------------------------------------


def test_build_anthropic_json_request_carries_temperature() -> None:
    request = build_anthropic_json_request("hello", "system prompt", SCHEMA, temperature=0.0)
    assert request["temperature"] == 0.0
    assert "inferenceConfig" not in request


def test_build_anthropic_json_request_omits_temperature_when_none() -> None:
    request = build_anthropic_json_request("hello", "system prompt", SCHEMA)
    assert "temperature" not in request


def test_build_anthropic_text_request_carries_temperature() -> None:
    request = build_anthropic_text_request("hello", "system prompt", temperature=0.4)
    assert request["temperature"] == 0.4


def test_build_anthropic_text_request_omits_temperature_when_none() -> None:
    request = build_anthropic_text_request("hello", "system prompt")
    assert "temperature" not in request


def test_bedrock_client_constructs_without_credentials_non_anthropic() -> None:
    client = BedrockClient("amazon.nova-lite-v1:0", region="us-east-1")
    assert client.model_id == "amazon.nova-lite-v1:0"
    assert client.model_key == "nova-lite"
    assert client._boto3_client is None
    assert client._anthropic_client is None


def test_bedrock_client_constructs_without_credentials_anthropic() -> None:
    client = BedrockClient("anthropic.claude-haiku-4-5", region="us-east-1")
    assert client.model_id == "anthropic.claude-haiku-4-5"
    assert client.model_key == "claude-haiku-4-5"


def test_bedrock_client_accepts_explicit_model_key() -> None:
    client = BedrockClient("some-future-model-id", region="us-east-1", model_key="gpt-oss-120b")
    assert client.model_key == "gpt-oss-120b"


def test_a_model_that_rejects_forced_tool_choice_gets_auto_and_an_instruction() -> None:
    """Sonnet 5.5, Opus 5.5 and Fable 5.2 answer `tool_choice: {"type":
    "tool"}` with a 400: the tool stays strict and the prompt asks for it."""
    request = build_anthropic_json_request("u", "s", {"type": "object"}, force_tool=False)
    assert request["tool_choice"] == {"type": "auto"}
    assert request["tools"][0]["strict"] is True
    assert GENERIC_TOOL_NAME in request["system"] and request["system"].startswith("s")


def test_infer_model_key_tells_current_models_from_their_predecessors() -> None:
    from portolan.extract.backends import infer_model_key
    from portolan.pricing import PRICING

    assert infer_model_key("anthropic.claude-sonnet-5-5") == "claude-sonnet-5-5"
    assert infer_model_key("anthropic.claude-sonnet-5") == "claude-sonnet-5"
    assert infer_model_key("anthropic.claude-haiku-5-5") == "claude-haiku-5-5"
    for key in ("claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5", "claude-fable-5-2"):
        assert key in PRICING
    assert PRICING["claude-haiku-5-5"].input_per_million == 0.10
