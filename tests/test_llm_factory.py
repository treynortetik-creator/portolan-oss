"""Tests for portolan.llm.factory.get_llm(). Construction only -- no
`.complete()` call, so no network for any provider.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from portolan.config import Settings
from portolan.llm import factory as factory_module
from portolan.llm.agent_sdk import AgentSDKClient
from portolan.llm.anthropic_api import AnthropicAPIClient
from portolan.llm.bedrock import BedrockClient
from portolan.llm.factory import PostgresUsageSink, get_llm
from portolan.llm.fake import FakeLLMClient
from portolan.llm.openrouter import OpenRouterClient


def _settings(tmp_path: Path, **overrides) -> Settings:
    defaults = {"database_url": "postgresql://localhost/x", "store_root": tmp_path}
    defaults.update(overrides)
    return Settings(_env_file=None, **defaults)


def test_get_llm_agent_sdk_provider(tmp_path: Path) -> None:
    cfg = _settings(tmp_path, llm_provider="agent_sdk", extraction_model="claude-haiku-4-5")
    client = get_llm(cfg)
    assert isinstance(client, AgentSDKClient)
    assert client.default_model == "claude-haiku-4-5"
    assert client.tenant == cfg.tenant


def test_get_llm_bedrock_provider(tmp_path: Path) -> None:
    cfg = _settings(
        tmp_path,
        llm_provider="bedrock",
        extraction_model="amazon.nova-lite-v1:0",
        aws_region="us-west-2",
    )
    client = get_llm(cfg)
    assert isinstance(client, BedrockClient)
    assert client.model_id == "amazon.nova-lite-v1:0"
    assert client.region == "us-west-2"


def test_get_llm_anthropic_provider(tmp_path: Path) -> None:
    cfg = _settings(
        tmp_path,
        llm_provider="anthropic",
        extraction_model="claude-haiku-4-5",
        anthropic_api_key="sk-ant-test",
    )
    client = get_llm(cfg)
    assert isinstance(client, AnthropicAPIClient)
    assert client.default_model == "claude-haiku-4-5"
    assert client.tenant == cfg.tenant


def test_get_llm_anthropic_provider_missing_key_raises_clear_error(tmp_path: Path) -> None:
    cfg = _settings(tmp_path, llm_provider="anthropic", anthropic_api_key=None)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        get_llm(cfg)


def test_get_llm_openrouter_provider(tmp_path: Path) -> None:
    cfg = _settings(
        tmp_path,
        llm_provider="openrouter",
        extraction_model="anthropic/claude-haiku-4.5",
        openrouter_api_key="sk-or-test",
        openrouter_provider_order=["anthropic"],
    )
    client = get_llm(cfg)
    assert isinstance(client, OpenRouterClient)
    assert client.default_model == "anthropic/claude-haiku-4.5"
    assert client.provider_order == ["anthropic"]
    assert client.tenant == cfg.tenant


def test_get_llm_openrouter_provider_missing_key_raises_clear_error(tmp_path: Path) -> None:
    cfg = _settings(tmp_path, llm_provider="openrouter", openrouter_api_key=None)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        get_llm(cfg)


def test_get_llm_fake_provider(tmp_path: Path) -> None:
    cfg = _settings(tmp_path, llm_provider="fake")
    client = get_llm(cfg)
    assert isinstance(client, FakeLLMClient)


def test_get_llm_creates_store_root(tmp_path: Path) -> None:
    store_root = tmp_path / "nested" / "store"
    cfg = _settings(tmp_path, store_root=store_root)
    get_llm(cfg)
    assert store_root.exists()


def test_get_llm_default_sink_is_postgres_backed(tmp_path: Path) -> None:
    """Review finding F12: the default sink used to be a JsonlSink under
    settings.store_root -- a container's ephemeral disk in production, so
    every metered cost vanished on the next deploy. Construction only;
    .record() is never called here, so no real database is touched."""
    cfg = _settings(tmp_path, database_url="postgresql://localhost/portolan_test_fix3")
    client = get_llm(cfg)
    assert isinstance(client.sink, PostgresUsageSink)
    assert client.sink.database_url == cfg.database_url.get_secret_value()


# --------------------------------------------------------------------------
# `provider=` override (infra-2b wave, 2026-09-18): real bug found live --
# `python -m portolan.bench extract --backend agent_sdk` silently went
# through OpenRouter because get_llm() obeyed only settings.llm_provider
# (LLM_PROVIDER in .env) and the CLI's own --backend choice was never
# passed in at all. An explicit `provider=` argument must win.
# --------------------------------------------------------------------------


def test_get_llm_explicit_provider_overrides_settings_llm_provider(tmp_path: Path) -> None:
    # The exact real-world scenario: .env/the environment says openrouter
    # (llm_provider="openrouter"), but the caller explicitly resolved
    # "agent_sdk" from its own source of truth (a CLI --backend flag) and
    # passes it here -- that must win, not the env-derived setting.
    cfg = _settings(
        tmp_path,
        llm_provider="openrouter",
        openrouter_api_key="sk-or-test",
        extraction_model="claude-sonnet-5",
    )
    client = get_llm(cfg, provider="agent_sdk")
    assert isinstance(client, AgentSDKClient)


def test_get_llm_provider_none_falls_back_to_settings_llm_provider(tmp_path: Path) -> None:
    # provider=None (the default -- every call site before this wave) is
    # unaffected: falls through to settings.llm_provider exactly as
    # before.
    cfg = _settings(tmp_path, llm_provider="openrouter", openrouter_api_key="sk-or-test")
    client = get_llm(cfg, provider=None)
    assert isinstance(client, OpenRouterClient)


def test_get_llm_explicit_provider_bedrock_overrides_settings_openrouter(tmp_path: Path) -> None:
    cfg = _settings(
        tmp_path,
        llm_provider="openrouter",
        openrouter_api_key="sk-or-test",
        extraction_model="amazon.nova-lite-v1:0",
    )
    client = get_llm(cfg, provider="bedrock")
    assert isinstance(client, BedrockClient)


def test_get_llm_defaults_to_module_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # No settings argument -- falls back to the module-level singleton.
    # Point that singleton's store_root at tmp_path first so this test
    # doesn't create a var/store/ directory in the real repo as a side
    # effect of running the suite.
    monkeypatch.setattr(factory_module, "default_settings", _settings(tmp_path))
    client = get_llm()
    assert hasattr(client, "complete")
