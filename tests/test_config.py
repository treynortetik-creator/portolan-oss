"""Tests for portolan.config.

`Settings` reads from real environment variables, the repo's `.env`, and
constructor kwargs, in that precedence order. Every test here that cares
about a specific value passes `_env_file=None` (a pydantic-settings
built-in override) so the dev machine's own `.env` -- which exists locally
so the rest of the app can import `portolan.config.settings` at all --
can't leak an unexpected value into an assertion.
"""

from __future__ import annotations

import pydantic
import pytest

from portolan.config import Settings, get_settings


def test_settings_reads_database_url_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/portolan_env_test")
    settings = get_settings()
    assert settings.database_url.get_secret_value() == "postgresql://localhost/portolan_env_test"


def test_settings_reads_aws_region_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/x")
    monkeypatch.setenv("AWS_REGION", "us-west-2")
    settings = get_settings()
    assert settings.aws_region == "us-west-2"


def test_settings_tenant_reads_portolan_tenant_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/x")
    monkeypatch.setenv("PORTOLAN_TENANT", "acme")
    settings = get_settings()
    assert settings.tenant == "acme"


def test_settings_reads_llm_fields_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/x")
    monkeypatch.setenv("LLM_PROVIDER", "bedrock")
    monkeypatch.setenv("EXTRACTION_MODEL", "amazon.nova-lite-v1:0")
    monkeypatch.setenv("COMPARE_MODEL", "amazon.nova-lite-v1:0")
    monkeypatch.setenv("STRONG_MODEL", "anthropic.claude-sonnet-5")
    monkeypatch.setenv("EMBED_PROVIDER", "titan")
    settings = get_settings()
    assert settings.llm_provider == "bedrock"
    assert settings.extraction_model == "amazon.nova-lite-v1:0"
    assert settings.compare_model == "amazon.nova-lite-v1:0"
    assert settings.strong_model == "anthropic.claude-sonnet-5"
    assert settings.embed_provider == "titan"


def test_settings_defaults_match_the_contract() -> None:
    settings = Settings(database_url="postgresql://localhost/x", _env_file=None)
    assert settings.tenant == "dev"
    assert str(settings.store_root) == "var/store"
    assert settings.llm_provider == "agent_sdk"
    assert settings.extraction_model == "claude-haiku-5-5"
    assert settings.compare_model == "claude-haiku-5-5"
    assert settings.strong_model == "claude-sonnet-5-5"
    assert settings.embed_provider == "fastembed"
    assert settings.aws_region == "us-east-1"
    assert settings.chat_question_max_chars == 32_000
    # Bug hunt 2026-09-18, finding 1: `POST /login`'s rate-limit knobs are
    # real, overridable `Settings` fields (portolan.web.routes.login's
    # `LoginRateLimiter`), never bare module constants.
    assert settings.login_rate_limit_per_ip == 5
    assert settings.login_rate_limit_window_seconds == 60.0
    assert settings.login_rate_limit_backoff_base_seconds == 60.0
    assert settings.login_rate_limit_backoff_max_seconds == 900.0
    assert settings.login_rate_limit_global_per_minute == 60
    assert settings.trust_forwarded_for is False


def test_settings_reads_login_rate_limit_per_ip_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PORTOLAN_LOGIN_RATE_LIMIT_PER_IP", "10")
    settings = get_settings()
    assert settings.login_rate_limit_per_ip == 10


def test_settings_reads_login_rate_limit_backoff_max_seconds_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PORTOLAN_LOGIN_RATE_LIMIT_BACKOFF_MAX_SECONDS", "120")
    settings = get_settings()
    assert settings.login_rate_limit_backoff_max_seconds == 120.0


def test_settings_reads_trust_forwarded_for_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Railway (and most deployments) sit behind a reverse proxy -- off
    by default so a caller can't just supply its own `X-Forwarded-For`
    to pick which IP the login throttle keys on."""
    monkeypatch.setenv("PORTOLAN_TRUST_FORWARDED_FOR", "true")
    settings = get_settings()
    assert settings.trust_forwarded_for is True


def test_settings_reads_chat_question_max_chars_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Owner instruction: "make the cap a setting with that default" --
    the chat composer's character cap must be a real, overridable
    `Settings` field, not a hardcoded constant."""
    monkeypatch.setenv("CHAT_QUESTION_MAX_CHARS", "5000")
    settings = get_settings()
    assert settings.chat_question_max_chars == 5000


def test_settings_accepts_field_name_alongside_alias() -> None:
    # populate_by_name=True: the Python field name works even though
    # `tenant`'s env alias is PORTOLAN_TENANT, not TENANT.
    settings = Settings(database_url="postgresql://localhost/x", tenant="direct", _env_file=None)
    assert settings.tenant == "direct"


def test_settings_defaults_database_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """A fresh checkout with no .env and no DATABASE_URL still imports and points at a local DB."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    settings = Settings(_env_file=None)
    assert settings.database_url.get_secret_value() == "postgresql://localhost/portolan"


def test_settings_rejects_unknown_llm_provider() -> None:
    with pytest.raises(pydantic.ValidationError):
        Settings(database_url="postgresql://localhost/x", llm_provider="bogus", _env_file=None)  # type: ignore[arg-type]


# --------------------------------------------------------------------
# Bug hunt 2026-09-18 (ops review, P1 finding 4): PORTOLAN_ROLE and
# friends were read by docker/entrypoint.sh but documented in neither
# .env.example nor this module.
# --------------------------------------------------------------------


def test_settings_defaults_portolan_role_to_web(monkeypatch: pytest.MonkeyPatch) -> None:
    """Matches docker/entrypoint.sh's own default -- an unset/dropped
    PORTOLAN_ROLE must mean "serve the web app," not fall through to
    anything else (the 2026-09-12 F4 outage this finding is a recurrence
    of).

    CI-vs-docker fix (2026-09-19): `_env_file=None` only turns off reading
    `.env` -- `Settings` (pydantic-settings `BaseSettings`) still reads a
    real ambient `PORTOLAN_ROLE` from the process environment same as
    always, and the `docker` CI job's own "run the suite in-container"
    step sets exactly that (`-e PORTOLAN_ROLE=test`, `.github/workflows/
    ci.yml`) so the whole container's test run sees it. This test means
    to assert the field's DEFAULT, so it must clear the ambient value
    first, same as `test_settings_defaults_database_url` already does for
    `DATABASE_URL` just above."""
    monkeypatch.delenv("PORTOLAN_ROLE", raising=False)
    settings = Settings(database_url="postgresql://localhost/x", _env_file=None)
    assert settings.portolan_role == "web"


def test_settings_reads_portolan_role_and_worker_fields_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/x")
    monkeypatch.setenv("PORTOLAN_ROLE", "worker")
    monkeypatch.setenv("PORTOLAN_CONNECTOR", "gong")
    monkeypatch.setenv("PORTOLAN_WORKER_POLL_SECONDS", "60")
    settings = get_settings()
    assert settings.portolan_role == "worker"
    assert settings.portolan_connector == "gong"
    assert settings.portolan_worker_poll_seconds == 60


def test_settings_rejects_unknown_portolan_role() -> None:
    with pytest.raises(pydantic.ValidationError):
        Settings(database_url="postgresql://localhost/x", portolan_role="bogus", _env_file=None)  # type: ignore[arg-type]


# --------------------------------------------------------------------
# Bug hunt 2026-09-18, P0 finding 1: the connection-pool sizing knobs
# `portolan.web.app.create_app`'s lifespan reads.
# --------------------------------------------------------------------


def test_settings_defaults_db_pool_sizes() -> None:
    settings = Settings(database_url="postgresql://localhost/x", _env_file=None)
    assert settings.db_pool_min_size == 1
    assert settings.db_pool_max_size == 10


def test_settings_reads_db_pool_sizes_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/x")
    monkeypatch.setenv("DB_POOL_MIN_SIZE", "2")
    monkeypatch.setenv("DB_POOL_MAX_SIZE", "25")
    settings = get_settings()
    assert settings.db_pool_min_size == 2
    assert settings.db_pool_max_size == 25


def test_settings_search_hnsw_ef_search_default_and_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # The default chosen from the ERB exam's ef_search sweep
    # (01-spec/bench-erb-search-2026-09-22.md, "Production path").
    settings = Settings(database_url="postgresql://localhost/x", _env_file=None)
    assert settings.search_hnsw_ef_search == 200
    monkeypatch.setenv("PORTOLAN_SEARCH_HNSW_EF_SEARCH", "120")
    assert get_settings().search_hnsw_ef_search == 120


@pytest.mark.parametrize("value", [0, 1001])
def test_settings_rejects_search_hnsw_ef_search_outside_pgvectors_range(value: int) -> None:
    with pytest.raises(pydantic.ValidationError):
        Settings(
            database_url="postgresql://localhost/x", search_hnsw_ef_search=value, _env_file=None
        )


def test_settings_rejects_unknown_embed_provider() -> None:
    with pytest.raises(pydantic.ValidationError):
        Settings(database_url="postgresql://localhost/x", embed_provider="bogus", _env_file=None)  # type: ignore[arg-type]


def test_get_settings_is_not_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/x")
    monkeypatch.setenv("PORTOLAN_TENANT", "first")
    first = get_settings()
    monkeypatch.setenv("PORTOLAN_TENANT", "second")
    second = get_settings()
    assert first.tenant == "first"
    assert second.tenant == "second"


def test_module_level_settings_singleton_is_importable() -> None:
    # The singleton was already constructed at import time (against this
    # dev machine's real environment/.env) -- just prove it's a usable
    # Settings instance with the right type on every field, not a stub.
    from portolan.config import settings

    assert isinstance(settings, Settings)
    assert settings.llm_provider in ("agent_sdk", "bedrock", "anthropic", "openrouter", "fake")


def test_settings_accepts_anthropic_and_openrouter_llm_providers() -> None:
    anthropic_cfg = Settings(
        database_url="postgresql://localhost/x", llm_provider="anthropic", _env_file=None
    )
    assert anthropic_cfg.llm_provider == "anthropic"

    openrouter_cfg = Settings(
        database_url="postgresql://localhost/x", llm_provider="openrouter", _env_file=None
    )
    assert openrouter_cfg.llm_provider == "openrouter"


def test_settings_anthropic_and_openrouter_keys_default_to_none() -> None:
    settings = Settings(database_url="postgresql://localhost/x", _env_file=None)
    assert settings.anthropic_api_key is None
    assert settings.openrouter_api_key is None
    assert settings.openrouter_provider_order is None


def test_settings_reads_anthropic_api_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/x")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-value")
    settings = get_settings()
    assert settings.anthropic_api_key.get_secret_value() == "sk-ant-test-value"


def test_settings_reads_openrouter_api_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/x")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-value")
    settings = get_settings()
    assert settings.openrouter_api_key.get_secret_value() == "sk-or-test-value"


def test_settings_reads_openrouter_provider_order_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/x")
    monkeypatch.setenv("OPENROUTER_PROVIDER_ORDER", '["anthropic", "together"]')
    settings = get_settings()
    assert settings.openrouter_provider_order == ["anthropic", "together"]


# --- mcp_org_key validation (security review 2026-09-19, S1) --------------
#
# Found: `PORTOLAN_MCP_ORG_KEY=" "` mounted `/mcp` and `Authorization:
# "Bearer  "` passed the auth gate -- a whitespace-only "key" is not a
# secret. The other direction: a key stored with incidental leading/
# trailing whitespace (a copy-paste artifact, a trailing newline from
# `echo >> .env`) silently bricked the endpoint with no log saying why.
# `mcp_org_key` is now stripped, and a stripped value under 32 characters
# (blank included) is treated as if it were never set at all, with one
# warning line naming nothing about the key's actual contents.

_REAL_KEY = "a" * 32  # the shortest value this validator still accepts


def test_mcp_org_key_of_valid_length_is_kept_verbatim() -> None:
    settings = Settings(database_url="postgresql://localhost/x", mcp_org_key=_REAL_KEY)
    assert settings.mcp_org_key.get_secret_value() == _REAL_KEY


def test_mcp_org_key_is_stripped_of_surrounding_whitespace() -> None:
    settings = Settings(database_url="postgresql://localhost/x", mcp_org_key=f"  {_REAL_KEY}\n")
    assert settings.mcp_org_key.get_secret_value() == _REAL_KEY


def test_mcp_org_key_unset_stays_none_with_no_warning(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("WARNING"):
        settings = Settings(database_url="postgresql://localhost/x")
    assert settings.mcp_org_key is None
    assert not caplog.records


def test_mcp_org_key_blank_is_treated_as_unset_and_warns(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("WARNING"):
        settings = Settings(database_url="postgresql://localhost/x", mcp_org_key="   ")
    assert settings.mcp_org_key is None
    assert len(caplog.records) == 1
    assert "PORTOLAN_MCP_ORG_KEY" in caplog.records[0].message


def test_mcp_org_key_under_32_chars_after_stripping_is_treated_as_unset_and_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    short_key = "a" * 31
    with caplog.at_level("WARNING"):
        settings = Settings(database_url="postgresql://localhost/x", mcp_org_key=f" {short_key} ")
    assert settings.mcp_org_key is None
    assert len(caplog.records) == 1
    # Never the key material itself, right or wrong (matches
    # routes/login.py's own "never the submitted key" logging rule).
    assert short_key not in caplog.records[0].message


def test_mcp_org_key_exactly_32_chars_after_stripping_is_kept() -> None:
    key = "b" * 32
    settings = Settings(database_url="postgresql://localhost/x", mcp_org_key=f"  {key}  ")
    assert settings.mcp_org_key.get_secret_value() == key


# --- secrets are unprintable (guardrails review 2026-09-22, rank 6) -------
#
# The secrets hook only sees the command an agent types; a script that prints the
# settings object, a traceback repr or a log line got every key out anyway.

_PLANTED = "planted-never-print-7f3a"
_SECRET_ENV = {
    "OPENROUTER_API_KEY": f"{_PLANTED}-openrouter",
    "ANTHROPIC_API_KEY": f"{_PLANTED}-anthropic",
    "PORTOLAN_SESSION_SECRET": f"{_PLANTED}-session",
    "PORTOLAN_ORG_KEY": f"{_PLANTED}-org",
    "PORTOLAN_MCP_ORG_KEY": f"{_PLANTED}-mcp-org-key-long-enough",
}
# Assembled at run time so the source never holds a DSN with an inline password.
_PLANTED_DSN = "postgresql://planted:" + _PLANTED + "-dbpass@localhost/portolan_test_x"


def test_no_settings_rendering_contains_a_secret(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging

    monkeypatch.setenv("DATABASE_URL", _PLANTED_DSN)
    for name, value in _SECRET_ENV.items():
        monkeypatch.setenv(name, value)
    settings = get_settings()

    # Control: every planted value really was loaded, so "absent below" means hidden.
    assert settings.database_url.get_secret_value() == _PLANTED_DSN
    assert settings.openrouter_api_key.get_secret_value() == _SECRET_ENV["OPENROUTER_API_KEY"]
    assert settings.anthropic_api_key.get_secret_value() == _SECRET_ENV["ANTHROPIC_API_KEY"]
    assert settings.session_secret.get_secret_value() == _SECRET_ENV["PORTOLAN_SESSION_SECRET"]
    assert settings.org_key.get_secret_value() == _SECRET_ENV["PORTOLAN_ORG_KEY"]
    assert settings.mcp_org_key.get_secret_value() == _SECRET_ENV["PORTOLAN_MCP_ORG_KEY"]

    print(settings)
    print(settings.__dict__)
    with caplog.at_level(logging.INFO):
        logging.getLogger("portolan.test_config").info("settings %s %r", settings, settings)
    renderings = {
        "repr": repr(settings),
        "str": str(settings),
        "format": f"{settings} {settings!r}",
        "model_dump": str(settings.model_dump()),
        "model_dump_json": settings.model_dump_json(),
        "vars": str(vars(settings)),
        "print": capsys.readouterr().out,
        "log": caplog.text,
    }
    assert renderings["log"] and renderings["print"]
    leaks = {name for name, text in renderings.items() if _PLANTED in text}
    assert not leaks, f"a secret setting was rendered by: {sorted(leaks)}"
