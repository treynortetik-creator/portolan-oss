"""Application settings. Per-customer configuration lives in that customer's
`.env` / environment, never in code (portolan-spec-v2.md section 11).

`get_settings()` builds a fresh `Settings()` from the environment / `.env` as
they are now, so a `monkeypatch.setenv(...)` is seen immediately. `settings` is
a module-level instance built once at import.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Relative to this file, not the cwd, so `Settings()` finds the repo's `.env`
# from any working directory.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ENV_FILE = _PROJECT_ROOT / ".env"

_logger = logging.getLogger(__name__)

# The shortest `mcp_org_key` treated as a real key (`Settings._validate_mcp_org_key`).
MIN_MCP_ORG_KEY_CHARS = 32


class Settings(BaseSettings):
    """Per-customer / per-dev-machine configuration.

    Precedence (pydantic-settings default, highest wins): keyword argument >
    environment variable > `.env` file > field default.

    A field with a `validation_alias` reads that variable (`PORTOLAN_`-prefixed
    where a bare name would be too generic); every other field reads its bare
    upper-cased name (`STORE_ROOT`, `LLM_PROVIDER`, ...).
    """

    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        # other modules' .env entries (AWS_PROFILE, ...) aren't ours to validate
        extra="ignore",
        # so Settings(tenant="acme") works, not just the PORTOLAN_TENANT alias
        populate_by_name=True,
    )

    # Every secret field is a `SecretStr`: repr, str, `model_dump()` and a printed or logged
    # settings object show `**********`, never the value (pinned by tests/test_config.py).
    # Read one with `.get_secret_value()`, or `secret_value()` below for an optional one.
    # Default so a fresh checkout imports; .env or the environment overrides it.
    database_url: SecretStr = SecretStr("postgresql://localhost/portolan")
    tenant: str = Field(default="dev", validation_alias="PORTOLAN_TENANT")
    store_root: Path = Path("var/store")
    llm_provider: Literal["agent_sdk", "bedrock", "anthropic", "openrouter", "fake"] = "agent_sdk"
    extraction_model: str = "claude-haiku-4-5"  # or a Bedrock model id when llm_provider="bedrock"
    compare_model: str = "claude-haiku-4-5"
    strong_model: str = "claude-sonnet-5"

    # Chat's own model, so chat does not ride the batch extraction model.
    # Resolved by `portolan.web.routes.chat.resolve_chat_model`: unset falls back
    # to `strong_model`, and under `llm_provider="agent_sdk"` a value that is not
    # a `claude-...` id is replaced by a Claude default.
    chat_model: str | None = Field(default=None, validation_alias="CHAT_MODEL")

    # The chat question cap passed to `portolan.web.bounds.require_question_length`.
    # Same default as `bounds.MAX_QUESTION_CHARS`, duplicated to avoid an import cycle.
    chat_question_max_chars: int = Field(default=32_000, validation_alias="CHAT_QUESTION_MAX_CHARS")

    # "agent": chat answers through the tool loop in `portolan/web/chat_agent.py` (seeded
    # search, then search / open_note / graph_neighbors), falling back to "single", the
    # one-call answer, on any error, timeout or client without tool calls.
    # 01-spec/chat-agent-2026-09-25.md.
    chat_mode: Literal["single", "agent"] = Field(
        default="agent", validation_alias="PORTOLAN_CHAT_MODE"
    )
    # An OpenRouter model id: the loop needs `OpenRouterClient.complete_with_tools`.
    chat_agent_model: str = Field(
        default="anthropic/claude-sonnet-5", validation_alias="PORTOLAN_CHAT_AGENT_MODEL"
    )
    # Tool calls the model may make per question; the seed search is not one of them.
    chat_agent_max_tool_calls: int = Field(
        default=6, ge=0, le=20, validation_alias="PORTOLAN_CHAT_AGENT_MAX_TOOL_CALLS"
    )
    # Dollars per question: no research turn starts that could take the loop past it.
    chat_agent_max_usd: float = Field(
        default=0.15, gt=0, validation_alias="PORTOLAN_CHAT_AGENT_MAX_USD"
    )
    # Wall clock per question, checked between steps and passed to each model request.
    chat_agent_timeout_seconds: float = Field(
        default=45.0, gt=0, validation_alias="PORTOLAN_CHAT_AGENT_TIMEOUT_SECONDS"
    )
    embed_provider: Literal["fastembed", "titan", "fake"] = "fastembed"
    aws_region: str = "us-east-1"

    # `hnsw.ef_search` for search()'s vector leg, set per query (`SET LOCAL`). pgvector's
    # default 40 lost recall at 1.36M chunks; 200 matches 400's held-out recall on the ERB
    # exam at less latency (01-spec/bench-erb-search-2026-09-22.md, "Production path").
    # pgvector allows 1-1000.
    search_hnsw_ef_search: int = Field(
        default=200, ge=1, le=1000, validation_alias="PORTOLAN_SEARCH_HNSW_EF_SEARCH"
    )

    # search()'s local cross-encoder over its fused top `search_rerank_pool` notes
    # (`portolan/search/rerank.py`); "off" skips it. On CPU a model this size costs about a
    # second per note on the dev laptop (01-spec/bench-erb-search-2026-09-22.md, "Reranker
    # and query routing").
    search_rerank: Literal[
        "off",
        "bge-reranker-v2-m3",
        "qwen3-reranker-0.6b",
        "ms-marco-minilm-l6",
        "jina-reranker-v1-turbo-en",
    ] = Field(default="off", validation_alias="PORTOLAN_SEARCH_RERANK")
    search_rerank_pool: int = Field(
        default=30, ge=1, le=100, validation_alias="PORTOLAN_SEARCH_RERANK_POOL"
    )
    # Keyword-led or equal fusion weights per query, from the keyword leg's coverage of
    # the query (`portolan.search.query._route_weights`); off keeps one fixed pair. On by
    # default from 01-spec/integration-2026-09-25.md.
    search_fusion_routing: bool = Field(
        default=True, validation_alias="PORTOLAN_SEARCH_FUSION_ROUTING"
    )
    # Recency from each note's newest claim date (`notes.content_date`, migration 0035)
    # instead of its last write; off keeps `notes.updated` (`_note_recency`). Off by
    # default: on, it cost one of 15 paraphrased demo questions
    # (01-spec/content-dates-2026-09-25.md). `since=` reads the content date either way.
    search_recency_content_date: bool = Field(
        default=False, validation_alias="PORTOLAN_SEARCH_RECENCY_CONTENT_DATE"
    )
    # Where fastembed caches the embedding model; the reranker's files go there too.
    # Unset: fastembed's default, `<tmp>/fastembed_cache`.
    fastembed_cache_path: str | None = Field(default=None, validation_alias="FASTEMBED_CACHE_PATH")

    # Read by `docker/entrypoint.sh` from the shell before any Python runs;
    # declared here so they are named and typed too. Defaults match the entrypoint's.
    portolan_role: Literal["web", "worker", "migrate", "test"] = Field(
        default="web", validation_alias="PORTOLAN_ROLE"
    )
    portolan_connector: str = Field(default="file", validation_alias="PORTOLAN_CONNECTOR")
    portolan_worker_poll_seconds: int = Field(
        default=300, validation_alias="PORTOLAN_WORKER_POLL_SECONDS"
    )

    # The web app's `psycopg_pool.ConnectionPool` (`create_app`). `max_size` stays
    # under the 40 threads sync routes run on: connections are the scarcer resource.
    db_pool_min_size: int = Field(default=1, validation_alias="DB_POOL_MIN_SIZE")
    db_pool_max_size: int = Field(default=10, validation_alias="DB_POOL_MAX_SIZE")

    # A directory with a `pg_dump`/`pg_restore` matching the server's major
    # version, searched first by `portolan.db._resolve_pg_binary` (before the
    # Homebrew/PGDG locations and PATH). Only needed when none of those match.
    pg_bin_dir: str | None = Field(default=None, validation_alias="PG_BIN_DIR")

    # llm_provider="anthropic" (portolan/llm/anthropic_api.py). None so a fresh
    # checkout imports; get_llm() raises naming ANTHROPIC_API_KEY when it is needed.
    anthropic_api_key: SecretStr | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")

    # llm_provider="openrouter" (portolan/llm/openrouter.py); same None default.
    openrouter_api_key: SecretStr | None = Field(
        default=None, validation_alias="OPENROUTER_API_KEY"
    )

    # Optional OpenRouter provider order, e.g. ["anthropic", "together"]. None lets
    # OpenRouter choose.
    openrouter_provider_order: list[str] | None = Field(
        default=None, validation_alias="OPENROUTER_PROVIDER_ORDER"
    )

    # Auth secrets, with no literal fallback. `create_app` refuses to build without
    # `session_secret` and `org_key`; `build_http_app` refuses without `mcp_org_key`
    # unless OAuth is configured. The `None` defaults exist only so a fresh checkout imports.
    session_secret: SecretStr | None = Field(
        default=None, validation_alias="PORTOLAN_SESSION_SECRET"
    )
    org_key: SecretStr | None = Field(default=None, validation_alias="PORTOLAN_ORG_KEY")
    mcp_org_key: SecretStr | None = Field(default=None, validation_alias="PORTOLAN_MCP_ORG_KEY")

    @field_validator("mcp_org_key")
    @classmethod
    def _validate_mcp_org_key(cls, value: SecretStr | None) -> SecretStr | None:
        """Strip surrounding whitespace: a stored trailing newline could never match
        a `Bearer` header. A stripped value shorter than `MIN_MCP_ORG_KEY_CHARS`,
        blank included, is treated as unset, so no transport is gated by a
        non-secret; one warning is logged, never naming the value.
        """
        if value is None:
            return None
        stripped = value.get_secret_value().strip()
        if len(stripped) < MIN_MCP_ORG_KEY_CHARS:
            _logger.warning(
                "PORTOLAN_MCP_ORG_KEY is set but is blank or shorter than %d characters "
                "after stripping whitespace -- treating it as unset. Unless OAuth is "
                "configured, the MCP streamable-HTTP transport will not be mounted (/mcp "
                "will 404) until a real key is configured.",
                MIN_MCP_ORG_KEY_CHARS,
            )
            return None
        return SecretStr(stripped)

    # This deployment's public origin, for hosted MCP citation links
    # (`tool_open_source`'s `original_url`) and the default OAuth resource.
    # Unset gives a relative `/originals/<slug>`.
    public_base_url: str | None = Field(default=None, validation_alias="PORTOLAN_PUBLIC_URL")

    # Outgoing mail for `python -m portolan.report monday --send`
    # (`smtp://user:password@host:587`, or `smtps://` for implicit TLS). Unset, --send refuses.
    smtp_url: SecretStr | None = Field(default=None, validation_alias="PORTOLAN_SMTP_URL")
    smtp_from: str | None = Field(default=None, validation_alias="PORTOLAN_SMTP_FROM")

    # The pipeline always records CRM snapshots; on, it also writes the changes between them
    # as claims at the end of a run. Off by default: on the graph bench's demo graph the extra
    # CRM-change note moved multi-hop answer support 80.77 to 76.92
    # (01-spec/monday-email-2026-09-25.md). `python -m portolan.crm changes` writes them either way.
    crm_change_claims: bool = Field(default=False, validation_alias="PORTOLAN_CRM_CHANGE_CLAIMS")

    # OAuth on /mcp for claude.ai and ChatGPT connectors (portolan/mcp_server/oauth.py,
    # docs/mcp-hosted.md). Off unless the issuer is set. None of these is a secret: Portolan
    # is only the resource server and checks signed tokens against the issuer's public keys.
    mcp_oauth_issuer: str | None = Field(default=None, validation_alias="PORTOLAN_MCP_OAUTH_ISSUER")
    # The canonical URL of /mcp that tokens must name as their audience; unset means
    # PORTOLAN_PUBLIC_URL + "/mcp".
    mcp_oauth_resource: str | None = Field(
        default=None, validation_alias="PORTOLAN_MCP_OAUTH_RESOURCE"
    )
    # The organization whose members may use this deployment: a token must carry this value
    # in `mcp_oauth_org_claim`. Required, because an identity provider signs tokens for anyone
    # who can sign in to it, not only this customer's people.
    mcp_oauth_org_id: str | None = Field(default=None, validation_alias="PORTOLAN_MCP_OAUTH_ORG_ID")
    mcp_oauth_org_claim: str = Field(
        default="org_id", validation_alias="PORTOLAN_MCP_OAUTH_ORG_CLAIM"
    )
    # Space-separated; every one must be granted. "openid" because WorkOS AuthKit grants only
    # the OpenID scopes (openid, profile, email, offline_access).
    mcp_oauth_required_scopes: str = Field(
        default="openid", validation_alias="PORTOLAN_MCP_OAUTH_REQUIRED_SCOPES"
    )
    # Optional override; unset means the `jwks_uri` in the issuer's own metadata.
    mcp_oauth_jwks_url: str | None = Field(
        default=None, validation_alias="PORTOLAN_MCP_OAUTH_JWKS_URL"
    )

    @field_validator(
        "mcp_oauth_issuer",
        "mcp_oauth_resource",
        "mcp_oauth_org_id",
        "mcp_oauth_jwks_url",
        mode="before",
    )
    @classmethod
    def _blank_is_unset(cls, value: object) -> object:
        """A Railway variable left empty arrives as "" -- that means unset, not a value."""
        if isinstance(value, str):
            value = value.strip()
            return value or None
        return value

    # Web app sign-in (`portolan.web.signin`, 01-spec/pilot-sign-in-2026-09-25.md). "none" is
    # the shared org key plus a typed name; "workos" is WorkOS AuthKit, and `create_app`
    # refuses to start unless the WorkOS client ID, API key, public URL and at least one of
    # the organization or allowed domains below are set.
    web_auth: Literal["none", "workos"] = Field(
        default="none", validation_alias="PORTOLAN_WEB_AUTH"
    )
    # Public: it appears in every sign-in URL the browser follows.
    workos_client_id: str | None = Field(default=None, validation_alias="PORTOLAN_WORKOS_CLIENT_ID")
    workos_api_key: SecretStr | None = Field(
        default=None, validation_alias="PORTOLAN_WORKOS_API_KEY"
    )
    # The WorkOS organization whose members may sign in to the web app; unset falls back to
    # `mcp_oauth_org_id`, so the web app and the MCP connector share one member list.
    web_auth_org_id: str | None = Field(default=None, validation_alias="PORTOLAN_WEB_AUTH_ORG_ID")
    # Comma- or space-separated email domains (e.g. "acme.com, acme.co.uk"). When set, a
    # person's verified email must be at one of them, on top of the organization check.
    web_auth_allowed_domains: str = Field(
        default="", validation_alias="PORTOLAN_WEB_AUTH_ALLOWED_DOMAINS"
    )
    # Comma- or space-separated emails who may open /admin/usage and /admin/people under
    # web_auth="workos".
    admin_emails: str = Field(default="", validation_alias="PORTOLAN_ADMIN_EMAILS")

    @field_validator("workos_client_id", "web_auth_org_id", mode="before")
    @classmethod
    def _blank_web_auth_is_unset(cls, value: object) -> object:
        if isinstance(value, str):
            value = value.strip()
            return value or None
        return value

    @field_validator("workos_api_key")
    @classmethod
    def _blank_api_key_is_unset(cls, value: SecretStr | None) -> SecretStr | None:
        if value is None:
            return None
        stripped = value.get_secret_value().strip()
        return SecretStr(stripped) if stripped else None

    # Session cookie `Secure` flag. Turn off only for local development over plain
    # http://127.0.0.1, never for anything reachable over the network.
    cookie_secure: bool = Field(default=True, validation_alias="PORTOLAN_COOKIE_SECURE")

    # `/docs`, `/redoc`, `/openapi.json`. Off by default; when on, `create_app`
    # still gates them behind `require_session`.
    enable_api_docs: bool = Field(default=False, validation_alias="PORTOLAN_ENABLE_API_DOCS")

    # `POST /login` throttle (`portolan.web.routes.login.LoginRateLimiter`): too many
    # wrong keys from one IP in the window trip a lockout that doubles per lockout up
    # to the max; the global limit caps failures across all IPs, so the one shared
    # key cannot be guessed faster by spreading guesses over many addresses.
    login_rate_limit_per_ip: int = Field(
        default=5, validation_alias="PORTOLAN_LOGIN_RATE_LIMIT_PER_IP"
    )
    login_rate_limit_window_seconds: float = Field(
        default=60.0, validation_alias="PORTOLAN_LOGIN_RATE_LIMIT_WINDOW_SECONDS"
    )
    login_rate_limit_backoff_base_seconds: float = Field(
        default=60.0, validation_alias="PORTOLAN_LOGIN_RATE_LIMIT_BACKOFF_BASE_SECONDS"
    )
    login_rate_limit_backoff_max_seconds: float = Field(
        default=900.0, validation_alias="PORTOLAN_LOGIN_RATE_LIMIT_BACKOFF_MAX_SECONDS"
    )
    login_rate_limit_global_per_minute: int = Field(
        default=60, validation_alias="PORTOLAN_LOGIN_RATE_LIMIT_GLOBAL_PER_MINUTE"
    )

    # Take the login client IP from `X-Forwarded-For`. Turn on only behind one
    # trusted proxy: without one the header is client-supplied and would let
    # anyone bypass the per-IP throttle.
    trust_forwarded_for: bool = Field(
        default=False, validation_alias="PORTOLAN_TRUST_FORWARDED_FOR"
    )

    # Pilot activity log (`portolan.activity`, migration 0037): rows older than this
    # many days are deleted at web startup and by `python -m portolan.activity purge`.
    # 0 keeps every row.
    activity_retention_days: int = Field(
        default=180, ge=0, validation_alias="PORTOLAN_ACTIVITY_RETENTION_DAYS"
    )

    # External signals (`portolan.signals`, 01-spec/external-signals-2026-09-23.md).
    # Off by default: `python -m portolan.signals scan` refuses to run, and the weekly
    # report prints no "Public signals" section, until this is true.
    external_signals_enabled: bool = Field(
        default=False, validation_alias="PORTOLAN_EXTERNAL_SIGNALS"
    )
    # The relevance judge. Public information only, so any model is allowed here. Luna
    # scored 8 of 8 with no false alarm on the 2026-09-23 live check; Nova Lite 6 of 8
    # (01-spec/external-signals-2026-09-23.md, "Judge quality").
    external_signals_model: str = Field(
        default="openai/gpt-5.6-luna", validation_alias="PORTOLAN_EXTERNAL_SIGNALS_MODEL"
    )
    # Sent on every page, feed and robots.txt request, so a site can tell who is asking.
    external_signals_user_agent: str = Field(
        default="PortolanSignals/0.1 (weekly public-page check; respects robots.txt)",
        validation_alias="PORTOLAN_EXTERNAL_SIGNALS_USER_AGENT",
    )


def secret_value(secret: SecretStr | None) -> str | None:
    """The plain text of an optional secret setting, for the one call that needs it."""
    return None if secret is None else secret.get_secret_value()


def get_settings() -> Settings:
    """Build a fresh `Settings` from the current environment/`.env`.

    Not cached, so an environment change is seen on the next call
    (`tests/test_config.py::test_get_settings_is_not_cached`).
    """
    return Settings()


settings = get_settings()
