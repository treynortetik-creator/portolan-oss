"""`get_llm()`: build the `LLMClient` for `settings.llm_provider`. Prefer this
factory to constructing a client directly, so a provider switch is one
`.env` edit (`LLM_PROVIDER=...`), not a code change.
"""

from __future__ import annotations

from portolan.config import Settings
from portolan.config import settings as default_settings
from portolan.llm.agent_sdk import AgentSDKClient
from portolan.llm.anthropic_api import AnthropicAPIClient
from portolan.llm.base import LLMClient
from portolan.llm.bedrock import BedrockClient
from portolan.llm.fake import FakeLLMClient
from portolan.llm.openrouter import OpenRouterClient
from portolan.metering import UsageEvent, UsageSink


class PostgresUsageSink:
    """`UsageSink` backed by the `usage_events` table, so metered cost
    survives a redeploy (spec sec 11 bills chat by consumption).

    Opens one short-lived autocommit connection per event: model calls are
    not a hot loop, so a pool is not worth it here.
    """

    def __init__(self, database_url: str) -> None:
        self.database_url = database_url

    def record(self, event: UsageEvent) -> None:
        import psycopg
        from psycopg.types.json import Jsonb

        conn = psycopg.connect(self.database_url, autocommit=True)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO usage_events
                        (tenant, surface, model_key, input_tokens, output_tokens,
                         cached_tokens, cost_estimate, request_id, extra)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        event.tenant,
                        event.surface,
                        event.model_key,
                        event.input_tokens,
                        event.output_tokens,
                        event.cached_tokens,
                        event.cost_estimate,
                        event.request_id or None,
                        Jsonb(event.extra),
                    ),
                )
        finally:
            conn.close()


def _default_sink(cfg: Settings) -> UsageSink:
    """A `PostgresUsageSink` on `settings.database_url`. Also creates
    `store_root` if it is missing."""
    cfg.store_root.mkdir(parents=True, exist_ok=True)
    return PostgresUsageSink(cfg.database_url.get_secret_value())


def get_llm(settings: Settings | None = None, *, provider: str | None = None) -> LLMClient:
    """Construct the `LLMClient` named by `provider`, else by
    `settings.llm_provider` (`settings` defaults to the
    `portolan.config.settings` singleton).

    A caller that already chose a provider (a CLI flag) must pass it:
    otherwise `.env`'s `LLM_PROVIDER` silently wins over that choice.

    `provider="fake"` returns a `FakeLLMClient` with no canned responses;
    tests that need replies construct `FakeLLMClient` directly.
    """
    cfg = settings or default_settings
    sink = _default_sink(cfg)
    effective_provider = provider or cfg.llm_provider

    if effective_provider == "fake":
        return FakeLLMClient([], sink=sink, tenant=cfg.tenant)
    if effective_provider == "bedrock":
        return BedrockClient(cfg.extraction_model, cfg.aws_region, sink=sink, tenant=cfg.tenant)
    if effective_provider == "anthropic":
        if not cfg.anthropic_api_key:
            raise RuntimeError(
                "llm_provider='anthropic' needs ANTHROPIC_API_KEY set (in the environment "
                "or .env) -- portolan.config.Settings.anthropic_api_key is unset."
            )
        return AnthropicAPIClient(
            api_key=cfg.anthropic_api_key.get_secret_value(),
            default_model=cfg.extraction_model,
            sink=sink,
            tenant=cfg.tenant,
        )
    if effective_provider == "openrouter":
        if not cfg.openrouter_api_key:
            raise RuntimeError(
                "llm_provider='openrouter' needs OPENROUTER_API_KEY set (in the environment "
                "or .env) -- portolan.config.Settings.openrouter_api_key is unset."
            )
        # The app's ledger is its usage sink (Postgres usage_events), not a SpendMeter.
        return OpenRouterClient(
            spend_key="app",
            api_key=cfg.openrouter_api_key.get_secret_value(),
            default_model=cfg.extraction_model,
            provider_order=cfg.openrouter_provider_order,
            sink=sink,
            tenant=cfg.tenant,
        )
    # "agent_sdk", and any unrecognized value, fall through to the Agent SDK.
    return AgentSDKClient(default_model=cfg.extraction_model, sink=sink, tenant=cfg.tenant)
