"""Brief -> source text.

Three ways words get produced, in the order the module tries them:

1. **API mode** — the Anthropic SDK, one call per source, asked to plant
   each fact where its brief says and wrap it in ``⟦F01⟧ ... ⟦/F01⟧``.
2. **Manual mode, staged** — a human (or a delegated subagent, itself
   Claude, briefed with the exact facts/positions/phrasing) has already
   hand-written the source and saved it under
   ``corpus/generator/_staging/<type>/<source_id>.txt`` (or ``.json`` for
   ticket comments), markers and all. This module just reads it back.
3. **Manual mode, procedural filler** — for noise sources and the
   no-facts distractors, where realism matters far less than volume and
   variety, deterministic seeded-random template generation stands in for
   hand-written prose. No model, no marker discipline needed (there is
   nothing to plant), just plausible filler.

`cli.py` decides which path applies to a given brief and calls the
matching function here.
"""

from __future__ import annotations

import random
import re
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from corpus.generator.briefs import Brief
from corpus.generator.structured import email_header_block, seconds_to_hhmmss

# --------------------------------------------------------------------------
# Mode detection
# --------------------------------------------------------------------------


def _env_or_dotenv(project_root: Path, key: str) -> str | None:
    """`key`'s value from the real environment, else from `<project_root>/.env`
    (a bare `KEY=` line does not count as set), else `None`. Same lookup
    `detect_mode` already did inline for `ANTHROPIC_API_KEY`; factored out
    so the `OPENROUTER_API_KEY` check below can reuse it."""
    import os

    value = os.environ.get(key)
    if value:
        return value
    env_path = project_root / ".env"
    if env_path.exists():
        try:
            for line in env_path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if stripped.startswith(f"{key}=") and stripped != f"{key}=":
                    return stripped.split("=", 1)[1]
        except OSError:
            pass
    return None


def detect_mode(project_root: Path) -> tuple[str, str]:
    """Return (mode, reason). mode is "openrouter", "agent_sdk", "api", or
    "manual".

    Detection order:
    1. `OPENROUTER_API_KEY` in the environment or in `<project_root>/.env`
       -> openrouter mode. Checked FIRST and preferred over every other
       mode when set: it is the only mode that writes the full corpus at
       real, un-reduced call lengths (12k-25k tokens) through a live model
       cheap enough to do that at scale (`CORPUS_WRITER_MODEL`, default
       `openai/gpt-5.6-luna`, ~$0.20/M input tokens) -- see prose.py's
       `generate_via_openrouter` and GENERATOR.md.
    2. The `claude` CLI is on PATH -> agent_sdk mode: local Claude Code
       login, no API key (adr-001-language-and-stack.md "Local model
       access"). Cheap check on purpose -- PATH presence, not an actual
       smoke call -- since `detect_mode` runs on every "auto" invocation.
       Preferred over API mode because it's the zero-setup path on a dev
       machine that's already logged in to Claude Code.
    3. `ant auth status` reports an active profile -> API mode, zero-arg client.
    4. `ANTHROPIC_API_KEY` in the environment or in `<project_root>/.env`.
    5. Otherwise, manual mode.
    """
    import shutil

    if _env_or_dotenv(project_root, "OPENROUTER_API_KEY"):
        return "openrouter", "OPENROUTER_API_KEY set (environment or .env)"

    if shutil.which("claude"):
        return "agent_sdk", "the `claude` CLI is on PATH"

    try:
        result = subprocess.run(
            ["ant", "auth", "status"], capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            return "api", "`ant auth status` reported an active profile"
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass

    import os

    if os.environ.get("ANTHROPIC_API_KEY"):
        return "api", "ANTHROPIC_API_KEY set in the environment"

    env_path = project_root / ".env"
    if env_path.exists():
        try:
            for line in env_path.read_text(encoding="utf-8").splitlines():
                if (
                    line.strip().startswith("ANTHROPIC_API_KEY=")
                    and line.strip() != "ANTHROPIC_API_KEY="
                ):
                    return "api", f"ANTHROPIC_API_KEY found in {env_path}"
        except OSError:
            pass

    return (
        "manual",
        "no `claude` CLI on PATH, no `ant` CLI profile, no ANTHROPIC_API_KEY in the "
        "environment or .env",
    )


# --------------------------------------------------------------------------
# API mode
# --------------------------------------------------------------------------

STABLE_SYSTEM_PROMPT = """You are generating one source document for a synthetic corpus that \
tests a customer-intelligence extraction pipeline. You will be given a brief for exactly one \
source: its type, date, participants, target length, and a list of facts to plant.

Rules, followed exactly:
1. Write the full source in the voice of its participants — realistic hesitations, tangents, \
and small talk on calls; terse language on tickets; ordinary email register with a signature.
2. Plant each listed fact at the sentence that most naturally carries it, positioned where the \
brief's `position` says (early = first 25% of the source, middle = 35-65%, late = last 25%), \
spoken/written by the named speaker, honoring the exact `phrasing` constraint.
3. Wrap ONLY the sentence that carries a fact in markers: ⟦F01⟧ that sentence. ⟦/F01⟧ — using \
that fact's exact id. Exactly one marker pair per listed fact. Nothing else in the source is \
marked. This applies EVEN WHEN a fact's phrasing says "no constraint — state plainly": stating it \
plainly changes the WORDS you use, never whether you wrap it. Every single fact listed below MUST \
have its marker pair somewhere in your reply — before sending your reply, check it against the \
fact list one more time and add any marker you skipped. Example: given a fact with no phrasing \
constraint, correct output is "...and ⟦F09⟧the delivery arrived Tuesday⟦/F09⟧, which was..." — \
NOT the same sentence with the marker glyphs left out.
4. Never write any of the words or phrases in that fact's `forbidden` list, in any form.
5. Calls: every line starts with `[HH:MM:SS] Speaker Name:` and timestamps advance realistically \
at roughly 150 words per minute. Tickets: a JSON object with numbered comments. Emails: standard \
headers (Message-ID, From, To, Date, Subject) then a blank line then the body.
6. Do not invent additional facts beyond what the brief lists. Everything else in the source is \
plausible color: small talk, scheduling, unrelated shop talk.
7. Tickets ONLY: reply with ONE raw JSON object, no markdown code fence, no prose before or \
after, exactly this shape: {"subject": "<short subject line>", "comments": \
[{"number": 1, "author": "<requester name or 'Halvard Support'>", "body": "<comment text, \
markers here if this comment carries a fact>"}, ...]}. 2-4 comments, first from the requester.
8. Emails ONLY: reply with ONLY the body text (no headers -- those are added separately), \
starting with a greeting and ending with a sign-off, paragraphs separated by one blank line.
"""


def build_user_prompt(brief: Brief) -> str:
    lines = [
        f"source_id: {brief.source_id}",
        f"source_type: {brief.source_type}",
        f"date: {brief.date}",
        "participants: "
        + ", ".join(f"{p.name} ({p.role or 'unknown role'})" for p in brief.participants),
    ]
    if brief.extra.get("target_words"):
        lines.append(f"target_length_words: {brief.extra['target_words']}")
    if brief.is_distractor:
        lines.append(f"distractor_instruction: {brief.distractor_instruction}")
    if brief.is_noise:
        lines.append(f"topic (no facts — plausible color only): {brief.topic}")
    for f in brief.facts:
        lines.append(
            "\n".join(
                [
                    f"- fact {f.fact_id} ({f.fact_type}), position={f.position}, speaker={f.speaker}:",
                    f"  claim: {f.claim}",
                    f"  phrasing: {f.phrasing or '(no constraint — state plainly)'}",
                    f"  forbidden: {f.forbidden or '(none)'}",
                    f"  quote_required: {f.quote_required}",
                ]
            )
        )
    return "\n".join(lines)


def generate_via_api(client: Any, brief: Brief, model: str = "claude-sonnet-5") -> str:
    """One call per source. Streaming, adaptive thinking, cached stable
    system prompt. Returns the raw text (markers still in place)."""
    with client.messages.stream(
        model=model,
        max_tokens=16000,
        thinking={"type": "adaptive"},
        system=[
            {
                "type": "text",
                "text": STABLE_SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": build_user_prompt(brief)}],
    ) as stream:
        final_message = stream.get_final_message()
    parts = [block.text for block in final_message.content if block.type == "text"]
    return "\n".join(parts)


# --------------------------------------------------------------------------
# Agent SDK mode
# --------------------------------------------------------------------------


def generate_via_agent_sdk(brief: Brief, model: str = "claude-sonnet-5") -> str:
    """One call per source via `portolan.llm.AgentSDKClient` -- the local
    Claude Code login, no API key (adr-001-language-and-stack.md "Local
    model access"). Same marker contract as `generate_via_api`: same
    stable system prompt, same user prompt, returns the raw text with
    ⟦F01⟧...⟦/F01⟧ markers still in place.

    Per the project's ground rule that every model call goes through
    `portolan.llm.LLMClient` (build-contracts.md "Ground rules"), this
    goes through the same client the extraction pipeline uses rather than
    calling `claude_agent_sdk` directly. `surface="chat"` is the
    least-wrong bucket in `portolan.metering`'s closed surface list for an
    unstructured, no-schema text completion like this one; it's corpus
    tooling, not the production pipeline, so metering precision here
    doesn't affect a real bill.
    """
    from portolan.llm.agent_sdk import AgentSDKClient

    client = AgentSDKClient(default_model=model)
    result = client.complete(
        system=STABLE_SYSTEM_PROMPT,
        user=build_user_prompt(brief),
        model=model,
        surface="chat",
    )
    return result.text or ""


# --------------------------------------------------------------------------
# OpenRouter mode
# --------------------------------------------------------------------------

CORPUS_WRITER_MODEL_ENV = "CORPUS_WRITER_MODEL"
# ~$0.20/M input, ~$1.20/M output on OpenRouter as of 2026-09-13 (confirmed
# live against https://openrouter.ai/api/v1/models) -- writes well and is
# cheap enough to draft the whole corpus (roughly 400k output tokens, about
# $0.60) for a fraction of what a frontier model would cost on the same job.
DEFAULT_CORPUS_WRITER_MODEL = "openai/gpt-5.6-luna"

# Stop generating and report rather than keep spending once OpenRouter's own
# reported `usage.cost` (summed across every call this run has made) crosses
# this line -- the budget the build instructions set, 5x the ~$0.60 estimate.
CORPUS_GENERATION_COST_LIMIT_USD = 3.0


class CorpusBudgetExceeded(RuntimeError):
    """Raised by the CLI (not this module) when cumulative OpenRouter cost
    for this generation run crosses `CORPUS_GENERATION_COST_LIMIT_USD`."""


def openrouter_writer_model() -> str:
    """`CORPUS_WRITER_MODEL` from the environment, else
    `DEFAULT_CORPUS_WRITER_MODEL`. A bare `CORPUS_WRITER_MODEL=` (empty)
    counts as unset, same convention `_env_or_dotenv` uses for API keys."""
    import os

    return os.environ.get(CORPUS_WRITER_MODEL_ENV) or DEFAULT_CORPUS_WRITER_MODEL


def build_openrouter_client(project_root: Path) -> Any:
    """One `portolan.llm.openrouter.OpenRouterClient` for the whole
    generation run -- the same class `portolan.llm.factory.get_llm()`
    would construct for `llm_provider="openrouter"`, with the API key and
    provider-order this project already reads from `.env`
    (`portolan.config.Settings.openrouter_api_key`/
    `openrouter_provider_order`).

    Deliberately does NOT go through `get_llm()` itself: that factory's
    default sink (`portolan.llm.factory.PostgresUsageSink`) writes every
    call into the `usage_events` table of whatever database
    `settings.database_url` points at, which would make `python -m
    corpus.generator.cli generate` -- a standalone script with no database
    of its own -- fail on a fresh checkout with no Postgres running, or
    silently mis-bill a customer database it has no business touching.
    This module's own `portolan.metering.InMemorySink` instead keeps every
    call's tokens/cost in memory, exactly like `portolan.bench.extraction`
    does for the same reason (see that module and `LLMClient.using_sink`) --
    the CLI reads `client.sink.events` back after generation to report and
    enforce `CORPUS_GENERATION_COST_LIMIT_USD`.
    """
    from portolan.llm.openrouter import OpenRouterClient
    from portolan.metering import InMemorySink

    api_key = _env_or_dotenv(project_root, "OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "prose.build_openrouter_client: OPENROUTER_API_KEY not set in the environment or .env"
        )
    provider_order_raw = _env_or_dotenv(project_root, "OPENROUTER_PROVIDER_ORDER")
    provider_order = None
    if provider_order_raw:
        import json as _json

        try:
            provider_order = _json.loads(provider_order_raw)
        except ValueError:
            provider_order = None
    return OpenRouterClient(
        spend_key="synthetic",
        api_key=api_key,
        default_model=openrouter_writer_model(),
        provider_order=provider_order,
        sink=InMemorySink(),
        tenant="corpus-generator",
    )


def cumulative_cost_usd(client: Any) -> float:
    """Sum of `cost_estimate` (OpenRouter's own reported `usage.cost` --
    `portolan.llm.openrouter._MeteredOpenRouterContext` prefers it over the
    static pricing-table estimate whenever OpenRouter returned one) across
    every call `client` (an `OpenRouterClient` built by
    `build_openrouter_client`) has made so far this process."""
    return sum(e.cost_estimate or 0.0 for e in client.sink.events)


def _max_tokens_for_brief(brief: Brief) -> int:
    """Completion-token budget for one `generate_via_openrouter` call.

    Calls are the expensive, long-form case (target 12k-25k tokens of
    transcript per `corpus/truth-table.yaml`'s `length_tokens`/reduced-mode
    target); tickets and emails are always short (200-1,500 / 200-800
    tokens, per GENERATOR.md -- never a "reduced" mode for these two).
    30% headroom over the brief's own target covers marker glyphs and the
    natural variance of a model's own tokenizer versus this project's
    words/0.75 estimate, without leaving so much slack that a runaway
    completion goes unnoticed."""
    target_words = brief.extra.get("target_words")
    if target_words:
        target_tokens = round(target_words / 0.75)
        return min(40000, round(target_tokens * 1.3) + 1000)
    return 4096


def generate_via_openrouter(
    client: Any, brief: Brief, model: str | None = None, system: str | None = None
) -> str:
    """One call per source through `client` (an `OpenRouterClient`,
    normally from `build_openrouter_client`). Same marker contract as
    `generate_via_api`/`generate_via_agent_sdk`: same stable system prompt
    (or `system`, when the caller passes one — see `CHAT_SYSTEM_PROMPT`/
    `NOTE_SYSTEM_PROMPT` below, added 2026-09-19 for the Slack/Teams/CRM-
    note source kinds, whose "one short message" shape doesn't fit the
    call/ticket/email-specific rules in `STABLE_SYSTEM_PROMPT`), same user
    prompt built from the brief, returns the raw text with
    ⟦F01⟧...⟦/F01⟧ markers still in place for `markers.py` to process.
    `surface="chat"` for the same reason `generate_via_agent_sdk` uses it --
    this is corpus tooling, not the production pipeline, so which metering
    bucket it lands in doesn't affect a real customer bill.
    """
    result = client.complete(
        system=system or STABLE_SYSTEM_PROMPT,
        user=build_user_prompt(brief),
        model=model or client.default_model,
        surface="chat",
        max_tokens=_max_tokens_for_brief(brief),
    )
    return result.text or ""


# --------------------------------------------------------------------------
# Free-teacher rotation — added 2026-09-19 for the synthetic-data pipeline
# (build instructions item 2). `CORPUS_WRITER_MODEL`/`generate_via_
# openrouter` above are unchanged and still write the ORIGINAL
# `corpus/truth-table.yaml` corpus with one paid-but-cheap model
# (`generate` subcommand); everything below is used only by the `synth`
# subcommand, which must spend nothing (build instructions: "Free models
# only; if a request would cost money, do not make it").
# --------------------------------------------------------------------------

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"

# Preference order, not an allowlist -- `discover_free_models` only ever
# returns ids it actually saw in a live `GET /api/v1/models` response
# (build instructions: "Do not hardcode ids you have not seen in that
# response"). A family here is a *substring* to look for among the ids
# that already end in `:free`; live-checked against OpenRouter 2026-09-19,
# every one of these five families had at least one `:free` id ---
# `deepseek/deepseek-v4-flash-0731:free`, `z-ai/glm-5.2:free`,
# `nvidia/nemotron-3-ultra-550b-a55b:free`,
# `nvidia/nemotron-3-super-120b-a12b:free`, `qwen/qwen3.8-27b:free`,
# `google/gemma-4-31b-it:free` -- but this list is a *preference*, not a
# requirement: a future run where one of these has been retired still
# works, just rotates across whichever other free ids remain.
PREFERRED_FREE_TEACHER_FAMILIES: tuple[str, ...] = (
    "deepseek/deepseek-v4-flash",
    "z-ai/glm-5.2",
    "nvidia/nemotron-3-ultra",
    "nvidia/nemotron-3-super",
    "qwen/qwen3.8-27b",
    "google/gemma-4-31b",
)

# `discover_free_models` returns EVERY `:free` id (build instructions:
# rotate across "several teachers" -- not "every free model OpenRouter
# happens to host"). The CLI slices its result to this many, taking the
# first N in the preference order `discover_free_models` already applies
# -- exactly the preferred families when at least this many are present,
# padded with other free ids (still alphabetical, never fabricated) only
# when fewer are.
MAX_TEACHER_ROTATION_SIZE = 6


def discover_free_models(
    api_key: str | None = None,
    *,
    transport: Any | None = None,
    timeout: float = 30.0,
) -> list[str]:
    """`GET /api/v1/models`, filtered to ids ending in `:free`, ordered
    with `PREFERRED_FREE_TEACHER_FAMILIES` first (in that order, one id per
    family -- the first `:free` id containing that family's substring) and
    every other free id after, alphabetically. Never raises on a network
    failure or an unexpected shape -- returns `[]`, which the CLI treats as
    "nothing to rotate across" and reports rather than crashes on.

    `transport` (an `httpx.BaseTransport`, typically `httpx.MockTransport`
    in tests) is the same test seam `portolan.llm.openrouter` uses -- no
    real network call in any test.
    """
    import httpx

    try:
        http_client = httpx.Client(transport=transport, timeout=timeout)
        try:
            headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
            response = http_client.get(OPENROUTER_MODELS_URL, headers=headers)
        finally:
            http_client.close()
        if response.status_code >= 400:
            return []
        payload = response.json()
    except Exception:  # noqa: BLE001 - best-effort discovery, never fatal
        return []

    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return []
    free_ids = sorted(
        {
            row["id"]
            for row in data
            if isinstance(row, dict)
            and isinstance(row.get("id"), str)
            and row["id"].endswith(":free")
        }
    )

    ordered: list[str] = []
    seen: set[str] = set()
    for family in PREFERRED_FREE_TEACHER_FAMILIES:
        for model_id in free_ids:
            if family in model_id and model_id not in seen:
                ordered.append(model_id)
                seen.add(model_id)
                break  # one id per preferred family
    for model_id in free_ids:
        if model_id not in seen:
            ordered.append(model_id)
            seen.add(model_id)
    return ordered


OPENROUTER_KEY_URL = "https://openrouter.ai/api/v1/key"


def free_model_daily_requests_remaining(
    api_key: str,
    *,
    transport: Any | None = None,
    timeout: float = 15.0,
) -> int | None:
    """`GET /api/v1/key` (bearer auth), read down to the free-model daily
    request budget for this key. Per OpenRouter's own published API
    reference for this endpoint (checked 2026-09-21 -- no live account was
    queried, this key was not available in this environment; re-verify
    against a real response if these names ever look wrong), the response
    shape is:

        {"data": {
            "label": str, "limit": float | None, "limit_reset": str | None,
            "limit_remaining": float | None,       # PAID credit balance, NOT this
            "usage": float, "usage_daily": float,   # PAID credit spend, NOT this
            "usage_weekly": float, "usage_monthly": float,
            "is_free_tier": bool,
            "free_model_daily_requests": {          # <-- what we want
                "used": int, "limit": int, "remaining": int
            },
        }}

    Returns `data.free_model_daily_requests.remaining` -- the count of
    `:free`-model requests left today for this key -- or `None` on ANY
    failure (bad status, network error, missing/malformed field): a
    best-effort check, never fatal, same contract as `discover_free_
    models`. The caller (`scripts/data/synth_batches.py`) treats `None` as
    "couldn't check" and logs-and-continues rather than blocking a run.

    `transport` is the same `httpx.BaseTransport` test seam `discover_
    free_models` uses -- no real network call in any test.
    """
    import httpx

    try:
        http_client = httpx.Client(transport=transport, timeout=timeout)
        try:
            response = http_client.get(
                OPENROUTER_KEY_URL, headers={"Authorization": f"Bearer {api_key}"}
            )
        finally:
            http_client.close()
        if response.status_code >= 400:
            return None
        payload = response.json()
        remaining = payload["data"]["free_model_daily_requests"]["remaining"]
        return int(remaining)
    except Exception:  # noqa: BLE001 - best-effort check, never fatal
        return None


class DailyLimitExceeded(RuntimeError):
    """Raised by `TokenBucketLimiter.acquire` once the day's request count
    has been used up -- the CLI catches this to stop a synth run cleanly
    and report rather than spin forever waiting for a window that won't
    reopen for hours."""


# --------------------------------------------------------------------------
# Paid teachers -- added 2026-09-22 so `synth` can be pointed at exact,
# non-`:free` OpenRouter ids instead of waiting on the account-wide
# 1,000-requests/day free tier. `--teachers` (cli.py's `synth` subcommand
# and `scripts/data/synth_batches.py`) bypasses `discover_free_models`
# entirely and rotates across exactly the given ids via the SAME
# `TeacherRotation`/`TokenBucketLimiter`/`generate_via_free_teacher`
# machinery above -- neither cares whether an id is free or paid, only
# `synth()`'s caller does (budget tracking, rpm, which checks to skip).
# --------------------------------------------------------------------------

# Every stop message `PaidBudgetExceeded` raises contains this exact
# substring -- callers (`cli.py`'s `synth` CLI branch,
# `scripts/data/synth_batches.py`'s `run_batches`) check for it in a
# `synth_report.json`'s `stop_reason` to tell "stopped because the paid
# dollar budget ran out" apart from every other `stopped_early` reason
# (daily free-tier budget, a hard failure) and map ONLY this one to exit
# code 4 -- the same stop_reason-text-matching convention `synth_batches.
# py`'s `_stopped_for_daily_budget` already uses for the free-tier case.
PAID_BUDGET_STOP_MARKER = "paid-teacher budget-usd ceiling reached"


class PaidBudgetExceeded(DailyLimitExceeded):
    """Raised (from inside `cli.py`'s `_synth_generate_unit`) the moment a
    paid-teacher run's shared `portolan.training.spend.SpendMeter` shows
    `corpus_usd(corpus_key) >= budget_usd` for THIS run's own synthetic-
    corpus spend -- not the meter's shared `total_usd()` across every
    corpus, which a concurrent, unrelated run (e.g. silver-labeling)
    sharing the same `spend.json` file can inflate past the ceiling on its
    own (fixed 2026-09-22) -- checked before every live call, never
    mid-call. Subclasses `DailyLimitExceeded` on purpose: every existing
    `except prose.DailyLimitExceeded` in `cli.py`'s `_synth_process_*`
    functions already stops that kind's loop and records `state.
    stop_reason` cleanly without needing its own new except clause --
    `PAID_BUDGET_STOP_MARKER` in the message is what lets a caller tell
    the two apart afterwards."""


class MixedTeacherListError(ValueError):
    """Raised by `teachers_are_paid` when `--teachers` mixes free
    (':free'-suffixed) and paid ids -- they can never share one rotation
    (different budgets, different rpm, different cost semantics), so a
    mixed list is a configuration error, not something to silently
    partition."""


def teachers_are_paid(teachers: list[str]) -> bool:
    """`True` when every id in `teachers` is PAID (no ':free' suffix),
    `False` when every id is FREE (':free'-suffixed). `teachers` must be
    non-empty. Raises `MixedTeacherListError` when the list has both."""
    if not teachers:
        raise ValueError("--teachers list must not be empty")
    free = [t for t in teachers if t.endswith(":free")]
    paid = [t for t in teachers if not t.endswith(":free")]
    if free and paid:
        raise MixedTeacherListError(
            f"--teachers mixes free and paid ids -- free={free} paid={paid}; "
            "a rotation must be all-free or all-paid, never both"
        )
    return bool(paid)


@dataclass
class TokenBucketLimiter:
    """A minute window plus a day window over ONE shared budget (build
    instructions: "about 20 requests a minute and 1,000 a day across free
    models" -- shared across every teacher, not per-model). `acquire()`
    blocks (via `sleep`) until a slot in the current minute is free, and
    raises `DailyLimitExceeded` once the day's budget is gone. `clock`/
    `sleep` are injectable so a test can drive this with a fake clock and
    assert exact sleep calls, never a real wait.
    """

    requests_per_minute: int = 20
    requests_per_day: int = 1000
    clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep

    _minute_window_start: float = field(default=0.0, init=False)
    _minute_count: int = field(default=0, init=False)
    _day_window_start: float = field(default=0.0, init=False)
    _day_count: int = field(default=0, init=False)
    _started: bool = field(default=False, init=False)

    def _ensure_started(self) -> None:
        if not self._started:
            now = self.clock()
            self._minute_window_start = now
            self._day_window_start = now
            self._started = True

    def acquire(self) -> None:
        self._ensure_started()
        now = self.clock()

        if now - self._day_window_start >= 86400.0:
            self._day_window_start = now
            self._day_count = 0
        if self._day_count >= self.requests_per_day:
            raise DailyLimitExceeded(
                f"TokenBucketLimiter: {self.requests_per_day} requests/day budget used up"
            )

        if now - self._minute_window_start >= 60.0:
            self._minute_window_start = now
            self._minute_count = 0
        if self._minute_count >= self.requests_per_minute:
            wait_seconds = 60.0 - (now - self._minute_window_start)
            if wait_seconds > 0:
                self.sleep(wait_seconds)
            self._minute_window_start = self.clock()
            self._minute_count = 0

        self._minute_count += 1
        self._day_count += 1

    @property
    def day_count(self) -> int:
        return self._day_count


@dataclass
class TeacherRotation:
    """Round-robins across `teachers` (a non-empty list of free model ids).
    `next(exclude=...)` skips one model id when possible (used to retry a
    verification failure with a *different* teacher, build instructions
    item 2) -- falls back to repeating it only when `teachers` has exactly
    one entry. Tracks per-teacher request and 429/rate-limit counts for the
    run report.

    Per-model cooldowns (added 2026-09-21, coordinator heads-up on
    free-tier 429 saturation): `record_rate_limit` puts a model on a
    cooldown -- 60s, doubling on repeat failures, capped at 15 minutes
    (`BASE_COOLDOWN_SECONDS`/`MAX_COOLDOWN_SECONDS`) -- and `mark_success`
    clears it. `next_available`/`wait_for_available` are cooldown-aware
    picks used by `generate_via_free_teacher`; plain `next()` is unchanged
    (blind round robin, no cooldown awareness) and kept for existing
    callers/tests. `clock`/`sleep` are injectable (same convention as
    `TokenBucketLimiter`) so a test can drive the wait with a fake clock.
    """

    BASE_COOLDOWN_SECONDS = 30.0
    MAX_COOLDOWN_SECONDS = 900.0

    teachers: list[str]
    clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep
    _index: int = field(default=0, init=False)
    request_counts: dict[str, int] = field(default_factory=dict, init=False)
    rate_limit_counts: dict[str, int] = field(default_factory=dict, init=False)
    _cooldown_until: dict[str, float] = field(default_factory=dict, init=False)
    _cooldown_duration: dict[str, float] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if not self.teachers:
            raise ValueError("TeacherRotation needs at least one teacher model id")
        for t in self.teachers:
            self.request_counts.setdefault(t, 0)
            self.rate_limit_counts.setdefault(t, 0)

    def next(self, exclude: str | None = None) -> str:
        n = len(self.teachers)
        for _ in range(n):
            candidate = self.teachers[self._index % n]
            self._index += 1
            if candidate != exclude or n == 1:
                self.request_counts[candidate] += 1
                return candidate
        candidate = self.teachers[self._index % n]
        self._index += 1
        self.request_counts[candidate] += 1
        return candidate

    def is_cooling(self, model: str) -> bool:
        return self._cooldown_until.get(model, 0.0) > self.clock()

    def cooldown_remaining(self, model: str) -> float:
        return max(0.0, self._cooldown_until.get(model, 0.0) - self.clock())

    def record_rate_limit(self, model: str, *, retry_after: float | None = None) -> None:
        """Count a 429/5xx against `model` and put it on a cooldown.
        `retry_after` (seconds), when the caller has one -- e.g. an
        `X-RateLimit-Reset`/`Retry-After` header value -- is honored
        directly instead of the doubling schedule; `generate_via_free_
        teacher` cannot supply one today (`portolan.llm.openrouter`'s
        `OpenRouterClient` does not surface response headers past its own
        internal same-model retry loop -- see that function's docstring),
        so in practice every call here uses the doubling default."""
        self.rate_limit_counts[model] = self.rate_limit_counts.get(model, 0) + 1
        if retry_after is not None:
            duration = min(self.MAX_COOLDOWN_SECONDS, max(0.0, retry_after))
        else:
            duration = min(
                self.MAX_COOLDOWN_SECONDS,
                self._cooldown_duration.get(model, self.BASE_COOLDOWN_SECONDS) * 2,
            )
        self._cooldown_duration[model] = duration
        self._cooldown_until[model] = self.clock() + duration

    def mark_success(self, model: str) -> None:
        self._cooldown_duration.pop(model, None)
        self._cooldown_until.pop(model, None)

    def next_available(self, exclude: str | None = None) -> str | None:
        """One round-robin pass for a model that is neither `exclude` nor
        currently cooling down. Returns `None` (state NOT advanced) when
        every model is excluded/cooling -- the caller decides whether to
        wait (`wait_for_available`) or give up."""
        n = len(self.teachers)
        for offset in range(n):
            idx = (self._index + offset) % n
            candidate = self.teachers[idx]
            if candidate != exclude and not self.is_cooling(candidate):
                self._index = idx + 1
                self.request_counts[candidate] += 1
                return candidate
        return None

    def wait_for_available(self, exclude: str | None = None) -> str:
        """`next_available`, but when every model is excluded/cooling,
        sleep until the EARLIEST cooldown expires and try again -- never
        skip a document ahead of that wait (coordinator heads-up,
        2026-09-21: "only skip a document when every model is cooling,
        and then sleep until the earliest cooldown expires rather than
        skipping ahead")."""
        while True:
            model = self.next_available(exclude)
            if model is not None:
                return model
            now = self.clock()
            candidates = [m for m in self.teachers if m != exclude] or list(self.teachers)
            remaining = [max(0.0, self._cooldown_until.get(m, now) - now) for m in candidates]
            self.sleep(min(remaining) if remaining else 0.0)

    @property
    def total_requests(self) -> int:
        return sum(self.request_counts.values())

    @property
    def total_rate_limits(self) -> int:
        return sum(self.rate_limit_counts.values())


_RETRYABLE_STATUS_RE = re.compile(r"status (\d{3})")
_RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})


def _is_retryable_status_error(exc: Exception) -> bool:
    """`True` for an `LLMError` whose message names a 429 or 5xx status --
    `OpenRouterClient._raise_for_status`'s message format is
    `"...request failed with status {code} (model=...): ..."`. Anything
    else (400/401/404/etc, or a message this doesn't recognize) is a real
    failure, not a rate/availability signal, and should propagate."""
    m = _RETRYABLE_STATUS_RE.search(str(exc))
    return bool(m) and int(m.group(1)) in _RETRYABLE_STATUSES


def generate_via_free_teacher(
    client: Any,
    brief: Brief,
    rotation: TeacherRotation,
    limiter: TokenBucketLimiter,
    *,
    system: str | None = None,
    exclude_model: str | None = None,
) -> tuple[str, str]:
    """Get a response from a free teacher, retrying across different
    models on a 429/upstream 5xx rather than giving up (coordinator
    heads-up, 2026-09-21, on free-tier saturation): picks a model via
    `rotation.wait_for_available` (skipping `exclude_model` for the FIRST
    pick only -- the "retry a verification failure with a different
    teacher" path callers already use), waits for a rate-limiter slot,
    makes the call. On a retryable `LLMError` (`_is_retryable_status_error`
    -- 429 or 5xx; `OpenRouterClient`'s own internal same-model retry
    budget, `_MAX_RETRIES=5`, has already been exhausted before this ever
    surfaces), the failing model is put on a cooldown
    (`rotation.record_rate_limit`) and the NEXT available model is tried
    immediately; when every model is cooling, `wait_for_available` sleeps
    until the earliest cooldown expires rather than skip the document
    ahead of that wait. A non-retryable `LLMError` (or `DailyLimitExceeded`
    from `limiter.acquire()`) propagates immediately, uncaught. Returns
    `(raw_text, model_used)`; marks that model's cooldown state cleared on
    success (`rotation.mark_success`).

    `OpenRouterClient` does not surface `Retry-After`/`X-RateLimit-*`
    header values to callers (`portolan.llm.base.LLMError` carries only a
    message string), so this cannot pass a `retry_after` into
    `record_rate_limit` today -- every cooldown here uses the 60s-doubling
    default. `TeacherRotation.record_rate_limit` still accepts one for a
    caller that has it (see that method's docstring).
    """
    from portolan.llm.base import LLMError

    first_exclude = exclude_model
    while True:
        model = rotation.wait_for_available(exclude=first_exclude)
        first_exclude = None  # only the caller's own exclusion applies to the first pick
        limiter.acquire()
        try:
            text = generate_via_openrouter(client, brief, model=model, system=system)
        except LLMError as exc:
            if _is_retryable_status_error(exc):
                rotation.record_rate_limit(model)
                continue
            raise
        rotation.mark_success(model)
        return text, model


CHAT_SYSTEM_PROMPT = """You are generating ONE short chat message for a synthetic corpus that \
tests a customer-intelligence extraction pipeline (Slack or Microsoft Teams, an account channel \
between a hospital customer and a vendor's support/sales team). You will be given a brief: who is \
speaking, the date, and (if any) one fact to plant.

Rules, followed exactly:
1. Write ONLY the message text for the named speaker -- one to three short sentences, casual chat \
register (lowercase is fine, contractions, no greeting, no sign-off, no email-style header).
2. If a fact is listed, plant it in this message, wrapped in markers: ⟦F01⟧ that sentence. ⟦/F01⟧ \
-- using that fact's exact id, honoring its `phrasing` constraint exactly, spoken in the voice of \
the named speaker. Every fact listed MUST have its marker pair somewhere in your reply.
3. Never write any word or phrase in that fact's `forbidden` list, in any form.
4. Do not invent additional facts. Reply with ONLY the message text -- no quotes, no markdown, no \
speaker name prefix (that is added separately).
"""

NOTE_SYSTEM_PROMPT = """You are generating ONE short CRM activity note for a synthetic corpus that \
tests a customer-intelligence extraction pipeline -- a vendor rep's free-text log of a customer \
touchpoint (a call recap, a site-visit note, a follow-up). You will be given a brief: who logged \
it, the date, and (if any) one fact to plant.

Rules, followed exactly:
1. Write 2-4 short sentences in a rep's own internal note register -- terse, third person or \
first person, no greeting, no sign-off.
2. If a fact is listed, plant it in this note, wrapped in markers: ⟦F01⟧ that sentence. ⟦/F01⟧ -- \
using that fact's exact id, honoring its `phrasing` constraint exactly. Every fact listed MUST \
have its marker pair somewhere in your reply.
3. Never write any word or phrase in that fact's `forbidden` list, in any form.
4. Do not invent additional facts. Reply with ONLY the note text -- no quotes, no markdown, no \
subject line (that is added separately).
"""


# --------------------------------------------------------------------------
# Manual mode: staged hand-written content
# --------------------------------------------------------------------------


def staging_dir(project_root: Path) -> Path:
    return project_root / "corpus" / "generator" / "_staging"


def staged_path(project_root: Path, brief: Brief) -> Path:
    subdir = {"call": "calls", "ticket": "tickets", "email": "email"}[brief.source_type]
    ext = "json" if brief.source_type == "ticket" else "txt"
    return staging_dir(project_root) / subdir / f"{brief.source_id}.{ext}"


def read_staged(project_root: Path, brief: Brief) -> str | None:
    path = staged_path(project_root, brief)
    if path.exists():
        return path.read_text(encoding="utf-8")
    return None


# --------------------------------------------------------------------------
# Manual mode: procedural filler (noise + no-fact distractors)
# --------------------------------------------------------------------------

_SMALL_TALK = [
    "How's the week treating you over there?",
    "Did you all get through the holiday weekend okay?",
    "I heard it's been busy on your end lately.",
    "Thanks for making time for this today.",
    "Let me pull up your account here real quick.",
    "Appreciate you hopping on with me.",
]

_FILLER_TOPIC_LINES = [
    "Nothing major to flag from our side this quarter.",
    "Things have been pretty steady, honestly.",
    "We haven't run into anything out of the ordinary.",
    "The team's been happy with how things are running.",
    "I just wanted to check in and make sure nothing's fallen through the cracks.",
    "No open issues that I'm tracking right now.",
    "Everything's been humming along fine on our end.",
    "We're in good shape, nothing urgent.",
]

_SCHEDULING_LINES = [
    "Let's find a date that works for the training session.",
    "I can send over a couple of time slots for next week.",
    "Does a Tuesday or Wednesday morning work better for your team?",
    "We'll want the ICU staff and a couple of the float pool nurses there.",
    "I'll follow up with a calendar invite once we lock in a date.",
    "Should we plan for an hour, or do you want the full ninety minutes?",
]

_CLOSINGS = [
    "Sounds good — I'll follow up over email with the details.",
    "Great, talk soon. Thanks again for the time.",
    "Perfect, I'll get that sent over today.",
    "Appreciate it. Have a good rest of your week.",
]


def _wrap_line(speaker: str, text: str) -> str:
    return f"{speaker}: {text}"


def generate_filler_call(brief: Brief, project_root: Path | None = None) -> str:
    """Deterministic, seeded-random filler transcript for a noise call or a
    no-fact distractor call (D04). No facts, no markers — just plausible,
    varied small talk and shop talk paced at ~150 words/minute."""
    rng = random.Random(f"call:{brief.source_id}")
    names = [p.name for p in brief.participants] or ["Rep", "Customer"]
    target_words = brief.extra.get("target_words") or 2200
    is_training_scheduling = "training" in (brief.topic or "") or brief.distractor_id == "D04"

    lines: list[str] = []
    elapsed_seconds = 0.0
    words_so_far = 0

    def add(speaker: str, text: str) -> None:
        nonlocal elapsed_seconds, words_so_far
        lines.append(f"[{seconds_to_hhmmss(int(elapsed_seconds))}] {_wrap_line(speaker, text)}")
        word_count = len(text.split())
        elapsed_seconds += word_count / 150 * 60 + 1.5  # ~150 wpm plus a turn-taking pause
        words_so_far += word_count

    add(names[0], rng.choice(_SMALL_TALK))
    add(
        names[1 % len(names)], "Doing alright, thanks for asking. What did you want to cover today?"
    )

    bank = _SCHEDULING_LINES if is_training_scheduling else _FILLER_TOPIC_LINES
    idx = 0
    while words_so_far < target_words - 60:
        speaker = names[idx % len(names)]
        line = bank[idx % len(bank)]
        # vary repeated lines slightly so a long filler call doesn't loop verbatim
        if idx >= len(bank):
            line = f"{line} {rng.choice(_SMALL_TALK)}"
        add(speaker, line)
        idx += 1

    add(names[0], rng.choice(_CLOSINGS))
    add(names[1 % len(names)], "You too, take care.")
    return "\n".join(lines)


_TICKET_BODIES = {
    "Password reset for the HalvardCare portal": [
        "I'm locked out of the portal after too many failed logins. Can you reset it?",
        "Reset link sent to the account email on file. Let us know once you're back in.",
        "That worked, thanks — closing this out.",
    ],
    "How do I run a self-test on a Tidewater T2 pump?": [
        "New tech on our floor is asking how to run the built-in self-test. Where's that documented?",
        "It's in the quick-start card, section 3 — hold the check button for 3 seconds at power-on.",
        "Found it, thank you.",
    ],
}

_GENERIC_TICKET_BODIES = [
    "Wanted to flag this for tracking, nothing urgent.",
    "Following up — any update on this?",
    "Confirming this is resolved on our end, thanks for the quick turnaround.",
    "Checking with our biomed team and will circle back.",
    "That resolved it, appreciate the help.",
]


def generate_filler_ticket(brief: Brief) -> dict[str, Any]:
    """Deterministic filler ticket JSON (no facts, no markers)."""
    rng = random.Random(f"ticket:{brief.source_id}")
    subject = brief.topic or "General question"
    requester_name = brief.participants[0].name if brief.participants else "Customer"
    scripted = _TICKET_BODIES.get(subject)
    if scripted:
        bodies = scripted
    else:
        n = rng.randint(2, 3)
        bodies = [rng.choice(_GENERIC_TICKET_BODIES) for _ in range(n)]
    comments = [
        {"number": i + 1, "author": requester_name if i == 0 else "Halvard Support", "body": b}
        for i, b in enumerate(bodies)
    ]
    return {"subject": subject, "comments": comments}


_GENERIC_EMAIL_BODIES = [
    "Just checking in — nothing pressing on our end right now.",
    "Wanted to confirm the schedule below works for everyone.",
    "Attaching nothing new here, just closing the loop from our call.",
    "Let me know if anything changes on your end before then.",
    "Appreciate the quick turnaround as always.",
]


def generate_filler_email_body(brief: Brief) -> str:
    """Deterministic filler email body (no facts, no markers)."""
    rng = random.Random(f"email:{brief.source_id}")
    greeting_name = brief.participants[-1].name.split()[0] if brief.participants else "there"
    sender_name = brief.participants[0].name if brief.participants else "Halvard Medical"
    paragraphs = [
        f"Hi {greeting_name},",
        rng.choice(_GENERIC_EMAIL_BODIES),
        "Best,\n" + sender_name,
    ]
    return "\n\n".join(paragraphs)


def generate_email_file(
    brief: Brief, subject: str, body: str, from_addr: str, to_addrs: list[str]
) -> str:
    header = email_header_block(brief.source_id, brief.date, from_addr, to_addrs, subject)
    return f"{header}\n\n{body}\n"


# --------------------------------------------------------------------------
# Manual mode: procedural filler for the two new chat source kinds and for
# CRM activity notes (added 2026-09-19). Same rationale as the call/ticket/
# email fillers above: noise sources carry no facts, so realism matters far
# less than volume/variety, and deterministic seeded-random filler avoids
# spending free-tier OpenRouter budget on text nothing will ever score.
# --------------------------------------------------------------------------

_CHAT_CUSTOMER_LINES = [
    "hey, quick one when you get a sec",
    "anyone around? got a question about the portal",
    "following up on this from earlier",
    "no rush, just wanted to get this on your radar",
    "thanks for the quick turnaround last time by the way",
]

_CHAT_REP_LINES = [
    "on it, give me a few minutes",
    "got it, thanks for flagging",
    "looping in the right person on our side now",
    "makes sense, I'll check and get back to you today",
    "appreciate the heads up",
]


def generate_filler_chat_message(source_id: str, is_customer: bool) -> str:
    """Deterministic filler for one noise Slack/Teams message — no facts,
    no markers."""
    rng = random.Random(f"chat:{source_id}")
    bank = _CHAT_CUSTOMER_LINES if is_customer else _CHAT_REP_LINES
    return rng.choice(bank)


def generate_filler_chat_reply(source_id: str) -> str:
    """Deterministic filler for a thread reply under a fact-bearing chat
    root message — a short acknowledgement, never a new fact."""
    rng = random.Random(f"chat-reply:{source_id}")
    return rng.choice(
        [
            "thanks, appreciate it",
            "sounds good",
            "got it, thank you",
            "perfect, thanks for the quick reply",
            "great, will keep an eye out",
        ]
    )


_NOTE_OPENERS = [
    "Logged after today's touchpoint.",
    "Quick note from the call this morning.",
    "Following up after the site visit.",
    "Capturing this before I forget.",
]

_NOTE_CLOSERS = [
    "Nothing else to flag right now.",
    "Will keep monitoring and update if anything changes.",
    "No action needed on our side at this time.",
    "Filed for the account record.",
]


def generate_filler_note_body(source_id: str) -> str:
    """Deterministic filler for a noise CRM activity note — no facts, no
    markers."""
    rng = random.Random(f"note:{source_id}")
    return f"{rng.choice(_NOTE_OPENERS)} {rng.choice(_NOTE_CLOSERS)}"
