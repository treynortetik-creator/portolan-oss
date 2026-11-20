"""The one wrapper every model call goes through (portolan-spec-v2.md, section 11).

"Every model call goes through one wrapper writing tenant, surface, model,
tokens, cost." This module is that wrapper. Nothing in the ingest pipeline,
merge step, consolidation, chat, or reporting should call a model SDK
directly without going through `metered()` — that is how the weekly report's
usage rollup and the customer's bill both stay accurate.

The sink is a small Protocol on purpose: today it's in-memory (tests), a
JSONL file (local dev / bench runs), or Postgres (`PostgresSink`, polish
wave 2026-09-12 — added because a real end-to-end `portolan.pipeline.run`
was metering every call into an `InMemorySink` that got thrown away when
the process exited, leaving `usage_events` permanently empty). No caller
changes when a new sink is added — one more class with a `record()` method
is the whole contract.
"""

from __future__ import annotations

import json
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol

import psycopg
from psycopg.types.json import Jsonb

from portolan.pricing import UnknownModelError, estimate_cost

# Closed list per the build instructions. Keeping it a Literal (rather than
# a free string) means a typo in a call site is a type-checker error, not a
# silent new surface nobody budgets for.
Surface = Literal["extract", "merge", "consolidate", "audit", "chat", "report", "embed"]


def _utc_now_iso() -> str:
    """Absolute UTC timestamp, ISO 8601. Never a relative time — see CLAUDE.md
    style rule and the lint's "relative-time language" check in the spec."""
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class UsageEvent:
    """One row of the usage ledger. Immutable once recorded — a sink appends,
    it never edits, mirroring the "nothing is deleted" rule the rest of the
    graph follows."""

    tenant: str
    surface: Surface
    model_key: str
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    # None when `model_key` wasn't in `portolan.pricing.PRICING` at record
    # time (review finding F3) -- `extra["pricing_error"]` carries why.
    cost_estimate: float | None
    request_id: str
    timestamp: str
    extra: dict[str, Any] = field(default_factory=dict)


class UsageSink(Protocol):
    """Anything that can durably record a UsageEvent. Implement this once for
    Postgres and every caller of `metered()` gets it for free."""

    def record(self, event: UsageEvent) -> None: ...


class InMemorySink:
    """Collects events in a list. Used by tests and by short-lived scripts
    that want to inspect what they spent before the process exits."""

    def __init__(self) -> None:
        self.events: list[UsageEvent] = []

    def record(self, event: UsageEvent) -> None:
        self.events.append(event)


class JsonlSink:
    """Appends one JSON object per line to a file. Good enough for local
    development and for the bench harness; not a substitute for the
    Postgres usage table in production (concurrent writers, querying by
    tenant/surface/day all want a real database)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def record(self, event: UsageEvent) -> None:
        # Open-append-close per event rather than holding a file handle:
        # this sink is meant for low-volume dev use, and keeping no state
        # between calls means multiple JsonlSink instances pointed at the
        # same path never fight over a shared handle.
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(event)) + "\n")


class PostgresSink:
    """Durable `UsageSink`: one `usage_events` row per event, over `conn`
    (`portolan/db/migrations/0001_init.sql`'s `usage_events` table).

    Added polish wave 2026-09-12 to close a real gap: `portolan.pipeline.
    run` metered every extraction/merge call through whatever `InMemorySink`
    the LLM client happened to default to, which lived and died with the
    process — a real end-to-end run made real model calls and left
    `usage_events` at zero rows. `run()` now defaults to `PostgresSink(conn)`
    when no sink is given explicitly (every test still passes its own,
    in-memory, exactly as before).

    Also keeps an in-memory `events` list, the same shape `InMemorySink`
    exposes — `portolan.pipeline._sink_events` and `RunReport.
    usage_by_surface` both read a sink's `.events` attribute generically,
    so this is a drop-in for either, not a second, differently-shaped
    sink; the durable write is a side effect on top; `events` is never
    read back from the database to reconstruct it.

    `record()` is guarded by its own lock: `run()`'s extraction fan-out
    (spec sec 5.1, "extraction fans out") can end up sharing one
    `PostgresSink` across several worker threads (any `LLMClient` besides
    `AgentSDKClient` — see `portolan.pipeline.run`'s own docstring on why
    that one gets a fresh per-worker client instead), and a single
    `psycopg.Connection` is not safe for concurrent `execute()` calls from
    more than one thread at once. The lock serializes only the write
    itself (one small INSERT + commit), never the model call around it.
    """

    def __init__(self, conn: psycopg.Connection) -> None:
        self.conn = conn
        self.events: list[UsageEvent] = []
        self._lock = threading.Lock()

    def record(self, event: UsageEvent) -> None:
        with self._lock:
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO usage_events
                        (tenant, surface, model_key, input_tokens, output_tokens,
                         cached_tokens, cost_estimate, request_id, created, extra)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        event.tenant,
                        event.surface,
                        event.model_key,
                        event.input_tokens,
                        event.output_tokens,
                        event.cached_tokens,
                        event.cost_estimate,
                        event.request_id,
                        datetime.fromisoformat(event.timestamp),
                        Jsonb(event.extra),
                    ),
                )
            self.conn.commit()
            self.events.append(event)


@dataclass
class _MeteringHandle:
    """The mutable object `metered()` yields. Callers set token counts on it
    as they learn them from the model response; everything else is filled
    in when the context manager exits.

    `cost_override` (bug hunt 2026-09-18, finding 9): when a caller sets
    this to a non-`None` value, it wins over the plain `estimate_cost`
    calculation below -- the same escape hatch `portolan.llm.openrouter`'s
    and `portolan.llm.agent_sdk`'s own local `metered()`-equivalents
    already had to build for themselves (this module used to be outside
    their build's owned files), for a caller that needs to price
    something `estimate_cost`'s flat `(input, output, cached)` shape can't
    express -- e.g. `portolan.pricing.estimate_cost_with_cache`'s
    cache-WRITE premium tier, which `AnthropicAPIClient` now uses.
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


@contextmanager
def metered(
    sink: UsageSink,
    tenant: str,
    surface: Surface,
    model_key: str,
) -> Iterator[_MeteringHandle]:
    """Wrap one model call. Usage:

        with metered(sink, tenant, "extract", "nova-lite") as usage:
            response = call_the_model(...)
            usage.input_tokens = response.usage.input_tokens
            usage.output_tokens = response.usage.output_tokens

    The event is recorded exactly once, on exit, whether the block raised or
    not. On an exception, `extra["error"]` is set to `str(exc)` before the
    event is recorded and the exception is re-raised unchanged — metering a
    failed call is as important as metering a successful one, since a model
    that errors out after emitting a huge prompt still cost money.

    Review finding F3: computing cost used to happen inside this `finally`,
    unguarded. `estimate_cost` raises `UnknownModelError` for any
    `model_key` not in `portolan.pricing.PRICING` (which `infer_model_key`
    hands back unchanged for any model id outside its hint list) — so a
    *successful* call against an unrecognised model id used to lose its
    result entirely: the pricing error replaced whatever the caller
    returned (or, worse, replaced a real exception already in flight, two
    `__context__` levels deep), and nothing was ever recorded in `sink`.
    Cost is computed in its own `try/except` now: an unknown model records
    the event anyway, with `cost_estimate=None` and
    `extra["pricing_error"]` set, and the caller's own result or exception
    is preserved exactly as it was. Metering must never be the reason a
    successful call's result — or a real failure's exception — doesn't
    reach the caller.
    """
    handle = _MeteringHandle(
        tenant=tenant,
        surface=surface,
        model_key=model_key,
        request_id=f"req_{uuid.uuid4().hex}",
    )
    try:
        yield handle
    except Exception as exc:
        handle.extra = {**handle.extra, "error": str(exc)}
        raise
    finally:
        if handle.cost_override is not None:
            cost = handle.cost_override
        else:
            try:
                cost = estimate_cost(
                    handle.model_key,
                    handle.input_tokens,
                    handle.output_tokens,
                    handle.cached_tokens,
                )
            except UnknownModelError as exc:
                cost = None
                handle.extra = {**handle.extra, "pricing_error": str(exc)}

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
        sink.record(event)
