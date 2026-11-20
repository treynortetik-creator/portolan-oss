"""The pipeline runner (build-contracts.md "Pipeline runner";
portolan-spec-v2.md sec 5.1, sec 6.2).

    intake -> filter -> extract -> resolve -> merge -> write -> index

`run()` drives every `SourceRef` a `Connector` lists through: fetch -> intake
-> prepare -> filter -> (if included) segment -> extract each segment ->
merge the segments into one `ExtractionResult` -> quote gate ->
`portolan.merge.merge_extraction` -> mark the source `done`/`partial` ->
update the coverage ledger. Each stage wrapped in `_retry_stage` is tried up
to three times with exponential backoff; one that still fails writes a
`dead_letter` row, marks the source `failed` with `last_error`, and the run
moves on (spec sec 6.2). The writes and code between those stages (the
filter decision, building envelopes, the quote gate, marking status) are
not retried, but a failure there dead-letters the source the same way
(`tests/test_pipeline_dead_letter.py`).

`redrive()` re-runs every source with a not-yet-cleared `dead_letter` row
through the same per-source pipeline.

CLI:

    python -m portolan.pipeline --connector file --path corpus/out \\
        [--llm fake|agent_sdk|bedrock|anthropic|openrouter] [--embed fake|fastembed] \\
        [--limit N] [--since YYYY-MM-DD] [--dry-run] [--force] [--samples N]
    python -m portolan.pipeline redrive [--connector file --path corpus/out] \\
        [--llm ...] [--embed fake|fastembed] [--samples N]

**Extraction fan-out** (spec sec 5.1: "Extraction fans out; merge and write
are single-writer"). `run()` processes sources in windows of
`PORTOLAN_EXTRACT_CONCURRENCY` (default 4, read from the environment). Within
a window, `_prepare_one` (writes to `conn`) runs one source at a time, in
order, on the main thread; `_extract_segments` (no `conn` access) runs on a
`ThreadPoolExecutor`; `_finish_one` (writes to `conn`) then runs each result
in the window's original order on the main thread. Only extraction overlaps.

An `AgentSDKClient` or `OpenRouterClient` is cloned once per worker thread
(`_worker_backend`), each clone with a private sink replayed into the run's
sink after the pool finishes; any other `LLMClient` is shared across workers.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import psycopg
import yaml
from psycopg.types.json import Jsonb
from pydantic import BaseModel, Field

from portolan import db
from portolan.config import Settings, get_settings, secret_value
from portolan.connectors.base import Connector, SourceRef
from portolan.context import build_account_context, load_company_context
from portolan.crm.changes import refresh_crm_changes
from portolan.crm.snapshots import record_snapshot
from portolan.extract.backends import Backend, LLMClientBackend
from portolan.extract.prompt import DEFAULT_PROMPT_VARIANT, build_system_prompt
from portolan.extract.quote_gate import quote_is_supported
from portolan.extract.schema import (
    QUOTE_REQUIRED_TYPES,
    CustomerExtension,
    Fact,
    RegistryContextEntry,
    SegmentRef,
    SourceEnvelope,
    SourceType,
    Vocabulary,
    merge_customer_extension,
)
from portolan.extract.schema import (
    Participant as ExtractParticipant,
)
from portolan.extract.segment import merge_segment_results
from portolan.extract.segment import segment as segment_text
from portolan.ingest.filter import apply_filter_decision, customer_filter
from portolan.ingest.intake import intake
from portolan.ingest.ledger import record as record_coverage
from portolan.ingest.models import RawSource, SourceRow
from portolan.ingest.prepare import prepare_text
from portolan.lint.report import render_markdown, run_lint, write_changelog_row
from portolan.llm.base import LLMClient
from portolan.merge import merge_extraction
from portolan.metering import InMemorySink, PostgresSink, UsageSink
from portolan.notes import append_changelog
from portolan.registry import Registry
from portolan.search.embed import Embedder
from portolan.search.index import index_pending
from portolan.store import LocalStore, Store

logger = logging.getLogger(__name__)

# Extraction worker threads; read from the environment, not `Settings`.
EXTRACT_CONCURRENCY_ENV = "PORTOLAN_EXTRACT_CONCURRENCY"
DEFAULT_EXTRACT_CONCURRENCY = 4


def _extract_concurrency() -> int:
    raw = os.environ.get(EXTRACT_CONCURRENCY_ENV, "")
    try:
        value = int(raw)
    except ValueError:
        value = DEFAULT_EXTRACT_CONCURRENCY
    return max(1, value)


# The build_* and load_customer_extension entries are public so other passes build
# envelopes exactly as the pipeline does.
__all__ = [
    "RedriveReport",
    "RunReport",
    "build_envelope_participants",
    "build_envelopes",
    "build_registry_context",
    "load_customer_extension",
    "main",
    "redrive",
    "run",
]

_MAX_ATTEMPTS = 3
_BACKOFF_BASE_SECONDS = 0.05
_SAMPLE_RATE = 0.02


# --------------------------------------------------------------------------
# Reports
# --------------------------------------------------------------------------


class UsageTotals(BaseModel):
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    cost_estimate: float = 0.0


class RunReport(BaseModel):
    """build-contracts.md "Pipeline runner": "sources seen/filtered/
    extracted/partial/failed, facts extracted, merge counts (from
    MergeReports), tokens and cost by surface (from the metering sink),
    wall time, dead-letter count."""

    sources_seen: int = 0
    sources_filtered: int = 0
    sources_extracted: int = 0
    sources_partial: int = 0
    sources_failed: int = 0
    # Unchanged sources already extracted with this run's signature, or excluded
    # again for the same reason, are skipped (`--force` bypasses); counted here.
    sources_skipped: int = 0
    # `--dry-run` counts a source that would be extracted here instead of extracting it.
    sources_would_extract: int = 0
    facts_extracted: int = 0
    merge_counts: dict[str, int] = Field(default_factory=dict)
    usage_by_surface: dict[str, UsageTotals] = Field(default_factory=dict)
    # This run's metered events with no price (`extra["pricing_error"]`), which
    # `usage_by_surface` counts as $0; visible here so a low bill is not silent.
    usage_pricing_errors: int = 0
    wall_time_seconds: float = 0.0
    dead_letter_count: int = 0
    index_chunks_written: int = 0
    # CRM changes between snapshots, written as claims by code (`portolan.crm`).
    crm_changes_written: int = 0


class RedriveReport(BaseModel):
    attempted: int = 0
    succeeded: int = 0
    still_failing: int = 0


# --------------------------------------------------------------------------
# Internal per-source result
# --------------------------------------------------------------------------


@dataclass
class _SourceOutcome:
    # "failed" is produced only by `run()`'s loop, for a source whose stage raised
    # `_DeadLettered`; `redrive()` catches `_DeadLettered` directly.
    status: str  # "filtered" | "done" | "partial" | "failed" | "skipped"
    facts_count: int = 0
    merge_report: Any = None


class _DeadLettered(Exception):
    """Signals that a stage already dead-lettered and recorded this
    source's failure; the caller just counts it and moves on."""


@dataclass
class _PreparedSource:
    """What `_prepare_one` hands `_extract_segments`: everything needed to
    run the segment-extraction loop with no further `conn` access.

    `text` is the whole, unsegmented prepared text the envelopes were built
    from; the quote gate checks against it because a merged fact's quote may
    come from any segment."""

    ref: SourceRef
    raw: RawSource
    source_row: SourceRow
    envelopes: list[SourceEnvelope]
    text: str


@dataclass
class _ExtractedSource:
    """What `_extract_segments` hands `_finish_one`: the prepared source plus
    the extraction results.

    `sink` is the per-source `InMemorySink` passed to every
    `backend.extract()` call. A client with `using_sink()` meters the call
    into it (not its own `.sink`), so `_finish_one` must replay it into the
    run's sink or that extraction usage is lost.

    `results` is one entry per extraction pass (`--samples N`), each the
    per-envelope `ExtractionResult` list that pass produced. Every call meters
    its own event, so N passes bill N times.
    """

    prepared: _PreparedSource
    results: list[list[Any]]  # list[pass] of list[ExtractionResult]
    segment_failures: list[str]
    sink: InMemorySink


# --------------------------------------------------------------------------
# Retry
# --------------------------------------------------------------------------


def _retry_stage(
    fn: Callable[[], Any],
    *,
    max_attempts: int = _MAX_ATTEMPTS,
    backoff_seconds: float = _BACKOFF_BASE_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> Any:
    """Call `fn()`, retrying on any exception up to `max_attempts` times
    total with exponential backoff between attempts. Re-raises the last
    exception if every attempt fails."""
    last_exc: BaseException | None = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - re-raised below once retries are exhausted
            last_exc = exc
            if attempt < max_attempts - 1:
                sleep(backoff_seconds * (2**attempt))
    assert last_exc is not None
    raise last_exc


# --------------------------------------------------------------------------
# Dead letter + source status
# --------------------------------------------------------------------------


def _write_dead_letter(
    conn: psycopg.Connection,
    *,
    source_id: str,
    stage: str,
    attempt: int,
    error: str,
    payload: dict[str, Any] | None = None,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO dead_letter (source_id, stage, attempt, error, payload) "
            "VALUES (%s, %s, %s, %s, %s)",
            (source_id, stage, attempt, error, Jsonb(payload or {})),
        )
    conn.commit()


def _clear_dead_letters(conn: psycopg.Connection, source_id: str) -> None:
    """Close every pending dead-letter row for `source_id` once the source
    has reached an outcome again, by redrive or by a later run: /coverage
    lists pending rows as not ingested, and redrive re-runs them."""
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE dead_letter SET redriven_at = now() "
            "WHERE source_id = %s AND redriven_at IS NULL",
            (source_id,),
        )
    conn.commit()


def _mark_source_status(
    conn: psycopg.Connection,
    slug: str,
    status: str,
    *,
    last_error: str | None = None,
    extraction_model: str | None = None,
    extraction_prompt_variant: str | None = None,
    extraction_samples: int | None = None,
) -> None:
    """`extraction_model`/`extraction_prompt_variant`/`extraction_samples` are
    recorded only when passed (`_finish_one`, on a source that reached merge);
    `COALESCE` keeps the stored value otherwise (a `failed` mark from
    `_fail_source`) rather than overwriting it with `NULL`.
    """
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE sources SET extraction_status = %s, last_error = %s, "
            "attempts = attempts + 1, "
            "extraction_model = COALESCE(%s, extraction_model), "
            "extraction_prompt_variant = COALESCE(%s, extraction_prompt_variant), "
            "extraction_samples = COALESCE(%s, extraction_samples) "
            "WHERE slug = %s",
            (
                status,
                last_error,
                extraction_model,
                extraction_prompt_variant,
                extraction_samples,
                slug,
            ),
        )
    conn.commit()


def _fail_source(
    conn: psycopg.Connection,
    *,
    source_id: str,
    source_type: str,
    day: date,
    stage: str,
    error: BaseException,
    sources_slug: str | None = None,
    seen: int = 1,
    attempt: int = _MAX_ATTEMPTS,
) -> None:
    """`attempt`: how many tries the stage had; 1 for a stage `_retry_stage`
    does not wrap.

    A connection left in a failed transaction by `error` is rolled back
    first, so the dead letter can still be written.

    `seen`: 1 when this failure ends the source's processing before the
    ledger counted it as seen (fetch through filter), 0 after (extract,
    merge), so each processed source counts as seen exactly once.

    `source_id` is the connector-scoped id `dead_letter` is keyed by, which
    `redrive()` looks up in `connector.list_sources`; `dead_letter` has no FK
    into `sources` and must not follow source versioning.

    `sources_slug` is the `sources.slug` to mark `failed`: `source_row.slug`
    once `intake()` returned one (`<source_id>--v<n>` for a changed,
    re-crawled source; provenance-chain.md "Source versioning"). Defaults to
    `source_id` for a fetch/intake failure, before any row exists.
    """
    if conn.info.transaction_status == psycopg.pq.TransactionStatus.INERROR:
        conn.rollback()
    message = str(error)
    _write_dead_letter(conn, source_id=source_id, stage=stage, attempt=attempt, error=message)
    _mark_source_status(conn, sources_slug or source_id, "failed", last_error=f"{stage}: {message}")
    record_coverage(conn, source_type, day, seen=seen, failed=1)


# --------------------------------------------------------------------------
# Customer extension / vocabulary
# --------------------------------------------------------------------------


def load_customer_extension(connector: Connector) -> CustomerExtension | None:
    """`customer.yaml` next to the corpus a `FileConnector` reads (a
    sibling of the truth table, one level above its `root`) -- optional,
    never required. A connector with no `.root` (a real vendor connector,
    not a file-based one) simply has no such file to find."""
    root = getattr(connector, "root", None)
    if root is None:
        return None
    path = Path(root).parent / "customer.yaml"
    if not path.is_file():
        return None
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return CustomerExtension.model_validate(data)


def _truth_table_path(connector: Connector) -> Path | None:
    """`<connector.root's parent>/truth-table.yaml` (for a `FileConnector` at
    `corpus/out`, `corpus/truth-table.yaml`), resolved like
    `load_customer_extension`'s `customer.yaml` and independent of
    `store_root`. `None` for a connector with no `.root` or no such file.
    """
    root = getattr(connector, "root", None)
    if root is None:
        return None
    candidate = Path(root).parent / "truth-table.yaml"
    return candidate if candidate.is_file() else None


def _maybe_import_registry(conn: psycopg.Connection, connector: Connector) -> None:
    """(a) "if the connector exposes CRM paths and the registry is empty,
    import it ... with the truth-table meta when [it] exists, else
    without"."""
    crm_paths_fn = getattr(connector, "crm_paths", None)
    if crm_paths_fn is None:
        return
    paths = crm_paths_fn()
    if paths is None:
        return

    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM registry_entities")
        (count,) = cur.fetchone()
    if count > 0:
        return

    products = competitors = internal_people = supplier = branches = None
    account_aliases: dict[str, list[str]] | None = None

    truth_table_path = _truth_table_path(connector)
    if truth_table_path is not None:
        data = yaml.safe_load(truth_table_path.read_text(encoding="utf-8")) or {}
        meta = data.get("meta") or {}
        products = meta.get("products")
        competitors = meta.get("competitors")
        internal_people = meta.get("internal_people")
        supplier = meta.get("supplier")
        branches = meta.get("branches")
        account_aliases = {}
        for account in data.get("accounts") or []:
            crm_id = account.get("crm_id")
            aliases = account.get("aliases")
            if crm_id and aliases:
                account_aliases[crm_id] = list(aliases)

    Registry().import_crm(
        conn,
        paths.accounts,
        paths.contacts,
        products=products,
        competitors=competitors,
        internal_people=internal_people,
        account_aliases=account_aliases or None,
        supplier=supplier,
        branches=branches,
    )
    conn.commit()


# --------------------------------------------------------------------------
# Envelope building
# --------------------------------------------------------------------------


def build_envelope_participants(raw: RawSource, registry: Registry) -> list[ExtractParticipant]:
    result: list[ExtractParticipant] = []
    for participant in raw.participants:
        registry_id = None
        if participant.email:
            entity = registry.by_email(participant.email)
            registry_id = entity.id if entity else None
        result.append(
            ExtractParticipant(
                name=participant.name, role=participant.role, registry_id=registry_id
            )
        )
    return result


def build_registry_context(
    registry: Registry, matched_entities: list[str]
) -> list[RegistryContextEntry]:
    """ "registry_context = the entities the filter matched plus
    ancestors". Ids that don't resolve against the registry (an unvalidated
    connector account hint, say) are skipped, not raised on.
    """
    seen: dict[str, RegistryContextEntry] = {}
    for entity_id in matched_entities:
        entity = registry.get(entity_id)
        if entity is None:
            continue
        ids_to_add = [entity_id, *registry.ancestors(entity_id)]
        for eid in ids_to_add:
            if eid in seen:
                continue
            ent = registry.get(eid)
            if ent is None:
                continue
            seen[eid] = RegistryContextEntry(
                name=ent.canonical_name, kind=ent.kind, registry_id=ent.id, aliases=ent.aliases
            )
    return list(seen.values())


def _token_estimate(text: str) -> int:
    return max(1, len(text) // 4)


def build_envelopes(
    raw: RawSource,
    prepared_text: str,
    *,
    participants: list[ExtractParticipant],
    registry_context: list[RegistryContextEntry],
    vocabulary: Vocabulary,
    account_context: dict[str, str] | None = None,
) -> list[SourceEnvelope]:
    common = dict(
        source_id=raw.source_id,
        source_type=raw.source_type,
        source_date=raw.source_date,
        participants=participants,
        registry_context=registry_context,
        vocabulary=vocabulary,
        account_context=account_context or {},
    )

    if _token_estimate(prepared_text) <= 4000:
        return [SourceEnvelope(**common, text=prepared_text, segment=None)]

    segments = segment_text(prepared_text, max_tokens=4000)
    return [
        SourceEnvelope(
            **common, text=seg.text, segment=SegmentRef(index=seg.index, total=seg.total)
        )
        for seg in segments
    ]


# --------------------------------------------------------------------------
# Metering rollup
# --------------------------------------------------------------------------


def _sink_events(llm: LLMClient) -> list[Any] | None:
    sink = getattr(llm, "sink", None)
    events = getattr(sink, "events", None)
    return events if isinstance(events, list) else None


def _usage_by_surface(events: list[Any]) -> dict[str, UsageTotals]:
    totals: dict[str, UsageTotals] = {}
    for event in events:
        bucket = totals.setdefault(event.surface, UsageTotals())
        bucket.calls += 1
        bucket.input_tokens += event.input_tokens
        bucket.output_tokens += event.output_tokens
        bucket.cached_tokens += event.cached_tokens
        # `cost_estimate` is None for an unpriced model; the event is still counted.
        bucket.cost_estimate += event.cost_estimate or 0.0
    return totals


def _accumulate_merge_counts(target: dict[str, int], report: Any) -> None:
    """Fold a `MergeReport` (or dict) into a flat `{key: count}` dict: every
    top-level int field by name, plus every nested dict's int values under
    `"<field>.<subkey>"` (e.g. `outcome_counts.new`). Booleans are skipped.
    """
    if report is None:
        return
    if hasattr(report, "model_dump"):
        data = report.model_dump()
    elif isinstance(report, dict):
        data = report
    else:
        return
    for key, value in data.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            target[key] = target.get(key, 0) + value
        elif isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, int) and not isinstance(sub_value, bool):
                    combined = f"{key}.{sub_key}"
                    target[combined] = target.get(combined, 0) + sub_value


# --------------------------------------------------------------------------
# Filter audit
# --------------------------------------------------------------------------


def _run_filter_audit(
    conn: psycopg.Connection,
    *,
    raw: RawSource,
    prepared,
    registry: Registry,
    decision,
    vocabulary: Vocabulary,
    backend: Backend,
    system_prompt: str,
    settings: Settings,
    sink: UsageSink,
    sleep: Callable[[float], None],
) -> None:
    """Run a sampled *excluded* source through the extractor anyway (spec sec
    5.2), so the filter's miss rate is measurable. The source stays excluded
    (no status change, no `merge_extraction`); its facts are logged as a
    `filter_audit` changelog row. A failure is logged, never dead-lettered.

    `sink` is the run's usage sink: `local_sink`'s events are replayed into it
    in a `finally`, so a billed call is recorded even when a later segment
    raises.
    """
    from portolan.metering import InMemorySink

    local_sink = InMemorySink()
    try:
        participants = build_envelope_participants(raw, registry)
        registry_context = build_registry_context(registry, decision.matched_entities)
        account_context = build_account_context(conn, registry_context)
        envelopes = build_envelopes(
            raw,
            prepared.text,
            participants=participants,
            registry_context=registry_context,
            vocabulary=vocabulary,
            account_context=account_context,
        )
        facts_found = 0
        fact_types: dict[str, int] = {}
        for envelope in envelopes:
            result = _retry_stage(
                lambda env=envelope: backend.extract(
                    env, system_prompt, local_sink, settings.tenant
                ),
                sleep=sleep,
            )
            facts_found += len(result.facts)
            for fact in result.facts:
                fact_types[fact.type] = fact_types.get(fact.type, 0) + 1
        append_changelog(
            conn,
            "filter_audit",
            f"sampled excluded source {raw.source_id}: {facts_found} fact(s) found by "
            f"extraction despite exclusion (reason: {decision.reason})",
            {
                "source_id": raw.source_id,
                "filter_reason": decision.reason,
                "facts_found": facts_found,
                "fact_types": fact_types,
            },
        )
    except Exception as exc:  # noqa: BLE001 - an audit measurement must never fail the source
        append_changelog(
            conn,
            "filter_audit",
            f"sampled excluded source {raw.source_id}: audit extraction failed: {exc}",
            {"source_id": raw.source_id, "filter_reason": decision.reason, "error": str(exc)},
        )
    finally:
        for event in local_sink.events:
            sink.record(event)


# --------------------------------------------------------------------------
# Per-source processing
# --------------------------------------------------------------------------


def _prepare_one(
    *,
    conn: psycopg.Connection,
    connector: Connector,
    ref: SourceRef,
    registry: Registry,
    store: Store,
    settings: Settings,
    backend: Backend,
    system_prompt: str,
    vocabulary: Vocabulary,
    sink: UsageSink,
    sleep: Callable[[float], None],
    dry_run: bool = False,
    force: bool = False,
    extraction_model: str | None = None,
    prompt_variant: str | None = None,
    samples: int = 1,
) -> _PreparedSource | _SourceOutcome:
    """fetch -> intake -> skip check -> prepare -> filter -> (if excluded and
    sampled, the filter audit) -> build envelopes.

    Every write here goes through `conn`, so this is never fanned out: `run()`
    calls it on the main thread, one source at a time, in order.

    Returns a `_PreparedSource` for an included source, `_SourceOutcome(
    "filtered")` for an excluded one, or `_SourceOutcome("skipped")` when,
    without `force`, the source is already `done`/`partial` with this run's
    `extraction_model`, `prompt_variant` and `samples`, or is unchanged and
    excluded again for the same reason. A dry run writes neither the filter
    decision nor the ledger and skips the filter audit (it calls the
    extractor). Raises `_DeadLettered` when fetch, intake, prepare or filter
    fails.

    A `None` `extraction_model` never matches a stored one, so omitting it
    disables skipping.
    """
    try:
        raw: RawSource = _retry_stage(lambda: connector.fetch(ref), sleep=sleep)
    except Exception as exc:
        # No `raw.source_date` exists yet: `ref.updated_at` is the only day.
        _fail_source(
            conn,
            source_id=ref.source_id,
            source_type=ref.source_type.value,
            day=ref.updated_at.date(),
            stage="fetch",
            error=exc,
        )
        raise _DeadLettered from exc

    # The ledger is keyed by the source's own date (`raw.source_date` from here
    # on). A source counts as `seen` once per processing that ends in an outcome;
    # not when skipped, and on a dry run only if it fails (`_fail_source` writes).
    try:
        source_row: SourceRow = _retry_stage(
            lambda: intake(raw, conn=conn, store=store), sleep=sleep
        )
    except Exception as exc:
        _fail_source(
            conn,
            source_id=raw.source_id,
            source_type=raw.source_type.value,
            day=raw.source_date,
            stage="intake",
            error=exc,
        )
        raise _DeadLettered from exc

    # Before the skip check, so a graph built before snapshots existed gets them on a re-run.
    if raw.source_type == SourceType.CRM and not dry_run:
        record_snapshot(conn, raw, source_row)

    # Unchanged content returns the same row from `intake()`; if it already
    # reached `done`/`partial` with this run's signature, re-extracting would
    # only re-bill the same result.
    if (
        not force
        and source_row.extraction_status in ("done", "partial")
        and extraction_model is not None
        and source_row.extraction_model == extraction_model
        and source_row.extraction_prompt_variant == prompt_variant
        and source_row.extraction_samples == samples
    ):
        if not dry_run:
            _clear_dead_letters(conn, raw.source_id)
        return _SourceOutcome(status="skipped")

    # From here on every write to `sources` uses `source_row.slug`, which intake
    # may have versioned to `<source_id>--v<n>`; `dead_letter` writes and clears
    # stay keyed by `raw.source_id`.
    try:
        prepared = _retry_stage(lambda: prepare_text(raw), sleep=sleep)
    except Exception as exc:
        _fail_source(
            conn,
            source_id=raw.source_id,
            source_type=raw.source_type.value,
            day=raw.source_date,
            stage="prepare",
            error=exc,
            sources_slug=source_row.slug,
        )
        raise _DeadLettered from exc

    try:
        decision = _retry_stage(
            lambda: customer_filter(
                raw, prepared, registry, sample_rate=_SAMPLE_RATE, seed=settings.tenant
            ),
            sleep=sleep,
        )
    except Exception as exc:
        _fail_source(
            conn,
            source_id=raw.source_id,
            source_type=raw.source_type.value,
            day=raw.source_date,
            stage="filter",
            error=exc,
            sources_slug=source_row.slug,
        )
        raise _DeadLettered from exc

    if (
        not force
        and not decision.include
        and source_row.extraction_status == "filtered"
        and source_row.filter_reason == decision.reason
    ):
        # Unchanged source, same exclusion as last time: already counted,
        # already audited if sampled (the audit is a paid extraction).
        if not dry_run:
            _clear_dead_letters(conn, raw.source_id)
        return _SourceOutcome(status="skipped")

    if not dry_run:
        try:
            apply_filter_decision(conn, source_row.slug, decision)
            if not decision.include:
                _clear_dead_letters(conn, raw.source_id)
            record_coverage(
                conn,
                raw.source_type.value,
                raw.source_date,
                seen=1,
                filtered={} if decision.include else {decision.reason: 1},
                sampled=1 if decision.sampled else 0,
            )
        except Exception as exc:
            _fail_source(
                conn,
                source_id=raw.source_id,
                source_type=raw.source_type.value,
                day=raw.source_date,
                stage="filter",
                error=exc,
                sources_slug=source_row.slug,
                attempt=1,
            )
            raise _DeadLettered from exc
    if not decision.include:
        if decision.sampled and not dry_run:
            # Spec sec 5.2: the sampled excluded sources still go through the
            # extractor, to measure the filter's miss rate. Never on a dry run.
            _run_filter_audit(
                conn,
                raw=raw,
                prepared=prepared,
                registry=registry,
                decision=decision,
                vocabulary=vocabulary,
                backend=backend,
                system_prompt=system_prompt,
                settings=settings,
                sink=sink,
                sleep=sleep,
            )
        return _SourceOutcome(status="filtered")

    try:
        participants = build_envelope_participants(raw, registry)
        registry_context = build_registry_context(registry, decision.matched_entities)
        account_context = build_account_context(conn, registry_context)
        envelopes = build_envelopes(
            raw,
            prepared.text,
            participants=participants,
            registry_context=registry_context,
            vocabulary=vocabulary,
            account_context=account_context,
        )
    except Exception as exc:
        # The ledger already counted this source as seen, except on a dry run.
        _fail_source(
            conn,
            source_id=raw.source_id,
            source_type=raw.source_type.value,
            day=raw.source_date,
            stage="envelope",
            error=exc,
            sources_slug=source_row.slug,
            seen=1 if dry_run else 0,
            attempt=1,
        )
        raise _DeadLettered from exc
    return _PreparedSource(
        ref=ref, raw=raw, source_row=source_row, envelopes=envelopes, text=prepared.text
    )


def _extract_segments(
    prepared: _PreparedSource,
    *,
    backend: Backend,
    system_prompt: str,
    settings: Settings,
    sleep: Callable[[float], None],
    samples: int = 1,
) -> _ExtractedSource:
    """One `backend.extract()` call per envelope, retried per the usual
    policy. Touches no `conn`, which is why `run()` may run it on a worker
    thread. Never raises: segment failures go to `segment_failures`, and
    `_finish_one` decides whether an empty result is fatal.

    Runs the per-envelope loop `max(1, samples)` independent times, one
    results list per pass; `_finish_one` unions the passes.
    """
    # One sink for the whole source, carried out on `_ExtractedSource.sink` so
    # `_finish_one` can replay it (see that class).
    sink = InMemorySink()
    all_pass_results: list[list[Any]] = []
    segment_failures: list[str] = []
    for _ in range(max(1, samples)):
        pass_results = []
        for envelope in prepared.envelopes:
            label = (
                f"{envelope.segment.index}/{envelope.segment.total}" if envelope.segment else "1/1"
            )
            try:
                result = _retry_stage(
                    lambda env=envelope: backend.extract(env, system_prompt, sink, settings.tenant),
                    sleep=sleep,
                )
                pass_results.append(result)
            except Exception as exc:  # noqa: BLE001 - recorded, not re-raised: other segments may still succeed
                segment_failures.append(f"segment {label}: {exc}")
        all_pass_results.append(pass_results)
    return _ExtractedSource(
        prepared=prepared, results=all_pass_results, segment_failures=segment_failures, sink=sink
    )


# --------------------------------------------------------------------------
# Mechanical quote gate (01-spec/extraction-final-2026-09-18.md)
#
# A quote-requiring fact whose quote is not in the source text is kept out of
# `merge_extraction` and parked in `review_queue` (`kind='quote_unsupported'`)
# for a human: facts are never dropped (migration 0001's `review_queue` comment).
# --------------------------------------------------------------------------


def _gate_quotes(facts: list[Fact], *, source_text: str) -> tuple[list[Fact], list[dict[str, Any]]]:
    """Split `facts` into `(kept, gated)`, keeping order. A fact whose `type` is
    not in `QUOTE_REQUIRED_TYPES` is always kept; a quote-requiring fact is
    kept only when `quote_gate.quote_is_supported` finds its quote in
    `source_text`.

    `gated` is one dict per dropped fact, shaped as `review_queue.payload
    ["fact"]`; the caller writes the row."""
    kept: list[Fact] = []
    gated: list[dict[str, Any]] = []
    for fact in facts:
        if fact.type not in QUOTE_REQUIRED_TYPES or quote_is_supported(fact.quote, source_text):
            kept.append(fact)
            continue
        gated.append(
            {
                "id": fact.id,
                "type": fact.type,
                "entities": list(fact.entities),
                "claim": fact.claim,
                "quote": fact.quote,
                "locator": fact.locator,
                "speaker": fact.speaker,
                "valid_from": fact.valid_from.isoformat() if fact.valid_from else None,
                "valid_to": fact.valid_to.isoformat() if fact.valid_to else None,
                "severity": fact.severity,
                "sentiment": fact.sentiment,
                "confidence": fact.confidence,
            }
        )
    return kept, gated


def _queue_quote_unsupported(
    conn: psycopg.Connection, *, source_slug: str, fact_payload: dict[str, Any]
) -> None:
    """One `review_queue` row (`kind='quote_unsupported'`) for a gated fact:
    `reason` is the one-line explanation shown on the review card; `fact` is
    the full fact, for "Keep as claim" to rebuild a `Fact` from.

    Deduped by `(source_slug, locator, quote)`: a redriven source re-extracts
    from scratch and a deterministic model gates the same quote again.

    Commits on its own: it runs before `merge_extraction`'s transaction opens."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id FROM review_queue
            WHERE kind = 'quote_unsupported' AND source_slug = %s
              AND payload -> 'fact' ->> 'locator' = %s
              AND payload -> 'fact' ->> 'quote' = %s
            """,
            (source_slug, fact_payload["locator"], fact_payload["quote"]),
        )
        if cur.fetchone() is not None:
            return
        cur.execute(
            "INSERT INTO review_queue (kind, payload, source_slug) VALUES (%s, %s, %s)",
            (
                "quote_unsupported",
                Jsonb(
                    {
                        "reason": (
                            "The model's quote for this fact was not found in the source text."
                        ),
                        "fact": fact_payload,
                    }
                ),
                source_slug,
            ),
        )
    conn.commit()


def _finish_one(
    extracted: _ExtractedSource,
    *,
    conn: psycopg.Connection,
    llm: LLMClient,
    embedder: Embedder,
    registry: Registry,
    merge_extraction: Callable[..., Any],
    sink: UsageSink,
    sleep: Callable[[float], None],
    company_context_updated_at: datetime | None = None,
    extraction_model: str | None = None,
    prompt_variant: str | None = None,
    samples: int = 1,
) -> _SourceOutcome:
    """quote gate -> merge -> mark source status -> record coverage.

    `extraction_model`/`prompt_variant`/`samples` are stamped on the source
    row with its `done`/`partial` status, for `_prepare_one`'s skip check.
    `company_context_updated_at` is forwarded to `merge_extraction`.

    Writes to `conn`, so it runs single-threaded, in source order (spec sec
    5.1: "merge and write are single-writer"). Raises `_DeadLettered` when
    extraction produced nothing or merge fails.

    The source is marked `done`/`partial` only after `merge_extraction` has
    returned, so the status is never visible before the source's graph
    writes are committed.

    `extracted.sink`'s events are replayed into `sink` (the run's usage sink)
    first, so a source that is then dead-lettered still records its billed
    calls.
    """
    for event in extracted.sink.events:
        sink.record(event)

    prepared = extracted.prepared
    raw, source_row, envelopes = prepared.raw, prepared.source_row, prepared.envelopes
    pass_results, segment_failures = extracted.results, extracted.segment_failures

    # The true segment count comes from the splitter, not segment 1's self-report.
    segments_total = envelopes[0].segment.total if envelopes[0].segment else 1

    # Merge each pass's per-envelope results; a pass where every envelope
    # failed contributes nothing.
    pass_merged = [
        merge_segment_results(results, segments_total=segments_total)
        for results in pass_results
        if results
    ]

    if not pass_merged:
        error = RuntimeError("; ".join(segment_failures) or "extraction produced no results")
        _fail_source(
            conn,
            source_id=raw.source_id,
            source_type=raw.source_type.value,
            day=raw.source_date,
            stage="extract",
            error=error,
            sources_slug=source_row.slug,
            seen=0,
        )
        raise _DeadLettered from error

    try:
        # Union the passes with the same dedup rule; a single pass is used as
        # is, never merged a second time.
        merged = (
            pass_merged[0]
            if len(pass_merged) == 1
            else merge_segment_results(pass_merged, segments_total=len(pass_merged))
        )

        # The quote gate runs on the unioned facts before any reach
        # `merge_extraction`, so a gated fact never becomes a claim.
        kept_facts, gated_facts = _gate_quotes(merged.facts, source_text=prepared.text)
        if gated_facts:
            merged.facts = kept_facts
            for fact_payload in gated_facts:
                _queue_quote_unsupported(
                    conn, source_slug=source_row.slug, fact_payload=fact_payload
                )
    except Exception as exc:
        _fail_source(
            conn,
            source_id=raw.source_id,
            source_type=raw.source_type.value,
            day=raw.source_date,
            stage="quote_gate",
            error=exc,
            sources_slug=source_row.slug,
            seen=0,
            attempt=1,
        )
        raise _DeadLettered from exc
    logger.info(
        "source %s: quote gate kept=%d gated=%d",
        source_row.slug,
        len(kept_facts),
        len(gated_facts),
    )

    try:
        merge_report = _retry_stage(
            lambda: merge_extraction(
                conn,
                llm,
                embedder,
                registry,
                merged,
                source_row,
                company_context_updated_at=company_context_updated_at,
            ),
            sleep=sleep,
        )
    except Exception as exc:
        _fail_source(
            conn,
            source_id=raw.source_id,
            source_type=raw.source_type.value,
            day=raw.source_date,
            stage="merge",
            error=exc,
            sources_slug=source_row.slug,
            seen=0,
        )
        raise _DeadLettered from exc

    status = "partial" if segment_failures else "done"
    last_error = "; ".join(segment_failures) if segment_failures else None
    try:
        _mark_source_status(
            conn,
            source_row.slug,
            status,
            last_error=last_error,
            extraction_model=extraction_model,
            extraction_prompt_variant=prompt_variant,
            extraction_samples=samples,
        )
        record_coverage(
            conn,
            raw.source_type.value,
            raw.source_date,
            extracted=1 if status == "done" else 0,
            partial=1 if status == "partial" else 0,
        )
        _clear_dead_letters(conn, raw.source_id)
    except Exception as exc:
        # The merge is committed; a redrive re-merges, and merge dedups.
        _fail_source(
            conn,
            source_id=raw.source_id,
            source_type=raw.source_type.value,
            day=raw.source_date,
            stage="status",
            error=exc,
            sources_slug=source_row.slug,
            seen=0,
            attempt=1,
        )
        raise _DeadLettered from exc
    return _SourceOutcome(status=status, facts_count=len(merged.facts), merge_report=merge_report)


def _process_one(
    *,
    conn: psycopg.Connection,
    connector: Connector,
    ref: SourceRef,
    registry: Registry,
    store: Store,
    settings: Settings,
    backend: Backend,
    system_prompt: str,
    vocabulary: Vocabulary,
    llm: LLMClient,
    embedder: Embedder,
    merge_extraction: Callable[..., Any],
    sink: UsageSink,
    sleep: Callable[[float], None],
    company_context_updated_at: datetime | None = None,
    extraction_model: str | None = None,
    prompt_variant: str | None = None,
    samples: int = 1,
) -> _SourceOutcome:
    """The three phases `run()` uses (`_prepare_one`/`_extract_segments`/
    `_finish_one`) for one source, sequentially, for `redrive()`.

    `force=True` on `_prepare_one`: a redriven source always gets another
    attempt. The extraction signature is still forwarded, so a successful
    redrive stamps the row as a normal run would.
    """
    prepared = _prepare_one(
        conn=conn,
        connector=connector,
        ref=ref,
        registry=registry,
        store=store,
        settings=settings,
        backend=backend,
        system_prompt=system_prompt,
        vocabulary=vocabulary,
        sink=sink,
        sleep=sleep,
        force=True,
        extraction_model=extraction_model,
        prompt_variant=prompt_variant,
        samples=samples,
    )
    if isinstance(prepared, _SourceOutcome):
        return prepared

    extracted = _extract_segments(
        prepared,
        backend=backend,
        system_prompt=system_prompt,
        settings=settings,
        sleep=sleep,
        samples=samples,
    )
    return _finish_one(
        extracted,
        conn=conn,
        llm=llm,
        embedder=embedder,
        registry=registry,
        merge_extraction=merge_extraction,
        sink=sink,
        extraction_model=extraction_model,
        prompt_variant=prompt_variant,
        samples=samples,
        sleep=sleep,
        company_context_updated_at=company_context_updated_at,
    )


# --------------------------------------------------------------------------
# run() / redrive()
# --------------------------------------------------------------------------


def run(
    conn: psycopg.Connection,
    *,
    connector: Connector,
    llm: LLMClient,
    embedder: Embedder,
    registry: Registry,
    store: Store,
    settings: Settings,
    since: date | None = None,
    limit: int | None = None,
    dry_run: bool = False,
    force: bool = False,
    prompt_variant: str = DEFAULT_PROMPT_VARIANT,
    samples: int = 1,
    sink: UsageSink | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> RunReport:
    """Run every listed source through the pipeline (module docstring,
    "Extraction fan-out").

    `samples`: independent extraction passes per source, unioned; stamped on
    the source as `extraction_samples`.

    `sink`: where every metered call this run makes lands. `None` (the CLI)
    means `portolan.metering.PostgresSink(conn)`; pass an in-memory sink to
    keep usage off the database.

    `dry_run`: fetch/intake/prepare/filter still run (so the report can show
    what would happen), but the extractor is never called, no note or claim is
    written, and neither the index nor the lint runs.

    `force`: re-extract every included source, ignoring the skip check.

    `prompt_variant`: the `build_system_prompt` variant, stamped on every
    source this run extracts.
    """
    started = time.monotonic()
    report = RunReport()

    _maybe_import_registry(conn, connector)
    registry = Registry.load(conn)  # pick up anything import_crm just wrote

    customer_extension = load_customer_extension(connector)
    # Company context is read once per run. Its `updated` stamp goes on every
    # claim this run inserts, recording which version it was extracted under.
    company_context = load_company_context(conn)
    system_prompt = build_system_prompt(
        customer_extension,
        company_context=company_context.text if company_context else None,
        variant=prompt_variant,
    )
    company_context_updated_at = company_context.updated if company_context else None
    vocabulary = merge_customer_extension(customer_extension).vocabulary
    extraction_model = settings.extraction_model

    effective_sink: UsageSink = sink if sink is not None else PostgresSink(conn)
    llm.sink = effective_sink  # type: ignore[attr-defined]

    # Discover the model's JSON strategy once, before extraction: unprimed, an
    # OpenRouter client assumes native `json_schema`, which some endpoints do
    # not support (01-spec/corpus-fix-2026-09-19.md). The answer is cached
    # process-wide, so per-thread clones share it.
    warm_json_strategy = getattr(llm, "warm_json_strategy", None)
    if warm_json_strategy is not None:
        warm_json_strategy(settings.extraction_model)

    backend: Backend = LLMClientBackend(llm, model=settings.extraction_model)

    events = _sink_events(llm)
    usage_start = len(events) if events is not None else 0

    refs = list(connector.list_sources(since))
    if limit is not None:
        refs = refs[:limit]
    report.sources_seen = len(refs)

    concurrency = _extract_concurrency()
    thread_local = threading.local()
    worker_sinks: list[InMemorySink] = []

    def _worker_backend() -> Backend:
        """This worker thread's own `Backend`, built once per thread and cached
        in `thread_local`. `AgentSDKClient` and `OpenRouterClient` get a clone
        with a private sink (replayed into `effective_sink` after the pool):
        `using_sink()` swaps `self.sink`/`self.tenant` without a lock, so one
        instance must not be shared across threads. Other clients are shared.
        """
        cached = getattr(thread_local, "backend", None)
        if cached is not None:
            return cached
        from portolan.llm.agent_sdk import AgentSDKClient
        from portolan.llm.openrouter import OpenRouterClient

        if isinstance(llm, AgentSDKClient):
            worker_sink = InMemorySink()
            worker_sinks.append(worker_sink)
            worker_llm: LLMClient = AgentSDKClient(
                default_model=llm.default_model, sink=worker_sink, tenant=llm.tenant
            )
        elif isinstance(llm, OpenRouterClient):
            worker_sink = InMemorySink()
            worker_sinks.append(worker_sink)
            worker_llm = OpenRouterClient(
                spend_key=llm.spend_key,
                spend_meter=llm.spend_meter,
                api_key=secret_value(settings.openrouter_api_key),
                default_model=llm.default_model,
                provider_order=llm.provider_order,
                sink=worker_sink,
                tenant=llm.tenant,
            )
        else:
            worker_llm = llm
        cached = LLMClientBackend(worker_llm, model=settings.extraction_model)
        thread_local.backend = cached
        return cached

    def _extract_in_worker(prepared: _PreparedSource) -> _ExtractedSource:
        return _extract_segments(
            prepared,
            backend=_worker_backend(),
            system_prompt=system_prompt,
            settings=settings,
            sleep=sleep,
            samples=samples,
        )

    def _count_outcome(outcome: _SourceOutcome) -> None:
        if outcome.status == "failed":
            report.sources_failed += 1
        elif outcome.status == "filtered":
            report.sources_filtered += 1
        elif outcome.status == "skipped":
            report.sources_skipped += 1
        elif outcome.status == "done":
            report.sources_extracted += 1
            report.facts_extracted += outcome.facts_count
            _accumulate_merge_counts(report.merge_counts, outcome.merge_report)
        elif outcome.status == "partial":
            report.sources_partial += 1
            report.facts_extracted += outcome.facts_count
            _accumulate_merge_counts(report.merge_counts, outcome.merge_report)

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for window_start in range(0, len(refs), concurrency):
            window = refs[window_start : window_start + concurrency]

            # Phase 1: fetch/intake/prepare/filter, one at a time, in
            # order -- every write here goes through `conn`.
            window_items: list[_PreparedSource | _SourceOutcome] = []
            for ref in window:
                try:
                    item = _prepare_one(
                        conn=conn,
                        connector=connector,
                        ref=ref,
                        registry=registry,
                        store=store,
                        settings=settings,
                        backend=backend,
                        system_prompt=system_prompt,
                        vocabulary=vocabulary,
                        sink=effective_sink,
                        sleep=sleep,
                        dry_run=dry_run,
                        force=force,
                        extraction_model=extraction_model,
                        prompt_variant=prompt_variant,
                        samples=samples,
                    )
                except _DeadLettered:
                    item = _SourceOutcome(status="failed")
                window_items.append(item)

            if dry_run:
                # Stop before extraction: an included source is counted as
                # would-extract; any other outcome is counted as on a real run.
                for item in window_items:
                    if isinstance(item, _PreparedSource):
                        report.sources_would_extract += 1
                    else:
                        _count_outcome(item)
                continue

            # Phase 2: fan out this window's segment-extraction calls
            # (spec sec 5.1, "extraction fans out") -- no `conn` access,
            # safe to overlap.
            futures = {
                i: pool.submit(_extract_in_worker, item)
                for i, item in enumerate(window_items)
                if isinstance(item, _PreparedSource)
            }
            for i, future in futures.items():
                window_items[i] = future.result()

            # Phase 3: merge/write, strictly sequential, in the window's
            # (== `refs`'s) original order -- the single-writer stage.
            for item in window_items:
                if isinstance(item, _SourceOutcome):
                    outcome = item
                else:
                    try:
                        outcome = _finish_one(
                            item,
                            conn=conn,
                            llm=llm,
                            embedder=embedder,
                            registry=registry,
                            merge_extraction=merge_extraction,
                            sink=effective_sink,
                            sleep=sleep,
                            company_context_updated_at=company_context_updated_at,
                            extraction_model=extraction_model,
                            prompt_variant=prompt_variant,
                            samples=samples,
                        )
                    except _DeadLettered:
                        outcome = _SourceOutcome(status="failed")
                _count_outcome(outcome)

    # Replay every worker clone's private sink into `effective_sink` from the
    # main thread, the only thread that may touch a `conn`-backed sink.
    for worker_sink in worker_sinks:
        for event in worker_sink.events:
            effective_sink.record(event)

    if events is not None:
        run_events = events[usage_start:]
        report.usage_by_surface = _usage_by_surface(run_events)
        report.usage_pricing_errors = sum(1 for e in run_events if e.extra.get("pricing_error"))

    if not dry_run:
        if settings.crm_change_claims:
            crm_report = refresh_crm_changes(conn, Registry.load(conn))
            conn.commit()
            report.crm_changes_written = crm_report.changes_written
        index_report = index_pending(conn, embedder)
        report.index_chunks_written = index_report.chunks_written

    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM dead_letter WHERE redriven_at IS NULL")
        (report.dead_letter_count,) = cur.fetchone()

    report.wall_time_seconds = time.monotonic() - started

    if not dry_run:
        lint_report = run_lint(conn, store)
        markdown = render_markdown(lint_report)
        lint_dir = Path(settings.store_root).parent / "lint"
        lint_dir.mkdir(parents=True, exist_ok=True)
        (lint_dir / f"lint-{lint_report.today:%Y-%m-%d}.md").write_text(markdown)

        write_changelog_row(
            conn,
            kind="pipeline_run",
            summary=(
                f"pipeline run: seen={report.sources_seen} extracted={report.sources_extracted} "
                f"partial={report.sources_partial} failed={report.sources_failed}"
            ),
            details=report.model_dump(mode="json"),
        )
        conn.commit()

    return report


def redrive(
    conn: psycopg.Connection,
    *,
    connector: Connector,
    llm: LLMClient,
    embedder: Embedder,
    registry: Registry,
    store: Store,
    settings: Settings,
    prompt_variant: str = DEFAULT_PROMPT_VARIANT,
    samples: int = 1,
    sleep: Callable[[float], None] = time.sleep,
) -> RedriveReport:
    """Re-run every source with a not-yet-cleared `dead_letter` row, once
    each, through the same per-source pipeline (fetch -> intake -> ... ->
    merge). A source that succeeds gets `redriven_at` set on all its rows
    (never deleted -- nothing is hard-deleted, spec sec 8); a source still
    failing, or one the connector no longer lists, stays pending.
    `attempted`/`succeeded`/`still_failing` count sources, not rows.
    `prompt_variant` and `samples` mean what they mean for `run()`.
    """
    report = RedriveReport()

    # One pass per source, however many failed runs left a row for it:
    # each pass is a full, paid extraction.
    with conn.cursor() as cur:
        cur.execute(
            "SELECT source_id FROM dead_letter WHERE redriven_at IS NULL "
            "GROUP BY source_id ORDER BY min(id)"
        )
        source_ids = [row[0] for row in cur.fetchall()]
    if not source_ids:
        return report

    customer_extension = load_customer_extension(connector)
    company_context = load_company_context(conn)
    system_prompt = build_system_prompt(
        customer_extension,
        company_context=company_context.text if company_context else None,
        variant=prompt_variant,
    )
    company_context_updated_at = company_context.updated if company_context else None
    vocabulary = merge_customer_extension(customer_extension).vocabulary

    # Prime the JSON strategy once, as `run()` does.
    warm_json_strategy = getattr(llm, "warm_json_strategy", None)
    if warm_json_strategy is not None:
        warm_json_strategy(settings.extraction_model)

    backend: Backend = LLMClientBackend(llm, model=settings.extraction_model)

    refs_by_source_id = {ref.source_id: ref for ref in connector.list_sources(None)}

    for source_id in source_ids:
        report.attempted += 1
        ref = refs_by_source_id.get(source_id)
        if ref is None:
            report.still_failing += 1
            continue
        try:
            _process_one(
                conn=conn,
                connector=connector,
                ref=ref,
                registry=registry,
                store=store,
                settings=settings,
                backend=backend,
                system_prompt=system_prompt,
                vocabulary=vocabulary,
                llm=llm,
                embedder=embedder,
                merge_extraction=merge_extraction,
                sink=llm.sink,  # type: ignore[attr-defined]
                sleep=sleep,
                company_context_updated_at=company_context_updated_at,
                extraction_model=settings.extraction_model,
                prompt_variant=prompt_variant,
                samples=samples,
            )
        except _DeadLettered:
            report.still_failing += 1
            continue

        _clear_dead_letters(conn, source_id)
        report.succeeded += 1

    return report


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _build_connector(name: str, path: Path) -> Connector:
    if name == "file":
        from portolan.connectors.file_connector import FileConnector

        return FileConnector(path)
    from portolan.connectors.registry import get_connector_class

    connector_cls = get_connector_class(name)
    raise ValueError(
        f"--connector {name!r} ({connector_cls.__name__}) needs vendor configuration this CLI "
        "does not collect from flags; construct it in a script and call portolan.pipeline.run() "
        "directly. Only --connector file is wired up here."
    )


def _build_llm(kind: str, settings: Settings, *, corpus_path: Path | None) -> LLMClient:
    if kind == "fake":
        from portolan.fake_extractor import FakePlacements, build_fake_llm_client

        if corpus_path is None:
            raise ValueError("--llm fake requires --path (the corpus root) to build placements")
        return build_fake_llm_client(FakePlacements(corpus_path), tenant=settings.tenant)
    if kind == "agent_sdk":
        from portolan.llm.agent_sdk import AgentSDKClient

        return AgentSDKClient(default_model=settings.extraction_model, tenant=settings.tenant)
    if kind == "bedrock":
        from portolan.llm.bedrock import BedrockClient

        return BedrockClient(settings.extraction_model, settings.aws_region, tenant=settings.tenant)
    if kind == "anthropic":
        from portolan.llm.anthropic_api import AnthropicAPIClient

        if not settings.anthropic_api_key:
            raise ValueError("--llm anthropic needs ANTHROPIC_API_KEY set")
        return AnthropicAPIClient(
            api_key=settings.anthropic_api_key.get_secret_value(),
            default_model=settings.extraction_model,
            tenant=settings.tenant,
        )
    if kind == "openrouter":
        from portolan.llm.openrouter import OpenRouterClient

        if not settings.openrouter_api_key:
            raise ValueError("--llm openrouter needs OPENROUTER_API_KEY set")
        return OpenRouterClient(
            spend_key="pipeline",
            api_key=settings.openrouter_api_key.get_secret_value(),
            default_model=settings.extraction_model,
            provider_order=settings.openrouter_provider_order,
            tenant=settings.tenant,
        )
    raise ValueError(f"unknown --llm {kind!r}")


def _build_embedder(kind: str) -> Embedder:
    if kind == "fake":
        from portolan.search.embed import FakeEmbedder

        return FakeEmbedder()
    if kind == "fastembed":
        from portolan.search.embed import FastEmbedEmbedder

        return FastEmbedEmbedder()
    raise ValueError(f"unknown --embed {kind!r}")


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--connector", default="file")
    parser.add_argument("--path", type=Path, default=Path("corpus/out"))
    parser.add_argument(
        "--llm",
        choices=["fake", "agent_sdk", "bedrock", "anthropic", "openrouter"],
        default="fake",
    )
    parser.add_argument("--embed", choices=["fake", "fastembed"], default="fake")
    # Extract each included source N independent times and union the results
    # (`merge_segment_results`). On `redrive` too, to match a normal run.
    parser.add_argument("--samples", type=int, default=1)


def _run_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m portolan.pipeline")
    _add_common_args(parser)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--since", default=None)
    parser.add_argument("--dry-run", action="store_true")
    # Re-extract every included source, ignoring the stored extraction signature.
    parser.add_argument("--force", action="store_true")
    return parser


def _redrive_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m portolan.pipeline redrive")
    _add_common_args(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    if argv and argv[0] == "redrive":
        args = _redrive_parser().parse_args(argv[1:])
        settings = get_settings()
        conn = db.connect()
        try:
            db.apply_migrations(conn)
            connector = _build_connector(args.connector, args.path)
            llm = _build_llm(args.llm, settings, corpus_path=args.path)
            embedder = _build_embedder(args.embed)
            store = LocalStore(settings.store_root)
            registry = Registry.load(conn)
            report = redrive(
                conn,
                connector=connector,
                llm=llm,
                embedder=embedder,
                registry=registry,
                store=store,
                settings=settings,
                samples=args.samples,
            )
            print(report.model_dump_json(indent=2))
        finally:
            conn.close()
        return 0

    args = _run_parser().parse_args(argv)
    settings = get_settings()
    conn = db.connect()
    try:
        db.apply_migrations(conn)
        connector = _build_connector(args.connector, args.path)
        llm = _build_llm(args.llm, settings, corpus_path=args.path)
        embedder = _build_embedder(args.embed)
        store = LocalStore(settings.store_root)
        registry = Registry.load(conn)
        since = date.fromisoformat(args.since) if args.since else None
        report = run(
            conn,
            connector=connector,
            llm=llm,
            embedder=embedder,
            registry=registry,
            store=store,
            settings=settings,
            since=since,
            limit=args.limit,
            dry_run=args.dry_run,
            force=args.force,
            samples=args.samples,
        )
        print(report.model_dump_json(indent=2))
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
