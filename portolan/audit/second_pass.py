"""`second_pass`: re-read each sampled source's stored original, prepare
and segment it exactly like the pipeline's first pass did, and extract it
again with a stronger backend (`01-spec/quality-loop.md` "The production
audit": "re-extract with Sonnet or Opus, full document, same schema").

Re-reading from the store rather than re-using whatever text the first
pass happened to extract from is the point: it is how this catches a
first-pass bug in `prepare_text`/segmentation itself, not just a model
miss on identical input.

**Fixed, polish wave 2026-09-12 (was a known limitation, flagged in the
wave report for "whichever wave next touches `portolan/ingest`" -- this
one).** `sources.metadata` now carries the connector's original
`RawSource.metadata` under a `"raw"` key (migration `0009_sources_
metadata.sql`, `portolan.ingest.intake.intake`). `_reconstruct_raw_source`
below uses it directly when present: exactly right for every source type,
including ticket/crm/rss, whose `RawSource.text` is a *rendering* the
connector built from structured data (`_render_crm_row_text`, `_fetch_
rss`'s title+summary join) that re-deriving from the stored bytes alone
could never recover -- before this fix, a second-pass CRM/RSS source got
fed something other than what the first pass actually saw, not just a
coarser locator. Falls back to the pre-fix re-derivation (still lossy for
crm/rss) only for a source ingested before `0009` landed, whose
`metadata` carries no `"raw"` key.
"""

from __future__ import annotations

import json

import psycopg

from portolan import pipeline
from portolan.context import load_company_context
from portolan.extract.backends import Backend, LLMClientBackend
from portolan.extract.prompt import build_system_prompt
from portolan.extract.schema import ExtractionResult, SourceType, merge_customer_extension
from portolan.extract.segment import merge_segment_results
from portolan.ingest.filter import customer_filter
from portolan.ingest.models import RawSource, SourceRow
from portolan.ingest.prepare import prepare_text
from portolan.llm.base import LLMClient
from portolan.metering import InMemorySink, UsageSink
from portolan.registry import Registry
from portolan.store import Store
from portolan.util import clock

__all__ = ["AuditBackend", "SecondPassResults", "second_pass"]


# --------------------------------------------------------------------------
# AuditBackend: LLMClientBackend, metered as surface="audit"
# --------------------------------------------------------------------------


class AuditBackend(LLMClientBackend):
    """One-line alias for `LLMClientBackend(client, model=model,
    surface="audit")` (polish wave 2026-09-12: `LLMClientBackend` grew the
    `surface` constructor argument this class used to need a ~40-line,
    near-verbatim subclass for -- review finding #11, fix wave
    2026-09-12). Kept, rather than deleted outright, only because
    `portolan.audit.__main__`/`__init__` (outside this fix's owned paths)
    still import this name; a caller with no reason to keep the old name
    should just write `LLMClientBackend(client, surface="audit")`
    directly."""

    def __init__(self, client: LLMClient, model: str | None = None) -> None:
        super().__init__(client, model=model, surface="audit")


# --------------------------------------------------------------------------
# Reconstructing a RawSource from the store (module docstring)
# --------------------------------------------------------------------------


def _reconstruct_raw_source(source: SourceRow, store: Store) -> RawSource:
    """`RawSource` for `prepare_text`, rebuilt from the stored original
    bytes plus `source.metadata["raw"]` when present (module docstring).

    `source_date`'s fallback (review finding 13, both branches below) is
    `portolan.util.clock.today()`, not stdlib `date.today()` -- the
    process host's own local zone, the exact wall-clock leak `clock` was
    written to remove. `source.source_date` is expected to always be set
    in practice (this only matters for a source ingested with no known
    date at all).
    """
    data = store.get(source.s3_key) if source.s3_key else b""
    source_type = SourceType(source.source_type)
    text = data.decode("utf-8", errors="replace")

    stored_raw = source.metadata.get("raw") if isinstance(source.metadata, dict) else None
    if stored_raw:
        metadata: dict[str, object] = dict(stored_raw)
        if source_type == SourceType.EMAIL:
            # The stored bytes are the full RFC822 message (headers +
            # body); `metadata` already carries message_id/subject/
            # from_email verbatim from intake, but `RawSource.text` for
            # email is body-only (`file_connector._fetch_email`), so the
            # headers still have to be split off the bytes either way.
            from portolan.connectors.file_connector import _parse_email_headers

            _headers, body = _parse_email_headers(text)
            body = body.strip("\n")
            text = (body + "\n") if body else ""
        elif source_type == SourceType.CRM:
            # The stored bytes are `json.dumps(row)` (`_fetch_crm`);
            # `RawSource.text` is `_render_crm_row_text(object_type, row)`,
            # a rendering, not the raw JSON -- rebuild it the same way the
            # connector did, now that `metadata["object"]` survives.
            from portolan.connectors.file_connector import _render_crm_row_text

            try:
                row = json.loads(text)
            except json.JSONDecodeError:
                row = {}
            text = _render_crm_row_text(metadata.get("object", "record"), row)
        elif source_type == SourceType.RSS:
            # The stored bytes are `json.dumps(item)` (`_fetch_rss`);
            # `RawSource.text` is `f"{title}\n\n{summary}"`, also a
            # rendering, not the raw JSON.
            try:
                item = json.loads(text)
            except json.JSONDecodeError:
                item = {}
            text = f"{item.get('title', '')}\n\n{item.get('summary', '')}".strip() + "\n"
        # TICKET: `_prepare_ticket` (portolan.ingest.prepare) reads
        # `raw.metadata["comments"]` directly and ignores `raw.text`
        # whenever comments are given, so the raw JSON blob text is fine
        # as-is here -- nothing to rebuild.

        return RawSource(
            source_id=source.slug,
            source_type=source_type,
            source_date=source.source_date or clock.today(),
            participants=source.participants,
            text=text,
            original_bytes=data,
            metadata=metadata,
        )

    # Legacy fallback: a source ingested before migration 0009 landed, so
    # `metadata["raw"]` was never stored -- re-derive best-effort from the
    # stored bytes alone, exactly as this function did before the fix
    # (still lossy for crm/rss: see the module docstring).
    metadata = {}
    if source_type == SourceType.TICKET:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict) and parsed.get("comments"):
            metadata["comments"] = parsed["comments"]
    elif source_type == SourceType.EMAIL:
        from portolan.connectors.file_connector import _parse_email_headers

        headers, body = _parse_email_headers(text)
        metadata["message_id"] = headers.get("Message-ID", source.slug)
        text = body
    elif source_type == SourceType.RSS:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            metadata["link"] = parsed.get("link", "")
    elif source_type == SourceType.CRM:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            metadata["object"] = parsed.get("object", parsed.get("kind", "record"))
            metadata["crm_id"] = parsed.get("crm_id", source.slug)

    return RawSource(
        source_id=source.slug,
        source_type=source_type,
        source_date=source.source_date or clock.today(),
        participants=source.participants,
        text=text,
        original_bytes=data,
        metadata=metadata,
    )


# --------------------------------------------------------------------------
# second_pass
# --------------------------------------------------------------------------


class SecondPassResults(dict[str, ExtractionResult]):
    """`dict[source_slug, ExtractionResult]` (the documented return type of
    `second_pass`) that also carries the `UsageSink` every `strong_backend.
    extract()` call recorded into and, per source, the failure (if any)
    that kept a source out of the dict -- mirrors `portolan.bench.
    extraction.ExtractionRunResult`'s pattern of a plain-dict subclass
    with reporting extras a caller can ignore."""

    def __init__(
        self,
        results: dict[str, ExtractionResult],
        *,
        sink: UsageSink,
        failures: dict[str, str],
    ) -> None:
        super().__init__(results)
        self.sink = sink
        self.failures = failures


def second_pass(
    conn: psycopg.Connection,
    store: Store,
    strong_backend: Backend,
    sources: list[SourceRow],
    *,
    registry: Registry | None = None,
    connector=None,
    tenant: str = "audit",
    sink: UsageSink | None = None,
) -> SecondPassResults:
    """Re-read, prepare, segment and re-extract every source in `sources`
    with `strong_backend` (`01-spec/quality-loop.md` "The production
    audit"). A source whose original cannot be re-read or re-extracted
    (e.g. `s3_key` missing, the store has no such key, a segment call
    raises) is recorded in the result's `.failures` rather than raising --
    one bad source in the sample must not abort the whole week's audit,
    mirroring `portolan.pipeline`'s own "one bad source never aborts a
    run" policy.

    **Review finding G9 (fix wave 2026-09-12): same envelope as the first
    pass, on purpose.** This used to build a bare `SourceEnvelope` (no
    `registry_context`, no `vocabulary`), validate against `build_system_
    prompt()` with no `CustomerExtension`, strip participants of their
    resolved `registry_id`, always call `segment_text` (so a short source
    the first pass sent as `segment=None` arrived here as "segment 1/1"),
    and merge segments without `segments_total=`. quality-loop.md's whole
    premise -- "same schema, same document, stronger model" -- only holds
    once every one of those agrees with what the first pass actually saw;
    otherwise every measured miss is confounded with a weaker envelope,
    not a weaker model. Fixed by routing every envelope through the exact
    helpers `portolan.pipeline` uses for the first pass:
    `build_envelope_participants`, `build_registry_context`,
    `build_envelopes` (the same <=4000-token segment-or-not decision --
    its default 200-token overlap is unchanged, since that is
    `portolan.extract.segment.segment`'s own default either way), and
    `load_customer_extension`.

    `registry` defaults to `Registry.load(conn)` -- `conn` is genuinely
    needed now, not just kept for signature parity. `connector` is
    optional (for `load_customer_extension`, which needs a `.root` to
    find a `customer.yaml`); without one this still fixes the segmenting-
    decision and registry-context parity, just without a customer
    extension layered on top (matching a first-pass run against a
    connector with no `customer.yaml` either).
    """
    registry = registry if registry is not None else Registry.load(conn)
    customer_extension = (
        pipeline.load_customer_extension(connector) if connector is not None else None
    )
    # Same company context as the first pass (`pipeline.run`): the prompt's
    # own examples name a fictional vendor, so this is the only vendor it sees.
    company_context = load_company_context(conn)
    system_prompt = build_system_prompt(
        customer_extension, company_context=company_context.text if company_context else None
    )
    vocabulary = merge_customer_extension(customer_extension).vocabulary
    active_sink = sink if sink is not None else InMemorySink()

    results: dict[str, ExtractionResult] = {}
    failures: dict[str, str] = {}

    for source in sources:
        try:
            raw = _reconstruct_raw_source(source, store)
            prepared = prepare_text(raw)
            # `sample_rate`/`seed` don't matter here -- only `matched_
            # entities` is used, to build the same registry_context the
            # first pass's own filter decision produced.
            decision = customer_filter(raw, prepared, registry, sample_rate=0.0, seed=tenant)
            participants = pipeline.build_envelope_participants(raw, registry)
            registry_context = pipeline.build_registry_context(registry, decision.matched_entities)
            envelopes = pipeline.build_envelopes(
                raw,
                prepared.text,
                participants=participants,
                registry_context=registry_context,
                vocabulary=vocabulary,
            )
            per_segment = [
                strong_backend.extract(envelope, system_prompt, active_sink, tenant)
                for envelope in envelopes
            ]
            segments_total = envelopes[0].segment.total if envelopes[0].segment else 1
            results[source.slug] = merge_segment_results(per_segment, segments_total=segments_total)
        except Exception as exc:  # noqa: BLE001 - recorded, not raised: see docstring
            failures[source.slug] = str(exc)

    return SecondPassResults(results, sink=active_sink, failures=failures)
