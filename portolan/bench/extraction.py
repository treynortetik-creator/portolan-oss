"""Run extraction over the bench corpus and score it against the truth
table. See `01-spec/quality-loop.md` ("The bench") for the metrics this
implements and `01-spec/build-contracts.md` ("Bench") for the signatures.
"""

from __future__ import annotations

import re
import time
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, NamedTuple, Protocol

from pydantic import BaseModel, ConfigDict, Field

from portolan.bench.corpus_loader import DEFAULT_CORPUS_ROOT, load_prepared_source
from portolan.bench.locators import locator_matches_any
from portolan.bench.truth import Manifest, PlantedFact, Truth
from portolan.extract.backends import Backend
from portolan.extract.prompt import build_system_prompt

# The containment logic (glyph folding, exact containment, fuzzy token-
# subsequence containment, the 0.75 threshold) moved to `portolan.extract.
# quote_gate` 2026-09-19 ("mechanical quote gate to production") so
# `portolan.pipeline` could enforce the identical rule live, not a second
# definition that could drift from this one. Imported here under the same
# private names this module always used, so every call site below AND
# every existing test (`tests/test_bench_extraction.py`, which imports
# `_quote_supported_by_text`/`_fuzzy_quote_containment` directly from
# THIS module) is unchanged -- see `quote_gate`'s own docstring for why
# the bench and the production gate now share one implementation.
from portolan.extract.quote_gate import (
    _FUZZY_QUOTE_THRESHOLD,
    _fold_glyphs,
    _fuzzy_quote_containment,
    _normalize_for_containment,
    _quote_supported_by_text,
)
from portolan.extract.schema import (
    QUOTE_REQUIRED_TYPES,
    Coverage,
    EntityKind,
    ExtractionResult,
    Fact,
    RegistryContextEntry,
    SegmentRef,
    SourceEnvelope,
    Vocabulary,
)
from portolan.extract.segment import merge_segment_results
from portolan.extract.segment import segment as segment_text
from portolan.metering import InMemorySink, UsageSink

TENANT = "bench"

# Rule 8: "type: other is allowed" and, per the build instructions, an
# extraction-side `other` matched against any planted type (or vice versa)
# is partial credit rather than a clean miss or a clean hit.
_OTHER_TYPE = "other"

# `var/bench/diag/diagnosis.md` (2026-09-12, the first real bench run's
# post-mortem): `commitment`/`decision`, `complaint`/`risk`, and
# `request`/`question` are the three base-type pairs the checklist itself
# describes as adjacent (a promise vs. the decision it follows from; a
# problem report vs. the threat it implies; an ask vs. the unanswered
# version of the same ask) — a model landing on the "wrong" half of one of
# these pairs got the *fact* right and the *classification* debatable, so
# it earns the same half credit as an `other` mismatch rather than a clean
# miss.
_NEAR_MATCH_TYPE_PAIRS: frozenset[frozenset[str]] = frozenset(
    {
        frozenset({"commitment", "decision"}),
        frozenset({"complaint", "risk"}),
        frozenset({"request", "question"}),
    }
)

# Claim-similarity threshold, justified in `var/bench/diag/diagnosis.md`
# against the weighted-token-set score below (replacing the first real
# run's `difflib.SequenceMatcher` character-ratio score, which sat every
# genuine paraphrase pair at 0.34-0.42 -- below its own 0.45 cutoff, so
# real matches scored as misses on text alone even when type/entities/
# locator all lined up).
_CLAIM_SIMILARITY_THRESHOLD = 0.30

# 2026-09-18 harness review (`01-spec/harness-review-2026-09-18.md`,
# matcher decisions section): the STRICT threshold above is unchanged --
# a wrong fact type or a locator outside the window is a real matcher
# rule, not a bug, and stays a miss. The LENIENT recall column exists only
# to make visible how many of those strict misses are near-misses (same
# fact, wrong type / no locator credit / a slightly-under-threshold
# paraphrase) versus facts genuinely absent from the model's output
# (`_evidence_basis`'s "not_extracted", `01-spec/extraction-miss-analysis-
# 2026-09-18.md` section 3). 0.20 sits below every genuine-paraphrase
# score that section's own hand-read found (0.269-0.833) and still above
# the near-zero scores of an unrelated produced fact that merely shares an
# account name.
_LENIENT_CLAIM_SIMILARITY_THRESHOLD = 0.20

_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "by",
        "for",
        "from",
        "had",
        "has",
        "have",
        "in",
        "into",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "our",
        "per",
        "said",
        "says",
        "than",
        "that",
        "the",
        "their",
        "then",
        "there",
        "they",
        "this",
        "to",
        "was",
        "were",
        "will",
        "with",
        "would",
    }
)

_TOKEN_SPLIT_RE = re.compile(r"[^\w.%$-]+")
_DIGIT_RE = re.compile(r"\d")

# "Normalisation" (the fix instructions' word) for numbers has to cover
# spelled-out counts, not just digits: the truth table itself mixes both
# ("Larkspur Memorial Hospital has forty Tidewater T2 pumps..." vs a ticket's
# own "4 days late") and a real extractor no more reliably picks one form
# than the corpus does. Without this, "four" and "4" share no token at all
# and a genuine paraphrase loses exactly the salient word that identifies
# it. Covers the range this corpus actually uses (unit counts, day counts,
# percents, pump counts) -- not a general number-word parser.
_NUMBER_WORDS: dict[str, str] = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
    "thirty": "30",
    "forty": "40",
    "fifty": "50",
    "sixty": "60",
    "seventy": "70",
    "eighty": "80",
    "ninety": "90",
    "hundred": "100",
}


def _normalize_claim_tokens(text: str) -> list[str]:
    """Lowercase, tokenize on whitespace/punctuation while keeping the
    internal characters a date, decimal, percent or dollar figure needs
    (`2026-08-04`, `4,800`, `12%`) as one token, drop empty/bare-
    punctuation leftovers, and fold spelled-out numbers to digits
    (`_NUMBER_WORDS`) so "four" and "4" are the same token."""
    lowered = text.lower().replace(",", "")
    tokens = (tok.strip(".-") for tok in _TOKEN_SPLIT_RE.split(lowered))
    return [_NUMBER_WORDS.get(t, t) for t in tokens if t]


# --------------------------------------------------------------------------
# run_extraction
# --------------------------------------------------------------------------


class ExtractionRunResult(dict[str, ExtractionResult]):
    """`dict[source_id, ExtractionResult]` — the documented return type of
    `run_extraction` — that also carries what `score_extraction` needs to
    report tokens, cost, wall time, and per-source failures: the
    `UsageSink` every `backend.extract()` call recorded into, per-source/
    total wall time, and `failed_sources` (`{source_id: error message}` for
    a source `run_extraction` caught an exception on and skipped rather
    than let take the whole run down -- Group 3 of the review fix wave: the
    first real bench run crashed outright on one bad source). A caller that
    only wants the results dict needs nothing extra; `score_extraction`
    reads the rest via `getattr` so a plain `dict` is still accepted.
    """

    def __init__(
        self,
        results: dict[str, ExtractionResult],
        *,
        wall_time_s: dict[str, float],
        sink: UsageSink,
        total_wall_time_s: float,
        failed_sources: dict[str, str] | None = None,
        source_texts: dict[str, str] | None = None,
    ) -> None:
        super().__init__(results)
        self.wall_time_s = wall_time_s
        self.sink = sink
        self.total_wall_time_s = total_wall_time_s
        self.failed_sources = failed_sources or {}
        # The exact prepared (locator-annotated) text `run_extraction` fed
        # the backend for each source, keyed by source_id. Used by
        # `score_extraction` to tell "unplanted but really in the source"
        # (`quality-loop.md`'s D01-D04 "trivia the corpus actually
        # contains") apart from an outright invention -- a plain dict of
        # canned results built directly in a test carries none of this
        # (`getattr(results, "source_texts", {})` at the read site), which
        # is fine: that classification only matters for a real run.
        self.source_texts = source_texts or {}


# --------------------------------------------------------------------------
# registry_context / vocabulary from the truth table
#
# Diagnosis (var/bench/diag/diagnosis.md): with no registry_context at all
# (corpus_loader.py's own docstring already flagged this as unfilled), a
# real produced fact about, say, S-call-009 never once names the account
# ("Larkspur Memorial Hospital" / `larkspur-memorial`) in its `entities`
# list -- not because the model can't resolve names it's given (it
# correctly uses "T2", "HalvardCare", "Derek Hollis", "Felix Arden", all of
# which line up with truth-table aliases), but because a real phone call
# between a rep and a customer contact routinely never says the hospital's
# own name out loud, and nothing in the envelope told the model which
# account this call is about. No scoring leniency can credit a mention
# that was never produced.
#
# `ManifestSource.accounts`/`.participants` already carry exactly what a
# real ingest connector would know before reading a word of the transcript
# (which CRM account this call/ticket belongs to, who's on it) -- the
# corpus generator wrote these fields for this purpose. Filling
# `registry_context` from them (and `vocabulary` from the truth table's
# static product/competitor list, the same for every source, per
# extraction-schema-v0.md's "customer's own product names ... and known
# competitors") is completing a stand-in the bench always intended to fill
# in, not leaking which facts are planted: identity resolution (who is on
# this call, which hospital, what products this vendor sells) is
# fact-independent context a connector supplies up front, not the answer
# to any one claim.
# --------------------------------------------------------------------------


def _account_registry_entries(manifest_source: Any, truth: Truth) -> list[RegistryContextEntry]:
    entries: list[RegistryContextEntry] = []
    accounts_by_id = {a.id: a for a in truth.table.accounts}
    for account_id in getattr(manifest_source, "accounts", None) or []:
        account = accounts_by_id.get(account_id)
        if account is None:
            continue
        entries.append(
            RegistryContextEntry(
                name=account.name,
                kind=EntityKind.ACCOUNT,
                registry_id=account.id,
                aliases=list(account.aliases),
            )
        )
    return entries


def _contact_registry_entries(manifest_source: Any, truth: Truth) -> list[RegistryContextEntry]:
    entries: list[RegistryContextEntry] = []
    contacts_by_id = {c.id: c for c in truth.table.contacts}
    internal_by_id = {p["id"]: p for p in (truth.table.meta or {}).get("internal_people", [])}
    for person_id in getattr(manifest_source, "participants", None) or []:
        contact = contacts_by_id.get(person_id)
        if contact is not None:
            entries.append(
                RegistryContextEntry(
                    name=contact.name, kind=EntityKind.CONTACT, registry_id=contact.id
                )
            )
            continue
        internal = internal_by_id.get(person_id)
        if internal is not None:
            entries.append(
                RegistryContextEntry(
                    name=internal["name"],
                    kind=EntityKind.INTERNAL_PERSON,
                    registry_id=f"internal:{internal['id']}",
                )
            )
    return entries


def _registry_context_for(manifest_source: Any, truth: Truth) -> list[RegistryContextEntry]:
    """Every account and participant contact/internal person the manifest
    already says belongs to this source, resolved against the truth
    table's own id/name/alias data -- the identity-resolution half of
    `registry_context`, independent of which facts happen to be planted."""
    return _account_registry_entries(manifest_source, truth) + _contact_registry_entries(
        manifest_source, truth
    )


def _vocabulary_from_truth(truth: Truth) -> Vocabulary:
    """The supplier's static product and competitor names
    (`truth-table.yaml`'s `meta.products`/`meta.competitors`) -- the same
    for every source, exactly like a customer extension's own vocabulary
    block (extraction-schema-v0.md's "customer extension"), not tied to
    any one source's planted facts."""
    meta = truth.table.meta or {}
    products = [p["name"] for p in meta.get("products", []) if p.get("name")]
    competitors = [c["name"] for c in meta.get("competitors", []) if c.get("name")]
    return Vocabulary(products=products, competitors=competitors)


def _build_envelope(
    prepared_participants: list,
    source_id: str,
    source_type,
    source_date,
    text: str,
    segment_ref: SegmentRef | None,
    registry_context: list[RegistryContextEntry] | None = None,
    vocabulary: Vocabulary | None = None,
) -> SourceEnvelope:
    return SourceEnvelope(
        source_id=source_id,
        source_type=source_type,
        source_date=source_date,
        participants=prepared_participants,
        registry_context=registry_context or [],
        vocabulary=vocabulary or Vocabulary(),
        text=text,
        segment=segment_ref,
    )


class _SourceOutcome:
    """One source's `run_extraction` attempt: either `result` is set (and
    `error` is None) or vice versa. A plain tuple would work too, but this
    is one usage site clearer to read, and gives the thread-pool path
    (`concurrency > 1`, where outcomes arrive out of manifest order) and the
    sequential path exactly the same shape to fold into the run's
    dictionaries."""

    __slots__ = ("source_id", "result", "error", "elapsed_s", "text")

    def __init__(
        self,
        source_id: str,
        result: ExtractionResult | None,
        error: str | None,
        elapsed_s: float,
        text: str,
    ) -> None:
        self.source_id = source_id
        self.result = result
        self.error = error
        self.elapsed_s = elapsed_s
        self.text = text


def _extract_one_pass(
    prepared: Any,
    backend: Backend,
    prompt: str,
    active_sink: UsageSink,
    tenant: str,
    *,
    segment: bool,
    max_segment_tokens: int,
    segment_overlap_tokens: int,
    registry_context: list[RegistryContextEntry] | None,
    vocabulary: Vocabulary | None,
) -> ExtractionResult:
    """One independent extraction attempt over `prepared` -- segmented and
    merged back into one `ExtractionResult` when `segment` is True, a
    single call otherwise. Factored out of `_extract_one_source` so
    `--samples N` (2026-09-18 harness review, determinism section) can
    call it N times per source and union the results (`merge_segment_
    results` again, this time across samples rather than segments) without
    duplicating the segmentation logic."""
    if segment:
        segments = segment_text(
            prepared.text,
            max_tokens=max_segment_tokens,
            overlap_tokens=segment_overlap_tokens,
        )
        per_segment_results = []
        for seg in segments:
            envelope = _build_envelope(
                prepared.participants,
                prepared.source_id,
                prepared.source_type,
                prepared.source_date,
                seg.text,
                SegmentRef(index=seg.index, total=seg.total),
                registry_context,
                vocabulary,
            )
            per_segment_results.append(backend.extract(envelope, prompt, active_sink, tenant))
        # F10: pass the true segment count rather than trust segment
        # 1's self-reported total.
        return merge_segment_results(per_segment_results, segments_total=len(segments))

    envelope = _build_envelope(
        prepared.participants,
        prepared.source_id,
        prepared.source_type,
        prepared.source_date,
        prepared.text,
        None,
        registry_context,
        vocabulary,
    )
    return backend.extract(envelope, prompt, active_sink, tenant)


def _extract_one_source(
    manifest_source: Any,
    backend: Backend,
    prompt: str,
    active_sink: UsageSink,
    tenant: str,
    *,
    segment: bool,
    corpus_root: Path | str,
    max_segment_tokens: int,
    segment_overlap_tokens: int,
    registry_context: list[RegistryContextEntry] | None = None,
    vocabulary: Vocabulary | None = None,
    samples: int = 1,
) -> _SourceOutcome:
    """`samples` (default 1, today's unchanged behavior): the number of
    INDEPENDENT extraction passes to run over this source, unioned into one
    `ExtractionResult` via `merge_segment_results`'s existing per-source
    dedup rule (same type, overlapping entities, locators within window,
    claim similarity above its own 0.6 threshold -- `portolan.extract.
    segment._facts_match`) -- the same dedup already used to merge a
    segmented source's per-segment results, reused here to merge a
    source's per-SAMPLE results instead. Cheap self-consistency check: with
    temperature not fixed at 0 across every provider in this checkout (see
    `01-spec/harness-review-2026-09-18.md`, determinism section), N
    independent runs of the SAME source can disagree, and `--samples N`
    measures whether their union recovers more planted facts than one run
    alone, and what that costs in invented facts."""
    source_start = time.perf_counter()
    try:
        prepared = load_prepared_source(manifest_source, corpus_root=corpus_root)

        pass_results = [
            _extract_one_pass(
                prepared,
                backend,
                prompt,
                active_sink,
                tenant,
                segment=segment,
                max_segment_tokens=max_segment_tokens,
                segment_overlap_tokens=segment_overlap_tokens,
                registry_context=registry_context,
                vocabulary=vocabulary,
            )
            for _ in range(max(1, samples))
        ]
        result = (
            pass_results[0]
            if len(pass_results) == 1
            else merge_segment_results(pass_results, segments_total=len(pass_results))
        )
    except Exception as exc:  # noqa: BLE001 - one bad source must not sink the whole run
        # Group 3: the first real bench run crashed outright the moment one
        # source's extraction raised. A source that fails is recorded and
        # skipped, not fatal to the rest of the run.
        return _SourceOutcome(
            manifest_source.id,
            None,
            f"{type(exc).__name__}: {exc}",
            time.perf_counter() - source_start,
            "",
        )

    return _SourceOutcome(
        manifest_source.id, result, None, time.perf_counter() - source_start, prepared.text
    )


def run_extraction(
    manifest: Manifest,
    backend: Backend,
    *,
    source_ids: list[str] | None = None,
    segment: bool = True,
    corpus_root: Path | str = DEFAULT_CORPUS_ROOT,
    system_prompt: str | None = None,
    sink: UsageSink | None = None,
    tenant: str = TENANT,
    max_segment_tokens: int = 4000,
    segment_overlap_tokens: int = 200,
    concurrency: int = 1,
    backend_factory: Callable[[], Backend] | None = None,
    truth: Truth | None = None,
    samples: int = 1,
) -> ExtractionRunResult:
    """Extract every (or, with `source_ids`, a chosen subset of) source in
    `manifest`, using `corpus_loader` for the locator-annotated text and
    `portolan.extract.segment` to segment long sources and merge their
    per-segment results back into one `ExtractionResult` per source.

    `samples` (default 1, today's unchanged behavior; determinism section,
    `01-spec/harness-review-2026-09-18.md`): run each source's (optionally
    segmented) extraction this many independent times and union the
    results via `merge_segment_results`'s existing dedup rule
    (`_extract_one_source`'s own docstring). Every sample is its own
    `backend.extract()` call -- tokens/cost naturally sum across samples in
    `sink` (a real cost, not a reporting artifact), and per-source wall
    time is the sum of all `samples` passes, both exactly reflecting what
    running `--samples N` for real costs.

    Every `backend.extract()` call records a `UsageEvent` (tokens, cost)
    into `sink` (a fresh `InMemorySink` if none is given); wall time is
    measured per source and in total. Both are attached to the returned
    `ExtractionRunResult` for `score_extraction` to read, along with the
    exact prepared text each source was extracted from (`source_texts`).

    `truth`, when given, fills each source's envelope with `registry_context`
    (the account and participant contacts `ManifestSource.accounts`/
    `.participants` already say belong to it, resolved to id/name/aliases
    against the truth table) and `vocabulary` (the supplier's static
    product/competitor names) -- see `_registry_context_for` /
    `_vocabulary_from_truth`'s docstrings for why: without this, diagnosis
    found a real produced fact never names the account at all, since
    nothing in the envelope says which account a call belongs to and a real
    call routinely never says the hospital's own name out loud. `None`
    (the default) reproduces the exact old behavior -- an empty
    `registry_context`/`vocabulary` on every envelope -- so every caller
    that doesn't pass `truth` (including every existing test) is unaffected.

    `concurrency` (default 1: today's sequential behavior, unchanged) runs
    sources through a `ThreadPoolExecutor` of that many workers -- each
    `backend.extract()` call is 100s-plus of I/O wait against the Agent SDK
    CLI subprocess (see `portolan/llm/agent_sdk.py`), so this is purely an
    I/O-bound speedup, not a CPU one. When `backend_factory` is given, each
    worker thread calls it exactly once and keeps the `Backend` it returns
    for every source that thread handles -- for `LLMClientBackend` this
    means a fresh `AgentSDKClient` (and so a fresh `claude_agent_sdk.query()`
    subprocess-per-call chain) per worker, never one shared across threads.
    Without a factory, every worker shares the single `backend` argument,
    which is what `FakeBackend` (no subprocess, no per-instance state to
    race on) wants for a concurrency test.
    """
    prompt = system_prompt if system_prompt is not None else build_system_prompt()
    active_sink = sink if sink is not None else InMemorySink()
    wanted = set(source_ids) if source_ids is not None else {src.id for src in manifest.sources}
    wanted_sources = [src for src in manifest.sources if src.id in wanted]

    # Computed once for the whole run (identical for every source) rather
    # than per source, and left `None` (-> `_build_envelope`'s own empty
    # defaults) when no `truth` was given.
    shared_vocabulary = _vocabulary_from_truth(truth) if truth is not None else None

    results: dict[str, ExtractionResult] = {}
    wall_time_s: dict[str, float] = {}
    failed_sources: dict[str, str] = {}
    source_texts: dict[str, str] = {}
    run_start = time.perf_counter()

    def _record(outcome: _SourceOutcome) -> None:
        if outcome.error is not None:
            failed_sources[outcome.source_id] = outcome.error
            return
        wall_time_s[outcome.source_id] = outcome.elapsed_s
        results[outcome.source_id] = outcome.result  # type: ignore[assignment]
        source_texts[outcome.source_id] = outcome.text

    if concurrency <= 1:
        for manifest_source in wanted_sources:
            registry_context = (
                _registry_context_for(manifest_source, truth) if truth is not None else None
            )
            _record(
                _extract_one_source(
                    manifest_source,
                    backend,
                    prompt,
                    active_sink,
                    tenant,
                    segment=segment,
                    corpus_root=corpus_root,
                    max_segment_tokens=max_segment_tokens,
                    segment_overlap_tokens=segment_overlap_tokens,
                    registry_context=registry_context,
                    vocabulary=shared_vocabulary,
                    samples=samples,
                )
            )
    else:
        # thread-local: `backend_factory()` runs once per worker thread
        # (its first task on that thread), not once per source -- a
        # `threading.local` cache keyed by nothing but the current thread
        # gives every task on that thread the same backend instance.
        import threading

        local = threading.local()

        def _worker(manifest_source: Any) -> _SourceOutcome:
            if backend_factory is not None:
                thread_backend = getattr(local, "backend", None)
                if thread_backend is None:
                    thread_backend = backend_factory()
                    local.backend = thread_backend
            else:
                thread_backend = backend
            registry_context = (
                _registry_context_for(manifest_source, truth) if truth is not None else None
            )
            return _extract_one_source(
                manifest_source,
                thread_backend,
                prompt,
                active_sink,
                tenant,
                segment=segment,
                corpus_root=corpus_root,
                max_segment_tokens=max_segment_tokens,
                segment_overlap_tokens=segment_overlap_tokens,
                registry_context=registry_context,
                vocabulary=shared_vocabulary,
                samples=samples,
            )

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            for outcome in pool.map(_worker, wanted_sources):
                _record(outcome)

    total_wall_time_s = time.perf_counter() - run_start
    return ExtractionRunResult(
        results,
        wall_time_s=wall_time_s,
        failed_sources=failed_sources,
        sink=active_sink,
        total_wall_time_s=total_wall_time_s,
        source_texts=source_texts,
    )


# --------------------------------------------------------------------------
# score_extraction
# --------------------------------------------------------------------------


class JudgeClient(Protocol):
    """Structurally the same shape as `portolan.llm.LLMClient` (build-
    contracts.md), duplicated here rather than imported: wave A1 owns that
    module and it may not exist yet. Any object with a matching `complete`
    method works, including a hand-rolled fake in a test."""

    def complete(
        self,
        *,
        system: str,
        user: str,
        json_schema: dict | None = None,
        model: str | None = None,
        surface: str,
        max_tokens: int = 4096,
    ) -> Any: ...


class ProducedFactRecord(BaseModel):
    """Everything about one fact a backend actually produced, kept verbatim
    for offline diagnosis: `01-spec/quality-loop.md`'s bench is meant to
    answer "which model does best" and "how do we prompt better" without a
    re-run, and the first real run's report only ever showed a truncated
    claim string -- diagnosing it required re-running the extractor by hand
    to see the raw output. This is that raw output, one record per produced
    fact, independent of whether it matched anything planted."""

    model_config = ConfigDict(extra="forbid")

    id: str
    type: str
    claim: str
    entities: list[str] = Field(default_factory=list)
    locator: str
    quote: str | None = None
    confidence: float


def _produced_record(fact: Fact) -> ProducedFactRecord:
    return ProducedFactRecord(
        id=fact.id,
        type=fact.type.value,
        claim=fact.claim,
        entities=list(fact.entities),
        locator=fact.locator,
        quote=fact.quote,
        confidence=fact.confidence,
    )


_PROPER_NOUN_RE = re.compile(r"\b[A-Z][a-zA-Z']+(?:\s+[A-Z][a-zA-Z']+){0,3}\b")
_NUMBER_LITERAL_RE = re.compile(r"\d[\d,.$%]*")


def _salient_claim_fragments(claim: str) -> set[str]:
    """Numbers/money/percents and capitalized name-like phrases pulled out
    of a claim -- the parts of a durable-language claim (fact rule 2) that
    should show up, at least as substrings, in the source it's really drawn
    from. A claim invented from nothing shares almost none of these with
    any real passage."""
    fragments = set(_NUMBER_LITERAL_RE.findall(claim))
    fragments.update(m.strip() for m in _PROPER_NOUN_RE.findall(claim))
    return {f for f in fragments if f}


# --------------------------------------------------------------------------
# EvidenceContext -- 2026-09-18 sanity audit
# (`01-spec/sanity-audit-2026-09-18.md`, `01-spec/extraction-accuracy-
# campaign-2026-09-18.md`'s Opus sanity rule).
#
# Reading every one of Opus's 51, Sonnet's 16, and Haiku's 8 `invented`-
# labeled facts against their source text (script: `scripts/
# bench_sanity_audit.py`) found ZERO real fabrications and a small, highly
# mechanical set of scorer false-positive causes, almost all downstream of
# one root cause already named in `01-spec/extraction-miss-analysis-
# 2026-09-18.md`: "call transcripts never say the account's own name aloud,
# so the fragment fallback can't find it." The same is true of the
# supplier's OWN name ("Halvard"/"Halvard Medical" -- a rep almost never
# says their own employer's name to the customer on a call) and of a CRM
# row's `as of <date>` framing (the date is the source's own `source_date`
# envelope field, never a column in the row itself). None of these are
# inventions: they are exactly the identity-resolution context
# `_build_envelope`/`run_extraction` already hands the model up front
# (`registry_context`, `vocabulary`, `source_date`) -- a fragment that only
# matches ONE of those, never the raw text, is citing real context, not
# hallucinating.
# --------------------------------------------------------------------------


class EvidenceContext(NamedTuple):
    """Extra grounding available to the fragment fallback, beyond the raw
    source text: every name/alias the model was actually given for this
    source (`registry_context`'s accounts/contacts/internal people, the
    static product/competitor `vocabulary`, the supplier's own name) plus
    the source's own `source_date` (as `{"2026","09","11"}`-shaped
    components, since a hyphenated ISO date in a claim -- `_NUMBER_LITERAL_
    RE` doesn't span `-` -- is extracted as three separate digit
    fragments). Empty (the default) reproduces the pre-2026-09-18 sanity-
    audit behavior exactly: fragments must appear in the raw text alone."""

    context_strings: frozenset[str] = frozenset()
    date_tokens: frozenset[str] = frozenset()


_EMPTY_EVIDENCE_CONTEXT = EvidenceContext()

_POSSESSIVE_RE = re.compile(r"['’]s\b")


def _normalize_context_string(s: str) -> str:
    """Fold `s` onto the same footing for a context/fragment comparison:
    glyph-fold (curly apostrophe, em/en dash), drop a trailing possessive
    (`'s`/`'s`), treat `-`/`_`/`:` as word separators, lowercase, and
    collapse whitespace -- so "Halvard's", "HalvardCare's", and a CRM row's
    "Ashgrove Health --- HalvardCare" all compare on the same footing."""
    folded = _fold_glyphs(_POSSESSIVE_RE.sub("", s)).lower()
    spaced = re.sub(r"[-_:]", " ", folded)
    return re.sub(r"\s+", " ", spaced).strip()


def _context_strings_for_source(manifest_source: Any, truth: Truth) -> frozenset[str]:
    """Every name/alias `_build_envelope` puts in front of the model for
    this one source: `registry_context`'s accounts/contacts/internal
    people (`_registry_context_for`, already source-scoped) plus the
    corpus-wide supplier name and product/competitor vocabulary
    (`truth.table.meta` -- read directly rather than through `Vocabulary`/
    `_vocabulary_from_truth`, which drops each product/competitor's own
    `aliases`, real identity-resolution context read straight from the
    prompt's own source data)."""
    strings: set[str] = set()
    for entry in _registry_context_for(manifest_source, truth):
        strings.add(_normalize_context_string(entry.name))
        for alias in entry.aliases:
            strings.add(_normalize_context_string(alias))
    meta = truth.table.meta or {}
    supplier_name = (meta.get("supplier") or {}).get("name")
    if supplier_name:
        strings.add(_normalize_context_string(supplier_name))
    for group_key in ("products", "competitors"):
        for row in meta.get(group_key, []):
            if row.get("name"):
                strings.add(_normalize_context_string(row["name"]))
            for alias in row.get("aliases", []):
                strings.add(_normalize_context_string(alias))
    return frozenset(s for s in strings if s)


def _date_tokens_for_source(manifest_source: Any) -> frozenset[str]:
    """`{"2026", "09", "11"}` for a source dated 2026-09-11 -- the envelope's
    own `source_date`, split the same way a hyphenated ISO date in a claim
    ends up split by `_NUMBER_LITERAL_RE` (which does not span `-`)."""
    date = getattr(manifest_source, "date", None)
    if date is None:
        return frozenset()
    return frozenset(date.isoformat().split("-"))


def evidence_context_for_source(source_id: str, truth: Truth) -> EvidenceContext:
    """Build the `EvidenceContext` for one source, best-effort: an unknown
    `source_id` (shouldn't happen against a real manifest, but a test
    fixture may not carry one) yields the empty context rather than
    raising, same fallback convention `rescore_unplanted_classifications`
    already uses for a missing/unreadable source text."""
    try:
        manifest_source = truth.manifest.source(source_id)
    except KeyError:
        return _EMPTY_EVIDENCE_CONTEXT
    return EvidenceContext(
        context_strings=_context_strings_for_source(manifest_source, truth),
        date_tokens=_date_tokens_for_source(manifest_source),
    )


_TEXT_GROUNDED = "text"
_CONTEXT_GROUNDED = "context"
_UNGROUNDED = "none"


def _fragment_grounding(fragment: str, text: str, context: EvidenceContext) -> str:
    """Is `fragment` (one salient claim fragment) grounded, and by which
    KIND of signal -- `_TEXT_GROUNDED` (the raw source text itself, in one
    of four increasingly tolerant forms) or `_CONTEXT_GROUNDED` (only
    `EvidenceContext` -- the envelope's own `registry_context`/
    `vocabulary`/supplier name/`source_date`, never the text)? The
    distinction is what 2026-09-18's follow-up audit needed
    (`01-spec/sanity-audit-2026-09-18.md` "closing the vacuous-fragment-
    vote hole"): crediting a fragment via context alone is fine (real
    context, not a fabrication) for ONE fragment among several, but if
    EVERY hit on a claim is context-only, nothing about the claim was ever
    actually checked against the source text -- see `_fragment_vote`.

    Text-grounded, in order: (1) literal, case-sensitive substring of the
    raw text -- the original, unchanged rule. (2) the same, after dropping
    a fragment's trailing sentence/list punctuation (`_NUMBER_LITERAL_RE`
    includes `.`/`,` in its character class, so a number at the end of a
    claim sentence picks up a trailing `.` -- "30." -- that (almost) never
    appears right after the digits in the source). (3) case-insensitive,
    glyph-folded containment (`_normalize_for_containment`) -- an
    abbreviation like "CRM" against a row's own lowercase `crm_id=...`
    field. (4) currency/comma-insensitive: the fragment's bare digits (3+,
    to avoid a coincidental short-run match) appear as a substring of the
    text's own bare digits -- "$2,350,000" and "arr=2350000" are the same
    number. Context-grounded: (5) the fragment's bare digits (any length)
    equal one of the source's own `EvidenceContext.date_tokens` -- a
    hyphenated date fragment ("09", "11") the model is only citing because
    the envelope's `source_date` told it so, not because the raw text
    repeats it. (6) `EvidenceContext.context_strings`: the fragment,
    normalized (`_normalize_context_string`), is a substring of -- or
    contains as a substring -- one registered name/alias/product/
    competitor/supplier string. Every one of (2)-(6) is additive: none
    replace the original literal check, they only widen what counts as
    "grounded" to context the model was actually given."""
    if fragment in text:
        return _TEXT_GROUNDED
    stripped = fragment.rstrip(".,")
    if stripped != fragment and stripped in text:
        return _TEXT_GROUNDED
    # Case-insensitive literal containment: a produced abbreviation like
    # "CRM" (proper-noun-shaped: one capital then letters, per
    # `_PROPER_NOUN_RE`) against a CRM row's own `crm_id=...` field never
    # matches signal (1)'s case-sensitive check even though the row is
    # unmistakably a CRM record -- found in the 2026-09-18 sanity audit
    # rescoring a non-Claude model's output. Folds glyphs too (curly
    # quotes/dashes), matching `_normalize_for_containment`'s treatment of
    # a quote.
    if _normalize_for_containment(stripped) in _normalize_for_containment(text):
        return _TEXT_GROUNDED
    digits = re.sub(r"[^0-9]", "", stripped)
    if len(digits) >= 3 and digits in re.sub(r"[^0-9]", "", text):
        return _TEXT_GROUNDED
    if digits and digits in context.date_tokens:
        return _CONTEXT_GROUNDED
    if context.context_strings:
        frag_norm = _normalize_context_string(stripped)
        if frag_norm:
            for ctx_string in context.context_strings:
                if ctx_string and (frag_norm in ctx_string or ctx_string in frag_norm):
                    return _CONTEXT_GROUNDED
    return _UNGROUNDED


_FRAGMENT_SUPPORT_THRESHOLD = 0.6


def _fragment_vote(
    claim: str, text: str, context: EvidenceContext = _EMPTY_EVIDENCE_CONTEXT
) -> tuple[bool, bool]:
    """`(supported, text_grounded)`. `supported`: the existing 60%
    hit-ratio vote over `claim`'s salient fragments, unchanged. `text_
    grounded`: True iff at least one COUNTED hit was `_TEXT_GROUNDED` --
    i.e. the claim is not being credited on `EvidenceContext` alone with
    zero words of it ever checked against the actual source text. A claim
    whose only salient fragment is a real account/supplier/product name
    (no number, no other proper noun) passes `supported` trivially the
    moment that name is in `registry_context` -- true of every source it's
    ever mentioned in, whether or not THIS claim is. That fragment alone
    is not evidence for the REST of the claim (`01-spec/sanity-audit-
    2026-09-18.md`'s follow-up finding, spot-checking Nova Lite: a soft
    fabrication like "Ashgrove Health is unhappy with support" would
    otherwise pass with the account name as its only, vacuously-context-
    matched fragment). `_evidence_basis` uses `text_grounded` to decide
    whether the fragment vote alone is enough, or whether it also needs
    `_window_supported` to corroborate."""
    fragments = _salient_claim_fragments(claim)
    if not fragments:
        return False, False
    groundings = [_fragment_grounding(f, text, context) for f in fragments]
    hits = sum(1 for g in groundings if g != _UNGROUNDED)
    supported = (hits / len(fragments)) >= _FRAGMENT_SUPPORT_THRESHOLD
    text_grounded = any(g == _TEXT_GROUNDED for g in groundings)
    return supported, text_grounded


def _fragments_supported(
    claim: str, text: str, context: EvidenceContext = _EMPTY_EVIDENCE_CONTEXT
) -> bool:
    """The salient-fragment fallback, factored out so it can be checked
    (and reported) independently of whether a quote was present -- 2026-
    09-18 fix instructions item 3 ("if the quote fails both [exact and
    fuzzy], still fall through to the fragment check instead of returning
    False"). `context` widens what counts as a "hit" per fragment
    (`_fragment_grounding`) without changing the 60% vote threshold
    itself. This is the plain boolean vote only -- see `_fragment_vote`
    for the `text_grounded` companion `_evidence_basis` actually gates on;
    kept as its own function because existing callers (tests included)
    want a plain bool and because a vacuously context-only "supported"
    vote is still real information (reported in `_evidence_basis`'s
    docstring), just not sufficient alone."""
    supported, _ = _fragment_vote(claim, text, context)
    return supported


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")

# How many consecutive sentences form one "window" the fallback below
# checks a claim's content words against, and what fraction of those words
# (deduplicated, stopwords dropped, number-words folded to digits via the
# same `_normalize_claim_tokens` the claim-similarity score uses) must
# appear somewhere in the best-scoring window. 2026-09-18 sanity audit:
# every one of Opus's 51 / Sonnet's 16 / Haiku's 8 `invented`-labeled facts
# was resolved by `EvidenceContext` and the punctuation/currency/date
# normalization above ALONE -- none of them needed this fallback to flip.
# Read across all 75 anyway (as a diagnostic, `var/bench/2026-09-18-sanity/
# opus-invented-audit.md`), the best-window overlap of every one of those
# (all independently confirmed genuine, all already credited some other
# way) ranged 0.20-0.94, median 0.47 -- too wide, on its own, to certify a
# safe floor, because none of those data points were load-bearing (the
# fragment/context vote had already passed). Two constructed adversarial
# controls (`tests/test_bench_extraction.py`'s
# `test_window_fallback_rejects_constructed_fabrications`) -- a fact naming
# a real registered account/supplier but inventing everything else
# (a competitor, a dollar figure, a future date) against an unrelated real
# transcript -- score 0.02-0.10 on this same metric: nowhere near even the
# lowest genuine value. `_WINDOW_CONTENT_THRESHOLD` is set to match
# `_FRAGMENT_SUPPORT_THRESHOLD` (0.6) precisely because this fallback is
# the last resort before `invented` and has none of the fragment check's
# specificity (numbers, proper nouns) -- it should only fire when nearly
# every content word of the claim genuinely reappears together in one
# small window, not merely "more than not."
_WINDOW_SENTENCES = 3
_WINDOW_CONTENT_THRESHOLD = 0.6


def _claim_content_words(text: str) -> set[str]:
    return {tok for tok in _normalize_claim_tokens(text) if tok not in _STOPWORDS and len(tok) > 1}


def _window_supported(claim: str, text: str) -> bool:
    """Last-resort fallback (2026-09-18 sanity audit, fix instructions item
    3's third bullet): even with no quote and no fragment (a purely
    descriptive claim -- no number, no name -- has nothing for
    `_fragments_supported` to check at all, since `_salient_claim_fragments`
    returns empty and that function returns False outright), most of the
    claim's own content words reappearing together in one small window of
    the source is still real textual grounding. Deliberately coarser than
    the fragment check (no per-token weighting) and gated at the same 0.6
    vote threshold for exactly that reason -- see `_WINDOW_CONTENT_
    THRESHOLD`'s comment."""
    claim_words = _claim_content_words(claim)
    if not claim_words:
        return False
    sentences = [s for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    if not sentences:
        return False
    for i in range(len(sentences)):
        window = " ".join(sentences[i : i + _WINDOW_SENTENCES])
        window_words = _claim_content_words(window)
        if not window_words:
            continue
        hits = sum(1 for w in claim_words if w in window_words)
        if (hits / len(claim_words)) >= _WINDOW_CONTENT_THRESHOLD:
            return True
    return False


# The mutually exclusive answers to "what, if anything, ties this
# produced fact back to the source text" -- kept as a plain frozenset
# (rather than an enum) since `UnplantedFact.evidence_basis` stores the
# string directly and nothing else needs to iterate it as a type.
# "window" added 2026-09-18 sanity audit (`_window_supported`).
#
# "judge_supported"/"judge_unsupported"/"judge_unverifiable" added
# 2026-09-18, wave 3A (`01-spec/extraction-accuracy-campaign-2026-09-18.md`,
# owner decision 08:27): the disputed middle the deterministic tiers above
# cannot decide -- a produced fact whose evidence is genuinely spread
# across the source too widely for the fragment/window check to certify
# (`opus-invented-audit.md`'s `S-call-n03/f1`, "stretched/busy winter",
# split across two turns ~70 sentences apart). `_evidence_basis` itself is
# untouched (still deterministic-only, still exactly the function every
# existing test above calls directly) -- these three values are set by
# `portolan.bench.judge.apply_evidence_judge`, a POST-PROCESSING pass over
# an already-scored `ExtractionScore` that only ever looks at entries
# `_evidence_basis` already called "none", reached ONLY when every
# deterministic path failed. `judge_supported`: the judge says the source
# supports the claim AND its cited sentences, read back, actually contain
# at least 60% of the claim's own content words (the "string-checkable"
# requirement -- a judge verdict is never trusted on its say-so alone).
# `judge_unsupported`: the judge says no. `judge_unverifiable`: the judge
# said yes but its own citation didn't hold up on the string check --
# reported as a THIRD bucket, distinct from both "confirmed genuine" and
# "confirmed invented" (see `UnplantedFact.is_unverifiable`), never folded
# into either the deterministic or the adjudicated invented count.
EVIDENCE_BASIS_VALUES: frozenset[str] = frozenset(
    {
        "exact_quote",
        "fuzzy_quote",
        "fragments",
        "window",
        "judge_supported",
        "judge_unsupported",
        "judge_unverifiable",
        "none",
    }
)


def _evidence_basis(
    claim: str,
    quote: str | None,
    text: str,
    context: EvidenceContext = _EMPTY_EVIDENCE_CONTEXT,
) -> str:
    """`_plausible_in_source`'s full decision, exposed as the five-way
    verdict `UnplantedFact.evidence_basis` reports instead of just the
    collapsed bool `in_source_text`.

    2026-09-18 fix (`01-spec/model-bench-2026-09-18.md` "Corrected
    scoring"): the original version returned `False` -- invented -- the
    moment a *given* quote failed exact containment, never trying the
    fragment fallback that a no-quote fact always got. A near-verbatim
    quote (one changed word, a punctuation difference, a curly apostrophe,
    an ellipsis, a dropped filler word) failed that exact check even
    though the fact plainly has a textual basis, so real extractions were
    mislabeled `invented` -- and disproportionately so for a model that
    quotes loosely rather than copy-pasting, which inflated its invented
    count relative to a model that fabricates outright but happens to
    quote verbatim when it does.

    2026-09-18 sanity audit fix (`01-spec/sanity-audit-2026-09-18.md`):
    `context` (default empty -- exact prior behavior) lets the fragment
    check credit a fragment the model could only have gotten from the
    envelope's own `registry_context`/`vocabulary`/`source_date`
    (`EvidenceContext`), and a claim with no fragment at all -- or one
    whose fragments still fail everything else -- gets one more try via
    `_window_supported`. Every signal now gets a chance; a fact is
    `invented` only when exact containment, fuzzy containment, the
    fragment fallback (context-aware), AND the sentence-window fallback
    all fail.

    2026-09-18 follow-up fix, same doc ("closing the vacuous-fragment-
    vote hole"): the fragment vote is accepted OUTRIGHT only when at
    least one of its hits is grounded in the actual source text
    (`_fragment_vote`'s `text_grounded`). A claim whose ONLY salient
    fragments are names -- an account, the supplier, a product -- and
    whose every hit came from `EvidenceContext` rather than the text
    itself was passing with literally zero words of the claim ever
    checked against the source (spot-checked on Nova Lite's saved output:
    a real account name is real context, but it is not evidence for
    whatever the rest of an otherwise-unverified claim says about that
    account). When the vote passes but is context-only, `_window_
    supported` must ALSO agree before the fact is credited -- exactly the
    same bar a fragment-free claim already had to clear.
    """
    if not text:
        return "none"
    if quote and quote.strip():
        if _quote_supported_by_text(quote, text):
            return "exact_quote"
        if _fuzzy_quote_containment(quote, text) >= _FUZZY_QUOTE_THRESHOLD:
            return "fuzzy_quote"
    fragments_ok, text_grounded = _fragment_vote(claim, text, context)
    if fragments_ok and text_grounded:
        return "fragments"
    if _window_supported(claim, text):
        return "window"
    return "none"


def _plausible_in_source(
    claim: str,
    quote: str | None,
    text: str,
    context: EvidenceContext = _EMPTY_EVIDENCE_CONTEXT,
) -> bool:
    """Heuristic (no model call needed) for "is this produced-but-unplanted
    fact really something the source text says" vs. an outright invention.

    `01-spec/quality-loop.md`'s ask: distractor sources (D01-D04) can
    contain real trivia that simply isn't a planted fact -- S-call-002's
    genuine scheduling chatter, S-email-021-thread's quoted-reply content
    the generator wrote beyond F21's own definition -- and the report
    should say which unplanted facts are that (`unplanted_plausible`)
    versus which have no textual basis at all (`invented`). A verbatim (or
    near-verbatim, see `_evidence_basis`) quote settles it outright
    (quotes are required for the five evidentiary types, so most facts
    that need one will have one); absent a quote, or when the quote fails
    both containment checks, most of the claim's own numbers/dates/money/
    percent figures and capitalized name-like phrases should appear
    somewhere in the text (directly, or via `context` -- see
    `EvidenceContext`), or -- last resort -- most of its content words
    should reappear together in one small window of the text.
    """
    return _evidence_basis(claim, quote, text, context) != "none"


class MatchedFact(BaseModel):
    """A planted fact matched to a produced fact."""

    model_config = ConfigDict(extra="forbid")

    fact_id: str
    source_id: str
    source_type: str = ""
    fact_type: str
    produced_type: str
    credit: float
    similarity: float
    produced_claim: str
    # Full produced-fact record (redundant with produced_type/produced_claim
    # above, kept for those two fields' existing callers) -- entities,
    # locator, quote and confidence are what diagnosis needs and the two
    # summary fields never carried.
    produced: ProducedFactRecord | None = None
    # Set only when `score_extraction(..., judge=...)` promoted this match
    # from `unmatched` on a judge PARAPHRASE verdict (the mechanical
    # criteria alone said "claim_similarity" was the only thing that
    # failed) -- distinguishes a judge-assisted match from a purely
    # mechanical one when reading the report.
    via_judge: bool = False


class UnmatchedFact(BaseModel):
    """A planted fact no produced fact matched, with the closest candidate
    (by claim similarity, regardless of whether it passed) and every
    criterion that candidate failed."""

    model_config = ConfigDict(extra="forbid")

    fact_id: str
    source_id: str
    source_type: str = ""
    fact_type: str
    claim: str | None
    locator: str
    failed_criteria: list[str] = Field(default_factory=list)
    best_candidate_claim: str | None = None
    best_similarity: float = 0.0
    judge_verdict: str | None = None
    # Full record of the closest candidate produced fact (None only when the
    # source produced nothing at all) -- so diagnosis can see its entities/
    # locator/quote/confidence, not just the claim text.
    best_candidate: ProducedFactRecord | None = None


class UnplantedFact(BaseModel):
    """A produced fact matching no planted fact. Not automatically an
    error: `is_distractor_leak` marks the real defect (anything at all
    produced from a D01-D04 distractor source); everything else is just
    reported (a legitimate fact in noise, an over-split duplicate, or a
    genuine invention) rather than judged.

    `in_source_text` (quality-loop.md's ask: distinguish "trivia the corpus
    actually contains" from "not in the text at all") is set by
    `_evidence_basis`'s heuristic -- true when the produced fact's quote
    (exact or near-verbatim, `evidence_basis in {"exact_quote",
    "fuzzy_quote"}`) or, failing that, most of its salient numbers/dates/
    names (`evidence_basis == "fragments"`) actually appears in the source
    text `run_extraction` fed the backend. This is what tells apart, e.g.,
    a real leak of S-call-002's genuine scheduling chatter (in the text,
    just not a planted fact -- D04 says this source should yield NOTHING,
    so it's still leakage, but not a hallucination) from a fact with no
    textual basis at all (`classification == "invented"`, `evidence_basis
    == "none"`).

    `evidence_basis` (added 2026-09-18, the invented-classifier fix): which
    of the four signals `_evidence_basis` found, one of
    `EVIDENCE_BASIS_VALUES` -- kept alongside the collapsed `in_source_text`
    bool so a report can show, per model, how each supported fact was
    actually supported (a model that quotes loosely leans on
    `fuzzy_quote`; one that fabricates fluently but never quotes leans on
    `fragments` or `none`)."""

    model_config = ConfigDict(extra="forbid")

    source_id: str
    fact_type: str
    claim: str
    locator: str
    is_distractor_leak: bool = False
    produced_id: str = ""
    in_source_text: bool = False
    evidence_basis: str = "none"
    # 2026-09-18 wave 3A: set only by `portolan.bench.judge.
    # apply_evidence_judge`, only on an entry whose judge verdict was
    # `supported` but whose cited sentences failed the string check (see
    # `EVIDENCE_BASIS_VALUES`'s docstring) -- a THIRD classification,
    # neither "confirmed genuine" nor "confirmed invented". Always False
    # on a deterministic-only score (every score before this wave, and
    # every score run with `--judge off`).
    is_unverifiable: bool = False

    @property
    def classification(self) -> str:
        if self.is_unverifiable:
            return "unverifiable"
        return "unplanted_plausible" if self.in_source_text else "invented"


class TypeBreakdown(BaseModel):
    model_config = ConfigDict(extra="forbid")

    planted: int = 0
    matched_credit: float = 0.0
    produced: int = 0
    unplanted: int = 0

    @property
    def recall(self) -> float | None:
        return self.matched_credit / self.planted if self.planted else None

    @property
    def precision(self) -> float | None:
        return self.matched_credit / self.produced if self.produced else None


class ExtractionScore(BaseModel):
    """Everything `report.py` needs to render an extraction bench run."""

    model_config = ConfigDict(extra="forbid")

    sources_scored: int
    planted_total: int
    matched_count: int
    produced_total: int
    recall: float
    precision: float | None
    distractor_leakage: int
    quote_presence_rate: float | None
    quote_missing: list[str] = Field(default_factory=list)
    locator_presence_rate: float
    locator_missing: list[str] = Field(default_factory=list)
    entity_accuracy: float | None
    entity_accuracy_checked: int
    # Added 2026-09-17 (company-context wave, 01-spec/company-context-
    # eval-2026-09-17.md): of the facts the bipartite match already
    # accepted as the SAME fact (`matched`), what fraction got the exact
    # `fact_type` right -- not the 0.5-credit near-match pairs `_type_
    # credit` treats as compatible for matching purposes, exact string
    # equality against the truth table's own type. `None` when nothing
    # matched at all, same "n/a" convention as `precision`/`entity_
    # accuracy` above.
    fact_type_accuracy: float | None = None
    # -- Lenient recall (2026-09-18 harness review, `01-spec/
    # harness-review-2026-09-18.md`) -----------------------------------
    # A SECOND, separately-computed recall number, never a replacement for
    # the STRICT `recall` above: type-agnostic, locator ignored, claim
    # similarity floor 0.20 instead of 0.30 (`_evaluate_pair_lenient`).
    # Exists to make visible how much of the gap between "recall" and 100%
    # is a scorer-strictness near-miss versus a fact genuinely absent from
    # the model's output -- read the two side by side, never `lenient_
    # recall` alone as if it were the score. All fields default so an
    # older saved report (from before this field existed) still validates.
    lenient_recall: float | None = None
    lenient_matched_count: int = 0
    # Planted fact_ids that are a MISS under strict (absent from
    # `matched`) but WOULD be credited under the lenient matcher -- the
    # "near-miss credited" column: a wrong fact type (kept a strict miss
    # on purpose -- that's a real extractor error) or a claim similarity
    # between 0.20 and 0.30, or a locator the strict window rejected.
    # Populated fact-by-fact so a later wave can read exactly which facts
    # these are without re-deriving the set.
    near_miss_credited_fact_ids: list[str] = Field(default_factory=list)
    near_miss_credited_count: int = 0
    lenient_by_fact_type: dict[str, TypeBreakdown] = Field(default_factory=dict)
    lenient_by_source_type: dict[str, TypeBreakdown] = Field(default_factory=dict)
    matched: list[MatchedFact] = Field(default_factory=list)
    unmatched: list[UnmatchedFact] = Field(default_factory=list)
    unplanted: list[UnplantedFact] = Field(default_factory=list)
    by_fact_type: dict[str, TypeBreakdown] = Field(default_factory=dict)
    by_source_type: dict[str, TypeBreakdown] = Field(default_factory=dict)
    # Every produced fact, verbatim, keyed by source_id -- so a diagnosis
    # pass never needs a re-run to see what the model actually said (type,
    # claim, entities, locator, quote, confidence). Populated for every
    # source in `results`, including ones with no planted facts at all.
    produced_by_source: dict[str, list[ProducedFactRecord]] = Field(default_factory=dict)
    tokens_input: int = 0
    tokens_output: int = 0
    tokens_cached: int = 0
    cost_estimate: float = 0.0
    wall_time_s: float = 0.0
    # {source_id: error message} for a source run_extraction caught an
    # exception on and skipped (Group 3 of the review fix wave). Empty
    # for a plain `{source_id: ExtractionResult}` dict passed to
    # score_extraction directly (a FakeBackend canned-result test), since
    # only run_extraction ever populates this.
    failed_sources: dict[str, str] = Field(default_factory=dict)

    # Fix wave 2026-09-22 (`01-spec/bench-deepseek-v41-flash-2026-09-22.md`):
    # sum of `ExtractionResult.coerced_sentiment_count` across every
    # source in `results` -- how many facts had `sentiment` defaulted
    # (`null` -> `0`) or coerced (a string like "negative" mapped onto
    # the int scale) rather than kept as-sent. Was previously invisible
    # in the scorecard entirely; defaults to `0` so an older saved report
    # (from before this field existed) still validates.
    coerced_sentiment: int = 0

    # -- Layered judge (2026-09-18, wave 3A) ----------------------------
    # All `None`/0/empty on a score never passed through `portolan.bench.
    # judge.apply_evidence_judge`/`apply_match_judge` (every score before
    # this wave, and every `--judge off` run) -- backward compatible with
    # every saved report on disk. `invented_deterministic` is what the
    # unplanted-fact count would be reading `evidence_basis` alone before
    # any judge tier runs (the old, sole "invented" number); `invented_
    # adjudicated` is the same count after the judge tier resolved some of
    # the disputed middle -- never a replacement for the deterministic
    # number, always reported beside it. `unverifiable_count` is its own
    # third bucket (`UnplantedFact.is_unverifiable`), never folded into
    # either invented number. `judge_calls`/`judge_cache_hits` count real
    # model calls vs. cache hits for the evidence tier; `judge_matches_
    # calls`/`judge_matches_cache_hits` the same for the match tier.
    invented_deterministic: int | None = None
    invented_adjudicated: int | None = None
    unverifiable_count: int = 0
    judge_calls: int = 0
    judge_cache_hits: int = 0
    # `adjudicated_recall` (match-side, `--judge-matches`): STRICT `recall`
    # plus one full credit per planted fact_id in `near_miss_credited_
    # fact_ids` the judge confirmed states the same fact as the closest
    # produced candidate -- never a replacement for `recall` (the
    # headline) or `lenient_recall`, reported beside both.
    adjudicated_recall: float | None = None
    adjudicated_matched_fact_ids: list[str] = Field(default_factory=list)
    judge_matches_calls: int = 0
    judge_matches_cache_hits: int = 0

    # -- Reporting-fix fields (2026-09-18, bench-provenance wave, 01-spec/
    # bench-provenance-2026-09-18.md) -----------------------------------
    # `report.py` prints several "pct (a / b)" parentheticals as the
    # reader's proof they can multiply a/b back out to the percentage.
    # Before this wave, some of those parentheticals printed the wrong
    # pair -- an instance/fact_id COUNT standing in where a CREDIT sum
    # belonged (recall/precision give half credit on a near-miss type
    # match), or a denominator with no numerator at all (fact-type/entity
    # accuracy). These fields are the exact numerator each such
    # percentage is computed from, captured once here so report.py never
    # has to re-derive (or mis-derive) them. None of this changes what
    # any metric IS -- every field below was already computed as a local
    # variable inside this function; this just keeps it on the object.
    # All default so an older saved report (from before this field
    # existed) still validates through `--rescore`/`--rescore-dir`.
    #
    # Numerator of `recall`: `sum(credit_by_fact_id.values())`.
    recall_credit_sum: float = 0.0
    # Numerator of `lenient_recall`: `sum(lenient_credit_by_fact_id.
    # values())`. Lenient credit is always 1.0 per matched fact_id today
    # (no near-miss partial credit on the lenient side), so this equals
    # `lenient_matched_count` on every score computed so far -- kept as
    # its own field anyway, both for symmetry with `recall_credit_sum`
    # and in case a later wave adds lenient partial credit.
    lenient_credit_sum: float = 0.0
    # Numerator of `precision`: `sum(m.credit for m in matched)`.
    precision_credit_sum: float = 0.0
    # `len(matched)` -- the count of matched-PAIR INSTANCES (one entry per
    # source/produced-fact assignment in `matched`), as opposed to
    # `matched_count` (distinct PLANTED fact_ids with credit, `len(
    # credit_by_fact_id)`). The two coincide unless the same fact_id was
    # matched more than once across sources. `fact_type_accuracy`'s own
    # denominator is `len(matched)`, not `matched_count` -- this is the
    # number that actually belongs beside it in the report.
    matched_instance_count: int = 0
    # Numerator of `fact_type_accuracy`: `sum(1 for m in matched if
    # m.produced_type == m.fact_type)`.
    fact_type_correct_count: int = 0
    # Numerator of `entity_accuracy`: the count of resolved entity
    # mentions whose registry_id matched the truth table's own, alongside
    # the existing `entity_accuracy_checked` denominator.
    entity_accuracy_correct: int = 0


def _type_credit(planted_type: str, produced_type: str) -> tuple[bool, float]:
    """(compatible, credit). Exact match is full credit; `other` on either
    side, or one of the three adjacent-pair confusions in
    `_NEAR_MATCH_TYPE_PAIRS`, is compatible at half credit; anything else is
    incompatible."""
    if planted_type == produced_type:
        return True, 1.0
    if _OTHER_TYPE in (planted_type, produced_type):
        return True, 0.5
    if frozenset((planted_type, produced_type)) in _NEAR_MATCH_TYPE_PAIRS:
        return True, 0.5
    return False, 0.0


def _is_salient_token(token: str) -> bool:
    """Numbers and dates carry a fact's identity -- "four days late" and
    "two days late" are different facts even though every other word
    matches -- far more than surrounding prose does."""
    return bool(_DIGIT_RE.search(token))


def _weighted_claim_tokens(text: str, entity_tokens: frozenset[str]) -> Counter[str]:
    """Tokenize and weight: numbers/dates 3x, the planted fact's own entity
    name/alias words 2x (so two well-formed claims about *different*
    accounts don't score high on connective prose alone), everything else
    (minus stopwords) 1x."""
    counts: Counter[str] = Counter()
    for token in _normalize_claim_tokens(text):
        if token in _STOPWORDS:
            continue
        if _is_salient_token(token):
            weight = 3
        elif token in entity_tokens:
            weight = 2
        else:
            weight = 1
        counts[token] += weight
    return counts


def _claim_similarity(
    a: str | None, b: str | None, entity_tokens: frozenset[str] = frozenset()
) -> float:
    """Weighted token-set overlap: a multiset (Counter) intersection over
    union, not `difflib.SequenceMatcher`'s character-run ratio.

    The planted claim is written in durable third person ("Larkspur
    Memorial Hospital reported a Tidewater T2 consumable delivery arriving
    four days late on 2026-08-04"); a real extractor's claim about the same
    fact rarely repeats that phrasing verbatim ("Larkspur Memorial's
    Tidewater T2 order arrived 4 days late on 2026-08-04"). Those two
    sentences share almost no long character run -- exactly what
    `SequenceMatcher.ratio()` rewards -- but share nearly every salient
    token. `_CLAIM_SIMILARITY_THRESHOLD` is calibrated to this scale in
    `var/bench/diag/diagnosis.md`, not the old character-ratio one.
    """
    if not a or not b:
        return 0.0
    ca = _weighted_claim_tokens(a, entity_tokens)
    cb = _weighted_claim_tokens(b, entity_tokens)
    if not ca or not cb:
        return 0.0
    intersection = sum((ca & cb).values())
    union = sum((ca | cb).values())
    return intersection / union if union else 0.0


def _entity_name_tokens(entity_ids: list[str], truth: Truth) -> frozenset[str]:
    """Every normalized word in `entity_ids`' canonical names/aliases
    (`_claim_similarity`'s entity-token weighting) -- e.g. `larkspur-
    memorial` contributes {"larkspur", "memorial", "hospital", ...} from
    its own name plus every alias variant."""
    tokens: set[str] = set()
    for entity_id in entity_ids:
        for name in truth.entity_names(entity_id):
            tokens.update(_normalize_claim_tokens(name))
    return frozenset(tokens)


def _evaluate_pair(
    planted: PlantedFact, produced: Fact, truth: Truth
) -> tuple[bool, float, list[str], float]:
    """(is_match, credit, failed_criteria, claim_similarity) for one
    (planted, produced) pair, per the four AND'd criteria: compatible type,
    entity overlap, locator within window, claim similarity >= threshold."""
    type_ok, credit = _type_credit(planted.type, produced.type)
    entities_ok = truth.entities_overlap(planted.entities, produced.entities)
    locator_ok = locator_matches_any(planted.source_type, planted.locators, produced.locator)
    entity_tokens = _entity_name_tokens(planted.entities, truth)
    similarity = _claim_similarity(planted.claim, produced.claim, entity_tokens)
    similarity_ok = similarity >= _CLAIM_SIMILARITY_THRESHOLD

    failed: list[str] = []
    if not type_ok:
        failed.append("type")
    if not entities_ok:
        failed.append("entities")
    if not locator_ok:
        failed.append("locator")
    if not similarity_ok:
        failed.append("claim_similarity")

    is_match = type_ok and entities_ok and locator_ok and similarity_ok
    return is_match, (credit if is_match else 0.0), failed, similarity


def _evaluate_pair_lenient(
    planted: PlantedFact, produced: Fact, truth: Truth
) -> tuple[bool, float]:
    """(is_match, similarity) under the LENIENT matcher (2026-09-18 harness
    review, `01-spec/harness-review-2026-09-18.md`): type-agnostic (any
    produced type counts as the same fact -- a wrong type is a real
    extractor error and stays a miss in the STRICT score this is never
    allowed to replace, `ExtractionScore.recall`, but the underlying fact
    was still found), locator ignored entirely (the window rules exist to
    keep two DIFFERENT same-account facts apart, which claim similarity
    already does most of that job on its own), similarity floor lowered
    from 0.30 to `_LENIENT_CLAIM_SIMILARITY_THRESHOLD`. Entities are still
    required: an unresolved or wrong entity mention is judged a real
    extraction gap, not a scorer-strictness artifact, so it is required in
    both matchers. No partial credit here (unlike `_type_credit`) -- the
    lenient score answers one binary question, "was this fact found at
    all", not "how well was it typed"."""
    if not truth.entities_overlap(planted.entities, produced.entities):
        return False, 0.0
    entity_tokens = _entity_name_tokens(planted.entities, truth)
    similarity = _claim_similarity(planted.claim, produced.claim, entity_tokens)
    return similarity >= _LENIENT_CLAIM_SIMILARITY_THRESHOLD, similarity


_FACT_ID_RE = re.compile(r"^([A-Za-z]*)(\d+)([A-Za-z]*)$")


def _fact_id_sort_key(fact_id: str) -> tuple[str, int, str]:
    """Natural sort for truth-table fact ids (`F6` before `F11b` before
    `F60`, not lexical `F11b` < `F6` < `F60`) -- purely cosmetic, for a
    stable and readable `near_miss_credited_fact_ids` list."""
    m = _FACT_ID_RE.match(fact_id)
    if not m:
        return (fact_id, 0, "")
    prefix, digits, suffix = m.groups()
    return (prefix, int(digits), suffix)


def _touch(bucket: dict[str, TypeBreakdown], key: str) -> TypeBreakdown:
    return bucket.setdefault(key, TypeBreakdown())


def score_extraction(
    truth: Truth,
    results: dict[str, ExtractionResult],
    *,
    judge: JudgeClient | None = None,
) -> ExtractionScore:
    """Score `results` (as returned by `run_extraction`, or any
    `{source_id: ExtractionResult}` mapping — e.g. `FakeBackend` canned
    results built directly in a test) against `truth`.

    Only sources present in `results` are scored, so a partial run (via
    `run_extraction(..., source_ids=[...])`) scores against exactly the
    planted facts in those sources, not the whole corpus.
    """
    # An `ExtractionRunResult` (a real `run_extraction` call) carries the
    # exact prepared text each source was extracted from; a plain dict
    # (canned `FakeBackend` results built directly in a test) carries none,
    # in which case every `UnplantedFact.in_source_text` is False rather
    # than erroring -- same `getattr` fallback pattern as `.sink` below.
    source_texts: dict[str, str] = dict(getattr(results, "source_texts", {}) or {})

    matched: list[MatchedFact] = []
    unmatched: list[UnmatchedFact] = []
    unplanted: list[UnplantedFact] = []
    by_fact_type: dict[str, TypeBreakdown] = {}
    by_source_type: dict[str, TypeBreakdown] = {}

    quote_required_total = 0
    quote_present_total = 0
    quote_missing: list[str] = []
    locator_present_total = 0
    locator_missing: list[str] = []
    produced_total = 0

    entity_checked = 0
    entity_correct = 0

    distractor_source_ids = truth.distractor_source_ids()

    # `EvidenceContext` per source, built lazily and cached (2026-09-18
    # sanity audit): every source's registry_context/vocabulary/date is
    # cheap to derive from `truth` but there is no reason to redo it once
    # per produced fact when a source can carry a dozen.
    evidence_context_cache: dict[str, EvidenceContext] = {}

    def _context_for(source_id: str) -> EvidenceContext:
        if source_id not in evidence_context_cache:
            evidence_context_cache[source_id] = evidence_context_for_source(source_id, truth)
        return evidence_context_cache[source_id]

    # Recall is measured per unique fact_id, not per (source, fact_id)
    # placement: a fact planted in two sources (e.g. F02, `cross_source_
    # dedup`) is one fact for recall purposes — it counts as recalled if
    # extraction found it in at least one of its sources; that a real
    # extractor found it in both is a merge-stage question
    # (`expectations.merge.cross_source_single_claim`, scored by
    # `graph.py`), not an extraction-recall one. `by_source_type` stays at
    # placement granularity on purpose: it answers "how well does this
    # channel do", which is a per-placement question even for a fact
    # planted in more than one channel.
    credit_by_fact_id: dict[str, float] = {}
    type_by_fact_id: dict[str, str] = {}
    unique_planted_ids_seen: set[str] = set()
    produced_by_source: dict[str, list[ProducedFactRecord]] = {}
    # Lenient-matcher bookkeeping, parallel to the strict dicts above --
    # see `_evaluate_pair_lenient` and `ExtractionScore.lenient_recall`.
    lenient_credit_by_fact_id: dict[str, float] = {}
    lenient_by_fact_type: dict[str, TypeBreakdown] = {}
    lenient_by_source_type: dict[str, TypeBreakdown] = {}

    # Fix wave 2026-09-22 (`01-spec/bench-deepseek-v41-flash-2026-09-22.md`):
    # `parse_lenient` coercing a fact's `sentiment` (a `null` default or a
    # string like "negative" mapped onto the int scale) was previously
    # invisible in the scorecard -- summed across every source's result
    # here so the report can show how often it happens.
    coerced_sentiment_total = 0

    for source_id, result in results.items():
        coerced_sentiment_total += result.coerced_sentiment_count
        planted_list = truth.planted(source_id)
        produced_list = list(result.facts)
        produced_total += len(produced_list)
        produced_by_source[source_id] = [_produced_record(f) for f in produced_list]
        source_type = planted_list[0].source_type if planted_list else None
        if source_type is None:
            # No planted facts for this source (noise, or a distractor):
            # still need its source_type for the by_source_type bucket and
            # for distractor-leak reporting.
            try:
                source_type = truth.manifest.source(source_id).type
            except KeyError:
                source_type = "unknown"

        # -- locator / quote presence, over every produced fact -----------
        for produced in produced_list:
            if produced.locator and produced.locator.strip():
                locator_present_total += 1
            else:
                locator_missing.append(f"{source_id}:{produced.id}")
            if produced.type in QUOTE_REQUIRED_TYPES:
                quote_required_total += 1
                if produced.quote and produced.quote.strip():
                    quote_present_total += 1
                else:
                    quote_missing.append(f"{source_id}:{produced.id}")

        # -- entity accuracy: a resolved registry_id against the truth ----
        for mention in result.entities:
            if mention.registry_id is None:
                continue
            expected = truth.identify_entity(mention.mention)
            if expected is None:
                continue  # not in the truth table's index; not checkable
            entity_checked += 1
            if expected == mention.registry_id:
                entity_correct += 1

        # -- bipartite match, planted facts vs. produced facts -------------
        pair_scores: list[tuple[bool, float, float, PlantedFact, int, list[str]]] = []
        for planted in planted_list:
            for i, produced in enumerate(produced_list):
                is_match, credit, failed, similarity = _evaluate_pair(planted, produced, truth)
                pair_scores.append((is_match, credit, similarity, planted, i, failed))

        # Greedy 1:1 assignment: strong (passing, high-credit, high-
        # similarity) pairs claimed first, so one produced fact cannot be
        # spent twice on two different planted facts.
        pair_scores.sort(key=lambda t: (not t[0], -t[1], -t[2]))
        matched_fact_ids: set[str] = set()
        assigned_produced: set[int] = set()

        for is_match, credit, similarity, planted, i, _failed in pair_scores:
            if not is_match:
                continue
            if planted.fact_id in matched_fact_ids or i in assigned_produced:
                continue
            matched_fact_ids.add(planted.fact_id)
            assigned_produced.add(i)
            produced = produced_list[i]
            matched.append(
                MatchedFact(
                    fact_id=planted.fact_id,
                    source_id=source_id,
                    source_type=planted.source_type,
                    fact_type=planted.type,
                    produced_type=produced.type.value,
                    credit=credit,
                    similarity=similarity,
                    produced_claim=produced.claim,
                    produced=_produced_record(produced),
                )
            )
            _touch(by_source_type, planted.source_type).matched_credit += credit
            credit_by_fact_id[planted.fact_id] = max(
                credit_by_fact_id.get(planted.fact_id, 0.0), credit
            )

        # -- lenient bipartite match, same source, independent of strict --
        # (`_evaluate_pair_lenient`: type-agnostic, locator ignored,
        # similarity >= 0.20). A separate greedy 1:1 assignment over the
        # same planted/produced lists -- a produced fact can be "spent" by
        # both the strict and the lenient match (they're answering two
        # different questions), just not twice within the same one.
        lenient_pair_scores: list[tuple[bool, float, PlantedFact, int]] = []
        for planted in planted_list:
            for i, produced in enumerate(produced_list):
                is_match, similarity = _evaluate_pair_lenient(planted, produced, truth)
                lenient_pair_scores.append((is_match, similarity, planted, i))
        lenient_pair_scores.sort(key=lambda t: (not t[0], -t[1]))
        lenient_matched_fact_ids: set[str] = set()
        lenient_assigned_produced: set[int] = set()
        for is_match, _similarity, planted, i in lenient_pair_scores:
            if not is_match:
                continue
            if planted.fact_id in lenient_matched_fact_ids or i in lenient_assigned_produced:
                continue
            lenient_matched_fact_ids.add(planted.fact_id)
            lenient_assigned_produced.add(i)
            _touch(lenient_by_source_type, planted.source_type).matched_credit += 1.0
            lenient_credit_by_fact_id[planted.fact_id] = 1.0

        for planted in planted_list:
            type_by_fact_id[planted.fact_id] = planted.type
            unique_planted_ids_seen.add(planted.fact_id)
            _touch(by_source_type, planted.source_type).planted += 1
            _touch(lenient_by_source_type, planted.source_type).planted += 1
            if planted.fact_id in matched_fact_ids:
                continue
            # "Best" candidate ranks by fewest failed criteria first, then
            # by similarity -- not similarity alone. Two produced facts
            # about the same account/product share entity-name tokens
            # (weighted 2x in `_claim_similarity`) whether or not they're
            # the same FACT: an unrelated produced fact naming the same
            # account can out-score, on raw text overlap, the real match's
            # own awkwardly-phrased claim, even though it fails type and/or
            # locator outright. Ranking by failure count first means the
            # reported "closest candidate" -- and judge eligibility, which
            # requires `failed_criteria == ["claim_similarity"]` exactly --
            # reflect the produced fact that is actually closest overall,
            # not just the one with the most incidental word overlap.
            best: tuple[int, float, list[str], Fact | None] = (
                4,
                0.0,
                ["no_produced_facts"],
                None,
            )
            for produced in produced_list:
                _, _, failed, similarity = _evaluate_pair(planted, produced, truth)
                is_better = len(failed) < best[0] or (
                    len(failed) == best[0] and similarity >= best[1]
                )
                if is_better:
                    best = (len(failed), similarity, failed, produced)
            _, best_similarity, best_failed, best_produced = best
            unmatched.append(
                UnmatchedFact(
                    fact_id=planted.fact_id,
                    source_id=source_id,
                    source_type=planted.source_type,
                    fact_type=planted.type,
                    claim=planted.claim,
                    locator=planted.locator,
                    failed_criteria=best_failed,
                    best_candidate_claim=(best_produced.claim if best_produced else None),
                    best_similarity=best_similarity,
                    best_candidate=(
                        _produced_record(best_produced) if best_produced is not None else None
                    ),
                )
            )

        for i, produced in enumerate(produced_list):
            if i in assigned_produced:
                continue
            is_leak = source_id in distractor_source_ids
            basis = _evidence_basis(
                produced.claim,
                produced.quote,
                source_texts.get(source_id, ""),
                _context_for(source_id),
            )
            unplanted.append(
                UnplantedFact(
                    source_id=source_id,
                    fact_type=produced.type.value,
                    claim=produced.claim,
                    locator=produced.locator,
                    is_distractor_leak=is_leak,
                    produced_id=produced.id,
                    in_source_text=basis != "none",
                    evidence_basis=basis,
                )
            )
            _touch(by_fact_type, produced.type.value).produced += 1
            _touch(by_fact_type, produced.type.value).unplanted += 1
            _touch(by_source_type, source_type).produced += 1
            _touch(by_source_type, source_type).unplanted += 1

        for i in assigned_produced:
            produced = produced_list[i]
            _touch(by_fact_type, produced.type.value).produced += 1
            _touch(by_source_type, source_type).produced += 1

    if judge is not None:
        promoted, unmatched = _run_judge_and_promote(judge, unmatched)
        # The produced fact a promotion is built from may already sit in
        # `unplanted` (the greedy match never assigned it, since claim
        # similarity was below threshold) -- drop it there so the report
        # doesn't show the same produced fact as both matched and
        # unplanted. `by_fact_type`/`by_source_type`'s `.produced`/
        # `.unplanted` tallies are left as originally counted (still
        # informationally true: the model DID produce it) rather than
        # unwound here.
        promoted_keys = {(m.source_id, m.produced.id) for m in promoted if m.produced is not None}
        unplanted = [u for u in unplanted if (u.source_id, u.produced_id) not in promoted_keys]
        for jm in promoted:
            matched.append(jm)
            _touch(by_source_type, jm.source_type).matched_credit += jm.credit
            credit_by_fact_id[jm.fact_id] = max(credit_by_fact_id.get(jm.fact_id, 0.0), jm.credit)

    # by_fact_type is filled in here (rather than inside the per-source
    # loop, like by_source_type) because its `.planted`/`.matched_credit`
    # need the same unique-fact_id dedup as the top-level recall.
    for fact_id in unique_planted_ids_seen:
        bucket = _touch(by_fact_type, type_by_fact_id[fact_id])
        bucket.planted += 1
        bucket.matched_credit += credit_by_fact_id.get(fact_id, 0.0)
        lenient_bucket = _touch(lenient_by_fact_type, type_by_fact_id[fact_id])
        lenient_bucket.planted += 1
        lenient_bucket.matched_credit += lenient_credit_by_fact_id.get(fact_id, 0.0)

    planted_total = len(unique_planted_ids_seen)
    recall_credit_sum = sum(credit_by_fact_id.values())
    precision_credit_sum = sum(m.credit for m in matched)
    distractor_leakage = sum(1 for u in unplanted if u.is_distractor_leak)

    recall = recall_credit_sum / planted_total if planted_total else 1.0
    precision = precision_credit_sum / produced_total if produced_total else None

    # Lenient recall + the "near-miss credited" column (2026-09-18 harness
    # review): `near_miss_credited_fact_ids` is every planted fact the
    # STRICT match missed entirely (not in `credit_by_fact_id`, i.e. not
    # even the judge could promote it -- `credit_by_fact_id` already
    # reflects any judge promotion by this point) but the LENIENT matcher
    # found. Sorted for a stable, diffable report.
    lenient_credit_sum = sum(lenient_credit_by_fact_id.values())
    lenient_recall = lenient_credit_sum / planted_total if planted_total else 1.0
    near_miss_credited_fact_ids = sorted(
        set(lenient_credit_by_fact_id) - set(credit_by_fact_id), key=_fact_id_sort_key
    )

    # `results.sink` (an `ExtractionRunResult`) is an `InMemorySink` by
    # default and so has `.events`; a plain `dict[source_id,
    # ExtractionResult]` (e.g. hand-built `FakeBackend` canned results in a
    # test) has no `sink` at all, in which case tokens/cost report as zero
    # rather than erroring.
    usage_events = getattr(getattr(results, "sink", None), "events", [])

    # Company-context wave, 2026-09-17: exact fact-type match rate over
    # `matched` -- see `ExtractionScore.fact_type_accuracy`'s own
    # docstring for why this is exact equality, not `_type_credit`'s
    # 0.5-credit near-match tolerance.
    fact_type_correct_count = sum(1 for m in matched if m.produced_type == m.fact_type)
    fact_type_accuracy = fact_type_correct_count / len(matched) if matched else None

    return ExtractionScore(
        sources_scored=len(results),
        planted_total=planted_total,
        matched_count=len(credit_by_fact_id),
        produced_total=produced_total,
        recall=recall,
        precision=precision,
        recall_credit_sum=recall_credit_sum,
        lenient_credit_sum=lenient_credit_sum,
        precision_credit_sum=precision_credit_sum,
        matched_instance_count=len(matched),
        fact_type_correct_count=fact_type_correct_count,
        entity_accuracy_correct=entity_correct,
        distractor_leakage=distractor_leakage,
        quote_presence_rate=(
            quote_present_total / quote_required_total if quote_required_total else None
        ),
        quote_missing=quote_missing,
        locator_presence_rate=(locator_present_total / produced_total if produced_total else 1.0),
        locator_missing=locator_missing,
        entity_accuracy=(entity_correct / entity_checked if entity_checked else None),
        entity_accuracy_checked=entity_checked,
        fact_type_accuracy=fact_type_accuracy,
        lenient_recall=lenient_recall,
        lenient_matched_count=len(lenient_credit_by_fact_id),
        near_miss_credited_fact_ids=near_miss_credited_fact_ids,
        near_miss_credited_count=len(near_miss_credited_fact_ids),
        lenient_by_fact_type=lenient_by_fact_type,
        lenient_by_source_type=lenient_by_source_type,
        matched=matched,
        unmatched=unmatched,
        unplanted=unplanted,
        by_fact_type=by_fact_type,
        by_source_type=by_source_type,
        produced_by_source=produced_by_source,
        tokens_input=sum(e.input_tokens for e in usage_events),
        tokens_output=sum(e.output_tokens for e in usage_events),
        tokens_cached=sum(e.cached_tokens for e in usage_events),
        # cost_estimate is None on an event whose model_key had no pricing
        # row (review finding F3 -- metered() now records such an event
        # instead of raising); treat it as 0 for the run total rather than
        # letting one unpriced call blow up the whole sum.
        cost_estimate=sum(e.cost_estimate or 0.0 for e in usage_events),
        wall_time_s=getattr(results, "total_wall_time_s", 0.0),
        failed_sources=dict(getattr(results, "failed_sources", {}) or {}),
        coerced_sentiment=coerced_sentiment_total,
    )


# --------------------------------------------------------------------------
# rescore_unplanted_classifications -- 2026-09-18 fix instructions item 4:
# "check python -m portolan.bench extract --help for a re-score-from-saved-
# output mode and add one if it does not exist." `python -m portolan.bench
# extract --rescore <saved-report.json>` (see __main__.py's `_cmd_rescore`)
# wraps this.
# --------------------------------------------------------------------------


def rescore_unplanted_classifications(
    score: ExtractionScore,
    manifest: Manifest,
    *,
    corpus_root: Path | str = DEFAULT_CORPUS_ROOT,
    truth: Truth | None = None,
) -> ExtractionScore:
    """Re-run ONLY `UnplantedFact.in_source_text`/`evidence_basis` (via the
    current, fixed `_evidence_basis`) for every entry in
    `score.unplanted`, against the corpus on disk -- no backend call, and
    no other field touched: recall, precision, matched, unmatched,
    tokens, cost, wall time, and every other count on `score` is carried
    forward exactly as saved. This is the whole point -- 2026-09-18's
    finding was that recall is unaffected, only the invented
    classification was wrong.

    A previously *written* `ExtractionScore` (loaded from a saved bench
    report's own JSON, `ExtractionScore.model_validate(data["extraction"])`)
    never kept the exact locator-annotated text `run_extraction` fed the
    backend for each source (`ExtractionRunResult.source_texts` is a
    run-time-only attribute of the run result, not part of the score that
    gets written to disk) -- but the corpus on disk is static, so
    re-deriving each source's prepared text via `load_prepared_source`
    against `manifest` reproduces exactly what the original run scored
    against, as long as `corpus/out` hasn't changed since. Quotes are read
    from `score.produced_by_source` (the one place a saved report keeps
    them; `UnplantedFact` itself never carried one).

    `truth` (optional, 2026-09-18 sanity audit): when given, every fragment
    check also gets that source's `EvidenceContext` (registry_context/
    vocabulary/source_date) -- same widening `score_extraction` always
    applies now. `None` (the default, and every call before this wave)
    reproduces the exact prior behavior: text-only fragment matching, no
    context credit. `_cmd_rescore` passes a loaded `Truth`; a caller with
    only a `Manifest` in hand (or a test that doesn't care) is unaffected.
    """
    source_ids = {u.source_id for u in score.unplanted}
    texts: dict[str, str] = {}
    for source_id in source_ids:
        try:
            manifest_source = manifest.source(source_id)
            texts[source_id] = load_prepared_source(manifest_source, corpus_root=corpus_root).text
        except (KeyError, FileNotFoundError, ValueError):
            texts[source_id] = ""

    def _quote_for(u: UnplantedFact) -> str | None:
        for rec in score.produced_by_source.get(u.source_id, []):
            if rec.id == u.produced_id:
                return rec.quote
        return None

    context_cache: dict[str, EvidenceContext] = {}

    def _context_for(source_id: str) -> EvidenceContext:
        if truth is None:
            return _EMPTY_EVIDENCE_CONTEXT
        if source_id not in context_cache:
            context_cache[source_id] = evidence_context_for_source(source_id, truth)
        return context_cache[source_id]

    rescored: list[UnplantedFact] = []
    for u in score.unplanted:
        text = texts.get(u.source_id, "")
        basis = _evidence_basis(u.claim, _quote_for(u), text, _context_for(u.source_id))
        rescored.append(
            u.model_copy(update={"in_source_text": basis != "none", "evidence_basis": basis})
        )

    return score.model_copy(update={"unplanted": rescored})


def _fact_from_record(rec: ProducedFactRecord) -> Fact:
    """Rebuild a `Fact` good enough to re-run the bipartite match against,
    from a saved `ProducedFactRecord` (`ExtractionScore.produced_by_source`
    -- every scored run already keeps the raw produced fact, per-source,
    verbatim). `sentiment` (required, no default on `Fact`) is set to 0 and
    `speaker`/`valid_from`/`valid_to`/`severity` are left `None`: none of
    these were ever persisted on `ProducedFactRecord` and nothing
    `score_extraction` reads touches them -- only `type`, `claim`,
    `entities`, `locator`, `quote`, `confidence` feed the matcher."""
    return Fact(
        id=rec.id,
        type=rec.type,
        claim=rec.claim,
        entities=list(rec.entities),
        locator=rec.locator,
        quote=rec.quote,
        confidence=rec.confidence,
        sentiment=0,
    )


def rescore_full(
    score: ExtractionScore,
    manifest: Manifest,
    truth: Truth,
    *,
    corpus_root: Path | str = DEFAULT_CORPUS_ROOT,
    judge: JudgeClient | None = None,
) -> ExtractionScore:
    """Full re-score from a saved report's `produced_by_source` -- no
    backend call, no new model output. Unlike `rescore_unplanted_
    classifications` (which touches ONLY `UnplantedFact.in_source_text`/
    `evidence_basis` and carries every other field forward exactly as
    saved), this rebuilds a `Fact` per produced record
    (`_fact_from_record`) and re-runs `score_extraction` end to end against
    the CURRENT matcher code -- recall, precision, fact-type accuracy,
    strict AND lenient recall, near-miss-credited, by_fact_type/
    by_source_type, matched/unmatched/unplanted, all of it.

    This is what makes a matcher or locator fix (e.g. the 2026-09-18 RSS
    locator-format fix in `portolan.bench.locators`) show up in a
    re-score of OLD saved output, which `rescore_unplanted_
    classifications` deliberately does not do (recall was, by design, the
    one thing that fix left untouched).

    Two things a saved report cannot reconstruct and this does NOT
    attempt to: `tokens_input`/`tokens_output`/`tokens_cached`/
    `cost_estimate`/`wall_time_s` (no raw `UsageEvent`s are persisted) and
    `entity_accuracy`/`entity_accuracy_checked`/`entity_accuracy_correct`
    (no raw `EntityMention` list is persisted, only the collapsed rate and
    its numerator/denominator). All three are carried forward from
    `score` unchanged, same convention `rescore_unplanted_
    classifications` already uses for what it can't recompute either.
    """
    source_texts: dict[str, str] = {}
    results: dict[str, ExtractionResult] = {}
    for source_id, records in score.produced_by_source.items():
        try:
            manifest_source = manifest.source(source_id)
            source_texts[source_id] = load_prepared_source(
                manifest_source, corpus_root=corpus_root
            ).text
        except (KeyError, FileNotFoundError, ValueError):
            source_texts[source_id] = ""
        results[source_id] = ExtractionResult(
            source_id=source_id,
            summary="",
            entities=[],
            facts=[_fact_from_record(r) for r in records],
            actions=[],
            coverage=Coverage(segments_read=1, segments_total=1),
        )

    run_result = ExtractionRunResult(
        results,
        wall_time_s={},
        sink=InMemorySink(),
        total_wall_time_s=0.0,
        failed_sources=dict(score.failed_sources),
        source_texts=source_texts,
    )
    rescored = score_extraction(truth, run_result, judge=judge)
    return rescored.model_copy(
        update={
            "tokens_input": score.tokens_input,
            "tokens_output": score.tokens_output,
            "tokens_cached": score.tokens_cached,
            "cost_estimate": score.cost_estimate,
            "wall_time_s": score.wall_time_s,
            "entity_accuracy": score.entity_accuracy,
            "entity_accuracy_checked": score.entity_accuracy_checked,
            # `entity_accuracy_correct` is the numerator behind those two
            # -- equally unrecomputable here (no raw `EntityMention` list
            # persisted; `_fact_from_record` never sets `entities` on the
            # rebuilt `ExtractionResult`, so a fresh `score_extraction`
            # call above found 0 entities to check), so it's carried
            # forward from the saved score exactly like its two siblings.
            "entity_accuracy_correct": score.entity_accuracy_correct,
        }
    )


def _run_judge_and_promote(
    judge: JudgeClient, unmatched: list[UnmatchedFact]
) -> tuple[list[MatchedFact], list[UnmatchedFact]]:
    """Grade the unmatched remainder with a strong-model judge (off by
    default; `01-spec/quality-loop.md`'s "a judge call grades only the
    unmatched remainder"). Every entry gets `judge_verdict` set, best-effort
    (a judge failure is recorded on the fact rather than raised, since the
    judge pass is a diagnostic extra, not part of the mechanical score).

    Only entries whose *sole* mechanical failure was `claim_similarity` are
    candidates for promotion -- type, entities and locator already agreed,
    so the only open question is "is this the same fact in different
    words", exactly what the judge is asked. A PARAPHRASE verdict on one of
    those promotes it: moved into the returned `matched` list (credit from
    `_type_credit`, since type compatibility was never in question) and
    dropped from the returned `unmatched` list. Every other entry -- a
    genuine MISS/NOT_A_FACT verdict, a judge error, or an entry that also
    failed type/entities/locator -- stays in `unmatched` unchanged; a
    locator or entity mismatch is a more mechanical, meaningful failure
    that a judge's read of the claim text alone should not paper over.
    """
    system = (
        "You are grading a candidate miss from an extraction bench. Given "
        "the planted fact and the closest candidate the extractor actually "
        "produced, answer with exactly one word: MISS (the extractor "
        "genuinely missed this fact), PARAPHRASE (the candidate is the same "
        "fact in different words and should not count as a miss), or "
        "NOT_A_FACT (neither text describes a real fact)."
    )
    promoted: list[MatchedFact] = []
    remaining: list[UnmatchedFact] = []

    for entry in unmatched:
        user = (
            f"Planted fact ({entry.fact_type}): {entry.claim}\n"
            f"Closest candidate produced: {entry.best_candidate_claim or '(nothing produced)'}"
        )
        try:
            result = judge.complete(system=system, user=user, surface="audit", max_tokens=16)
            verdict = (getattr(result, "text", None) or "").strip().upper()
            entry.judge_verdict = verdict or None
        except Exception as exc:  # pragma: no cover - defensive, judge is optional
            entry.judge_verdict = f"judge_error: {exc}"

        eligible = (
            entry.failed_criteria == ["claim_similarity"] and entry.best_candidate is not None
        )
        if eligible and entry.judge_verdict == "PARAPHRASE":
            candidate = entry.best_candidate
            assert candidate is not None  # `eligible` just checked this
            _, credit = _type_credit(entry.fact_type, candidate.type)
            promoted.append(
                MatchedFact(
                    fact_id=entry.fact_id,
                    source_id=entry.source_id,
                    source_type=entry.source_type,
                    fact_type=entry.fact_type,
                    produced_type=candidate.type,
                    credit=credit,
                    similarity=entry.best_similarity,
                    produced_claim=candidate.claim,
                    produced=candidate,
                    via_judge=True,
                )
            )
        else:
            remaining.append(entry)

    return promoted, remaining
