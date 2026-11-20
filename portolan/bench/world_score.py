"""Extraction bench on a v2 world (`01-spec/world-generator-v2-2026-09-22.md`
section 7, "Extraction on test worlds"): run a backend over a world's
sources and score it against the world's span labels.

Matching is by character span, not by claim wording. A produced fact is
placed in the prepared text by its quote (`resolve_quote_span`, exact tier
first), or, when it has no quote, by the locator unit it cites. It then
lands in exactly one category:

- **matched**: its span overlaps a label (IoU >= 0.5, or one span inside
  the other, the rule spec section 3.3 uses) and the types are compatible
  (`extraction._type_credit`: equal, `other`, or one of the three adjacent
  pairs);
- **type_confusion**: it overlaps a label, but the types are incompatible
  (reported as a confusion, not an invention);
- **decoy_hit**: it overlaps a planted decoy and no label;
- **off_span**: it overlaps nothing labeled.

The three numbers, per fact type and overall:

1. **Quote-verbatim**: produced facts whose quote resolves exactly
   (`exact`) or after glyph and whitespace folding (`normalized`), out of
   all produced facts. A stitched quote (two spans joined by "...") and a
   missing quote both count as failures.
2. **Substantiated**: matched facts plus disputed facts three judges
   ruled real (`portolan.bench.disputes`), out of all produced facts.
3. **Omission versus invention**: omission is labeled facts (one per
   `(doc_id, fact_id)`) with no matched produced fact, out of labeled
   facts. Invention is decoy hits plus off-span facts the judges did not
   overturn; type confusions are shown beside them.

Breakdowns: by fact type and document kind (both sides), by label source,
directive and reference mode (label side; a produced fact inherits the
label it hit, else `unlabeled`), and by noise operation (a fact touched by
two operations counts in both rows).
"""

from __future__ import annotations

import datetime as dt
import re
import time
from collections import Counter
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from portolan.bench.extraction import ExtractionRunResult, _type_credit
from portolan.bench.world_v2 import (
    LABEL_SOURCE_ORDER,
    LoadedWorld,
    WorldDecoy,
    WorldLabel,
    WorldSource,
    locator_region,
)
from portolan.extract.backends import Backend
from portolan.extract.schema import (
    Coverage,
    EntityKind,
    ExtractionResult,
    Fact,
    FactType,
    Participant,
    RegistryContextEntry,
    Vocabulary,
)
from portolan.extract.segment import merge_segment_results
from portolan.metering import InMemorySink, UsageSink
from portolan.training.quote_span import resolve_quote_span

__all__ = [
    "BREAKDOWNS",
    "MATCH_IOU",
    "LabelOutcome",
    "ProducedOutcome",
    "Tally",
    "WorldScore",
    "classify_world",
    "perfect_world_results",
    "quote_tier",
    "run_world_extraction",
    "score_world",
    "spans_match",
]

TENANT = "bench"
MATCH_IOU = 0.5  # spec section 3.3
BREAKDOWNS: tuple[str, ...] = (
    "fact_type",
    "doc_kind",
    "label_source",
    "directive",
    "reference_mode",
    "noise",
)
UNLABELED = "unlabeled"

QuoteTier = Literal["exact", "normalized", "stitched", "unresolved", "missing"]
Category = Literal["matched", "type_confusion", "decoy_hit", "off_span"]
DISPUTED_CATEGORIES: frozenset[str] = frozenset({"decoy_hit", "off_span"})


# --------------------------------------------------------------------------
# Spans
# --------------------------------------------------------------------------


def span_iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    inter = min(a[1], b[1]) - max(a[0], b[0])
    if inter <= 0:
        return 0.0
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def spans_match(a: tuple[int, int], b: tuple[int, int]) -> bool:
    """IoU >= 0.5, or one span inside the other (spec section 3.3)."""
    if min(a[1], b[1]) - max(a[0], b[0]) <= 0:
        return False
    if span_iou(a, b) >= MATCH_IOU:
        return True
    return (a[0] <= b[0] and b[1] <= a[1]) or (b[0] <= a[0] and a[1] <= b[1])


def quote_tier(quote: str | None, text: str) -> tuple[QuoteTier, tuple[int, int] | None]:
    if not quote or not quote.strip():
        return "missing", None
    resolved = resolve_quote_span(quote, text)
    if resolved is None:
        return "unresolved", None
    if resolved.tier == "exact":
        return "exact", resolved.span
    if resolved.tier.startswith("stitched"):
        return "stitched", resolved.span
    return "normalized", resolved.span


# --------------------------------------------------------------------------
# Outcomes
# --------------------------------------------------------------------------


class ProducedOutcome(BaseModel):
    """One produced fact and where it landed."""

    model_config = ConfigDict(extra="forbid")

    source_id: str
    produced_id: str
    fact_type: str
    claim: str
    quote: str | None = None
    locator: str
    entities: list[str] = Field(default_factory=list)
    quote_tier: str
    span: tuple[int, int] | None = None
    span_basis: Literal["quote", "locator", "none"] = "none"
    category: str
    label_key: str | None = None
    label_type: str | None = None
    exact_type: bool | None = None
    decoy_id: str | None = None
    decoy_class_id: str | None = None
    iou: float | None = None
    duplicate: bool = False
    # Set when the fact was disputed: "supported", "not_supported" or
    # "not_judged" (budget or no judges).
    judge_decision: str | None = None


class LabelOutcome(BaseModel):
    """One labeled fact, `(doc_id, fact_id)`, and whether it was produced."""

    model_config = ConfigDict(extra="forbid")

    key: str
    doc_id: str
    fact_id: str
    fact_type: str
    label_source: str
    directive: str
    reference_mode: str
    doc_kind: str
    noise: list[str] = Field(default_factory=list)
    spans: list[tuple[int, int]] = Field(default_factory=list)
    text: str
    found: bool = False
    matched_produced: list[str] = Field(default_factory=list)
    confused_as: list[str] = Field(default_factory=list)


def _label_units(world: LoadedWorld, source_id: str) -> dict[str, list[WorldLabel]]:
    units: dict[str, list[WorldLabel]] = {}
    for label in world.labels_for(source_id):
        units.setdefault(label.key, []).append(label)
    rank = {name: i for i, name in enumerate(LABEL_SOURCE_ORDER)}
    for group in units.values():
        group.sort(key=lambda lab: rank.get(lab.label_source, len(rank)))
    return units


def _label_outcome(
    world: LoadedWorld, source: WorldSource, group: list[WorldLabel]
) -> LabelOutcome:
    primary = group[0]
    noise: list[str] = []
    for op in [*source.noise, *(op for lab in group for op in lab.noise)]:
        if op not in noise:
            noise.append(op)
    return LabelOutcome(
        key=primary.key,
        doc_id=primary.doc_id,
        fact_id=primary.fact_id,
        fact_type=primary.fact_type,
        label_source=primary.label_source,
        directive=world.directive_for(primary),
        reference_mode=world.reference_mode_for(primary),
        doc_kind=source.kind,
        noise=noise,
        spans=[lab.span for lab in group],
        text=primary.text,
    )


def _best_overlap(
    span: tuple[int, int], candidates: Iterable[tuple[str, tuple[int, int]]]
) -> tuple[str, float] | None:
    best: tuple[str, float] | None = None
    for cid, cspan in candidates:
        if spans_match(span, cspan):
            iou = span_iou(span, cspan)
            if best is None or iou > best[1]:
                best = (cid, iou)
    return best


def _classify_fact(
    fact: Fact,
    source: WorldSource,
    labels: dict[str, LabelOutcome],
    decoys: list[WorldDecoy],
) -> ProducedOutcome:
    tier, span = quote_tier(fact.quote, source.text)
    basis: Literal["quote", "locator", "none"] = "quote" if span is not None else "none"
    if span is None and tier == "missing":
        span = locator_region(source, fact.locator)
        basis = "locator" if span is not None else "none"

    base: dict[str, Any] = dict(
        source_id=source.source_id,
        produced_id=fact.id,
        fact_type=fact.type,
        claim=fact.claim,
        quote=fact.quote,
        locator=fact.locator,
        entities=list(fact.entities),
        quote_tier=tier,
        span=span,
        span_basis=basis,
    )
    if span is None:
        return ProducedOutcome(**base, category="off_span")

    compatible: list[tuple[str, tuple[int, int]]] = []
    incompatible: list[tuple[str, tuple[int, int]]] = []
    for key, lab in labels.items():
        ok, _ = _type_credit(lab.fact_type, fact.type)
        for lspan in lab.spans:
            (compatible if ok else incompatible).append((key, lspan))

    hit = _best_overlap(span, compatible)
    if hit is not None:
        lab = labels[hit[0]]
        return ProducedOutcome(
            **base,
            category="matched",
            label_key=lab.key,
            label_type=lab.fact_type,
            exact_type=lab.fact_type == fact.type,
            iou=round(hit[1], 4),
        )

    confusion = _best_overlap(span, incompatible)
    decoy_hit = _best_overlap(span, ((d.decoy_id, d.span) for d in decoys))
    # A decoy wins a tie: the extractor took the near-fact the decoy was
    # written to be, not the labeled fact beside it.
    if decoy_hit is not None and (confusion is None or decoy_hit[1] >= confusion[1]):
        decoy = next(d for d in decoys if d.decoy_id == decoy_hit[0])
        return ProducedOutcome(
            **base,
            category="decoy_hit",
            decoy_id=decoy.decoy_id,
            decoy_class_id=decoy.decoy_class_id,
            iou=round(decoy_hit[1], 4),
        )
    if confusion is not None:
        lab = labels[confusion[0]]
        return ProducedOutcome(
            **base,
            category="type_confusion",
            label_key=lab.key,
            label_type=lab.fact_type,
            exact_type=False,
            iou=round(confusion[1], 4),
        )
    return ProducedOutcome(**base, category="off_span")


def classify_world(
    world: LoadedWorld, results: dict[str, ExtractionResult]
) -> tuple[list[ProducedOutcome], list[LabelOutcome]]:
    """Every produced fact's category and every labeled fact's outcome,
    for the sources in `results` only."""
    produced: list[ProducedOutcome] = []
    label_outcomes: list[LabelOutcome] = []
    for source_id in sorted(results):
        source = world.sources.get(source_id)
        if source is None:
            continue
        units = {
            key: _label_outcome(world, source, group)
            for key, group in _label_units(world, source_id).items()
        }
        decoys = world.decoys_for(source_id)
        for fact in results[source_id].facts:
            outcome = _classify_fact(fact, source, units, decoys)
            if outcome.category == "matched" and outcome.label_key:
                lab = units[outcome.label_key]
                outcome.duplicate = lab.found
                lab.found = True
                lab.matched_produced.append(outcome.produced_id)
            elif outcome.category == "type_confusion" and outcome.label_key:
                units[outcome.label_key].confused_as.append(outcome.fact_type)
            produced.append(outcome)
        label_outcomes.extend(units[k] for k in sorted(units))
    return produced, label_outcomes


# --------------------------------------------------------------------------
# Tallies
# --------------------------------------------------------------------------


def _rate(n: float, d: float) -> float | None:
    return n / d if d else None


class Tally(BaseModel):
    """Counts for one bucket. Every rate is computed from these by code."""

    model_config = ConfigDict(extra="forbid")

    produced: int = 0
    quote_exact: int = 0
    quote_normalized: int = 0
    quote_stitched: int = 0
    quote_unresolved: int = 0
    quote_missing: int = 0
    matched: int = 0
    duplicates: int = 0
    judge_supported: int = 0
    decoy_hits: int = 0
    decoy_overturned: int = 0
    type_confusions: int = 0
    off_span: int = 0
    off_span_overturned: int = 0
    labeled: int = 0
    labeled_found: int = 0
    labeled_confused: int = 0

    @property
    def quote_verbatim(self) -> float | None:
        return _rate(self.quote_exact + self.quote_normalized, self.produced)

    @property
    def substantiated_count(self) -> int:
        return self.matched + self.judge_supported

    @property
    def substantiated(self) -> float | None:
        return _rate(self.substantiated_count, self.produced)

    @property
    def omitted(self) -> int:
        return self.labeled - self.labeled_found

    @property
    def omission(self) -> float | None:
        return _rate(self.omitted, self.labeled)

    @property
    def invented_count(self) -> int:
        return (self.decoy_hits - self.decoy_overturned) + (
            self.off_span - self.off_span_overturned
        )

    @property
    def invention(self) -> float | None:
        return _rate(self.invented_count, self.produced)

    def add_produced(self, p: ProducedOutcome) -> None:
        self.produced += 1
        tier_field = {
            "exact": "quote_exact",
            "normalized": "quote_normalized",
            "stitched": "quote_stitched",
            "unresolved": "quote_unresolved",
            "missing": "quote_missing",
        }[p.quote_tier]
        setattr(self, tier_field, getattr(self, tier_field) + 1)
        overturned = p.judge_decision == "supported"
        if p.category == "matched":
            self.matched += 1
            self.duplicates += int(p.duplicate)
        elif p.category == "type_confusion":
            self.type_confusions += 1
        elif p.category == "decoy_hit":
            self.decoy_hits += 1
            self.decoy_overturned += int(overturned)
            self.judge_supported += int(overturned)
        else:
            self.off_span += 1
            self.off_span_overturned += int(overturned)
            self.judge_supported += int(overturned)

    def add_label(self, lab: LabelOutcome) -> None:
        self.labeled += 1
        self.labeled_found += int(lab.found)
        self.labeled_confused += int(not lab.found and bool(lab.confused_as))

    def rates(self) -> dict[str, float | None]:
        return {
            "quote_verbatim": self.quote_verbatim,
            "quote_exact": _rate(self.quote_exact, self.produced),
            "quote_normalized": _rate(self.quote_normalized, self.produced),
            "substantiated": self.substantiated,
            "omission": self.omission,
            "invention": self.invention,
            "decoy_hit_rate": _rate(self.decoy_hits - self.decoy_overturned, self.produced),
            "type_confusion_rate": _rate(self.type_confusions, self.produced),
            "off_span_rate": _rate(self.off_span - self.off_span_overturned, self.produced),
        }


def _produced_keys(
    p: ProducedOutcome, labels: dict[str, LabelOutcome], sources: dict[str, WorldSource]
) -> dict[str, list[str]]:
    lab = labels.get(p.label_key) if p.label_key else None
    source = sources[p.source_id]
    return {
        "fact_type": [p.fact_type],
        "doc_kind": [source.kind],
        "label_source": [lab.label_source if lab else UNLABELED],
        "directive": [lab.directive if lab else UNLABELED],
        "reference_mode": [lab.reference_mode if lab else UNLABELED],
        "noise": list(source.noise) or ["none"],
    }


def _label_keys(lab: LabelOutcome) -> dict[str, list[str]]:
    return {
        "fact_type": [lab.fact_type],
        "doc_kind": [lab.doc_kind],
        "label_source": [lab.label_source],
        "directive": [lab.directive],
        "reference_mode": [lab.reference_mode],
        "noise": list(lab.noise) or ["none"],
    }


def build_tallies(
    world: LoadedWorld, produced: list[ProducedOutcome], labels: list[LabelOutcome]
) -> tuple[Tally, dict[str, dict[str, Tally]], dict[str, int]]:
    overall = Tally()
    breakdowns: dict[str, dict[str, Tally]] = {name: {} for name in BREAKDOWNS}
    by_key = {lab.key: lab for lab in labels}
    confusions: Counter[str] = Counter()
    for p in produced:
        overall.add_produced(p)
        for dim, keys in _produced_keys(p, by_key, world.sources).items():
            for key in keys:
                breakdowns[dim].setdefault(key, Tally()).add_produced(p)
        if p.category == "type_confusion":
            confusions[f"{p.label_type} -> {p.fact_type}"] += 1
    for lab in labels:
        overall.add_label(lab)
        for dim, keys in _label_keys(lab).items():
            for key in keys:
                breakdowns[dim].setdefault(key, Tally()).add_label(lab)
    sorted_breakdowns = {dim: dict(sorted(buckets.items())) for dim, buckets in breakdowns.items()}
    return overall, sorted_breakdowns, dict(sorted(confusions.items()))


# --------------------------------------------------------------------------
# The score
# --------------------------------------------------------------------------


class DisputeSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    judge_models: list[str] = Field(default_factory=list)
    estimate_usd: float = 0.0
    ceiling_usd: float = 0.0
    spent_usd: float = 0.0
    max_usd: float = 0.0
    calls: int = 0
    stopped: str | None = None
    # category -> {"disputed", "judged", "overturned", "upheld", "not_judged"}
    by_category: dict[str, dict[str, int]] = Field(default_factory=dict)
    corrections_path: str | None = None


class WorldScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    world_id: str
    world_dir: str
    input_format: str
    date: dt.date
    backend: str
    extractor_model: str | None = None
    prompt_variant: str | None = None
    samples: int = 1
    sources_scored: int = 0
    sources_failed: dict[str, str] = Field(default_factory=dict)
    sources_skipped: dict[str, str] = Field(default_factory=dict)
    labels_used: int = 0
    labels_excluded: dict[str, int] = Field(default_factory=dict)
    decoys_used: int = 0
    decoys_excluded: dict[str, int] = Field(default_factory=dict)
    overall: Tally = Field(default_factory=Tally)
    breakdowns: dict[str, dict[str, Tally]] = Field(default_factory=dict)
    type_confusions: dict[str, int] = Field(default_factory=dict)
    produced: list[ProducedOutcome] = Field(default_factory=list)
    labels: list[LabelOutcome] = Field(default_factory=list)
    disputes: DisputeSummary | None = None
    tokens_input: int = 0
    tokens_output: int = 0
    cost_usd: float = 0.0
    wall_time_s: float = 0.0


def score_world(
    world: LoadedWorld,
    results: dict[str, ExtractionResult],
    *,
    backend: str,
    extractor_model: str | None = None,
    prompt_variant: str | None = None,
    samples: int = 1,
    judge_decisions: dict[tuple[str, str], str] | None = None,
    disputes: DisputeSummary | None = None,
) -> WorldScore:
    """Score `results` against `world`. `judge_decisions` maps
    `(source_id, produced_id)` to a dispute decision; only decoy hits and
    off-span facts read it."""
    produced, labels = classify_world(world, results)
    for p in produced:
        if p.category in DISPUTED_CATEGORIES:
            p.judge_decision = (judge_decisions or {}).get((p.source_id, p.produced_id))
    overall, breakdowns, confusions = build_tallies(world, produced, labels)

    sink = getattr(results, "sink", None)
    events = list(getattr(sink, "events", []) or [])
    scored = [sid for sid in results if sid in world.sources]
    scored_set = set(scored)
    return WorldScore(
        world_id=world.world_id,
        world_dir=str(world.root),
        input_format=world.input_format,
        date=dt.date.today(),
        backend=backend,
        extractor_model=extractor_model,
        prompt_variant=prompt_variant,
        samples=samples,
        sources_scored=len(scored),
        sources_failed=dict(getattr(results, "failed_sources", {}) or {}),
        sources_skipped=dict(world.skipped_sources),
        labels_used=sum(1 for lab in world.labels if lab.doc_id in scored_set),
        labels_excluded=dict(world.excluded_labels),
        decoys_used=sum(1 for d in world.decoys if d.doc_id in scored_set),
        decoys_excluded=dict(world.excluded_decoys),
        overall=overall,
        breakdowns=breakdowns,
        type_confusions=confusions,
        produced=produced,
        labels=labels,
        disputes=disputes,
        tokens_input=sum(e.input_tokens for e in events),
        tokens_output=sum(e.output_tokens for e in events),
        cost_usd=round(sum(e.cost_estimate or 0.0 for e in events), 6),
        wall_time_s=round(float(getattr(results, "total_wall_time_s", 0.0) or 0.0), 3),
    )


# --------------------------------------------------------------------------
# Running a backend over a world
# --------------------------------------------------------------------------


def _registry_context(world: LoadedWorld, source: WorldSource) -> list[RegistryContextEntry]:
    account = world.accounts.get(source.account_id or "")
    if account is None:
        return []
    return [
        RegistryContextEntry(
            name=account.name,
            kind=EntityKind.ACCOUNT,
            registry_id=account.id,
            aliases=list(account.aliases),
        )
    ]


def _extract_source(
    world: LoadedWorld,
    source: WorldSource,
    backend: Backend,
    prompt: str,
    sink: UsageSink,
    samples: int,
) -> ExtractionResult:
    # Imported here: `portolan.pipeline` pulls in the database layer, which
    # the reader and the scorer do not need.
    from portolan.ingest.models import RawSource
    from portolan.pipeline import build_envelopes

    raw = RawSource(
        source_id=source.source_id,
        source_type=source.source_type,
        source_date=source.date,
        text=source.text,
    )
    envelopes = build_envelopes(
        raw,
        source.text,
        participants=[Participant(name=name) for name in source.participants],
        registry_context=_registry_context(world, source),
        vocabulary=Vocabulary(products=world.products, competitors=world.competitors),
    )
    passes: list[ExtractionResult] = []
    for _ in range(max(1, samples)):
        parts = [backend.extract(env, prompt, sink, TENANT) for env in envelopes]
        passes.append(
            parts[0] if len(parts) == 1 else merge_segment_results(parts, segments_total=len(parts))
        )
    if len(passes) == 1:
        return passes[0]
    return merge_segment_results(passes, segments_total=len(passes))


def run_world_extraction(
    world: LoadedWorld,
    backend: Backend,
    *,
    system_prompt: str,
    source_ids: list[str] | None = None,
    samples: int = 1,
    concurrency: int = 1,
    backend_factory: Callable[[], Backend] | None = None,
    sink: UsageSink | None = None,
) -> ExtractionRunResult:
    """Extract every (or each named) source of `world` the way production
    does (`portolan.pipeline.build_envelopes`: one envelope up to 4,000
    tokens, else segments merged back), `samples` times merged. One failed
    source is recorded and skipped, not fatal."""
    active_sink = sink if sink is not None else InMemorySink()
    wanted = (
        sorted(world.sources)
        if source_ids is None
        else [sid for sid in source_ids if sid in world.sources]
    )
    results: dict[str, ExtractionResult] = {}
    wall: dict[str, float] = {}
    failed: dict[str, str] = {}
    texts: dict[str, str] = {}
    start = time.perf_counter()

    import threading

    local = threading.local()

    def one(sid: str) -> tuple[str, ExtractionResult | None, str | None, float]:
        t0 = time.perf_counter()
        use = backend
        if backend_factory is not None and concurrency > 1:
            use = getattr(local, "backend", None) or backend_factory()
            local.backend = use
        try:
            result = _extract_source(
                world, world.sources[sid], use, system_prompt, active_sink, samples
            )
            return sid, result, None, time.perf_counter() - t0
        except Exception as exc:  # noqa: BLE001 - one bad source must not sink the run
            return sid, None, f"{type(exc).__name__}: {exc}", time.perf_counter() - t0

    if concurrency <= 1:
        outcomes = [one(sid) for sid in wanted]
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            outcomes = list(pool.map(one, wanted))
    for sid, result, error, elapsed in outcomes:
        if error is not None or result is None:
            failed[sid] = error or "no result"
            continue
        results[sid] = result
        wall[sid] = elapsed
        texts[sid] = world.sources[sid].text
    return ExtractionRunResult(
        results,
        wall_time_s=wall,
        sink=active_sink,
        total_wall_time_s=time.perf_counter() - start,
        failed_sources=failed,
        source_texts=texts,
    )


# --------------------------------------------------------------------------
# A perfect extractor, for wiring checks (`--backend fake`)
# --------------------------------------------------------------------------

_WORD_RE = re.compile(r"\S+")


def _first_words(text: str, n: int) -> str:
    """`text` cut after its `n`th word, keeping the original characters so
    the result is still an exact substring."""
    words = list(_WORD_RE.finditer(text))
    if len(words) <= n:
        return text.strip()
    return text[words[0].start() : words[n - 1].end()]


def perfect_world_results(
    world: LoadedWorld, source_ids: list[str] | None = None
) -> dict[str, ExtractionResult]:
    """One fact per labeled fact, quoting its label span (at most the 40
    words fact rule 5 allows). Scores quote-verbatim, substantiated and
    omission perfectly by construction."""
    wanted = sorted(world.sources) if source_ids is None else source_ids
    out: dict[str, ExtractionResult] = {}
    for sid in wanted:
        if sid not in world.sources:
            continue
        facts: list[Fact] = []
        for i, (_key, group) in enumerate(sorted(_label_units(world, sid).items()), start=1):
            label = group[0]
            try:
                fact_type = FactType(label.fact_type)
            except ValueError:
                fact_type = FactType.OTHER
            facts.append(
                Fact(
                    id=f"f{i}",
                    type=fact_type,
                    claim=world.claim_for(label) or label.text,
                    entities=[label.account_id] if label.account_id else [],
                    locator=label.locator or "L1",
                    quote=_first_words(label.text, 40),
                    sentiment=0,
                    confidence=0.9,
                )
            )
        out[sid] = ExtractionResult(
            source_id=sid,
            summary="perfect world fixture",
            facts=facts,
            coverage=Coverage(segments_read=1, segments_total=1),
        )
    return out
