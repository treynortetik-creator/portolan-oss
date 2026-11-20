"""Combine pass-1 facts with a reviewer's verdicts and sweep, and score
the result with the bench's existing scorer.

Wave 1D (`01-spec/extraction-accuracy-campaign-2026-09-18.md`). This
module owns the COMBINE step and the scoring wrapper around it; it does
not talk to a model (that's `portolan.extract.review_pass`) and it does
not run the pipeline end to end (that's `scripts/bench_two_pass.py`).

Three combine modes, matching the DESIGN's ablations:

- `combined`: pass-1 facts minus rejects, with fixes applied, plus the
  sweep's missed facts -- the full two-pass result.
- `verify_only`: pass-1 facts minus rejects, with fixes applied. No sweep
  facts added -- isolates what VERIFY alone contributes.
- `sweep_only`: pass-1 facts exactly as pass 1 produced them (every
  reviewer verdict ignored), plus the sweep's missed facts -- isolates
  what the SWEEP alone contributes.

Every mode's final per-source fact list is deduped with the SAME dedup
rule already used to merge one source's own overlapping segments
(`portolan.extract.segment._merge_facts` -- same type, overlapping
entities, locators within window, claim similarity above threshold),
imported here rather than reimplemented, per the DESIGN's "deduped by the
existing dedup."

Ownership note: `portolan/bench/extraction.py` is imported from, never
edited -- `score_extraction` and its supporting types are used exactly as
that module defines them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from portolan.bench.extraction import (
    ExtractionRunResult,
    ExtractionScore,
    ProducedFactRecord,
    score_extraction,
)
from portolan.bench.truth import Truth
from portolan.extract.review_pass import FactVerdict, ReviewedMissedFact, ReviewResult
from portolan.extract.schema import Coverage, ExtractionResult, Fact
from portolan.extract.segment import _merge_facts
from portolan.metering import InMemorySink

CombineMode = Literal["combined", "verify_only", "sweep_only"]
COMBINE_MODES: tuple[CombineMode, ...] = ("combined", "verify_only", "sweep_only")

# Prefix on the synthetic id given to every fact the sweep adds -- lets a
# later reader (score_combined's caller, and the report's "planted facts
# the sweep recovered" count) tell a sweep-added fact apart from a
# surviving/fixed pass-1 fact by id alone, no side channel needed.
SWEEP_FACT_ID_PREFIX = "sweep"


@dataclass
class CombineDiagnostics:
    """What happened combining one source's pass-1 facts with one
    reviewer's verdicts -- everything the report's "five examples of
    correct/wrong rejections" and the reject/invention cross-tab need,
    without re-deriving it from the raw facts later."""

    source_id: str
    mode: CombineMode
    # (pass1_fact_id, reject_reason) for every fact this mode rejected.
    rejected: list[tuple[str, str | None]] = field(default_factory=list)
    fixed: list[str] = field(default_factory=list)
    missed_added: int = 0
    # Human-readable notes on anything dropped for being structurally
    # invalid (e.g. a `fix` that produced a quote over 40 words for a
    # quote-required type) -- mirrors ExtractionResult.parse_lenient's
    # RejectedFact philosophy: record it, don't let it break the run.
    dropped_invalid: list[str] = field(default_factory=list)


def _produced_to_fact(rec: ProducedFactRecord) -> Fact:
    return Fact(
        id=rec.id,
        type=rec.type,
        claim=rec.claim,
        entities=list(rec.entities),
        locator=rec.locator,
        quote=rec.quote,
        sentiment=0,
        confidence=rec.confidence,
    )


def _missed_to_fact(mf: ReviewedMissedFact, index: int) -> Fact:
    return Fact(
        id=f"{SWEEP_FACT_ID_PREFIX}{index}",
        type=mf.type,
        claim=mf.claim,
        entities=list(mf.entities),
        locator=mf.locator,
        quote=mf.quote,
        sentiment=0,
        # Below pass 1's typical produced confidence on purpose: a sweep
        # fact came from a second, narrower pass explicitly hunting for
        # what the first pass missed, not from the same confident read
        # pass 1 gave its own finds. Not read by score_extraction (which
        # scores type/claim/entities/locator/quote, never confidence) --
        # kept plausible only for anything downstream that does read it.
        confidence=0.6,
    )


def _apply_verdict(fact: Fact, verdict: FactVerdict | None) -> tuple[Fact | None, str | None]:
    """`(resulting_fact_or_None, reject_reason_or_None)` for one pass-1
    fact under one verdict. `None` verdict (the reviewer's reply never
    mentioned this fact_id) is treated as `keep` -- an omitted fact is
    not a rejection; see the module docstring's combine modes."""
    if verdict is None or verdict.verdict == "keep":
        return fact, None
    if verdict.verdict == "reject":
        return None, verdict.reject_reason or verdict.reason or "unspecified"

    updates: dict[str, object] = {}
    if verdict.fixed_type:
        updates["type"] = verdict.fixed_type
    if verdict.fixed_claim:
        updates["claim"] = verdict.fixed_claim
    if verdict.fixed_quote is not None:
        updates["quote"] = verdict.fixed_quote
    if not updates:
        return fact, None
    return fact.model_copy(update=updates), None


def combine_source(
    source_id: str,
    pass1_facts: list[ProducedFactRecord],
    review: ReviewResult,
    *,
    mode: CombineMode,
) -> tuple[list[Fact], CombineDiagnostics]:
    """Combine one source's pass-1 facts with one reviewer reply, per
    `mode`. Returns the final (deduped) `Fact` list for this source plus
    a `CombineDiagnostics` record of what happened."""
    diag = CombineDiagnostics(source_id=source_id, mode=mode)
    verdict_by_id = {v.fact_id: v for v in review.verdicts}

    survivors: list[Fact] = []
    for rec in pass1_facts:
        try:
            base_fact = _produced_to_fact(rec)
        except Exception as exc:  # noqa: BLE001 - one bad pass-1 fact must not sink combine
            diag.dropped_invalid.append(f"pass1 fact {rec.id} itself invalid: {exc}")
            continue

        if mode == "sweep_only":
            # VERIFY is switched off entirely: every pass-1 fact survives
            # exactly as pass 1 produced it, no verdict applied.
            survivors.append(base_fact)
            continue

        verdict = verdict_by_id.get(rec.id)
        fixed_or_kept, reject_reason = _apply_verdict(base_fact, verdict)
        if fixed_or_kept is None:
            diag.rejected.append((rec.id, reject_reason))
            continue
        if verdict is not None and verdict.verdict == "fix":
            try:
                # Re-validate: a fix's fixed_type/fixed_claim/fixed_quote
                # can violate a fact rule the reviewer didn't itself check
                # (a quote over forty words, a quote-required type with no
                # quote at all). Caught here rather than left to explode
                # later in _merge_facts/score_extraction.
                fixed_or_kept = Fact.model_validate(fixed_or_kept.model_dump())
            except Exception as exc:  # noqa: BLE001
                diag.dropped_invalid.append(f"fix on {rec.id} produced an invalid fact: {exc}")
                continue
            diag.fixed.append(rec.id)
        survivors.append(fixed_or_kept)

    if mode == "verify_only":
        merged = _merge_facts(survivors) if survivors else []
        return merged, diag

    missed_facts: list[Fact] = []
    for i, mf in enumerate(review.missed, start=1):
        try:
            missed_facts.append(_missed_to_fact(mf, i))
        except Exception as exc:  # noqa: BLE001 - one bad sweep item must not sink combine
            diag.dropped_invalid.append(f"missed[{i}] ({mf.type}) invalid: {exc}")
            continue
    diag.missed_added = len(missed_facts)

    combined_before_dedup = survivors + missed_facts
    merged = _merge_facts(combined_before_dedup) if combined_before_dedup else []
    return merged, diag


def score_combined(
    truth: Truth,
    combined_by_source: dict[str, list[Fact]],
    source_texts: dict[str, str],
) -> ExtractionScore:
    """Score one mode's combined per-source fact lists with the bench's
    own `score_extraction` -- same strict recall, invented count/rate,
    fact-type accuracy, `evidence_basis` breakdown every pass-1-alone
    report already uses, so the two are directly comparable.

    `source_texts` (the exact prepared text each source was extracted
    from) is required for `_evidence_basis`'s invented/plausible
    classification to mean anything -- the same requirement
    `rescore_unplanted_classifications` documents.
    """
    results: dict[str, ExtractionResult] = {
        source_id: ExtractionResult(
            source_id=source_id,
            summary="",
            facts=facts,
            coverage=Coverage(segments_read=1, segments_total=1),
        )
        for source_id, facts in combined_by_source.items()
    }
    run_result = ExtractionRunResult(
        results,
        wall_time_s={},
        sink=InMemorySink(),
        total_wall_time_s=0.0,
        source_texts=source_texts,
    )
    return score_extraction(truth, run_result)


def sweep_recovered_fact_ids(
    mode_score: ExtractionScore, baseline_score: ExtractionScore
) -> list[str]:
    """Planted `fact_id`s matched in `mode_score` but NOT in
    `baseline_score` -- the sweep's own marginal contribution, isolated
    from VERIFY's (a `fix` can also turn a former near-miss into a real
    match, e.g. correcting a mis-typed `decision` back to `request` --
    `01-spec/extraction-miss-analysis-2026-09-18.md`'s "mis_typed" near-
    miss class -- and that gain must not be counted as the sweep's).

    NOT id-based (`_merge_facts`, reused unmodified from
    `portolan.extract.segment`, unconditionally renumbers every fact
    `f1, f2, ...` after dedup -- see that function's own docstring -- so
    any id a sweep fact was given before merging is not reliably
    recoverable afterward). Comparing MATCHED fact_id sets between two
    scoring runs sidesteps that entirely: it needs no id convention to
    survive dedup, and works for the actual comparisons the report needs:
    `combined` vs. that same run's `verify_only` score (isolates the
    sweep, excluding fix-driven recoveries), and `sweep_only` vs. pass 1's
    OWN frozen score (isolates the sweep with VERIFY switched off
    entirely, so there is no fix contribution to exclude)."""
    mode_ids = {m.fact_id for m in mode_score.matched}
    baseline_ids = {m.fact_id for m in baseline_score.matched}
    return sorted(mode_ids - baseline_ids)


def rejected_fact_classification(
    pass1_score: ExtractionScore, source_id: str, pass1_fact_id: str
) -> str:
    """How pass 1's OWN scoring already classified the produced fact a
    reviewer rejected: `"matched"` (it was a real, credited planted-fact
    match -- rejecting it was a FALSE rejection), `"unplanted_plausible"`
    or `"invented"` (pass 1's own `UnplantedFact.classification`), or
    `"unknown"` if it can't be found (shouldn't happen for a fact_id that
    really came from pass 1's own produced_by_source).

    This reuses pass 1's ALREADY-COMPUTED classification rather than
    re-deriving invented/plausible from scratch -- the DESIGN's "how many
    of those were true inventions per the scorer" is answered by reading
    pass 1's own report, not by re-running `_evidence_basis`.
    """
    for m in pass1_score.matched:
        if m.source_id == source_id and m.produced is not None and m.produced.id == pass1_fact_id:
            return "matched"
    for u in pass1_score.unplanted:
        if u.source_id == source_id and u.produced_id == pass1_fact_id:
            return u.classification
    return "unknown"
