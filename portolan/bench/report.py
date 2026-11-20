"""Markdown (+ JSON twin) rendering for a bench run: an `ExtractionScore`,
a `GraphScore`, or both together.
"""

from __future__ import annotations

import datetime as dt
import json
from collections import Counter
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from portolan.bench.extraction import ExtractionScore
from portolan.bench.graph import GraphScore


class BenchReport(BaseModel):
    """Everything one bench run produces, ready to render and write."""

    model_config = ConfigDict(extra="forbid")

    backend: str
    date: dt.date
    extraction: ExtractionScore | None = None
    graph: GraphScore | None = None
    # Determinism recording (2026-09-18 harness review, `01-spec/
    # harness-review-2026-09-18.md`): `--temperature` and `--samples`,
    # recorded so a later wave can see "which runs used what" even though
    # `temperature_requested` currently has NO effect on the model call --
    # see `portolan.bench.__main__._cmd_extract`'s own warning. Optional,
    # defaulting to `None`/1 so an older saved report still validates.
    temperature_requested: float | None = None
    samples: int = 1

    # -- Provenance stamp (2026-09-18, bench-provenance wave, 01-spec/
    # bench-provenance-2026-09-18.md) -----------------------------------
    # A run's own JSON previously carried only `backend, date, extraction,
    # graph, samples, temperature_requested` at the top level -- the
    # extractor MODEL id and the PROMPT VARIANT appeared nowhere except a
    # directory name (which a later wave's re-run/re-score can easily
    # lose track of). These fields make a saved run self-describing: a
    # reader should never have to infer what produced these numbers from
    # a path. All optional/defaulted so an older saved report (from
    # before this field existed) still validates through `--rescore`/
    # `--rescore-dir`. Populated by `portolan.bench.__main__` for a live
    # `extract` run and carried/re-derived by its rescore paths; `None`
    # wherever the caller building a `BenchReport` directly (a test, or a
    # hand-built report) doesn't supply one.
    prompt_variant: str | None = None
    # The extractor model id actually used, e.g. "amazon.nova-lite-v1:0"
    # (`--model`) or the literal string "<provider default>" when no
    # `--model` was given to a live backend, or "fake" for the `fake`
    # backend (no real model at all).
    extractor_model: str | None = None
    # `git rev-parse HEAD` of the checkout that produced this run --
    # "the harness git commit" a reader would otherwise have no way to
    # pin down from the JSON alone.
    harness_commit: str | None = None
    truth_table_path: str | None = None
    truth_table_sha256: str | None = None
    # sha256 of the sorted, newline-joined list of relative file paths
    # under the corpus root actually used (`--corpus`) -- a lightweight
    # stand-in for "which corpus snapshot" that changes if any file is
    # added, removed or renamed, without hashing every file's content.
    corpus_listing_sha256: str | None = None
    # Always "single run" -- one `BenchReport`/one saved JSON is always
    # exactly one execution, never an aggregate. A written-up MEDIAN
    # across several saved reports is a separate artifact (a markdown
    # table built from N of these) and must carry its own "median of N
    # runs" label there, never implied by this field.
    run_basis: str = "single run"


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def wilson_interval(successes: float, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """The Wilson score interval (95% by default) for `successes` of `n`.
    The corpus plants a few dozen facts, so one fact is more than a point of
    recall: the interval says how much of a difference between two runs is
    the runs and how much is the size of the exam. `successes` may be
    fractional (half credit for an adjacent fact type)."""
    if n <= 0:
        return None
    p = min(1.0, max(0.0, successes / n))
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


# Printed under the headline. "invented" is decided by words, not meaning
# (`portolan.bench.extraction._evidence_basis`): a verbatim or near-verbatim
# quote, most of the claim's names and numbers, or most of its content words
# inside three sentences of the source.
INVENTED_FOOTNOTE = (
    "*invented* counts a produced fact that has no footing in its source: no quote found "
    "there, and neither its names and numbers nor its wording close together anywhere in "
    "the text. A claim that misreads words the source does contain (the wrong account for "
    "a real quote, a reversed meaning) is not counted here. Zero invented means nothing "
    "was made up from nothing; it does not mean every claim is right."
)


def _fmt_credit(value: float) -> str:
    """One decimal place, always -- 2026-09-18 bench-provenance wave
    (01-spec/bench-provenance-2026-09-18.md). Every "pct (a / b)"
    parenthetical in this module must print the exact numerator a
    percentage was divided from; several printed an unrelated COUNT
    instead (e.g. distinct matched fact_ids where a CREDIT sum -- which
    can be fractional, a near-miss type match earns 0.5 -- belonged).
    Fixed format regardless of whether THIS run happens to contain any
    half-credit match, so a whole-number credit sum (`56.0`) reads in the
    same shape as a fractional one (`43.5`) instead of the format
    silently changing run to run."""
    return f"{value:.1f}"


def _invented_count(score: ExtractionScore) -> int:
    """Produced facts with no basis in the source text at all, excluding
    distractor leaks (`score.distractor_leakage` already counts those
    separately -- this must not double-count the same fact under both
    numbers). This is the number `01-spec/model-bench-2026-09-12.md`'s own
    reading calls out as the one that matters on a hand-written corpus:
    "precision as printed is not the useful number ... most 'unplanted'
    facts are plausible, not wrong. The column that matters is invented,"
    since a corpus this small and hand-written can't give precision a
    denominator worth trusting -- most of what a strong extractor finds
    beyond the planted set is real content the corpus happens to contain,
    not a hallucination.

    2026-09-18, wave 3A: also excludes `is_unverifiable` -- a fact the
    judge tier moved to its own third bucket (`UnplantedFact.
    classification == "unverifiable"`) is deliberately neither "confirmed
    genuine" nor "confirmed invented"; folding it into this headline
    number would silently re-merge the two buckets `apply_evidence_judge`
    exists to keep apart. `is_unverifiable` defaults `False`, so this is
    the exact same count as before on every score that never went through
    the judge tier.
    """
    return sum(
        1
        for u in score.unplanted
        if not u.is_distractor_leak and not u.in_source_text and not u.is_unverifiable
    )


def _invented_rate(score: ExtractionScore) -> float | None:
    """Invented divided by produced -- `None` (rendered "n/a") when
    nothing was produced, same convention as every other rate in this
    module."""
    if not score.produced_total:
        return None
    return _invented_count(score) / score.produced_total


def _render_extraction_section(score: ExtractionScore) -> list[str]:
    lines = ["## Extraction", ""]
    lines.append("| metric | value |")
    lines.append("|---|---|")
    lines.append(f"| sources scored | {score.sources_scored} |")
    lines.append(
        f"| recall (strict) | {_pct(score.recall)} "
        f"({_fmt_credit(score.recall_credit_sum)} credit / {score.planted_total} planted) |"
    )
    lines.append(
        f"| recall (lenient) | {_pct(score.lenient_recall)} "
        f"({_fmt_credit(score.lenient_credit_sum)} credit / {score.planted_total} planted) |"
    )
    lines.append(
        f"| near-miss credited (lenient minus strict) | {score.near_miss_credited_count} |"
    )
    lines.append(
        f"| precision | {_pct(score.precision)} "
        f"({_fmt_credit(score.precision_credit_sum)} credit / {score.produced_total} produced) |"
    )
    lines.append(f"| distractor leakage | {score.distractor_leakage} |")
    non_leak = [u for u in score.unplanted if not u.is_distractor_leak]
    lines.append(f"| unplanted (non-leak) | {len(non_leak)} |")
    lines.append(
        f"| unplanted — plausible in source | "
        f"{sum(1 for u in score.unplanted if u.in_source_text)} |"
    )
    lines.append(f"| unplanted — invented (no textual basis) | {_invented_count(score)} |")
    # -- Layered judge (2026-09-18, wave 3A, portolan.bench.judge) -------
    # Present only on a score that actually went through apply_evidence_
    # judge/apply_match_judge (`invented_deterministic` is None on every
    # older saved report and every `--judge off` run) -- "both numbers",
    # per the build instructions, never one replacing the other.
    if score.invented_deterministic is not None:
        lines.append(f"| invented (deterministic) | {score.invented_deterministic} |")
        lines.append(f"| invented (adjudicated) | {score.invented_adjudicated} |")
        lines.append(f"| unverifiable (judge, third bucket) | {score.unverifiable_count} |")
        lines.append(
            f"| evidence judge calls / cache hits | "
            f"{score.judge_calls} / {score.judge_cache_hits} |"
        )
    if score.adjudicated_recall is not None:
        lines.append(f"| recall (adjudicated) | {_pct(score.adjudicated_recall)} |")
        lines.append(
            f"| match judge calls / cache hits | "
            f"{score.judge_matches_calls} / {score.judge_matches_cache_hits} |"
        )
    if non_leak:
        basis_counts = Counter(u.evidence_basis for u in non_leak)
        basis_line = ", ".join(
            f"{basis}={basis_counts.get(basis, 0)}"
            for basis in ("exact_quote", "fuzzy_quote", "fragments", "none")
        )
        lines.append(f"| unplanted (non-leak) — evidence basis | {basis_line} |")
    lines.append(f"| quote presence (required types) | {_pct(score.quote_presence_rate)} |")
    lines.append(f"| locator presence | {_pct(score.locator_presence_rate)} |")
    lines.append(f"| coerced sentiment (null/string -> int) | {score.coerced_sentiment} |")
    lines.append(
        f"| entity accuracy | {_pct(score.entity_accuracy)} "
        f"({score.entity_accuracy_correct}/{score.entity_accuracy_checked} checked) |"
    )
    if score.entity_mentions:
        exact = score.entity_accuracy_exact_id_correct
        checked = score.entity_accuracy_checked
        lines.append(
            f"| entity accuracy, exact id string (scorer before 2026-11-20) | "
            f"{_pct(exact / checked if checked else None)} ({exact}/{checked} checked) |"
        )
    lines.append(
        f"| fact-type accuracy | {_pct(score.fact_type_accuracy)} "
        f"({score.fact_type_correct_count}/{score.matched_instance_count} matched instances) |"
    )
    tokens = f"{score.tokens_input}/{score.tokens_output}/{score.tokens_cached}"
    lines.append(f"| tokens (in/out/cached) | {tokens} |")
    lines.append(f"| cost estimate | ${score.cost_estimate:,.4f} |")
    lines.append(f"| wall time | {score.wall_time_s:.1f}s |")
    lines.append("")

    if score.by_fact_type:
        lines.append("### By fact type")
        lines.append("")
        # "matched (credit)" (2026-09-18 bench-provenance wave): without
        # it, `recall`'s numerator was invisible -- a reader could see the
        # percentage and `planted` but not the credit sum they combine to
        # produce it. `breakdown.matched_credit` is exactly that.
        lines.append("| type | recall | matched (credit) | planted | produced | unplanted |")
        lines.append("|---|---|---|---|---|---|")
        for fact_type, breakdown in sorted(score.by_fact_type.items()):
            lines.append(
                f"| {fact_type} | {_pct(breakdown.recall)} | "
                f"{_fmt_credit(breakdown.matched_credit)} | {breakdown.planted} | "
                f"{breakdown.produced} | {breakdown.unplanted} |"
            )
        lines.append("")

    if score.by_source_type:
        lines.append("### By source type")
        lines.append("")
        lines.append("| source type | recall | matched (credit) | planted | produced | unplanted |")
        lines.append("|---|---|---|---|---|---|")
        for source_type, breakdown in sorted(score.by_source_type.items()):
            lines.append(
                f"| {source_type} | {_pct(breakdown.recall)} | "
                f"{_fmt_credit(breakdown.matched_credit)} | {breakdown.planted} | "
                f"{breakdown.produced} | {breakdown.unplanted} |"
            )
        lines.append("")

    if score.lenient_by_fact_type:
        lines.append("### By fact type — lenient recall (strict recall alongside)")
        lines.append("")
        # Both matched-credit numerators shown (2026-09-18 bench-
        # provenance wave) -- same reasoning as the strict-only table
        # above, doubled since this table prints two recall percentages.
        lines.append(
            "| type | strict recall | strict matched (credit) | lenient recall | "
            "lenient matched (credit) | planted |"
        )
        lines.append("|---|---|---|---|---|---|")
        for fact_type, lenient_breakdown in sorted(score.lenient_by_fact_type.items()):
            strict_breakdown = score.by_fact_type.get(fact_type)
            strict_credit = strict_breakdown.matched_credit if strict_breakdown else 0.0
            lines.append(
                f"| {fact_type} | {_pct(strict_breakdown.recall if strict_breakdown else None)} | "
                f"{_fmt_credit(strict_credit)} | {_pct(lenient_breakdown.recall)} | "
                f"{_fmt_credit(lenient_breakdown.matched_credit)} | {lenient_breakdown.planted} |"
            )
        lines.append("")

    if score.lenient_by_source_type:
        lines.append("### By source type — lenient recall (strict recall alongside)")
        lines.append("")
        lines.append(
            "| source type | strict recall | strict matched (credit) | lenient recall | "
            "lenient matched (credit) | planted |"
        )
        lines.append("|---|---|---|---|---|---|")
        for source_type, lenient_breakdown in sorted(score.lenient_by_source_type.items()):
            strict_breakdown = score.by_source_type.get(source_type)
            strict_recall = strict_breakdown.recall if strict_breakdown else None
            strict_credit = strict_breakdown.matched_credit if strict_breakdown else 0.0
            lines.append(
                f"| {source_type} | {_pct(strict_recall)} | {_fmt_credit(strict_credit)} | "
                f"{_pct(lenient_breakdown.recall)} | "
                f"{_fmt_credit(lenient_breakdown.matched_credit)} | {lenient_breakdown.planted} |"
            )
        lines.append("")

    if score.near_miss_credited_fact_ids:
        lines.append(
            f"### Near-miss credited ({score.near_miss_credited_count} planted facts -- "
            "a strict miss the lenient matcher would credit)"
        )
        lines.append("")
        lines.append(", ".join(score.near_miss_credited_fact_ids))
        lines.append("")

    worst = sorted(score.unmatched, key=lambda u: u.best_similarity)[:10]
    if worst:
        lines.append("### Ten worst misses")
        lines.append("")
        lines.append("| fact | source | type | best sim. | failed on | planted claim |")
        lines.append("|---|---|---|---|---|---|")
        for u in worst:
            claim = (u.claim or "").replace("|", "\\|")
            if len(claim) > 90:
                claim = claim[:87] + "..."
            failed = ", ".join(u.failed_criteria)
            lines.append(
                f"| {u.fact_id} | {u.source_id} | {u.fact_type} | {u.best_similarity:.2f} | "
                f"{failed} | {claim} |"
            )
        lines.append("")

    if score.distractor_leakage:
        lines.append("### Distractor leakage (must be zero)")
        lines.append("")
        lines.append(
            "Each leaked fact is labeled with whether the source text really "
            "contains it (`in corpus text` — e.g. S-call-002's genuine "
            "scheduling chatter, S-email-021-thread's quoted-reply content "
            "beyond F21 — real trivia the extractor should still have left "
            "out, not a hallucination) or has no textual basis at all "
            "(`invented`)."
        )
        lines.append("")
        for u in score.unplanted:
            if u.is_distractor_leak:
                claim = u.claim.replace("|", "\\|")
                label = "in corpus text" if u.in_source_text else "invented"
                lines.append(f"- **{u.source_id}** ({u.fact_type}, {label}): {claim}")
        lines.append("")

    if score.failed_sources:
        # Group 3 of the review fix wave: run_extraction now catches a
        # per-source exception and skips that source instead of crashing
        # the whole run; this is the list of what it skipped and why.
        lines.append(f"### Failed sources ({len(score.failed_sources)})")
        lines.append("")
        lines.append("| source | error |")
        lines.append("|---|---|")
        for source_id, error in sorted(score.failed_sources.items()):
            lines.append(f"| {source_id} | {error.replace('|', '\\|')} |")
        lines.append("")

    return lines


def _render_graph_section(score: GraphScore) -> list[str]:
    lines = ["## Graph", ""]
    lines.append(f"{score.passed_count} passed, {score.failed_count} failed.")
    lines.append("")
    lines.append("| category | expectation | result | detail |")
    lines.append("|---|---|---|---|")
    for r in score.results:
        mark = "pass" if r.passed else "**FAIL**"
        detail = r.detail.replace("|", "\\|")
        lines.append(f"| {r.category} | {r.name} | {mark} | {detail} |")
    lines.append("")
    return lines


def render_markdown(report: BenchReport) -> str:
    lines = [f"# Bench report — {report.backend} — {report.date.isoformat()}", ""]
    if report.temperature_requested is not None or report.samples != 1:
        lines.append(
            f"temperature_requested={report.temperature_requested} "
            f"(recorded only -- see determinism note below), samples={report.samples}"
        )
        lines.append("")

    # Provenance stamp (2026-09-18 bench-provenance wave): every field
    # printed here is on `BenchReport` itself -- see that model's own
    # docstring comments for what each one is and why. Omitted line by
    # line wherever the field is `None` (an older saved report, or a
    # `BenchReport` built by hand without it), so this never prints a
    # misleading "unknown" in place of a value nobody supplied; `basis`
    # always prints (`run_basis` always has a value).
    provenance = [f"basis: {report.run_basis}"]
    if report.prompt_variant is not None:
        provenance.append(f"prompt variant: {report.prompt_variant}")
    if report.extractor_model is not None:
        provenance.append(f"model: {report.extractor_model}")
    if report.harness_commit is not None:
        provenance.append(f"harness commit: {report.harness_commit}")
    if report.truth_table_path is not None and report.truth_table_sha256 is not None:
        provenance.append(
            f"truth table: {report.truth_table_path} (sha256 {report.truth_table_sha256})"
        )
    if report.corpus_listing_sha256 is not None:
        provenance.append(f"corpus listing sha256: {report.corpus_listing_sha256}")
    lines.append("Provenance: " + "; ".join(provenance))
    lines.append("")

    if report.extraction is not None:
        lines.append("## Headline")
        lines.append("")
        lines.append("| | |")
        lines.append("|---|---|")
        lines.append(f"| extraction recall (strict) | {_pct(report.extraction.recall)} |")
        interval = wilson_interval(
            report.extraction.recall_credit_sum, report.extraction.planted_total
        )
        if interval is not None:
            lines.append(
                f"| extraction recall (strict), 95% interval | {_pct(interval[0])} to "
                f"{_pct(interval[1])} (n = {report.extraction.planted_total} planted facts) |"
            )
        lines.append(f"| extraction recall (lenient) | {_pct(report.extraction.lenient_recall)} |")
        lines.append(f"| extraction precision | {_pct(report.extraction.precision)} |")
        # Fix instructions item 3: precision isn't meaningful on a
        # hand-written corpus (most "unplanted" facts are real content the
        # corpus happens to contain, not a hallucination -- see
        # `01-spec/model-bench-2026-09-12.md`'s own reading). `invented`
        # (no textual basis at all, distractor leaks excluded -- those are
        # already `distractor leakage` below) and its rate over produced
        # are the numbers that actually distinguish a disciplined
        # extractor from an over-producing one.
        lines.append(f"| invented | {_invented_count(report.extraction)} |")
        lines.append(f"| invented rate | {_pct(_invented_rate(report.extraction))} |")
        lines.append(f"| fact-type accuracy | {_pct(report.extraction.fact_type_accuracy)} |")
        lines.append(f"| distractor leakage | {report.extraction.distractor_leakage} |")
        if report.graph is not None:
            lines.append(
                f"| graph expectations | {report.graph.passed_count}/"
                f"{report.graph.passed_count + report.graph.failed_count} passed |"
            )
        lines.append("")
        lines.append(INVENTED_FOOTNOTE)
        lines.append("")
    elif report.graph is not None:
        lines.append("## Headline")
        lines.append("")
        lines.append(
            f"graph expectations: {report.graph.passed_count}/"
            f"{report.graph.passed_count + report.graph.failed_count} passed"
        )
        lines.append("")

    if report.extraction is not None:
        lines.extend(_render_extraction_section(report.extraction))
    if report.graph is not None:
        lines.extend(_render_graph_section(report.graph))

    return "\n".join(lines).rstrip() + "\n"


def write(report: BenchReport, out_dir: Path | str) -> tuple[Path, Path]:
    """Render `report` and write `bench-YYYY-MM-DD-<backend>.md` plus a JSON
    twin (`report.model_dump_json`) with the same basename, under `out_dir`
    (created if it doesn't exist). Returns `(md_path, json_path)`."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    basename = f"bench-{report.date.isoformat()}-{report.backend}"

    md_path = out / f"{basename}.md"
    md_path.write_text(render_markdown(report), encoding="utf-8")

    json_path = out / f"{basename}.json"
    json_path.write_text(
        json.dumps(json.loads(report.model_dump_json()), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return md_path, json_path
