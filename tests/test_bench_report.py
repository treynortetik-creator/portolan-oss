"""Tests for `portolan.bench.report`: the report renders and writes both
a markdown file and a JSON twin."""

from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path

from portolan.bench.extraction import (
    ExtractionScore,
    MatchedFact,
    TypeBreakdown,
    UnmatchedFact,
    UnplantedFact,
    score_extraction,
)
from portolan.bench.fixtures import lossy_results
from portolan.bench.graph import ExpectationResult, GraphScore
from portolan.bench.report import BenchReport, render_markdown, write
from portolan.bench.truth import Truth

REPO_ROOT = Path(__file__).resolve().parents[1]
TRUTH_TABLE_PATH = REPO_ROOT / "corpus" / "truth-table.yaml"
MANIFEST_PATH = REPO_ROOT / "corpus" / "out" / "manifest.json"


def _truth() -> Truth:
    return Truth.load(TRUTH_TABLE_PATH, MANIFEST_PATH)


def _sample_graph_score() -> GraphScore:
    return GraphScore(
        results=[
            ExpectationResult(
                category="merge", name="cross_source_single_claim:F02", passed=True, detail="ok"
            ),
            ExpectationResult(
                category="filter",
                name="junk_filtered:S-email-040",
                passed=False,
                detail="no filter_reason recorded",
                evidence={"row": None},
            ),
        ]
    )


def test_render_markdown_includes_headline_and_worst_misses() -> None:
    truth = _truth()
    results = lossy_results(truth, drop=3, wrong_locator=2, invent=2, seed=3)
    extraction_score = score_extraction(truth, results)
    report = BenchReport(
        backend="fake",
        date=dt.date(2026, 9, 12),
        extraction=extraction_score,
        graph=_sample_graph_score(),
    )

    markdown = render_markdown(report)

    assert "Bench report — fake — 2026-09-12" in markdown
    assert "## Headline" in markdown
    assert "## Extraction" in markdown
    assert "Ten worst misses" in markdown
    assert "## Graph" in markdown
    assert "cross_source_single_claim:F02" in markdown
    assert "**FAIL**" in markdown  # the one failing expectation is visibly marked
    # Fix wave 2026-09-22 (`01-spec/bench-deepseek-v41-flash-2026-09-22.md`):
    # coerced-sentiment count is always in the Extraction table, even when
    # zero, so a reader never has to guess whether it was tracked.
    assert "coerced sentiment" in markdown


def test_write_produces_markdown_and_json_twin(tmp_path: Path) -> None:
    truth = _truth()
    extraction_score = score_extraction(truth, lossy_results(truth, drop=1, seed=5))
    report = BenchReport(backend="fake", date=dt.date(2026, 9, 12), extraction=extraction_score)

    md_path, json_path = write(report, tmp_path)

    assert md_path == tmp_path / "bench-2026-09-12-fake.md"
    assert json_path == tmp_path / "bench-2026-09-12-fake.json"
    assert md_path.exists()
    assert json_path.exists()

    md_text = md_path.read_text(encoding="utf-8")
    assert "## Extraction" in md_text

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    planted_total = len(truth.planted_fact_ids())  # 73 as of truth-table v1.1 (2026-09-13)
    assert payload["backend"] == "fake"
    assert payload["extraction"]["planted_total"] == planted_total
    assert payload["extraction"]["matched_count"] == planted_total - 1


def test_render_markdown_handles_graph_only_report() -> None:
    report = BenchReport(backend="graph", date=dt.date(2026, 9, 12), graph=_sample_graph_score())
    markdown = render_markdown(report)
    assert "## Graph" in markdown
    assert "## Extraction" not in markdown
    assert "1/2 passed" in markdown


def test_render_markdown_lists_failed_sources() -> None:
    # Group 3 of the review fix wave.
    truth = _truth()
    extraction_score = score_extraction(truth, lossy_results(truth, drop=1, seed=5))
    extraction_score = extraction_score.model_copy(
        update={"failed_sources": {"S-call-999": "RuntimeError: boom"}}
    )
    report = BenchReport(backend="fake", date=dt.date(2026, 9, 12), extraction=extraction_score)

    markdown = render_markdown(report)

    assert "Failed sources (1)" in markdown
    assert "S-call-999" in markdown
    assert "boom" in markdown


def test_render_markdown_omits_failed_sources_section_when_empty() -> None:
    truth = _truth()
    extraction_score = score_extraction(truth, lossy_results(truth, drop=1, seed=5))
    report = BenchReport(backend="fake", date=dt.date(2026, 9, 12), extraction=extraction_score)

    markdown = render_markdown(report)

    assert "Failed sources" not in markdown


# --------------------------------------------------------------------
# Fix instructions item 3: invented + invented rate in the headline.
# Precision isn't a meaningful number on this hand-written corpus (most
# "unplanted" facts are real content the corpus contains, not a
# hallucination) -- invented count and rate are.
# --------------------------------------------------------------------


def test_render_markdown_headline_includes_invented_and_invented_rate() -> None:
    truth = _truth()
    results = lossy_results(truth, invent=2, seed=3)
    extraction_score = score_extraction(truth, results)
    report = BenchReport(backend="fake", date=dt.date(2026, 9, 12), extraction=extraction_score)

    markdown = render_markdown(report)
    headline = markdown.split("## Extraction")[0]

    assert "| invented |" in headline
    assert "| invented rate |" in headline
    # The two invented facts `lossy_results(invent=2, ...)` appended have
    # no textual basis anywhere in the corpus and aren't distractor leaks
    # -- both existing columns (recall, precision) must still be present
    # too (fix instructions item 3: "keep the existing columns").
    assert "| invented | 2 |" in headline
    assert "| extraction recall (strict) |" in headline
    assert "| extraction precision |" in headline


def test_render_markdown_invented_rate_is_invented_over_produced() -> None:
    truth = _truth()
    results = lossy_results(truth, invent=2, seed=3)
    extraction_score = score_extraction(truth, results)
    report = BenchReport(backend="fake", date=dt.date(2026, 9, 12), extraction=extraction_score)

    markdown = render_markdown(report)
    expected_rate = 2 / extraction_score.produced_total

    assert f"| invented rate | {expected_rate * 100:.1f}% |" in markdown


def test_render_markdown_invented_rate_is_na_when_nothing_produced() -> None:
    truth = _truth()
    # Every planted fact dropped from every source and nothing invented:
    # produced_total is 0.
    all_ids = {p.fact_id for p in truth.all_planted()}
    results = lossy_results(truth, drop=len(all_ids), seed=1)
    extraction_score = score_extraction(truth, results)
    report = BenchReport(backend="fake", date=dt.date(2026, 9, 12), extraction=extraction_score)

    markdown = render_markdown(report)

    assert extraction_score.produced_total == 0
    assert "| invented | 0 |" in markdown
    assert "| invented rate | n/a |" in markdown


# --------------------------------------------------------------------
# 2026-09-18 bench-provenance wave (01-spec/bench-provenance-2026-09-18.md,
# ISSUE 1 / FIX 1 / TEST 1): recall/precision/fact-type/entity accuracy
# are CREDIT-weighted (a near-miss type match earns 0.5 credit), but
# their "pct (a / b)" parentheticals printed an unrelated COUNT in place
# of the credit sum, or a denominator with no numerator at all -- so a/b
# didn't equal the printed percentage. This score is built BY HAND
# (never through `score_extraction`) specifically so it contains at
# least one half-credit match (`m3` below, credit=0.5) -- exactly the
# case that would have caught the bug and that an all-integer-credit
# score (every real score in this file's other tests, and most real
# runs) cannot.
# --------------------------------------------------------------------


def _synthetic_half_credit_score() -> ExtractionScore:
    matched = [
        MatchedFact(
            fact_id="F1",
            source_id="S-call-001",
            source_type="call",
            fact_type="complaint",
            produced_type="complaint",
            credit=1.0,
            similarity=0.9,
            produced_claim="claim one",
        ),
        MatchedFact(
            fact_id="F2",
            source_id="S-call-001",
            source_type="call",
            fact_type="decision",
            produced_type="decision",
            credit=1.0,
            similarity=0.9,
            produced_claim="claim two",
        ),
        # The half-credit near-miss: a compatible-but-wrong fact type
        # (`_type_credit`'s 0.5 tier) -- `produced_type` deliberately
        # differs from `fact_type`, so this instance also fails
        # `fact_type_accuracy`'s exact-match check (fact_type_correct_
        # count stays 2, not 3, out of matched_instance_count=3).
        MatchedFact(
            fact_id="F3",
            source_id="S-email-021",
            source_type="email",
            fact_type="request",
            produced_type="other",
            credit=0.5,
            similarity=0.5,
            produced_claim="claim three",
        ),
    ]
    unmatched = [
        UnmatchedFact(
            fact_id="F4",
            source_id="S-email-021",
            source_type="email",
            fact_type="decision",
            claim="a missed planted fact",
            locator="L1",
        )
    ]
    unplanted = [
        UnplantedFact(
            source_id="S-call-001",
            fact_type="complaint",
            claim="extra fact one",
            locator="L2",
            produced_id="p1",
        ),
        UnplantedFact(
            source_id="S-email-021",
            fact_type="decision",
            claim="extra fact two",
            locator="L3",
            produced_id="p2",
        ),
    ]
    planted_total = 4  # F1, F2, F3, F4
    produced_total = len(matched) + len(unplanted)  # 5
    recall_credit_sum = 2.5  # 1.0 + 1.0 + 0.5
    lenient_credit_sum = 4.0  # every planted fact credited leniently
    precision_credit_sum = 2.5  # sum(m.credit for m in matched)
    fact_type_correct_count = 2  # F1, F2 exact type match; F3 does not
    matched_instance_count = len(matched)  # 3
    entity_accuracy_correct = 3
    entity_accuracy_checked = 4

    return ExtractionScore(
        sources_scored=2,
        planted_total=planted_total,
        matched_count=3,  # distinct planted fact_ids with credit: F1, F2, F3
        produced_total=produced_total,
        recall=recall_credit_sum / planted_total,
        precision=precision_credit_sum / produced_total,
        recall_credit_sum=recall_credit_sum,
        lenient_credit_sum=lenient_credit_sum,
        precision_credit_sum=precision_credit_sum,
        matched_instance_count=matched_instance_count,
        fact_type_correct_count=fact_type_correct_count,
        entity_accuracy_correct=entity_accuracy_correct,
        distractor_leakage=0,
        quote_presence_rate=1.0,
        locator_presence_rate=1.0,
        entity_accuracy=entity_accuracy_correct / entity_accuracy_checked,
        entity_accuracy_checked=entity_accuracy_checked,
        fact_type_accuracy=fact_type_correct_count / matched_instance_count,
        lenient_recall=lenient_credit_sum / planted_total,
        lenient_matched_count=4,
        near_miss_credited_fact_ids=["F4"],
        near_miss_credited_count=1,
        matched=matched,
        unmatched=unmatched,
        unplanted=unplanted,
        by_fact_type={
            "complaint": TypeBreakdown(planted=1, matched_credit=1.0, produced=2, unplanted=1),
            "decision": TypeBreakdown(planted=2, matched_credit=1.0, produced=2, unplanted=1),
            "request": TypeBreakdown(planted=1, matched_credit=0.5, produced=1, unplanted=0),
        },
        by_source_type={
            "call": TypeBreakdown(planted=2, matched_credit=2.0, produced=2, unplanted=1),
            "email": TypeBreakdown(planted=2, matched_credit=0.5, produced=3, unplanted=1),
        },
        lenient_by_fact_type={
            "complaint": TypeBreakdown(planted=1, matched_credit=1.0),
            "decision": TypeBreakdown(planted=2, matched_credit=2.0),
            "request": TypeBreakdown(planted=1, matched_credit=1.0),
        },
        lenient_by_source_type={
            "call": TypeBreakdown(planted=2, matched_credit=2.0),
            "email": TypeBreakdown(planted=2, matched_credit=2.0),
        },
    )


# Matches every "NN.N% (a<junk>/ b<junk>)" parenthetical this module's
# `_render_extraction_section` prints -- e.g. "62.5% (2.5 credit / 4
# planted)", "75.0% (3/4 checked)", "66.7% (2/3 matched instances)".
# `[^()/]*` deliberately can't cross another "(" or ")" so it never spans
# past the parenthetical it started in.
_PCT_FRACTION_RE = re.compile(r"(\d+\.\d+)%\s*\(([\d.]+)[^()/]*/\s*([\d.]+)")


def test_every_pct_a_of_b_parenthetical_is_internally_consistent() -> None:
    score = _synthetic_half_credit_score()
    report = BenchReport(backend="synthetic", date=dt.date(2026, 9, 18), extraction=score)

    markdown = render_markdown(report)
    matches = _PCT_FRACTION_RE.findall(markdown)

    # Sanity: the regex actually found the lines this test exists to
    # check (recall strict/lenient, precision, entity accuracy,
    # fact-type accuracy) -- five is a floor, not an exact count.
    assert len(matches) >= 5

    for pct_str, a_str, b_str in matches:
        pct = float(pct_str)
        a, b = float(a_str), float(b_str)
        assert b != 0, f"zero denominator in {pct_str}% ({a_str}/{b_str})"
        computed_pct = 100 * a / b
        # Both the credit numerator and the percentage are independently
        # rounded to one decimal for display; 0.1 percentage points
        # covers that compounded rounding with room to spare.
        assert abs(computed_pct - pct) < 0.1, (
            f"printed {pct_str}% but ({a_str} / {b_str}) = {computed_pct:.4f}%"
        )


def _parse_markdown_table(markdown: str, heading: str) -> list[list[str]]:
    """Rows (as lists of stripped cell strings, header/separator rows
    dropped) of the first markdown table under `### {heading}`."""
    section = markdown.split(f"### {heading}")[1]
    body = section.lstrip("\n").split("\n\n")[0]
    rows = [line for line in body.splitlines() if line.strip().startswith("|")]
    data_rows = [r for r in rows if not set(r.replace("|", "").strip()) <= {"-", " "}][1:]
    return [[cell.strip() for cell in r.strip().strip("|").split("|")] for r in data_rows]


def test_breakdown_table_matched_credit_column_matches_recall_percent() -> None:
    # The by-fact-type/by-source-type breakdown tables don't use the
    # "pct (a/b)" single-cell shape (recall% and matched-credit are
    # separate table columns), so the regex test above can't see them --
    # covered separately here. Before this wave's fix, `matched (credit)`
    # didn't exist as a column at all: a reader could see the recall
    # percentage and `planted` but had no way to verify one against the
    # other.
    score = _synthetic_half_credit_score()
    report = BenchReport(backend="synthetic", date=dt.date(2026, 9, 18), extraction=score)
    markdown = render_markdown(report)

    for heading in ("By fact type", "By source type"):
        rows = _parse_markdown_table(markdown, heading)
        assert rows, f"no data rows found under ### {heading}"
        for row in rows:
            _label, recall_str, credit_str, planted_str = row[0], row[1], row[2], row[3]
            planted = float(planted_str)
            credit = float(credit_str)
            expected_pct = 100 * credit / planted if planted else 0.0
            actual_pct = float(recall_str.rstrip("%"))
            assert abs(expected_pct - actual_pct) < 0.1, (
                f"{heading} row {row}: {credit_str}/{planted_str} = "
                f"{expected_pct:.4f}% but recall column printed {recall_str}"
            )
