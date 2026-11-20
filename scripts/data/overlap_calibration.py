"""Calibrate the mechanical overlap gate (`portolan.training.overlap_gate`)
before labeling with `--mode single` (coordinator course-correction,
2026-09-21, final design item 2: "Mechanical gate only, calibrated on
gold ... Set the threshold from the calibration set: pick the value that
keeps at least 95 percent of the 20 gold facts and drops the most
corruptions").

Reuses the SAME gold-facts-with-quotes loader and corruption builder as
`scripts/data/judge_calibration.py` (`load_gold_facts_with_quotes`,
`build_calibration_items`) -- the same 20 Halvard gold facts from
`corpus/truth-table.yaml` with an extractable planted quote (never
`corpus/out/` itself), each with the same three corruptions
(wrong_type, quote_swapped, added_detail).

**Same limitation as the judge calibration, worked around differently
here**: without `corpus/out/` access there is no real document text, and
`truth-table.yaml`'s own planted quote fragments (extracted from
`phrasing`, often 2-4 words -- "Larkspur", "4 days late") never contain
the claim's own entity names, so using a bare fragment as `doc_text`
made EVERY gold item fail the "names must all appear" rule (measured:
0% gold-keep-rate at every threshold -- a synthetic-harness artifact,
not a gate defect). `doc_text` here is instead `f"{claim} {quote}"` --
the gold claim and its planted quote fragment concatenated, standing in
for "a real document that states this claim and contains this quote".
This validates the gate's actual discriminating logic (numeral/name
coverage catches `added_detail`; quote-not-in-document catches
`quote_swapped`) faithfully; it cannot validate real-world short-quote-
vs-long-claim coverage the way a real document would, since no real
document is available under the hold-out rule.

`wrong_type` is included as a corruption for consistency with the judge
calibration's protocol, but this gate has NO type check at all (it only
ever looks at claim/quote content) -- expect its drop rate to sit near
0%, and the printed table says so plainly rather than hiding it.

Usage:
    uv run python scripts/data/overlap_calibration.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from portolan.training.overlap_gate import (
    CalibrationCase,
    choose_threshold,
    sweep_thresholds,
)
from scripts.data.judge_calibration import _ADDED_DETAIL_SUFFIX, load_gold_facts_with_quotes

REPO_ROOT = Path(__file__).resolve().parents[2]

THRESHOLD_SWEEP: tuple[float, ...] = tuple(round(0.05 * i, 2) for i in range(4, 20))  # 0.20 .. 0.95
MIN_GOLD_KEEP_RATE = 0.95


def build_overlap_cases(truth_table_path: Path) -> list[CalibrationCase]:
    gold_facts = load_gold_facts_with_quotes(truth_table_path)
    n = len(gold_facts)
    cases: list[CalibrationCase] = []
    for i, fact in enumerate(gold_facts):
        claim = fact["claim"]
        quote = fact["quote"]
        # One sentence, no internal period, so `extract_surrounding_
        # sentence`'s sentence-split never separates the claim's own
        # entity names/numerals from the quote fragment into two
        # different "sentences" (see module docstring).
        doc_text = f"{claim.rstrip('.')} -- {quote}"

        cases.append(
            CalibrationCase(
                label="gold", claim=claim, quote=quote, doc_text=doc_text, expected_keep=True
            )
        )
        # wrong_type never touches claim/quote -- this gate has no type
        # check, so it is structurally identical to "gold" for this
        # gate's own purposes. Kept as its own labeled case (rather than
        # silently dropped) so the printed table shows its ~0% drop rate
        # honestly, per the coordinator's own corruption protocol.
        cases.append(
            CalibrationCase(
                label="wrong_type", claim=claim, quote=quote, doc_text=doc_text, expected_keep=False
            )
        )
        swapped_quote = gold_facts[(i + 1) % n]["quote"]
        cases.append(
            CalibrationCase(
                label="quote_swapped",
                claim=claim,
                quote=swapped_quote,
                doc_text=doc_text,
                expected_keep=False,
            )
        )
        cases.append(
            CalibrationCase(
                label="added_detail",
                claim=claim + _ADDED_DETAIL_SUFFIX,
                quote=quote,
                doc_text=doc_text,
                expected_keep=False,
            )
        )
    return cases


def print_table(rows) -> None:
    print(
        f"{'threshold':>9}  {'gold_keep':>9}  {'wrong_type':>10}  {'quote_swap':>10}  "
        f"{'added_detail':>12}  {'overall_drop':>12}"
    )
    for row in rows:
        wt = row.corruption_drop_rate_by_label.get("wrong_type", 0.0)
        qs = row.corruption_drop_rate_by_label.get("quote_swapped", 0.0)
        ad = row.corruption_drop_rate_by_label.get("added_detail", 0.0)
        print(
            f"{row.threshold:>9.2f}  {row.gold_keep_rate:>9.1%}  {wt:>10.1%}  {qs:>10.1%}  "
            f"{ad:>12.1%}  {row.overall_corruption_drop_rate:>12.1%}"
        )


def main(argv: list[str] | None = None) -> int:
    truth_table_path = REPO_ROOT / "corpus" / "truth-table.yaml"
    cases = build_overlap_cases(truth_table_path)
    gold_n = sum(1 for c in cases if c.expected_keep)
    print(f"calibration cases: {len(cases)} ({gold_n} gold facts x 4 variants)")

    rows = sweep_thresholds(cases, list(THRESHOLD_SWEEP))
    print_table(rows)

    chosen = choose_threshold(rows, min_gold_keep_rate=MIN_GOLD_KEEP_RATE)
    if chosen is None:
        print(
            f"\nno threshold in the sweep keeps >= {MIN_GOLD_KEEP_RATE:.0%} of gold facts -- "
            "widen THRESHOLD_SWEEP or revisit the gate itself",
            file=sys.stderr,
        )
        return 1

    print(
        f"\nchosen threshold: {chosen.threshold:.2f}  "
        f"(gold_keep_rate={chosen.gold_keep_rate:.1%}, "
        f"overall_corruption_drop_rate={chosen.overall_corruption_drop_rate:.1%})"
    )
    print(
        "note: wrong_type's drop rate reflects that this gate has no type check at all -- "
        "it only ever looks at claim/quote content, so a mismatched emitted type is not "
        "something this gate can catch by design."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
