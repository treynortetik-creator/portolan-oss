"""Calibrate the typed judge (`portolan.training.typed_judge`) before
trusting it to gate silver labels (coordinator course-correction,
2026-09-21, item 3: "Calibrate before labeling").

Takes the Halvard Medical planted gold facts from `corpus/truth-
table.yaml` -- calibration only, never training, and never `corpus/
out/` itself (the hold-out rule: this script reads only the YAML truth
table, the same "never read corpus/out/" boundary
`portolan.training.pipeline.assert_not_holdout` enforces for the real
silver-labeling pipeline). `truth-table.yaml`'s `facts[].phrasing` field
carries the fact's real planted wording as a double-quoted substring for
some (not all) facts -- e.g. `'"two shipments showed up days late"; do
not say the branch name'` -- those double-quoted fragments are the only
verbatim quotes available without opening `corpus/out/`, so calibration
runs only over the facts that have one (20 of 77 as of 2026-09-21; see
this script's own printed count).

For each usable gold fact, builds three corruptions:
  - **wrong_type**: the fact's `emitted_type` (what a downstream
    extractor claims to have found) is set to a different production
    fact type than the true one -- the judge's own type question is
    always asked independent of any emitted type, so this tests whether
    the KEEP GATE correctly rejects a mismatched emitted type once the
    judge (correctly) names the true one.
  - **quote_swapped**: the claim keeps its real quote, then the whole
    thing is proven with a DIFFERENT fact's quote instead -- the judge
    should call this unfaithful.
  - **added_detail**: a fabricated, clearly-invented detail (a dollar
    figure never in any source) is appended to the claim -- the judge
    should flag it as adding something the quote doesn't say.

Prints accuracy per question (faithful / type / added_detail), a
confusion table for the type question, and overall agreement with gold
(does the keep/drop decision match the expected label). Run once against
Jev, again against Kev once `docs/kev-local.md` documents its URL.

Usage:
    set -a; source .env; set +a
    uv run python scripts/data/judge_calibration.py
    uv run python scripts/data/judge_calibration.py --judge kev --base-url http://localhost:8787/v1/systemone
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from corpus.generator.prose import _env_or_dotenv
from portolan.extract.schema import FactType
from portolan.training.pipeline import assert_not_holdout
from portolan.training.typed_judge import (
    JEV_URL_DEFAULT,
    JevClient,
    ask_typed_questions,
    fact_type_criteria,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

_QUOTE_RE = re.compile(r'"([^"]{8,})"')
_ADDED_DETAIL_SUFFIX = " -- an estimated $8,400 in unplanned costs, per the account's own tally."

_FACT_TYPE_ORDER = list(FactType)


@dataclass
class CalibrationItem:
    fact_id: str
    variant: str  # "gold" | "wrong_type" | "quote_swapped" | "added_detail"
    claim: str
    quote: str
    true_type: str
    emitted_type: str
    expected_faithful: bool
    expected_added_detail: bool
    expect_type_check: bool  # whether this variant's type prediction counts toward type accuracy
    expected_keep: bool


def load_gold_facts_with_quotes(truth_table_path: Path) -> list[dict[str, Any]]:
    """Never `corpus/out/` -- only `truth-table.yaml` itself. Reuses the
    same hold-out guard the real silver-labeling pipeline enforces
    (`portolan.training.pipeline.assert_not_holdout`) rather than a
    second, possibly-drifting check."""
    import yaml

    assert_not_holdout(truth_table_path)
    data = yaml.safe_load(truth_table_path.read_text(encoding="utf-8"))
    usable = []
    for fact in data["facts"]:
        phrasing = fact.get("phrasing") or ""
        match = _QUOTE_RE.search(phrasing)
        if not match:
            continue
        usable.append(
            {
                "id": fact["id"],
                "type": fact["type"],
                "claim": fact["claim"],
                "quote": match.group(1),
            }
        )
    return usable


def _wrong_type_for(true_type: str) -> str:
    idx = _FACT_TYPE_ORDER.index(FactType(true_type))
    return _FACT_TYPE_ORDER[(idx + 1) % len(_FACT_TYPE_ORDER)].value


def build_calibration_items(gold_facts: list[dict[str, Any]]) -> list[CalibrationItem]:
    items: list[CalibrationItem] = []
    n = len(gold_facts)
    for i, fact in enumerate(gold_facts):
        true_type = fact["type"]
        claim = fact["claim"]
        quote = fact["quote"]

        items.append(
            CalibrationItem(
                fact_id=fact["id"],
                variant="gold",
                claim=claim,
                quote=quote,
                true_type=true_type,
                emitted_type=true_type,
                expected_faithful=True,
                expected_added_detail=False,
                expect_type_check=True,
                expected_keep=True,
            )
        )

        wrong_type = _wrong_type_for(true_type)
        items.append(
            CalibrationItem(
                fact_id=fact["id"],
                variant="wrong_type",
                claim=claim,
                quote=quote,
                true_type=true_type,
                emitted_type=wrong_type,
                expected_faithful=True,
                expected_added_detail=False,
                expect_type_check=True,
                expected_keep=False,
            )
        )

        swapped_quote = gold_facts[(i + 1) % n]["quote"]
        items.append(
            CalibrationItem(
                fact_id=fact["id"],
                variant="quote_swapped",
                claim=claim,
                quote=swapped_quote,
                true_type=true_type,
                emitted_type=true_type,
                expected_faithful=False,
                expected_added_detail=False,
                expect_type_check=False,  # quote no longer matches the claim's content
                expected_keep=False,
            )
        )

        items.append(
            CalibrationItem(
                fact_id=fact["id"],
                variant="added_detail",
                claim=claim + _ADDED_DETAIL_SUFFIX,
                quote=quote,
                true_type=true_type,
                emitted_type=true_type,
                expected_faithful=True,
                expected_added_detail=True,
                expect_type_check=True,
                expected_keep=False,
            )
        )
    return items


def run_calibration(client: JevClient, items: list[CalibrationItem]) -> list[dict[str, Any]]:
    criteria = fact_type_criteria()
    rows: list[dict[str, Any]] = []
    for i, item in enumerate(items, start=1):
        verdict = ask_typed_questions(client, item.claim, item.quote, criteria=criteria)
        actual_keep = verdict.keep(item.emitted_type)
        rows.append(
            {
                "item": item,
                "verdict": verdict,
                "actual_keep": actual_keep,
                "faithful_correct": verdict.faithful == item.expected_faithful,
                "added_detail_correct": verdict.added_detail == item.expected_added_detail,
                "type_correct": (
                    verdict.predicted_type == item.true_type if item.expect_type_check else None
                ),
                "keep_correct": actual_keep == item.expected_keep,
            }
        )
        if i % 20 == 0:
            print(f"  ... {i}/{len(items)} calibration items judged", file=sys.stderr)
    return rows


def print_report(rows: list[dict[str, Any]], judge_name: str) -> None:
    total = len(rows)
    faithful_acc = sum(r["faithful_correct"] for r in rows) / total
    added_detail_acc = sum(r["added_detail_correct"] for r in rows) / total
    type_rows = [r for r in rows if r["type_correct"] is not None]
    type_acc = sum(r["type_correct"] for r in type_rows) / len(type_rows) if type_rows else 0.0
    keep_acc = sum(r["keep_correct"] for r in rows) / total

    print(f"\n=== Judge calibration report: {judge_name} ===")
    print(f"items: {total} ({total // 4} gold facts x 4 variants)")
    print(f"(a) faithful accuracy:      {faithful_acc:.1%}")
    print(f"(b) fact_type accuracy:     {type_acc:.1%}  (n={len(type_rows)})")
    print(f"(c) added_detail accuracy:  {added_detail_acc:.1%}")
    print(f"overall keep/drop agreement with gold: {keep_acc:.1%}")

    print("\ntype confusion (true -> predicted), gold + wrong_type + added_detail variants:")
    confusion: Counter[tuple[str, str]] = Counter()
    for r in type_rows:
        confusion[(r["item"].true_type, r["verdict"].predicted_type)] += 1
    for (true_t, pred_t), count in sorted(confusion.items()):
        marker = "" if true_t == pred_t else "  <-- MISS"
        print(f"  {true_t:<20} -> {pred_t:<20} {count}{marker}")

    print("\nby variant:")
    for variant in ("gold", "wrong_type", "quote_swapped", "added_detail"):
        variant_rows = [r for r in rows if r["item"].variant == variant]
        if not variant_rows:
            continue
        acc = sum(r["keep_correct"] for r in variant_rows) / len(variant_rows)
        print(f"  {variant:<15} keep/drop correct: {acc:.1%}  (n={len(variant_rows)})")

    lowest = min(faithful_acc, type_acc, added_detail_acc)
    if lowest < 0.85:
        print(
            f"\n*** {judge_name} is below 85% on at least one question ({lowest:.1%}) -- keep "
            "the two-free-teacher agreement mode available behind --judge none. ***"
        )
    else:
        print(f"\n{judge_name} clears the 85% bar on all three questions.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge", choices=["jev", "kev"], default="jev")
    parser.add_argument("--base-url", default=None, help="override the judge endpoint URL")
    parser.add_argument(
        "--truth-table",
        type=Path,
        default=REPO_ROOT / "corpus" / "truth-table.yaml",
        help="path to the hand corpus's own truth table (never corpus/out/ itself)",
    )
    args = parser.parse_args(argv)

    gold_facts = load_gold_facts_with_quotes(args.truth_table)
    print(f"usable gold facts with an extractable planted quote: {len(gold_facts)}")
    if not gold_facts:
        print("error: no gold facts with a quotable phrasing found", file=sys.stderr)
        return 2
    items = build_calibration_items(gold_facts)

    if args.judge == "jev":
        api_key = _env_or_dotenv(REPO_ROOT, "TYPESAFE_API_KEY")
        if not api_key:
            print("error: TYPESAFE_API_KEY not set in the environment or .env", file=sys.stderr)
            return 2
        client = JevClient(api_key=api_key, base_url=args.base_url or JEV_URL_DEFAULT)
    else:
        base_url = args.base_url or _env_or_dotenv(REPO_ROOT, "KEV_BASE_URL")
        if not base_url:
            print(
                "error: --judge kev needs --base-url or KEV_BASE_URL "
                "(docs/kev-local.md, once documented)",
                file=sys.stderr,
            )
            return 2
        api_key = _env_or_dotenv(REPO_ROOT, "KEV_API_KEY")  # may be None for a local Kev
        client = JevClient(api_key=api_key, base_url=base_url)

    print(f"running {len(items)} calibration items through {args.judge}...")
    t0 = time.perf_counter()
    rows = run_calibration(client, items)
    elapsed = time.perf_counter() - t0
    print(f"done in {elapsed:.1f}s ({elapsed / len(items):.2f}s/item)")

    print_report(rows, args.judge)
    return 0


if __name__ == "__main__":
    sys.exit(main())
