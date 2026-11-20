#!/usr/bin/env python3
"""Scan spec docs for "NN.N% (a / b)"-style parentheticals and flag any
where a/b doesn't equal the printed percentage within rounding.

2026-09-18 bench-provenance wave, DELIVERABLE 3 (01-spec/
bench-provenance-2026-09-18.md): `portolan/bench/report.py`'s own version
of this bug (ISSUE 1 -- a credit-weighted percentage's parenthetical
printing an unrelated instance COUNT instead of the credit sum, or a
denominator with no numerator) is fixed in code and covered by a test.
This script is the read-only survey for every OTHER place in the spec
docs where a printed number and its stated basis might disagree --
whether hand-copied wrong, computed against a since-fixed bug, or simply
a typo. Read-only: it never edits a doc, only prints findings.

    uv run python scripts/bench_report_mismatch_scan.py
    uv run python scripts/bench_report_mismatch_scan.py --glob "01-spec/*.md" --tolerance 0.2
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# "NN.N% (a<words>/ b<words>)" -- covers both the plain "76.7% (57/73)"
# shape and the labelled "14.1% (41 credit / 309 produced)" shape.
# `[^()%/]*` deliberately can't cross another "(", ")", "%" or "/", so it
# never spans past the parenthetical it started in or swallows a second
# percentage/fraction on the same line (e.g. a strict-then-lenient pair).
PCT_FRACTION_RE = re.compile(
    r"(\d+(?:\.\d+)?)%\s*\(([\d,]+(?:\.\d+)?)[^()%/]*/\s*([\d,]+(?:\.\d+)?)"
)


def _num(s: str) -> float:
    return float(s.replace(",", ""))


def scan_file(path: Path, tolerance_pp: float) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    findings: list[dict] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for m in PCT_FRACTION_RE.finditer(line):
            pct_str, a_str, b_str = m.groups()
            pct = _num(pct_str)
            a, b = _num(a_str), _num(b_str)
            if b == 0:
                findings.append(
                    {
                        "file": str(path.relative_to(REPO_ROOT)),
                        "line": lineno,
                        "text": line.strip(),
                        "printed_pct": pct,
                        "a": a,
                        "b": b,
                        "computed_pct": None,
                        "diff_pp": None,
                        "mismatch": True,
                        "reason": "zero denominator",
                    }
                )
                continue
            computed = 100 * a / b
            diff = abs(computed - pct)
            findings.append(
                {
                    "file": str(path.relative_to(REPO_ROOT)),
                    "line": lineno,
                    "text": line.strip(),
                    "printed_pct": pct,
                    "a": a,
                    "b": b,
                    "computed_pct": round(computed, 4),
                    "diff_pp": round(diff, 4),
                    "mismatch": diff > tolerance_pp,
                    "reason": None,
                }
            )
    return findings


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--glob",
        default="01-spec/*2026-09-18*.md",
        help="glob (relative to the repo root) of files to scan",
    )
    ap.add_argument(
        "--tolerance",
        type=float,
        default=0.1,
        help="allowed drift in percentage points before flagging a mismatch -- a percentage "
        "printed to one decimal has up to ~0.05pp of its own rounding error, so 0.1 leaves a "
        "little margin without hiding a real mismatch",
    )
    args = ap.parse_args()

    paths = sorted(REPO_ROOT.glob(args.glob))
    all_findings: list[dict] = []
    for p in paths:
        all_findings.extend(scan_file(p, args.tolerance))

    mismatches = [f for f in all_findings if f["mismatch"]]
    print(
        f"scanned {len(paths)} file(s) matching {args.glob!r}: "
        f"{len(all_findings)} 'pct (a/b)' pattern(s) found, "
        f"{len(mismatches)} mismatch(es) beyond {args.tolerance}pp"
    )
    for f in mismatches:
        if f["reason"]:
            print(f"MISMATCH {f['file']}:{f['line']}: {f['reason']} ({f['a']:g}/{f['b']:g})")
        else:
            print(
                f"MISMATCH {f['file']}:{f['line']}: printed {f['printed_pct']:g}% but "
                f"({f['a']:g}/{f['b']:g}) = {f['computed_pct']:g}% (diff {f['diff_pp']:g}pp)"
            )
        print(f"    {f['text']}")
    if not mismatches:
        print("no mismatches found")
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
