"""Side-by-side table for the fine-tune shakedown (2026-09-22) from saved
bench reports: `python -m portolan.bench extract` JSON plus the
`parser-*.json` sidecar it writes next to it.

Everything here is read from the saved reports or computed by code from
them; nothing is estimated. Two numbers the bench itself does not
compute are added, both mechanical:

- **quote verbatim**: of produced facts that carry a quote, how many
  quotes are an exact substring of the text the model was shown (the
  bench's own `corpus_loader` preparation, re-run here), and how many
  match after collapsing whitespace and case.
- **omissions**: planted facts in the slice the strict matcher found no
  produced fact for (`len(unmatched)`), next to the bench's own invented
  count (`portolan.bench.report._invented_count`).

"Claim substantiated by its quote" needs a judge; the bench does not
compute it and neither does this script.

Usage:
    uv run python scripts/train/shakedown_scores.py \\
        --report base=var/bench/.../base --report tuned=var/bench/.../tuned
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from portolan.bench.corpus_loader import DEFAULT_CORPUS_ROOT, load_prepared_source  # noqa: E402
from portolan.bench.extraction import ExtractionScore  # noqa: E402
from portolan.bench.report import _invented_count  # noqa: E402
from portolan.bench.truth import DEFAULT_MANIFEST_PATH, Manifest  # noqa: E402


def _norm(text: str) -> str:
    return " ".join(text.split()).lower()


def _find_report(directory: Path) -> tuple[Path, Path | None]:
    reports = sorted(directory.glob("bench-*.json"))
    if len(reports) != 1:
        raise SystemExit(f"{directory}: expected one bench-*.json, found {len(reports)}")
    parser = directory / f"parser-{reports[0].stem}.json"
    return reports[0], parser if parser.is_file() else None


def score_row(report_dir: Path, texts: dict[str, str]) -> dict[str, Any]:
    report_path, parser_path = _find_report(report_dir)
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    score = ExtractionScore.model_validate(saved["extraction"])
    parser = json.loads(parser_path.read_text(encoding="utf-8")) if parser_path else {}

    quoted = [
        (sid, fact.quote)
        for sid, facts in score.produced_by_source.items()
        for fact in facts
        if fact.quote
    ]
    exact = sum(1 for sid, quote in quoted if quote in texts[sid])
    normalized = sum(1 for sid, quote in quoted if _norm(quote) in _norm(texts[sid]))
    invented = _invented_count(score)
    return {
        "report": str(report_path),
        "extractor_model": saved.get("extractor_model"),
        "sources_scored": score.sources_scored,
        "failed_sources": len(score.failed_sources),
        "planted": score.planted_total,
        "produced": score.produced_total,
        "recall_strict": score.recall,
        "recall_lenient": score.lenient_recall,
        "precision": score.precision,
        "fact_type_accuracy": score.fact_type_accuracy,
        "quote_presence_required_types": score.quote_presence_rate,
        "quotes_produced": len(quoted),
        "quote_verbatim_exact": exact,
        "quote_verbatim_normalized": normalized,
        "locator_presence": score.locator_presence_rate,
        "omissions_strict": len(score.unmatched),
        "invented": invented,
        "unplanted_in_source": sum(
            1 for u in score.unplanted if u.in_source_text and not u.is_distractor_leak
        ),
        "distractor_leakage": score.distractor_leakage,
        "coerced_sentiment": score.coerced_sentiment,
        "parser_rejected_facts": parser.get("rejected_facts"),
        "parser_coercions": parser.get("coercions_total"),
        "parser_coercions_by_field": parser.get("coercions_by_field"),
        "model_calls": parser.get("model_calls"),
        "retry_calls": parser.get("retry_calls"),
        "finish_reasons": parser.get("finish_reasons"),
        "wall_time_s": score.wall_time_s,
        "tokens_input": score.tokens_input,
        "tokens_output": score.tokens_output,
        "tokens_cached": score.tokens_cached,
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def render(rows: dict[str, dict[str, Any]]) -> str:
    names = list(rows)
    keys = [k for k in next(iter(rows.values())) if k not in ("report",)]
    lines = ["| metric | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
    for key in keys:
        cells = []
        for name in names:
            row = rows[name]
            value = row[key]
            if key.startswith("quote_verbatim") and row["quotes_produced"]:
                value = f"{value} / {row['quotes_produced']} ({value / row['quotes_produced']:.1%})"
            cells.append(_fmt(value))
        lines.append(f"| {key} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report", action="append", required=True, help="name=bench output dir")
    p.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_ROOT)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    p.add_argument("--out", type=Path, default=None, help="write the rows as JSON here")
    args = p.parse_args(argv)

    manifest = Manifest.load(args.manifest)
    texts = {s.id: load_prepared_source(s, args.corpus).text for s in manifest.sources}
    rows = {}
    for spec in args.report:
        name, _, directory = spec.partition("=")
        rows[name] = score_row(Path(directory), texts)
    print(render(rows))
    if args.out:
        args.out.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
