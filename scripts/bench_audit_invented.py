#!/usr/bin/env python3
"""Audit every fact a saved bench run currently labels ``invented`` and
show whether the *old* `_plausible_in_source` classifier (exact quote
containment only, `False` the moment a given quote failed it) actually had
grounds to call it that, versus the corrected classifier
(`portolan.bench.extraction._evidence_basis`: exact quote, fuzzy quote, or
salient-fragment fallback).

2026-09-18 finding (orchestrator): the "invented" count in
`01-spec/model-bench-2026-09-18.md` is inflated because a near-verbatim
quote (one changed word, punctuation, a curly apostrophe, an ellipsis, a
dropped filler word) failed the old classifier's exact-substring check and
fell straight to `invented` instead of the fragment fallback a no-quote
fact always got.

Usage::

    uv run python scripts/bench_audit_invented.py \\
        [--root var/bench] [--corpus corpus/out] \\
        [--manifest corpus/out/manifest.json] [--out var/bench/audit-2026-09-18]

Reads every saved run it can find under ``--root`` at the two locations the
2026-09-18 fix instructions named:

- ``<root>/2026-09-18/<model>/*.json`` -- today's three-model bench
  (absent on this machine as of 2026-09-18; reported as "no saved run
  found", not fabricated).
- ``<root>/company-context-2026-09-17/{with,without}/*.json`` -- the two
  Nova Lite company-context runs, present on disk.

For each run found, prints one summary line (invented as labeled under the
old rule, of which quote-present / fuzzy-supported / fragment-supported,
and the corrected invented count under the new rule) and writes every
*disputed* item -- labeled invented by the old rule but textually
supported by the new one -- to ``<out>/<label>.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from portolan.bench.corpus_loader import load_prepared_source  # noqa: E402
from portolan.bench.extraction import (  # noqa: E402
    _evidence_basis,
    _fragments_supported,
    _fuzzy_quote_containment,
    _quote_supported_by_text,
)
from portolan.bench.truth import Manifest  # noqa: E402

# Today's three 2026-09-18 model runs, named the way `var/bench/compare/`'s
# existing convention slugs a model id (`/` -> `_`) -- used only to report
# "no saved run found" by name when the directory is absent, not to
# fabricate a result.
EXPECTED_2026_09_18_MODELS = [
    "qwen_qwen3.8-27b",
    "amazon_nova-lite-v1",
    "moonshotai_kimi-k2",
]


def _find_report_json(run_dir: Path) -> Path | None:
    """The one `*-openrouter.json` (or any `*.json`) bench report file
    directly inside `run_dir`, or `None` if the directory doesn't exist or
    has no JSON report in it."""
    if not run_dir.is_dir():
        return None
    candidates = sorted(run_dir.glob("*.json"))
    return candidates[0] if candidates else None


def _discover_runs(root: Path) -> dict[str, Path]:
    """label -> saved JSON report path, for every run this audit knows how
    to find. Missing directories are silently skipped here -- the caller
    reports "no saved run found" for every entry in
    `EXPECTED_2026_09_18_MODELS` that never got a hit."""
    runs: dict[str, Path] = {}

    today_root = root / "2026-09-18"
    if today_root.is_dir():
        for model_dir in sorted(today_root.iterdir()):
            if not model_dir.is_dir():
                continue
            report = _find_report_json(model_dir)
            if report is not None:
                runs[f"2026-09-18/{model_dir.name}"] = report

    context_root = root / "company-context-2026-09-17"
    if context_root.is_dir():
        for variant in ("without", "with"):
            variant_dir = context_root / variant
            report = _find_report_json(variant_dir)
            if report is not None:
                runs[f"company-context-2026-09-17/{variant} (amazon_nova-lite-v1)"] = report

    return runs


class _SourceTextCache:
    """Lazy, memoized `source_id -> prepared text` lookup. The saved JSON
    report never stored the exact text `run_extraction` fed the backend
    (`ExtractionRunResult.source_texts` is a run-time-only attribute), but
    the corpus on disk is static, so re-deriving it via
    `load_prepared_source` against the manifest reproduces the same
    locator-annotated text the original run scored against."""

    def __init__(self, manifest: Manifest, corpus_root: Path) -> None:
        self._manifest = manifest
        self._corpus_root = corpus_root
        self._cache: dict[str, str] = {}

    def get(self, source_id: str) -> str:
        if source_id not in self._cache:
            try:
                manifest_source = self._manifest.source(source_id)
                prepared = load_prepared_source(manifest_source, corpus_root=self._corpus_root)
                self._cache[source_id] = prepared.text
            except (KeyError, FileNotFoundError, ValueError) as exc:
                print(f"    ! could not load source text for {source_id}: {exc}", file=sys.stderr)
                self._cache[source_id] = ""
        return self._cache[source_id]


def _produced_quote(
    produced_by_source: dict, source_id: str, produced_id: str
) -> tuple[str | None, str]:
    """(quote, claim) for one produced fact, read from the saved report's
    `produced_by_source` (the only place a saved report keeps the quote --
    `UnplantedFact` itself never carried one)."""
    for rec in produced_by_source.get(source_id, []):
        if rec.get("id") == produced_id:
            return rec.get("quote"), rec.get("claim", "")
    return None, ""


def audit_run(label: str, report_path: Path, source_cache: _SourceTextCache, out_dir: Path) -> None:
    data = json.loads(report_path.read_text(encoding="utf-8"))
    extraction = data.get("extraction")
    if extraction is None:
        print(f"{label}: {report_path} has no 'extraction' section, skipping")
        return

    unplanted = extraction.get("unplanted", [])
    produced_by_source = extraction.get("produced_by_source", {})

    # "Invented as labeled" == report.py's own `_invented_count`: a
    # non-leak unplanted fact the OLD classifier (this saved JSON's own
    # `in_source_text`, computed by whatever code wrote the file) called
    # False.
    labeled_invented = [
        u for u in unplanted if not u["is_distractor_leak"] and not u["in_source_text"]
    ]

    quote_present = 0
    fuzzy_supported = 0
    fragment_supported = 0
    corrected_supported = 0
    disputed: list[dict] = []

    for u in labeled_invented:
        source_id = u["source_id"]
        text = source_cache.get(source_id)
        quote, claim = _produced_quote(produced_by_source, source_id, u["produced_id"])
        claim = claim or u["claim"]

        has_quote = bool(quote and quote.strip())
        fuzzy_score = _fuzzy_quote_containment(quote, text) if has_quote and text else 0.0
        exact_ok = bool(has_quote and text and _quote_supported_by_text(quote, text))
        fragments_ok = bool(text) and _fragments_supported(claim, text)
        basis = _evidence_basis(claim, quote, text)
        supported = basis != "none"

        if has_quote:
            quote_present += 1
        if basis == "fuzzy_quote":
            fuzzy_supported += 1
        if fragments_ok:
            fragment_supported += 1
        if supported:
            corrected_supported += 1
            disputed.append(
                {
                    "source_id": source_id,
                    "produced_id": u["produced_id"],
                    "fact_type": u["fact_type"],
                    "claim": claim,
                    "quote": quote,
                    "has_quote": has_quote,
                    "exact_containment": exact_ok,
                    "fuzzy_containment_score": round(fuzzy_score, 3),
                    "fragments_supported": fragments_ok,
                    "evidence_basis": basis,
                    "source_text_excerpt": text[:2000],
                }
            )

    corrected_invented = len(labeled_invented) - corrected_supported

    print(f"\n{label}")
    print(f"  saved report:            {report_path.relative_to(REPO_ROOT)}")
    print(f"  invented as labeled:     {len(labeled_invented)}")
    print(f"    of which quote-present:      {quote_present}")
    print(f"    of which fuzzy-supported:     {fuzzy_supported}")
    print(f"    of which fragment-supported:  {fragment_supported}")
    print(f"  corrected invented count: {corrected_invented}")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = label.split(" ")[0].replace("/", "_") + ".json"
    out_path = out_dir / out_name
    out_path.write_text(
        json.dumps(
            {
                "label": label,
                "report_path": str(report_path.relative_to(REPO_ROOT)),
                "invented_as_labeled": len(labeled_invented),
                "quote_present": quote_present,
                "fuzzy_supported": fuzzy_supported,
                "fragment_supported": fragment_supported,
                "corrected_invented": corrected_invented,
                "disputed_items": disputed,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"  disputed items saved:    {out_path.relative_to(REPO_ROOT)} ({len(disputed)} items)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(REPO_ROOT / "var" / "bench"))
    parser.add_argument("--corpus", default=str(REPO_ROOT / "corpus" / "out"))
    parser.add_argument("--manifest", default=str(REPO_ROOT / "corpus" / "out" / "manifest.json"))
    parser.add_argument("--out", default=str(REPO_ROOT / "var" / "bench" / "audit-2026-09-18"))
    args = parser.parse_args(argv)

    root = Path(args.root)
    corpus_root = Path(args.corpus)
    manifest = Manifest.load(Path(args.manifest))
    source_cache = _SourceTextCache(manifest, corpus_root)
    out_dir = Path(args.out)

    runs = _discover_runs(root)

    found_today = {label.split("/", 1)[1] for label in runs if label.startswith("2026-09-18/")}
    for model in EXPECTED_2026_09_18_MODELS:
        if model not in found_today:
            print(f"\n2026-09-18/{model}: NO SAVED RUN FOUND under {root / '2026-09-18' / model}")

    if not runs:
        print(f"\nNo saved bench runs found at all under {root}.")
        return 1

    for label, report_path in sorted(runs.items()):
        audit_run(label, report_path, source_cache, out_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
