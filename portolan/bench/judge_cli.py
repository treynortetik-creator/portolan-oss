"""CLI for the layered evidence/match judge (`portolan.bench.judge`).

Ownership note (wave 3A, `01-spec/extraction-accuracy-campaign-2026-09-18.md`):
`portolan/bench/__main__.py` is another agent's active worktree ("editing
its flag plumbing") -- this wave's build instructions say to put every new
CLI flag in its own module here and add "at most a two-line hook" to
`__main__.py`. `__main__.py` already has a boolean `--judge` flag on its
`extract` subcommand (the pre-existing, unrelated paraphrase-promotion
judge, `extraction._run_judge_and_promote`) -- a three-way `--judge
{off,cached,live}` flag of the SAME name on the SAME subparser would
collide with it (argparse forbids re-adding an option string, and even if
it didn't, the two features answer different questions). Rather than
touch that flag or its wiring (outside this wave's ownership, and actively
being edited elsewhere), this module registers its OWN subcommand --
`python -m portolan.bench judge` -- with its own `--judge`/`--judge-matches`
three-way flags, free of any collision. `add_judge_subcommand` is the
entire two-line hook `__main__.py` needs:

    from portolan.bench.judge_cli import add_judge_subcommand
    add_judge_subcommand(sub)

`sub` is the `argparse._SubParsersAction` `build_parser()` already builds
(`parser.add_subparsers(dest="command", required=True)`) -- adding one
more subcommand to it needs nothing else from that module.

Usage (mirrors `extract --rescore-dir`'s directory-recursive shape, per
the build instructions' own example -- "re-score every saved run with
`--rescore-dir --judge cached`"):

    python -m portolan.bench judge --rescore-dir var/bench/2026-09-18-models \\
        --out var/bench/2026-09-18-judge/models --judge live --judge-matches live

    python -m portolan.bench judge --rescore-dir var/bench/2026-09-18-models \\
        --out var/bench/2026-09-18-judge/models --judge cached
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from portolan.bench.corpus_loader import DEFAULT_CORPUS_ROOT
from portolan.bench.extraction import ExtractionScore, rescore_full
from portolan.bench.judge import (
    DEFAULT_JUDGE_MODEL,
    DEFAULT_VERDICTS_PATH,
    Judge,
    VerdictStore,
    apply_evidence_judge,
    apply_match_judge,
)
from portolan.bench.provenance import provenance_kwargs
from portolan.bench.report import BenchReport, write
from portolan.bench.truth import DEFAULT_MANIFEST_PATH, DEFAULT_TRUTH_TABLE_PATH, Manifest, Truth

_JUDGE_MODE_CHOICES = ["off", "cached", "live"]


def _cmd_judge(args: argparse.Namespace) -> int:
    manifest = Manifest.load(args.manifest)
    truth = Truth.for_manifest(manifest, truth_table_path=args.truth_table)
    store = VerdictStore(args.verdicts_path)
    judge = (
        Judge(model=args.judge_model, log_path=args.judge_log)
        if "live" in (args.judge, args.judge_matches)
        else None
    )

    src_dir = Path(args.rescore_dir)
    out_root = Path(args.out)
    found = sorted(
        p for p in src_dir.rglob("bench-*.json") if not p.stem.endswith(("-rescored", "-judged"))
    )
    if not found:
        raise SystemExit(f"no bench-*.json files found under {src_dir}")

    rows: list[dict[str, object]] = []
    for saved_path in found:
        saved = json.loads(saved_path.read_text(encoding="utf-8"))
        extraction_data = saved.get("extraction")
        if extraction_data is None:
            print(f"skip {saved_path} (no 'extraction' section)")
            continue
        score = ExtractionScore.model_validate(extraction_data)
        # Deterministic full re-score first (current matcher code, exactly
        # what `extract --rescore-dir` does) -- the judge tier below only
        # ever looks at what THAT leaves classified "none"/near-miss.
        rescored = rescore_full(score, manifest, truth, corpus_root=args.corpus)
        rescored = apply_evidence_judge(
            rescored,
            manifest,
            truth,
            mode=args.judge,
            store=store,
            judge=judge,
            corpus_root=args.corpus,
            max_workers=args.judge_workers,
        )
        rescored = apply_match_judge(
            rescored,
            manifest,
            truth,
            mode=args.judge_matches,
            store=store,
            judge=judge,
            corpus_root=args.corpus,
            max_workers=args.judge_workers,
        )

        rel = saved_path.relative_to(src_dir)
        dest_dir = out_root / rel.parent
        backend_label = f"{saved.get('backend', 'unknown')}-judged"
        # Provenance stamp (2026-09-18 bench-provenance follow-up,
        # 01-spec/bench-provenance-2026-09-18.md): metadata only -- git
        # commit and file hashes, never a judge verdict, prompt, or model
        # choice -- so stamping it here doesn't touch this module's own
        # judge logic. `prompt_variant`/`extractor_model` carry forward
        # from the ORIGINAL saved report's own stamp (this command re-
        # scores and judges; it never re-extracts, so the underlying
        # run's model/variant didn't change).
        report = BenchReport(
            backend=backend_label,
            date=dt.date.today(),
            extraction=rescored,
            **provenance_kwargs(
                corpus=args.corpus,
                truth_table=args.truth_table,
                prompt_variant=saved.get("prompt_variant"),
                extractor_model=saved.get("extractor_model"),
            ),
        )
        md_path, json_path = write(report, dest_dir)

        adjudicated_recall = (
            rescored.adjudicated_recall
            if rescored.adjudicated_recall is not None
            else rescored.recall
        )
        row = {
            "source": str(saved_path),
            "judged_md": str(md_path),
            "judged_json": str(json_path),
            "recall_strict": rescored.recall,
            "recall_lenient": rescored.lenient_recall,
            "recall_adjudicated": adjudicated_recall,
            "invented_deterministic": rescored.invented_deterministic,
            "invented_adjudicated": rescored.invented_adjudicated,
            "unverifiable_count": rescored.unverifiable_count,
            "judge_calls": rescored.judge_calls,
            "judge_cache_hits": rescored.judge_cache_hits,
            "judge_matches_calls": rescored.judge_matches_calls,
            "judge_matches_cache_hits": rescored.judge_matches_cache_hits,
        }
        rows.append(row)
        print(
            f"{saved_path}: recall {rescored.recall:.3f} "
            f"(lenient {rescored.lenient_recall or 0.0:.3f}, adjudicated {adjudicated_recall:.3f}) "
            f"invented det={rescored.invented_deterministic} adj={rescored.invented_adjudicated} "
            f"unverifiable={rescored.unverifiable_count} "
            f"judge_calls={rescored.judge_calls}+{rescored.judge_matches_calls} "
            f"cache_hits={rescored.judge_cache_hits}+{rescored.judge_matches_cache_hits}"
        )

    out_root.mkdir(parents=True, exist_ok=True)
    summary_path = out_root / "judge-summary.json"
    summary_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {summary_path}")
    print(f"verdicts cache: {len(store)} entries at {store.path}")
    return 0


def add_judge_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--rescore-dir",
        required=True,
        metavar="DIR",
        help="recursively find every saved bench-*.json report under DIR (same convention as "
        "'extract --rescore-dir') and re-score each with the current matcher, then run the "
        "layered judge tiers on top.",
    )
    parser.add_argument("--out", default="var/bench", help="where judged reports are written")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--truth-table", default=str(DEFAULT_TRUTH_TABLE_PATH))
    parser.add_argument("--corpus", default=str(DEFAULT_CORPUS_ROOT))
    parser.add_argument(
        "--judge",
        choices=_JUDGE_MODE_CHOICES,
        default="cached",
        help="evidence-judge mode for the disputed 'invented' middle: off (deterministic only), "
        "cached (default -- use corpus/verdicts.yaml, never call a model), live (call for cache "
        "misses, then cache the result).",
    )
    parser.add_argument(
        "--judge-matches",
        choices=_JUDGE_MODE_CHOICES,
        default="off",
        help="match-judge mode for lenient near-misses on the recall side: off (default), "
        "cached, live -- same semantics as --judge.",
    )
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument(
        "--judge-workers",
        type=int,
        default=8,
        help="thread pool size for LIVE judge calls (each a real CLI subprocess round trip) -- "
        "cache hits and 'off'/'cached' misses never touch this pool.",
    )
    parser.add_argument("--verdicts-path", default=str(DEFAULT_VERDICTS_PATH))
    parser.add_argument(
        "--judge-log",
        default=None,
        metavar="JSONL_PATH",
        help="append every judge call (source, tokens, model) to this file; off by default",
    )
    parser.set_defaults(func=_cmd_judge)


def add_judge_subcommand(subparsers: argparse._SubParsersAction) -> None:
    """The entire hook `__main__.py` needs -- one new subparser, `judge`,
    registered on the SAME `subparsers` object `build_parser()` already
    has (`sub`), so `python -m portolan.bench judge ...` reaches
    `_cmd_judge` above."""
    judge_p = subparsers.add_parser(
        "judge", help="layered evidence/match judge over a directory of saved bench runs"
    )
    add_judge_args(judge_p)
