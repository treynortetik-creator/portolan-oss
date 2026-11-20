"""CLI for WP9: `python -m portolan.bench extract --world <dir>` (a v2 world
bench), `--dispute-judges` on `extract` (three-judge disputes, on a world or
on the Halvard exam, report only), and `python -m portolan.bench disputes
--report <saved bench json>` (the same judges over a saved Halvard run).

Spend: live extraction on a world, and extraction on the Halvard exam when
`--dispute-judges` is given, is recorded under corpus key `bench_extract`;
judges under `bench_judges` (`portolan.training.spend`).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path
from typing import Any

from portolan.bench.disputes import (
    CORPUS_KEY,
    DEFAULT_JUDGES,
    BudgetEstimateExceeded,
    Dispute,
    DisputeRun,
    JudgePanel,
    OpenRouterJudgeBackend,
    fetch_openrouter_prices,
    halvard_disputes,
    resolve_disputes,
    summarize,
    write_dispute_logs,
)
from portolan.bench.world_report import render_disputes_markdown, write_world_report
from portolan.bench.world_score import (
    DISPUTED_CATEGORIES,
    DisputeSummary,
    perfect_world_results,
    run_world_extraction,
    score_world,
)
from portolan.bench.world_v2 import LoadedWorld, load_world
from portolan.llm.base import TEMPERATURE_ENV_VAR
from portolan.training.spend import SpendMeter, default_spend_path

__all__ = [
    "EXTRACT_CORPUS_KEY",
    "EXIT_JUDGES_OVER_BUDGET",
    "add_disputes_subcommand",
    "add_world_args",
    "cmd_extract_world",
    "halvard_dispute_step",
    "make_judge_panel",
    "parse_judges",
]

EXTRACT_CORPUS_KEY = "bench_extract"
_REPO_VAR = Path(__file__).resolve().parents[2] / "var"
DEFAULT_MAX_USD = 0.50
EXIT_JUDGES_OVER_BUDGET = 3


def parse_judges(spec: str) -> tuple[str, ...]:
    if spec.strip() == "default":
        return DEFAULT_JUDGES
    models = tuple(m.strip() for m in spec.split(",") if m.strip())
    if len(models) != 3 or len(set(models)) != 3:
        raise SystemExit(
            f"--dispute-judges needs three different model ids (or 'default'), got {models}"
        )
    return models


def _spend_path(args: argparse.Namespace) -> Path:
    """`--spend-path`, else the machine's one meter (a worktree's var/ maps to the main
    checkout's), never a path relative to wherever the command was run."""
    return Path(args.spend_path) if args.spend_path else default_spend_path(_REPO_VAR)


def make_judge_panel(models: tuple[str, ...]) -> JudgePanel:
    """Live panel: OpenRouter, prices read from its public model list."""
    from portolan.config import get_settings, secret_value

    key = secret_value(get_settings().openrouter_api_key)
    if not key:
        raise SystemExit("--dispute-judges needs OPENROUTER_API_KEY in the environment")
    prices = fetch_openrouter_prices(models)
    return JudgePanel(models=models, prices=prices, backend=OpenRouterJudgeBackend(key))


def add_world_args(extract_p: argparse.ArgumentParser) -> None:
    extract_p.add_argument(
        "--world",
        default=None,
        metavar="DIR",
        help="score a v2 world directory (labels.jsonl plus sources.jsonl or docs/*.json) "
        "instead of the Halvard exam; --corpus/--manifest/--truth-table are ignored",
    )
    extract_p.add_argument(
        "--dispute-judges",
        default=None,
        metavar="M1,M2,M3",
        help="three different model ids (or 'default': "
        f"{','.join(DEFAULT_JUDGES)}) that vote on every produced fact the labels do not "
        "account for; proposals go to <out>/corrections.jsonl and are never applied",
    )
    extract_p.add_argument(
        "--max-usd",
        type=float,
        default=DEFAULT_MAX_USD,
        help="judge spend cap: refuse to start when the expected judge cost is above it, "
        "and never start a call whose ceiling (full max_tokens, client retries) could "
        "cross it",
    )
    extract_p.add_argument(
        "--spend-path",
        default=None,
        help="SpendMeter file (corpus keys bench_extract, bench_judges); default: this "
        "machine's shared meter, the main checkout's var/data/silver/spend.json",
    )


def _meter_extraction(meter: SpendMeter, results: Any) -> float:
    total = 0.0
    for event in getattr(getattr(results, "sink", None), "events", []) or []:
        cost = event.cost_estimate or 0.0
        if cost > 0:
            meter.record(model=event.model_key, corpus=EXTRACT_CORPUS_KEY, amount_usd=cost)
            total += cost
    return total


def _run_judges(
    disputes: list[Dispute],
    texts: dict[str, str],
    args: argparse.Namespace,
    meter: SpendMeter,
) -> DisputeRun | None:
    models = parse_judges(args.dispute_judges)
    if not disputes:
        print("no disputed facts; no judge called", flush=True)
        return DisputeRun(
            outcomes=[], models=models, estimate_usd=0.0, ceiling_usd=0.0, max_usd=args.max_usd
        )
    panel = make_judge_panel(models)
    try:
        return resolve_disputes(
            disputes,
            texts,
            panel,
            max_usd=args.max_usd,
            meter=meter,
            corpus_key=CORPUS_KEY,
            progress=lambda line: print(line, flush=True),
        )
    except BudgetEstimateExceeded as exc:
        print(str(exc), file=sys.stderr)
        return None


def _summary(run: DisputeRun, corrections: Path) -> DisputeSummary:
    return DisputeSummary(
        judge_models=list(run.models),
        estimate_usd=round(run.estimate_usd, 6),
        ceiling_usd=round(run.ceiling_usd, 6),
        spent_usd=round(run.spent_usd, 6),
        max_usd=run.max_usd,
        calls=run.calls,
        stopped=run.stopped,
        by_category=summarize(run),
        corrections_path=str(corrections),
    )


def _world_disputes(world: LoadedWorld, score: Any) -> list[Dispute]:
    out: list[Dispute] = []
    for p in score.produced:
        if p.category not in DISPUTED_CATEGORIES:
            continue
        source = world.sources[p.source_id]
        out.append(
            Dispute(
                source_id=p.source_id,
                produced_id=p.produced_id,
                category=p.category,
                fact_type=p.fact_type,
                claim=p.claim,
                quote=p.quote,
                locator=p.locator,
                entities=p.entities,
                span=p.span,
                decoy_class_id=p.decoy_class_id,
                source_type=source.source_type.value,
                source_date=source.date.isoformat(),
            )
        )
    return out


def cmd_extract_world(args: argparse.Namespace) -> int:
    from portolan.extract.backends import FakeBackend
    from portolan.extract.prompt import build_system_prompt

    world = load_world(args.world)
    print(
        f"world {world.world_id}: {len(world.sources)} sources, {len(world.labels)} labels, "
        f"{len(world.decoys)} decoys (from {world.input_format}); labels excluded "
        f"{world.excluded_labels or 'none'}; sources skipped {len(world.skipped_sources)}",
        flush=True,
    )
    meter = SpendMeter(_spend_path(args))
    backend_factory = None
    if args.backend == "fake":
        backend = FakeBackend(perfect_world_results(world, args.source_ids))
        model = "fake"
    elif args.backend is None:
        raise SystemExit("--backend is required with --world")
    else:
        from portolan.bench.__main__ import _live_backend, _live_backend_factory

        backend = _live_backend(args.backend, args.model)
        if args.concurrency > 1:
            backend_factory = _live_backend_factory(args.backend, args.model)
        model = args.model or "<provider default>"
        print(f"backend resolved: provider={args.backend!r} model={model!r}", file=sys.stderr)
        os.environ[TEMPERATURE_ENV_VAR] = str(args.temperature)

    results = run_world_extraction(
        world,
        backend,
        system_prompt=build_system_prompt(variant=args.variant),
        source_ids=args.source_ids,
        samples=args.samples,
        concurrency=args.concurrency,
        backend_factory=backend_factory,
    )
    if args.backend != "fake":
        spent = _meter_extraction(meter, results)
        print(f"extraction spend ${spent:.4f} recorded under {EXTRACT_CORPUS_KEY}", flush=True)

    common = dict(
        backend=args.backend,
        extractor_model=model,
        prompt_variant=args.variant,
        samples=args.samples,
    )
    score = score_world(world, results, **common)
    exit_code = 0
    if args.dispute_judges:
        disputes = _world_disputes(world, score)
        texts = {sid: s.text for sid, s in world.sources.items()}
        run = _run_judges(disputes, texts, args, meter)
        if run is None:
            exit_code = EXIT_JUDGES_OVER_BUDGET
        else:
            corrections, _ = write_dispute_logs(
                run,
                args.out,
                context={
                    "world_id": world.world_id,
                    "extractor_model": model,
                    "prompt_variant": args.variant,
                },
            )
            score = score_world(
                world,
                results,
                **common,
                judge_decisions=run.decisions(),
                disputes=_summary(run, corrections),
            )
    md, js = write_world_report(score, args.out)
    print(f"wrote {md}")
    print(f"wrote {js}")
    o = score.overall
    print(
        json.dumps(
            {
                "produced": o.produced,
                "labeled": o.labeled,
                "rates": {
                    k: (round(v, 4) if v is not None else None) for k, v in o.rates().items()
                },
            }
        )
    )
    return exit_code


# --------------------------------------------------------------------------
# Halvard (report only)
# --------------------------------------------------------------------------


def _halvard_run(
    score: Any,
    texts: dict[str, str],
    source_meta: dict[str, tuple[str | None, str | None]],
    args: argparse.Namespace,
    meter: SpendMeter,
    context: dict[str, Any],
) -> int:
    disputes = halvard_disputes(score, texts, source_meta=source_meta)
    run = _run_judges(disputes, texts, args, meter)
    if run is None:
        return EXIT_JUDGES_OVER_BUDGET
    out = Path(args.out)
    corrections, disputes_path = write_dispute_logs(run, out, context=context)
    summary = _summary(run, corrections).model_dump()
    title = f"Halvard exam disputes, {dt.date.today().isoformat()} (report only)"
    outcomes = [o.model_dump() for o in run.outcomes]
    md = out / f"disputes-{dt.date.today().isoformat()}.md"
    md.write_text(render_disputes_markdown(title, summary, outcomes), encoding="utf-8")
    (out / f"disputes-{dt.date.today().isoformat()}.json").write_text(
        json.dumps({"summary": summary, "outcomes": outcomes}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    invented = summary["by_category"].get("invented", {})
    print(f"wrote {corrections}")
    print(f"wrote {disputes_path}")
    print(f"wrote {md}")
    print(
        f"judges overturned {invented.get('overturned', 0)} of {invented.get('disputed', 0)} "
        f"'invented' facts; spent ${run.spent_usd:.4f} under {CORPUS_KEY}"
    )
    return 0


def _source_meta(manifest: Any) -> dict[str, tuple[str | None, str | None]]:
    return {s.id: (s.type, s.date.isoformat() if s.date else None) for s in manifest.sources}


def halvard_dispute_step(
    args: argparse.Namespace, score: Any, run_result: Any, manifest: Any, extractor_model: str
) -> int:
    """After `extract` scored the Halvard exam: meter the extraction, then
    run the judges on its unplanted facts. The truth table is not touched."""
    meter = SpendMeter(_spend_path(args))
    if args.backend != "fake":
        spent = _meter_extraction(meter, run_result)
        print(f"extraction spend ${spent:.4f} recorded under {EXTRACT_CORPUS_KEY}", flush=True)
    texts = dict(getattr(run_result, "source_texts", {}) or {})
    return _halvard_run(
        score,
        texts,
        _source_meta(manifest),
        args,
        meter,
        {"exam": "halvard", "extractor_model": extractor_model, "prompt_variant": args.variant},
    )


def _cmd_disputes(args: argparse.Namespace) -> int:
    from portolan.bench.corpus_loader import load_prepared_source
    from portolan.bench.extraction import ExtractionScore
    from portolan.bench.truth import Manifest

    saved = json.loads(Path(args.report).read_text(encoding="utf-8"))
    if "extraction" not in saved:
        raise SystemExit(f"{args.report} has no 'extraction' section")
    score = ExtractionScore.model_validate(saved["extraction"])
    if args.source_ids:
        wanted = set(args.source_ids)
        score.unplanted = [u for u in score.unplanted if u.source_id in wanted]
    manifest = Manifest.load(args.manifest)
    texts: dict[str, str] = {}
    for sid in sorted({u.source_id for u in score.unplanted}):
        try:
            texts[sid] = load_prepared_source(manifest.source(sid), corpus_root=args.corpus).text
        except (KeyError, ValueError, OSError) as exc:
            print(f"skip {sid}: {exc}", file=sys.stderr)
    return _halvard_run(
        score,
        texts,
        _source_meta(manifest),
        args,
        SpendMeter(_spend_path(args)),
        {
            "exam": "halvard",
            "report": str(args.report),
            "extractor_model": saved.get("extractor_model"),
            "prompt_variant": saved.get("prompt_variant"),
        },
    )


def add_disputes_subcommand(sub: argparse._SubParsersAction) -> None:
    from portolan.bench.corpus_loader import DEFAULT_CORPUS_ROOT
    from portolan.bench.truth import DEFAULT_MANIFEST_PATH

    p = sub.add_parser(
        "disputes",
        help="three judges over a saved Halvard bench report's unplanted facts (report only)",
    )
    p.add_argument("--report", required=True, help="a saved bench-*.json from `extract`")
    p.add_argument(
        "--dispute-judges", default="default", metavar="M1,M2,M3", help="three model ids"
    )
    p.add_argument("--max-usd", type=float, default=DEFAULT_MAX_USD)
    p.add_argument("--spend-path", default=None, help="default: the shared meter")
    p.add_argument("--corpus", default=str(DEFAULT_CORPUS_ROOT))
    p.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH))
    p.add_argument("--source-ids", nargs="*", default=None)
    p.add_argument("--out", default="var/bench/disputes")
    p.set_defaults(func=_cmd_disputes)
