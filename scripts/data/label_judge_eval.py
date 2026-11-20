"""Calibrate the label judge (`portolan.training.label_judge`) against by-eye
verdicts, and compare judge models.

Input: an evaluation directory (default `var/data/label_judge/2026-09-22/`)
holding `eval_verdicts.jsonl` (one silver fact per line: its type, claim,
quote, the reader's verdict `correct` / `wrong_type` / `not_a_fact`, the
types a correct label may carry, and its `split`) and `eval_sources.jsonl`
(each source's text, company-context key, and its facts' eval ids in the order
the pipeline would send them). The verdicts are evaluation data only; they
never become labels.

For each model and the chosen split, one judge call per source (every fact of
the source in one call, as the pipelines send them), then four conditions:
no judge, the v2 decoy rule only, the judge only, both. Per condition:

- precision: kept facts that are real facts carrying an acceptable type,
  over kept facts;
- recall: real facts (`correct` or `wrong_type`) kept, over real facts;
- type accuracy: kept real facts with an acceptable type, over kept real facts.

The decoy rule is computed by the WP5 code itself (`scripted_spans`,
`_touching`) on the v2 documents under `--v2-docs`; it never touches a public
corpus. Paid calls are metered under `label_judge`; `--max-usd` caps this
run's growth of that key, checked before every call.

Usage:
    uv run python scripts/data/label_judge_eval.py --split dev \\
        --models qwen/qwen3-235b-a22b-2507,deepseek/deepseek-v3.2 --max-usd 0.5
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from portolan.training.label_judge import (  # noqa: E402
    LABEL_JUDGE_SPEND_KEY,
    JudgeFact,
    judge_facts,
    openrouter_judge_call,
)
from portolan.training.pipeline import assert_not_holdout  # noqa: E402
from portolan.training.spend import SpendMeter, default_spend_path  # noqa: E402

DEFAULT_EVAL_DIR = REPO_ROOT / "var" / "data" / "label_judge" / "2026-09-22"
REAL = ("correct", "wrong_type")
CONDITIONS = ("no_judge", "decoy_rule", "judge", "both")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def v2_decoy_drops(rows: list[dict[str, Any]], v2_docs: Path) -> set[str]:
    """Eval ids of v2 facts the WP5 decoy rule drops (any overlap with a
    planted decoy span, recovered unclosed-marker decoys included)."""
    from corpus.generator.v2.completeness import (
        TeacherFact,
        _touching,
        load_written_docs,
        scripted_spans,
        source_inputs,
    )
    from corpus.generator.v2.models import Split
    from corpus.generator.v2.packs.schema import load_pack
    from corpus.generator.v2.planner import plan_world
    from corpus.generator.v2.simulator import simulate_world

    pack = load_pack(REPO_ROOT / "corpus" / "generator" / "v2" / "packs" / "medical_devices.yaml")
    world = simulate_world(pack, seed=42, months=8, split=Split("train"))
    plans = plan_world(world, pack)
    decoys = {d.id: d for d in plans.decoys}
    by_plan = plans.doc_by_id()
    docs = load_written_docs(v2_docs)
    out: set[str] = set()
    spans_by_plan: dict[str, Any] = {}
    for r in rows:
        if r["source"] != "synth_v2":
            continue
        plan_id = r["plan_id"]
        if plan_id not in spans_by_plan:
            doc, plan = docs[plan_id], by_plan[plan_id]
            sources = source_inputs(doc, plan)
            spans_by_plan[plan_id] = (
                [s for s in scripted_spans(doc, sources, world, decoys) if s.kind == "decoy"],
                {s.source_id: s.text for s in sources},
            )
        decoy_spans, texts = spans_by_plan[plan_id]
        fact = TeacherFact(
            teacher_model=r["teacher_model"],
            role="completeness",
            doc_id=r["doc_id"],
            envelope=0,
            fact_type=r["fact_type"],
            claim=r["claim"],
            quote=r["quote"],
            char_start=r["char_start"],
            char_end=r["char_end"],
        )
        if _touching(fact, decoy_spans, texts) is not None:
            out.add(r["eval_id"])
    return out


def metrics(rows: list[dict[str, Any]], dropped: set[str], types: dict[str, str]) -> dict[str, Any]:
    kept = [r for r in rows if r["eval_id"] not in dropped]
    real = [r for r in rows if r["verdict"] in REAL]
    kept_real = [r for r in kept if r["verdict"] in REAL]
    good = [r for r in kept_real if types.get(r["eval_id"], r["fact_type"]) in r["ok_types"]]
    return {
        "facts": len(rows),
        "kept": len(kept),
        "precision": round(len(good) / len(kept), 4) if kept else None,
        "recall": round(len(kept_real) / len(real), 4) if real else None,
        "type_accuracy": round(len(good) / len(kept_real), 4) if kept_real else None,
        "good": len(good),
        "real": len(real),
        "kept_real": len(kept_real),
        "not_a_fact_kept": sum(1 for r in kept if r["verdict"] == "not_a_fact"),
        "not_a_fact": sum(1 for r in rows if r["verdict"] == "not_a_fact"),
    }


def judge_sources(
    model: str,
    sources: list[dict[str, Any]],
    *,
    fact_of: dict[str, JudgeFact],
    wanted: set[str],
    call: Any,
    budget_left: Any,
    workers: int,
) -> tuple[dict[str, dict[str, Any]], float, Counter]:
    """One judge call per source; the verdict for each wanted eval id."""
    results: dict[str, dict[str, Any]] = {}
    how_counts: Counter = Counter()
    costs: list[float] = []
    lock = threading.Lock()

    def run(source: dict[str, Any]) -> None:
        ids = source["eval_ids"]
        result, how = judge_facts(
            source["text"],
            [fact_of[i] for i in ids],
            model=model,
            call=call,
            company_context=source["company_context"],
            source_date=source.get("source_date"),
            budget_left=budget_left,
        )
        with lock:
            costs.append(result.cost_usd)
            how_counts[how] += 1
            for eid, j in zip(ids, result.judgements, strict=True):
                if eid in wanted:
                    results[eid] = {
                        "verdict": None if j is None else j.verdict,
                        "fact_type": None if j is None else j.fact_type,
                        "reason": None if j is None else j.reason,
                        "error": result.error,
                    }

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(run, sources))
    return results, sum(costs), how_counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--split", choices=("dev", "test", "all"), required=True)
    parser.add_argument("--models", required=True, help="comma-separated OpenRouter model ids")
    parser.add_argument("--max-usd", type=float, required=True)
    parser.add_argument("--out", type=Path, default=None, help="default: <eval-dir>/runs/<split>")
    parser.add_argument(
        "--v2-docs",
        type=Path,
        default=REPO_ROOT / "var" / "data" / "synth_v2" / "smoke_wp4fix",
        help="the WP4 output the v2 eval facts come from",
    )
    parser.add_argument("--env-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    for path in (args.eval_dir, args.v2_docs):
        assert_not_holdout(path)

    every = _load_jsonl(args.eval_dir / "eval_verdicts.jsonl")
    sources = _load_jsonl(args.eval_dir / "eval_sources.jsonl")
    rows = [r for r in every if args.split in ("all", r["split"])]
    wanted = {r["eval_id"] for r in rows}
    by_id = {r["eval_id"]: r for r in rows}
    fact_of = {r["eval_id"]: JudgeFact(r["fact_type"], r["claim"], r["quote"]) for r in every}
    sources = [s for s in sources if set(s["eval_ids"]) & wanted]
    out_dir = args.out or args.eval_dir / "runs" / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    decoy = v2_decoy_drops(rows, args.v2_docs)

    from corpus.generator.prose import _env_or_dotenv

    api_key = _env_or_dotenv(args.env_root, "OPENROUTER_API_KEY")
    if not api_key:
        print("error: OPENROUTER_API_KEY not found", file=sys.stderr)
        return 2
    spend = SpendMeter(default_spend_path(REPO_ROOT / "var"))
    start = spend.corpus_usd(LABEL_JUDGE_SPEND_KEY)
    lock = threading.Lock()

    def budget_left() -> bool:
        with lock:
            return spend.corpus_usd(LABEL_JUDGE_SPEND_KEY) - start < args.max_usd

    call = openrouter_judge_call(api_key, spend)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    summary: dict[str, Any] = {"split": args.split, "facts": len(rows), "models": {}}
    base = {c: metrics(rows, decoy if c == "decoy_rule" else set(), {}) for c in CONDITIONS[:2]}
    summary["no_judge"] = base["no_judge"]
    summary["decoy_rule"] = base["decoy_rule"]
    summary["decoy_rule_drops"] = sorted(decoy)

    for model in models:
        log_path = out_dir / f"{model.replace('/', '_')}.jsonl"
        results, cost, how_counts = judge_sources(
            model,
            sources,
            fact_of=fact_of,
            wanted=wanted,
            call=call,
            budget_left=budget_left,
            workers=args.workers,
        )
        judge_drop = {e for e, v in results.items() if v["verdict"] == "drop"}
        types = {e: v["fact_type"] for e, v in results.items() if v["verdict"] == "retype"}
        with log_path.open("w", encoding="utf-8") as f:
            for eid in sorted(results):
                r = by_id[eid]
                f.write(
                    json.dumps(
                        {
                            "eval_id": eid,
                            "truth": r["verdict"],
                            "ok_types": r["ok_types"],
                            "fact_type": r["fact_type"],
                            "claim": r["claim"],
                            "decoy_rule_drop": eid in decoy,
                            **{f"judge_{k}": v for k, v in results[eid].items()},
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        summary["models"][model] = {
            "calls": dict(how_counts),
            "cost_usd": round(cost, 6),
            "cost_per_fact_usd": round(cost / len(results), 6) if results else None,
            "no_verdict": sum(1 for v in results.values() if v["verdict"] is None),
            "verdicts": dict(Counter(v["verdict"] for v in results.values())),
            "judge": metrics(rows, judge_drop, types),
            "both": metrics(rows, judge_drop | decoy, types),
        }
        print(json.dumps({model: summary["models"][model]}, indent=1), flush=True)

    summary["spend_usd_this_run"] = round(spend.corpus_usd(LABEL_JUDGE_SPEND_KEY) - start, 6)
    tag = "_".join(m.split("/")[-1] for m in models)[:120]
    (out_dir / f"summary_{tag}.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("no_judge", "decoy_rule", "spend_usd_this_run")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
