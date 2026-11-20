"""CLI for WP10: `python -m portolan.bench questions build|score`.

`build --world DIR` plans a v2 world's search-evaluation questions in code,
has an open-weight model word them, validates them (BM25 top 50, one
open-weight judge) and writes `questions.jsonl`, `dropped.jsonl`,
`intents.jsonl`, `calls.jsonl` and `report.json`. `--plan-only` stops before
any model call and prints the cost estimate; `--dry-run` runs every stage
with fixture models (no network, `DRY_RUN` marker, never real questions).

`score --world DIR --questions FILE` scores a retrieval pass (BM25 over the
world, or `--retrieved FILE` from any system) and, with `--answers`, an
answer pass judged by ERB's prompts.

Spend: every paid call is recorded under `v2_search_eval` on the machine's
one meter; `--max-usd` caps this run's growth of that key.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from portolan.bench.world_questions import (
    DEFAULT_COUNTS,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_WORDING_MODELS,
    KINDS,
    SPEND_KEY,
    BuildConfig,
    CallLog,
    SearchEvalBudget,
    build_intents,
    build_questions,
    dispatch_call,
    estimate_run_cost,
    fixture_judge_call,
    fixture_wording_call,
    load_question_world,
    model_allowed,
    model_price,
    openrouter_model_call,
    world_bm25_index,
    write_build_outputs,
)
from portolan.bench.world_search_score import (
    answer_questions,
    bm25_retrieve,
    load_questions,
    load_retrieved,
    score_answers,
    score_retrieval,
)
from portolan.training.spend import SpendMeter, default_spend_path, shared_var_root

__all__ = ["add_questions_subcommand", "parse_counts"]

_REPO_VAR = Path(__file__).resolve().parents[2] / "var"
DEFAULT_MAX_USD = 1.0
DEFAULT_ANSWER_MODEL = "qwen/qwen3-235b-a22b-2507"
EXIT_OVER_BUDGET = 3


def parse_counts(spec: str | None) -> dict[str, int]:
    """`single_fact=12,recency=8`; kinds left out keep their default."""
    counts = dict(DEFAULT_COUNTS)
    if not spec:
        return counts
    for part in spec.split(","):
        if not part.strip():
            continue
        kind, _, n = part.partition("=")
        kind = kind.strip()
        if kind not in KINDS or not n.strip().isdigit():
            raise SystemExit(f"--counts: bad entry {part!r} (kinds: {', '.join(KINDS)})")
        counts[kind] = int(n)
    return counts


def _default_out(world_id: str, dry_run: bool, dry_run_world: bool) -> Path:
    suffix = ("-dryrun-world" if dry_run_world else "") + ("-dryrun" if dry_run else "")
    return shared_var_root(_REPO_VAR) / "bench" / "world_questions" / f"{world_id}{suffix}"


def _meter(args: argparse.Namespace) -> SpendMeter:
    path = Path(args.spend_path) if args.spend_path else default_spend_path(_REPO_VAR)
    return SpendMeter(path)


def _api_key(args: argparse.Namespace) -> str:
    from portolan.bench.erb_search import read_openrouter_key

    env = Path(args.env_file) if args.env_file else shared_var_root(_REPO_VAR).parent / ".env"
    return read_openrouter_key(env)


def _cmd_build(args: argparse.Namespace) -> int:
    w = load_question_world(args.world)
    cfg = BuildConfig(
        counts=parse_counts(args.counts),
        seed=args.seed,
        wording_models=tuple(m.strip() for m in args.wording_models.split(",") if m.strip()),
        judge_model=args.judge_model,
        concurrency=args.concurrency,
        dry_run=args.dry_run,
    )
    intents, supply = build_intents(w, cfg.counts, seed=cfg.seed)
    estimate = estimate_run_cost(w, intents, cfg)
    out = Path(args.out) if args.out else _default_out(w.world_id, args.dry_run, w.dry_run_world)
    kind = " (dry-run world: fixture text)" if w.dry_run_world else ""
    print(f"world {w.world_id}{kind}: {len(w.sources)} sources, {len(w.fact_docs)} labeled facts")
    print(f"candidates by kind: {supply}")
    planned = {k: sum(1 for i in intents if i.kind == k) for k in KINDS}
    print(f"planned: {planned}")
    print(
        f"estimate: expected ${estimate['expected_usd']:.4f}, ceiling "
        f"${estimate['ceiling_usd']:.4f}, {estimate['calls_expected']} calls "
        f"(cap ${args.max_usd:.2f})"
    )
    if args.plan_only:
        out.mkdir(parents=True, exist_ok=True)
        with (out / "intents.jsonl").open("w", encoding="utf-8") as fh:
            for i in intents:
                fh.write(json.dumps(asdict(i), ensure_ascii=False) + "\n")
        plan = {"world_id": w.world_id, "supply": supply, "estimate": estimate, "plan_only": True}
        (out / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
        print(f"plan only: wrote {out / 'intents.jsonl'} and {out / 'plan.json'}; no model call")
        return 0

    if args.dry_run:
        call = dispatch_call(fixture_wording_call(w), fixture_judge_call(w))
        budget = SearchEvalBudget(args.max_usd)
    elif w.dry_run_world:
        print("refused: a dry-run world holds fixture text; use --plan-only or --dry-run")
        return 2
    else:
        refused = [m for m in (*cfg.wording_models, cfg.judge_model) if not model_allowed(m)]
        if refused:
            print(f"refused: not open-weight or not on the allowlist: {refused}", file=sys.stderr)
            return 2
        if estimate["expected_usd"] > args.max_usd:
            print("refused: the expected cost is above --max-usd", file=sys.stderr)
            return EXIT_OVER_BUDGET
        meter = _meter(args)
        call = openrouter_model_call(_api_key(args), meter)
        budget = SearchEvalBudget(args.max_usd, meter)
    log = CallLog(out / "calls.jsonl")
    result = build_questions(
        w, cfg, call=call, budget=budget, log=log, intents=intents, supply=supply
    )
    write_build_outputs(out, result, estimate)
    report = result.report()
    print(f"kept {report['kept']} of {report['planned']} (drop rate {report['drop_rate']:.1%})")
    for kind, row in report["by_kind"].items():
        print(f"  {kind}: {row}")
    print(f"drops by reason: {report['drops_by_reason']}")
    print(f"spent ${report['spent_usd']:.4f} under {SPEND_KEY}; wrote {out}")
    return 0 if not result.budget_stopped else EXIT_OVER_BUDGET


class _FixtureChat:
    """`--dry-run` stand-in for `erb_answers.OpenRouterChat`: answers with
    the first retrieved document, and judges a fact valid when its first
    five words appear in the answer."""

    def complete(self, model: str, prompt: str, *, max_tokens: int = 1200) -> str:
        del model, max_tokens
        if "## Gold Answer" in prompt:
            return '{"reason": "fixture", "aligned": "yes"}'
        if "## Statement" in prompt:
            answer = prompt.split("## Answer", 1)[1].split("## Statement", 1)[0].lower()
            statement = prompt.split("## Statement", 1)[1].lower()
            words = [t for t in statement.replace("`", " ").split() if t.isalpha()][:5]
            return "yes" if words and all(t in answer for t in words) else "no"
        context = prompt.split("## Context Documents", 1)[-1]
        return context.strip()[:400]


def _estimate_answers(questions, retrieved, texts, k, answer_model, judge_model) -> float:
    total = 0.0
    for q in questions:
        top = retrieved.get(q["question_id"], [])[:k]
        chars = 1500 + len(q["question"]) + sum(len(texts[d][1]) for d in top if d in texts)
        pin, pout = model_price(answer_model)
        total += (chars / 3 * pin + 1500 * pout) / 1e6
        jin, jout = model_price(judge_model)
        facts = len(q.get("answer_facts") or [])
        total += (1 + facts) * (3000 / 3 * jin + 300 * jout) / 1e6
    return 2 * total


def _cmd_score(args: argparse.Namespace) -> int:
    w = load_question_world(args.world)
    questions = load_questions(args.questions)
    if args.limit:
        questions = questions[: args.limit]
    if args.retrieved:
        retrieved = load_retrieved(args.retrieved)
        label = Path(args.retrieved).stem
    else:
        retrieved = bm25_retrieve(world_bm25_index(w), questions, args.k)
        label = "bm25"
    result: dict[str, Any] = {
        "world_id": w.world_id,
        "questions_file": str(args.questions),
        "retrieval": label,
        "search": score_retrieval(questions, retrieved, args.k),
    }
    overall = result["search"]["overall"]
    print(
        f"{label}: recall@{args.k} {overall['average_recall_pct']}, invalid extra "
        f"{overall['average_invalid_extra_docs']}, fact coverage "
        f"{overall['ours_fact_coverage_pct']}, MRR {overall['ours_mrr_at_10']} "
        f"over {overall['count']} questions"
    )
    for kind, row in result["search"]["by_kind"].items():
        print(f"  {kind}: recall {row['average_recall_pct']} (n={row['count']})")
    if args.answers:
        texts = {sid: (f"{s.kind} {s.date}", s.text) for sid, s in w.sources.items()}
        if args.dry_run:
            chat: Any = _FixtureChat()
        else:
            for m in (args.answer_model, args.answer_judge_model):
                if not model_allowed(m):
                    print(f"refused: {m} is not an allowlisted open-weight model", file=sys.stderr)
                    return 2
            estimate = _estimate_answers(
                questions, retrieved, texts, args.k, args.answer_model, args.answer_judge_model
            )
            print(f"answer pass estimate (ceiling): ${estimate:.4f} (cap ${args.max_usd:.2f})")
            if estimate > args.max_usd:
                print("refused: the estimate is above --max-usd", file=sys.stderr)
                return EXIT_OVER_BUDGET
            from portolan.bench.erb_answers import OpenRouterChat

            meter = _meter(args)
            chat = OpenRouterChat(
                _api_key(args),
                meter=meter,
                corpus=SPEND_KEY,
                budget_usd=meter.corpus_usd(SPEND_KEY) + args.max_usd,
            )
        answers = answer_questions(chat, args.answer_model, questions, retrieved, texts, args.k)
        result["answers"] = score_answers(chat, args.answer_judge_model, questions, answers)
        result["answer_model"] = args.answer_model
        result["answer_judge_model"] = args.answer_judge_model
        a = result["answers"]["overall"]
        print(
            f"answers: correctness {a.get('average_correctness_pct')}, completeness "
            f"{a.get('average_completeness_pct')}, combined "
            f"{a.get('combined_correctness_completeness_score')}"
        )
    out = Path(args.out) if args.out else Path(args.questions).parent / f"score-{label}.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


def add_questions_subcommand(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("questions", help="WP10: search-eval questions from a v2 world")
    qsub = p.add_subparsers(dest="questions_command", required=True)

    b = qsub.add_parser("build", help="plan, word and validate questions for one world")
    b.add_argument("--world", required=True, metavar="DIR", help="an exported v2 world")
    b.add_argument("--out", default=None, help="default: var/bench/world_questions/<world>")
    b.add_argument(
        "--counts",
        default=None,
        help=f"per kind, e.g. single_fact=12,recency=8 (default {DEFAULT_COUNTS})",
    )
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--plan-only", action="store_true", help="intents and estimate, no call")
    b.add_argument("--dry-run", action="store_true", help="fixture models, no network")
    b.add_argument("--max-usd", type=float, default=DEFAULT_MAX_USD)
    b.add_argument("--wording-models", default=",".join(DEFAULT_WORDING_MODELS))
    b.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    b.add_argument("--concurrency", type=int, default=4)
    b.add_argument("--env-file", default=None, help="default: the main checkout's .env")
    b.add_argument("--spend-path", default=None, help="default: the machine's shared meter")
    b.set_defaults(func=_cmd_build)

    s = qsub.add_parser("score", help="score retrieval (and answers) on built questions")
    s.add_argument("--world", required=True, metavar="DIR")
    s.add_argument("--questions", required=True, help="a questions.jsonl from `build`")
    s.add_argument("--retrieved", default=None, help="jsonl of question_id, retrieved[]")
    s.add_argument("--k", type=int, default=10)
    s.add_argument("--limit", type=int, default=0)
    s.add_argument("--answers", action="store_true", help="also run the answer pass")
    s.add_argument("--answer-model", default=DEFAULT_ANSWER_MODEL)
    s.add_argument("--answer-judge-model", default=DEFAULT_JUDGE_MODEL)
    s.add_argument("--dry-run", action="store_true", help="fixture answer model and judges")
    s.add_argument("--max-usd", type=float, default=DEFAULT_MAX_USD)
    s.add_argument("--env-file", default=None)
    s.add_argument("--spend-path", default=None)
    s.add_argument("--out", default=None, help="default: score-<retrieval>.json beside it")
    s.set_defaults(func=_cmd_score)
