"""`python -m portolan.bench chat-exam seed|build|run|compare|rejudge|rescore|regrade`
(01-spec/chat-exam-2026-09-29.md).

- `seed [--graph demo|world|both]`: build the local graphs (no model call).
- `build`: write the cases from the seeded graphs: `evals/chat-exam/{train,smoke,split}.json`
  and the locked test file. Prints counts only.
- `run --split train|test|both|smoke --mode agent|single --model ID --max-usd N`: answer,
  check, judge, score; results under `var/evals/chat-exam/<run>/`, test transcripts only
  under `var/evals/locked/`. Prints aggregates, and for test only pass/fail ids.
- `compare A.json B.json`: the acceptance rule; prints KEEP, REVERT or NOISE.
- `regrade --run DIR --judge-model ID`: grade a run's train answers twice with a judge;
  prints its flip rate, its agreement with the run's verdicts, and its cost.
- `rejudge --run DIR --judge-model ID`: grade a whole run again with another judge, answers
  unchanged; writes a new run directory beside it.
- `rescore --run DIR`: score a run against the exam as it stands now with the run's own
  answers and verdicts; no model call.
- `run --reuse DIR`: answers from an earlier run of the same config for the same questions.

Every paid call goes through `OpenRouterClient` under spend key `chat_exam`; no run passes
`HARD_CAP_USD` for that key.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
import sys
import threading
from pathlib import Path
from typing import Any

from portolan.bench import chat_exam as ex
from portolan.bench.chat_exam_cases import (
    CASE_TYPES,
    ExamCase,
    build_all_cases,
    load_split,
    split_cases,
    write_exam,
)
from portolan.bench.chat_exam_graphs import GRAPHS, connect_graph, graph_counts, seed_graph

__all__ = ["add_chat_exam_subcommand"]

EXIT_OVER_BUDGET = 3


def _url(args: argparse.Namespace, graph: str) -> str:
    return getattr(args, f"{graph}_url", None) or GRAPHS[graph].default_url


def _embedder() -> Any:
    from portolan.search.embed import FastEmbedEmbedder

    return FastEmbedEmbedder()


def _cmd_seed(args: argparse.Namespace) -> int:
    names = list(GRAPHS) if args.graph == "both" else [args.graph]
    embedder = _embedder()
    for name in names:
        counts = seed_graph(name, _url(args, name), embedder)
        print(f"seeded {name}: {counts}")
    return 0


def _cmd_build(args: argparse.Namespace) -> int:
    conns = {g: connect_graph(_url(args, g)) for g in GRAPHS}
    try:
        for g, conn in conns.items():
            if graph_counts(conn).get("notes", 0) == 0:
                print(f"graph {g} is empty: run `chat-exam seed` first", file=sys.stderr)
                return 2
        cases = build_all_cases(conns["demo"], conns["world"])
    finally:
        for conn in conns.values():
            conn.close()
    split = split_cases(cases)
    counts = write_exam(cases, split)
    n = {name: sum(counts[name].values()) for name in ("train", "test")}
    dropped = sorted(c.id for c in cases if c.dropped)
    print(f"{n['train'] + n['test']} cases: train {n['train']}, test {n['test']}")
    if dropped:
        print(f"  dropped (gold incomplete): {' '.join(dropped)}")
    for name in ("train", "test"):
        row = ", ".join(f"{t} {counts[name].get(t, 0)}" for t in CASE_TYPES)
        print(f"  {name}: {row}")
    print("wrote evals/chat-exam/{train,smoke,split}.json and the locked test file")
    return 0


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def _cases_for(args: argparse.Namespace) -> list[tuple[str, ExamCase]]:
    splits = ["train", "test"] if args.split == "both" else [args.split]
    out = [(s, c) for s in splits for c in load_split(s)]
    if args.ids:
        wanted = {i.strip() for i in args.ids.split(",") if i.strip()}
        out = [(s, c) for s, c in out if c.id in wanted]
    if args.limit:
        rng = random.Random(args.seed)
        out = rng.sample(out, min(args.limit, len(out)))
        out.sort(key=lambda sc: (sc[0], sc[1].id))
    return out


class _Conns:
    """One connection per graph per worker thread."""

    def __init__(self, urls: dict[str, str]) -> None:
        self.urls = urls
        self._local = threading.local()
        self._all: list[Any] = []
        self._lock = threading.Lock()

    def get(self, graph: str) -> Any:
        cache = getattr(self._local, "conns", None)
        if cache is None:
            cache = self._local.conns = {}
        if graph not in cache:
            conn = connect_graph(self.urls[graph])
            cache[graph] = conn
            with self._lock:
                self._all.append(conn)
        return cache[graph]

    def close(self) -> None:
        for conn in self._all:
            conn.close()


def _client(key: str, meter: Any, model: str) -> tuple[Any, Any]:
    from portolan.llm.openrouter import OpenRouterClient
    from portolan.metering import InMemorySink

    sink = InMemorySink()
    client = OpenRouterClient(
        spend_key=ex.SPEND_KEY, spend_meter=meter, api_key=key, default_model=model, sink=sink,
        tenant="test",
    )  # fmt: skip
    return client, sink


def _sink_cost(sink: Any) -> float:
    return sum(e.cost_estimate or 0.0 for e in sink.events)


def _api_key(args: argparse.Namespace) -> str:
    from portolan.bench.erb_search import read_openrouter_key

    env = Path(args.env_file) if args.env_file else ex.shared_var_root(ex._VAR).parent / ".env"
    return read_openrouter_key(env)


def _agent_limits(args: argparse.Namespace) -> dict[str, Any]:
    from portolan.config import Settings
    from portolan.web.chat_agent import AgentLimits

    limits = dataclasses.asdict(AgentLimits.from_settings(Settings()))
    if args.max_tool_calls is not None:
        limits["max_tool_calls"] = args.max_tool_calls
    return limits


def _judge_all(
    items: list[tuple[str, ExamCase, dict[str, Any]]],
    client: Any,
    judge_model: str,
    budget: ex.Budget,
    workers: int,
) -> dict[str, dict[str, list[bool]] | None]:
    def one(item: tuple[str, ExamCase, dict[str, Any]]) -> tuple[str, Any]:
        _split, case, t = item
        if not budget.acquire(ex.JUDGE_RESERVE_USD):
            return case.id, "budget"
        try:
            return case.id, ex.judge_case(client, judge_model, case, t, t["evidence"])
        except Exception as exc:  # noqa: BLE001 -- a judge error leaves the case unscored
            return case.id, f"error: {type(exc).__name__}: {str(exc)[:200]}"
        finally:
            budget.release(ex.JUDGE_RESERVE_USD)

    return dict(ex.pool_run(items, one, workers))


def _flip_stats(
    pairs: list[tuple[ExamCase, dict[str, Any], dict[str, list[bool]], dict[str, list[bool]]]],
) -> dict[str, Any]:
    verdicts = flips = score_flips = pass_flips = 0
    first, second = [], []
    for case, t, v1, v2 in pairs:
        for key in ("stated", "supported"):
            for a, b in zip(v1[key], v2[key], strict=True):
                verdicts += 1
                flips += a != b
        s1 = ex.score_case(case, t, t["checks"], v1)
        s2 = ex.score_case(case, t, t["checks"], v2)
        first.append(s1["score"])
        second.append(s2["score"])
        score_flips += s1["score"] != s2["score"]
        pass_flips += s1["pass"] != s2["pass"]
    return {
        "regraded": len(pairs),
        "verdicts_compared": verdicts,
        "verdict_flips": flips,
        "verdict_flip_rate": round(flips / verdicts, 4) if verdicts else None,
        "score_flips": score_flips,
        "pass_flips": pass_flips,
        "judge_sd": round(ex.per_case_noise_sd(first, second), 4) if len(pairs) > 1 else None,
    }


def _repeat_stats(
    sample: list[tuple[str, ExamCase]],
    first: dict[str, Any],
    first_verdicts: dict[str, Any],
    second: dict[str, Any],
    second_verdicts: dict[str, Any],
) -> dict[str, Any]:
    """The same cases answered twice by the same config: how far a case's score moves
    between runs (answer and judge noise together)."""
    a, b = [], []
    for _s, case in sample:
        if "not_run" in second.get(case.id, {"not_run": 1}):
            continue
        s1 = ex.score_case(case, first[case.id], first[case.id]["checks"], _v(first_verdicts, case))
        s2 = ex.score_case(
            case, second[case.id], second[case.id]["checks"], _v(second_verdicts, case)
        )
        if s1["score"] is None or s2["score"] is None:
            continue
        a.append(s1["score"])
        b.append(s2["score"])
    return {
        "repeated": len(a),
        "repeat_score_changes": sum(1 for x, y in zip(a, b, strict=True) if x != y),
        "repeat_mean_diff_pct": round(100 * (sum(b) - sum(a)) / len(a), 1) if a else None,
        "answer_sd": round(ex.per_case_noise_sd(a, b), 4) if len(a) > 1 else None,
    }


def _v(verdicts: dict[str, Any], case: ExamCase) -> dict[str, list[bool]] | None:
    v = verdicts.get(case.id)
    return v if isinstance(v, dict) else None


def _grade(
    selected: list[tuple[str, ExamCase]],
    by_id: dict[str, Any],
    repeats: dict[str, Any] | None,
    client: Any,
    judge_model: str,
    budget: ex.Budget,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The judge's verdict (or why there is none) per answer that needs one, and the noise
    block: a regraded sample's flips and, with `repeats`, the answer re-run's spread."""

    def judge(items: list[tuple[str, ExamCase]], ts: dict[str, Any]) -> dict[str, Any]:
        need = [
            (s, c, ts[c.id])
            for s, c in items
            if "not_run" not in ts[c.id] and ex.needs_judge(c, ts[c.id], ts[c.id]["checks"])
        ]
        return _judge_all(need, client, judge_model, budget, args.workers)

    verdicts = judge(selected, by_id)
    noise: dict[str, Any] = {}
    judged = [(s, c, by_id[c.id]) for s, c in selected if isinstance(verdicts.get(c.id), dict)]
    if args.regrade and judged:
        sample = random.Random(args.seed).sample(judged, min(args.regrade, len(judged)))
        again = _judge_all(sample, client, judge_model, budget, args.workers)
        pairs = [
            (c, t, verdicts[c.id], again[c.id])
            for _s, c, t in sample
            if isinstance(again.get(c.id), dict)
        ]
        noise = _flip_stats(pairs)
        noise["judged_share"] = round(len(judged) / len(selected), 4)
    if repeats:
        cases = {c.id: (s, c) for s, c in selected}
        sample2 = [cases[i] for i in repeats["sample"] if i in cases]
        second = repeats["transcripts"]
        noise.update(_repeat_stats(sample2, by_id, verdicts, second, judge(sample2, second)))
    return verdicts, noise


def _answer_all(
    args: argparse.Namespace, selected: list[tuple[str, ExamCase]], cfg: ex.RunConfig,
    budget: ex.Budget, meter: Any, key: str,
) -> tuple[dict[str, Any], dict[str, Any] | None] | None:  # fmt: skip
    """Every selected case through the chat path, then (with `--repeat`) a sample of train
    cases a second time. None when a graph is not seeded."""
    from portolan.bench.chat_exam import _LockedEmbedder

    conns = _Conns({g: _url(args, g) for g in GRAPHS})
    try:
        for g in sorted({c.graph for _, c in selected}):
            if graph_counts(conns.get(g)).get("notes", 0) == 0:
                print(f"graph {g} is empty: run `chat-exam seed` first", file=sys.stderr)
                return None
        embedder = _LockedEmbedder(_embedder())
        reserve = (cfg.limits.get("max_usd", 0.15) + 0.05) if cfg.mode == "agent" else 0.06

        def answer(item: tuple[str, ExamCase]) -> dict[str, Any]:
            _split, case = item
            if not budget.acquire(reserve):
                return {"id": case.id, "not_run": "budget"}
            try:
                client, sink = _client(key, meter, cfg.model)
                conn = conns.get(case.graph)
                t = ex.answer_case(
                    case, conn, embedder, client, cfg, cost_of=lambda: _sink_cost(sink)
                )
                t["checks"] = ex.check_answer(conn, t)
                t["evidence"] = ex.evidence_for(conn, t)
                return t
            finally:
                budget.release(reserve)

        old, old_repeats = _reusable(args.reuse) if args.reuse else ({}, {})
        by_id: dict[str, Any] = {}
        fresh = []
        for split, case in selected:
            prior = old.get((case.graph, case.question))
            if prior is None:
                fresh.append((split, case))
            else:
                by_id[case.id] = {**prior, "id": case.id, "reused_from": args.reuse}
        by_id.update({t["id"]: t for t in ex.pool_run(fresh, answer, args.workers)})
        repeats = None
        if args.repeat:
            pool = [(s, c) for s, c in selected if s == "train" and "not_run" not in by_id[c.id]]
            sample = random.Random(args.seed + 1).sample(pool, min(args.repeat, len(pool)))
            second = {t["id"]: t for t in ex.pool_run(sample, answer, args.workers)}
            repeats = {"sample": [c.id for _, c in sample], "transcripts": second}
        elif old_repeats:
            again = {
                c.id: {**old_repeats[(c.graph, c.question)], "id": c.id}
                for s, c in selected
                if s == "train" and (c.graph, c.question) in old_repeats
            }
            repeats = {"sample": sorted(again), "transcripts": again} if again else None
        return by_id, repeats
    finally:
        conns.close()


def _reusable(run: str) -> tuple[dict[tuple[str, str], Any], dict[tuple[str, str], Any]]:
    """A finished run's answers (and its re-run answers) by graph and question: the same
    config answering the same question on the same graph needs no second paid call. Test
    answers are read from the locked side, in-process."""
    run_dir = Path(run)
    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    graph_of = {r["id"]: r.get("graph") for r in results["rows"]}
    ts = dict(json.loads((run_dir / "train-transcripts.json").read_text(encoding="utf-8")))
    locked = ex.locked_results_root() / run_dir.name / "test-transcripts.json.gz"
    if locked.is_file():
        ts.update(ex.read_locked(locked)["transcripts"])
    graphs = {i: graph_of.get(i) or ("world" if i.startswith("w-") else "demo") for i in ts}
    old = {(graphs[i], t["question"]): t for i, t in ts.items() if "not_run" not in t}
    repeats: dict[tuple[str, str], Any] = {}
    path = run_dir / "repeat-transcripts.json"
    if path.is_file():
        for i, t in json.loads(path.read_text(encoding="utf-8"))["transcripts"].items():
            if "not_run" not in t:
                repeats[(graphs.get(i) or "demo", t["question"])] = t
    return old, repeats


def _refresh_checks(
    args: argparse.Namespace, cases: dict[str, ExamCase], transcripts: list[dict[str, Any]]
) -> None:
    """Each transcript's code checks and judge evidence, from the graph as it stands, so a
    reused or re-graded answer is checked and shown to the judge the current way."""
    conns = _Conns({g: _url(args, g) for g in GRAPHS})
    try:
        for t in transcripts:
            if "not_run" in t or t["id"] not in cases:
                continue
            conn = conns.get(cases[t["id"]].graph)
            t["checks"] = ex.check_answer(conn, t)
            t["evidence"] = ex.evidence_for(conn, t)
    finally:
        conns.close()


def _cmd_run(args: argparse.Namespace) -> int:
    selected = _cases_for(args)
    if not selected:
        print("no cases selected", file=sys.stderr)
        return 2
    cfg = ex.RunConfig(
        mode=args.mode,
        model=args.model,
        judge_model=args.judge_model,
        limits=_agent_limits(args) if args.mode == "agent" else {},
        label=args.label or "",
    )
    meter = ex.default_meter()
    budget = ex.Budget(meter, args.max_usd)
    if budget.cap_usd - budget.start_usd <= 0:
        print(
            f"refused: {ex.SPEND_KEY} has ${budget.start_usd:.4f} of its "
            f"${ex.HARD_CAP_USD:.2f} cap spent",
            file=sys.stderr,
        )
        return EXIT_OVER_BUDGET
    key = _api_key(args)
    print(
        f"{len(selected)} cases ({args.split}), {cfg.mode} {cfg.model}, judge {cfg.judge_model}; "
        f"cap ${budget.cap_usd - budget.start_usd:.2f} this run "
        f"(${budget.start_usd:.4f} of ${ex.HARD_CAP_USD:.2f} spent under {ex.SPEND_KEY})"
    )
    answered = _answer_all(args, selected, cfg, budget, meter, key)
    if answered is None:
        return 2
    by_id, repeats = answered
    cases = {c.id: c for _, c in selected}
    _refresh_checks(
        args, cases, [*by_id.values(), *((repeats or {}).get("transcripts") or {}).values()]
    )
    paths = ex.RunPaths.new(args.label or f"{args.mode}-{args.split}")
    _checkpoint(selected, by_id, repeats, cfg, paths, args.split)
    judge_client, judge_sink = _client(key, meter, cfg.judge_model)
    verdicts, noise = _grade(selected, by_id, repeats, judge_client, cfg.judge_model, budget, args)
    _finish(selected, by_id, repeats, verdicts, noise, cfg, paths, args.split, budget, judge_sink)
    return 0


def _checkpoint(
    selected: list[tuple[str, ExamCase]],
    by_id: dict[str, Any],
    repeats: dict[str, Any] | None,
    cfg: ex.RunConfig,
    paths: ex.RunPaths,
    split_name: str,
) -> None:
    """The paid part's output, saved before grading: if grading fails, `rejudge` picks the
    run up from here without answering again."""
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    rows = [{"id": c.id, "split": s} for s, c in selected]
    stub = {"config": cfg.public(), "split": split_name, "rows": rows, "checkpoint": True}
    (paths.run_dir / "results.json").write_text(json.dumps(stub, indent=1) + "\n")
    train_t = {
        c.id: by_id[c.id] for s, c in selected if s != "test" and "not_run" not in by_id[c.id]
    }
    (paths.run_dir / "train-transcripts.json").write_text(
        json.dumps(train_t, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if repeats:
        (paths.run_dir / "repeat-transcripts.json").write_text(
            json.dumps(repeats, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    test_ids = [c.id for s, c in selected if s == "test" and "not_run" not in by_id[c.id]]
    if test_ids:
        ex.write_locked(
            paths.locked_dir / "test-transcripts.json.gz",
            {"transcripts": {i: by_id[i] for i in test_ids}, "rows": []},
        )


def _finish(
    selected: list[tuple[str, ExamCase]],
    by_id: dict[str, Any],
    repeats: dict[str, Any] | None,
    verdicts: dict[str, Any],
    noise: dict[str, Any],
    cfg: ex.RunConfig,
    paths: ex.RunPaths,
    split_name: str,
    budget: ex.Budget,
    judge_sink: Any,
) -> dict[str, Any]:
    """Score every case, add the noise-based minimum difference, write the run's files
    (test transcripts and verdicts to the locked directory only) and print the summary."""
    rows: list[dict[str, Any]] = []
    for split, case in selected:
        t = by_id[case.id]
        base = {"id": case.id, "split": split, "type": case.type, "graph": case.graph}
        if "not_run" in t:
            rows.append(
                {**base, "score": None, "pass": None, "outcome": "not_run", "infra": ["budget"]}
            )
            continue
        v = verdicts.get(case.id)
        verdict = v if isinstance(v, dict) else None
        infra = []
        if t["error"]:
            infra.append("error")
        if t["truncated"]:
            infra.append("truncated")
        if t["fallback"]:
            infra.append(f"fallback:{str(t['fallback']).split(':')[0]}")
        if isinstance(v, str):
            infra.append(f"judge:{v.split(':')[0]}")
        rows.append(
            {
                **base,
                **ex.score_case(case, t, t["checks"], verdict),
                "cost_usd": t["cost_usd"],
                "wall_s": t["wall_s"],
                "infra": infra,
                "checks": t["checks"],
                "verdict": verdict,
                "gold": [g.to_dict() for g in case.gold],
                "question": case.question,
            }
        )

    n_by_split = {s: sum(1 for r in rows if r["split"] == s and r.get("score") is not None)
                  for s in ("train", "test", "smoke")}  # fmt: skip
    # Judge noise touches only judged answers; the answer re-run carries both kinds. Each
    # estimate's upper bound depends on how many pairs it came from.
    sds, bounds = [], []
    if noise.get("judge_sd") is not None:
        sds.append(noise["judge_sd"] * noise["judged_share"] ** 0.5)
        bounds.append(ex.sd_upper_bound(sds[-1], noise["regraded"]))
    if noise.get("answer_sd") is not None:
        sds.append(noise["answer_sd"])
        bounds.append(ex.sd_upper_bound(sds[-1], noise["repeated"]))
    if sds:
        noise["per_case_sd"] = round(max(sds), 4)
        noise["per_case_sd_upper90"] = round(max(bounds), 4)
        noise["min_diff_pct"] = {
            s: round(100 * ex.min_difference(noise["per_case_sd_upper90"], n), 1)
            for s, n in n_by_split.items()
            if n
        }
    summary = {
        s: ex.summarize([r for r in rows if r["split"] == s])
        if any(r["split"] == s for r in rows)
        else None
        for s in ("train", "test", "smoke")
    }
    infra_counts: dict[str, int] = {}
    for r in rows:
        for flag in r.get("infra") or []:
            infra_counts[flag] = infra_counts.get(flag, 0) + 1
    answers_usd = sum(t.get("cost_usd", 0.0) for t in by_id.values())
    if repeats:
        answers_usd += sum(t.get("cost_usd", 0.0) for t in repeats["transcripts"].values())
    results = {
        "started_at": paths.run_dir.name,
        "git_sha": ex.git_sha(),
        "config": cfg.public(),
        "split": split_name,
        "summary": summary,
        "noise": noise or None,
        "infra": infra_counts,
        "reused_answers": sum(1 for t in by_id.values() if t.get("reused_from")),
        "spend_usd": {
            "answers": round(answers_usd, 4),
            "judge": round(ex._sink_total(judge_sink), 4),
            "this_command_meter": round(budget.spent(), 4),
        },
        "headroom_warning": [
            s
            for s, v in summary.items()
            if v and v["mean_pct"] is not None and v["mean_pct"] >= 100 * ex.HEADROOM_WARN
        ],
        "rows": rows,
    }

    readable = dict(results)
    readable["rows"] = [r if r["split"] != "test" else ex.public_row(r) for r in rows]
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    (paths.run_dir / "results.json").write_text(
        json.dumps(readable, indent=1) + "\n", encoding="utf-8"
    )
    train_t = {
        c.id: by_id[c.id] for s, c in selected if s != "test" and "not_run" not in by_id[c.id]
    }
    (paths.run_dir / "train-transcripts.json").write_text(
        json.dumps(train_t, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if repeats:
        (paths.run_dir / "repeat-transcripts.json").write_text(
            json.dumps(repeats, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    (paths.run_dir / "review.html").write_text(ex.review_html(readable, train_t), encoding="utf-8")
    test_ids = [c.id for s, c in selected if s == "test"]
    if test_ids:
        ex.write_locked(
            paths.locked_dir / "test-transcripts.json.gz",
            {
                "transcripts": {i: by_id[i] for i in test_ids if "not_run" not in by_id[i]},
                "rows": [r for r in rows if r["split"] == "test"],
            },
        )
    _print_summary(results, rows)
    print(f"wrote {paths.run_dir / 'results.json'} and review.html")
    return results


def _cmd_rejudge(args: argparse.Namespace) -> int:
    """A finished run graded again by another judge, answers unchanged: written as a new
    run directory beside it (test transcripts read from, and written to, the locked side)."""
    run_dir = Path(args.run)
    old = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    by_id: dict[str, Any] = json.loads(
        (run_dir / "train-transcripts.json").read_text(encoding="utf-8")
    )
    locked = ex.locked_results_root() / run_dir.name / "test-transcripts.json.gz"
    if locked.is_file():
        by_id.update(ex.read_locked(locked)["transcripts"])
    repeats_path = run_dir / "repeat-transcripts.json"
    repeats = (
        json.loads(repeats_path.read_text(encoding="utf-8")) if repeats_path.is_file() else None
    )
    cases = {c.id: c for name in ("train", "test", "smoke") for c in load_split(name)}
    stale = [i for i, t in by_id.items() if i not in cases or cases[i].question != t["question"]]
    if stale:
        print(
            f"refused: {len(stale)} answers are to questions the exam no longer asks under "
            "those ids (the exam was rebuilt); use `run --reuse` instead",
            file=sys.stderr,
        )
        return 2
    selected = [(r["split"], cases[r["id"]]) for r in old["rows"]]
    for _s, c in selected:
        by_id.setdefault(c.id, {"id": c.id, "not_run": "budget"})
    _refresh_checks(
        args, cases, [*by_id.values(), *((repeats or {}).get("transcripts") or {}).values()]
    )
    cfg = ex.RunConfig(**{**old["config"], "judge_model": args.judge_model})
    meter = ex.default_meter()
    budget = ex.Budget(meter, args.max_usd)
    client, sink = _client(_api_key(args), meter, args.judge_model)
    verdicts, noise = _grade(selected, by_id, repeats, client, args.judge_model, budget, args)
    tag = f"rejudged-{args.judge_model.replace('/', '_')}"
    paths = ex.RunPaths(
        ex.results_root() / f"{run_dir.name}-{tag}",
        ex.locked_results_root() / f"{run_dir.name}-{tag}",
    )
    _finish(selected, by_id, repeats, verdicts, noise, cfg, paths, old["split"], budget, sink)
    return 0


def _cmd_rescore(args: argparse.Namespace) -> int:
    """A finished run scored against the exam as it stands now, with the run's own answers
    and verdicts: no model call. A case the exam dropped, or whose question changed, is left
    out; the halves are the exam's current ones. The run's noise measurements carry over."""
    run_dir = Path(args.run)
    old = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    by_id: dict[str, Any] = json.loads(
        (run_dir / "train-transcripts.json").read_text(encoding="utf-8")
    )
    verdicts = {r["id"]: r.get("verdict") for r in old["rows"] if r.get("verdict")}
    locked = ex.locked_results_root() / run_dir.name / "test-transcripts.json.gz"
    if locked.is_file():
        stored = ex.read_locked(locked)
        by_id.update(stored["transcripts"])
        verdicts.update({r["id"]: r.get("verdict") for r in stored["rows"] if r.get("verdict")})
    selected = [
        (split, c)
        for split in ("train", "test")
        for c in load_split(split)
        if c.id in by_id and by_id[c.id]["question"] == c.question
    ]
    derived = ("per_case_sd", "per_case_sd_upper90", "min_diff_pct")
    noise = {k: v for k, v in (old.get("noise") or {}).items() if k not in derived}
    cfg = ex.RunConfig(**old["config"])
    paths = ex.RunPaths(
        ex.results_root() / f"{run_dir.name}-rescored",
        ex.locked_results_root() / f"{run_dir.name}-rescored",
    )
    budget = ex.Budget(ex.default_meter(), 0.0)
    no_calls = type("NoCalls", (), {"events": []})()
    _finish(selected, by_id, None, verdicts, noise, cfg, paths, "both", budget, no_calls)
    print(
        f"rescored {len(selected)} of {len(old['rows'])} cases; noise carried from {run_dir.name}"
    )
    return 0


def _print_summary(results: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    for split, s in results["summary"].items():
        if s is None:
            continue
        print(
            f"{split}: {s['mean_pct']}% (95% CI {s['ci95_pct']}), {s['passed']}/{s['n']} passed, "
            f"scored {s['scored']}; by type "
            + ", ".join(f"{t} {v['mean_pct']}% (n={v['n']})" for t, v in s["by_type"].items())
        )
        mine = [r for r in rows if r["split"] == split]
        if split == "test":
            print("  test passed: " + " ".join(r["id"] for r in mine if r.get("pass")))
            print("  test failed: " + " ".join(r["id"] for r in mine if not r.get("pass")))
        else:
            for r in mine:
                score = "-" if r.get("score") is None else f"{r['score']:.2f}"
                infra = " ".join(r.get("infra") or [])
                print(f"  {r['id']:<10} {score:>5} {r.get('outcome')} {infra}")
    noise = results.get("noise") or {}
    if "regraded" in noise:
        print(
            f"judge regrade: {noise['regraded']} answers, verdict flip rate "
            f"{noise['verdict_flip_rate']}, score flips {noise['score_flips']}, pass flips "
            f"{noise['pass_flips']}, judge sd {noise.get('judge_sd')}"
        )
    if "repeated" in noise:
        print(
            f"answer re-run: {noise['repeated']} cases, {noise['repeat_score_changes']} scores "
            f"moved, mean {noise['repeat_mean_diff_pct']} points, sd {noise.get('answer_sd')}"
        )
    if noise:
        print(
            f"per-case sd {noise.get('per_case_sd')} "
            f"(90% upper bound {noise.get('per_case_sd_upper90')}); "
            f"minimum difference worth acting on {noise.get('min_diff_pct')} points"
        )
    print(f"infra: {results['infra'] or 'none'}; spend {results['spend_usd']}")
    if results["headroom_warning"]:
        print(f"WARNING: headroom: {results['headroom_warning']} at or above 95%")


# ---------------------------------------------------------------------------
# compare, regrade
# ---------------------------------------------------------------------------


def _cmd_compare(args: argparse.Namespace) -> int:
    a = json.loads(Path(args.a).read_text(encoding="utf-8"))
    b = json.loads(Path(args.b).read_text(encoding="utf-8"))
    try:
        out = ex.compare_results(a, b)
    except ValueError as exc:
        print(f"refused: {exc}; `rejudge` one run with the other's judge first", file=sys.stderr)
        return 2
    for split, s in out["splits"].items():
        if s is None:
            print(f"{split}: no common scored cases")
            continue
        print(
            f"{split}: A {s['a_pct']}% -> B {s['b_pct']}% ({s['diff_pct']:+.1f} points, 95% CI "
            f"{s['diff_ci95_pct']}), minimum difference {s['min_diff_pct']} points, n={s['n']}"
        )
    print(f"{out['decision']}: {out['why']}")
    return 0


def _cmd_regrade(args: argparse.Namespace) -> int:
    run_dir = Path(args.run)
    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    transcripts = json.loads((run_dir / "train-transcripts.json").read_text(encoding="utf-8"))
    cases = {
        i: c
        for c in load_split("train")
        if (i := c.id) in transcripts and transcripts[i]["question"] == c.question
    }
    rows = {r["id"]: r for r in results["rows"] if r["split"] == "train"}
    _refresh_checks(args, cases, [transcripts[i] for i in cases])
    items = [
        ("train", cases[i], transcripts[i])
        for i in sorted(cases)
        if ex.needs_judge(cases[i], transcripts[i], transcripts[i]["checks"])
    ]
    if args.limit:
        items = random.Random(args.seed).sample(items, min(args.limit, len(items)))
    meter = ex.default_meter()
    budget = ex.Budget(meter, args.max_usd)
    client, sink = _client(_api_key(args), meter, args.judge_model)
    first = _judge_all(items, client, args.judge_model, budget, args.workers)
    second = _judge_all(items, client, args.judge_model, budget, args.workers)
    pairs = [
        (c, t, first[c.id], second[c.id])
        for _s, c, t in items
        if isinstance(first.get(c.id), dict) and isinstance(second.get(c.id), dict)
    ]
    stats = _flip_stats(pairs)
    agree = total = 0
    for c, _t, v1, _v2 in pairs:
        ref = (rows.get(c.id) or {}).get("verdict")
        if not ref:
            continue
        for key in ("stated", "supported"):
            for x, y in zip(ref[key], v1[key], strict=False):
                total += 1
                agree += x == y
    stats["agreement_with_run_judge"] = round(agree / total, 4) if total else None
    stats["judge_model"] = args.judge_model
    stats["run_judge_model"] = results["config"]["judge_model"]
    stats["failed"] = len(items) - len(pairs)
    stats["cost_usd"] = round(ex._sink_total(sink), 4)
    print(json.dumps(stats, indent=1))
    out = run_dir / f"regrade-{args.judge_model.replace('/', '_')}.json"
    out.write_text(json.dumps(stats, indent=1) + "\n", encoding="utf-8")
    return 0


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------


def _graph_url_args(p: argparse.ArgumentParser) -> None:
    for g in GRAPHS:
        p.add_argument(f"--{g}-url", default=None, help=f"default {GRAPHS[g].default_url}")


def _key_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--env-file", default=None, help="default: the main checkout's .env")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)


def add_chat_exam_subcommand(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("chat-exam", help="the chat exam: seed, build, run, compare, regrade")
    csub = p.add_subparsers(dest="chat_exam_command", required=True)

    s = csub.add_parser("seed", help="build the local graphs (no model call)")
    s.add_argument("--graph", choices=("demo", "world", "both"), default="both")
    _graph_url_args(s)
    s.set_defaults(func=_cmd_seed)

    b = csub.add_parser("build", help="write the cases and the split from the seeded graphs")
    _graph_url_args(b)
    b.set_defaults(func=_cmd_build)

    r = csub.add_parser("run", help="answer, check, judge and score one split")
    r.add_argument("--split", choices=("train", "test", "both", "smoke"), default="train")
    r.add_argument("--mode", choices=("agent", "single"), default="agent")
    r.add_argument("--model", default=ex.DEFAULT_MODEL)
    r.add_argument("--judge-model", default=ex.DEFAULT_JUDGE_MODEL)
    r.add_argument("--max-usd", type=float, required=True, help="this run's cap")
    r.add_argument("--max-tool-calls", type=int, default=None, help="agent mode override")
    r.add_argument("--ids", default="", help="comma-separated case ids")
    r.add_argument("--limit", type=int, default=0, help="a seeded random sample of this size")
    r.add_argument("--regrade", type=int, default=0, help="grade this many answers twice")
    r.add_argument(
        "--repeat", type=int, default=0, help="answer this many train cases a second time"
    )
    r.add_argument("--label", default="")
    r.add_argument(
        "--reuse",
        default="",
        metavar="RUN_DIR",
        help="take answers (and re-run answers) from a finished run of the same config for "
        "cases asking the same question on the same graph; answer only the rest",
    )
    _graph_url_args(r)
    _key_args(r)
    r.set_defaults(func=_cmd_run)

    c = csub.add_parser("compare", help="KEEP, REVERT or NOISE for run B against run A")
    c.add_argument("a", help="baseline results.json")
    c.add_argument("b", help="candidate results.json")
    c.set_defaults(func=_cmd_compare)

    j = csub.add_parser("rejudge", help="grade a finished run again with another judge")
    j.add_argument("--run", required=True, help="a run directory under var/evals/chat-exam/")
    j.add_argument("--judge-model", required=True)
    j.add_argument("--regrade", type=int, default=0, help="grade this many answers twice")
    j.add_argument("--max-usd", type=float, required=True)
    _graph_url_args(j)
    _key_args(j)
    j.set_defaults(func=_cmd_rejudge)

    s2 = csub.add_parser("rescore", help="score a finished run against the current exam, no calls")
    s2.add_argument("--run", required=True, help="a run directory under var/evals/chat-exam/")
    s2.set_defaults(func=_cmd_rescore)

    g = csub.add_parser("regrade", help="a judge's consistency on a run's train answers")
    g.add_argument("--run", required=True, help="a run directory under var/evals/chat-exam/")
    g.add_argument("--judge-model", default=ex.DEFAULT_JUDGE_MODEL)
    g.add_argument("--limit", type=int, default=0)
    g.add_argument("--max-usd", type=float, required=True)
    _graph_url_args(g)
    _key_args(g)
    g.set_defaults(func=_cmd_regrade)
