"""Reranker and query-routing exam (`portolan.bench.rerank_eval`), $0: local
models only, the ERB index in `portolan_erb`, the demo graph in a local
`portolan_test*` database seeded by `scripts/bench/demo_search.py`.

    uv run python scripts/bench/rerank_exam.py legs --set erb
    uv run python scripts/bench/rerank_exam.py legs --set demo --demo-url <portolan_test_*>
    uv run python scripts/bench/rerank_exam.py route --demo-url <same>
    uv run python scripts/bench/rerank_exam.py score --model bge-reranker-v2-m3 --pool 50
    uv run python scripts/bench/rerank_exam.py table --demo-url <same>
    uv run python scripts/bench/rerank_exam.py latency --model ms-marco-minilm-l6 --demo-url <same>

- `legs`: each question's retrieval legs, once (`var/bench/rerank/legs-<set>.json`,
  note text included, so it stays in var/).
- `route`: the routing grid on ERB dev, the paraphrases and the demo questions;
  chooses by `rerank_eval.choose` and writes `route.json`. The chosen values
  then go into `portolan/search/query.py` by hand, and every later step reads
  them from there.
- `score`: every note a configuration with this pool could rerank, scored by the
  model one pair at a time (`scores-<model>.json`, resumable).
- `table`: every configuration through `_rank` itself; the choice; held-out at
  the choice; writes `table.json` and, with `--results`, the ids-and-numbers file
  for 01-spec/.
- `latency`: `search()` itself with the real model (no recorded scores), per
  question, for the load-noted wall-clock check of the estimate.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

import psycopg

from portolan.bench.demo_search import DEMO_QUESTIONS, PARAPHRASE_QUESTIONS
from portolan.bench.erb import (
    DEV_SPLIT_SEED,
    ERB_SLICES,
    TOP_K,
    default_erb_root,
    load_questions,
    load_uuid_index,
    select_in_slice_questions,
    split_dev_heldout,
)
from portolan.bench.erb_search import CachingEmbedder, assert_erb_database, erb_database_url
from portolan.bench.rerank_eval import (
    ROUTE_GRID,
    choose,
    evaluate,
    load_legs,
    pool_slugs,
    record_legs,
    route_profile,
    save_legs,
    score_candidates,
    summarize,
    write_json,
)
from portolan.search import query as q
from portolan.search.embed import FastEmbedEmbedder
from portolan.search.rerank import RERANK_MODELS, OnnxReranker
from portolan.util.redact import redact_dsn

POOLS = (30, 50)


def _log(msg: str) -> None:
    load = ", ".join(f"{x:.1f}" for x in os.getloadavg())
    print(f"[{datetime.now(UTC):%H:%M:%S}Z load {load}] {msg}", flush=True)


def _erb_questions(root: Path) -> list[dict]:
    uuid_index = load_uuid_index(root)
    questions = load_questions(root / "questions.jsonl")
    sel = select_in_slice_questions(questions, ERB_SLICES, uuid_index=uuid_index)
    dev, held = split_dev_heldout(x["question_id"] for x in sel.questions)
    half = {**dict.fromkeys(dev, "dev"), **dict.fromkeys(held, "held_out")}
    return [{**x, "set": "erb", "half": half[x["question_id"]]} for x in sel.questions]


def _demo_questions() -> list[dict]:
    out = []
    for name, qs in (("demo", DEMO_QUESTIONS), ("paraphrase", PARAPHRASE_QUESTIONS)):
        for x in qs:
            out.append(
                {
                    "question_id": x.qid,
                    "question": x.question,
                    "expected_doc_ids": list(x.expected),
                    "question_type": name,
                    "set": name,
                    "half": "all",
                }
            )
    return out


def _demo_conn(url: str) -> psycopg.Connection:
    name = urlsplit(url).path.lstrip("/")
    host = urlsplit(url).hostname or ""
    if not name.startswith("portolan_test") or host not in {"localhost", "127.0.0.1", "::1", ""}:
        raise SystemExit(f"refusing {redact_dsn(url)}: local portolan_test* only")
    return psycopg.connect(url, autocommit=True)


def _erb_conn(args: argparse.Namespace) -> psycopg.Connection:
    url = erb_database_url(args.pg_host, args.pg_port)
    assert_erb_database(url)
    return psycopg.connect(url, autocommit=True)


def _sets(args: argparse.Namespace) -> dict[str, tuple[list[dict], psycopg.Connection]]:
    """{"erb": (questions, conn), "demo": (demo + paraphrase questions, conn)}"""
    out = {}
    if "erb" in args.sets:
        out["erb"] = (_erb_questions(Path(args.erb_root).expanduser()), _erb_conn(args))
    if "demo" in args.sets:
        out["demo"] = (_demo_questions(), _demo_conn(args.demo_url))
    return out


def cmd_legs(args: argparse.Namespace) -> None:
    if args.set == "erb":
        questions, conn = _erb_questions(Path(args.erb_root).expanduser()), _erb_conn(args)
    else:
        questions, conn = _demo_questions(), _demo_conn(args.demo_url)
    embedder = CachingEmbedder(FastEmbedEmbedder())
    _log(f"legs: {len(questions)} {args.set} questions")
    legs = record_legs(conn, embedder, questions, progress=_log)
    save_legs(Path(args.out_dir) / f"legs-{args.set}.json", legs)
    secs = sorted(v["seconds"] for v in legs.values())
    _log(f"wrote legs-{args.set}.json; legs median {secs[len(secs) // 2] * 1000:.0f} ms")


def _load_all_legs(out_dir: Path, sets: dict) -> dict[str, dict]:
    return {name: load_legs(out_dir / f"legs-{name}.json") for name in sets}


def _groups(questions: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for x in questions:
        if x["set"] == "erb":
            groups.setdefault(f"erb_{x['half']}", []).append(x)
            groups.setdefault("erb_all", []).append(x)
        else:
            groups.setdefault(x["set"], []).append(x)
    return groups


def _run_config(sets, legs, *, fusion_routing, model=None, scores=None, pool=30) -> dict:
    """{group: rows} for one configuration over every loaded set."""
    out: dict[str, list[dict]] = {}
    for name, (questions, conn) in sets.items():
        for group, qs in _groups(questions).items():
            out[group] = evaluate(
                conn,
                qs,
                legs[name],
                fusion_routing=fusion_routing,
                model=model,
                scores=None if scores is None else scores,
                pool=pool,
            )
    return out


def _rule_inputs(rows: dict[str, list[dict]]) -> dict[str, float]:
    return {
        "erb_dev": summarize(rows["erb_dev"])["average_recall_pct"],
        "paraphrase": summarize(rows["paraphrase"])["average_recall_pct"],
        "demo": summarize(rows["demo"])["average_recall_pct"],
    }


def cmd_route(args: argparse.Namespace) -> None:
    args.sets = ["erb", "demo"]
    sets = _sets(args)
    legs = _load_all_legs(Path(args.out_dir), sets)
    for name, (questions, _c) in sets.items():
        for group, qs in _groups(questions).items():
            cov = sorted(legs[name][x["question_id"]][0].coverage for x in qs)
            quart = [round(cov[int(p * (len(cov) - 1))], 3) for p in (0, 0.25, 0.5, 0.75, 1)]
            print(f"coverage {group:12s} n={len(cov):3d} min/q1/median/q3/max {quart}")
    grid: dict[str, dict] = {}
    for threshold, semantic in ROUTE_GRID:
        with route_profile(threshold, semantic):
            rows = _run_config(sets, legs, fusion_routing=True)
        name = f"route t{threshold} v{semantic}"
        grid[name] = {
            "threshold": threshold,
            "semantic_vector_weight": semantic,
            **_rule_inputs(rows),
            "erb_held_out": summarize(rows["erb_held_out"])["average_recall_pct"],
        }
        g = grid[name]
        print(
            f"{name:22s} ERB dev {g['erb_dev']:6.2f}  paraphrase {g['paraphrase']:6.2f}  "
            f"demo {g['demo']:6.2f}  (held-out {g['erb_held_out']:6.2f})"
        )
    chosen = choose(dict(grid))
    print(f"chosen: {chosen}")
    write_json(
        Path(args.out_dir) / "route.json",
        {
            "run_at": datetime.now(UTC).isoformat(),
            "keyword_profile": list(q.ROUTE_KEYWORD_WEIGHTS),
            "rule": choose.__doc__,
            "grid": grid,
            "chosen": grid[chosen],
        },
    )


def _scores_path(out_dir: Path, model: str) -> Path:
    return out_dir / f"scores-{model}.json"


def cmd_score(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    sets = _sets(args)
    legs = _load_all_legs(out_dir, sets)
    path = _scores_path(out_dir, args.model)
    scores: dict[str, dict[str, list[float]]] = (
        json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    )
    reranker = OnnxReranker(RERANK_MODELS[args.model], allow_download=False)
    halves = set(args.halves.split(","))
    n_pairs = 0
    t_start = time.monotonic()
    for name, (questions, conn) in sets.items():
        todo = [x for x in questions if x["half"] in halves or x["set"] != "erb"]
        for i, x in enumerate(todo):
            qid = x["question_id"]
            leg = legs[name][qid][0]
            # --pool 0: every fused candidate (the legs' union), so any routing
            # or pool can be ranked later; else the top `pool` under either routing.
            slugs: list[str] = []
            for routing in (False, True):
                for s in pool_slugs(conn, leg, pool=args.pool or 10_000, fusion_routing=routing):
                    if s not in slugs:
                        slugs.append(s)
            done = scores.setdefault(qid, {})
            new = score_candidates(conn, reranker, leg, slugs, done)
            done.update(new)
            n_pairs += len(new)
            if new:
                write_json(path, scores)
            if (i + 1) % 10 == 0 or i + 1 == len(todo):
                rate = (time.monotonic() - t_start) / max(1, n_pairs)
                _log(
                    f"{args.model} {name}: {i + 1}/{len(todo)}, {n_pairs} pairs, {rate:.2f} s/pair"
                )


def cmd_table(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    args.sets = ["erb", "demo"]
    sets = _sets(args)
    legs = _load_all_legs(out_dir, sets)
    configs: dict[str, dict] = {"live": {"fusion_routing": False}}
    configs["routing"] = {"fusion_routing": True}
    for model in args.models.split(","):
        path = _scores_path(out_dir, model)
        if not path.is_file():
            continue
        scores = json.loads(path.read_text(encoding="utf-8"))
        for pool in POOLS:
            for routing, label in ((False, "rerank"), (True, "both")):
                configs[f"{label} {model} @{pool}"] = {
                    "fusion_routing": routing,
                    "model": model,
                    "scores": scores,
                    "pool": pool,
                }
    table: dict[str, dict] = {}
    all_rows: dict[str, dict] = {}
    for name, cfg in configs.items():
        try:
            rows = _run_config(sets, legs, **cfg)
        except KeyError as exc:  # a pool this model was not scored for
            print(f"skip {name}: unscored note {exc}")
            continue
        table[name] = {group: summarize(r) for group, r in rows.items()}
        all_rows[name] = rows
        t = table[name]
        print(
            f"{name:44s} dev {t['erb_dev']['average_recall_pct']:6.2f}  "
            f"held {t['erb_held_out']['average_recall_pct']:6.2f}  "
            f"para {t['paraphrase']['average_recall_pct']:6.2f}  "
            f"demo {t['demo']['average_recall_pct']:6.2f}  "
            f"ERB p50/p95 {t['erb_all']['p50_ms']:.0f}/{t['erb_all']['p95_ms']:.0f} ms  "
            f"demo p50/p95 {t['demo']['p50_ms']:.0f}/{t['demo']['p95_ms']:.0f} ms"
        )
    chosen = choose({n: _rule_inputs(all_rows[n]) for n in table})
    print(f"chosen: {chosen}; held-out at the choice {table[chosen]['erb_held_out']}")
    payload = {
        "run_at": datetime.now(UTC).isoformat(),
        "split_seed": DEV_SPLIT_SEED,
        "route": {
            "threshold": q.ROUTE_COVERAGE_THRESHOLD,
            "keyword_weights": list(q.ROUTE_KEYWORD_WEIGHTS),
            "semantic_weights": list(q.ROUTE_SEMANTIC_WEIGHTS),
        },
        "rule": choose.__doc__,
        "chosen": chosen,
        "table": table,
    }
    write_json(out_dir / "table.json", payload)
    if args.results:
        slim = {
            name: {
                group: [
                    {
                        k: r[k]
                        for k in (
                            "question_id",
                            "retrieved_doc_ids",
                            "document_recall_pct",
                            "first_gold_rank",
                            "coverage",
                            "seconds",
                            "model_seconds",
                        )
                    }
                    for r in rows
                ]
                for group, rows in groups.items()
                if group != "erb_all"
            }
            for name, groups in all_rows.items()
        }
        write_json(Path(args.results), {**payload, "rows": slim})
        print(f"wrote {args.results}")


def cmd_latency(args: argparse.Namespace) -> None:
    """`search()` itself, real model, per question: wall clock with the query
    vector cached, rerank pool from the flag, routing from the flag."""
    sets = _sets(args)
    reranker = (
        OnnxReranker(RERANK_MODELS[args.model], allow_download=False) if args.model else False
    )
    if reranker:
        reranker.score("warm up", ["warm up"])
    embedder = CachingEmbedder(FastEmbedEmbedder())
    out = {}
    for name, (questions, conn) in sets.items():
        qs = questions[: args.limit] if args.limit else questions
        for x in qs:
            embedder.embed([x["question"]])
        times = []
        for _rep in range(args.reps):
            for x in qs:
                t0 = time.monotonic()
                q.search(
                    conn,
                    embedder,
                    x["question"],
                    k=TOP_K,
                    fusion_routing=args.routing,
                    reranker=reranker,
                    rerank_pool=args.pool,
                )
                times.append({"seconds": time.monotonic() - t0})
        from portolan.bench.rerank_eval import latency_summary

        out[name] = {"n": len(times), **latency_summary(times)}
        _log(f"{name}: {out[name]}")
    print(json.dumps(out))


def main() -> None:
    parser = argparse.ArgumentParser(prog="rerank_exam")
    parser.add_argument("--out-dir", default="var/bench/rerank")
    parser.add_argument("--erb-root", default=str(default_erb_root()))
    parser.add_argument("--pg-host", default="localhost")
    parser.add_argument("--pg-port", type=int, default=5432)
    parser.add_argument(
        "--demo-url", default="postgresql://localhost:5432/portolan_test_rerank_demo"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    legs = sub.add_parser("legs")
    legs.add_argument("--set", choices=("erb", "demo"), required=True)
    sub.add_parser("route")
    score = sub.add_parser("score")
    score.add_argument("--model", choices=sorted(RERANK_MODELS), required=True)
    score.add_argument("--pool", type=int, default=0, help="0: every fused candidate")
    score.add_argument("--sets", default="demo,erb", type=lambda s: s.split(","))
    score.add_argument("--halves", default="dev,held_out")
    table = sub.add_parser("table")
    table.add_argument("--models", default=",".join(RERANK_MODELS))
    table.add_argument("--results", default="")
    lat = sub.add_parser("latency")
    lat.add_argument("--model", choices=sorted(RERANK_MODELS), default=None)
    lat.add_argument("--pool", type=int, default=30)
    lat.add_argument("--routing", action="store_true")
    lat.add_argument("--sets", default="demo", type=lambda s: s.split(","))
    lat.add_argument("--limit", type=int, default=0)
    lat.add_argument("--reps", type=int, default=1)
    args = parser.parse_args()
    {
        "legs": cmd_legs,
        "route": cmd_route,
        "score": cmd_score,
        "table": cmd_table,
        "latency": cmd_latency,
    }[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
