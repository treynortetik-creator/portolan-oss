"""Graph retrieval bench (`portolan.bench.graph_search`, `portolan.bench.graph_world`).

    uv run python scripts/bench/graph_retrieval.py                        # demo graph, fake
    uv run python scripts/bench/graph_retrieval.py --embedder fastembed   # local bge-large
    uv run python scripts/bench/graph_retrieval.py --graph world --world medical_devices:test:7

`--graph demo` seeds the demo graph (fake extractor over `corpus/out`); `--graph world`
builds a v2 world's graph from its simulated truth (`graph_world.seed_world_graph`). Each
goes into its own local `portolan_test_graph_*` database (public schema dropped first)
unless `--reuse` finds it seeded. Every mode and hop count runs over every question set;
the table and `var/bench/graph_retrieval/<graph>-<embedder>.json` carry requirement
recall at 10, answer support, MRR@10 and latency. No paid call: both embedders are local.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from portolan import db as db_module
from portolan.bench.graph_search import (
    MULTI_HOP_QUESTIONS,
    evaluate,
    graph_params,
    rescoring_overhead,
    single_hop_questions,
    window_questions,
)
from portolan.search.embed import FakeEmbedder, FastEmbedEmbedder
from portolan.util.redact import redact_dsn

_LOCAL = {"localhost", "127.0.0.1", "::1", ""}


def _connect_fresh(url: str, reuse: bool):
    name = urlsplit(url).path.lstrip("/")
    host = urlsplit(url).hostname or ""
    if not name.startswith("portolan_test") or host not in _LOCAL:
        raise SystemExit(f"refusing {redact_dsn(url)}: local portolan_test* only")
    conn = db_module.connect(url)
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass('public.notes') IS NOT NULL")
        seeded = False
        if cur.fetchone()[0]:
            cur.execute("SELECT count(*) > 0 FROM notes")
            seeded = cur.fetchone()[0]
    if reuse and seeded:
        conn.commit()
        conn.autocommit = True
        return conn, False
    with conn.cursor() as cur:
        cur.execute("SET synchronous_commit = off")
        cur.execute("DROP SCHEMA IF EXISTS public CASCADE")
        cur.execute("CREATE SCHEMA public")
    conn.commit()
    db_module.apply_migrations(conn)
    conn.commit()
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("SET synchronous_commit = off")
    return conn, True


def _parse_params(spec: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in filter(None, spec.split(",")):
        name, value = item.split("=", 1)
        try:
            out[name] = float(value) if "." in value else int(value)
        except ValueError:
            out[name] = value
    return out


def _erb_overhead(args: argparse.Namespace, embedder: Any) -> None:
    """The graph modes' own cost at ERB's size (152,697 notes, no links): read-only on
    the local `portolan_erb`, the first 20 dev questions. ERB documents carry no links
    the graph could use (write-up, "ERB"), so only the cost is measured there."""
    from portolan.bench.erb import (
        default_erb_root,
        load_questions,
        select_in_slice_questions,
        split_dev_heldout,
    )
    from portolan.bench.erb_search import assert_erb_database, erb_database_url

    url = args.database_url or erb_database_url()
    assert_erb_database(url)
    conn = db_module.connect(url)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("SET default_transaction_read_only = on")
    selected = select_in_slice_questions(load_questions(default_erb_root() / "questions.jsonl"))
    dev, _held_out = split_dev_heldout(q["question_id"] for q in selected.questions)
    questions = [q["question"] for q in selected.questions if q["question_id"] in set(dev)][:20]
    result = rescoring_overhead(conn, embedder, questions)
    print(json.dumps(result, indent=2))
    path = Path(args.out_dir) / "erb-overhead.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="graph_retrieval")
    parser.add_argument("--graph", choices=("demo", "world", "erb"), default="demo")
    parser.add_argument("--world", default="medical_devices:test:7", help="pack:split:seed")
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--embedder", choices=("fake", "fastembed"), default="fake")
    parser.add_argument("--reuse", action="store_true", help="skip seeding if notes exist")
    parser.add_argument("--modes", default="off,expand,entity,ppr")
    parser.add_argument("--hops", default="1,2")
    parser.add_argument("--sets", default="single,multi,window")
    parser.add_argument("--out-dir", default="var/bench/graph_retrieval")
    parser.add_argument(
        "--params",
        default="",
        help="graph module overrides, e.g. COMBINE=add,STEP_DECAY=0.7 (label them in --tag)",
    )
    parser.add_argument("--tag", default="", help="suffix for the output file name")
    parser.add_argument(
        "--reembed",
        action="store_true",
        help="re-chunk and re-embed every note with --embedder first (a copied database)",
    )
    args = parser.parse_args()
    overrides = _parse_params(args.params)

    embedder = FastEmbedEmbedder() if args.embedder == "fastembed" else FakeEmbedder()
    if args.graph == "erb":
        _erb_overhead(args, embedder)
        return
    url = args.database_url or (
        f"postgresql://localhost:5432/portolan_test_graph_{args.graph}_{args.embedder}"
    )
    conn, fresh = _connect_fresh(url, args.reuse)
    if args.reembed and not fresh:
        from portolan.search.index import reindex_all

        with conn.cursor() as cur:
            cur.execute("DELETE FROM note_index_state")
        print(f"re-embedded: {reindex_all(conn, embedder)}")
    world_questions: dict[str, Any] = {}
    if args.graph == "demo":
        from portolan.bench.demo_search import seed_demo_graph

        if fresh:
            seed_demo_graph(conn, Path(tempfile.mkdtemp()), embedder)
        sets = {
            "single": single_hop_questions(),
            "multi": MULTI_HOP_QUESTIONS,
            "window": window_questions(conn),
        }
    else:
        from portolan.bench import graph_world

        world, plan = graph_world.build_world(args.world)
        if fresh:
            report = graph_world.seed_world_graph(
                conn, world, plan, Path(tempfile.mkdtemp()), embedder
            )
            print(f"seeded world graph: {report}")
        world_questions = graph_world.world_questions(conn, world, plan)
        sets = {name: world_questions[name] for name in ("single", "multi", "window")}
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM notes")
        n_notes = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM links WHERE kind IN ('body', 'parent')")
        n_links = cur.fetchone()[0]
    print(f"graph {args.graph}: {n_notes} notes, {n_links} body/parent links")

    wanted_sets = [s for s in args.sets.split(",") if s]
    runs: dict[str, Any] = {}
    for set_name in wanted_sets:
        questions = sets[set_name]
        print(f"\n== {set_name} ({len(questions)} questions)")
        print(
            f"{'mode':<14}{'recall@10':>10}{'support':>9}{'ans@10':>8}{'ans sup':>9}"
            f"{'hit@1':>8}{'mrr@10':>8}{'p50 ms':>9}{'p95 ms':>9}{'pool':>7}{'win prec':>10}"
        )
        configs: list[tuple[str, str, int, bool, bool]] = []
        for use_window in [False, True] if set_name == "window" else [False]:
            for mode in args.modes.split(","):
                hop_list = [1] if mode == "off" else [int(h) for h in args.hops.split(",")]
                for hops in hop_list:
                    label = mode if mode == "off" else f"{mode}/h{hops}"
                    configs.append(
                        (label + ("+win" if use_window else ""), mode, hops, use_window, False)
                    )
        if set_name == "window":
            configs.insert(1, ("off+since", "off", 1, False, True))
        for label, mode, hops, use_window, use_since in configs:
            with graph_params(**overrides):
                out = evaluate(
                    conn,
                    embedder,
                    questions,
                    mode=mode,
                    hops=hops,
                    use_window=use_window,
                    use_since=use_since,
                )
            runs[f"{set_name}:{label}"] = out
            o = out["overall"]
            wp = o.get("window_precision_pct")
            print(
                f"{label:<14}{o['requirement_recall_pct']:>10.1f}"
                f"{o['supported_pct']:>9.1f}{o['answer_recall_pct']:>8.1f}"
                f"{o['answer_supported_pct']:>9.1f}{o['hit_at_1_pct']:>8.1f}"
                f"{o['mrr_at_10']:>8.3f}{o['p50_ms']:>9.1f}{o['p95_ms']:>9.1f}"
                f"{o['mean_pool']:>7.0f}{(f'{wp:.1f}' if wp is not None else '-'):>10}"
            )

    suffix = f"-{args.tag}" if args.tag else ""
    path = Path(args.out_dir) / f"{args.graph}-{args.embedder}{suffix}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "params": {k: str(v) for k, v in overrides.items()},
        "graph": args.graph,
        "world": args.world if args.graph == "world" else None,
        "embedder": args.embedder,
        "notes": n_notes,
        "links": n_links,
        "questions": {
            name: [
                {
                    "qid": q.qid,
                    "question": q.question,
                    "category": q.category,
                    "hop": q.hop,
                    "requirements": [list(r) for r in q.requirements],
                    "window": [d.isoformat() for d in q.window] if q.window else None,
                }
                for q in sets[name]
            ]
            for name in wanted_sets
        },
        "runs": runs,
    }
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
