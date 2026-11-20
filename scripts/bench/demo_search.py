"""Search check on the demo graph (`portolan.bench.demo_search`).

    uv run python scripts/bench/demo_search.py                       # FakeEmbedder
    uv run python scripts/bench/demo_search.py --embedder fastembed  # local bge-large
    uv run python scripts/bench/demo_search.py --embedder fastembed --reuse \\
        --questions paraphrase --weights 1.0/0.2,1.0/1.0             # paraphrase check

Seeds the graph into `--database-url` (a local `portolan_test*` database,
whose public schema is dropped first) unless `--reuse` finds it already
seeded, then prints recall@10, hit@1 and MRR@10 per question and overall
for each `--weights` pair (keyword/vector; default: search()'s own), and
writes `var/bench/demo_search/<embedder>[-<questions>].json`. With
`--questions paraphrase` it also prints each question's lexeme overlap
with its gold notes. No paid call: both embedders run locally.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

from portolan import db as db_module
from portolan.bench.demo_search import (
    DEMO_QUESTIONS,
    PARAPHRASE_QUESTIONS,
    evaluate_demo_search,
    fusion_weights,
    lexeme_overlap,
    seed_demo_graph,
)
from portolan.search.embed import FakeEmbedder, FastEmbedEmbedder
from portolan.util.redact import redact_dsn

QUESTION_SETS = {"demo": DEMO_QUESTIONS, "paraphrase": PARAPHRASE_QUESTIONS}


def _weights(spec: str) -> list[tuple[float, float] | None]:
    if not spec:
        return [None]
    pairs = []
    for item in spec.split(","):
        fts, vector = item.split("/")
        pairs.append((float(fts), float(vector)))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(prog="demo_search")
    parser.add_argument(
        "--database-url", default="postgresql://localhost:5432/portolan_test_demo_search"
    )
    parser.add_argument("--embedder", choices=("fake", "fastembed"), default="fake")
    parser.add_argument("--reuse", action="store_true", help="skip seeding if notes exist")
    parser.add_argument("--questions", choices=sorted(QUESTION_SETS), default="demo")
    parser.add_argument(
        "--weights", default="", help="comma list of keyword/vector RRF weights, e.g. 1.0/0.2"
    )
    parser.add_argument("--out-dir", default="var/bench/demo_search")
    args = parser.parse_args()

    name = urlsplit(args.database_url).path.lstrip("/")
    host = urlsplit(args.database_url).hostname or ""
    if not name.startswith("portolan_test") or host not in {"localhost", "127.0.0.1", "::1", ""}:
        raise SystemExit(f"refusing {redact_dsn(args.database_url)}: local portolan_test* only")
    embedder = FastEmbedEmbedder() if args.embedder == "fastembed" else FakeEmbedder()
    questions = QUESTION_SETS[args.questions]

    conn = db_module.connect(args.database_url)
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass('public.notes') IS NOT NULL")
        has_notes_table = cur.fetchone()[0]
        seeded = False
        if has_notes_table:
            cur.execute("SELECT count(*) > 0 FROM notes")
            seeded = cur.fetchone()[0]
    if not (args.reuse and seeded):
        with conn.cursor() as cur:
            cur.execute("DROP SCHEMA public CASCADE")
            cur.execute("CREATE SCHEMA public")
        conn.commit()
        db_module.apply_migrations(conn)
        conn.commit()
        conn.autocommit = True
        seed_demo_graph(conn, Path(tempfile.mkdtemp()), embedder)
    conn.commit()
    conn.autocommit = True

    overlap = {q.qid: round(lexeme_overlap(conn, q), 3) for q in questions}
    runs = {}
    for pair in _weights(args.weights):
        label = "default" if pair is None else f"{pair[0]}/{pair[1]}"
        with fusion_weights(*pair) if pair else contextlib.nullcontext():
            out = evaluate_demo_search(conn, embedder, questions=questions)
        runs[label] = out
        print(f"weights {label}")
        for row in out["rows"]:
            print(
                f"  {row['question_id']}  recall {row['document_recall_pct']:6.1f}  "
                f"first gold rank {row['first_gold_rank']}  "
                f"lexeme overlap {overlap[row['question_id']]:.2f}"
            )
        o = out["overall"]
        print(
            f"overall ({args.embedder}, {args.questions}, {label}): "
            f"recall@10 {o['average_recall_pct']:.2f}  hit@1 {o['ours_hit_at_1_pct']:.2f}  "
            f"mrr@10 {o['ours_mrr_at_10']:.3f}  n={o['count']}"
        )
    mean_overlap = sum(overlap.values()) / len(overlap)
    print(f"mean lexeme overlap with gold notes: {mean_overlap:.3f}")

    suffix = "" if args.questions == "demo" else f"-{args.questions}"
    path = Path(args.out_dir) / f"{args.embedder}{suffix}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = runs["default"] if list(runs) == ["default"] else {"runs": runs}
    payload = {**payload, "lexeme_overlap": overlap, "mean_lexeme_overlap": mean_overlap}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
