"""EnterpriseRAG-Bench as Portolan's search exam (checklist item 27).

Thin driver over `portolan.bench.erb` (pure) and `portolan.bench.erb_search`
(database, embedder, passes). The OpenRouter key comes from the environment
(source the repo `.env` into the one process; it is never printed), or from
`--env-file`. Subcommands, in the order a run uses them:

    uv run python scripts/bench/erb_exam.py select
    uv run python scripts/bench/erb_exam.py plan
    uv run python scripts/bench/erb_exam.py load --fresh --reuse-plan --spend-file S
    uv run python scripts/bench/erb_exam.py load --reuse-plan --shard K/4 --spend-file S
    uv run python scripts/bench/erb_exam.py load --reuse-plan --indexes-only --spend-file S
    uv run python scripts/bench/erb_exam.py run --exact-vectors --spend-file S
    uv run python scripts/bench/erb_exam.py run --exact-vectors --local-embedder --spend-file S
    uv run python scripts/bench/erb_exam.py tune
    uv run python scripts/bench/erb_exam.py run --spend-file S      # needs the HNSW index
    uv run python scripts/bench/erb_exam.py ef                      # needs the HNSW index
    uv run python scripts/bench/erb_exam.py weights --demo-results D.json P.json
    uv run python scripts/bench/erb_exam.py answers --spend-file S  # item 28 sample
    uv run python scripts/bench/erb_exam.py repair --context results_exact.json:fused_exact

(S = the shared spend meter, `<main checkout>/var/data/silver/spend.json`.)

- `select` -- which questions fall inside the four slices, and how that
  was derived (no database, no network).
- `plan` -- chunk the corpus exactly as the loader will and estimate the
  embedding bill (no database, no network).
- `load` -- (re)create the local `portolan_erb` database, migrate it,
  check the (remaining) estimate against `--budget-usd`, then chunk, embed
  and write. Resumable; `--shard K/N` runs N writers in parallel. HNSW,
  both GIN indexes and the chunk-to-note btree are dropped for the bulk
  load and rebuilt at the end with the migration's own definitions.
- `run` -- the retrieval passes and ERB's metrics; writes
  `var/bench/erb_exam/results.json` (or `results_exact.json` with
  `--exact-vectors`): ids and numbers only, no ERB text. `--local-embedder`
  embeds the questions with local fastembed (the same model, $0). The
  exact scan is cached in `exact_scan.json`; delete it after reloading
  the index.
- `tune` -- chooses the fusion weights on the fixed dev half only and
  reports every pass on dev, held-out and all; writes `tune.json`.
- `ef` -- `search()` through the HNSW index at several `hnsw.ef_search`
  values; chooses one on the dev half and writes `ef_sweep.json`.
- `weights` -- `search()` at several vector fusion weights; with the demo
  graph's results, chooses one and writes `weight_sweep.json`.
- `answers` -- ERB's answer evaluation on a fixed 20-question sample.
- `repair` -- the same sample through the cited-answer gate, the repair
  pass and the support judge, every mode in one run (`repair.json`).

ERB never enters training (README canary `9d998ebd-...`): this driver
writes ERB text only into the `portolan_erb` database. Nothing lands in
`var/data/silver`, `var/data/synth`, `var/data/training` or `corpus/`
except the dollar amount recorded against corpus key `erb_exam` in the
shared spend meter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import psycopg

from portolan import db
from portolan.bench.erb import (
    CANARY_GUID,
    DEV_SPLIT_SEED,
    ERB_SLICES,
    Bm25Index,
    IterStats,
    aggregate,
    default_erb_root,
    extract_document_content,
    iter_documents,
    load_questions,
    load_uuid_index,
    select_in_slice_questions,
    split_dev_heldout,
    strip_nul,
)
from portolan.bench.erb_search import (
    BGE_QUERY_INSTRUCTION,
    EF_SEARCH_GRID,
    EF_SEARCH_RECALL_TOLERANCE,
    EMBED_MODEL,
    EMBED_PRICE_PER_MILLION,
    PASS_NAMES,
    SPEND_CORPUS,
    VECTOR_ONLY_K,
    VECTOR_WEIGHT_GRID,
    BudgetExceeded,
    CachingEmbedder,
    LoadReport,
    OpenRouterEmbedder,
    assert_erb_database,
    choose_ef_search,
    choose_vector_weight,
    ensure_erb_database,
    erb_database_url,
    estimate_embedding_cost,
    exact_vector_scan,
    find_bge_tokenizer,
    fusion_rows,
    latency_summary,
    load_documents,
    load_scan,
    read_openrouter_key,
    run_ef_sweep,
    run_exact_passes,
    run_passes,
    run_weight_sweep,
    save_scan,
    tune_fusion_weights,
    write_json,
)
from portolan.search.embed import FastEmbedEmbedder
from portolan.training.spend import SpendMeter, default_spend_path
from portolan.util.redact import redact_dsn

# The four secondary indexes a bulk load maintains row by row at great
# cost: dropped before loading and rebuilt once afterwards, each with its
# definition from portolan/db/migrations/0001_init.sql (the `IF NOT EXISTS`
# form, so a rebuild after a crash-and-resume is idempotent).
DEFERRED_INDEXES = {  # built in this order: cheap first, HNSW last
    "idx_chunks_note_slug": "CREATE INDEX IF NOT EXISTS idx_chunks_note_slug ON chunks (note_slug)",
    "idx_notes_search": "CREATE INDEX IF NOT EXISTS idx_notes_search ON notes USING gin (search)",
    "idx_chunks_search": "CREATE INDEX IF NOT EXISTS idx_chunks_search ON chunks USING gin (search)",
    "idx_chunks_embedding_hnsw": (
        "CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw "
        "ON chunks USING hnsw (embedding vector_cosine_ops)"
    ),
}


def _log(msg: str) -> None:
    print(f"[{datetime.now(UTC):%H:%M:%S}Z] {msg}", flush=True)


def _paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    root = Path(args.erb_root).expanduser()
    return root, root / "generated_data" / "sources", root / "questions.jsonl"


def _require_clone(root: Path) -> None:
    readme = root / "README.md"
    if not readme.is_file():
        sys.exit(f"ERB clone not found at {root} (set --erb-root or ERB_ROOT)")
    if CANARY_GUID not in readme.read_text(encoding="utf-8"):
        sys.exit(f"{root}/README.md does not carry the ERB canary; is this the right clone?")


def _selection(args: argparse.Namespace):
    root, _sources, questions_path = _paths(args)
    _require_clone(root)
    uuid_index = load_uuid_index(root)
    questions = load_questions(questions_path)
    return uuid_index, select_in_slice_questions(questions, ERB_SLICES, uuid_index=uuid_index)


def cmd_select(args: argparse.Namespace) -> None:
    _uuid_index, sel = _selection(args)
    print(sel.describe(ERB_SLICES))
    if sel.derivations_agree is False:
        sys.exit("the two derivations disagree -- not trustworthy")


def _plan(args: argparse.Namespace, uuid_index):
    _root, sources, _q = _paths(args)
    stats = IterStats()
    tok = find_bge_tokenizer()
    _log(f"plan: tokenizer {'found at ' + str(tok) if tok else 'not found (chars/3)'}")
    report = estimate_embedding_cost(
        iter_documents(sources, ERB_SLICES, uuid_index=uuid_index, stats=stats),
        tokenizer_path=tok,
        progress=_log,
    )
    out = report.as_dict()
    out["iteration"] = {
        "files": stats.files,
        "parse_failures": stats.parse_failures,
        "missing_dsid": stats.missing_dsid,
        "duplicate_dsid_skipped": stats.duplicate_dsid_skipped,
        "nul_docs_cleaned": stats.nul_docs,
        "documents_per_source": dict(stats.per_source),
    }
    return report, out


def cmd_plan(args: argparse.Namespace) -> None:
    root, _s, _q = _paths(args)
    _require_clone(root)
    uuid_index = load_uuid_index(root)
    report, out = _plan(args, uuid_index)
    print(json.dumps(out, indent=2))
    print(
        f"ESTIMATE: {out['total_tokens']:,} tokens x ${EMBED_PRICE_PER_MILLION}/M "
        f"({EMBED_MODEL}) = ${report.estimated_usd:.4f}"
    )
    write_json(Path(args.out_dir) / "plan.json", out)


def _meter(args: argparse.Namespace) -> SpendMeter:
    # Default: the machine's one meter (a worktree's var/ maps to the main checkout's),
    # never a path relative to wherever the command was run.
    if args.spend_file:
        return SpendMeter(Path(args.spend_file))
    return SpendMeter(default_spend_path(Path(__file__).resolve().parents[2] / "var"))


def _embedder(args: argparse.Namespace, meter: SpendMeter) -> OpenRouterEmbedder:
    key = read_openrouter_key(Path(args.env_file) if args.env_file else None)
    return OpenRouterEmbedder(
        key,
        meter=meter,
        corpus=SPEND_CORPUS,
        budget_usd=args.budget_usd,
        max_inputs_per_request=args.request_size,
    )


def _connect(args: argparse.Namespace) -> tuple[str, psycopg.Connection]:
    url = erb_database_url(args.pg_host, args.pg_port)
    assert_erb_database(url)
    return url, psycopg.connect(url, autocommit=True)


def cmd_load(args: argparse.Namespace) -> None:
    root, sources, _q = _paths(args)
    _require_clone(root)
    uuid_index = load_uuid_index(root)
    meter = _meter(args)
    already = meter.corpus_usd(SPEND_CORPUS)

    plan_path = Path(args.out_dir) / "plan.json"
    if args.reuse_plan and plan_path.is_file():
        plan_out = json.loads(plan_path.read_text(encoding="utf-8"))
        estimate = float(plan_out["estimated_usd"])
        _log(f"plan: reusing {plan_path} (written by `plan` over the same corpus)")
    else:
        report, plan_out = _plan(args, uuid_index)
        estimate = report.estimated_usd
    print(
        f"ESTIMATE: {plan_out['total_tokens']:,} tokens over {plan_out['total_chunks']:,} chunks "
        f"x ${EMBED_PRICE_PER_MILLION}/M ({EMBED_MODEL}) = ${estimate:.4f}; "
        f"already recorded under {SPEND_CORPUS!r}: ${already:.4f}; cap ${args.budget_usd:.2f}",
        flush=True,
    )
    write_json(Path(args.out_dir) / "plan.json", plan_out)
    # On a resume, only the chunks not yet in portolan_erb will be paid for.
    remaining = estimate
    if not args.fresh:
        url0 = erb_database_url(args.pg_host, args.pg_port)
        assert_erb_database(url0)
        try:
            with psycopg.connect(url0) as c0, c0.cursor() as cur:
                cur.execute("SELECT note_slug FROM note_index_state")
                done = [row[0] for row in cur.fetchall()]
            # Pro-rate the plan's per-source tokens by documents already
            # loaded per source (cheap; a chunks count(*) scans gigabytes).
            per = plan_out["per_source"]
            done_by_source: dict[str, int] = {}
            for dsid in done:
                src = (uuid_index or {}).get(dsid, "?/").split("/", 1)[0]
                done_by_source[src] = done_by_source.get(src, 0) + 1
            remaining_tokens = sum(
                v["tokens"] * max(0.0, 1.0 - done_by_source.get(src, 0) / v["documents"])
                for src, v in per.items()
            )
            remaining = remaining_tokens * EMBED_PRICE_PER_MILLION / 1e6
            print(
                f"RESUME: {len(done):,} notes already loaded; remaining estimate ${remaining:.4f}"
            )
        except psycopg.Error:
            remaining = estimate
    if remaining + already > args.budget_usd:
        sys.exit(
            f"STOP: remaining estimate ${remaining:.4f} + already-spent ${already:.4f} exceeds "
            f"the ${args.budget_usd:.2f} cap. Nothing was embedded."
        )
    if args.dry_run:
        print("dry run: stopping before any paid call")
        return

    url = erb_database_url(args.pg_host, args.pg_port)
    created = ensure_erb_database(url, fresh=args.fresh)
    _log(f"database {redact_dsn(url)} {'created' if created else 'exists'}")
    conn = psycopg.connect(url)
    applied = db.apply_migrations(conn)
    conn.commit()  # apply_migrations leaves its read of schema_migrations open when nothing is new
    conn.autocommit = True
    _log(f"migrations applied: {len(applied)}")
    with conn.cursor() as cur:
        for name in DEFERRED_INDEXES:
            cur.execute(f"DROP INDEX IF EXISTS {name}")
        # Autovacuum re-reading a table that is being bulk-loaded is pure
        # contention; paused here, switched back on once indexes are built.
        for table in ("notes", "chunks"):
            cur.execute(f"ALTER TABLE {table} SET (autovacuum_enabled = false)")
        # Session-only bulk-load settings on this loader connection. With
        # 128 MB of shared_buffers every chunk row's foreign-key check
        # (a SELECT ... FOR KEY SHARE on its note) and random btree insert
        # became disk reads; notes and their chunks are written in the same
        # transaction, so referential integrity holds by construction and
        # is re-verified by query after the load (db_counts below).
        cur.execute("SET synchronous_commit = off")
        cur.execute("SELECT rolsuper FROM pg_roles WHERE rolname = current_user")
        if cur.fetchone()[0]:
            cur.execute("SET session_replication_role = replica")
            _log("bulk load: FK triggers off for this session (verified by query afterwards)")
    _log(f"dropped {', '.join(DEFERRED_INDEXES)} for the bulk load (rebuilt afterwards)")

    embedder = _embedder(args, meter)
    stats = IterStats()
    t0 = time.monotonic()
    docs = iter_documents(sources, ERB_SLICES, uuid_index=uuid_index, stats=stats)
    shard_k, shard_n = (int(x) for x in args.shard.split("/"))
    if shard_n > 1:
        # Disjoint shards so several loader processes (each with its own
        # Postgres backend) can write in parallel; the write path, not the
        # embedding API, was the bottleneck on the loaded dev machine.
        docs = (d for d in docs if _shard_of(d.dsid, shard_n) == shard_k)
        _log(f"shard {shard_k}/{shard_n}: loading only this shard, indexes left to the last run")
    try:
        if args.indexes_only:
            load = LoadReport()
            _log("--indexes-only: skipping the document pass")
        else:
            load = load_documents(
                conn,
                docs,
                embedder,
                batch_docs=args.batch_docs,
                workers=args.workers,
                request_size=args.request_size,
                progress=_log,
            )
    finally:
        embedder.close()
    _log(f"load: {load}")
    _log(
        f"embedder: {embedder.requests} requests, {embedder.prompt_tokens:,} tokens, "
        f"${embedder.spent_usd:.4f} this run, providers {dict(embedder.providers)}"
    )
    if shard_n > 1:
        _log(f"shard {shard_k}/{shard_n} done: {load}")
        conn.close()
        return
    with conn.cursor() as cur:
        cur.execute(
            "SELECT set_config('maintenance_work_mem', %s, false)", (args.maintenance_work_mem,)
        )
        cur.execute(
            "SELECT set_config('max_parallel_maintenance_workers', %s, false)",
            (str(args.index_workers),),
        )
        idx_seconds: dict[str, float] = {}
        for name, ddl in DEFERRED_INDEXES.items():
            t_idx = time.monotonic()
            _log(f"building {name} (migration 0001 definition)...")
            cur.execute(ddl)
            idx_seconds[name] = round(time.monotonic() - t_idx, 1)
            _log(f"built {name} in {idx_seconds[name]}s")
        for table in ("notes", "chunks"):
            cur.execute(f"ALTER TABLE {table} RESET (autovacuum_enabled)")
        cur.execute("VACUUM (ANALYZE) notes")  # also sets the visibility map
        cur.execute("VACUUM (ANALYZE) chunks")
        cur.execute("SELECT count(*) FROM notes")
        n_notes = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM chunks")
        n_chunks = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM chunks WHERE embedding IS NULL")
        n_null = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM note_index_state")
        n_state = cur.fetchone()[0]
        cur.execute(
            "SELECT count(*) FROM chunks c WHERE NOT EXISTS "
            "(SELECT 1 FROM notes n WHERE n.slug = c.note_slug)"
        )
        orphan_chunks = cur.fetchone()[0]
        cur.execute(
            "SELECT count(*) FROM notes n WHERE NOT EXISTS "
            "(SELECT 1 FROM note_index_state s WHERE s.note_slug = n.slug)"
        )
        unindexed_notes = cur.fetchone()[0]
    summary = {
        "loaded_at": datetime.now(UTC).isoformat(),
        "seconds_total": round(time.monotonic() - t0, 1),
        "index_build_seconds": idx_seconds,
        "load": load.__dict__,
        "iteration": {
            "files": stats.files,
            "parse_failures": stats.parse_failures,
            "missing_dsid": stats.missing_dsid,
            "duplicate_dsid_skipped": stats.duplicate_dsid_skipped,
            "nul_docs_cleaned": stats.nul_docs,
            "documents_per_source": dict(stats.per_source),
        },
        "db_counts": {
            "notes": n_notes,
            "chunks": n_chunks,
            "chunks_without_embedding": n_null,
            "note_index_state": n_state,
            "chunks_without_a_note": orphan_chunks,
            "notes_without_index_state": unindexed_notes,
        },
        "embedder": {
            "model": EMBED_MODEL,
            "requests": embedder.requests,
            "prompt_tokens": embedder.prompt_tokens,
            "spent_usd_this_run": round(embedder.spent_usd, 6),
            "providers": dict(embedder.providers),
        },
        "spend_meter_erb_exam_usd": meter.corpus_usd(SPEND_CORPUS),
    }
    print(json.dumps(summary, indent=2))
    write_json(Path(args.out_dir) / "load.json", summary)
    conn.close()


def _shard_of(dsid: str, n: int) -> int:
    return int(hashlib.sha1(dsid.encode("utf-8")).hexdigest(), 16) % n


def _original_text_reader(sources: Path, uuid_index: dict[str, str]):
    cache: dict[str, str | None] = {}

    def read(dsid: str) -> str | None:
        if dsid in cache:
            return cache[dsid]
        rel = uuid_index.get(dsid)
        text: str | None = None
        if rel:
            path = sources / rel
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                title, content = extract_document_content(data)
                text = strip_nul(f"{title}\n\n{content}")
            except (OSError, ValueError):
                text = None
        if len(cache) > 5000:
            cache.clear()
        cache[dsid] = text
        return text

    return read


def cmd_run(args: argparse.Namespace) -> None:
    root, sources, _q = _paths(args)
    _require_clone(root)
    uuid_index, sel = _selection(args)
    if uuid_index is None:
        sys.exit("uuid_index.json is required for the provenance check")
    print(sel.describe(ERB_SLICES), flush=True)
    questions = sel.questions[: args.limit] if args.limit else sel.questions
    passes = tuple(args.passes.split(",")) if args.passes else PASS_NAMES

    meter = _meter(args)
    n_query_embeds = len(questions) * (2 if "portolan_vector_bge_instruction" in passes else 1)
    est_tokens = (
        sum(len(q["question"]) for q in questions) // 3 * (n_query_embeds // len(questions))
    )
    est_usd = 0.0 if args.local_embedder else est_tokens * EMBED_PRICE_PER_MILLION / 1e6
    already = meter.corpus_usd(SPEND_CORPUS)
    how = "local fastembed, no paid call" if args.local_embedder else "chars/3"
    print(
        f"ESTIMATE (query embeddings): ~{est_tokens:,} tokens ({how}) = ${est_usd:.6f}; "
        f"already under {SPEND_CORPUS!r}: ${already:.4f}; cap ${args.budget_usd:.2f}",
        flush=True,
    )
    if already + est_usd > args.budget_usd:
        sys.exit("STOP: estimate exceeds the cap")

    bm25 = None
    if args.exact_vectors or {"bm25", "rrf_bm25_vector"} & set(passes):
        import pickle

        # Local cache only (var/ is gitignored); its vocabulary is ERB-derived.
        cache = Path(args.out_dir) / "bm25.pkl"
        t0 = time.monotonic()
        if cache.is_file():
            with cache.open("rb") as fh:
                bm25 = pickle.load(fh)
            _log(f"bm25: loaded {cache}")
        else:
            _log("bm25: building over the same documents...")
            bm25 = Bm25Index()
            stats = IterStats()
            for n, doc in enumerate(
                iter_documents(sources, ERB_SLICES, uuid_index=uuid_index, stats=stats), 1
            ):
                bm25.add(doc.dsid, doc.index_text)
                if n % 25000 == 0:
                    _log(f"  bm25: {n} documents")
            bm25.finalize()
            with cache.open("wb") as fh:
                pickle.dump(bm25, fh, protocol=5)
        _log(
            f"bm25: {len(bm25.doc_ids)} docs, {len(bm25.vocab):,} terms, "
            f"{bm25.postings:,} postings in {time.monotonic() - t0:.0f}s"
        )

    url, conn = _connect(args)
    _log(f"connected to {redact_dsn(url)}")
    needs_embeddings = args.exact_vectors or bool(set(passes) - {"portolan_fts", "bm25"})
    # Accounting only: the local model never calls out, so nothing is spent.
    embedder = OpenRouterEmbedder("unused", meter=meter, budget_usd=0.0)
    if needs_embeddings and not args.local_embedder:
        embedder = _embedder(args, meter)
    query_model = FastEmbedEmbedder() if args.local_embedder else embedder
    cached = CachingEmbedder(query_model)
    instruction = CachingEmbedder(query_model, prefix=BGE_QUERY_INSTRUCTION)
    results: dict = {}
    try:
        if args.exact_vectors:
            # One streaming scan for every query (see exact_vector_scan);
            # the vector leg is then exactly what pgvector returns without
            # an HNSW index, and the fusion is Portolan's own. The scan
            # depends only on the questions and the index, so it is cached.
            keys = [x["question_id"] + s for x in questions for s in ("", "#instruction")]
            scan_path = Path(args.scan_cache or Path(args.out_dir) / "exact_scan.json")
            scan = load_scan(scan_path) if scan_path.is_file() else {}
            if all(k in scan for k in keys):
                _log(f"exact scan: reusing {scan_path}")
            else:
                vecs: dict[str, list[float]] = {}
                for x in questions:
                    vecs[x["question_id"]] = cached.embed([x["question"]])[0]
                    vecs[x["question_id"] + "#instruction"] = instruction.embed([x["question"]])[0]
                _log(f"exact scan: {len(vecs)} query vectors, one pass over chunks...")
                scan = exact_vector_scan(conn, vecs, 3 * VECTOR_ONLY_K, progress=_log)
                save_scan(scan_path, scan)
                _log(f"exact scan: saved {scan_path}")
            plain = {x["question_id"]: scan[x["question_id"]] for x in questions}
            instructed = {
                x["question_id"]: scan[x["question_id"] + "#instruction"] for x in questions
            }
            results = run_exact_passes(
                conn,
                questions,
                plain=plain,
                instructed=instructed,
                bm25=bm25,
                original_text=_original_text_reader(sources, uuid_index),
                progress=_log,
            )
        else:
            results = run_passes(
                conn,
                cached,
                questions,
                bm25=bm25,
                original_text=_original_text_reader(sources, uuid_index),
                instruction_embedder=instruction,
                passes=passes,
                progress=_log,
            )
    except BudgetExceeded as exc:
        sys.exit(f"STOP: {exc}")
    finally:
        embedder.close()

    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM note_index_state")
        n_notes = cur.fetchone()[0]
        cur.execute("SELECT reltuples::bigint FROM pg_class WHERE relname = 'chunks'")
        n_chunks = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM pg_indexes WHERE indexname = 'idx_chunks_embedding_hnsw'")
        has_hnsw = bool(cur.fetchone()[0])
    results["meta"] = {
        "run_at": datetime.now(UTC).isoformat(),
        "questions": len(questions),
        "question_selection": {
            "total": sel.total,
            "without_gold": sel.without_gold,
            "rule_a_source_types": sel.by_source_types,
            "rule_b_gold_paths": sel.by_gold_paths,
            "rules_agree": sel.derivations_agree,
            "by_type": sel.by_type,
            "by_source_types": sel.by_source_combo,
            "distinct_gold_docs": sel.distinct_gold_docs,
        },
        "index": {
            "notes_indexed": n_notes,
            "chunks_estimate_pg_class": n_chunks,
            "hnsw_index_present": has_hnsw,
        },
        "vector_leg": "exact streaming scan" if args.exact_vectors else "pgvector query",
        "embed_model": EMBED_MODEL,
        "query_embedder": (
            f"local fastembed {FastEmbedEmbedder.MODEL_NAME}"
            if args.local_embedder
            else f"OpenRouter {EMBED_MODEL}"
        ),
        "query_embedding_requests": embedder.requests,
        "query_embedding_usd": round(embedder.spent_usd, 6),
        "spend_meter_erb_exam_usd": meter.corpus_usd(SPEND_CORPUS),
        "top_k": 10,
        "erb_metric_source": "src/scripts/answer_evaluation/metrics_based_eval.py "
        "(score_answer, compute_stats_for_group), --no-correction",
    }
    out = Path(args.out_dir) / ("results_exact.json" if args.exact_vectors else "results.json")
    write_json(out, results)
    _log(f"wrote {out}")
    for name, s in results["summary"].items():
        o = s["overall"]
        print(
            f"{name:36s} recall@10 {o['average_recall_pct']:6.2f}  "
            f"invalid_extra {o['average_invalid_extra_docs']:5.2f}  "
            f"hit@1 {o['ours_hit_at_1_pct']:6.2f}  hit@10 {o['ours_hit_at_10_pct']:6.2f}  "
            f"mrr {o['ours_mrr_at_10']:.4f}  n={o['count']}  {s['mean_seconds']}s/q"
        )
    print(json.dumps(results["fused_diagnostics"], indent=2))
    print(json.dumps(results["citation_gate"], indent=2))


def cmd_tune(args: argparse.Namespace) -> None:
    """Choose the fusion weights on the dev half only, then report every
    pass on both halves (and all 157). Reads the legs `run --exact-vectors`
    recorded (ids and numbers only); no database, no network."""
    _uuid_index, sel = _selection(args)
    results_path = Path(args.results or Path(args.out_dir) / "results_exact.json")
    results = json.loads(results_path.read_text(encoding="utf-8"))
    legs = results["legs"]
    questions = [x for x in sel.questions if x["question_id"] in legs]
    dev_ids, held_ids = split_dev_heldout(x["question_id"] for x in questions)
    halves = {"dev": set(dev_ids), "held_out": set(held_ids)}

    def per_half(rows: list[dict]) -> dict:
        out = {
            h: aggregate([r for r in rows if r["question_id"] in ids]) for h, ids in halves.items()
        }
        out["all"] = aggregate(rows)
        return out

    tuned = tune_fusion_weights([x for x in questions if x["question_id"] in halves["dev"]], legs)
    w = tuned["chosen"]["vector_weight"]
    passes: dict[str, dict] = {}
    before = Path(args.before)
    if before.is_file():
        old = json.loads(before.read_text(encoding="utf-8"))["rows"]
        passes["before: fused, AND keyword leg, 1.0/1.2"] = per_half(old["fused_exact"])
        passes["before: AND keyword leg alone"] = per_half(old["portolan_fts"])
    now = results["rows"]
    ran = results.get("fusion_weights", {})
    passes[f"fused as run, {ran.get('fts')}/{ran.get('vector')}"] = per_half(now["fused_exact"])
    for label, vw in (("old weights 1.0/1.2", 1.2), (f"chosen 1.0/{w}", w)):
        passes[f"fused, BM25 keyword leg, {label}"] = per_half(
            fusion_rows(questions, legs, fts_weight=1.0, vector_weight=vw)
        )
    passes["BM25 keyword leg alone"] = per_half(now["portolan_fts"])
    passes["vector leg alone"] = per_half(now["portolan_vector_exact"])
    if "bm25" in now:
        passes["reference BM25 (in memory) alone"] = per_half(now["bm25"])
        passes[f"reference: rrf(in-memory BM25, vector), 1.0/{w}"] = per_half(
            fusion_rows(questions, legs, fts_weight=1.0, vector_weight=w, keyword_leg="bm25")
        )
    out = {
        "results_file": str(results_path),
        "split": {"seed": DEV_SPLIT_SEED, "dev": dev_ids, "held_out": held_ids},
        "selection_rule": "dev recall@10, then dev MRR@10, then weight closest to 1:1",
        "grid_dev_only": tuned["grid"],
        "chosen": tuned["chosen"],
        "passes": passes,
    }
    write_json(Path(args.out_dir) / "tune.json", out)
    print(f"split: dev {len(dev_ids)}, held-out {len(held_ids)} (seed {DEV_SPLIT_SEED!r})")
    print("grid on dev only (keyword weight 1.0):")
    for r in tuned["grid"]:
        d = r["dev"]
        print(
            f"  vector {r['vector_weight']:<5} recall@10 {d['average_recall_pct']:6.2f}  "
            f"mrr {d['ours_mrr_at_10']:.4f}"
        )
    print(f"chosen: vector weight {w}")
    for name, p in passes.items():
        cells = "  ".join(
            f"{h} {p[h]['average_recall_pct']:6.2f}/{p[h]['ours_mrr_at_10']:.3f}"
            for h in ("dev", "held_out", "all")
        )
        print(f"  {name:52s} {cells}")


def cmd_ef(args: argparse.Namespace) -> None:
    """`search()` through the HNSW index at each `hnsw.ef_search` value,
    every question, local query embeddings ($0). Chooses a value on the dev
    half (`choose_ef_search`) and reports dev, held-out and all, with the
    exact vector step from `results_exact.json` beside them. Writes
    `ef_sweep.json` (ids and numbers only)."""
    _uuid_index, sel = _selection(args)
    questions = sel.questions[: args.limit] if args.limit else sel.questions
    dev_ids, held_ids = split_dev_heldout(x["question_id"] for x in sel.questions)
    halves = {"dev": set(dev_ids), "held_out": set(held_ids)}
    ef_values = [int(v) for v in args.ef_values.split(",")]

    url, conn = _connect(args)
    _log(f"connected to {redact_dsn(url)}; ef_search {ef_values}, {len(questions)} questions")
    embedder = CachingEmbedder(FastEmbedEmbedder())
    for x in questions:  # embed up front so no timed search pays for the model
        embedder.embed([x["question"]])
    rows = run_ef_sweep(conn, embedder, questions, ef_values=ef_values, progress=_log)

    def per_half(pass_rows: list[dict]) -> dict:
        out = {}
        for half, ids in (*halves.items(), ("all", halves["dev"] | halves["held_out"])):
            subset = [r for r in pass_rows if r["question_id"] in ids]
            out[half] = {**aggregate(subset), **latency_summary(subset)}
        return out

    table = {f"ef{ef}": per_half(rows[str(ef)]) for ef in ef_values}
    exact_path = Path(args.exact_results or Path(args.out_dir) / "results_exact.json")
    if exact_path.is_file():
        exact_rows = json.loads(exact_path.read_text(encoding="utf-8"))["rows"]["fused_exact"]
        wanted = {x["question_id"] for x in questions}
        table["exact"] = per_half([r for r in exact_rows if r["question_id"] in wanted])
    chosen = choose_ef_search(
        {ef: table[f"ef{ef}"]["dev"]["average_recall_pct"] for ef in ef_values},
        tolerance=EF_SEARCH_RECALL_TOLERANCE,
    )
    out = {
        "run_at": datetime.now(UTC).isoformat(),
        "split": {"seed": DEV_SPLIT_SEED, "dev": dev_ids, "held_out": held_ids},
        "selection_rule": (
            f"smallest ef_search whose dev recall@10 is within {EF_SEARCH_RECALL_TOLERANCE} "
            "points of the grid's best dev recall (dev numbers only)"
        ),
        "chosen_ef_search": chosen,
        "exact_results_file": str(exact_path) if exact_path.is_file() else None,
        "query_embedder": f"local fastembed {FastEmbedEmbedder.MODEL_NAME}",
        "table": table,
        "rows": rows,
    }
    write_json(Path(args.out_dir) / "ef_sweep.json", out)
    for name, p in table.items():
        cells = "  ".join(
            f"{h} {p[h]['average_recall_pct']:6.2f}/{p[h]['ours_mrr_at_10']:.3f} "
            f"p50 {p[h]['p50_ms']:.0f} p95 {p[h]['p95_ms']:.0f}ms"
            for h in ("dev", "held_out", "all")
        )
        print(f"  {name:6s} {cells}")
    print(f"chosen on dev: ef_search {chosen}")
    conn.close()


def cmd_weights(args: argparse.Namespace) -> None:
    """`search()` through the HNSW index (the configured ef_search) at
    keyword weight 1.0 and each vector weight, every question, local query
    embeddings ($0). With `--demo-results` (the JSON files
    `scripts/bench/demo_search.py --weights` wrote for the paraphrase and
    demo sets), chooses a weight with `choose_vector_weight`: ERB dev and
    paraphrase recall, demo recall as a constraint. Writes
    `weight_sweep.json` (ids and numbers only)."""
    from portolan import config

    _uuid_index, sel = _selection(args)
    questions = sel.questions[: args.limit] if args.limit else sel.questions
    dev_ids, held_ids = split_dev_heldout(x["question_id"] for x in sel.questions)
    halves = {"dev": set(dev_ids), "held_out": set(held_ids)}
    weights = [float(w) for w in args.vector_weights.split(",")]

    ef = config.settings.search_hnsw_ef_search
    previous = Path(args.out_dir) / "weight_sweep.json"
    rows = {}
    if args.reuse and previous.is_file():
        old = json.loads(previous.read_text(encoding="utf-8"))
        if old["ef_search"] == ef and all(str(w) in old["rows"] for w in weights):
            rows = {str(w): old["rows"][str(w)] for w in weights}
            _log(f"reusing the search rows in {previous}")
    if not rows:
        url, conn = _connect(args)
        _log(f"connected to {redact_dsn(url)}; vector weights {weights}, ef_search {ef}")
        embedder = CachingEmbedder(FastEmbedEmbedder())
        for x in questions:
            embedder.embed([x["question"]])
        rows = run_weight_sweep(conn, embedder, questions, vector_weights=weights, progress=_log)
        conn.close()

    table: dict[str, dict] = {}
    for w in weights:
        pass_rows = rows[str(w)]
        table[str(w)] = {
            half: aggregate([r for r in pass_rows if r["question_id"] in ids])
            for half, ids in (*halves.items(), ("all", halves["dev"] | halves["held_out"]))
        }
    demo: dict[str, dict] = {}
    for path in args.demo_results:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        name = "paraphrase" if "paraphrase" in Path(path).stem else "demo"
        demo[name] = {
            label: run["overall"]["average_recall_pct"] for label, run in data["runs"].items()
        }
    chosen = None
    if {"paraphrase", "demo"} <= set(demo):

        def by_weight(recalls: dict[str, float]) -> dict[float, float]:
            return {float(label.split("/")[1]): v for label, v in recalls.items()}

        chosen = choose_vector_weight(
            {w: table[str(w)]["dev"]["average_recall_pct"] for w in weights},
            by_weight(demo["paraphrase"]),
            by_weight(demo["demo"]),
        )
    out = {
        "run_at": datetime.now(UTC).isoformat(),
        "ef_search": ef,
        "split": {"seed": DEV_SPLIT_SEED, "dev": dev_ids, "held_out": held_ids},
        "selection_rule": (
            "among weights keeping demo recall at 100, max of min(ERB dev recall, paraphrase "
            "recall); ties to higher paraphrase recall, then higher ERB dev, then smaller weight"
        ),
        "chosen_vector_weight": chosen,
        "erb": table,
        "demo_search": demo,
        "rows": rows,
    }
    write_json(Path(args.out_dir) / "weight_sweep.json", out)
    for w in weights:
        t = table[str(w)]
        para = demo.get("paraphrase", {}).get(f"1.0/{w}")
        dm = demo.get("demo", {}).get(f"1.0/{w}")
        print(
            f"  1.0/{w:<4} ERB dev {t['dev']['average_recall_pct']:6.2f}  held-out "
            f"{t['held_out']['average_recall_pct']:6.2f}  all {t['all']['average_recall_pct']:6.2f}"
            f"  paraphrase {para}  demo {dm}"
        )
    print(f"chosen: vector weight {chosen}")


def _answers_estimate(docs_chars: list[int], facts: list[int], gen: str, judge: str) -> float:
    """Upper-bound dollar estimate for the answer sample (chars/3 tokens,
    full max_tokens output, both modes), from `portolan.pricing`."""
    from portolan.pricing import PRICING

    g, j = PRICING[gen], PRICING[judge]
    total = 0.0
    for chars, n_facts in zip(docs_chars, facts, strict=True):
        gen_in = (chars + 2500) / 3
        total += 2 * (gen_in * g.input_per_million + 1500 * g.output_per_million) / 1e6
        judge_in = (1 + n_facts) * (6000 / 3 + 800)  # answer (<=1500 tok) + template
        total += (
            2 * (judge_in * j.input_per_million + (1 + n_facts) * 300 * j.output_per_million) / 1e6
        )
    return total


def cmd_answers(args: argparse.Namespace) -> None:
    """Checklist item 28, first half: ERB's answer evaluation (their judge
    prompts, verbatim) on a fixed 20-question sample, answers generated
    from the fused top 10, once with ERB's own prompt and once with
    Portolan's citation contract and gate. No three-judge gold correction."""
    from concurrent.futures import ThreadPoolExecutor

    from portolan.bench.erb_answers import (
        OpenRouterChat,
        format_context_documents,
        gate_cited_answer,
        generate_answer,
        parse_cited_answer,
        sample_questions,
        score_answer_text,
        strip_dsid_references,
        summarize,
    )

    root, sources, _q = _paths(args)
    _require_clone(root)
    uuid_index, sel = _selection(args)
    # The fused top 10 from `run` (pgvector) if present, else `run --exact-vectors`.
    fused_rows = None
    for fname, key in (("results.json", "fused"), ("results_exact.json", "fused_exact")):
        path = Path(args.out_dir) / fname
        if path.is_file():
            rows_by_pass = json.loads(path.read_text(encoding="utf-8"))["rows"]
            if key in rows_by_pass:
                fused_rows, source_pass = rows_by_pass[key], f"{fname}:{key}"
                break
    if fused_rows is None:
        sys.exit("no fused results yet: run `run` or `run --exact-vectors` first")
    _log(f"answers: context from {source_pass}")
    fused = {r["question_id"]: r["retrieved_doc_ids"] for r in fused_rows}
    sample = sample_questions(sel.questions, args.sample)

    def _doc(dsid: str) -> tuple[str, str, str]:
        data = json.loads((sources / uuid_index[dsid]).read_text(encoding="utf-8"))
        title, content = extract_document_content(data)
        return dsid, strip_nul(title), strip_nul(content)

    contexts = {q["question_id"]: [_doc(d) for d in fused[q["question_id"]]] for q in sample}
    est = _answers_estimate(
        [sum(len(t) + len(c) for _d, t, c in contexts[q["question_id"]]) for q in sample],
        [len(q.get("answer_facts") or []) for q in sample],
        args.gen_model,
        args.judge_model,
    )
    meter = _meter(args)
    already = meter.corpus_usd(SPEND_CORPUS)
    print(
        f"ESTIMATE (answers, upper bound): ${est:.4f} for {len(sample)} questions x 2 modes; "
        f"generator {args.gen_model}, judge {args.judge_model}; already under "
        f"{SPEND_CORPUS!r}: ${already:.4f}; cap ${args.budget_usd:.2f}",
        flush=True,
    )
    if already + est > args.budget_usd:
        sys.exit("STOP: estimate exceeds the cap")

    key = read_openrouter_key(Path(args.env_file) if args.env_file else None)
    chat = OpenRouterChat(key, meter=meter, corpus=SPEND_CORPUS, budget_usd=args.budget_usd)

    def _one(q: dict) -> tuple[dict, dict]:
        qid = q["question_id"]
        docs = contexts[qid]
        context = format_context_documents(docs)
        doc_texts = {d: f"{t}\n\n{c}" for d, t, c in docs}
        plain = generate_answer(chat, args.gen_model, q["question"], context, cited=False)
        plain_judged = strip_dsid_references(plain)
        cited_raw = generate_answer(chat, args.gen_model, q["question"], context, cited=True)
        sentences = parse_cited_answer(cited_raw)
        gate = gate_cited_answer(sentences, doc_texts)
        gated_text = strip_dsid_references(gate.text)
        row = {
            "question_id": qid,
            "question_type": q["question_type"],
            "gold_in_context": bool(set(q["expected_doc_ids"]) & set(fused[qid])),
            "erb_plain": score_answer_text(chat, args.judge_model, q, plain_judged),
            "portolan_cited": score_answer_text(chat, args.judge_model, q, gated_text),
            "gate": {
                "sentences": len(sentences),
                "kept": len(gate.kept),
                "dropped": len(gate.dropped),
                "quotes": gate.quotes_total,
                "quotes_supported": gate.quotes_supported,
            },
        }
        texts = {"question_id": qid, "erb_plain": plain, "portolan_cited_raw": cited_raw}
        return row, texts

    rows: list[dict] = []
    texts: list[dict] = []
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for row, text in pool.map(_one, sample):
                rows.append(row)
                texts.append(text)
                _log(
                    f"  {row['question_id']}: plain correct={row['erb_plain']['answer_correct']} "
                    f"cited correct={row['portolan_cited']['answer_correct']} "
                    f"gate kept {row['gate']['kept']}/{row['gate']['sentences']}"
                )
    except BudgetExceeded as exc:
        sys.exit(f"STOP: {exc}")
    finally:
        chat.close()

    def _gate_totals(key: str) -> int:
        return sum(r["gate"][key] for r in rows)

    out = {
        "run_at": datetime.now(UTC).isoformat(),
        "sample": [q["question_id"] for q in sample],
        "sample_seed": 20260922,
        "generator": args.gen_model,
        "judge": args.judge_model,
        "context": f"fused top 10 documents ({source_pass}), full text, ERB format_context_documents",
        "erb_plain": summarize([r["erb_plain"] for r in rows]),
        "portolan_cited": summarize([r["portolan_cited"] for r in rows]),
        "gate": {
            "sentences": _gate_totals("sentences"),
            "kept": _gate_totals("kept"),
            "dropped": _gate_totals("dropped"),
            "quotes": _gate_totals("quotes"),
            "quotes_supported": _gate_totals("quotes_supported"),
        },
        "gold_in_context": sum(1 for r in rows if r["gold_in_context"]),
        "rows": sorted(rows, key=lambda r: r["question_id"]),
        "chat_calls": chat.calls,
        "chat_usd": round(chat.spent_usd, 6),
        "chat_usd_per_model": {k: round(v, 6) for k, v in chat.per_model_usd.items()},
        "spend_meter_erb_exam_usd": meter.corpus_usd(SPEND_CORPUS),
    }
    write_json(Path(args.out_dir) / "answers.json", out)
    # Model text derived from ERB documents stays local (var/ is gitignored).
    with (Path(args.out_dir) / "answers_text.jsonl").open("w", encoding="utf-8") as fh:
        for t in sorted(texts, key=lambda t: t["question_id"]):
            fh.write(json.dumps(t) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=2))


def _context_rows(args: argparse.Namespace) -> tuple[list[dict], str]:
    """The fused rows answers are built from: `--context FILE:PASS`, else
    the first of results.json:fused, results_exact.json:fused_exact."""
    candidates = [("results.json", "fused"), ("results_exact.json", "fused_exact")]
    if args.context:
        fname, _, key = args.context.partition(":")
        candidates = [(fname, key)]
    for fname, key in candidates:
        path = Path(args.out_dir) / fname
        if path.is_file():
            rows_by_pass = json.loads(path.read_text(encoding="utf-8"))["rows"]
            if key in rows_by_pass:
                return rows_by_pass[key], f"{fname}:{key}"
    sys.exit(f"no fused rows for {candidates}: run `run` or `run --exact-vectors` first")


def cmd_repair(args: argparse.Namespace) -> None:
    """The cited-answer repair measurement (2026-09-23): the item-28 sample,
    same generator, judge and context, every mode in `ANSWER_MODES` in one
    run. Generation first, then gate, repair, support judge and ERB's
    judges; an upper-bound estimate is printed before each phase, and the
    client refuses any call once this run has spent `--run-cap-usd`."""
    from concurrent.futures import ThreadPoolExecutor

    from portolan.bench.erb_answers import (
        ANSWER_MODES,
        REPAIR_PROMPT,
        REPAIR_PROMPT_V2,
        OpenRouterChat,
        estimate_evaluation_usd,
        estimate_generation_usd,
        evaluate_question,
        evaluation_item,
        format_context_documents,
        generate_answers,
        sample_questions,
        strip_dsid_references,
        summarize_modes,
    )
    from portolan.report.support_judge import (
        ANSWER_SUPPORT_SYSTEM,
        ANSWER_SUPPORT_SYSTEM_IN_CONTEXT,
    )

    root, sources, _q = _paths(args)
    _require_clone(root)
    uuid_index, sel = _selection(args)
    fused_rows, source_pass = _context_rows(args)
    _log(f"repair: context from {source_pass}")
    fused = {r["question_id"]: r["retrieved_doc_ids"] for r in fused_rows}
    sample = sample_questions(sel.questions, args.sample)

    def _doc(dsid: str) -> tuple[str, str, str]:
        data = json.loads((sources / uuid_index[dsid]).read_text(encoding="utf-8"))
        title, content = extract_document_content(data)
        return dsid, strip_nul(title), strip_nul(content)

    docs = {q["question_id"]: [_doc(d) for d in fused[q["question_id"]]] for q in sample}
    contexts = {qid: format_context_documents(d) for qid, d in docs.items()}

    recorded_raw: dict[str, str] = {}
    recorded_scores: dict[str, dict] = {}
    if args.replay:
        for line in Path(args.replay).read_text(encoding="utf-8").splitlines():
            t = json.loads(line)
            recorded_raw[t["question_id"]] = t["portolan_cited_raw"]
    if args.replay_scores:
        for r in json.loads(Path(args.replay_scores).read_text(encoding="utf-8"))["rows"]:
            recorded_scores[r["question_id"]] = r["portolan_cited"]

    # --from-run: an earlier run's generations (and its scores, keyed by the
    # judged text) so a repair or support variant is measured on the same
    # answers without paying to generate or re-judge them.
    earlier_gens: dict[str, dict] = {}
    seed_scores: dict[str, dict] = {}
    if args.from_run:
        run_dir = Path(args.out_dir)
        earlier_rows = {
            r["question_id"]: r
            for r in json.loads((run_dir / f"{args.from_run}.json").read_text(encoding="utf-8"))[
                "rows"
            ]
        }
        text_path = run_dir / f"{args.from_run}_text.jsonl"
        for line in text_path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            qid = rec["question_id"]
            earlier_gens[qid] = {
                k: {"text": rec["raw"][k], "s": 0.0, "usd": 0.0} for k in ("plain", "v1", "v2")
            }
            if "recorded" in rec["raw"]:
                recorded_raw[qid] = rec["raw"]["recorded"]
            scores = seed_scores.setdefault(qid, {})
            row = earlier_rows[qid]
            scores[strip_dsid_references(rec["raw"]["plain"])] = row["erb_plain"]
            for mode, mode_rec in rec["modes"].items():
                s = row[mode]
                scores[strip_dsid_references(mode_rec["final_text"])] = {
                    "answer_correct": s["answer_correct"],
                    "completeness_pct": s["completeness_pct"],
                    "facts_valid": s["facts_valid"],
                }

    meter = _meter(args)
    already = meter.corpus_usd(SPEND_CORPUS)
    gen_est = (
        0.0
        if args.from_run
        else estimate_generation_usd([len(c) for c in contexts.values()], args.gen_model)
    )
    print(
        f"ESTIMATE (phase 1, generation, upper bound): ${gen_est:.4f} for {len(sample)} "
        f"questions x 3 prompts, {args.gen_model}"
        f"{f' (reused from {args.from_run})' if args.from_run else ''}; already under "
        f"{SPEND_CORPUS!r}: ${already:.4f}; this run's cap ${args.run_cap_usd:.2f}",
        flush=True,
    )
    if gen_est > args.run_cap_usd:
        sys.exit("STOP: estimate exceeds the run cap")

    def _items(gens_by_q: dict[str, dict]) -> list[dict]:
        return [
            evaluation_item(
                question=q,
                context_chars=len(contexts[q["question_id"]]),
                doc_texts={d: f"{t}\n\n{c}" for d, t, c in docs[q["question_id"]]},
                gens=gens_by_q[q["question_id"]],
                recorded_raw=recorded_raw.get(q["question_id"]),
                already_judged=("plain", "v1") if args.from_run else (),
                support_context_chars=args.support_context_chars,
            )
            for q in sample
        ]

    if args.estimate_only:
        # Phase 2 depends on the generated answers; without --from-run,
        # preview it with each question's replayed answer standing in.
        stand_in = earlier_gens or {
            q["question_id"]: {
                k: {"text": recorded_raw.get(q["question_id"], "")} for k in ("plain", "v1", "v2")
            }
            for q in sample
        }
        preview_est = estimate_evaluation_usd(_items(stand_in), args.gen_model, args.judge_model)
        print(f"ESTIMATE (phase 2, upper bound): ${preview_est:.4f}")
        return

    env_file = Path(args.env_root) / ".env" if args.env_root else Path(args.env_file)
    key = read_openrouter_key(env_file)
    chat = OpenRouterChat(
        key,
        meter=meter,
        corpus=SPEND_CORPUS,
        budget_usd=already + args.run_cap_usd,
        run_cap_usd=args.run_cap_usd,
    )
    repair_prompt = REPAIR_PROMPT_V2 if args.repair_prompt == "v2" else REPAIR_PROMPT
    support_system = (
        ANSWER_SUPPORT_SYSTEM_IN_CONTEXT if args.support_context_chars else ANSWER_SUPPORT_SYSTEM
    )
    try:
        if earlier_gens:
            gens = {q["question_id"]: earlier_gens[q["question_id"]] for q in sample}
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                gens = dict(
                    zip(
                        [q["question_id"] for q in sample],
                        pool.map(
                            lambda q: generate_answers(
                                chat, args.gen_model, q["question"], contexts[q["question_id"]]
                            ),
                            sample,
                        ),
                        strict=True,
                    )
                )
            _log(f"repair: generated {3 * len(gens)} answers, spent ${chat.spent_usd:.4f}")

        eval_est = estimate_evaluation_usd(_items(gens), args.gen_model, args.judge_model)
        print(
            f"ESTIMATE (phase 2, gate/repair/support/judges, upper bound): ${eval_est:.4f}; "
            f"this run so far ${chat.spent_usd:.4f} of ${args.run_cap_usd:.2f}",
            flush=True,
        )
        if chat.spent_usd + eval_est > args.run_cap_usd:
            sys.exit("STOP: estimate exceeds what is left of the run cap")

        def _one(q: dict) -> tuple[dict, dict]:
            qid = q["question_id"]
            row, record = evaluate_question(
                chat,
                gen_model=args.gen_model,
                judge_model=args.judge_model,
                question=q,
                docs=docs[qid],
                gens=gens[qid],
                recorded_raw=recorded_raw.get(qid),
                recorded_cited_score=recorded_scores.get(qid),
                seed_scores=seed_scores.get(qid),
                repair_prompt=repair_prompt,
                support_system=support_system,
                support_context_chars=args.support_context_chars,
            )
            row["gold_in_context"] = bool(set(q["expected_doc_ids"]) & set(fused[qid]))
            return row, record

        rows: list[dict] = []
        records: list[dict] = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for row, record in pool.map(_one, sample):
                rows.append(row)
                records.append(record)
                _log(
                    f"  {row['question_id']}: "
                    + " ".join(
                        f"{m}={'Y' if row[m]['answer_correct'] else 'n'}"
                        for m in ANSWER_MODES
                        if m in row
                    )
                )
    except BudgetExceeded as exc:
        sys.exit(f"STOP: {exc}")
    finally:
        chat.close()

    rows.sort(key=lambda r: r["question_id"])
    out = {
        "run_at": datetime.now(UTC).isoformat(),
        "sample": [q["question_id"] for q in sample],
        "sample_seed": 20260922,
        "generator": args.gen_model,
        "judge": args.judge_model,
        "support_judge": args.judge_model,
        "context": f"fused top 10 documents ({source_pass}), full text, ERB format_context_documents",
        "replay": bool(recorded_raw),
        "generations_from": args.from_run or "this run",
        "repair_prompt": args.repair_prompt,
        "support_variant": (
            f"in context, {args.support_context_chars} chars each side"
            if args.support_context_chars
            else "quotes and titles only"
        ),
        "modes": summarize_modes(rows, ANSWER_MODES),
        "gold_in_context": sum(1 for r in rows if r["gold_in_context"]),
        "rows": rows,
        "estimate_usd": {"generation": round(gen_est, 4), "evaluation": round(eval_est, 4)},
        "chat_calls": chat.calls,
        "chat_usd": round(chat.spent_usd, 6),
        "chat_usd_per_model": {k: round(v, 6) for k, v in chat.per_model_usd.items()},
        "support_diag_usd": round(sum(r["support_diag_usd"] for r in rows), 6),
        "spend_meter_erb_exam_usd": meter.corpus_usd(SPEND_CORPUS),
    }
    write_json(Path(args.out_dir) / f"{args.tag}.json", out)
    # Model text derived from ERB documents stays local (var/ is gitignored).
    with (Path(args.out_dir) / f"{args.tag}_text.jsonl").open("w", encoding="utf-8") as fh:
        for rec in sorted(records, key=lambda t: t["question_id"]):
            fh.write(json.dumps(rec) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=2))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="erb_exam")
    parser.add_argument("--erb-root", default=str(default_erb_root()))
    parser.add_argument("--out-dir", default="var/bench/erb_exam")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("select")
    sub.add_parser("plan")

    def _common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--pg-host", default="localhost")
        p.add_argument("--pg-port", type=int, default=5432)
        p.add_argument("--env-file", default=".env", help="read OPENROUTER_API_KEY from here")
        p.add_argument("--spend-file", default=None, help="default: the shared meter")
        p.add_argument("--budget-usd", type=float, default=5.0)
        p.add_argument("--request-size", type=int, default=64)

    load = sub.add_parser("load")
    _common(load)
    load.add_argument("--fresh", action="store_true", help="drop and recreate portolan_erb")
    load.add_argument("--dry-run", action="store_true")
    load.add_argument(
        "--reuse-plan", action="store_true", help="use <out-dir>/plan.json instead of re-planning"
    )
    load.add_argument("--batch-docs", type=int, default=300)
    load.add_argument("--workers", type=int, default=8)
    load.add_argument("--maintenance-work-mem", default="3GB")
    load.add_argument("--index-workers", type=int, default=3)
    load.add_argument(
        "--shard", default="0/1", help="K/N: load only shard K of N; N>1 skips the index build"
    )
    load.add_argument(
        "--indexes-only",
        action="store_true",
        help="after sharded loads: skip the document pass, just build indexes and count",
    )

    run = sub.add_parser("run")
    _common(run)
    run.add_argument("--limit", type=int, default=0)
    run.add_argument("--passes", default="", help=f"comma list from {','.join(PASS_NAMES)}")
    run.add_argument(
        "--exact-vectors",
        action="store_true",
        help="vector leg from one exact streaming scan (no HNSW needed); writes results_exact.json",
    )
    run.add_argument(
        "--local-embedder",
        action="store_true",
        help="embed the questions with local fastembed (same model, no key, $0)",
    )
    run.add_argument(
        "--scan-cache",
        default="",
        help="exact scan cache (default <out-dir>/exact_scan.json); reused when it covers "
        "every question",
    )

    tune = sub.add_parser("tune")
    tune.add_argument(
        "--results", default="", help="results_exact.json (default <out-dir>/results_exact.json)"
    )
    tune.add_argument(
        "--before",
        default="01-spec/bench-erb-search-2026-09-22.results.json",
        help="an earlier results file whose fused_exact rows are reported per half",
    )

    ef = sub.add_parser("ef")
    _common(ef)
    ef.add_argument("--limit", type=int, default=0)
    ef.add_argument(
        "--ef-values",
        default=",".join(str(v) for v in EF_SEARCH_GRID),
        help="comma list of hnsw.ef_search values",
    )
    ef.add_argument(
        "--exact-results",
        default="",
        help="results_exact.json whose fused_exact rows give the exact column "
        "(default <out-dir>/results_exact.json)",
    )

    weights = sub.add_parser("weights")
    _common(weights)
    weights.add_argument("--limit", type=int, default=0)
    weights.add_argument(
        "--vector-weights",
        default=",".join(str(w) for w in VECTOR_WEIGHT_GRID),
        help="comma list of vector weights (keyword weight 1.0)",
    )
    weights.add_argument(
        "--reuse",
        action="store_true",
        help="take the search rows from an earlier weight_sweep.json at the same ef_search",
    )
    weights.add_argument(
        "--demo-results",
        nargs="*",
        default=[],
        help="demo_search.py --weights outputs: <embedder>.json and <embedder>-paraphrase.json",
    )

    answers = sub.add_parser("answers")
    _common(answers)
    answers.add_argument("--sample", type=int, default=20)
    answers.add_argument("--gen-model", default="openai/gpt-5.6-luna")
    answers.add_argument("--judge-model", default="anthropic/claude-haiku-4.5")
    answers.add_argument("--workers", type=int, default=4)

    repair = sub.add_parser("repair")
    _common(repair)
    repair.add_argument("--sample", type=int, default=20)
    repair.add_argument("--gen-model", default="openai/gpt-5.6-luna")
    repair.add_argument("--judge-model", default="anthropic/claude-haiku-4.5")
    repair.add_argument("--workers", type=int, default=4)
    repair.add_argument(
        "--context",
        default="",
        help="FILE:PASS under --out-dir, e.g. results_exact.json:fused_exact",
    )
    repair.add_argument("--run-cap-usd", type=float, default=2.0)
    repair.add_argument(
        "--env-root",
        default="",
        help="a checkout whose env file holds the key (a worktree run names the main checkout)",
    )
    repair.add_argument(
        "--estimate-only", action="store_true", help="print the estimates, make no paid call"
    )
    repair.add_argument(
        "--tag", default="repair", help="output names: <tag>.json, <tag>_text.jsonl"
    )
    repair.add_argument(
        "--from-run",
        default="",
        help="tag of an earlier run under --out-dir whose generations and scores are reused",
    )
    repair.add_argument("--repair-prompt", choices=("v1", "v2"), default="v1")
    repair.add_argument(
        "--support-context-chars",
        type=int,
        default=0,
        help="show the support judge each quote inside this many chars of its passage",
    )
    repair.add_argument(
        "--replay",
        default="",
        help="an earlier answers_text.jsonl whose cited answers are replayed",
    )
    repair.add_argument(
        "--replay-scores",
        default="",
        help="that run's results file; its cited scores seed the judge cache for unchanged text",
    )

    args = parser.parse_args(argv)
    commands = {
        "select": cmd_select,
        "plan": cmd_plan,
        "load": cmd_load,
        "run": cmd_run,
        "tune": cmd_tune,
        "ef": cmd_ef,
        "weights": cmd_weights,
        "answers": cmd_answers,
        "repair": cmd_repair,
    }
    commands[args.cmd](args)


if __name__ == "__main__":
    try:
        main()
    except BudgetExceeded as exc:
        sys.exit(f"STOP: {exc}")
