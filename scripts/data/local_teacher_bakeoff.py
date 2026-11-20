"""Local-teacher bake-off (`01-spec/local-teachers-bakeoff-2026-09-22.md`):
can a small open-weight model on this Mac label silver data as well as Kimi
K2 does?

Every candidate runs the production path a silver record takes: the neutral
extraction prompt with the corpus's company context
(`scripts/data/silver_contexts.py`), `run_corpus_single` (the keep gate, the
negative cap off so every document gets a record), then the repair pass and
completeness audit (`scripts/data/repair_silver.py` `repair_corpus`), with
the repair call answered by the same local model. The reference is a Kimi K2
run of that same path on the same documents with the same prompt bytes
(checked by sha256).

    select  pick N documents (seeded, spread over the corpora) from the
            reference run and copy the reference's files for them
    label   run one local teacher over the selection, one process per model
    wp5     the WP5 completeness teacher's job on the v2 documents a person
            checked Kimi K2's labels on (`precision_review.jsonl`): which of
            the facts Kimi got right, and which it got wrong, does the local
            model also label?
    judge   the typed-judge calibration (`scripts/data/judge_calibration.py`'s
            80 items: gold, wrong type, swapped quote, added detail) with
            the local model answering the judge's three questions
    score   score every run against the reference; write scores.json/.md

Layout under `--out`: `selection.json`; `runs/<slug>/var/data/silver/<corpus>/`
(records, drops, timings, repaired records, repair log); `runs/<slug>/calls.jsonl`
(one line per extraction attempt: calls, retry, lenient parse, raw text,
parsed answer, seconds, peak memory); `runs/<slug>/run.json`.

Local calls cost nothing; each writes a $0 line naming the model under the
spend key `local_bakeoff` (`--spend-file`, the shared meter).

Usage:
    uv run python scripts/data/local_teacher_bakeoff.py select --reference-root <var> --out <dir>
    uv run --group mlx python scripts/data/local_teacher_bakeoff.py label \\
        --teacher local:mlx-community/Qwen3.5-4B-MLX-4bit --out <dir> --converted-root <dir>
    uv run python scripts/data/local_teacher_bakeoff.py score --out <dir>
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import re
import resource
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from corpus.generator.prose import TokenBucketLimiter
from portolan.extract.backends import LLMClientBackend
from portolan.extract.prompt import (
    DEFAULT_PROMPT_VARIANT,
    build_system_prompt,
    system_prompt_sha256,
)
from portolan.extract.schema import ExtractionResult, extraction_result_json_schema
from portolan.metering import InMemorySink
from portolan.training.bakeoff_scoring import (
    MIN_OVERLAP,
    LabelFact,
    overlap_ratio,
    precision_recall,
    vote_facts,
)
from portolan.training.local_teacher import (
    canonical_local_teacher,
    is_local_teacher,
    make_local_client,
    require_local_teacher,
)
from portolan.training.pipeline import (
    CORPUS_KINDS,
    CORPUS_LICENSES,
    assert_not_holdout,
    documents_for_corpus,
    run_corpus_single,
)
from portolan.training.records import (
    DroppedFact,
    TrainingRecord,
    dropped_path,
    failures_path,
    progress_path,
    read_records,
    records_path,
    repaired_records_path,
    repairs_log_path,
    silver_dir,
)
from portolan.training.repair import REPAIR_SCHEMA, RepairLogEntry
from portolan.training.spend import SpendMeter, default_spend_path
from portolan.training.typed_judge import build_fact_questions, fact_type_criteria
from scripts.data.silver_contexts import CORPUS_CONTEXTS

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPORA: tuple[str, ...] = ("fda_maude", "cfpb", "enron", "ubuntu_irc")
SPEND_KEY = "local_bakeoff"
DEFAULT_SEED = 20260922
REFERENCE_SLUG = "reference"

# the maintainer's bar, 2026-09-22: local labels replace Kimi K2 only if all hold.
BAR = {
    "json_valid": 0.95,
    "quote_verbatim": 0.95,
    "precision": 0.80,
    "recall": 0.70,
    "complete": 0.80,
    "docs_per_hour": 60.0,
}


def slug(teacher: str) -> str:
    return re.sub(r"[^a-z0-9.]+", "-", teacher.lower()).strip("-")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _git_sha() -> str:
    out = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return out.stdout.strip()


# --------------------------------------------------------------------------
# select
# --------------------------------------------------------------------------


def quotas(n: int, corpora: tuple[str, ...] = CORPORA) -> dict[str, int]:
    """`n` documents spread over `corpora` as evenly as possible, the
    remainder going to the first corpora."""
    base, extra = divmod(n, len(corpora))
    return {c: base + (1 if i < extra else 0) for i, c in enumerate(corpora)}


def select_documents(source_ids: dict[str, list[str]], n: int, seed: int) -> list[tuple[str, str]]:
    """`(corpus, source_id)` pairs: a seeded sample per corpus, in corpus order."""
    rng = random.Random(seed)
    picked = []
    for corpus, k in quotas(n).items():
        pool = sorted(set(source_ids.get(corpus, [])))
        if len(pool) < k:
            raise ValueError(f"{corpus}: {len(pool)} reference documents, need {k}")
        picked += [(corpus, sid) for sid in sorted(rng.sample(pool, k))]
    return picked


def core_documents(documents: list[dict[str, Any]], per_corpus: int | None) -> list[dict[str, Any]]:
    """The first `per_corpus` selected documents of each corpus, in selection
    order (all of them when `per_corpus` is None): a smaller stratified core
    every model runs when the full selection would take too long locally."""
    if per_corpus is None:
        return list(documents)
    seen: Counter = Counter()
    core = []
    for d in documents:
        if seen[d["corpus"]] < per_corpus:
            core.append(d)
            seen[d["corpus"]] += 1
    return core


def cmd_select(args: argparse.Namespace) -> int:
    ref_root: Path = args.reference_root
    assert_not_holdout(ref_root)
    source_ids: dict[str, list[str]] = {}
    for corpus in CORPORA:
        records = read_records(repaired_records_path(ref_root, corpus))
        source_ids[corpus] = [r.source_id for r in records]
    picked = select_documents(source_ids, args.n, args.seed)
    wanted = {c: {s for cc, s in picked if cc == c} for c in CORPORA}

    ref_var = args.out / "runs" / REFERENCE_SLUG / "var"
    documents = []
    teachers: set[str] = set()
    for corpus in CORPORA:
        src = silver_dir(ref_root, corpus)
        dst = silver_dir(ref_var, corpus)
        keep = wanted[corpus]
        for name in ("records.jsonl", "records.repaired.jsonl"):
            rows = [r for r in _read_jsonl(src / name) if r["source_id"] in keep]
            _write_jsonl(dst / name, rows)
        for name in ("dropped.jsonl", "repairs.jsonl"):
            rows = [r for r in _read_jsonl(src / name) if r["source_id"] in keep]
            _write_jsonl(dst / name, rows)
        for record in read_records(repaired_records_path(ref_var, corpus)):
            teachers.update(record.teachers)
            documents.append(
                {
                    "corpus": corpus,
                    "source_id": record.source_id,
                    "record_id": record.record_id,
                    "system_prompt_sha256": record.system_prompt_sha256,
                }
            )
    selection = {
        "seed": args.seed,
        "n": args.n,
        "reference": {"root": str(ref_root), "teachers": sorted(teachers)},
        "documents": documents,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    print(f"selected {len(documents)} documents: {dict(Counter(d['corpus'] for d in documents))}")
    return 0


# --------------------------------------------------------------------------
# label
# --------------------------------------------------------------------------


class RecordingClient:
    """Passes every call through to the wrapped client and keeps each
    reply's raw text, which `LLMClientBackend` does not return."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.texts: list[str] = []

    @property
    def model_id(self) -> str:
        return self.inner.model_id

    def using_sink(self, sink: Any, tenant: str) -> Any:
        return self.inner.using_sink(sink, tenant)

    def complete(
        self,
        *,
        system: str,
        user: str,
        json_schema: dict[str, Any] | None = None,
        model: str | None = None,
        surface: Any,
        max_tokens: int = 4096,
        temperature: float | None = None,
        validate_json: bool = True,
    ) -> Any:
        result = self.inner.complete(
            system=system,
            user=user,
            json_schema=json_schema,
            model=model,
            surface=surface,
            max_tokens=max_tokens,
            temperature=temperature,
            validate_json=validate_json,
        )
        self.texts.append(result.text)
        return result


class DiagnosticBackend:
    """The production `LLMClientBackend`, plus one `calls.jsonl` line per
    extraction attempt: model calls made, whether the one retry or the
    lenient parser was needed, raw replies, the parsed answer, time and
    peak memory. The pipeline retries a raised attempt once; each attempt
    gets its own line."""

    def __init__(self, recorder: RecordingClient, log_path: Path, corpus: str) -> None:
        self.recorder = recorder
        self.inner = LLMClientBackend(recorder, model=recorder.model_id, surface="extract")
        self.log_path = log_path
        self.corpus = corpus

    def extract(self, envelope: Any, system_prompt: str, sink: Any, tenant: str) -> Any:
        events_before = len(sink.events)
        texts_before = len(self.recorder.texts)
        started = time.perf_counter()
        line: dict[str, Any] = {"corpus": self.corpus, "source_id": envelope.source_id}
        try:
            result = self.inner.extract(envelope, system_prompt, sink, tenant)
        except Exception as exc:
            line["error"] = f"{type(exc).__name__}: {exc}"[:500]
            result = None
            raise
        finally:
            events = sink.events[events_before:]
            model_events = [e for e in events if "mlx" in e.extra]
            lenient = [e for e in events if "coercions" in e.extra]
            line.update(
                {
                    "seconds": round(time.perf_counter() - started, 3),
                    "calls": len(model_events),
                    "retry": any(e.extra["mlx"].get("retry") for e in model_events),
                    "lenient": bool(lenient),
                    "coercions": sum(e.extra.get("coercions", 0) for e in lenient),
                    "rejected_facts": sum(len(e.extra.get("rejected_facts", [])) for e in lenient),
                    "finish_reasons": [e.extra["mlx"].get("finish_reason") for e in model_events],
                    "input_tokens": sum(e.input_tokens for e in model_events),
                    "cached_tokens": sum(e.cached_tokens for e in model_events),
                    "output_tokens": sum(e.output_tokens for e in model_events),
                    "peak_memory_bytes": max(
                        (e.extra["mlx"].get("peak_memory_bytes", 0) for e in model_events),
                        default=0,
                    ),
                    "raw_texts": self.recorder.texts[texts_before:],
                    "result": result.model_dump(mode="json") if result is not None else None,
                }
            )
            with self.log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(line) + "\n")
        return result


class KeyedSpend(SpendMeter):
    """The shared meter with every line booked under `SPEND_KEY`."""

    def record(
        self, *, model: str, corpus: str, amount_usd: float, allow_zero: bool = False
    ) -> float:
        return super().record(
            model=model, corpus=SPEND_KEY, amount_usd=amount_usd, allow_zero=allow_zero
        )


def local_repair_call(client: Any, teacher: str, spend: SpendMeter, seconds: list[float]) -> Any:
    """A `portolan.training.repair.RepairCall` answered by the local model."""

    def call(model: str, system: str, user: str) -> tuple[dict[str, Any] | None, float]:
        if model != teacher:
            raise ValueError(f"repair for {model!r} sent to local teacher {teacher!r}")
        started = time.perf_counter()
        try:
            result = client.complete(
                system=system,
                user=user,
                json_schema=REPAIR_SCHEMA,
                model=client.model_id,
                surface="extract",
                max_tokens=1500,
            )
        finally:
            seconds.append(time.perf_counter() - started)
            spend.record(model=teacher, corpus=SPEND_KEY, amount_usd=0.0, allow_zero=True)
        return result.data, 0.0

    return call


def cmd_label(args: argparse.Namespace) -> int:
    from scripts.data.repair_silver import repair_corpus

    if not is_local_teacher(args.teacher):
        print("error: label runs local teachers only (local:<repo id>)", file=sys.stderr)
        return 2
    info = require_local_teacher(args.teacher)
    teacher = canonical_local_teacher(args.teacher)
    selection = json.loads((args.out / "selection.json").read_text())
    run_dir = args.out / "runs" / slug(teacher)
    run_var = run_dir / "var"
    assert_not_holdout(args.converted_root)
    assert_not_holdout(run_dir)
    run_var.mkdir(parents=True, exist_ok=True)
    converted_link = run_var / "data" / "converted"
    if not converted_link.exists():
        converted_link.parent.mkdir(parents=True, exist_ok=True)
        converted_link.symlink_to(args.converted_root.resolve())

    spend = KeyedSpend(path=args.spend_file)
    client = make_local_client(teacher, sink=InMemorySink(), tenant="local-bakeoff", progress=True)
    recorder = RecordingClient(client)
    schema = extraction_result_json_schema()
    started_at = time.time()
    t0 = time.perf_counter()
    client.load()
    load_seconds = time.perf_counter() - t0

    per_corpus: dict[str, Any] = {}
    for corpus in CORPORA:
        core = core_documents(selection["documents"], args.max_per_corpus)
        docs = [d for d in core if d["corpus"] == corpus]
        if args.corpus and corpus not in args.corpus:
            continue
        wanted = {d["source_id"] for d in docs}
        ref_shas = {d["system_prompt_sha256"] for d in docs}
        context = CORPUS_CONTEXTS[corpus]
        system_prompt = build_system_prompt(company_context=context, variant=DEFAULT_PROMPT_VARIANT)
        if ref_shas != {system_prompt_sha256(system_prompt)}:
            print(f"error: {corpus}: prompt differs from the reference's", file=sys.stderr)
            return 2
        documents = [
            d
            for d in documents_for_corpus(corpus, args.converted_root / corpus)
            if d.source_id in wanted
        ]
        if len(documents) != len(wanted):
            print(f"error: {corpus}: found {len(documents)} of {len(wanted)}", file=sys.stderr)
            return 2

        t0 = time.perf_counter()
        client.warm(system_prompt, schema)
        prefix_seconds = time.perf_counter() - t0
        backend = DiagnosticBackend(recorder, run_dir / "calls.jsonl", corpus)
        t0 = time.perf_counter()
        stats = run_corpus_single(
            corpus=corpus,
            kind=CORPUS_KINDS[corpus],
            license_text=CORPUS_LICENSES[corpus],
            documents=documents,
            system_prompt=system_prompt,
            prompt_variant=DEFAULT_PROMPT_VARIANT,
            teacher_model=teacher,
            overlap_threshold=0.0,
            backend_factory=lambda _model, b=backend: b,
            limiter=TokenBucketLimiter(requests_per_minute=10**9, requests_per_day=10**9),
            sink=client.sink,
            tenant="local-bakeoff",
            records_file=records_path(run_var, corpus),
            failures_file=failures_path(run_var, corpus),
            progress_file=progress_path(run_var, corpus),
            dropped_file=dropped_path(run_var, corpus),
            spend=spend,
            budget_usd=float("inf"),
            judges={},
            negative_cap_ratio=1.0,
            timings_file=silver_dir(run_var, corpus) / "timings.jsonl",
        )
        label_seconds = time.perf_counter() - t0

        repair_seconds: list[float] = []
        t0 = time.perf_counter()
        result = repair_corpus(
            var_root=run_var,
            out_root=run_var,
            corpus=corpus,
            call=local_repair_call(client, teacher, spend, repair_seconds),
            budget_left=lambda: True,
            company_context=context,
            workers=1,
        )
        repair_wall = time.perf_counter() - t0
        per_corpus[corpus] = {
            "documents": stats.documents_processed,
            "resumed": stats.documents_skipped_resumed,
            "failures": stats.failures,
            "prefix_seconds": round(prefix_seconds, 1),
            "label_seconds": round(label_seconds, 1),
            "repair_seconds": round(repair_wall, 1),
            "repair_calls": result.calls_made,
            "complete_records": result.complete_records,
            "incomplete_records": result.incomplete_records,
        }
        print(f"[{corpus}] {json.dumps(per_corpus[corpus])}", flush=True)

    run = {
        "teacher": teacher,
        "base_model": info.base_model,
        "revision": info.revision,
        "license": info.license,
        "git": _git_sha(),
        "started": started_at,
        "finished": time.time(),
        "load_seconds": round(load_seconds, 1),
        "corpora": per_corpus,
        "peak_memory_bytes": client.peak_memory_bytes(),
        "max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    previous = (
        json.loads((run_dir / "run.json").read_text()) if (run_dir / "run.json").exists() else {}
    )
    if previous.get("corpora"):
        run["corpora"] = {**previous["corpora"], **per_corpus}
    (run_dir / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    print(json.dumps(run, indent=2))
    return 0


# --------------------------------------------------------------------------
# wp5: hand-checked v2 labels
# --------------------------------------------------------------------------


def review_hits(
    review: list[dict[str, Any]], spans_by_doc: dict[str, list[tuple[int, int]]]
) -> dict[str, Any]:
    """For each hand-checked reference label (`doc_id`, `char_start`,
    `char_end`, `correct`), whether any of this labeler's spans on that
    source overlaps it by at least half the shorter span (type not
    required: the verdict says whether the span is a fact at all)."""
    per_fact = []
    for row in review:
        target = (row["char_start"], row["char_end"])
        hit = any(
            overlap_ratio(target, span) >= MIN_OVERLAP
            for span in spans_by_doc.get(row["doc_id"], [])
        )
        per_fact.append({"fact_id": row["fact_id"], "correct": row["correct"], "labeled": hit})
    wrong = [p for p in per_fact if not p["correct"]]
    right = [p for p in per_fact if p["correct"]]
    return {
        "wrong_total": len(wrong),
        "wrong_also_labeled": sum(p["labeled"] for p in wrong),
        "correct_total": len(right),
        "correct_found": sum(p["labeled"] for p in right),
        "labels_on_these_sources": sum(len(v) for v in spans_by_doc.values()),
        "per_fact": per_fact,
    }


def cmd_wp5(args: argparse.Namespace) -> int:
    from corpus.generator.v2.completeness import (
        CompletenessContext,
        TeacherCallError,
        TeacherRunBudget,
        source_inputs,
        teacher_pass,
        world_company_context,
    )
    from corpus.generator.v2.models import Split
    from corpus.generator.v2.packs.schema import load_pack
    from corpus.generator.v2.planner import plan_world
    from corpus.generator.v2.simulator import simulate_world
    from corpus.generator.v2.writer import WrittenDoc

    teacher = canonical_local_teacher(args.teacher)
    run_dir = args.out / "runs" / slug(teacher)
    run_dir.mkdir(parents=True, exist_ok=True)
    review = _read_jsonl(args.review)
    plan_ids = sorted({r["plan_id"] for r in review})
    pack = load_pack(REPO_ROOT / "corpus" / "generator" / "v2" / "packs" / f"{args.pack}.yaml")
    world = simulate_world(pack, seed=args.seed, months=args.months, split=Split(args.split))
    world_plan = plan_world(world, pack)
    plans = world_plan.doc_by_id()

    client = make_local_client(
        teacher, sink=InMemorySink(), tenant="local-bakeoff-wp5", progress=True
    )
    recorder = RecordingClient(client)
    backend = DiagnosticBackend(recorder, run_dir / "wp5_calls.jsonl", "synthetic_v2")
    ctx = CompletenessContext(
        world=world,
        world_plan=world_plan,
        backend_factory=lambda _model: backend,
        budget=TeacherRunBudget(cap_usd=args.budget_usd),
        teacher_model=teacher,
        company_context=world_company_context(world),
        sink=client.sink,
    )
    client.warm(ctx.system_prompt, extraction_result_json_schema())
    rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    failed: list[str] = []
    for plan_id in plan_ids:
        doc = WrittenDoc.model_validate_json(
            (args.wp4_dir / "docs" / f"{plan_id}.json").read_text(encoding="utf-8")
        )
        # One source at a time: a source the model fails on (twice, WP5's
        # rule) counts as no labels there instead of ending the run.
        for source in source_inputs(doc, plans[plan_id]):
            try:
                facts, _calls = teacher_pass(
                    ctx, [source], model=teacher, role="completeness", plan_id=plan_id, attempt=0
                )
            except TeacherCallError:
                failed.append(source.source_id)
                continue
            rows += [{"plan_id": plan_id, **f.model_dump(mode="json")} for f in facts]
    seconds = time.perf_counter() - t0
    _write_jsonl(run_dir / "wp5_facts.jsonl", rows)
    kept = [r for r in rows if r["gate"] == "kept" and r["char_start"] is not None]
    kept = [r for r in kept if not r["duplicate"]]
    spans: dict[str, list[tuple[int, int]]] = {}
    for r in kept:
        spans.setdefault(r["doc_id"], []).append((r["char_start"], r["char_end"]))
    report = {
        "teacher": teacher,
        "documents": plan_ids,
        "facts_emitted": len(rows),
        "facts_kept": len(kept),
        "failed_sources": failed,
        "seconds": round(seconds, 1),
        **review_hits(review, spans),
    }
    (run_dir / "wp5.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "per_fact"}, indent=2))
    return 0


def wp5_vote(out: Path, slugs: list[str], review: list[dict[str, Any]]) -> dict[str, Any]:
    """Two-of-three vote over the voters' kept WP5 facts (type and span)."""
    per_voter: list[dict[str, list[LabelFact]]] = []
    for s_ in slugs:
        facts: dict[str, list[LabelFact]] = {}
        for r in _read_jsonl(out / "runs" / s_ / "wp5_facts.jsonl"):
            if r["gate"] == "kept" and r["char_start"] is not None and not r["duplicate"]:
                facts.setdefault(r["doc_id"], []).append(
                    LabelFact(r["fact_type"], (r["char_start"], r["char_end"]))
                )
        per_voter.append(facts)
    docs = {d for v in per_voter for d in v}
    spans = {
        d: [f.span for f in vote_facts([v.get(d, []) for v in per_voter]) if f.span] for d in docs
    }
    return {"voters": slugs, **review_hits(review, spans)}


# --------------------------------------------------------------------------
# judge
# --------------------------------------------------------------------------


def judge_system_prompt() -> str:
    """The typed judge's own three questions (`typed_judge.build_fact_questions`)
    as one prompt a chat model answers in JSON."""
    questions = build_fact_questions()
    types = "\n".join(f"- {name}: {text}" for name, text in fact_type_criteria().items())
    return (
        "You check one extracted fact against the quote it was written from, and answer "
        "three questions about it.\n\n"
        f"faithful (true or false): {questions['faithful']['instructions']}\n\n"
        f"fact_type: {questions['fact_type']['instructions']} Choose one of:\n{types}\n\n"
        f"added_detail (true or false): {questions['added_detail']['instructions']}"
    )


JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "faithful": {"type": "boolean"},
        "fact_type": {"type": "string", "enum": list(fact_type_criteria())},
        "added_detail": {"type": "boolean"},
    },
    "required": ["faithful", "fact_type", "added_detail"],
}


def score_judge(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """`judge_calibration.print_report`'s numbers from rows of
    `{variant, true_type, emitted_type, expected_*, expect_type_check,
    answer}`; an unusable answer counts as wrong on every question."""

    def correct(row: dict[str, Any], key: str) -> bool:
        answer = row["answer"]
        return answer is not None and answer[key] == row[f"expected_{key}"]

    def keep(row: dict[str, Any]) -> bool | None:
        a = row["answer"]
        if a is None:
            return None
        return a["faithful"] and not a["added_detail"] and a["fact_type"] == row["emitted_type"]

    n = len(rows)
    typed = [r for r in rows if r["expect_type_check"]]
    by_variant = {}
    for variant in ("gold", "wrong_type", "quote_swapped", "added_detail"):
        vr = [r for r in rows if r["variant"] == variant]
        by_variant[variant] = _ratio(sum(keep(r) == r["expected_keep"] for r in vr), len(vr))
    return {
        "items": n,
        "invalid_answers": sum(1 for r in rows if r["answer"] is None),
        "faithful": _ratio(sum(correct(r, "faithful") for r in rows), n),
        "fact_type": _ratio(
            sum(
                r["answer"] is not None and r["answer"]["fact_type"] == r["true_type"]
                for r in typed
            ),
            len(typed),
        ),
        "added_detail": _ratio(sum(correct(r, "added_detail") for r in rows), n),
        "keep_agreement": _ratio(sum(keep(r) == r["expected_keep"] for r in rows), n),
        "keep_by_variant": by_variant,
    }


def reference_gold_facts(out: Path, n: int, seed: int) -> list[dict[str, Any]]:
    """`n` of the reference teacher's final facts on the selected documents
    (seeded), in the shape `judge_calibration.build_calibration_items` takes.
    Unlike the Halvard truth table, whose claims name accounts and dates
    their short quotes do not, each of these claims was written from its
    quote and passed the keep gate."""
    selection = json.loads((out / "selection.json").read_text())
    reference = _load_run(out, REFERENCE_SLUG, selection["documents"])
    facts = [
        {"id": f"{sid}:{i}", "type": f.fact_type, "claim": f.claim, "quote": f.quote}
        for sid, record in sorted(reference["records"].items())
        for i, f in enumerate(record.facts)
        if f.quote
    ]
    return random.Random(seed).sample(facts, min(n, len(facts)))


def cmd_judge(args: argparse.Namespace) -> int:
    from scripts.data.judge_calibration import (
        build_calibration_items,
        load_gold_facts_with_quotes,
    )

    teacher = canonical_local_teacher(args.teacher)
    run_dir = args.out / "runs" / slug(teacher)
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.gold == "halvard":
        gold = load_gold_facts_with_quotes(args.truth_table)
    else:
        gold = reference_gold_facts(args.out, args.n, DEFAULT_SEED)
    items = build_calibration_items(gold)
    spend = KeyedSpend(path=args.spend_file)
    client = make_local_client(teacher, sink=InMemorySink(), tenant="local-bakeoff-judge")
    system = judge_system_prompt()
    t0 = time.perf_counter()
    client.warm(system, JUDGE_SCHEMA)
    warm_seconds = time.perf_counter() - t0
    rows = []
    t0 = time.perf_counter()
    for item in items:
        result = client.complete(
            system=system,
            user=f'Claim: "{item.claim}"\nQuote: "{item.quote}"',
            json_schema=JUDGE_SCHEMA,
            model=client.model_id,
            surface="extract",
            max_tokens=200,
        )
        spend.record(model=teacher, corpus=SPEND_KEY, amount_usd=0.0, allow_zero=True)
        rows.append(
            {
                "fact_id": item.fact_id,
                "variant": item.variant,
                "true_type": item.true_type,
                "emitted_type": item.emitted_type,
                "expected_faithful": item.expected_faithful,
                "expected_added_detail": item.expected_added_detail,
                "expect_type_check": item.expect_type_check,
                "expected_keep": item.expected_keep,
                "answer": result.data,
                "raw": result.text,
            }
        )
    seconds = time.perf_counter() - t0
    report = {
        "teacher": teacher,
        **score_judge(rows),
        "warm_seconds": round(warm_seconds, 1),
        "seconds_per_item": round(seconds / len(items), 2),
        "items_per_hour": round(3600 * len(items) / seconds, 1),
        "peak_memory_bytes": client.peak_memory_bytes(),
    }
    suffix = "" if args.gold == "halvard" else f"-{args.gold}"
    report["gold"] = args.gold
    _write_jsonl(run_dir / f"judge{suffix}_items.jsonl", rows)
    (run_dir / f"judge{suffix}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


# --------------------------------------------------------------------------
# score
# --------------------------------------------------------------------------


def _load_run(out: Path, run_slug: str, documents: list[dict[str, Any]]) -> dict[str, Any]:
    run_dir = out / "runs" / run_slug
    run_var = run_dir / "var"
    records: dict[str, TrainingRecord] = {}
    raw_records: dict[str, TrainingRecord] = {}
    drops: dict[str, list[DroppedFact]] = {}
    repairs: list[RepairLogEntry] = []
    timings: dict[str, float] = {}
    for corpus in CORPORA:
        for r in read_records(repaired_records_path(run_var, corpus)):
            records[r.source_id] = r
        for r in read_records(records_path(run_var, corpus)):
            raw_records[r.source_id] = r
        for row in _read_jsonl(dropped_path(run_var, corpus)):
            d = DroppedFact.model_validate(row)
            drops.setdefault(d.source_id, []).append(d)
        repairs += [
            RepairLogEntry.model_validate(row)
            for row in _read_jsonl(repairs_log_path(run_var, corpus))
        ]
        for row in _read_jsonl(silver_dir(run_var, corpus) / "timings.jsonl"):
            timings[row["source_id"]] = row["seconds"]
    calls: dict[str, list[dict[str, Any]]] = {}
    for row in _read_jsonl(run_dir / "calls.jsonl"):
        calls.setdefault(row["source_id"], []).append(row)
    meta = json.loads((run_dir / "run.json").read_text()) if (run_dir / "run.json").exists() else {}
    return {
        "slug": run_slug,
        "records": records,
        "raw_records": raw_records,
        "drops": drops,
        "repairs": repairs,
        "timings": timings,
        "calls": calls,
        "meta": meta,
        "documents": [d["source_id"] for d in documents],
    }


def final_facts(run: dict[str, Any]) -> dict[str, list[LabelFact]]:
    return {
        sid: [LabelFact.from_training_fact(f) for f in record.facts]
        for sid, record in run["records"].items()
    }


def _ratio(a: float, b: float) -> float | None:
    return a / b if b else None


def run_metrics(run: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    docs = run["documents"]
    n = len(docs)
    # JSON validity: the answer validated strictly, first try or after the
    # production chain's one retry, with no lenient salvage and no failure.
    json_valid = first_try = 0
    for sid in docs:
        attempts = run["calls"].get(sid, [])
        ok = [a for a in attempts if not a.get("error")]
        if ok and not ok[-1]["lenient"]:
            json_valid += 1
            if len(attempts) == 1 and ok[-1]["calls"] == 1:
                first_try += 1

    # Quote verbatim: the teacher's own quote string (before the gate
    # resolves it to a span) is an exact substring of the record text.
    kept_verbatim = kept_total = kept_unknown = 0
    emitted_verbatim = emitted_quoted = emitted_total = 0
    repair_quotes = {
        (e.record_id, (e.returned or {}).get("claim")): (e.returned or {}).get("quote")
        for e in run["repairs"]
        if e.returned
    }
    for sid in docs:
        record = run["records"].get(sid)
        attempts = [a for a in run["calls"].get(sid, []) if a.get("result")]
        raw = ExtractionResult.model_validate(attempts[-1]["result"]) if attempts else None
        text = record.text if record is not None else ""
        raw_quotes = {f.claim: f.quote for f in raw.facts} if raw else {}
        if raw:
            emitted_total += len(raw.facts)
            for f in raw.facts:
                if f.quote:
                    emitted_quoted += 1
                    emitted_verbatim += int(f.quote in text)
        if record is None:
            continue
        for f in record.facts:
            if f.quote is None:
                continue
            kept_total += 1
            quote = raw_quotes.get(f.claim) or repair_quotes.get((record.record_id, f.claim))
            if quote is None:
                kept_unknown += 1
            elif quote in text:
                kept_verbatim += 1

    complete = sum(
        1
        for sid in docs
        if (r := run["records"].get(sid)) and r.label_audit and r.label_audit.status == "complete"
    )
    audits = [r.label_audit for sid in docs if (r := run["records"].get(sid)) and r.label_audit]
    label_seconds = sum(run["timings"].get(sid, 0.0) for sid in docs)
    repair_seconds = sum(
        c.get("repair_seconds", 0.0) for c in run["meta"].get("corpora", {}).values()
    )
    total_seconds = label_seconds + repair_seconds
    per_doc = [run["timings"][sid] for sid in docs if sid in run["timings"]]

    pred, ref = final_facts(run), final_facts(reference)
    # Filter proxy: would this model's first pass flag the documents the
    # reference found facts in (so a cheap local pass could skip the rest)?
    emitted_any = {
        sid: any(a.get("result") and a["result"]["facts"] for a in run["calls"].get(sid, []))
        for sid in docs
    }
    ref_any = {sid: bool(ref.get(sid)) for sid in docs}
    flagged = sum(1 for sid in docs if ref_any[sid] and emitted_any[sid])
    typed = precision_recall(pred, ref, documents=docs)
    span_only = precision_recall(pred, ref, documents=docs, require_type=False)
    emitted_per_doc = _ratio(emitted_total, n)
    return {
        "documents": n,
        "records": sum(1 for sid in docs if sid in run["records"]),
        "hard_failures": sum(
            1 for sid in docs if not any(not a.get("error") for a in run["calls"].get(sid, []))
        ),
        "json_valid": _ratio(json_valid, n),
        "json_valid_first_try": _ratio(first_try, n),
        "lenient_documents": sum(
            1 for sid in docs if any(a["lenient"] for a in run["calls"].get(sid, []))
        ),
        "facts_emitted_per_doc": emitted_per_doc,
        "facts_final_per_doc": _ratio(sum(len(pred.get(sid, [])) for sid in docs), n),
        "quote_verbatim_kept": _ratio(kept_verbatim, kept_total),
        "quote_verbatim_kept_counts": [kept_verbatim, kept_total, kept_unknown],
        "quote_verbatim_emitted": _ratio(emitted_verbatim, emitted_quoted),
        "complete": _ratio(complete, n),
        "repaired": sum(a.repaired for a in audits),
        "withdrawn": sum(a.withdrawn for a in audits),
        "unresolved": sum(a.unresolved for a in audits),
        "vs_reference": typed.as_dict(),
        "vs_reference_span_only": span_only.as_dict(),
        "type_agreement_on_span_matches": _ratio(typed.matched, span_only.matched),
        "filter_recall_of_documents_with_facts": _ratio(flagged, sum(ref_any.values())),
        "filter_skip_rate": _ratio(sum(not v for v in emitted_any.values()), n),
        "reference_documents_without_facts": sum(not v for v in ref_any.values()),
        "label_seconds_per_doc_mean": _ratio(label_seconds, len(per_doc)),
        "label_seconds_per_doc_median": statistics.median(per_doc) if per_doc else None,
        "repair_seconds_total": repair_seconds,
        "docs_per_hour": _ratio(3600 * len(per_doc), total_seconds),
        "load_seconds": run["meta"].get("load_seconds"),
        "prefix_seconds": {
            c: v.get("prefix_seconds") for c, v in run["meta"].get("corpora", {}).items()
        },
        "peak_memory_gb": (run["meta"].get("peak_memory_bytes") or 0) / 1e9,
        "max_rss_gb": (run["meta"].get("max_rss_bytes") or 0) / 1e9,
    }


def verdicts(m: dict[str, Any]) -> dict[str, str]:
    def check(value: float | None, bar: float) -> str:
        return "PASS" if value is not None and value >= bar else "FAIL"

    return {
        "json_valid": check(m.get("json_valid"), BAR["json_valid"]),
        "quote_verbatim": check(m.get("quote_verbatim_kept"), BAR["quote_verbatim"]),
        "precision": check(m["vs_reference"]["precision"], BAR["precision"]),
        "recall": check(m["vs_reference"]["recall"], BAR["recall"]),
        "complete": check(m.get("complete"), BAR["complete"]),
        "throughput": check(m.get("docs_per_hour"), BAR["docs_per_hour"]),
    }


def vote_metrics(runs: list[dict[str, Any]], reference: dict[str, Any]) -> dict[str, Any]:
    """A majority of the runs (two of three) on each fact. A document's vote
    counts as JSON-valid / complete when at least two voters' records are;
    its time is the voters' times added up (one Mac runs them in turn)."""
    docs = reference["documents"]
    finals = [final_facts(r) for r in runs]
    voted = {sid: vote_facts([f.get(sid, []) for f in finals]) for sid in docs}
    ref = final_facts(reference)
    typed = precision_recall(voted, ref, documents=docs)
    span_only = precision_recall(voted, ref, documents=docs, require_type=False)
    member = [run_metrics(r, reference) for r in runs]
    valid_docs = complete_docs = 0
    for sid in docs:
        valid = complete = 0
        for r in runs:
            attempts = [a for a in r["calls"].get(sid, []) if not a.get("error")]
            valid += int(bool(attempts) and not attempts[-1]["lenient"])
            rec = r["records"].get(sid)
            complete += int(bool(rec and rec.label_audit and rec.label_audit.status == "complete"))
        valid_docs += int(valid >= 2)
        complete_docs += int(complete >= 2)
    hours = sum(len(docs) / m["docs_per_hour"] for m in member if m["docs_per_hour"])
    return {
        "voters": [r["slug"] for r in runs],
        "json_valid": _ratio(valid_docs, len(docs)),
        "quote_verbatim_kept": min(
            (m["quote_verbatim_kept"] for m in member if m["quote_verbatim_kept"] is not None),
            default=None,
        ),
        "complete": _ratio(complete_docs, len(docs)),
        "facts_final_per_doc": _ratio(sum(len(v) for v in voted.values()), len(docs)),
        "vs_reference": typed.as_dict(),
        "vs_reference_span_only": span_only.as_dict(),
        "docs_per_hour": _ratio(len(docs), hours),
    }


def pairwise(runs: list[dict[str, Any]], docs: list[str]) -> dict[str, dict[str, float | None]]:
    """Typed F1 between every two runs (symmetric)."""
    finals = {r["slug"]: final_facts(r) for r in runs}
    table: dict[str, dict[str, float | None]] = {}
    for a, b in itertools.product(finals, repeat=2):
        table.setdefault(a, {})[b] = precision_recall(finals[a], finals[b], documents=docs).f1
    return table


def _fmt(value: Any, pct: bool = True) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.1%}" if pct else f"{value:.1f}"
    return str(value)


def render_markdown(scores: dict[str, Any]) -> str:
    lines = [
        f"# Local-teacher bake-off scores ({scores['documents']} documents)",
        "",
        f"Reference: {', '.join(scores['reference_teachers'])}. Bar: JSON valid >= 95%, quote "
        "verbatim >= 95% of kept facts, precision >= 0.80 and recall >= 0.70 against the "
        "reference (type + quote-span overlap), complete records >= 80%, >= 60 documents/hour.",
        "",
        f"Reference facts per document (final): "
        f"{_fmt(scores['reference']['facts_final_per_doc'], False)}; reference complete records: "
        f"{_fmt(scores['reference']['complete'])}.",
        "",
        "| model | JSON valid (first try) | quote verbatim, kept (emitted) | facts/doc emitted, final "
        "| precision | recall | span-only P / R | type agree | complete | s/doc label (median) "
        "| docs/hour | peak GB |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, m in scores["models"].items():
        v = m["vs_reference"]
        s = m["vs_reference_span_only"]
        lines.append(
            f"| {name} | {_fmt(m['json_valid'])} ({_fmt(m['json_valid_first_try'])}) "
            f"| {_fmt(m['quote_verbatim_kept'])} ({_fmt(m['quote_verbatim_emitted'])}) "
            f"| {_fmt(m['facts_emitted_per_doc'], False)}, {_fmt(m['facts_final_per_doc'], False)} "
            f"| {_fmt(v['precision'])} | {_fmt(v['recall'])} "
            f"| {_fmt(s['precision'])} / {_fmt(s['recall'])} "
            f"| {_fmt(m['type_agreement_on_span_matches'])} | {_fmt(m['complete'])} "
            f"| {_fmt(m['label_seconds_per_doc_mean'], False)} "
            f"({_fmt(m['label_seconds_per_doc_median'], False)}) "
            f"| {_fmt(m['docs_per_hour'], False)} | {m['peak_memory_gb']:.2f} |"
        )
    lines += ["", "## PASS / FAIL", "", "| model | " + " | ".join(BAR) + " |"]
    lines.append("|---|" + "---|" * len(BAR))
    for name, v in scores["verdicts"].items():
        lines.append(f"| {name} | " + " | ".join(v.values()) + " |")
    lines += ["", "## Majority vote (2 of 3)", ""]
    lines.append(
        "| voters | JSON valid | complete | facts/doc | precision | recall "
        "| span-only P / R | docs/hour |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for vote in scores["votes"]:
        v = vote["vs_reference"]
        s = vote["vs_reference_span_only"]
        lines.append(
            f"| {' + '.join(vote['voters'])} | {_fmt(vote['json_valid'])} "
            f"| {_fmt(vote['complete'])} | {_fmt(vote['facts_final_per_doc'], False)} "
            f"| {_fmt(v['precision'])} | {_fmt(v['recall'])} "
            f"| {_fmt(s['precision'])} / {_fmt(s['recall'])} | {_fmt(vote['docs_per_hour'], False)} |"
        )
    if scores.get("judges"):
        lines += [
            "",
            "## Judge calibration (80 items: gold, wrong type, swapped quote, added detail)",
            "",
            "| model | faithful | fact type | added detail | keep/drop | invalid | s/item |",
            "|---|---|---|---|---|---|---|",
        ]
        for name, j in scores["judges"].items():
            lines.append(
                f"| {name} | {_fmt(j['faithful'])} | {_fmt(j['fact_type'])} "
                f"| {_fmt(j['added_detail'])} | {_fmt(j['keep_agreement'])} "
                f"| {j['invalid_answers']} | {j['seconds_per_item']} |"
            )
    if scores.get("wp5"):
        lines += [
            "",
            "## Hand-checked v2 labels (WP5 precision review of Kimi K2)",
            "",
            "| labeler | Kimi's wrong labels also labeled (lower is better) "
            "| Kimi's correct labels found | labels on these sources | seconds |",
            "|---|---|---|---|---|",
        ]
        for name, w in scores["wp5"].items():
            lines.append(
                f"| {name} | {w['wrong_also_labeled']} of {w['wrong_total']} "
                f"| {w['correct_found']} of {w['correct_total']} "
                f"| {w['labels_on_these_sources']} | {w['seconds']} |"
            )
        for v in scores.get("wp5_votes", []):
            lines.append(
                f"| vote: {' + '.join(v['voters'])} | {v['wrong_also_labeled']} of "
                f"{v['wrong_total']} | {v['correct_found']} of {v['correct_total']} "
                f"| {v['labels_on_these_sources']} | |"
            )
    names = list(scores["pairwise_f1"])
    lines += ["", "## Pairwise agreement (typed F1)", "", "| | " + " | ".join(names) + " |"]
    lines.append("|---|" + "---|" * len(names))
    for a in names:
        lines.append(
            f"| {a} | " + " | ".join(_fmt(scores["pairwise_f1"][a][b]) for b in names) + " |"
        )
    return "\n".join(lines) + "\n"


def cmd_score(args: argparse.Namespace) -> int:
    selection = json.loads((args.out / "selection.json").read_text())
    documents = core_documents(selection["documents"], args.max_per_corpus)
    reference = _load_run(args.out, REFERENCE_SLUG, documents)
    slugs = args.runs or sorted(
        p.name for p in (args.out / "runs").iterdir() if p.is_dir() and p.name != REFERENCE_SLUG
    )
    runs = [_load_run(args.out, s, documents) for s in slugs]
    models = {r["slug"]: run_metrics(r, reference) for r in runs}
    judges = {}
    for s_ in slugs:
        for name in ("judge.json", "judge-reference.json"):
            path = args.out / "runs" / s_ / name
            if path.exists():
                judges[f"{s_} ({name.removesuffix('.json')})"] = json.loads(path.read_text())
    wp5 = {}
    for s_ in slugs:
        path = args.out / "runs" / s_ / "wp5.json"
        if path.exists():
            report = json.loads(path.read_text())
            wp5[s_] = {k: v for k, v in report.items() if k != "per_fact"}
    wp5_votes = []
    if args.review is not None:
        review = _read_jsonl(args.review)
        for trio in itertools.combinations(sorted(wp5), 3):
            vote = wp5_vote(args.out, list(trio), review)
            wp5_votes.append({k: v for k, v in vote.items() if k != "per_fact"})
    ranked = sorted(runs, key=lambda r: -(models[r["slug"]]["vs_reference"]["f1"] or 0))
    votes = [vote_metrics(list(trio), reference) for trio in itertools.combinations(ranked, 3)]
    for vote in votes:
        vote["verdicts"] = verdicts(vote)
    scores = {
        "documents": len(documents),
        "reference_teachers": selection["reference"]["teachers"],
        "reference": {
            "facts_final_per_doc": _ratio(
                sum(len(final_facts(reference).get(sid, [])) for sid in reference["documents"]),
                len(documents),
            ),
            "complete": _ratio(
                sum(
                    1
                    for sid in reference["documents"]
                    if (r := reference["records"].get(sid))
                    and r.label_audit
                    and r.label_audit.status == "complete"
                ),
                len(documents),
            ),
        },
        "bar": BAR,
        "models": models,
        "verdicts": {name: verdicts(m) for name, m in models.items()},
        "votes": votes,
        "pairwise_f1": pairwise([*runs, reference], reference["documents"]),
        "judges": judges,
        "wp5": wp5,
        "wp5_votes": wp5_votes,
    }
    (args.out / "scores.json").write_text(json.dumps(scores, indent=2) + "\n")
    (args.out / "scores.md").write_text(render_markdown(scores))
    print(render_markdown(scores))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sel = sub.add_parser("select")
    sel.add_argument("--reference-root", type=Path, required=True)
    sel.add_argument("--out", type=Path, required=True)
    sel.add_argument("--n", type=int, default=30)
    sel.add_argument("--seed", type=int, default=DEFAULT_SEED)
    lab = sub.add_parser("label")
    lab.add_argument("--teacher", required=True)
    lab.add_argument("--out", type=Path, required=True)
    lab.add_argument(
        "--converted-root", type=Path, default=REPO_ROOT / "var" / "data" / "converted"
    )
    lab.add_argument("--spend-file", type=Path, default=default_spend_path(REPO_ROOT / "var"))
    lab.add_argument("--corpus", action="append", choices=CORPORA)
    lab.add_argument("--max-per-corpus", type=int, default=None)
    jud = sub.add_parser("judge")
    jud.add_argument("--teacher", required=True)
    jud.add_argument("--out", type=Path, required=True)
    jud.add_argument("--truth-table", type=Path, default=REPO_ROOT / "corpus" / "truth-table.yaml")
    jud.add_argument("--spend-file", type=Path, default=default_spend_path(REPO_ROOT / "var"))
    jud.add_argument(
        "--gold",
        choices=["halvard", "reference"],
        default="halvard",
        help="halvard: judge_calibration's truth-table items; reference: the reference "
        "teacher's own final facts on the selected documents",
    )
    jud.add_argument("--n", type=int, default=30, help="--gold reference: facts to sample")
    wp5 = sub.add_parser("wp5")
    wp5.add_argument("--teacher", required=True)
    wp5.add_argument("--out", type=Path, required=True)
    wp5.add_argument("--wp4-dir", type=Path, required=True, help="WP4 run holding docs/<plan>.json")
    wp5.add_argument("--review", type=Path, required=True, help="precision_review.jsonl")
    wp5.add_argument("--pack", default="medical_devices")
    wp5.add_argument("--seed", type=int, default=42)
    wp5.add_argument("--months", type=int, default=8)
    wp5.add_argument("--split", default="train")
    wp5.add_argument("--budget-usd", type=float, default=0.01, help="local calls cost $0")
    sco = sub.add_parser("score")
    sco.add_argument("--out", type=Path, required=True)
    sco.add_argument("--runs", nargs="*")
    sco.add_argument("--max-per-corpus", type=int, default=None)
    sco.add_argument("--review", type=Path, default=None, help="precision_review.jsonl, for votes")
    args = parser.parse_args(argv)
    commands = {
        "select": cmd_select,
        "label": cmd_label,
        "wp5": cmd_wp5,
        "judge": cmd_judge,
        "score": cmd_score,
    }
    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
