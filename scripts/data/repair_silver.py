"""Re-gate every silver record and logged drop with the current keep gate,
repair what still fails with one call to the same teacher, and write
`records.repaired.jsonl` with a `label_audit` on every record
(`portolan.training.repair`).

Labeling runs keep appending to `records.jsonl` and `dropped.jsonl`, so
this script never writes them. It reads a snapshot (progress, then
records, then drops: a live run logs an envelope's drops before its
record and marks a source processed after its last record, so the
snapshot is consistent; a line with no newline yet is still being
written and is left out) and writes, per corpus, next to the live files:

- `records.repaired.jsonl`: every snapshotted record, audited, plus a
  record for each envelope the label-time negative cap never wrote whose
  facts the gate had dropped (`recover_unwritten_records`); the
  training-set builder prefers this file;
- `repairs.jsonl`: one line per repair call, also the cache a rerun uses
  instead of paying again;
- `repair_summary.json`: counts and the sha256 of the bytes read.

Paid calls go through `OpenRouterClient` and are metered in the shared
spend meter under the corpus key `silver_repair`; the run stops calling
once that key reaches `--budget-usd` (facts left then stay unresolved).

`--label-judge MODEL` (off by default) then runs the label judge
(`portolan.training.label_judge`) over every audited record with facts:
drops and retypes go into `records.repaired.jsonl` and the record's
`label_audit`, calls into `label_judge.jsonl` (also its cache), spend under
`label_judge`, capped by `--judge-budget-usd` for this run. A fact the judge
left without a verdict makes its record incomplete until a rerun.

Usage:
    uv run python scripts/data/repair_silver.py --all --var-root <var> --budget-usd 6
    uv run python scripts/data/repair_silver.py --corpus enron --var-root <var> --no-calls
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from portolan.training.label_judge import (
    LABEL_JUDGE_SPEND_KEY,
    JudgeCall,
    JudgeFact,
    JudgeLogEntry,
    judge_facts,
    judge_model_allowed,
    judge_record,
    openrouter_judge_call,
)
from portolan.training.pipeline import (
    CORPUS_KINDS,
    CORPUS_LICENSES,
    assert_not_holdout,
    document_envelopes,
    documents_for_corpus,
    load_progress,
)
from portolan.training.records import (
    DroppedFact,
    TrainingRecord,
    dropped_path,
    label_judge_log_path,
    progress_path,
    records_path,
    repaired_records_path,
    repairs_log_path,
    silver_dir,
)
from portolan.training.repair import (
    FINAL_OUTCOMES,
    REPAIR_SCHEMA,
    EnvelopeLookup,
    FailedFact,
    RepairCall,
    RepairLogEntry,
    assemble_record,
    map_drops_to_records,
    plan_record,
    recover_unwritten_records,
    repair_fact,
    repair_key,
)
from portolan.training.spend import SpendMeter, default_spend_path
from portolan.util import clock

REPO_ROOT = Path(__file__).resolve().parents[2]
VAR_ROOT = REPO_ROOT / "var"

# Corpora whose records may train (build_training_set.py provenance tables);
# saas_sales and synthetic are excluded there, so they are not repaired.
DEFAULT_CORPORA: tuple[str, ...] = ("fda_maude", "cfpb", "enron", "ubuntu_irc", "ami")
SPEND_CORPUS = "silver_repair"


# --------------------------------------------------------------------------
# Snapshot
# --------------------------------------------------------------------------


def read_complete_lines(path: Path) -> bytes:
    """`path`'s bytes up to its last newline: a line still being appended
    has no newline yet."""
    if not path.exists():
        return b""
    data = path.read_bytes()
    return data[: data.rfind(b"\n") + 1]


@dataclass
class Snapshot:
    records: list[TrainingRecord]
    drops: list[DroppedFact]
    inputs: dict[str, dict[str, Any]]
    # read before the records: a source is marked processed only after all
    # its records are written, so each of these has every record it will get
    processed_sources: set[str] = field(default_factory=set)


def take_snapshot(var_root: Path, corpus: str, copy_to: Path | None = None) -> Snapshot:
    processed = _read_processed(progress_path(var_root, corpus))
    record_bytes = read_complete_lines(records_path(var_root, corpus))
    drop_bytes = read_complete_lines(dropped_path(var_root, corpus))
    inputs: dict[str, dict[str, Any]] = {}
    for name, data in (("records.jsonl", record_bytes), ("dropped.jsonl", drop_bytes)):
        inputs[name] = {
            "lines": sum(1 for line in data.splitlines() if line.strip()),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
        if copy_to is not None:
            target = silver_dir(copy_to, corpus) / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    records = [
        TrainingRecord.model_validate_json(line)
        for line in record_bytes.decode("utf-8").splitlines()
        if line.strip()
    ]
    drops = [
        DroppedFact.model_validate_json(line)
        for line in drop_bytes.decode("utf-8").splitlines()
        if line.strip()
    ]
    return Snapshot(records=records, drops=drops, inputs=inputs, processed_sources=processed)


def _read_processed(path: Path) -> set[str]:
    """The live run rewrites its progress file in place; a read that lands
    mid-write recovers nothing rather than guessing."""
    try:
        return load_progress(path)
    except (json.JSONDecodeError, OSError):
        return set()


def _envelope_lookup(var_root: Path, corpus: str) -> EnvelopeLookup:
    """Envelopes rebuilt from `var/data/converted/<corpus>` exactly as the
    live run builds them, read only when a source needs them."""
    documents: dict[str, Any] | None = None

    def lookup(source_id: str) -> list[tuple[str, str]]:
        nonlocal documents
        if documents is None:
            converted = var_root / "data" / "converted" / corpus
            assert_not_holdout(converted)
            documents = (
                {d.source_id: d for d in documents_for_corpus(corpus, converted)}
                if converted.exists()
                else {}
            )
        document = documents.get(source_id)
        if document is None:
            return []
        out = []
        for envelope in document_envelopes(document):
            seg = envelope.segment
            record_id = source_id if seg is None else f"{source_id}#seg{seg.index}of{seg.total}"
            out.append((record_id, envelope.text))
        return out

    return lookup


# --------------------------------------------------------------------------
# One corpus
# --------------------------------------------------------------------------


@dataclass
class CorpusResult:
    corpus: str
    records: int = 0
    drops: int = 0
    orphan_drops: int = 0
    recovered_records: int = 0
    kept_facts: int = 0
    regated_facts: int = 0
    failed_facts: int = 0
    failed_by_reason: Counter = field(default_factory=Counter)
    calls_made: int = 0
    cached: int = 0
    skipped_unlogged: int = 0
    outcomes: Counter = field(default_factory=Counter)
    repaired_facts: int = 0
    withdrawn_facts: int = 0
    unresolved_facts: int = 0
    unlogged_facts: int = 0
    complete_records: int = 0
    incomplete_records: int = 0
    incomplete_with_unlogged: int = 0
    spend_usd: float = 0.0
    inputs: dict[str, Any] = field(default_factory=dict)
    judge_model: str | None = None
    judge_calls: int = 0
    judge_cached: int = 0
    judge_skipped: int = 0
    judge_dropped: int = 0
    judge_retyped: int = 0
    judge_unresolved: int = 0
    judge_spend_usd: float = 0.0


def read_repair_log(path: Path) -> list[RepairLogEntry]:
    if not path.exists():
        return []
    return [
        RepairLogEntry.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_records_atomically(path: Path, records: list[TrainingRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(record.model_dump_json())
            f.write("\n")
    tmp.replace(path)


def repair_corpus(
    *,
    var_root: Path,
    out_root: Path,
    corpus: str,
    call: RepairCall | None,
    budget_left: Callable[[], bool],
    company_context: str | None,
    teacher_override: str | None = None,
    workers: int = 1,
    snapshot_copy_to: Path | None = None,
    now: Callable[[], str] = lambda: clock.now().isoformat(),
    judge_model: str | None = None,
    judge_call: JudgeCall | None = None,
    judge_budget_left: Callable[[], bool] = lambda: True,
) -> CorpusResult:
    result = CorpusResult(corpus=corpus)
    snap = take_snapshot(var_root, corpus, copy_to=snapshot_copy_to)
    result.inputs = snap.inputs
    result.records = len(snap.records)
    result.drops = len(snap.drops)
    mapping = map_drops_to_records(snap.records, snap.drops)
    recovered, still_orphaned = recover_unwritten_records(
        mapping.orphans,
        processed_sources=snap.processed_sources,
        existing_record_ids={r.record_id for r in snap.records},
        envelopes_for=_envelope_lookup(var_root, corpus),
        corpus=corpus,
        kind=CORPUS_KINDS[corpus],
        license_text=CORPUS_LICENSES[corpus],
    )
    result.recovered_records = len(recovered)
    result.orphan_drops = len(still_orphaned)

    log_path = repairs_log_path(out_root, corpus)
    cache = {e.key: e for e in read_repair_log(log_path) if e.outcome in FINAL_OUTCOMES}
    plans = [
        plan_record(record, mapping.by_record.get(record.record_id, [])) for record in snap.records
    ] + [plan_record(record, drops) for record, drops in recovered]

    pending: dict[str, tuple[FailedFact, str]] = {}
    for plan in plans:
        result.kept_facts += plan.kept
        result.regated_facts += plan.regated
        for failed in plan.failed:
            result.failed_facts += 1
            result.failed_by_reason[failed.verdict.reason or ""] += 1
            key = repair_key(failed.record_id, failed.fact)
            if key in cache:
                result.cached += 1
            elif plan.unlogged:
                # the record stays incomplete whatever the repair returns
                result.skipped_unlogged += 1
            elif key not in pending:
                if teacher_override:
                    failed = FailedFact(
                        failed.record_id,
                        failed.source_id,
                        teacher_override,
                        failed.origin,
                        failed.fact,
                        failed.verdict,
                    )
                pending[key] = (failed, plan.record.text)

    lock = threading.Lock()
    fresh: dict[str, RepairLogEntry] = {}

    def run_one(key: str) -> None:
        failed, text = pending[key]
        use_call = call if call is not None and budget_left() else None
        entry = repair_fact(
            failed,
            corpus=corpus,
            text=text,
            company_context=company_context,
            call=use_call,
            now=now(),
        )
        with lock:
            fresh[key] = entry
            if use_call is not None:
                result.calls_made += 1
                result.spend_usd += entry.cost_usd
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(entry.model_dump_json())
                f.write("\n")

    if workers > 1 and len(pending) > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(run_one, list(pending)))
    else:
        for key in pending:
            run_one(key)

    audited: list[TrainingRecord] = []
    for plan in plans:
        entries = []
        for failed in plan.failed:
            key = repair_key(failed.record_id, failed.fact)
            entry = cache.get(key) or fresh.get(key)
            if entry is not None:
                entries.append(entry)
                result.outcomes[entry.outcome] += 1
        record = assemble_record(plan, entries, audited=now())
        audit = record.label_audit
        assert audit is not None
        result.repaired_facts += audit.repaired
        result.withdrawn_facts += audit.withdrawn
        result.unresolved_facts += audit.unresolved
        result.unlogged_facts += audit.unlogged
        if audit.status == "complete":
            result.complete_records += 1
        else:
            result.incomplete_records += 1
            if audit.unlogged:
                result.incomplete_with_unlogged += 1
        audited.append(record)

    if judge_model is not None:
        audited = _judge_records(
            audited,
            result=result,
            corpus=corpus,
            out_root=out_root,
            model=judge_model,
            call=judge_call,
            budget_left=judge_budget_left,
            company_context=company_context,
            workers=workers,
            now=now,
        )

    _write_records_atomically(repaired_records_path(out_root, corpus), audited)
    summary = asdict(result)
    summary["failed_by_reason"] = dict(result.failed_by_reason)
    summary["outcomes"] = dict(result.outcomes)
    summary["created"] = now()
    (silver_dir(out_root, corpus) / "repair_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return result


def read_judge_log(path: Path) -> list[JudgeLogEntry]:
    if not path.exists():
        return []
    return [
        JudgeLogEntry.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _judge_records(
    records: list[TrainingRecord],
    *,
    result: CorpusResult,
    corpus: str,
    out_root: Path,
    model: str,
    call: JudgeCall | None,
    budget_left: Callable[[], bool],
    company_context: str | None,
    workers: int,
    now: Callable[[], str],
) -> list[TrainingRecord]:
    """The label judge over every record with facts (one call per record,
    cached in `label_judge.jsonl`): drops and retypes applied by
    `judge_record`; a fact left without a verdict makes its record
    incomplete, and a rerun retries it."""
    result.judge_model = model
    log_path = label_judge_log_path(out_root, corpus)
    cache = {e.key: e for e in read_judge_log(log_path) if e.error is None}
    lock = threading.Lock()
    judged: dict[str, TrainingRecord] = {}

    def run_one(record: TrainingRecord) -> None:
        facts = [JudgeFact(f.fact_type, f.claim, f.quote or "") for f in record.facts]
        inputs = record.user_message_inputs or {}
        judgement, how = judge_facts(
            record.text,
            facts,
            model=model,
            call=call,
            company_context=company_context,
            source_date=inputs.get("source_date"),
            cache=cache,
            budget_left=budget_left,
        )
        new = judge_record(record, judgement)
        with lock:
            judged[record.record_id] = new
            result.judge_calls += how == "called"
            result.judge_cached += how == "cached"
            result.judge_skipped += how == "skipped"
            result.judge_spend_usd += judgement.cost_usd
            if how == "called":
                entry = JudgeLogEntry.from_result(
                    judgement, subject=record.record_id, facts=facts, created=now()
                )
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(entry.model_dump_json())
                    f.write("\n")

    todo = [r for r in records if r.facts]
    if workers > 1 and len(todo) > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(run_one, todo))
    else:
        for record in todo:
            run_one(record)

    out = []
    for record in records:
        new = judged.get(record.record_id, record)
        audit = new.label_audit
        if record.facts and audit is not None:
            result.judge_dropped += audit.judge_dropped
            result.judge_retyped += audit.judge_retyped
            result.judge_unresolved += audit.judge_unresolved
            if audit.judge_unresolved and record.label_audit.status == "complete":
                result.complete_records -= 1
                result.incomplete_records += 1
        out.append(new)
    return out


# --------------------------------------------------------------------------
# The paid call
# --------------------------------------------------------------------------


def openrouter_call(api_key: str, spend: SpendMeter) -> RepairCall:
    """One `OpenRouterClient` per worker thread; each call's real cost
    (the sink's `cost_estimate`) goes into `spend` under `silver_repair`,
    failed calls included."""
    from portolan.llm.openrouter import OpenRouterClient
    from portolan.metering import InMemorySink

    local = threading.local()

    def call(model: str, system: str, user: str) -> tuple[dict[str, Any] | None, float]:
        client = getattr(local, "client", None)
        if client is None:
            client = OpenRouterClient(
                spend_key=SPEND_CORPUS,
                api_key=api_key,
                default_model=model,
                sink=InMemorySink(),
                max_retries=3,
            )
            local.client = client
        before = len(client.sink.events)
        cost = 0.0
        try:
            result = client.complete(
                system=system,
                user=user,
                json_schema=REPAIR_SCHEMA,
                model=model,
                surface="extract",
                max_tokens=1500,
            )
        finally:
            cost = sum((e.cost_estimate or 0.0) for e in client.sink.events[before:])
            spend.record(model=model, corpus=SPEND_CORPUS, amount_usd=cost)
        return result.data, cost

    return call


def _company_contexts() -> dict[str, str]:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "silver_contexts_for_repair", Path(__file__).resolve().parent / "silver_contexts.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return dict(module.CORPUS_CONTEXTS)


def _print_result(r: CorpusResult) -> None:
    print(
        f"[{r.corpus}] records={r.records} drops={r.drops} orphan_drops={r.orphan_drops} "
        f"recovered_records={r.recovered_records} "
        f"kept={r.kept_facts} regated={r.regated_facts} failed={r.failed_facts} "
        f"{dict(r.failed_by_reason)} calls={r.calls_made} cached={r.cached} "
        f"skipped_unlogged={r.skipped_unlogged} "
        f"outcomes={dict(r.outcomes)} repaired={r.repaired_facts} "
        f"withdrawn={r.withdrawn_facts} unresolved={r.unresolved_facts} "
        f"unlogged={r.unlogged_facts} complete={r.complete_records} "
        f"incomplete={r.incomplete_records} (unlogged {r.incomplete_with_unlogged}) "
        f"spend_usd={r.spend_usd:.4f}"
    )
    if r.judge_model:
        print(
            f"[{r.corpus}] label judge {r.judge_model}: calls={r.judge_calls} "
            f"cached={r.judge_cached} skipped={r.judge_skipped} dropped={r.judge_dropped} "
            f"retyped={r.judge_retyped} unresolved={r.judge_unresolved} "
            f"spend_usd={r.judge_spend_usd:.4f}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", action="append", dest="corpora", choices=DEFAULT_CORPORA)
    parser.add_argument("--all", action="store_true", help=f"{', '.join(DEFAULT_CORPORA)}")
    parser.add_argument(
        "--var-root", type=Path, default=VAR_ROOT, help="var/ holding the silver data"
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="var/ to write outputs under (default: --var-root)",
    )
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=None,
        help="also copy the snapshotted records.jsonl and dropped.jsonl under this var/-shaped dir",
    )
    parser.add_argument(
        "--budget-usd", type=float, default=6.0, help="cap on the silver_repair key"
    )
    parser.add_argument("--no-calls", action="store_true", help="re-gate only; no paid calls")
    parser.add_argument(
        "--teacher", default=None, help="repair with this model instead of each record's teacher"
    )
    parser.add_argument(
        "--label-judge",
        default=None,
        metavar="MODEL",
        help="judge every audited record's facts with this open-weight model "
        "(portolan.training.label_judge; e.g. deepseek/deepseek-v4.1-flash); off by default",
    )
    parser.add_argument(
        "--judge-budget-usd",
        type=float,
        default=3.0,
        help="cap on this run's growth of the label_judge spend key",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-calls", type=int, default=None, help="stop calling after this many")
    parser.add_argument(
        "--spend-file",
        type=Path,
        default=None,
        help="the shared spend meter (default: <var-root>/data/silver/spend.json)",
    )
    parser.add_argument(
        "--env-root",
        type=Path,
        default=REPO_ROOT,
        help="directory whose .env holds OPENROUTER_API_KEY when the environment does not",
    )
    args = parser.parse_args(argv)
    if bool(args.corpora) == bool(args.all):
        print("error: pass --corpus NAME (repeatable) or --all", file=sys.stderr)
        return 2
    corpora = list(DEFAULT_CORPORA) if args.all else args.corpora
    out_root = args.out_root or args.var_root
    for path in (args.var_root, out_root, args.snapshot_dir):
        if path is not None:
            assert_not_holdout(path)

    if args.label_judge and not judge_model_allowed(args.label_judge):
        print(
            f"error: --label-judge {args.label_judge} is not on the training-set allowlist",
            file=sys.stderr,
        )
        return 2
    spend = SpendMeter(path=args.spend_file or default_spend_path(args.var_root))
    call: RepairCall | None = None
    judge_call: JudgeCall | None = None
    if not args.no_calls:
        from corpus.generator.prose import _env_or_dotenv

        api_key = _env_or_dotenv(args.env_root, "OPENROUTER_API_KEY")
        if not api_key:
            print(
                "error: OPENROUTER_API_KEY not in the environment or --env-root/.env",
                file=sys.stderr,
            )
            return 2
        call = openrouter_call(api_key, spend)
        if args.label_judge:
            judge_call = openrouter_judge_call(api_key, spend)

    judge_start = spend.corpus_usd(LABEL_JUDGE_SPEND_KEY)

    def judge_budget_left() -> bool:
        grown = spend.corpus_usd(LABEL_JUDGE_SPEND_KEY) - judge_start
        return grown < args.judge_budget_usd

    calls_started = 0
    calls_lock = threading.Lock()

    def budget_left() -> bool:
        nonlocal calls_started
        with calls_lock:
            if args.max_calls is not None and calls_started >= args.max_calls:
                return False
            if spend.is_corpus_over_budget(SPEND_CORPUS, args.budget_usd):
                return False
            calls_started += 1
            return True

    print(
        f"silver_repair spend before: ${spend.corpus_usd(SPEND_CORPUS):.4f} of ${args.budget_usd}"
    )
    contexts = _company_contexts()
    for corpus in corpora:
        result = repair_corpus(
            var_root=args.var_root,
            out_root=out_root,
            corpus=corpus,
            call=call,
            budget_left=budget_left,
            company_context=contexts.get(corpus),
            teacher_override=args.teacher,
            workers=args.workers,
            snapshot_copy_to=args.snapshot_dir,
            judge_model=args.label_judge,
            judge_call=judge_call,
            judge_budget_left=judge_budget_left,
        )
        _print_result(result)
    print(f"silver_repair spend after: ${spend.corpus_usd(SPEND_CORPUS):.4f} of ${args.budget_usd}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
