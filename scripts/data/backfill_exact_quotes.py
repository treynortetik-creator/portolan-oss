"""Backfill exact quote spans into an EXISTING `records.jsonl`
(feature/labeling-exact-quotes, build instructions item 3). The
orchestrator review in `01-spec/training-data-2026-09-19.md`
(2026-09-22 06:10 MST) found 48 of 280 kept Enron facts and 2 of 31 AMI
facts whose `quote` is not an exact substring of the record `text`
(some stitch two spans with `"..."`), another 30 differing only by
whitespace/case/MIME artifacts -- all written before exact-span
resolution existed (`portolan.training.pipeline.run_corpus_single`,
`portolan.training.quote_span.resolve_quote_span`, same fix). This
script re-resolves every fact already on disk the same way a live run
now does, so the paused labeling runs can resume against a clean
baseline instead of restarting from scratch.

**What it does, per corpus:**

1. Read every record from `var/data/silver/<corpus>/records.jsonl`.
2. Copy that file, unchanged, to `records.jsonl.bak-<timestamp>` --
   made every run (not just the first), so re-running after a bad edit
   still leaves a fresh recovery point.
3. For every fact: if it already carries a valid `quote_span`
   (`text[start:end] == quote`) leave it untouched -- the fast path that
   makes a second run over already-backfilled data a no-op (idempotent).
   Otherwise resolve it (`resolve_quote_span`); a fact that resolves gets
   its `quote`/`quote_span` rewritten to the exact span; a fact that
   doesn't is DROPPED and logged to `dropped.jsonl` with reason
   `"quote_not_exact_backfill"`.
4. A record left with zero facts after step 3 is removed from the
   rewritten file entirely, UNLESS it was already a negative (zero facts)
   BEFORE backfill -- a real "the model correctly found nothing" example
   stays; a record that lost every fact to unresolvable quotes is not
   silently turned into one.
5. Overwrite `records.jsonl` with the surviving, resolved records.

Idempotent: running this twice back to back on the same corpus produces
identical output the second time (nothing left to resolve, nothing left
to drop) -- exercised directly in
`tests/test_backfill_exact_quotes.py::test_backfill_is_idempotent`.

Usage:
    uv run python scripts/data/backfill_exact_quotes.py --corpus enron
    uv run python scripts/data/backfill_exact_quotes.py --all
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from portolan.training.pipeline import assert_not_holdout
from portolan.training.quote_span import resolve_quote_span
from portolan.training.records import (
    CORPORA,
    DroppedFact,
    TrainingFact,
    TrainingRecord,
    append_dropped,
    dropped_path,
    read_records,
    records_path,
)
from portolan.util import clock

REPO_ROOT = Path(__file__).resolve().parents[2]
VAR_ROOT = REPO_ROOT / "var"

# The four real corpora this backfill has ever had records for
# (`01-spec/training-data-2026-09-19.md`'s silver-labeling section).
# `synthetic` is deliberately excluded from `--all`: its quotes are
# already exact by construction (that same doc's synthetic-run entries:
# "every quote a substring of its text") and it has no `quote_span` field
# on disk yet either -- `--corpus synthetic` still runs the same
# resolver over it if ever needed, harmlessly (the fast path won't fire
# since there's no existing span, but every quote resolves at tier
# "exact").
DEFAULT_CORPORA: tuple[str, ...] = ("fda_maude", "ami", "enron", "ubuntu_irc")

_RAW_FACT_FIELDS = ("fact_type", "claim", "quote", "locator", "entities")


@dataclass
class BackfillResult:
    corpus: str
    records_before: int = 0
    records_after: int = 0
    facts_before: int = 0
    facts_after: int = 0
    facts_already_exact: int = 0
    facts_newly_resolved: int = 0
    facts_dropped: int = 0
    records_dropped: int = 0
    backup_path: Path | None = None


def _raw_fact_dict(fact: TrainingFact) -> dict:
    return {name: getattr(fact, name) for name in _RAW_FACT_FIELDS}


def _already_resolved(fact: TrainingFact, text: str) -> bool:
    """The idempotent fast path: a fact whose `quote_span` already names
    the exact substring it claims needs no re-resolution."""
    if fact.quote is None or fact.quote_span is None:
        return False
    start, end = fact.quote_span
    if not (0 <= start < end <= len(text)):
        return False
    return text[start:end] == fact.quote


def resolve_record_facts(
    record: TrainingRecord,
) -> tuple[list[TrainingFact], list[TrainingFact], int]:
    """Pure per-record step (no I/O): `(kept_facts, dropped_facts,
    already_exact_count)`. `kept_facts` carries the exact-resolved
    `quote`/`quote_span` for every fact that resolves (or was already
    resolved); `dropped_facts` is every fact that couldn't be, in its
    original (unresolved) form, for the caller to log."""
    kept: list[TrainingFact] = []
    dropped: list[TrainingFact] = []
    already_exact = 0

    for fact in record.facts:
        if _already_resolved(fact, record.text):
            kept.append(fact)
            already_exact += 1
            continue

        resolved = resolve_quote_span(fact.quote, record.text)
        if resolved is None:
            dropped.append(fact)
            continue

        assert record.text[resolved.span[0] : resolved.span[1]] == resolved.quote
        kept.append(fact.model_copy(update={"quote": resolved.quote, "quote_span": resolved.span}))

    return kept, dropped, already_exact


def backfill_records_file(
    records_file: Path,
    dropped_file: Path,
    *,
    corpus: str,
    now: Callable[[], str] = lambda: clock.now().isoformat(),
    make_backup: bool = True,
) -> BackfillResult:
    """Read `records_file` in full, resolve every fact's quote span, log
    every unresolvable fact to `dropped_file`, drop any record left with
    zero facts that wasn't already a negative, and overwrite
    `records_file` with the result -- after copying it, unmodified, to
    `records_file.bak-<timestamp>` first (`make_backup=False` only for
    tests that want to skip that I/O; the real CLI never does)."""
    result = BackfillResult(corpus=corpus)
    if not records_file.exists():
        return result

    original_records = list(read_records(records_file))
    result.records_before = len(original_records)
    result.facts_before = sum(len(r.facts) for r in original_records)

    backup_path: Path | None = None
    if make_backup:
        # Filesystem-safe timestamp: strip everything but digits/letters
        # from the ISO `now()` string ("2026-09-22T06:10:00-07:00" ->
        # "20260922T061000Z0700") rather than assuming any particular
        # ISO shape.
        timestamp = re.sub(r"[^0-9A-Za-z]", "", now())
        backup_path = records_file.with_name(f"{records_file.name}.bak-{timestamp}")
        shutil.copy2(records_file, backup_path)
        result.backup_path = backup_path

    kept_records: list[TrainingRecord] = []
    for record in original_records:
        was_negative = record.is_negative
        kept_facts, dropped_facts, already_exact = resolve_record_facts(record)
        result.facts_already_exact += already_exact
        result.facts_newly_resolved += len(kept_facts) - already_exact

        for fact in dropped_facts:
            teacher = record.teachers[0] if record.teachers else ""
            append_dropped(
                dropped_file,
                DroppedFact(
                    source_id=record.source_id,
                    reason="quote_not_exact_backfill",
                    fact=_raw_fact_dict(fact),
                    teacher=teacher,
                    created=now(),
                    prompt_variant=record.prompt_variant,
                    system_prompt_sha256=record.system_prompt_sha256,
                ),
            )
            result.facts_dropped += 1

        if not kept_facts and not was_negative:
            result.records_dropped += 1
            continue

        kept_records.append(record.model_copy(update={"facts": kept_facts}))

    with records_file.open("w", encoding="utf-8") as f:
        for record in kept_records:
            f.write(record.model_dump_json())
            f.write("\n")

    result.records_after = len(kept_records)
    result.facts_after = sum(len(r.facts) for r in kept_records)
    return result


def _print_table(results: list[BackfillResult]) -> None:
    header = (
        f"{'corpus':<12} {'records':>16} {'facts':>16} "
        f"{'resolved':>10} {'dropped':>8} {'rec_dropped':>12}"
    )
    print(header)
    print(f"{'':<12} {'before->after':>16} {'before->after':>16}")
    for r in results:
        records_str = f"{r.records_before}->{r.records_after}"
        facts_str = f"{r.facts_before}->{r.facts_after}"
        print(
            f"{r.corpus:<12} {records_str:>16} {facts_str:>16} "
            f"{r.facts_newly_resolved:>10} {r.facts_dropped:>8} {r.records_dropped:>12}"
        )
    total_before_r = sum(r.records_before for r in results)
    total_after_r = sum(r.records_after for r in results)
    total_before_f = sum(r.facts_before for r in results)
    total_after_f = sum(r.facts_after for r in results)
    total_resolved = sum(r.facts_newly_resolved for r in results)
    total_dropped = sum(r.facts_dropped for r in results)
    total_rec_dropped = sum(r.records_dropped for r in results)
    print(
        f"{'TOTAL':<12} {f'{total_before_r}->{total_after_r}':>16} "
        f"{f'{total_before_f}->{total_after_f}':>16} "
        f"{total_resolved:>10} {total_dropped:>8} {total_rec_dropped:>12}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus", action="append", dest="corpora", choices=CORPORA, help="corpus to backfill"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help=f"backfill every real corpus with records on disk: {', '.join(DEFAULT_CORPORA)}",
    )
    parser.add_argument(
        "--var-root",
        type=Path,
        default=VAR_ROOT,
        help="repo var/ directory (default: the repo root's var/)",
    )
    args = parser.parse_args(argv)

    if not args.corpora and not args.all:
        print("error: pass --corpus NAME (repeatable) or --all", file=sys.stderr)
        return 2
    if args.corpora and args.all:
        print("error: pass --corpus or --all, not both", file=sys.stderr)
        return 2

    corpora = args.corpora if args.corpora else list(DEFAULT_CORPORA)

    silver_root = args.var_root / "data" / "silver"
    assert_not_holdout(silver_root)

    results: list[BackfillResult] = []
    for corpus in corpora:
        records_file = records_path(args.var_root, corpus)
        dropped_file = dropped_path(args.var_root, corpus)
        result = backfill_records_file(records_file, dropped_file, corpus=corpus)
        if result.records_before == 0 and not records_file.exists():
            print(f"[{corpus}] no records.jsonl on disk, skipped")
            continue
        results.append(result)
        print(
            f"[{corpus}] records {result.records_before}->{result.records_after} "
            f"facts {result.facts_before}->{result.facts_after} "
            f"already_exact={result.facts_already_exact} "
            f"newly_resolved={result.facts_newly_resolved} "
            f"dropped_facts={result.facts_dropped} "
            f"dropped_records={result.records_dropped} "
            f"backup={result.backup_path}"
        )

    if results:
        print()
        _print_table(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
