"""Re-admit facts dropped by the (now-relaxed) overlap gate
(fix/overlap-gate-paraphrase, 2026-09-22). This rewrites the live
`records.jsonl` and `dropped.jsonl`, so never run it while a labeling run
appends to them; `scripts/data/repair_silver.py` re-gates and repairs into
new files instead. `scripts/data/silver_label.py
--mode single` logs every dropped fact to `dropped.jsonl`
(`portolan.training.records.DroppedFact`). Disabling the content-overlap
rule by default and normalizing the name rule
(`portolan.training.overlap_gate`) means a fact dropped under the OLD
gate may pass under the CURRENT one -- this script re-evaluates every
such drop and, for the ones that now pass, gets the fact into
`records.jsonl` without re-paying for a teacher call.

**What it does, per corpus:**

1. Read every entry from `var/data/silver/<corpus>/dropped.jsonl` whose
   `reason` starts with `"overlap_gate"` (`quote_gate` and
   `quote_not_exact*` entries failed a check this fix does not touch --
   left alone) and has no `regated_at` yet (the idempotency marker --
   see below).
2. For each one, rebuild the document it came from EXACTLY the way the
   live pipeline does (`portolan.training.pipeline.documents_for_corpus`
   -> `document_envelopes`, same defaults) and find the envelope whose
   text actually supports the dropped fact's quote
   (`portolan.extract.quote_gate.quote_is_supported` -- true by
   construction for every `overlap_gate_*` reason, since the overlap
   gate only ever runs after the quote gate passes). A `DroppedFact`
   only carries the document's bare `source_id`, never a long
   document's `#segN` segment suffix -- `run_corpus_single` logs a drop
   before it knows how many envelopes the document will have.
3. Re-run the CURRENT gate (`overlap_gate_reason`) against that
   envelope's text. Still fails -> stays dropped. Passes -> resolve an
   exact quote span (`resolve_quote_span`); resolves -> the fact is
   ADMITTED, `quote`/`quote_span` rewritten to the exact span, `regated:
   true` set, and appended to the document's record in `records.jsonl`
   (creating one when the document had no kept facts at all, updating it
   otherwise); doesn't resolve -> stays dropped (the same "supported but
   not exact" case `run_corpus_single` itself would have dropped as
   `quote_not_exact`).
4. Every entry this script LOOKED AT (admitted or not, envelope match or
   not) gets `regated_at` stamped in `dropped.jsonl` -- so a second run
   over the same corpus finds nothing left to evaluate (idempotent).
5. `records.jsonl` is copied to `records.jsonl.bak-<timestamp>` before
   being rewritten -- only when at least one fact was actually admitted
   (nothing to protect otherwise). `dropped.jsonl` is rewritten in place
   (no backup -- every entry's own content is unchanged, only
   `regated_at` is added).

Usage:
    uv run python scripts/data/regate_dropped.py --corpus saas_sales
    uv run python scripts/data/regate_dropped.py --corpus saas_sales --corpus enron
    uv run python scripts/data/regate_dropped.py --all
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from portolan.extract.quote_gate import quote_is_supported
from portolan.training.overlap_gate import overlap_gate_reason
from portolan.training.pipeline import (
    CORPUS_KINDS,
    CORPUS_LICENSES,
    DEFAULT_CHAT_WINDOW_SIZE,
    Document,
    assert_not_holdout,
    document_envelopes,
    documents_for_corpus,
)
from portolan.training.quote_span import resolve_quote_span
from portolan.training.records import (
    TEACHER_FACT_FIELDS,
    DroppedFact,
    TrainingFact,
    TrainingRecord,
    dropped_path,
    read_dropped,
    read_records,
    records_path,
    write_dropped,
)
from portolan.util import clock

REPO_ROOT = Path(__file__).resolve().parents[2]
VAR_ROOT = REPO_ROOT / "var"

# Every real corpus `run_corpus_single` (the only run mode that writes
# `dropped.jsonl`) can produce a drop for -- `pipeline.CORPUS_LICENSES`'s
# own keys, not `portolan.training.records.CORPORA` (that tuple predates
# `cfpb`/`saas_sales` and would reject them as an argparse choice).
DEFAULT_CORPORA: tuple[str, ...] = tuple(CORPUS_LICENSES.keys())

_RAW_FACT_FIELDS = ("fact_type", "claim", "quote", "locator", "entities")


@dataclass
class RegateResult:
    corpus: str
    dropped_entries_total: int = 0
    dropped_entries_eligible: int = 0
    dropped_entries_already_regated: int = 0
    facts_readmitted: int = 0
    facts_still_dropped: int = 0
    facts_no_envelope_match: int = 0
    records_created: int = 0
    records_updated: int = 0
    records_before: int = 0
    records_after: int = 0
    facts_before: int = 0
    facts_after: int = 0
    backup_path: Path | None = None


def _document_index(corpus: str, var_root: Path, *, chat_window: int) -> dict[str, Document]:
    """Every `Document` this corpus's `documents_for_corpus` yields,
    keyed by `source_id` -- built once per corpus so re-gating several
    drops from the same document doesn't re-read/re-window the source
    more than once."""
    converted_root = var_root / "data" / "converted" / corpus
    assert_not_holdout(converted_root)
    return {
        doc.source_id: doc
        for doc in documents_for_corpus(corpus, converted_root, window_size=chat_window)
    }


def _record_id_for(source_id: str, envelope: Any) -> str:
    if envelope.segment is None:
        return source_id
    seg = envelope.segment
    return f"{source_id}#seg{seg.index}of{seg.total}"


def _find_matching_envelope(quote: str | None, envelopes: list[Any]) -> Any | None:
    """The envelope, among `envelopes`, whose text actually supports
    `quote` -- true by construction for every `overlap_gate_*` drop
    reason (the overlap gate only ever runs after the quote gate
    passes), so this should always find exactly one match UNLESS the
    converted corpus file on disk changed since the original run. The
    first match wins on the rare chance more than one envelope's text
    happens to contain the same quote."""
    for envelope in envelopes:
        if quote_is_supported(quote, envelope.text):
            return envelope
    return None


def regate_corpus(
    var_root: Path,
    corpus: str,
    *,
    overlap_threshold: float,
    chat_window: int = DEFAULT_CHAT_WINDOW_SIZE,
    now: Callable[[], str] = lambda: clock.now().isoformat(),
    make_backup: bool = True,
) -> RegateResult:
    result = RegateResult(corpus=corpus)

    dropped_file = dropped_path(var_root, corpus)
    original_dropped = list(read_dropped(dropped_file))
    result.dropped_entries_total = len(original_dropped)
    if not original_dropped:
        records_file = records_path(var_root, corpus)
        existing = list(read_records(records_file))
        result.records_before = result.records_after = len(existing)
        result.facts_before = result.facts_after = sum(len(r.facts) for r in existing)
        return result

    records_file = records_path(var_root, corpus)
    existing_records = list(read_records(records_file))
    result.records_before = len(existing_records)
    result.facts_before = sum(len(r.facts) for r in existing_records)

    records_by_id: dict[str, TrainingRecord] = {r.record_id: r for r in existing_records}
    record_order: list[str] = [r.record_id for r in existing_records]

    envelope_cache: dict[str, list[Any] | None] = {}
    documents: dict[str, Document] | None = None

    def envelopes_for(source_id: str) -> list[Any] | None:
        nonlocal documents
        if source_id in envelope_cache:
            return envelope_cache[source_id]
        if documents is None:
            documents = _document_index(corpus, var_root, chat_window=chat_window)
        document = documents.get(source_id)
        envs = list(document_envelopes(document)) if document is not None else None
        envelope_cache[source_id] = envs
        return envs

    result.dropped_entries_already_regated = sum(
        1
        for e in original_dropped
        if e.reason.startswith("overlap_gate") and e.regated_at is not None
    )

    updated_dropped: list[DroppedFact] = []
    any_regated = False

    for entry in original_dropped:
        if entry.regated_at is not None or not entry.reason.startswith("overlap_gate"):
            updated_dropped.append(entry)
            continue

        result.dropped_entries_eligible += 1
        any_regated = True
        raw_fact = entry.fact
        quote = raw_fact.get("quote")

        envelopes = envelopes_for(entry.source_id)
        matched_envelope = _find_matching_envelope(quote, envelopes) if envelopes else None

        if matched_envelope is None:
            result.facts_no_envelope_match += 1
            updated_dropped.append(entry.model_copy(update={"regated_at": now()}))
            continue

        reason = overlap_gate_reason(
            claim=raw_fact.get("claim", ""),
            quote=quote,
            doc_text=matched_envelope.text,
            threshold=overlap_threshold,
        )
        if reason is not None:
            result.facts_still_dropped += 1
            updated_dropped.append(entry.model_copy(update={"regated_at": now()}))
            continue

        resolved = resolve_quote_span(quote, matched_envelope.text)
        if resolved is None:
            result.facts_still_dropped += 1
            updated_dropped.append(entry.model_copy(update={"regated_at": now()}))
            continue

        assert matched_envelope.text[resolved.span[0] : resolved.span[1]] == resolved.quote

        admitted = TrainingFact(
            fact_type=raw_fact.get("fact_type", "other"),
            claim=raw_fact.get("claim", ""),
            quote=resolved.quote,
            quote_span=resolved.span,
            locator=raw_fact.get("locator", ""),
            entities=list(raw_fact.get("entities") or []),
            judge=None,
            regated=True,
            **{k: raw_fact[k] for k in TEACHER_FACT_FIELDS if k in raw_fact},
        )

        record_id = _record_id_for(entry.source_id, matched_envelope)
        existing_record = records_by_id.get(record_id)
        if existing_record is not None:
            records_by_id[record_id] = existing_record.model_copy(
                update={"facts": [*existing_record.facts, admitted]}
            )
            result.records_updated += 1
        else:
            records_by_id[record_id] = TrainingRecord(
                record_id=record_id,
                corpus=corpus,
                source_id=entry.source_id,
                kind=CORPUS_KINDS[corpus],
                license=CORPUS_LICENSES[corpus],
                text=matched_envelope.text,
                facts=[admitted],
                teachers=[entry.teacher],
                # No sibling facts survive to compute a real agreement
                # ratio against (a record created here has exactly the
                # one re-admitted fact) -- 1.0 is honest for this record
                # in isolation, not a claim about the original envelope's
                # full emitted-fact count.
                agreement=1.0,
                created=now(),
                mode="single",
                # No teacher_answer: the drop log keeps single facts, not
                # the teacher's summary, so sft_format refuses this record.
                prompt_variant=entry.prompt_variant,
                system_prompt_sha256=entry.system_prompt_sha256,
                user_message_inputs=matched_envelope.model_dump(mode="json", exclude={"text"}),
            )
            record_order.append(record_id)
            result.records_created += 1

        result.facts_readmitted += 1
        updated_dropped.append(entry.model_copy(update={"regated_at": now()}))

    if any_regated:
        write_dropped(dropped_file, updated_dropped)

    if result.facts_readmitted > 0:
        if make_backup and records_file.exists():
            timestamp = re.sub(r"[^0-9A-Za-z]", "", now())
            backup_path = records_file.with_name(f"{records_file.name}.bak-{timestamp}")
            shutil.copy2(records_file, backup_path)
            result.backup_path = backup_path

        final_records = [records_by_id[rid] for rid in record_order]
        records_file.parent.mkdir(parents=True, exist_ok=True)
        with records_file.open("w", encoding="utf-8") as f:
            for record in final_records:
                f.write(record.model_dump_json())
                f.write("\n")

        result.records_after = len(final_records)
        result.facts_after = sum(len(r.facts) for r in final_records)
    else:
        result.records_after = result.records_before
        result.facts_after = result.facts_before

    return result


def _print_table(results: list[RegateResult]) -> None:
    header = (
        f"{'corpus':<12} {'records':>16} {'facts':>16} "
        f"{'eligible':>9} {'readmit':>8} {'still_drop':>10} {'no_env':>7}"
    )
    print(header)
    print(f"{'':<12} {'before->after':>16} {'before->after':>16}")
    for r in results:
        records_str = f"{r.records_before}->{r.records_after}"
        facts_str = f"{r.facts_before}->{r.facts_after}"
        print(
            f"{r.corpus:<12} {records_str:>16} {facts_str:>16} "
            f"{r.dropped_entries_eligible:>9} {r.facts_readmitted:>8} "
            f"{r.facts_still_dropped:>10} {r.facts_no_envelope_match:>7}"
        )
    total_before_r = sum(r.records_before for r in results)
    total_after_r = sum(r.records_after for r in results)
    total_before_f = sum(r.facts_before for r in results)
    total_after_f = sum(r.facts_after for r in results)
    total_eligible = sum(r.dropped_entries_eligible for r in results)
    total_readmit = sum(r.facts_readmitted for r in results)
    total_still = sum(r.facts_still_dropped for r in results)
    total_no_env = sum(r.facts_no_envelope_match for r in results)
    print(
        f"{'TOTAL':<12} {f'{total_before_r}->{total_after_r}':>16} "
        f"{f'{total_before_f}->{total_after_f}':>16} "
        f"{total_eligible:>9} {total_readmit:>8} {total_still:>10} {total_no_env:>7}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        action="append",
        dest="corpora",
        choices=DEFAULT_CORPORA,
        help="corpus to re-gate (repeatable)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help=f"re-gate every corpus that can have a dropped.jsonl: {', '.join(DEFAULT_CORPORA)}",
    )
    parser.add_argument(
        "--var-root",
        type=Path,
        default=VAR_ROOT,
        help="repo var/ directory (default: the repo root's var/)",
    )
    parser.add_argument(
        "--overlap-threshold",
        type=float,
        default=None,
        help="mechanical gate threshold to re-evaluate against "
        "(default: scripts/data/silver_label.py's current DEFAULT_OVERLAP_THRESHOLD)",
    )
    parser.add_argument(
        "--chat-window",
        type=int,
        default=DEFAULT_CHAT_WINDOW_SIZE,
        help="ubuntu_irc only: messages per rebuilt window (must match the live run's own "
        f"--chat-window to reproduce identical envelopes; default {DEFAULT_CHAT_WINDOW_SIZE})",
    )
    args = parser.parse_args(argv)

    if not args.corpora and not args.all:
        print("error: pass --corpus NAME (repeatable) or --all", file=sys.stderr)
        return 2
    if args.corpora and args.all:
        print("error: pass --corpus or --all, not both", file=sys.stderr)
        return 2

    corpora = args.corpora if args.corpora else list(DEFAULT_CORPORA)

    if args.overlap_threshold is not None:
        overlap_threshold = args.overlap_threshold
    else:
        # Imported lazily (not at module load) so this script never
        # fails to even parse --help when silver_label.py's own network-
        # touching imports (OpenRouter client, etc.) are unavailable --
        # the same reason scripts/data/overlap_calibration.py imports
        # from judge_calibration lazily-by-module rather than duplicating
        # the constant.
        from scripts.data.silver_label import DEFAULT_OVERLAP_THRESHOLD

        overlap_threshold = DEFAULT_OVERLAP_THRESHOLD

    silver_root = args.var_root / "data" / "silver"
    assert_not_holdout(silver_root)

    results: list[RegateResult] = []
    for corpus in corpora:
        result = regate_corpus(
            args.var_root, corpus, overlap_threshold=overlap_threshold, chat_window=args.chat_window
        )
        results.append(result)
        print(
            f"[{corpus}] dropped_total={result.dropped_entries_total} "
            f"eligible={result.dropped_entries_eligible} "
            f"already_regated={result.dropped_entries_already_regated} "
            f"readmitted={result.facts_readmitted} "
            f"still_dropped={result.facts_still_dropped} "
            f"no_envelope_match={result.facts_no_envelope_match} "
            f"records {result.records_before}->{result.records_after} "
            f"facts {result.facts_before}->{result.facts_after} "
            f"backup={result.backup_path}"
        )

    if results:
        print()
        _print_table(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
