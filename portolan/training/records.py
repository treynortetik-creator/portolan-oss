"""`TrainingRecord`: one silver-labeled fine-tuning example (silver-labels
build instructions item 3).

A record pairs the exact document text a teacher pair saw (chunked the
way `portolan.pipeline.build_envelopes` chunks any source -- see
`portolan.training.pipeline`) with the facts both teachers agreed on
(`portolan.training.agreement`), in the same shape the production
extractor emits: `fact_type`, `claim`, verbatim `quote`, `locator`, and
entity mentions. One JSONL line per record under
`var/data/silver/<corpus>/records.jsonl` -- gitignored, like every other
`var/` artifact (`CLAUDE.md`: "`var/` is gitignored and stays that way").

This module owns the schema, JSONL read/write, and the `--stats` mode
(counts per corpus and per `fact_type`, negatives count, teacher pairs
used) -- run directly (`uv run python -m portolan.training.records
--stats`) or via `scripts/data/silver_label.py --stats`, which imports
`compute_stats`/`print_stats` from here rather than re-implementing them.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "CORPORA",
    "KINDS",
    "DROP_REASONS",
    "TrainingFact",
    "LabelAudit",
    "TEACHER_FACT_FIELDS",
    "teacher_fact_fields",
    "TeacherAnswer",
    "teacher_answer_fields",
    "TrainingRecord",
    "DroppedFact",
    "SilverStats",
    "silver_dir",
    "records_path",
    "failures_path",
    "progress_path",
    "dropped_path",
    "REPAIRED_RECORDS_NAME",
    "repaired_records_path",
    "repairs_log_path",
    "label_judge_log_path",
    "append_record",
    "read_records",
    "iter_all_records",
    "append_dropped",
    "read_dropped",
    "write_dropped",
    "iter_all_dropped",
    "dropped_by_corpus",
    "compute_stats",
    "format_stats",
    "print_stats",
    "main",
]

# Every reason `run_corpus_single`/`scripts/data/backfill_exact_quotes.py`
# can log to `dropped.jsonl` (feature/labeling-exact-quotes, silver-labels
# build instructions item 2: "quote gate, overlap gate, numeral or name
# rule, quote_not_exact") -- `portolan.training.overlap_gate.
# overlap_gate_reason`'s own three reason strings, plus the two quote-gate
# reasons (live run vs. an existing record.jsonl's backfill).
DROP_REASONS: tuple[str, ...] = (
    "quote_gate",
    "overlap_gate_numeral",
    "overlap_gate_name",
    "overlap_gate_content",
    "quote_not_exact",
    "quote_not_exact_backfill",
)

# `synthetic` is listed per the build instructions' `corpus` field
# description even though this wave's pipeline (`portolan.training.
# pipeline`) only ever produces the first four -- kept in the closed list
# so a later wave over `corpus/generator`'s `synth` output validates
# against the same enum rather than inventing a second one.
CORPORA: tuple[str, ...] = ("fda_maude", "ami", "enron", "ubuntu_irc", "synthetic")
KINDS: tuple[str, ...] = ("ticket", "call", "email", "chat")


class JudgeVerdict(BaseModel):
    """One typed judge's (Jev or Kev, `portolan.training.typed_judge`)
    answer to the three gate questions for one fact -- `FactVerdict.
    to_record_dict()`'s shape, mirrored here as a real schema so a
    record round-trips through JSON without losing field types."""

    model_config = ConfigDict(extra="forbid")

    faithful: bool
    faithful_probability: float = Field(ge=0.0, le=1.0)
    predicted_type: str
    type_confidence: float = Field(ge=0.0, le=1.0)
    added_detail: bool
    added_detail_probability: float = Field(ge=0.0, le=1.0)


class TrainingFact(BaseModel):
    """One kept fact, in the production extractor's own field names
    (`portolan.extract.schema.Fact`) -- `fact_type` here, not `type`,
    per the build instructions' literal field name, since `type` shadows
    the Python builtin and every other field on this model already
    spells out what it holds (`claim`, `quote`, `locator`, `entities`).

    `judge` (coordinator course-correction, 2026-09-21, item 2: "Record
    BOTH verdicts per fact in the record (`judge: {jev: {...}, kev:
    {...}}`) when both endpoints are configured"): `None` for a record
    from the original two-free-teacher agreement mode (`--judge none`);
    `{"jev": JudgeVerdict, ...}` (and `"kev"` too when Kev was also
    configured) for the paid-teacher-plus-typed-judge mode."""

    model_config = ConfigDict(extra="forbid")

    fact_type: str
    claim: str
    quote: str | None = None
    # `[start, end]` into the OWNING record's `text` such that
    # `text[start:end] == quote` (feature/labeling-exact-quotes, build
    # instructions item 1) -- `None` for a fact with no quote, or one
    # written before exact-span resolution existed (a pre-existing
    # record.jsonl on disk until `backfill_exact_quotes.py` rewrites it;
    # `scripts/data/build_training_set.py` falls back to a bare `quote
    # in text` check for those).
    quote_span: tuple[int, int] | None = None
    locator: str
    entities: list[str] = Field(default_factory=list)
    judge: dict[str, JudgeVerdict] | None = None
    # `True` only for a fact `scripts/data/regate_dropped.py` re-admitted
    # from `dropped.jsonl` (fix/overlap-gate-paraphrase, 2026-09-22) --
    # this fact was ORIGINALLY dropped by the overlap gate, then kept on
    # re-evaluation against the current (relaxed) gate. `False` for
    # every fact a live run kept the first time. Purely informational:
    # nothing downstream treats a regated fact differently, but a future
    # audit of what changed under this fix can filter on it without
    # cross-referencing `dropped.jsonl`.
    regated: bool = False
    # The teacher's corrected fact from a repair call
    # (`portolan.training.repair`), which then passed the full gate.
    repaired: bool = False
    # The teacher's own type when the label judge retyped this fact
    # (`portolan.training.label_judge`); None when the type is the teacher's.
    teacher_fact_type: str | None = None
    # The teacher's own values for the rest of `Fact`, kept so a training
    # answer never needs placeholders (`portolan.training.sft_format`).
    # `None` on facts written before 2026-09-22; `severity`, `speaker` and
    # the validity dates can also be a teacher's genuine `None`.
    sentiment: int | None = None
    confidence: float | None = None
    severity: str | None = None
    speaker: str | None = None
    valid_from: str | None = None
    valid_to: str | None = None
    # v2 world records only (`corpus.generator.v2.export`): where the label
    # and its claim came from. The builder's provenance check requires a
    # teacher's claim on every `synthetic_v2` fact.
    label_source: str | None = None
    claim_source: str | None = None


class LabelAudit(BaseModel):
    """What happened to every fact the teacher emitted for one record
    (`scripts/data/repair_silver.py`). `complete` means each one ended kept,
    re-admitted by the fixed gate, repaired, or withdrawn by the teacher;
    the builder trains only on complete records."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["complete", "incomplete"]
    # None when the label-time drops were never logged, so the count is unknown
    teacher_facts: int | None
    kept: int
    regated: int
    repaired: int
    withdrawn: int
    unresolved: int
    unlogged: int
    audited: str
    # The label judge (`portolan.training.label_judge`), when it ran on this
    # record: its model, and the facts it dropped, retyped, or left without a
    # usable verdict (those make the record incomplete).
    judge_model: str | None = None
    judge_dropped: int = 0
    judge_retyped: int = 0
    judge_unresolved: int = 0


# `Fact` fields `TrainingFact` stores beyond type/claim/quote/locator/entities.
TEACHER_FACT_FIELDS: frozenset[str] = frozenset(
    {"sentiment", "confidence", "severity", "speaker", "valid_from", "valid_to"}
)


def teacher_fact_fields(fact: BaseModel) -> dict[str, Any]:
    """`TEACHER_FACT_FIELDS` of a teacher's `portolan.extract.schema.Fact`,
    JSON-ready."""
    return fact.model_dump(mode="json", include=set(TEACHER_FACT_FIELDS))


class TeacherAnswer(BaseModel):
    """The non-fact half of the teacher's `ExtractionResult` for one
    envelope, as emitted (entities and actions as JSON dicts)."""

    model_config = ConfigDict(extra="forbid")

    summary: str
    entities: list[dict[str, Any]] = Field(default_factory=list)
    actions: list[dict[str, Any]] = Field(default_factory=list)
    unknown_entities: list[str] = Field(default_factory=list)
    coverage: dict[str, int]


def teacher_answer_fields(result: BaseModel) -> dict[str, Any]:
    """A `TeacherAnswer` dict from a teacher's `ExtractionResult`."""
    return result.model_dump(
        mode="json", include={"summary", "entities", "actions", "unknown_entities", "coverage"}
    )


class TrainingRecord(BaseModel):
    """One line of `var/data/silver/<corpus>/records.jsonl`.

    `mode` (coordinator course-correction, 2026-09-21): `"agreement"` for
    the original two-free-open-weight-teacher design (`--judge none`,
    `portolan.training.pipeline.run_corpus`) -- `teachers` holds both
    model ids and `agreement` is the fraction of either teacher's facts
    the pair agreed on. `"judge"` for the paid-teacher-plus-typed-judge
    design (`--judge jev|kev|both`,
    `portolan.training.pipeline.run_corpus_judged`) -- `teachers` holds
    the single paid extractor's model id and `agreement` is the fraction
    of its quote-gate-passing facts the judge(s) kept."""

    model_config = ConfigDict(extra="forbid")

    record_id: str
    corpus: str
    source_id: str
    kind: str
    license: str
    text: str
    facts: list[TrainingFact] = Field(default_factory=list)
    teachers: list[str]
    agreement: float = Field(ge=0.0, le=1.0)
    created: str
    mode: str = "agreement"
    label_audit: LabelAudit | None = None
    # The prompt the teacher saw and the rest of its answer, so
    # `portolan.training.sft_format` can rebuild the example exactly. All
    # `None` on records written before 2026-09-22 (review-training-data-
    # 2026-09-22.md section 4). `user_message_inputs` is the
    # `SourceEnvelope` minus `text` (which is `text` above).
    prompt_variant: str | None = None
    system_prompt_sha256: str | None = None
    user_message_inputs: dict[str, Any] | None = None
    teacher_answer: TeacherAnswer | None = None
    # v2 world records only: the vendor note the teacher got (each world has
    # its own vendor, so no per-corpus note fits), and the world and split
    # the builder splits by.
    company_context: str | None = None
    world_id: str | None = None
    world_split: str | None = None

    @property
    def is_negative(self) -> bool:
        """A document with zero kept facts -- in `"agreement"` mode,
        both teachers agreeing there was nothing to extract (build
        instructions item 2: "the model must learn to abstain"); in
        `"judge"` mode, the paid extractor found nothing, or the judge
        kept none of what it found."""
        return len(self.facts) == 0


class DroppedFact(BaseModel):
    """One dropped fact (feature/labeling-exact-quotes, build
    instructions item 2): a fact a teacher emitted that never made it
    into a `TrainingRecord`, logged so the gate can be audited instead of
    silently discarding it. `reason` is one of `DROP_REASONS`; `fact` is
    the raw `{fact_type, claim, quote, locator, entities}` dict as the
    teacher produced it (pre-resolution -- there is no exact span to
    store for a drop). `teacher` is the single teacher model id (`--mode
    single`'s own `teachers` list always has exactly one entry)."""

    model_config = ConfigDict(extra="forbid")

    source_id: str
    reason: str
    fact: dict
    teacher: str
    created: str
    # `None` until `scripts/data/regate_dropped.py` (fix/overlap-gate-
    # paraphrase, 2026-09-22) evaluates this entry against the CURRENT
    # gate -- set on every entry it looks at, admitted or not, so a
    # second run finds nothing left to re-evaluate (idempotent) without
    # needing to inspect `records.jsonl` to find out what already
    # happened. Only entries whose `reason` starts with `"overlap_gate"`
    # are ever looked at in the first place -- `quote_gate` and
    # `quote_not_exact*` failed a check this fix does not touch, so an
    # entry with one of THOSE reasons is left `None` forever, by design,
    # not because it hasn't been considered.
    regated_at: str | None = None
    # Same meaning as on `TrainingRecord`; carried so a re-admitted fact's
    # new record can say which prompt produced it.
    prompt_variant: str | None = None
    system_prompt_sha256: str | None = None


def silver_dir(var_root: Path, corpus: str) -> Path:
    return var_root / "data" / "silver" / corpus


def records_path(var_root: Path, corpus: str) -> Path:
    return silver_dir(var_root, corpus) / "records.jsonl"


def failures_path(var_root: Path, corpus: str) -> Path:
    return silver_dir(var_root, corpus) / "failures.jsonl"


def progress_path(var_root: Path, corpus: str) -> Path:
    return silver_dir(var_root, corpus) / ".progress.json"


def dropped_path(var_root: Path, corpus: str) -> Path:
    return silver_dir(var_root, corpus) / "dropped.jsonl"


# Written by `scripts/data/repair_silver.py` next to the live files, which
# labeling runs keep appending to; the training-set builder prefers it.
REPAIRED_RECORDS_NAME = "records.repaired.jsonl"


def repaired_records_path(var_root: Path, corpus: str) -> Path:
    return silver_dir(var_root, corpus) / REPAIRED_RECORDS_NAME


def label_judge_log_path(var_root: Path, corpus: str) -> Path:
    """The label judge's call log and cache (`scripts/data/repair_silver.py
    --label-judge`)."""
    return silver_dir(var_root, corpus) / "label_judge.jsonl"


def repairs_log_path(var_root: Path, corpus: str) -> Path:
    return silver_dir(var_root, corpus) / "repairs.jsonl"


def append_record(path: Path, record: TrainingRecord) -> None:
    """Append one JSONL line, creating `path`'s parent directory if
    needed. Never rewrites the whole file -- a long-lived run (build
    instructions item 4) must survive being killed mid-run with every
    record written so far intact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(record.model_dump_json())
        f.write("\n")


def read_records(path: Path) -> Iterator[TrainingRecord]:
    """Every record in `path`, in file order. Yields nothing (not an
    error) when `path` doesn't exist yet -- a corpus with no run started."""
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            yield TrainingRecord.model_validate_json(stripped)


def iter_all_records(
    var_root: Path, corpora: Sequence[str] | None = None
) -> Iterator[TrainingRecord]:
    """Every record across `corpora` (default: every subdirectory present
    under `var_root/data/silver/`, sorted) -- used by `--stats`."""
    base = var_root / "data" / "silver"
    if not base.exists():
        return
    names = corpora if corpora else sorted(p.name for p in base.iterdir() if p.is_dir())
    for name in names:
        yield from read_records(records_path(var_root, name))


def append_dropped(path: Path, dropped: DroppedFact) -> None:
    """Append one dropped-fact JSONL line, creating `path`'s parent
    directory if needed -- the same append-only discipline as
    `append_record`, for the same reason (a long-lived run must survive
    being killed mid-run with every drop logged so far intact)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(dropped.model_dump_json())
        f.write("\n")


def write_dropped(path: Path, dropped: Sequence[DroppedFact]) -> None:
    """Overwrite `path` with exactly `dropped`, one JSONL line each, in
    order -- the rewrite counterpart to `append_dropped`'s append-only
    write, for a caller that needs to change an EXISTING entry (`scripts/
    data/regate_dropped.py` stamping `regated_at` -- fix/overlap-gate-
    paraphrase, 2026-09-22) rather than add a new one. Creates `path`'s
    parent directory if needed, same as `append_dropped`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for entry in dropped:
            f.write(entry.model_dump_json())
            f.write("\n")


def read_dropped(path: Path) -> Iterator[DroppedFact]:
    """Every dropped fact in `path`, in file order. Yields nothing when
    `path` doesn't exist yet -- a corpus whose run never dropped a
    fact, or hasn't started."""
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            yield DroppedFact.model_validate_json(stripped)


def iter_all_dropped(var_root: Path, corpora: Sequence[str] | None = None) -> Iterator[DroppedFact]:
    """Every dropped fact across `corpora` (same default-corpus rule as
    `iter_all_records`) -- used by `--stats`."""
    base = var_root / "data" / "silver"
    if not base.exists():
        return
    names = corpora if corpora else sorted(p.name for p in base.iterdir() if p.is_dir())
    for name in names:
        yield from read_dropped(dropped_path(var_root, name))


@dataclass
class SilverStats:
    """`compute_stats`'s result: everything `--stats` prints.

    `dropped_by_reason`/`total_dropped` (feature/labeling-exact-quotes,
    build instructions item 2) come from `dropped.jsonl`, passed
    separately from `records` since a drop never became a
    `TrainingRecord` -- `dropped` is `None` when the caller didn't pass
    any (e.g. an old call site, or a corpus with no drop log yet), kept
    distinct from "read the log and found zero drops" (`total_dropped ==
    0`, `dropped_by_reason` empty)."""

    total_records: int = 0
    per_corpus: Counter = field(default_factory=Counter)
    per_fact_type: Counter = field(default_factory=Counter)
    negatives_per_corpus: Counter = field(default_factory=Counter)
    total_negatives: int = 0
    teacher_pairs: Counter = field(default_factory=Counter)
    kept_facts_with_quote: int = 0
    dropped_by_reason: Counter = field(default_factory=Counter)
    total_dropped: int = 0

    @property
    def total_facts(self) -> int:
        return sum(self.per_fact_type.values())

    @property
    def exact_quote_rate(self) -> float | None:
        """Of every quote-bearing fact that reached exact-span
        resolution (kept, quote-bearing facts, plus facts dropped
        specifically for failing that resolution -- `"quote_not_exact"`
        or `"quote_not_exact_backfill"`), the fraction that resolved and
        was actually kept. `None` when there is nothing to divide by (no
        kept quote-bearing facts and no `quote_not_exact*` drops
        recorded) rather than a misleading 0.0 or 1.0."""
        not_exact_drops = self.dropped_by_reason.get(
            "quote_not_exact", 0
        ) + self.dropped_by_reason.get("quote_not_exact_backfill", 0)
        denominator = self.kept_facts_with_quote + not_exact_drops
        if denominator == 0:
            return None
        return self.kept_facts_with_quote / denominator


def _teacher_pair_key(teachers: Sequence[str]) -> str:
    """A stable, order-independent key for a teacher pair, so `qwen +
    deepseek` and `deepseek + qwen` count as the same pair."""
    return " + ".join(sorted(teachers))


def compute_stats(
    records: Iterable[TrainingRecord], dropped: Iterable[DroppedFact] | None = None
) -> SilverStats:
    """`records` drives everything except `dropped_by_reason*`/
    `total_dropped`/half of `exact_quote_rate`'s denominator -- those
    come from `dropped` (feature/labeling-exact-quotes, build
    instructions item 2), an optional second iterable of `DroppedFact`
    since a drop log is per-corpus (`dropped.jsonl`) and a caller with no
    interest in drops (or an older one, before this existed) can still
    call this with just `records`."""
    stats = SilverStats()
    for record in records:
        stats.total_records += 1
        stats.per_corpus[record.corpus] += 1
        stats.teacher_pairs[_teacher_pair_key(record.teachers)] += 1
        if record.is_negative:
            stats.total_negatives += 1
            stats.negatives_per_corpus[record.corpus] += 1
        else:
            for fact in record.facts:
                stats.per_fact_type[fact.fact_type] += 1
                if fact.quote:
                    stats.kept_facts_with_quote += 1

    for drop in dropped or ():
        stats.total_dropped += 1
        stats.dropped_by_reason[drop.reason] += 1

    return stats


def dropped_by_corpus(var_root: Path, corpora: Sequence[str] | None = None) -> dict[str, Counter]:
    """`{corpus: Counter(reason -> count)}` for every corpus with a
    `dropped.jsonl` (feature/labeling-exact-quotes, build instructions
    item 2: "`--stats` reports drops by reason per corpus"). `DroppedFact`
    itself carries no `corpus` field (`dropped.jsonl` is already scoped
    to one corpus's own directory, same as `records.jsonl`), so this
    reads each corpus's log separately rather than trying to recover the
    corpus from a combined `iter_all_dropped` stream."""
    base = var_root / "data" / "silver"
    if not base.exists():
        return {}
    names = corpora if corpora else sorted(p.name for p in base.iterdir() if p.is_dir())
    result: dict[str, Counter] = {}
    for name in names:
        counts: Counter = Counter()
        for drop in read_dropped(dropped_path(var_root, name)):
            counts[drop.reason] += 1
        if counts:
            result[name] = counts
    return result


def format_stats(stats: SilverStats, dropped_per_corpus: dict[str, Counter] | None = None) -> str:
    lines: list[str] = []
    lines.append(
        f"records: {stats.total_records}  facts: {stats.total_facts}  "
        f"negatives: {stats.total_negatives}"
    )
    lines.append("")
    lines.append("per corpus:")
    for corpus, count in sorted(stats.per_corpus.items()):
        negatives = stats.negatives_per_corpus.get(corpus, 0)
        lines.append(f"  {corpus:<12} records={count:<6} negatives={negatives}")
    lines.append("")
    lines.append("per fact_type:")
    if stats.per_fact_type:
        for fact_type, count in stats.per_fact_type.most_common():
            lines.append(f"  {fact_type:<20} {count}")
    else:
        lines.append("  (none)")
    lines.append("")
    lines.append("teacher pairs used:")
    if stats.teacher_pairs:
        for pair, count in stats.teacher_pairs.most_common():
            lines.append(f"  {pair:<60} {count}")
    else:
        lines.append("  (none)")
    lines.append("")
    exact_rate = stats.exact_quote_rate
    rate_str = f"{exact_rate:.1%}" if exact_rate is not None else "n/a (no quote-bearing facts)"
    lines.append(f"exact-quote rate: {rate_str}  (kept={stats.kept_facts_with_quote})")
    lines.append("")
    lines.append(f"dropped: {stats.total_dropped}")
    if stats.dropped_by_reason:
        for reason, count in stats.dropped_by_reason.most_common():
            lines.append(f"  {reason:<28} {count}")
    lines.append("")
    lines.append("dropped by reason, per corpus:")
    if dropped_per_corpus:
        for corpus in sorted(dropped_per_corpus):
            counts = dropped_per_corpus[corpus]
            total = sum(counts.values())
            breakdown = ", ".join(f"{r}={c}" for r, c in counts.most_common())
            lines.append(f"  {corpus:<12} total={total:<5} {breakdown}")
    else:
        lines.append("  (none)")
    return "\n".join(lines)


def print_stats(stats: SilverStats, dropped_per_corpus: dict[str, Counter] | None = None) -> None:
    print(format_stats(stats, dropped_per_corpus))


def _default_var_root() -> Path:
    return Path(__file__).resolve().parents[2] / "var"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Silver-label training record stats (portolan.training.records)."
    )
    parser.add_argument(
        "--stats", action="store_true", help="print per-corpus/per-fact_type counts and exit"
    )
    parser.add_argument(
        "--corpus",
        action="append",
        dest="corpora",
        choices=CORPORA,
        help="limit to one corpus (repeatable); default: every corpus with records on disk",
    )
    parser.add_argument(
        "--var-root",
        type=Path,
        default=_default_var_root(),
        help="repo var/ directory (default: the repo root's var/, resolved from this file)",
    )
    args = parser.parse_args(argv)

    if not args.stats:
        parser.print_help()
        return 0

    records = iter_all_records(args.var_root, args.corpora)
    dropped = iter_all_dropped(args.var_root, args.corpora)
    stats = compute_stats(records, dropped)
    print_stats(stats, dropped_by_corpus(args.var_root, args.corpora))
    return 0


if __name__ == "__main__":
    sys.exit(main())
