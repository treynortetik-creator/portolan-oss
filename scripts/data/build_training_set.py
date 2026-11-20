"""Build the fine-tuning training set from the silver-labeled corpora on
disk, curate it, split it, and write `train.jsonl`, `val.jsonl`,
`manifest.json` and `data_card.md`.

Inputs: `var/data/silver/<corpus>/records.jsonl` for every corpus in
`portolan.training.pipeline.CORPUS_LICENSES`, every
`var/data/silver/synthetic/*.jsonl` batch, and the `training_records.jsonl`
of every exported v2 world under `var/data/synth_v2/worlds/{train,
validation}/` (corpus `synthetic_v2`, `corpus.generator.v2.export`; a dry-run
world or one without an export manifest is skipped). When a corpus has
`records.repaired.jsonl` (`scripts/data/repair_silver.py`), its records
replace the ones with the same `record_id`; records labeled after the
repair run come from `records.jsonl` unaudited. Never `corpus/out/`:
`assert_not_holdout` runs on every path this script opens.

**The one rule for every step: drop the record, never edit it.** A step
that finds one bad fact drops the whole record. Removing a fact from a
record would turn its span into a taught "this is not a fact", which is
a label nobody wrote (review 2026-09-22, finding 5); the builder before
this change did exactly that for failed quotes and for the `other` cap.

**Pipeline** (review `01-spec/reviews/review-training-data-2026-09-22.md`):

0. Contamination (`01-spec/world-generator-v2-2026-09-22.md` section 6,
   work package WP8; coordinator review 2026-09-22). Every record's text
   is checked against `var/contamination/` (built by `scripts/data/
   build_contamination_signatures.py`) for a MinHash near-duplicate, a
   shared 13-word sequence, ERB's canary string, or a blocklisted proper
   name against the Halvard exam corpus or EnterpriseRAG-Bench.
   Near-duplicate, exact-13-gram-overlap and canary hits are a hard gate
   for EVERY corpus. A blocklisted-name hit is a hard gate only for
   `CONTAMINATION_HARD_GATE_CORPORA` (sources this project GENERATES:
   `synthetic`, `synthetic_v2`) -- for a real public corpus a name hit is
   counted and written to `<out>/contamination_review.jsonl`, never
   dropped on its own, because an ordinary English word that also
   happens to be a blocklisted name (the review's examples: "Link",
   "Memorial", "Riverside", "NICE") says nothing about an FDA report or
   an Enron email. The blocklist itself is already filtered for
   distinctiveness before it reaches this script
   (`contamination.name_drop_reason`: job titles, names made of common
   words, and names or single tokens found in a real-corpus background
   are left out -- see that function's docstring for the calibration). Missing
   signatures REFUSE the run (`contamination.ContaminationSignaturesMissing`)
   unless `--no-contamination-check` is passed explicitly.
1. Provenance (CLAUDE.md, training-data provenance rule). A record is
   dropped when
   - it is a v1 synthetic negative: templated filler from
     `corpus.generator.prose.generate_filler_*`, no teacher recorded;
   - any teacher id is not on `ALLOWED_TEACHER_IDS`, or no teacher is
     recorded at all (fail closed; drops are counted per teacher id);
   - the label judge that kept, dropped or retyped its facts
     (`label_audit.judge_model`) is not on `ALLOWED_TEACHER_IDS`;
   - its corpus's document text was written by a closed model
     (`CORPUS_DOCUMENT_TEXT`: SaaS sales is GPT-4 dialogue);
   - its claims come from code templates, not a model
     (`CORPUS_CLAIM_SOURCE`: every v1 synthetic positive);
   - it is a v2 world record (`synthetic_v2`) with a fact whose claim is
     not a teacher's (`claim_source`), or from a test world, a world with
     no split, or the held-out vertical.
2. Verification: every fact carries a non-blank quote (a fact with none
   drops the record: `"" in text` is always true, so nothing else would
   check it), and every quote must equal `text[quote_span]` (or be a
   substring of `text` when no span was stored).
3. Label completeness: a record with no facts whose teacher DID emit
   facts is a negative made by the gate, not by the teacher, and is
   dropped, even when the teacher later withdrew them all. A positive
   with a teacher fact that never ended kept, repaired or withdrawn is
   dropped too: `incomplete_labels` when its `label_audit` says so,
   `partial_labels` when it was never audited and `agreement < 1.0`
   (`--keep-partial-records` keeps those).
4. Document dedup (`portolan.training.curation.similarity_groups`):
   5-word-shingle Jaccard >= `--dedup-jaccard` (0.8, the recipes note,
   section 4) keeps one record per group: the one with the most facts,
   then the lowest record id.
5. Conflicting labels: when one normalized quote carries two or more
   fact types across records (the CFPB canned company response is
   labeled `other` 34 times, `commitment` 6 and `decision` 4), records
   with a minority type are dropped when one type holds at least
   `--conflict-majority` (0.75) of the occurrences; otherwise every
   record carrying the quote is dropped.
6. Repeat caps: a normalized claim+quote pair may appear
   `--max-pair-repeats` times (1: a second copy is a duplicated source),
   a normalized quote `--max-quote-repeats` times (3, world generator v2
   spec section 4). Records are visited in a seeded shuffle.
7. `other` cap: whole records with the highest `other` share are dropped
   until `other` is at most `--other-cap` of facts (0.15).
8. Negative cap: negatives at most `--negative-cap` of records (0.30,
   v2 spec open decision 5), solved against the post-drop pool.
9. Corpus share cap: no corpus above `--max-corpus-share` of records
   (0.5); skipped when only one corpus is left.
10. Split: 90/10 by a hash of the smallest `source_id` in each group of
    near-siblings (Jaccard >= `--split-group-jaccard`, 0.5, the v2 spec's
    "close sibling" line) joined with every record sharing a
    `source_id`, so a thread and its forwards never straddle the split.
    A `synthetic_v2` record goes to its world's split instead (v2 spec
    section 6: one world's style never lands in both train and validation).

**Cap math.** "Cap X at N percent" is solved for the share X has AFTER
the drop: `keep <= ratio * rest / (1 - ratio)`, since dropping X shrinks
the total along with X.

Usage:
    uv run python scripts/data/build_training_set.py --out var/data/training/2026-09-22b/
    uv run python scripts/data/build_training_set.py --var-root <snapshot> --out <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import statistics
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

from portolan.training import contamination
from portolan.training.curation import pair_key, quote_key, similarity_groups
from portolan.training.pipeline import CORPUS_LICENSES, assert_not_holdout
from portolan.training.records import (
    REPAIRED_RECORDS_NAME,
    TrainingRecord,
    read_records,
    records_path,
    repaired_records_path,
)
from portolan.util import clock

REPO_ROOT = Path(__file__).resolve().parents[2]
VAR_ROOT = REPO_ROOT / "var"

REAL_CORPORA: tuple[str, ...] = tuple(CORPUS_LICENSES.keys())
SYNTHETIC_CORPUS = "synthetic"
# v2 worlds, exported by `corpus/generator/v2/export.py` (WP7).
SYNTHETIC_V2_CORPUS = "synthetic_v2"
V2_WORLDS_DIR = Path("data") / "synth_v2" / "worlds"
V2_RECORDS_NAME = "training_records.jsonl"
V2_DRY_RUN_MARKER = "DRY_RUN"  # `scripts/data/synth_v2.py` marks a fixture-writer root
# World split -> builder bucket. Test worlds are the bench and never train.
V2_SPLIT_BUCKET: dict[str, str] = {"train": "train", "validation": "val"}
V2_TEACHER_CLAIM_SOURCES = frozenset({"teacher", "teacher_restatement"})
V2_PACKS_DIR = Path(__file__).resolve().parents[2] / "corpus" / "generator" / "v2" / "packs"

# WP8 name-blocklist scope (coordinator review, 2026-09-22): a
# blocklisted-name hit is a hard gate only for sources this project
# GENERATES -- everything else is `contamination`'s own module docstring
# calls "review-only" (counted, reported, never dropped on its own). The
# near-duplicate, exact-13-gram-overlap and canary checks are NOT scoped
# this way; they gate every corpus regardless of this set.
CONTAMINATION_HARD_GATE_CORPORA: frozenset[str] = frozenset({SYNTHETIC_CORPUS, SYNTHETIC_V2_CORPUS})

# --------------------------------------------------------------------------
# Provenance tables. Changing one of these is a provenance decision:
# record it in 01-spec/decisions-log.md with the evidence.
# --------------------------------------------------------------------------

# Exact model ids (lowercase, OpenRouter variant suffix such as ":free"
# stripped) of open-weight models whose output may reach a training
# document or label. Exact ids, not family substrings: "qwen" also
# matches closed API models (qwen-max, qwen-plus, qwen-flash), "mistral"
# matches mistral-medium. Sources: the labeling teachers in
# 01-spec/training-data-2026-09-19.md, the v1 synthetic writers on disk,
# and the v2 writer rotation in 01-spec/world-generator-v2-2026-09-22.md
# section 4. nvidia/nemotron-3-* were allowed by this script's original
# build instructions although portolan/training/teachers.py excludes the
# family for silver discovery; the conflict is open (review, problem list).
# Local open-weight teachers run on this Mac through MLX (`--teacher
# local:<repo id>`, portolan/training/local_teacher.py). Exact lowercase ids,
# each allowed only because its license permits training on its outputs;
# the license is the value. Must equal
# `portolan.training.local_teacher.permitted_local_teacher_ids()`
# (tests/test_local_teacher.py).
ALLOWED_LOCAL_TEACHERS: dict[str, str] = {
    "local:mlx-community/qwen3.5-4b-mlx-4bit": "Apache-2.0 (Qwen/Qwen3.5-4B)",
    "local:mlx-community/gemma-4-e4b-it-4bit": "Apache-2.0 (google/gemma-4-e4b-it, Gemma 4 license)",
    "local:mlx-community/ministral-3-8b-instruct-2512-4bit": (
        "Apache-2.0 (mistralai/Ministral-3-8B-Instruct-2512)"
    ),
    "local:mlx-community/phi-4-mini-instruct-4bit": "MIT (microsoft/Phi-4-mini-instruct)",
}

ALLOWED_TEACHER_IDS: frozenset[str] = frozenset(ALLOWED_LOCAL_TEACHERS) | frozenset(
    {
        # labeling teachers
        "deepseek/deepseek-v3.2",
        "deepseek/deepseek-v4.1-flash",
        "moonshotai/kimi-k2",
        "mistralai/mistral-nemo",
        # v1 synthetic writers
        "deepseek/deepseek-v4-flash",
        "qwen/qwen3-30b-a3b-instruct-2507",
        "qwen/qwen3-32b",
        "qwen/qwen3.8-27b",
        "mistralai/mistral-small-3.2-24b-instruct",
        "z-ai/glm-5.2",
        "nvidia/nemotron-3-ultra-550b-a55b",
        "nvidia/nemotron-3-super-120b-a12b",
        # v2 writer rotation not listed above
        "qwen/qwen3-235b-a22b-2507",
        "mistralai/mistral-small-2603",
        "z-ai/glm-4.7-flash",
        "openai/gpt-oss-120b",
    }
)

# Why a family is kept out, for the data card. Not consulted for the
# decision itself: anything not on ALLOWED_TEACHER_IDS is dropped.
EXCLUDED_TEACHER_NOTES: dict[str, str] = {
    "cohere/": "Cohere's terms restrict training on outputs (2026-09-21)",
    "google/gemma": "output-use terms not yet read (v2 spec section 4)",
    "meta-llama/": "output-use terms not yet read (v2 spec section 4)",
}

# Who wrote each corpus's document text. "human": people wrote it.
# "open_weight_writer": an open-weight model, checked per record through
# `teachers`. "closed_model": excluded by the provenance rule.
CORPUS_DOCUMENT_TEXT: dict[str, str] = {
    "fda_maude": "human",
    "ami": "human",
    "enron": "human",
    "ubuntu_irc": "human",
    "cfpb": "human",
    # GPT-4-generated dialogue: scripts/data/fetch_saas.py docstring and
    # var/data/MANIFEST.md section 7 "Known quirks", both citing the
    # dataset card.
    "saas_sales": "closed_model",
    SYNTHETIC_CORPUS: "open_weight_writer",
    SYNTHETIC_V2_CORPUS: "open_weight_writer",
}

# Who wrote each corpus's claims. "teacher": the model in `teachers`.
# "template": code (corpus/generator/worlds.py `_tpl_*`), which the
# provenance rule does not allow; every `synthetic` record is v1. A v2
# record's claims are checked per fact (`claim_source`), since a scripted
# label can keep its planted claim when no teacher confirmed it.
CORPUS_CLAIM_SOURCE: dict[str, str] = {
    **{c: "teacher" for c in REAL_CORPORA},
    SYNTHETIC_CORPUS: "template",
    SYNTHETIC_V2_CORPUS: "teacher",
}

OTHER_FACT_TYPE = "other"
NO_TEACHER = "(none recorded)"
# A type with fewer facts than this is flagged in the data card as too
# thin for the student to learn.
THIN_TYPE_FACTS = 20
CAP_RANDOM_SEED = 20260922


@dataclass(frozen=True)
class CurationConfig:
    negative_cap: float = 0.30
    other_cap: float = 0.15
    max_corpus_share: float = 0.50
    max_pair_repeats: int = 1
    max_quote_repeats: int = 3
    dedup_jaccard: float = 0.8
    split_group_jaccard: float = 0.5
    val_fraction: float = 0.10
    conflict_majority: float = 0.75
    drop_partial_records: bool = True


# --------------------------------------------------------------------------
# Contamination gate (WP8)
# --------------------------------------------------------------------------


def _contamination_gate(
    record: TrainingRecord,
    refs: contamination.ContaminationReferences,
    stats: BuildStats,
) -> str | None:
    """Runs `contamination.check_text` against `record.text` and applies
    WP8's per-corpus scope (coordinator review, 2026-09-22): near-
    duplicate, exact-13-gram-overlap and canary hits gate every corpus;
    a blocklisted-name hit gates only `CONTAMINATION_HARD_GATE_CORPORA`.
    On a corpus outside that set, a name hit is counted into
    `stats.contamination_review_*` and logged for the report, but never
    drops the record by itself.

    Returns a reason string (e.g. `"contaminated_near_duplicate"`,
    `"contaminated_blocklisted_name"`) for `_drop` to use when the record
    is dropped, keyed off the FIRST gating hit found (`check_text`'s own
    order: near-duplicate, exact overlap, canary, name) -- or `None` when
    the record survives. Every gating hit, not just the first, is still
    counted in `stats.contamination_by_reason`; a record can be dropped
    for one reason while other hits it also carried are recorded too."""
    hits = [h for h in contamination.check_text(record.text, refs) if h.exclusionary]
    if not hits:
        return None
    hard_gate_here = record.corpus in CONTAMINATION_HARD_GATE_CORPORA
    gating_hits = [h for h in hits if hard_gate_here or h.reason != "blocklisted_name"]
    review_hits = [h for h in hits if h not in gating_hits]

    if review_hits:
        for h in review_hits:
            stats.contamination_review_by_reason[h.reason] += 1
        stats.contamination_review_per_corpus[record.corpus] += 1
        stats.contamination_review.append(
            {
                "record_id": record.record_id,
                "corpus": record.corpus,
                "source_id": record.source_id,
                "reasons": [h.reason for h in review_hits],
                "detail": [h.detail for h in review_hits],
            }
        )

    if not gating_hits:
        return None
    for h in gating_hits:
        stats.contamination_by_reason[h.reason] += 1
    stats.contamination_per_corpus[record.corpus] += 1
    stats.contamination_dropped.append(
        {
            "record_id": record.record_id,
            "corpus": record.corpus,
            "source_id": record.source_id,
            "reasons": [h.reason for h in gating_hits],
            "detail": [h.detail for h in gating_hits],
        }
    )
    return f"contaminated_{gating_hits[0].reason}"


# --------------------------------------------------------------------------
# Provenance checks
# --------------------------------------------------------------------------


def normalize_model_id(model_id: str) -> str:
    """Lowercase, without an OpenRouter variant suffix (":free" serves
    the same weights as the paid id). A `local:` id keeps its prefix: it
    names MLX weights on this Mac, not an OpenRouter route."""
    low = model_id.strip().lower()
    if low.startswith("local:"):
        return low
    return low.split(":", 1)[0]


def teacher_allowed(model_id: str) -> bool:
    return normalize_model_id(model_id) in ALLOWED_TEACHER_IDS


def record_teachers_allowed(teachers: list[str]) -> bool:
    """True only when at least one teacher is recorded and every one is
    on the allowlist. A record with no recorded teacher has no
    provenance and fails closed."""
    return bool(teachers) and all(teacher_allowed(t) for t in teachers)


def _fact_is_verified(fact, text: str) -> bool:
    """`text[quote_span] == quote` when a span is present; `quote in
    text` otherwise. A fact with no quote has nothing to check."""
    if fact.quote is None:
        return True
    if fact.quote_span is not None:
        start, end = fact.quote_span
        if not (0 <= start < end <= len(text)):
            return False
        return text[start:end] == fact.quote
    return fact.quote in text


def _estimated_dropped_facts(record: TrainingRecord) -> int:
    """Facts the teacher emitted that the gate dropped, from `agreement`
    (kept / emitted at label time). Facts re-admitted later by
    `scripts/data/regate_dropped.py` carry `regated` and are not part of
    that original kept count."""
    if record.agreement >= 1.0:
        return 0
    kept_at_label_time = sum(1 for f in record.facts if not f.regated)
    if record.agreement <= 0.0:
        return 1 if not record.facts else 0
    emitted = round(kept_at_label_time / record.agreement)
    regated = sum(1 for f in record.facts if f.regated)
    return max(0, emitted - kept_at_label_time - regated)


def _unresolved_facts(record: TrainingRecord) -> int:
    """Teacher facts that ended neither kept, repaired nor withdrawn: from
    the record's audit when it has one, else estimated from `agreement`."""
    if record.label_audit is not None:
        return record.label_audit.unresolved + record.label_audit.unlogged
    return _estimated_dropped_facts(record)


def _gate_made_negative(record: TrainingRecord) -> bool:
    """No facts left, although the teacher emitted some (or may have)."""
    if not record.is_negative:
        return False
    if record.label_audit is not None:
        return record.label_audit.teacher_facts != 0
    return record.agreement < 1.0


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------


def list_input_files(var_root: Path) -> list[Path]:
    """Every input file, in a fixed order: real corpora (registry order,
    a corpus's `records.repaired.jsonl` before its `records.jsonl`), then
    synthetic batches by file name, then exported v2 train and validation
    worlds by path."""
    silver_root = var_root / "data" / "silver"
    assert_not_holdout(silver_root)
    paths = []
    for corpus in REAL_CORPORA:
        for path in (repaired_records_path(var_root, corpus), records_path(var_root, corpus)):
            assert_not_holdout(path)
            if path.exists():
                paths.append(path)
    synthetic_dir = silver_root / SYNTHETIC_CORPUS
    assert_not_holdout(synthetic_dir)
    if synthetic_dir.exists():
        paths.extend(sorted(synthetic_dir.glob("*.jsonl")))
    paths.extend(v2_world_files(var_root)[0])
    return paths


def v2_world_files(var_root: Path) -> tuple[list[Path], dict[str, str]]:
    """Exported v2 worlds' `training_records.jsonl` under the train and
    validation folders, and `world dir -> reason` for each world skipped:
    no export manifest, a dry run (fixture text), or a split the folder does
    not match."""
    root = var_root / V2_WORLDS_DIR
    assert_not_holdout(root)
    paths: list[Path] = []
    skipped: dict[str, str] = {}
    if (root / V2_DRY_RUN_MARKER).exists():
        return paths, {str(root): "dry-run root"}
    for split in V2_SPLIT_BUCKET:
        for records in sorted((root / split).glob(f"*/{V2_RECORDS_NAME}")):
            world = records.parent
            assert_not_holdout(world)
            manifest = world / "manifest.json"
            try:
                data = json.loads(manifest.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                skipped[str(world.relative_to(var_root))] = "no export manifest"
                continue
            if not data.get("export_version"):
                skipped[str(world.relative_to(var_root))] = "no export manifest"
            elif data.get("dry_run") is not False:
                skipped[str(world.relative_to(var_root))] = "dry run: fixture writer text"
            elif data.get("split") != split:
                skipped[str(world.relative_to(var_root))] = f"manifest split {data.get('split')!r}"
            else:
                paths.append(records)
    return paths, skipped


def load_records(paths: Iterable[Path]) -> list[TrainingRecord]:
    """Records from `paths` in order; a `records.jsonl` record whose id its
    corpus's `records.repaired.jsonl` already holds is skipped."""
    out: list[TrainingRecord] = []
    repaired_ids: dict[Path, set[str]] = {}
    for path in paths:
        if path.name == REPAIRED_RECORDS_NAME:
            ids = repaired_ids.setdefault(path.parent, set())
            for record in read_records(path):
                ids.add(record.record_id)
                out.append(record)
            continue
        skip = repaired_ids.get(path.parent, set())
        out.extend(r for r in read_records(path) if r.record_id not in skip)
    return out


def iter_silver_records(var_root: Path) -> Iterator[TrainingRecord]:
    yield from load_records(list_input_files(var_root))


def describe_inputs(var_root: Path, paths: Iterable[Path]) -> list[dict]:
    out = []
    for path in paths:
        data = path.read_bytes()
        out.append(
            {
                "path": str(path.relative_to(var_root)),
                "lines": sum(1 for line in data.splitlines() if line.strip()),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    return out


# --------------------------------------------------------------------------
# Stats
# --------------------------------------------------------------------------


@dataclass
class BuildStats:
    input_records: int = 0
    dropped_by_reason: Counter = field(default_factory=Counter)
    dropped_facts_by_reason: Counter = field(default_factory=Counter)
    dropped_by_reason_corpus: dict = field(default_factory=lambda: defaultdict(Counter))
    dropped_by_teacher: Counter = field(default_factory=Counter)
    provenance_violations: Counter = field(default_factory=Counter)
    input_records_per_corpus: Counter = field(default_factory=Counter)
    input_facts_per_corpus: Counter = field(default_factory=Counter)
    records_per_corpus: Counter = field(default_factory=Counter)
    facts_per_corpus: Counter = field(default_factory=Counter)
    negatives_per_corpus: Counter = field(default_factory=Counter)
    facts_per_type: Counter = field(default_factory=Counter)
    facts_per_corpus_type: dict = field(default_factory=lambda: defaultdict(Counter))
    records_per_teacher: Counter = field(default_factory=Counter)
    median_chars_per_corpus: dict = field(default_factory=dict)
    facts_per_600_words: dict = field(default_factory=dict)
    partial_records_per_corpus: Counter = field(default_factory=Counter)
    est_dropped_facts_per_corpus: Counter = field(default_factory=Counter)
    audited_records_per_corpus: Counter = field(default_factory=Counter)
    regated_facts_per_corpus: Counter = field(default_factory=Counter)
    repaired_facts_per_corpus: Counter = field(default_factory=Counter)
    withdrawn_facts_per_corpus: Counter = field(default_factory=Counter)
    duplicate_groups: int = 0
    near_sibling_pairs: int = 0
    split_groups_multi: int = 0
    max_quote_repeat_out: int = 0
    max_pair_repeat_out: int = 0
    caps: dict = field(default_factory=dict)
    negatives_total: int = 0
    negatives_train: int = 0
    negatives_val: int = 0
    teachers: set[str] = field(default_factory=set)
    train_records: int = 0
    val_records: int = 0
    train_facts: int = 0
    val_facts: int = 0
    inputs: list = field(default_factory=list)
    # Contamination gate (WP8). `contamination_enabled` is False only
    # when the caller passed no `contamination_refs`
    # (`--no-contamination-check`) -- distinct from "ran and found
    # nothing," which leaves the Counters empty but this True.
    contamination_enabled: bool = False
    # Hard-gate drops: hit-type counts (a record can carry more than one
    # gating hit) and dropped-RECORD counts per corpus.
    contamination_by_reason: Counter = field(default_factory=Counter)
    contamination_per_corpus: Counter = field(default_factory=Counter)
    contamination_dropped: list = field(default_factory=list)
    # Review-only: a blocklisted-name hit on a corpus outside
    # CONTAMINATION_HARD_GATE_CORPORA. Counted and logged, never dropped.
    contamination_review_by_reason: Counter = field(default_factory=Counter)
    contamination_review_per_corpus: Counter = field(default_factory=Counter)
    contamination_review: list = field(default_factory=list)
    # "<variant> <sha256[:16]>" per output record, or UNSTAMPED_PROMPT.
    records_per_prompt: Counter = field(default_factory=Counter)
    # v2 worlds: world id -> {"split", "input", "train", "val"}; skipped world dirs.
    v2_worlds: dict = field(default_factory=dict)
    v2_worlds_skipped: dict = field(default_factory=dict)


UNSTAMPED_PROMPT = "unstamped"


def prompt_stamp_key(record: TrainingRecord) -> str:
    if record.prompt_variant is None or record.system_prompt_sha256 is None:
        return UNSTAMPED_PROMPT
    return f"{record.prompt_variant} {record.system_prompt_sha256[:16]}"


def _drop(stats: BuildStats, record: TrainingRecord, reason: str) -> None:
    stats.dropped_by_reason[reason] += 1
    stats.dropped_facts_by_reason[reason] += len(record.facts)
    stats.dropped_by_reason_corpus[reason][record.corpus] += 1


# --------------------------------------------------------------------------
# Steps
# --------------------------------------------------------------------------


def provenance_failures(record: TrainingRecord) -> list[str]:
    """Every provenance check `record` fails, in a fixed order; empty when it
    passes. `scripts/train/mlx_lora.py` re-runs this on every record it is
    given, so a set this builder did not write cannot slip past it."""
    failures: list[str] = []
    if record.corpus == SYNTHETIC_CORPUS and record.is_negative:
        failures.append("v1_templated_negative")
    if not record.teachers:
        failures.append("no_teacher_recorded")
    elif any(not teacher_allowed(t) for t in record.teachers):
        failures.append("teacher_not_allowed")
    # The label judge's verdicts decide which facts stay, so it is held to the same list.
    judge = record.label_audit.judge_model if record.label_audit is not None else None
    if judge is not None and not teacher_allowed(judge):
        failures.append("judge_not_allowed")
    text_source = CORPUS_DOCUMENT_TEXT.get(record.corpus)
    if text_source is None:
        failures.append("corpus_provenance_unknown")
    elif text_source == "closed_model":
        failures.append("document_text_not_open_weight")
    if CORPUS_CLAIM_SOURCE.get(record.corpus) == "template" and not record.is_negative:
        failures.append("v1_template_claim")
    if record.corpus == SYNTHETIC_V2_CORPUS:
        failures.extend(v2_failures(record))
    return failures


def held_out_packs(packs_dir: Path = V2_PACKS_DIR) -> frozenset[str]:
    """Pack names whose YAML says `held_out: true` (the eighth vertical, test
    worlds only)."""
    import yaml

    out = set()
    for path in sorted(packs_dir.glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if isinstance(data, dict) and data.get("held_out") is True:
            out.add(path.stem)
    return frozenset(out)


_HELD_OUT_PACKS: frozenset[str] | None = None


def v2_failures(record: TrainingRecord) -> list[str]:
    """The v2 world rules: every claim is a teacher's, and the record comes
    from a train or validation world outside the held-out vertical."""
    global _HELD_OUT_PACKS
    if _HELD_OUT_PACKS is None:
        _HELD_OUT_PACKS = held_out_packs()
    failures: list[str] = []
    if any(f.claim_source not in V2_TEACHER_CLAIM_SOURCES for f in record.facts):
        failures.append("v2_claim_not_from_teacher")
    if not record.world_id or record.world_split not in V2_SPLIT_BUCKET:
        failures.append("v2_world_not_trainable")
    if record.world_id and any(
        record.world_id.startswith(f"world-{pack}-") for pack in _HELD_OUT_PACKS
    ):
        failures.append("v2_held_out_vertical")
    return failures


def _provenance_reason(record: TrainingRecord, stats: BuildStats) -> str | None:
    """The first provenance check `record` fails, or None. Every failing
    check is also counted in `stats.provenance_violations`, and every
    disallowed teacher id in `stats.dropped_by_teacher`."""
    failures = provenance_failures(record)
    if not record.teachers:
        stats.dropped_by_teacher[NO_TEACHER] += 1
    for t in sorted({t for t in record.teachers if not teacher_allowed(t)}):
        stats.dropped_by_teacher[t] += 1
    for f in failures:
        stats.provenance_violations[f] += 1
    return failures[0] if failures else None


def _fact_has_quote(fact) -> bool:
    return bool(fact.quote and fact.quote.strip())


def _dedup_documents(
    pool: list[TrainingRecord], config: CurationConfig, stats: BuildStats
) -> tuple[list[TrainingRecord], dict[int, str]]:
    """Keep one record per near-duplicate group; return the survivors and
    each survivor's split key (smallest source_id among its near-siblings
    and same-source records)."""
    groups = similarity_groups(
        [r.text for r in pool],
        dedup_threshold=config.dedup_jaccard,
        split_threshold=config.split_group_jaccard,
    )
    stats.near_sibling_pairs = len(groups.pairs)
    keep = [True] * len(pool)
    for members in groups.duplicate_groups:
        if len(members) < 2:
            continue
        stats.duplicate_groups += 1
        best = min(members, key=lambda i: (-len(pool[i].facts), pool[i].record_id))
        for i in members:
            if i != best:
                keep[i] = False
                _drop(stats, pool[i], "near_duplicate_document")

    parent = list(range(len(pool)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for members in groups.split_groups:
        for i in members[1:]:
            union(members[0], i)
    first_by_source: dict[str, int] = {}
    for i, r in enumerate(pool):
        if r.source_id in first_by_source:
            union(first_by_source[r.source_id], i)
        else:
            first_by_source[r.source_id] = i
    component_key: dict[int, str] = {}
    for i, r in enumerate(pool):
        root = find(i)
        if root not in component_key or r.source_id < component_key[root]:
            component_key[root] = r.source_id
    stats.split_groups_multi = sum(1 for m in groups.split_groups if len(m) > 1)

    survivors = [r for i, r in enumerate(pool) if keep[i]]
    # keyed by object identity: record ids are not guaranteed unique
    # across input files (a resumed labeling run could append twice)
    split_keys = {id(pool[i]): component_key[find(i)] for i in range(len(pool)) if keep[i]}
    return survivors, split_keys


def _drop_conflicting_labels(
    pool: list[TrainingRecord], config: CurationConfig, stats: BuildStats
) -> list[TrainingRecord]:
    """One normalized quote labeled with two or more fact types across
    records is label noise. When one type holds at least
    `config.conflict_majority` of that quote's occurrences, records
    carrying a minority type are dropped and the majority stays;
    otherwise every record carrying the quote is dropped."""
    types_by_quote: dict[str, Counter] = defaultdict(Counter)
    for r in pool:
        for f in r.facts:
            k = quote_key(f.quote)
            if k:
                types_by_quote[k][f.fact_type] += 1
    allowed_type: dict[str, str | None] = {}
    for k, counts in types_by_quote.items():
        if len(counts) < 2:
            continue
        top_type, top_n = counts.most_common(1)[0]
        share = top_n / sum(counts.values())
        allowed_type[k] = top_type if share >= config.conflict_majority else None
    stats.caps["conflicting_quotes"] = len(allowed_type)
    stats.caps["conflicting_quotes_resolved_by_majority"] = sum(
        1 for v in allowed_type.values() if v is not None
    )
    out = []
    for r in pool:
        bad = any(
            quote_key(f.quote) in allowed_type and allowed_type[quote_key(f.quote)] != f.fact_type
            for f in r.facts
        )
        if bad:
            _drop(stats, r, "conflicting_type_for_same_quote")
        else:
            out.append(r)
    return out


def _apply_repeat_caps(
    pool: list[TrainingRecord], config: CurationConfig, stats: BuildStats, rng: random.Random
) -> list[TrainingRecord]:
    order = list(range(len(pool)))
    rng.shuffle(order)
    pair_counts: Counter = Counter()
    quote_counts: Counter = Counter()
    keep = [False] * len(pool)
    for i in order:
        r = pool[i]
        pairs = {pair_key(f.claim, f.quote) for f in r.facts}
        quotes = {quote_key(f.quote) for f in r.facts} - {""}
        if any(pair_counts[p] >= config.max_pair_repeats for p in pairs):
            _drop(stats, r, "pair_repeat_cap")
            continue
        if any(quote_counts[q] >= config.max_quote_repeats for q in quotes):
            _drop(stats, r, "quote_repeat_cap")
            continue
        keep[i] = True
        pair_counts.update(pairs)
        quote_counts.update(quotes)
    stats.caps["max_pair_repeats"] = config.max_pair_repeats
    stats.caps["max_quote_repeats"] = config.max_quote_repeats
    return [r for i, r in enumerate(pool) if keep[i]]


def _cap_other(
    pool: list[TrainingRecord], config: CurationConfig, stats: BuildStats, rng: random.Random
) -> list[TrainingRecord]:
    def other_count(r: TrainingRecord) -> int:
        return sum(1 for f in r.facts if f.fact_type == OTHER_FACT_TYPE)

    total = sum(len(r.facts) for r in pool)
    other = sum(other_count(r) for r in pool)
    before = other / total if total else 0.0
    candidates = [i for i, r in enumerate(pool) if other_count(r)]
    tiebreak = {i: rng.random() for i in candidates}
    candidates.sort(key=lambda i: (-other_count(pool[i]) / len(pool[i].facts), tiebreak[i]))
    dropped: set[int] = set()
    for i in candidates:
        if total == 0 or other / total <= config.other_cap:
            break
        dropped.add(i)
        other -= other_count(pool[i])
        total -= len(pool[i].facts)
        _drop(stats, pool[i], "other_type_cap")
    stats.caps["other"] = {
        "ratio": config.other_cap,
        "share_before": round(before, 4),
        "share_after": round(other / total, 4) if total else 0.0,
        "records_dropped": len(dropped),
    }
    return [r for i, r in enumerate(pool) if i not in dropped]


def _cap_negatives(
    pool: list[TrainingRecord], config: CurationConfig, stats: BuildStats, rng: random.Random
) -> list[TrainingRecord]:
    negatives = [i for i, r in enumerate(pool) if r.is_negative]
    positives = len(pool) - len(negatives)
    max_neg = int(config.negative_cap * positives / (1 - config.negative_cap))
    excess = max(0, len(negatives) - max_neg)
    dropped = set(rng.sample(negatives, k=excess)) if excess else set()
    for i in dropped:
        _drop(stats, pool[i], "negative_cap")
    stats.caps["negatives"] = {
        "ratio": config.negative_cap,
        "share_before": round(len(negatives) / len(pool), 4) if pool else 0.0,
        "records_dropped": len(dropped),
    }
    return [r for i, r in enumerate(pool) if i not in dropped]


def _cap_corpus_share(
    pool: list[TrainingRecord], config: CurationConfig, stats: BuildStats, rng: random.Random
) -> list[TrainingRecord]:
    counts = Counter(r.corpus for r in pool)
    applied: dict[str, int] = {}
    if len(counts) < 2:
        stats.caps["corpus_share"] = {
            "ratio": config.max_corpus_share,
            "applied": "skipped: fewer than two corpora",
        }
        return pool
    drop_ids: set[int] = set()
    while True:
        corpus, n = counts.most_common(1)[0]
        rest = sum(counts.values()) - n
        if n <= config.max_corpus_share * (n + rest) or config.max_corpus_share >= 1.0:
            break
        keep_n = int(config.max_corpus_share * rest / (1 - config.max_corpus_share))
        members = sorted(
            (r for r in pool if r.corpus == corpus and id(r) not in drop_ids),
            key=lambda r: r.record_id,
        )
        victims = rng.sample(members, k=n - keep_n)
        for r in victims:
            drop_ids.add(id(r))
            _drop(stats, r, "corpus_share_cap")
        applied[corpus] = applied.get(corpus, 0) + len(victims)
        counts[corpus] = keep_n
    stats.caps["corpus_share"] = {"ratio": config.max_corpus_share, "records_dropped": applied}
    return [r for r in pool if id(r) not in drop_ids]


def _bucket(split_key: str, val_fraction: float) -> str:
    """90/10 split key: sha256 of the group's split key, low byte mod 100
    (unchanged from the original builder, so a lone source keeps the
    bucket it always had)."""
    digest = hashlib.sha256(split_key.encode("utf-8")).digest()
    return "val" if (digest[0] % 100) < int(val_fraction * 100) else "train"


def build_training_set(
    var_root: Path,
    *,
    records: list[TrainingRecord] | None = None,
    seed: int = CAP_RANDOM_SEED,
    config: CurationConfig | None = None,
    contamination_refs: contamination.ContaminationReferences | None = None,
) -> tuple[list[TrainingRecord], list[TrainingRecord], BuildStats]:
    """Returns `(train, val, stats)`. `records` overrides reading from
    `var_root` (tests).

    `contamination_refs` is `None` by default so every caller that
    predates WP8 -- including every existing test in this file -- keeps
    working unchanged, with the gate simply skipped. The hard "refuse to
    run without signatures" behavior
    (`contamination.ContaminationSignaturesMissing`) lives in `main()`,
    at the point where `var/contamination/` either does or doesn't exist
    on disk, not here in the disk-free pure function."""
    config = config or CurationConfig()
    stats = BuildStats()
    stats.contamination_enabled = contamination_refs is not None
    rng = random.Random(seed)
    if records is None:
        paths = list_input_files(var_root)
        stats.inputs = describe_inputs(var_root, paths)
        stats.v2_worlds_skipped = v2_world_files(var_root)[1]
        source_records = load_records(paths)
    else:
        source_records = list(records)
    stats.input_records = len(source_records)
    for r in source_records:
        stats.input_records_per_corpus[r.corpus] += 1
        stats.input_facts_per_corpus[r.corpus] += len(r.facts)
        if r.label_audit is not None:
            stats.audited_records_per_corpus[r.corpus] += 1
        if r.corpus == SYNTHETIC_V2_CORPUS:
            world = stats.v2_worlds.setdefault(
                r.world_id or "(none)",
                {"split": r.world_split, "input": 0, "train": 0, "val": 0},
            )
            world["input"] += 1

    # 0-3: contamination, provenance, verification, label completeness
    pool: list[TrainingRecord] = []
    for r in source_records:
        reason = None
        if contamination_refs is not None:
            reason = _contamination_gate(r, contamination_refs, stats)
        if reason is None:
            reason = _provenance_reason(r, stats)
        if reason is None and not all(_fact_has_quote(f) for f in r.facts):
            reason = "fact_without_quote"
        if reason is None and not all(_fact_is_verified(f, r.text) for f in r.facts):
            reason = "quote_verification_failed"
        if reason is None and _gate_made_negative(r):
            reason = "negative_after_gate_drops"
        if (
            reason is None
            and config.drop_partial_records
            and not r.is_negative
            and _unresolved_facts(r) > 0
        ):
            reason = "incomplete_labels" if r.label_audit is not None else "partial_labels"
        if reason is None:
            pool.append(r)
        else:
            _drop(stats, r, reason)

    # 4-6: dedup, conflicts, repeats
    pool, split_keys = _dedup_documents(pool, config, stats)
    pool = _drop_conflicting_labels(pool, config, stats)
    pool = _apply_repeat_caps(pool, config, stats, rng)

    # 7-9: mix caps
    pool = _cap_other(pool, config, stats, rng)
    pool = _cap_negatives(pool, config, stats, rng)
    pool = _cap_corpus_share(pool, config, stats, rng)

    # 10: split
    train: list[TrainingRecord] = []
    val: list[TrainingRecord] = []
    for r in pool:
        if r.corpus == SYNTHETIC_V2_CORPUS:
            # provenance already dropped every v2 record without a train or validation world
            bucket = V2_SPLIT_BUCKET[r.world_split or ""]
            stats.v2_worlds[r.world_id or "(none)"][bucket] += 1
        else:
            bucket = _bucket(split_keys.get(id(r), r.source_id), config.val_fraction)
        (val if bucket == "val" else train).append(r)
        if r.is_negative:
            stats.negatives_total += 1
            if bucket == "val":
                stats.negatives_val += 1
            else:
                stats.negatives_train += 1
    _final_stats(pool, stats)
    stats.train_records, stats.val_records = len(train), len(val)
    stats.train_facts = sum(len(r.facts) for r in train)
    stats.val_facts = sum(len(r.facts) for r in val)
    return train, val, stats


def _final_stats(pool: list[TrainingRecord], stats: BuildStats) -> None:
    chars: dict[str, list[int]] = defaultdict(list)
    words: Counter = Counter()
    pos_facts: Counter = Counter()
    pair_counts: Counter = Counter()
    quote_counts: Counter = Counter()
    for r in pool:
        stats.records_per_corpus[r.corpus] += 1
        stats.records_per_prompt[prompt_stamp_key(r)] += 1
        stats.facts_per_corpus[r.corpus] += len(r.facts)
        if r.is_negative:
            stats.negatives_per_corpus[r.corpus] += 1
        else:
            words[r.corpus] += len(r.text.split())
            pos_facts[r.corpus] += len(r.facts)
        chars[r.corpus].append(len(r.text))
        for f in r.facts:
            stats.facts_per_type[f.fact_type] += 1
            stats.facts_per_corpus_type[r.corpus][f.fact_type] += 1
        for t in r.teachers:
            stats.records_per_teacher[t] += 1
        stats.teachers.update(r.teachers)
        lost = _unresolved_facts(r)
        if lost and not r.is_negative:
            stats.partial_records_per_corpus[r.corpus] += 1
            stats.est_dropped_facts_per_corpus[r.corpus] += lost
        stats.regated_facts_per_corpus[r.corpus] += sum(1 for f in r.facts if f.regated)
        stats.repaired_facts_per_corpus[r.corpus] += sum(1 for f in r.facts if f.repaired)
        if r.label_audit is not None:
            stats.withdrawn_facts_per_corpus[r.corpus] += r.label_audit.withdrawn
        pair_counts.update({pair_key(f.claim, f.quote) for f in r.facts})
        quote_counts.update({quote_key(f.quote) for f in r.facts} - {""})
    stats.median_chars_per_corpus = {c: int(statistics.median(v)) for c, v in chars.items()}
    stats.facts_per_600_words = {
        c: round(pos_facts[c] / words[c] * 600, 1) for c in words if words[c]
    }
    stats.max_pair_repeat_out = max(pair_counts.values(), default=0)
    stats.max_quote_repeat_out = max(quote_counts.values(), default=0)


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------


def _write_jsonl(path: Path, records: list[TrainingRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(record.model_dump_json())
            f.write("\n")


def _write_jsonl_dicts(path: Path, entries: list[dict]) -> None:
    """Same shape as `_write_jsonl`, for plain dict log entries
    (`BuildStats.contamination_dropped`/`contamination_review`) rather
    than `TrainingRecord`s. Written whenever the contamination gate ran
    at all, even with zero entries, so the file's presence tells a
    reader the gate was on; skipped entirely under
    `--no-contamination-check`, where its absence says the gate was off."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry))
            f.write("\n")


def _prompt_hashes() -> dict[str, str]:
    """sha256 of the teacher system prompt for each corpus context, as
    this commit builds it (records do not store the prompt they were
    labeled under)."""
    try:
        from portolan.extract.prompt import build_system_prompt

        spec = importlib.util.spec_from_file_location(
            "silver_contexts_for_card", Path(__file__).resolve().parent / "silver_contexts.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return {
            c: hashlib.sha256(build_system_prompt(company_context=ctx).encode("utf-8")).hexdigest()[
                :16
            ]
            for c, ctx in sorted(module.CORPUS_CONTEXTS.items())
        }
    except Exception as exc:  # noqa: BLE001 - the card still writes without it
        return {"error": f"{type(exc).__name__}: {exc}"}


def _manifest(stats: BuildStats, config: CurationConfig, seed: int, now: str) -> dict:
    return {
        "created": now,
        "seed": seed,
        "config": asdict(config),
        "inputs": stats.inputs,
        "input_records": stats.input_records,
        "train_records": stats.train_records,
        "val_records": stats.val_records,
        "train_facts": stats.train_facts,
        "val_facts": stats.val_facts,
        "records_per_corpus": dict(stats.records_per_corpus),
        "facts_per_corpus": dict(stats.facts_per_corpus),
        "facts_per_type": dict(stats.facts_per_type),
        "negatives": {
            "total": stats.negatives_total,
            "train": stats.negatives_train,
            "val": stats.negatives_val,
            "cap_ratio": config.negative_cap,
        },
        "other_fact_cap_ratio": config.other_cap,
        "dropped_by_reason": dict(stats.dropped_by_reason),
        "dropped_facts_by_reason": dict(stats.dropped_facts_by_reason),
        "dropped_by_reason_corpus": {k: dict(v) for k, v in stats.dropped_by_reason_corpus.items()},
        "dropped_by_teacher": dict(stats.dropped_by_teacher),
        "label_audit": {
            "audited_input_records": dict(stats.audited_records_per_corpus),
            "regated_facts_out": dict(stats.regated_facts_per_corpus),
            "repaired_facts_out": dict(stats.repaired_facts_per_corpus),
            "withdrawn_facts_out": dict(stats.withdrawn_facts_per_corpus),
        },
        "provenance_violations": dict(stats.provenance_violations),
        "caps": stats.caps,
        "dedup": {
            "duplicate_groups": stats.duplicate_groups,
            "near_sibling_pairs": stats.near_sibling_pairs,
            "split_groups_with_siblings": stats.split_groups_multi,
        },
        "teachers": sorted(stats.teachers),
        "contamination": {
            "enabled": stats.contamination_enabled,
            "hard_gate_corpora": sorted(CONTAMINATION_HARD_GATE_CORPORA),
            "dropped_total": sum(stats.contamination_per_corpus.values()),
            "dropped_by_reason": dict(stats.contamination_by_reason),
            "dropped_per_corpus": dict(stats.contamination_per_corpus),
            "review_only_total": sum(stats.contamination_review_per_corpus.values()),
            "review_only_by_reason": dict(stats.contamination_review_by_reason),
            "review_only_per_corpus": dict(stats.contamination_review_per_corpus),
        },
        "records_per_prompt": dict(stats.records_per_prompt),
        "v2_worlds": {
            "split_rule": "every record of a world goes to its world's split; test worlds never",
            "worlds": dict(sorted(stats.v2_worlds.items())),
            "skipped": dict(sorted(stats.v2_worlds_skipped.items())),
        },
    }


def _pct(n: int, d: int) -> str:
    return f"{n / d:.0%}" if d else "n/a"


def write_data_card(
    path: Path, stats: BuildStats, config: CurationConfig, seed: int, now: str
) -> None:
    total_records = stats.train_records + stats.val_records
    total_facts = stats.train_facts + stats.val_facts
    lines: list[str] = []
    add = lines.append
    add(f"# Data card: {path.parent.name}")
    add("")
    add(f"Built {now} by `scripts/data/build_training_set.py`, seed {seed}.")
    add("Every number here was computed by the builder from the files listed under Inputs.")
    add("")
    add("## Totals")
    add("")
    add("| split | records | facts | negatives |")
    add("|---|---|---|---|")
    add(f"| train | {stats.train_records} | {stats.train_facts} | {stats.negatives_train} |")
    add(f"| val | {stats.val_records} | {stats.val_facts} | {stats.negatives_val} |")
    add(
        f"| all | {total_records} | {total_facts} | {stats.negatives_total} ({_pct(stats.negatives_total, total_records)}) |"
    )
    add("")
    add(
        f"Input records: {stats.input_records}. Kept: {total_records} ({_pct(total_records, stats.input_records)})."
    )
    add("")
    add("## By corpus")
    add("")
    add(
        "| corpus | input records | kept records | share of records | facts | share of facts | negatives | median chars | facts per 600 words | positives with gate-dropped facts | est. facts the gate dropped |"
    )
    add("|---|---|---|---|---|---|---|---|---|---|---|")
    corpora = sorted(set(stats.input_records_per_corpus) | set(stats.records_per_corpus))
    for c in corpora:
        kept = stats.records_per_corpus.get(c, 0)
        add(
            f"| {c} | {stats.input_records_per_corpus.get(c, 0)} | {kept} | {_pct(kept, total_records)} | "
            f"{stats.facts_per_corpus.get(c, 0)} | {_pct(stats.facts_per_corpus.get(c, 0), total_facts)} | "
            f"{stats.negatives_per_corpus.get(c, 0)} | {stats.median_chars_per_corpus.get(c, '-')} | "
            f"{stats.facts_per_600_words.get(c, '-')} | {stats.partial_records_per_corpus.get(c, 0)} | "
            f"{stats.est_dropped_facts_per_corpus.get(c, 0)} |"
        )
    add("")
    add("## Label completeness")
    add("")
    add(
        "A record trains only when every fact its teacher emitted ended kept, re-admitted by the fixed gate, repaired by the teacher, or withdrawn by the teacher (`scripts/data/repair_silver.py`, `label_audit` on the record). Counts of excluded records are input records; fact counts are for the kept records."
    )
    add("")
    add(
        "| corpus | audited input records | excluded: incomplete (audited) | excluded: partial (unaudited) | excluded: gate-made negative | re-admitted facts | repaired facts | withdrawn facts |"
    )
    add("|---|---|---|---|---|---|---|---|")
    for c in corpora:
        add(
            f"| {c} | {stats.audited_records_per_corpus.get(c, 0)} | "
            f"{stats.dropped_by_reason_corpus['incomplete_labels'].get(c, 0)} | "
            f"{stats.dropped_by_reason_corpus['partial_labels'].get(c, 0)} | "
            f"{stats.dropped_by_reason_corpus['negative_after_gate_drops'].get(c, 0)} | "
            f"{stats.regated_facts_per_corpus.get(c, 0)} | {stats.repaired_facts_per_corpus.get(c, 0)} | "
            f"{stats.withdrawn_facts_per_corpus.get(c, 0)} |"
        )
    add("")
    add("## By fact type")
    add("")
    types = [t for t, _ in stats.facts_per_type.most_common()]
    add(
        "| fact type | facts | share | "
        + " | ".join(c for c in corpora if stats.facts_per_corpus.get(c))
        + " |"
    )
    add("|---|---|---|" + "---|" * len([c for c in corpora if stats.facts_per_corpus.get(c)]))
    for t in types:
        per = " | ".join(
            str(stats.facts_per_corpus_type[c].get(t, 0))
            for c in corpora
            if stats.facts_per_corpus.get(c)
        )
        add(
            f"| {t} | {stats.facts_per_type[t]} | {_pct(stats.facts_per_type[t], total_facts)} | {per} |"
        )
    add("")
    thin = [t for t, n in sorted(stats.facts_per_type.items()) if n < THIN_TYPE_FACTS]
    add(
        f"Fact types with fewer than {THIN_TYPE_FACTS} facts: {', '.join(thin) if thin else 'none'}."
    )
    add("")
    add("## By teacher (kept records)")
    add("")
    add("| teacher | records |")
    add("|---|---|")
    for t, n in stats.records_per_teacher.most_common():
        add(f"| `{t}` | {n} |")
    add("")
    add("## Dropped records")
    add("")
    add("Each dropped record is counted once, under the first rule it failed, in pipeline order.")
    add("")
    add("| reason | records | facts | by corpus |")
    add("|---|---|---|---|")
    for reason, n in stats.dropped_by_reason.most_common():
        per = ", ".join(f"{c} {k}" for c, k in stats.dropped_by_reason_corpus[reason].most_common())
        add(f"| {reason} | {n} | {stats.dropped_facts_by_reason[reason]} | {per} |")
    add("")
    add("Provenance checks failed (a record can fail several):")
    add("")
    add("| check | records |")
    add("|---|---|")
    for check, n in stats.provenance_violations.most_common():
        add(f"| {check} | {n} |")
    add("")
    add("Records dropped per teacher id not on the allowlist:")
    add("")
    add("| teacher | records | note |")
    add("|---|---|---|")
    for t, n in stats.dropped_by_teacher.most_common():
        note = next(
            (v for k, v in EXCLUDED_TEACHER_NOTES.items() if t.lower().startswith(k)),
            "no provenance: no teacher on the record"
            if t == NO_TEACHER
            else "not on the allowlist",
        )
        add(f"| `{t}` | {n} | {note} |")
    if not stats.dropped_by_teacher:
        add("| (none) | 0 | |")
    add("")
    add("## Contamination (WP8)")
    add("")
    if not stats.contamination_enabled:
        add("Disabled for this run (`--no-contamination-check`). No record was gated on it.")
    else:
        add(
            f"Hard gate for every corpus: MinHash near-duplicate, exact 13-word overlap, ERB's "
            f"canary string. Hard gate ONLY for sources this project generates "
            f"(`{'`, `'.join(sorted(CONTAMINATION_HARD_GATE_CORPORA))}`): a blocklisted-name hit. "
            f"On every other corpus a name hit is review-only -- counted here and in "
            f"`contamination_review.jsonl`, never dropped by itself (coordinator review, "
            f"2026-09-22: an ordinary English word that is also a blocklisted name says nothing "
            f"about a real public-corpus record)."
        )
        add("")
        add("Hard-gate drops, by reason:")
        add("")
        add("| reason | hits | by corpus (records dropped) |")
        add("|---|---|---|")
        for reason, n in stats.contamination_by_reason.most_common():
            per = ", ".join(
                f"{c} {k}"
                for c, k in sorted(
                    stats.dropped_by_reason_corpus.get(f"contaminated_{reason}", {}).items()
                )
            )
            add(f"| {reason} | {n} | {per} |")
        if not stats.contamination_by_reason:
            add("| (none) | 0 | |")
        add("")
        add(
            f"Records dropped for contamination: {sum(stats.contamination_per_corpus.values())} "
            f"total. By corpus: "
            + (
                ", ".join(f"{c} {n}" for c, n in sorted(stats.contamination_per_corpus.items()))
                or "(none)"
            )
            + "."
        )
        add("")
        add(
            "Review-only name hits (real public corpora -- counted, NOT dropped, listed in "
            "`contamination_review.jsonl`):"
        )
        add("")
        add("| corpus | records with a review-only name hit |")
        add("|---|---|")
        for c, n in sorted(stats.contamination_review_per_corpus.items()):
            add(f"| {c} | {n} |")
        if not stats.contamination_review_per_corpus:
            add("| (none) | 0 |")
    add("")
    add("## v2 worlds (split by world)")
    add("")
    add(
        "Every record of a v2 world goes to its world's split (spec section 6); test worlds and "
        "the held-out vertical never reach train or validation."
    )
    add("")
    add("| world | split | input records | train | val |")
    add("|---|---|---|---|---|")
    for world_id, row in sorted(stats.v2_worlds.items()):
        add(f"| {world_id} | {row['split']} | {row['input']} | {row['train']} | {row['val']} |")
    if not stats.v2_worlds:
        add("| (none) | | 0 | 0 | 0 |")
    for world_dir, reason in sorted(stats.v2_worlds_skipped.items()):
        add(f"- skipped `{world_dir}`: {reason}")
    add("")
    add("## Caps and dedup")
    add("")
    add(
        f"- Document dedup: 5-word shingles, MinHash 128 x LSH 32x4, exact Jaccard >= {config.dedup_jaccard}: {stats.duplicate_groups} groups, {stats.dropped_by_reason.get('near_duplicate_document', 0)} records removed."
    )
    add(
        f"- Split grouping: Jaccard >= {config.split_group_jaccard}: {stats.near_sibling_pairs} sibling pairs, {stats.split_groups_multi} multi-record groups kept on one side of the split."
    )
    add(
        f"- Conflicting labels: {stats.caps.get('conflicting_quotes', 0)} quotes carried two or more fact types ({stats.caps.get('conflicting_quotes_resolved_by_majority', 0)} kept at a majority type of at least {config.conflict_majority:.0%}); {stats.dropped_by_reason.get('conflicting_type_for_same_quote', 0)} records removed."
    )
    add(
        f"- Claim+quote pair repeats <= {config.max_pair_repeats}: {stats.dropped_by_reason.get('pair_repeat_cap', 0)} records removed; max in output {stats.max_pair_repeat_out}."
    )
    add(
        f"- Quote repeats <= {config.max_quote_repeats}: {stats.dropped_by_reason.get('quote_repeat_cap', 0)} records removed; max in output {stats.max_quote_repeat_out}."
    )
    oc = stats.caps.get("other", {})
    add(
        f"- `other` <= {config.other_cap:.0%} of facts: before {oc.get('share_before', 0):.1%}, after {oc.get('share_after', 0):.1%}, {oc.get('records_dropped', 0)} records removed."
    )
    nc = stats.caps.get("negatives", {})
    add(
        f"- Negatives <= {config.negative_cap:.0%} of records: before {nc.get('share_before', 0):.1%}, {nc.get('records_dropped', 0)} records removed; final {_pct(stats.negatives_total, total_records)}."
    )
    cc = stats.caps.get("corpus_share", {})
    add(
        f"- Corpus share <= {config.max_corpus_share:.0%} of records: {cc.get('applied') or cc.get('records_dropped') or 'no corpus over the cap'}."
    )
    add(
        f"- Incomplete and partial-label records dropped: {'yes' if config.drop_partial_records else 'no'} (`--keep-partial-records` keeps them)."
    )
    add("")
    add("## Prompt")
    add("")
    add(
        "Records labeled from 2026-09-22 carry the variant and sha256 of the system prompt their teacher saw. `unstamped` records were labeled earlier, with the prompt whose examples named Halvard exam entities; which version is not recorded."
    )
    add("")
    add("| prompt (variant sha256[:16]) | records |")
    add("|---|---|")
    for key, count in sorted(stats.records_per_prompt.items()):
        add(f"| `{key}` | {count} |")
    add("")
    add(
        "A labeling run at this commit stamps `portolan.extract.prompt.build_system_prompt(company_context=CORPUS_CONTEXTS[corpus])` (variant `v6-type-precision`, production's default). First 16 hex of its sha256, per corpus context:"
    )
    add("")
    add("| corpus | system prompt sha256[:16] |")
    add("|---|---|")
    for c, h in _prompt_hashes().items():
        add(f"| {c} | `{h}` |")
    add("")
    add("## Inputs")
    add("")
    add("| file | lines | sha256 |")
    add("|---|---|---|")
    for item in stats.inputs:
        add(f"| `{item['path']}` | {item['lines']} | `{item['sha256']}` |")
    if not stats.inputs:
        add("| (records passed in memory) | | |")
    add("")
    add("## Settings")
    add("")
    for k, v in asdict(config).items():
        add(f"- `{k}`: {v}")
    add("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _print_summary(stats: BuildStats) -> None:
    print(f"input records: {stats.input_records}")
    print(f"train: records={stats.train_records} facts={stats.train_facts}")
    print(f"val:   records={stats.val_records} facts={stats.val_facts}")
    print(
        f"negatives: total={stats.negatives_total} (train={stats.negatives_train}, val={stats.negatives_val})"
    )
    print("records per corpus:")
    for corpus, count in sorted(stats.records_per_corpus.items()):
        print(f"  {corpus:<12} {count}")
    print("facts per type:")
    for fact_type, count in stats.facts_per_type.most_common():
        print(f"  {fact_type:<20} {count}")
    print("dropped by reason (records):")
    for reason, count in stats.dropped_by_reason.most_common():
        print(f"  {reason:<34} {count}")
    print("dropped per teacher:")
    for t, count in stats.dropped_by_teacher.most_common():
        print(f"  {t:<50} {count}")
    if stats.contamination_enabled:
        print("contamination, hard-gate hits by reason:")
        for reason, count in stats.contamination_by_reason.most_common():
            print(f"  {reason:<34} {count}")
        print("contamination, records dropped per corpus:")
        for corpus, count in sorted(stats.contamination_per_corpus.items()):
            print(f"  {corpus:<20} {count}")
        print("contamination, review-only name hits per corpus (NOT dropped):")
        for corpus, count in sorted(stats.contamination_review_per_corpus.items()):
            print(f"  {corpus:<20} {count}")


def main(argv: list[str] | None = None) -> int:
    defaults = CurationConfig()
    parser = argparse.ArgumentParser(description="Build and curate the fine-tuning set.")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument(
        "--var-root", type=Path, default=VAR_ROOT, help="var/ directory (default: repo var/)"
    )
    parser.add_argument("--seed", type=int, default=CAP_RANDOM_SEED)
    parser.add_argument("--negative-cap", type=float, default=defaults.negative_cap)
    parser.add_argument("--other-cap", type=float, default=defaults.other_cap)
    parser.add_argument("--max-corpus-share", type=float, default=defaults.max_corpus_share)
    parser.add_argument("--max-pair-repeats", type=int, default=defaults.max_pair_repeats)
    parser.add_argument("--max-quote-repeats", type=int, default=defaults.max_quote_repeats)
    parser.add_argument("--dedup-jaccard", type=float, default=defaults.dedup_jaccard)
    parser.add_argument("--split-group-jaccard", type=float, default=defaults.split_group_jaccard)
    parser.add_argument("--val-fraction", type=float, default=defaults.val_fraction)
    parser.add_argument("--conflict-majority", type=float, default=defaults.conflict_majority)
    parser.add_argument(
        "--keep-partial-records",
        dest="drop_partial_records",
        action="store_false",
        help="keep positives with a teacher fact that ended neither kept, repaired nor withdrawn",
    )
    parser.add_argument(
        "--no-contamination-check",
        action="store_true",
        help="skip the WP8 contamination gate when var/contamination/ has no signatures, or "
        "stale ones -- must be passed explicitly; the default is to refuse the run, and "
        "scripts/train/mlx_lora.py refuses a set built this way",
    )
    args = parser.parse_args(argv)
    config = CurationConfig(
        negative_cap=args.negative_cap,
        other_cap=args.other_cap,
        max_corpus_share=args.max_corpus_share,
        max_pair_repeats=args.max_pair_repeats,
        max_quote_repeats=args.max_quote_repeats,
        dedup_jaccard=args.dedup_jaccard,
        split_group_jaccard=args.split_group_jaccard,
        val_fraction=args.val_fraction,
        conflict_majority=args.conflict_majority,
        drop_partial_records=args.drop_partial_records,
    )
    assert_not_holdout(args.out)

    contamination_dir = args.var_root / "contamination"
    try:
        contamination_refs = contamination.load_references_or_refuse(
            contamination_dir, allow_missing=args.no_contamination_check
        )
    except contamination.ContaminationSignaturesMissing as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if contamination_refs is None:
        print(
            "WARNING: contamination check disabled (--no-contamination-check) -- "
            "records are NOT gated against the Halvard exam corpus or ERB.",
            file=sys.stderr,
        )

    train, val, stats = build_training_set(
        args.var_root, seed=args.seed, config=config, contamination_refs=contamination_refs
    )

    now = clock.now().isoformat()
    _write_jsonl(args.out / "train.jsonl", train)
    _write_jsonl(args.out / "val.jsonl", val)
    manifest = _manifest(stats, config, args.seed, now)
    # The trainer refuses a train/val file whose hash is not this one: no hand-made
    # or edited set trains next to a manifest this builder wrote.
    manifest["outputs"] = {
        name: {"sha256": hashlib.sha256((args.out / name).read_bytes()).hexdigest()}
        for name in ("train.jsonl", "val.jsonl")
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    write_data_card(args.out / "data_card.md", stats, config, args.seed, now)
    if contamination_refs is not None:
        _write_jsonl_dicts(args.out / "contamination_dropped.jsonl", stats.contamination_dropped)
        _write_jsonl_dicts(args.out / "contamination_review.jsonl", stats.contamination_review)

    print(f"wrote {args.out}/train.jsonl, val.jsonl, manifest.json, data_card.md")
    _print_summary(stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
