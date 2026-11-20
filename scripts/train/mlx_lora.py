"""Local LoRA fine-tune of a small student extractor with `mlx-lm`
(fine-tune shakedown, 2026-09-22; write-up in
`01-spec/finetune-shakedown-2026-09-22.md`).

Two phases, one command:

1. **Convert** a training set built by `scripts/data/build_training_set.py`
   (`train.jsonl`/`val.jsonl` of `TrainingRecord`s) into MLX-LM chat
   JSONL (`{"messages": [system, user, assistant]}`) with the one SFT
   formatter, `portolan.training.sft_format.
   production_sft_messages_for_record`: the system prompt the teacher saw
   (its corpus's company context, its variant, checked against the
   record's sha256) plus the JSON-schema block `MLXClient` appends
   (`portolan.llm.mlx_local.effective_system_prompt`), `build_user_message
   (envelope)`, and the teacher's own answer with no placeholders. The
   envelope is the one stored with the record, or, for a record without
   one, rebuilt from its source file the way silver labeling built it
   (`FileConnector` + `portolan.training.pipeline.document_envelopes`).
   Records without the teacher's summary, sentiment and confidence (every
   record written before 2026-09-22) are skipped, never filled.
2. **Train** a LoRA adapter with `mlx_lm`'s own LoRA layers and trainer.
   Writes `adapters/` plus `manifest.json` (base model id and license,
   training-set path and sha256, hyperparameters, git sha, wall time, peak
   memory) under `var/models/<run-id>/`.

**Prefix mode `frozen_base` (the M1 default).** The system prompt plus
schema is about 7,800 tokens and must be the same for every example (one
company context; `single_system_prompt` refuses a mixed set). Ordinary
LoRA over the full ~8,500-token sequence measured 11.4 GB peak and
135-274 s per step on the 16 GB M1 (attention backward is quadratic), so
this mode encodes the shared prefix once with the base model and trains
the adapter on everything after it (the user message and the answer),
attending to that frozen prefix. The adapter is served the same way
(`portolan.llm.mlx_local.MLXClient` reads `portolan_prefix_mode` from
`adapter_config.json`). On a GPU with room, `--prefix-mode standard` is
ordinary full-sequence LoRA.

**Hold-out.** Halvard (`corpus/out/`, `corpus/truth-table.yaml`) is the
exam and EnterpriseRAG-Bench is off limits: `assert_training_path_allowed`
refuses either as a training input, and every record is checked for any
13-word run shared with the exam text (`exam_shingles`, the one read of
`corpus/out/` this script makes, for comparison only); a hit excludes the
record.

**The builder's set, gated.** `main` refuses a set whose `manifest.json` is
missing, records `contamination.enabled` other than true, or does not hash to
its `train.jsonl`/`val.jsonl` (`assert_built_and_gated`), and re-runs the
builder's provenance check on every record (`assert_provenance`).

Usage (Apple Silicon, `uv sync --group mlx`):
    uv run --group mlx python scripts/train/mlx_lora.py \\
        --train-set var/data/training/2026-09-22-regate --run-id <id> --iters 150
    ... --convert-only   # phase 1 only, no MLX needed beyond the tokenizer check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import resource
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from portolan.extract.schema import (  # noqa: E402
    QUOTE_REQUIRED_TYPES,
    SourceEnvelope,
    SourceType,
    extraction_result_json_schema,
)
from portolan.llm.mlx_local import (  # noqa: E402
    PREFIX_MODE_FROZEN_BASE,
    PREFIX_MODE_STANDARD,
    effective_system_prompt,
)
from portolan.training.pipeline import assert_not_holdout  # noqa: E402
from portolan.training.records import TrainingRecord, read_records  # noqa: E402
from portolan.training.sft_format import (  # noqa: E402
    LegacyRecordError,
    SystemPromptMismatchError,
    envelope_for_record,
    production_sft_messages_for_record,
)
from scripts.data.build_training_set import provenance_failures  # noqa: E402
from scripts.data.silver_contexts import CORPUS_CONTEXTS  # noqa: E402

DEFAULT_BASE_MODEL = "mlx-community/Qwen3-1.7B-4bit"
EXAM_ROOT = REPO_ROOT / "corpus" / "out"
EXAM_TRUTH_TABLE = REPO_ROOT / "corpus" / "truth-table.yaml"
SHINGLE_WORDS = 13  # GPT-3's contamination n-gram (01-spec/research/training-data-recipes)

# saas_sales is out: its dialogue is GPT-4 text (build_training_set.CORPUS_DOCUMENT_TEXT),
# which the provenance rule excludes until the maintainer decides otherwise.
ALLOWED_CORPORA = frozenset(
    {"fda_maude", "ami", "enron", "ubuntu_irc", "cfpb", "synthetic", "synthetic_v2"}
)
FILE_CORPORA = frozenset({"fda_maude", "ami", "enron", "cfpb", "saas_sales"})

MAX_QUOTE_WORDS = 40  # fact rule 5

_SEGMENT_RE = re.compile(r"#seg(\d+)of(\d+)$")
_LABEL_RE = re.compile(r"\[([^\]\n]+)\]")  # used with .match(text, pos); no ^ anchor
_WORD_RE = re.compile(r"[a-z0-9]+")


# --------------------------------------------------------------------------
# Guards
# --------------------------------------------------------------------------


def assert_training_path_allowed(path: Path) -> None:
    """Refuse the Halvard exam (`corpus/out/` anywhere in the resolved path,
    or the truth table) and EnterpriseRAG-Bench (any path component naming
    it) as a training input."""
    resolved = Path(path).expanduser().resolve()
    assert_not_holdout(resolved)
    if resolved.parts[-2:] == ("corpus", "truth-table.yaml"):
        raise RuntimeError(
            f"hold-out violation: {path} is the exam answer key, never training data"
        )
    for part in resolved.parts:
        folded = part.lower().replace("-", "").replace("_", "")
        if "enterpriserag" in folded or folded == "erb":
            raise RuntimeError(
                f"hold-out violation: {path} is EnterpriseRAG-Bench material, never training data"
            )


def assert_built_and_gated(train_set: Path, inputs: dict[str, Path]) -> dict[str, Any]:
    """The trainer's own door (guardrails review 2026-09-22, rows 30-31): refuse a set
    `scripts/data/build_training_set.py` did not write, wrote with the contamination gate
    off, or whose train/val files changed after it wrote them. Returns the manifest."""
    manifest_path = train_set / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(
            f"{manifest_path} is missing: train only on a set build_training_set.py wrote"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("contamination") or {}).get("enabled") is not True:
        raise RuntimeError(
            f"{train_set} was built without the contamination gate (contamination.enabled is "
            "not true in manifest.json); rebuild it with current signatures"
        )
    outputs = manifest.get("outputs") or {}
    for path in inputs.values():
        recorded = (outputs.get(path.name) or {}).get("sha256")
        if recorded is None:
            raise RuntimeError(
                f"manifest.json records no sha256 for {path.name}; rebuild the set with the "
                "current build_training_set.py"
            )
        if recorded != sha256_file(path):
            raise RuntimeError(
                f"{path} is not the file build_training_set.py wrote (sha256 differs from "
                "manifest.json)"
            )
    return manifest


def assert_provenance(records: list[TrainingRecord], path: Path) -> None:
    """Re-run the builder's provenance allowlist on every record: a set built under an
    older allowlist, or edited by hand, is refused whole."""
    for record in records:
        failures = provenance_failures(record)
        if failures:
            raise RuntimeError(
                f"{path}: {record.record_id} fails the builder's provenance check "
                f"({', '.join(failures)}); refusing to train on this set"
            )


def assert_git_ignored(path: Path) -> None:
    """Refuse to write model artifacts anywhere git would pick them up."""
    resolved = Path(path).resolve()
    root = next(
        (p for p in (resolved, *resolved.parents) if (p / ".git").exists()),
        None,
    )
    if root is None:
        return  # outside any repository: nothing can be committed by accident
    result = subprocess.run(
        ["git", "-C", str(root), "check-ignore", "-q", str(resolved.relative_to(root))],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"{path} is not gitignored in {root}; model artifacts must stay out")


# --------------------------------------------------------------------------
# Exam-overlap check
# --------------------------------------------------------------------------


def _shingles(text: str, n: int = SHINGLE_WORDS) -> set[tuple[str, ...]]:
    words = _WORD_RE.findall(text.lower())
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def exam_shingles(exam_root: Path = EXAM_ROOT, truth_table: Path = EXAM_TRUTH_TABLE) -> set:
    """Every 13-word run in the Halvard exam's files and answer key. The one
    sanctioned read of `corpus/out/` in this script: used only to REFUSE
    training records that share text with the exam, never as an input."""
    shingles: set[tuple[str, ...]] = set()
    paths = sorted(p for p in exam_root.rglob("*") if p.is_file())
    if truth_table.is_file():
        paths.append(truth_table)
    for path in paths:
        try:
            shingles |= _shingles(path.read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            continue
    return shingles


def exam_overlap(record: TrainingRecord, exam: set) -> int:
    """How many of the record's 13-word runs (text, claims, quotes) the
    exam also contains."""
    blob = "\n".join([record.text, *(f.claim for f in record.facts)])
    return len(_shingles(blob) & exam)


# --------------------------------------------------------------------------
# Envelope rebuild
# --------------------------------------------------------------------------

_KIND_LAYOUT = {
    "call": (SourceType.CALL, "calls", (".txt",)),
    "ticket": (SourceType.TICKET, "tickets", (".json",)),
    "email": (SourceType.EMAIL, "email", (".txt", ".eml")),
}


def segment_of(record_id: str) -> tuple[int, int] | None:
    match = _SEGMENT_RE.search(record_id)
    return (int(match.group(1)), int(match.group(2))) if match else None


def source_root(record: TrainingRecord, var_root: Path) -> Path | None:
    """Where the record's source file lives: `var/data/converted/<corpus>/`
    for the file-shaped real corpora, `var/data/synth/<batch>/` for a
    synthetic record (`record_id` is `<batch>:<source_id>`)."""
    if record.corpus in FILE_CORPORA:
        return var_root / "data" / "converted" / record.corpus
    if record.corpus == "synthetic" and ":" in record.record_id:
        return var_root / "data" / "synth" / record.record_id.split(":", 1)[0]
    if record.corpus == "ubuntu_irc":
        return var_root / "data" / "converted" / "ubuntu_irc"
    return None


def _fetch_file_source(root: Path, record: TrainingRecord):
    from portolan.connectors.base import SourceRef
    from portolan.connectors.file_connector import FileConnector

    layout = _KIND_LAYOUT.get(record.kind)
    if layout is None:
        return None
    source_type, subdir, suffixes = layout
    for suffix in suffixes:
        path = root / subdir / f"{record.source_id}{suffix}"
        if path.is_file():
            ref = SourceRef(
                source_id=record.source_id,
                source_type=source_type,
                updated_at=datetime.now(UTC),
                ref={"path": str(path.relative_to(root))},
            )
            return FileConnector(root).fetch(ref)
    return None


def _chat_raw_source(record: TrainingRecord):
    """A synthetic Slack/Teams day file: no connector reads these, so build
    the `RawSource` the way `portolan.training.pipeline` builds a chat
    window (`SourceType.CHAT`, one line per message, speakers as
    participants); `prepare_text` then gives it `[Ln]` line locators."""
    from portolan.ingest.models import Participant, RawSource

    stem = Path(record.source_id).stem
    try:
        source_date = date.fromisoformat(stem)
    except ValueError:
        return None
    speakers: dict[str, None] = {}
    for line in record.text.splitlines():
        name, sep, _ = line.partition(": ")
        if sep and name.strip():
            speakers.setdefault(name.strip(), None)
    return RawSource(
        source_id=record.source_id,
        source_type=SourceType.CHAT,
        source_date=source_date,
        participants=[Participant(name=n) for n in speakers],
        text=record.text,
        metadata={},
    )


class EnvelopeIndex:
    """Rebuilds `SourceEnvelope`s for records, caching per source."""

    def __init__(self, var_root: Path) -> None:
        self.var_root = var_root
        self._by_source: dict[tuple[str, str], list[SourceEnvelope] | None] = {}
        self._ubuntu: dict[str, Any] | None = None

    def _ubuntu_documents(self, wanted: set[str]) -> dict[str, Any]:
        if self._ubuntu is None:
            from portolan.training.pipeline import documents_for_corpus

            found: dict[str, Any] = {}
            root = self.var_root / "data" / "converted" / "ubuntu_irc"
            for document in documents_for_corpus("ubuntu_irc", root):
                if document.source_id in wanted:
                    found[document.source_id] = document
                    if len(found) == len(wanted):
                        break
            self._ubuntu = found
        return self._ubuntu

    def envelopes(self, record: TrainingRecord, ubuntu_wanted: set[str]) -> list | None:
        from portolan.training.pipeline import Document, document_envelopes

        key = (record.corpus, record.record_id.split("#", 1)[0])
        if key in self._by_source:
            return self._by_source[key]
        envelopes = None
        root = source_root(record, self.var_root)
        if root is not None:
            assert_training_path_allowed(root)
            raw = None
            if record.corpus == "ubuntu_irc":
                document = self._ubuntu_documents(ubuntu_wanted).get(record.source_id)
                raw = document.raw if document is not None else None
            elif record.kind == "chat":
                raw = _chat_raw_source(record)
            elif root.is_dir():
                raw = _fetch_file_source(root, record)
            if raw is not None:
                document = Document(source_id=record.source_id, raw=raw, kind=record.kind)
                envelopes = document_envelopes(document)
        self._by_source[key] = envelopes
        return envelopes


def match_envelope(record: TrainingRecord, envelopes: list) -> tuple[Any, str | None]:
    """The envelope a record was labeled from, or `(None, reason)`."""
    seg = segment_of(record.record_id)
    if seg is not None:
        index, total = seg
        for env in envelopes:
            if env.segment is not None and (env.segment.index, env.segment.total) == (index, total):
                return env, None if env.text == record.text else "text_mismatch"
        return None, "segment_not_found"
    if len(envelopes) != 1:
        return None, "segmented_source_without_segment_id"
    env = envelopes[0]
    if record.corpus != "synthetic" and env.text != record.text:
        return None, "text_mismatch"
    return env, None


# --------------------------------------------------------------------------
# Target
# --------------------------------------------------------------------------


def locator_for_quote(text: str, quote: str) -> str | None:
    """The `[label]` of the nearest locator-annotated line at or above the
    quote's first occurrence in the envelope text."""
    pos = text.find(quote)
    if pos < 0:
        return None
    line_start = text.rfind("\n", 0, pos) + 1
    while True:
        match = _LABEL_RE.match(text, line_start)
        if match:
            return match.group(1)
        if line_start == 0:
            return None
        line_start = text.rfind("\n", 0, line_start - 1) + 1


@dataclass
class ConversionStats:
    records_in: int = 0
    records_out: int = 0
    negatives_out: int = 0
    facts_in: int = 0
    facts_out: int = 0
    skipped: Counter = field(default_factory=Counter)
    fact_drops: Counter = field(default_factory=Counter)
    locators_remapped: int = 0
    exam_overlap_records: list[str] = field(default_factory=list)
    per_corpus: Counter = field(default_factory=Counter)

    def to_dict(self) -> dict[str, Any]:
        return {
            "records_in": self.records_in,
            "records_out": self.records_out,
            "negatives_out": self.negatives_out,
            "facts_in": self.facts_in,
            "facts_out": self.facts_out,
            "records_skipped_by_reason": dict(self.skipped),
            "facts_dropped_by_reason": dict(self.fact_drops),
            "locators_remapped": self.locators_remapped,
            "exam_overlap_records": self.exam_overlap_records,
            "records_out_per_corpus": dict(self.per_corpus),
        }


def gate_facts(
    record: TrainingRecord, envelope: SourceEnvelope, stats: ConversionStats
) -> TrainingRecord | None:
    """The record with only facts whose quote is in the prompt text, within
    the 40-word rule, and whose locator resolves (remapped to the line
    holding the quote when the teacher's label is not in the text), or
    `None` when every fact of a positive record had to go (turning it into
    a false negative would teach under-extraction)."""
    text = envelope.text
    kept = []
    for fact in record.facts:
        quote = fact.quote
        if quote is not None and quote not in text:
            stats.fact_drops["quote_not_in_prompt_text"] += 1
            continue
        if (
            fact.fact_type in {t.value for t in QUOTE_REQUIRED_TYPES}
            and quote is not None
            and len(quote.split()) > MAX_QUOTE_WORDS
        ):
            stats.fact_drops["quote_over_40_words"] += 1
            continue
        locator = fact.locator
        if f"[{locator}]" not in text:
            new_locator = locator_for_quote(text, quote) if quote else None
            if new_locator is None:
                stats.fact_drops["locator_not_resolvable"] += 1
                continue
            stats.locators_remapped += 1
            fact = fact.model_copy(update={"locator": new_locator})
        kept.append(fact)
    if record.facts and not kept:
        return None
    return record.model_copy(update={"facts": kept})


def company_context_for(record: TrainingRecord) -> str | None:
    """The company context the record's teacher was given: the record's own
    (a v2 world's vendor note), else its silver corpus's note, else none (the
    record's sha256 check catches a wrong guess)."""
    if record.company_context is not None:
        return record.company_context
    return CORPUS_CONTEXTS.get(record.corpus)


def serve_system(system_prompt: str) -> str:
    """What `MLXClient` sends for a system prompt: plus its schema block."""
    return effective_system_prompt(system_prompt, extraction_result_json_schema())


def single_system_prompt(rows: list[dict[str, Any]]) -> str:
    """The one system prompt `frozen_base` can encode as a shared prefix;
    raises when the converted rows carry more than one (several company
    contexts or variants in one training set)."""
    prompts = {row["messages"][0]["content"] for row in rows}
    if len(prompts) != 1:
        raise SystemExit(
            f"frozen_base needs one shared system prompt; this set has {len(prompts)} "
            "(one per company context). Train one context at a time, or use "
            "--prefix-mode standard."
        )
    return prompts.pop()


def convert_records(
    records: list[TrainingRecord],
    index: EnvelopeIndex,
    exam: set,
    stats: ConversionStats,
) -> list[dict[str, Any]]:
    ubuntu_wanted = {r.source_id for r in records if r.corpus == "ubuntu_irc"}
    out: list[dict[str, Any]] = []
    for record in records:
        stats.records_in += 1
        stats.facts_in += len(record.facts)
        if record.corpus not in ALLOWED_CORPORA:
            raise RuntimeError(f"{record.record_id}: corpus {record.corpus!r} is not allowed")
        if exam and exam_overlap(record, exam):
            stats.exam_overlap_records.append(record.record_id)
            stats.skipped["exam_overlap_13gram"] += 1
            continue
        if record.user_message_inputs is not None:
            envelope = envelope_for_record(record)
        else:
            envelopes = index.envelopes(record, ubuntu_wanted)
            if not envelopes:
                stats.skipped["source_not_rebuilt"] += 1
                continue
            envelope, reason = match_envelope(record, envelopes)
            if envelope is None:
                stats.skipped[reason or "no_envelope"] += 1
                continue
        gated = gate_facts(record, envelope, stats)
        if gated is None:
            stats.skipped["all_facts_dropped"] += 1
            continue
        try:
            messages = production_sft_messages_for_record(
                gated,
                company_context=company_context_for(record),
                envelope=envelope,
                serve_system=serve_system,
            )
        except LegacyRecordError:
            stats.skipped["no_teacher_answer"] += 1
            continue
        except SystemPromptMismatchError:
            stats.skipped["system_prompt_mismatch"] += 1
            continue
        n_facts = len(json.loads(messages[2]["content"])["facts"])
        stats.records_out += 1
        stats.facts_out += n_facts
        stats.negatives_out += 0 if n_facts else 1
        stats.per_corpus[record.corpus] += 1
        out.append({"messages": messages, "record_id": record.record_id, "corpus": record.corpus})
    return out


# --------------------------------------------------------------------------
# Small utilities
# --------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_state() -> dict[str, Any]:
    def _git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True, check=False
        ).stdout.strip()

    return {
        "sha": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(_git("status", "--porcelain")),
    }


def model_card_license(model_dir: Path) -> str | None:
    """`license:` from the local snapshot's README front matter."""
    readme = model_dir / "README.md"
    if not readme.is_file():
        return None
    text = readme.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return None
    front = text.split("---", 2)[1]
    for line in front.splitlines():
        if line.strip().startswith("license:"):
            return line.split(":", 1)[1].strip()
    return None


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps({"messages": row["messages"]}, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------
# Training (MLX imported lazily from here down)
# --------------------------------------------------------------------------


class FrozenPrefixKV:
    """Per-layer stand-in for an mlx-lm KV cache: fixed prefix keys/values
    (encoded once by the base model) followed by whatever the current
    forward pass adds. Stateless across calls, so gradient checkpointing's
    recompute sees the same thing twice."""

    def __init__(self, keys: Any, values: Any) -> None:
        self.keys, self.values = keys, values
        self.offset = keys.shape[2]

    def update_and_fetch(self, keys: Any, values: Any) -> tuple[Any, Any]:
        import mlx.core as mx

        batch = keys.shape[0]
        k = mx.broadcast_to(self.keys, (batch, *self.keys.shape[1:]))
        v = mx.broadcast_to(self.values, (batch, *self.values.shape[1:]))
        return mx.concatenate([k, keys], axis=2), mx.concatenate([v, values], axis=2)

    def make_mask(self, n: int, return_array: bool = False, window_size: int | None = None):
        from mlx_lm.models.cache import create_attention_mask

        return create_attention_mask(
            n, offset=self.offset, return_array=return_array, window_size=window_size
        )


def encode_prefix(model: Any, prefix_ids: list[int]) -> list[tuple[Any, Any]]:
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    cache = make_prompt_cache(model)
    tokens = mx.array(prefix_ids)[None]
    for start in range(0, len(prefix_ids), 2048):
        model(tokens[:, start : start + 2048], cache=cache)
        mx.eval([c.state for c in cache])
    n = len(prefix_ids)
    kv = [(c.keys[..., :n, :], c.values[..., :n, :]) for c in cache]
    mx.eval(kv)
    return kv


def tokenize_example(tokenizer: Any, messages: list[dict[str, str]]) -> tuple[list[int], int]:
    """Full token ids and where the answer starts (thinking disabled, the
    same prompt `MLXClient` sends)."""
    full = list(tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False))
    prompt = list(
        tokenizer.apply_chat_template(
            messages[:-1],
            add_generation_prompt=True,
            enable_thinking=False,
            tokenize=True,
            return_dict=False,
        )
    )
    if full[: len(prompt)] != prompt:
        raise RuntimeError("chat template: the prompt is not a prefix of the full conversation")
    return full, len(prompt)


def run_training(args: argparse.Namespace, run_dir: Path, rows: dict[str, list]) -> dict[str, Any]:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from huggingface_hub import snapshot_download
    from mlx.utils import tree_flatten
    from mlx_lm import load
    from mlx_lm.tuner.trainer import TrainingArgs, train
    from mlx_lm.tuner.utils import linear_to_lora_layers

    frozen = args.prefix_mode == PREFIX_MODE_FROZEN_BASE
    system = (
        single_system_prompt(rows["train"] + rows["valid"])
        if frozen
        else rows["train"][0]["messages"][0]["content"]
    )
    model_dir = Path(snapshot_download(args.base_model))
    model, tokenizer = load(str(model_dir))
    prefix_ids = list(
        tokenizer.apply_chat_template(
            [{"role": "system", "content": system}], tokenize=True, return_dict=False
        )
    )
    skip_len = len(prefix_ids) if frozen else 0

    datasets: dict[str, list[tuple[list[int], int]]] = {}
    length_skips: Counter = Counter()
    suffix_lengths: dict[str, list[int]] = {}
    for split in ("train", "valid"):
        examples = []
        for row in rows[split]:
            full, offset = tokenize_example(tokenizer, row["messages"])
            if frozen and full[: len(prefix_ids)] != prefix_ids:
                raise RuntimeError(f"{row['record_id']}: system prefix tokens differ")
            seq = full[skip_len:]
            if len(seq) > args.max_seq_tokens:
                length_skips[split] += 1
                continue
            examples.append((seq, offset - skip_len))
        datasets[split] = examples
        suffix_lengths[split] = sorted(len(s) for s, _ in examples)

    prefix_kv: list[tuple[Any, Any]] = []
    t_prefix = time.perf_counter()
    if frozen:
        prefix_kv = encode_prefix(model, prefix_ids)
    prefix_seconds = time.perf_counter() - t_prefix

    def loss_fn(model: Any, batch: Any, lengths: Any) -> tuple[Any, Any]:
        inputs, targets = batch[:, :-1], batch[:, 1:]
        cache = [FrozenPrefixKV(k, v) for k, v in prefix_kv] if frozen else None
        logits = model(inputs, cache=cache)
        steps = mx.arange(1, targets.shape[1] + 1)
        mask = mx.logical_and(steps >= lengths[:, 0:1], steps <= lengths[:, 1:])
        ce = nn.losses.cross_entropy(logits, targets) * mask
        ntoks = mask.sum()
        return ce.astype(mx.float32).sum() / ntoks, ntoks

    mx.random.seed(args.seed)
    model.freeze()
    lora_parameters = {"rank": args.rank, "dropout": 0.0, "scale": args.lora_scale}
    linear_to_lora_layers(model, args.num_layers, lora_parameters)
    trainable = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    total = sum(v.size for _, v in tree_flatten(model.parameters()))

    adapter_dir = run_dir / "adapters"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    adapter_config = {
        "model": args.base_model,
        "fine_tune_type": "lora",
        "num_layers": args.num_layers,
        "lora_parameters": lora_parameters,
        "portolan_prefix_mode": args.prefix_mode,
        "portolan_prefix_tokens": len(prefix_ids),
    }
    (adapter_dir / "adapter_config.json").write_text(json.dumps(adapter_config, indent=2) + "\n")

    training_args = TrainingArgs(
        batch_size=1,
        iters=args.iters,
        val_batches=args.val_batches,
        steps_per_report=args.steps_per_report,
        steps_per_eval=args.steps_per_eval,
        steps_per_save=args.iters,
        adapter_file=adapter_dir / "adapters.safetensors",
        max_seq_length=args.max_seq_tokens + 1,
        grad_checkpoint=True,
    )
    reports: list[dict[str, Any]] = []

    class _Callback:
        def on_train_loss_report(self, info: dict[str, Any]) -> None:
            reports.append({"kind": "train", **info})

        def on_val_loss_report(self, info: dict[str, Any]) -> None:
            reports.append({"kind": "val", **info})

    np_seed = args.seed
    import numpy as np

    np.random.seed(np_seed)
    optimizer = optim.Adam(learning_rate=args.learning_rate)
    t_train = time.perf_counter()
    train(
        model=model,
        optimizer=optimizer,
        train_dataset=datasets["train"],
        val_dataset=datasets["valid"],
        args=training_args,
        loss=loss_fn,
        training_callback=_Callback(),
    )
    train_seconds = time.perf_counter() - t_train

    return {
        "base_model_dir": str(model_dir),
        "base_model_revision": model_dir.name,
        "base_model_license": model_card_license(model_dir),
        "prefix_tokens": len(prefix_ids),
        "prefix_encode_seconds": round(prefix_seconds, 1),
        "examples_used": {k: len(v) for k, v in datasets.items()},
        "examples_skipped_over_max_seq_tokens": dict(length_skips),
        "sequence_tokens_after_prefix": {
            k: {"min": v[0], "median": v[len(v) // 2], "max": v[-1]} if v else None
            for k, v in suffix_lengths.items()
        },
        "trainable_parameters": trainable,
        "total_parameters": total,
        "train_seconds": round(train_seconds, 1),
        "peak_metal_memory_gb": round(mx.get_peak_memory() / 1e9, 3),
        "loss_reports": reports,
        "adapter_files": sorted(p.name for p in adapter_dir.iterdir()),
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--train-set", type=Path, required=True, help="dir with train.jsonl, val.jsonl")
    p.add_argument("--run-id", required=True)
    p.add_argument("--var-root", type=Path, default=REPO_ROOT / "var", help="where var/data lives")
    p.add_argument("--models-root", type=Path, default=None, help="default: <var-root>/models")
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument(
        "--prefix-mode",
        choices=[PREFIX_MODE_FROZEN_BASE, PREFIX_MODE_STANDARD],
        default=PREFIX_MODE_FROZEN_BASE,
    )
    p.add_argument("--iters", type=int, default=150)
    p.add_argument("--learning-rate", type=float, default=5e-5)
    p.add_argument("--num-layers", type=int, default=16, help="LoRA on the last N layers")
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--lora-scale", type=float, default=20.0)
    p.add_argument("--max-seq-tokens", type=int, default=2048, help="after the frozen prefix")
    p.add_argument("--val-batches", type=int, default=16)
    p.add_argument("--steps-per-eval", type=int, default=50)
    p.add_argument("--steps-per-report", type=int, default=10)
    p.add_argument("--seed", type=int, default=20260922)
    p.add_argument("--convert-only", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.perf_counter()
    train_set = args.train_set.expanduser().resolve()
    assert_training_path_allowed(train_set)
    models_root = (args.models_root or args.var_root / "models").expanduser().resolve()
    run_dir = models_root / args.run_id
    assert_git_ignored(models_root)
    if (run_dir / "adapters" / "adapters.safetensors").exists() and not args.overwrite:
        raise SystemExit(f"{run_dir} already has adapters; pass --overwrite or a new --run-id")

    inputs = {
        "train": train_set / "train.jsonl",
        "valid": train_set / "val.jsonl",
    }
    for path in inputs.values():
        assert_training_path_allowed(path)
        if not path.is_file():
            raise SystemExit(f"missing {path}")
    assert_built_and_gated(train_set, inputs)
    records_by_split = {split: list(read_records(path)) for split, path in inputs.items()}
    for split, records in records_by_split.items():
        assert_provenance(records, inputs[split])

    exam = exam_shingles()
    index = EnvelopeIndex(args.var_root.expanduser().resolve())
    stats_by_split: dict[str, ConversionStats] = {}
    rows: dict[str, list[dict[str, Any]]] = {}
    for split, records in records_by_split.items():
        stats = ConversionStats()
        rows[split] = convert_records(records, index, exam, stats)
        stats_by_split[split] = stats
        _write_jsonl(run_dir / "data" / f"{split}.jsonl", rows[split])
        print(f"{split}: {json.dumps(stats.to_dict())}")

    system_prompts = Counter(
        hashlib.sha256(row["messages"][0]["content"].encode("utf-8")).hexdigest()
        for split_rows in rows.values()
        for row in split_rows
    )
    conversion = {
        "system_prompt_sha256_rows": dict(system_prompts),
        "prompt": "portolan.training.sft_format.production_sft_messages_for_record: "
        "build_system_prompt(company_context=<the record's corpus context>, variant=<the "
        "record's>) checked against the record's sha256, + JSON_INSTRUCTION_TEMPLATE schema "
        "block; build_user_message(stored or rebuilt envelope); the teacher's own answer",
        "exam_shingles": len(exam),
        "splits": {k: v.to_dict() for k, v in stats_by_split.items()},
        "outputs": {
            k: {"path": str(run_dir / "data" / f"{k}.jsonl"), "rows": len(v)}
            for k, v in rows.items()
        },
    }
    (run_dir / "conversion.json").write_text(json.dumps(conversion, indent=2) + "\n")
    if args.convert_only:
        print(f"wrote {run_dir / 'conversion.json'} (convert only)")
        return 0

    training = run_training(args, run_dir, rows)
    manifest = {
        "run_id": args.run_id,
        "created": datetime.now(UTC).isoformat(),
        "base_model": {
            "id": args.base_model,
            "license": training.pop("base_model_license"),
            "revision": training.pop("base_model_revision"),
            "local_dir": training.pop("base_model_dir"),
        },
        "training_set": {
            "path": str(train_set),
            "sha256": {k: sha256_file(v) for k, v in inputs.items()},
            "converted_sha256": {
                k: sha256_file(run_dir / "data" / f"{k}.jsonl") for k in ("train", "valid")
            },
        },
        "hyperparameters": {
            "prefix_mode": args.prefix_mode,
            "iters": args.iters,
            "batch_size": 1,
            "learning_rate": args.learning_rate,
            "optimizer": "adam",
            "num_layers": args.num_layers,
            "rank": args.rank,
            "lora_scale": args.lora_scale,
            "dropout": 0.0,
            "max_seq_tokens_after_prefix": args.max_seq_tokens,
            "grad_checkpoint": True,
            "mask_prompt": True,
            "seed": args.seed,
        },
        "git": git_state(),
        "conversion": "conversion.json",
        "training": training,
        "wall_seconds": round(time.perf_counter() - started, 1),
        "max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {run_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
