"""WP7: export one v2 world (`01-spec/world-generator-v2-2026-09-22.md`
section 1 "Output per world", section 6 "Splits and contamination", and
section 9's WP7 row).

Reads what the runner's stages left under a world directory -- WP4's
`writer/`, and WP5's `completeness/` and WP6's `noise/` when they ran -- and
writes at the world root:

| File | What |
|---|---|
| `files/...` | the final documents, and the CRM account records |
| `sources.jsonl` | one row per extractable source: the prepared text every label indexes (what `portolan.bench.world_v2` reads) |
| `labels.jsonl` | every label that ships: `scripted_code` (CRM fields), `scripted_writer` and `silver_completeness`, with claims and noise |
| `decoys.jsonl` | the verified decoy spans |
| `qa_report.json` | WP5's report, copied unchanged |
| `training_records.jsonl` | `TrainingRecord`s for the builder, train and validation worlds only |
| `manifest.json` | world id, split, pack, seed, counts, the caps applied, every exclusion with its reason, and the sha256 of every file above |

**Composition.** A document ships WP5's final version (the rewrite when WP5
regenerated it) with WP5's labels, or WP4's version and labels when WP5 did
not run. WP6's noised version replaces it only when WP6 noised that same
text (`NoisedDoc.noised_from_sha256`, or for WP6 output made from WP4's
documents the same unregenerated version) and every shipped label and decoy
comes out of WP6 again (joined by `noise.carrier_slot_id`, which gives WP5's
silver labels a slot) or is logged as dropped by a noise operation;
otherwise the document ships un-noised and the manifest says why.
Near-duplicates exist only in WP6's output and ship with WP6's labels.

**Per-type cap** (WP1 build note): no fact type above 12% of the exported
labels of a train or validation world. Whole documents are dropped, never
single labels (a dropped label would leave its fact in the text as a taught
"not a fact"), in a fixed seeded order, each drop lowering the labels over
the cap summed over types; a document holding a planted, verified fact of
an arc's event is never dropped. Test worlds are not capped (their
documents are the bench's gold), nor is an export with fewer than
`CAP_MIN_LABELS` labels.

**Training records.** One `TrainingRecord` per envelope that
`portolan.pipeline.build_envelopes` makes of each shipped source, built by
WP5's own `source_inputs` so the user message is the one the teacher got.
Each fact carries the label's exact span as its quote and the teacher's
claim and field values; a record is only as complete as WP5 says
(`label_audit` counts the teacher facts that ended unlabeled). The summary,
sentiment and confidence come from the teacher's own answer when WP5 kept
it; when it did not, they stay empty and `portolan.training.sft_format`
refuses the record (`LegacyRecordError`) rather than fill placeholders.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from corpus.generator.v2 import completeness as wp5
from corpus.generator.v2.models import Split, World
from corpus.generator.v2.noise import NoisedDoc, carrier_slot_id, prepared_sha256
from corpus.generator.v2.planner import (
    CRM_ACCOUNTS_PATH,
    DocPlan,
    DocRole,
    WorldDocPlan,
    crm_account_source_id,
    crm_prepared_prefix,
    render_crm_row_text,
)
from corpus.generator.v2.renderers import materialize_crm_accounts
from corpus.generator.v2.writer import WrittenDoc
from portolan.extract.prompt import build_system_prompt, system_prompt_sha256
from portolan.extract.schema import QUOTE_REQUIRED_TYPES
from portolan.ingest.locators import parse_locator
from portolan.training.pipeline import assert_not_holdout
from portolan.training.records import LabelAudit, TeacherAnswer, TrainingFact, TrainingRecord
from portolan.training.sft_format import LegacyRecordError, production_sft_messages_for_record

__all__ = [
    "EXPORT_VERSION",
    "TRAINING_CORPUS",
    "TYPE_CAP_SHARE",
    "CapReport",
    "ComposedDoc",
    "StageOutputs",
    "apply_type_cap",
    "compose_documents",
    "export_world",
    "load_stage_outputs",
    "protected_documents",
    "training_records_for",
]

EXPORT_VERSION = "wp7-2026-09-23.1"
TRAINING_CORPUS = "synthetic_v2"
RECORD_MODE = "v2_world"
TYPE_CAP_SHARE = 0.12  # WP1 build note: no fact type above 12% of a world's exported facts
# Below this many labels one document is a large part of a type's 12%, and
# whole-document drops can empty the world (dropping everything meets any
# cap); a whole world has about 400.
CAP_MIN_LABELS = 100
CAP_SEED = "wp7-type-cap"
TRAINABLE_SPLITS = frozenset({Split.TRAIN, Split.VALIDATION})
TEACHER_CLAIM_SOURCES = frozenset({"teacher", "teacher_restatement"})
MAX_QUOTE_WORDS = 40  # fact rule 5, enforced by `portolan.extract.schema.Fact`
LICENSE = (
    "synthetic: written by open-weight models in the Portolan v2 world generator; "
    "every person, company and product is fictional"
)
_QUOTE_REQUIRED = frozenset(t.value for t in QUOTE_REQUIRED_TYPES)
# Keys `labels.jsonl` needs on every row (`portolan.bench.world_v2.WorldLabel`).
_LABEL_KEYS = ("fact_id", "fact_type", "doc_id", "char_start", "char_end", "text")


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------


@dataclass
class StageOutputs:
    """What the upstream stages left for one world. `states` and `noised`
    are `None` when that stage did not run (no directory)."""

    written: dict[str, WrittenDoc]
    states: dict[str, wp5.DocCompleteness] | None = None
    noised: dict[str, NoisedDoc] | None = None
    noise_manifest: dict[str, Any] = field(default_factory=dict)
    qa_report_path: Path | None = None
    versions: dict[str, Any] = field(default_factory=dict)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def load_stage_outputs(
    writer_dir: Path, completeness_dir: Path | None = None, noise_dir: Path | None = None
) -> StageOutputs:
    """WP4's written documents (only those its `run_report.json` lists as
    written, when it has one: a checkpoint left by older code is not
    current), WP5's per-document states and WP6's noised documents."""
    report = _read_json(writer_dir / "run_report.json")
    current = {d["plan_id"] for d in report.get("per_document", []) if d.get("status") == "written"}
    written: dict[str, WrittenDoc] = {}
    for path in sorted((writer_dir / "docs").glob("*.json")):
        doc = WrittenDoc.model_validate_json(path.read_text(encoding="utf-8"))
        if doc.status == "written" and (not report or doc.plan_id in current):
            written[doc.plan_id] = doc
    out = StageOutputs(written=written)
    out.versions["writer"] = {
        "prompt_version": report.get("prompt_version"),
        "checks_version": report.get("checks_version"),
    }
    if completeness_dir is not None and (completeness_dir / "docs").is_dir():
        out.states = {}
        for path in sorted((completeness_dir / "docs").glob("*.json")):
            state = wp5.DocCompleteness.model_validate_json(path.read_text(encoding="utf-8"))
            out.states[state.plan_id] = state
        qa = completeness_dir / "qa_report.json"
        out.qa_report_path = qa if qa.exists() else None
        out.versions["completeness"] = _read_json(qa).get("pass_version")
    if noise_dir is not None and (noise_dir / "docs").is_dir():
        out.noised = {}
        for path in sorted((noise_dir / "docs").glob("*.json")):
            doc = NoisedDoc.model_validate_json(path.read_text(encoding="utf-8"))
            out.noised[doc.plan_id] = doc
        out.noise_manifest = _read_json(noise_dir / "noise_manifest.json")
        out.versions["noise"] = out.noise_manifest.get("noise_version")
    return out


# --------------------------------------------------------------------------
# Composition
# --------------------------------------------------------------------------


@dataclass
class ComposedDoc:
    """One document as it ships: its final text and files, and every label
    and decoy as a JSON-ready dict indexing that text."""

    plan: DocPlan
    doc: WrittenDoc
    labels: list[dict[str, Any]]
    decoys: list[dict[str, Any]]
    noise: list[dict[str, Any]] = field(default_factory=list)
    filed_account_id: str | None = None
    state: wp5.DocCompleteness | None = None
    # WP4's version, and the version WP5's teachers read (its rewrite, if any).
    original: WrittenDoc | None = None
    teacher_doc: WrittenDoc | None = None
    noise_applied: bool = False
    note: str | None = None

    @property
    def plan_id(self) -> str:
        return self.plan.id

    def type_counts(self) -> Counter:
        return Counter(str(x["fact_type"]) for x in self.labels)


def _dump(model: Any) -> dict[str, Any]:
    return model.model_dump(mode="json")


# What a noise operation changes on a label or decoy (`noise._new_fields`,
# `noise._finish`); every other field of a shipped row stays as WP5 wrote it.
_NOISE_FIELDS = (
    "doc_id",
    "char_start",
    "char_end",
    "text",
    "prepared_locator",
    "locator",
    "span_words",
    "noise",
    "asr_in_span",
    "remapped",
    "copy_of",
)


def _noise_join(
    labels: list[dict[str, Any]], decoys: list[dict[str, Any]], noised: NoisedDoc
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]] | str:
    """The shipped labels and decoys moved onto WP6's text, or why they
    cannot be: each must come out of WP6 under its carrier slot
    (`noise.carrier_slot_id`; copies too), or be logged as a span a noise
    operation dropped."""
    dropped = {(x.doc_id, x.char_start, x.char_end, x.text) for x in noised.label_drops}

    def join(rows: list[dict[str, Any]], out_rows: list[Any], what: str):
        by_slot: dict[str, list[dict[str, Any]]] = {}
        for x in out_rows:
            by_slot.setdefault(x.slot_id, []).append(_dump(x))
        joined: list[dict[str, Any]] = []
        for row in rows:
            slot = carrier_slot_id(row)
            hits = by_slot.pop(slot, None)
            if hits is None:
                key = (row["doc_id"], row["char_start"], row["char_end"], row["text"])
                if key in dropped:
                    continue
                return f"{what} {slot} is neither in WP6's output nor in its drop log"
            joined.extend({**row, **{k: hit[k] for k in _NOISE_FIELDS if k in hit}} for hit in hits)
        if by_slot:
            return f"WP6 has {what}s this document does not ship: {sorted(by_slot)}"
        return joined

    new_labels = join(labels, noised.labels, "label")
    if isinstance(new_labels, str):
        return new_labels
    new_decoys = join(decoys, noised.decoy_spans, "decoy")
    if isinstance(new_decoys, str):
        return new_decoys
    return new_labels, new_decoys


def _near_duplicate_labels(nd: NoisedDoc, world: World, manifest: dict[str, Any]) -> list[dict]:
    """WP6's labels on a near-duplicate, with the claim the bench scores
    against: the world fact's, or WP6's changed claim for the new fact."""
    claims = {f.id: f.claim for f in world.facts}
    claims.update(
        {f["id"]: f["claim"] for f in manifest.get("new_facts", []) if "id" in f and "claim" in f}
    )
    return [
        {**_dump(x), "claim": claims.get(x.fact_id, ""), "claim_source": "scripted"}
        for x in nd.labels
    ]


def compose_documents(
    world: World, world_plan: WorldDocPlan, stage: StageOutputs
) -> tuple[list[ComposedDoc], dict[str, str]]:
    """Every document that ships, and `plan_id -> reason` for each one that
    does not."""
    plans = world_plan.doc_by_id()
    claims = {f.id: f.claim for f in world.facts}
    noised = stage.noised or {}
    near = {
        pid: nd
        for pid, nd in noised.items()
        if pid not in stage.written
        and pid in plans
        and plans[pid].role == DocRole.NEAR_DUPLICATE
        and nd.status == "written"
    }
    docs: list[ComposedDoc] = []
    excluded: dict[str, str] = {}
    for pid in sorted(set(stage.written) | set(near)):
        plan = plans.get(pid)
        if plan is None:
            excluded[pid] = "not_in_world_plan"
            continue
        if pid in near:
            nd = near[pid]
            docs.append(
                ComposedDoc(
                    plan=plan,
                    doc=nd,
                    labels=_near_duplicate_labels(nd, world, stage.noise_manifest),
                    decoys=[_dump(x) for x in nd.decoy_spans],
                    noise=[_dump(op) for op in nd.noise],
                    filed_account_id=nd.filed_account_id or plan.account_id,
                    noise_applied=True,
                )
            )
            continue
        wp4_doc = stage.written[pid]
        state: wp5.DocCompleteness | None = None
        if stage.states is not None:
            state = stage.states.get(pid)
            if state is None:
                excluded[pid] = "completeness_missing"
                continue
            if state.status not in wp5.SHIPPED_STATUSES:
                excluded[pid] = f"completeness_{state.status}"
                continue
            base = state.final_doc(wp4_doc)
            labels = [_dump(x) for x in [*state.scripted_labels, *state.silver_labels]]
        else:
            base = wp4_doc
            labels = [
                {**_dump(x), "claim": claims.get(x.fact_id, ""), "claim_source": "scripted"}
                for x in wp4_doc.labels
            ]
        decoys = [_dump(x) for x in base.decoy_spans]
        composed = ComposedDoc(
            plan=plan,
            doc=base,
            labels=labels,
            decoys=decoys,
            filed_account_id=plan.account_id,
            state=state,
            original=wp4_doc,
            teacher_doc=base if state is not None else None,
        )
        nd = noised.get(pid)
        if nd is not None and nd.status == "written":
            if nd.noised_from_sha256 is not None:
                same_text = nd.noised_from_sha256 == prepared_sha256(base)
            else:  # noised from WP4's output, before WP5 ran
                regenerated = state is not None and state.regenerated_doc is not None
                same_text = not regenerated and nd.input_fingerprint == wp4_doc.input_fingerprint
            if not same_text:
                composed.note = "noise_not_applied: WP6 noised another version of this document"
            else:
                joined = _noise_join(labels, decoys, nd)
                if isinstance(joined, str):
                    composed.note = f"noise_not_applied: {joined}"
                else:
                    composed.doc = nd
                    composed.labels, composed.decoys = joined
                    composed.noise = [_dump(op) for op in nd.noise]
                    composed.filed_account_id = nd.filed_account_id or plan.account_id
                    composed.noise_applied = True
        docs.append(composed)
    return docs, excluded


# --------------------------------------------------------------------------
# Per-type cap
# --------------------------------------------------------------------------


def protected_documents(docs: list[ComposedDoc], world: World) -> set[str]:
    """Documents holding a planted, verified fact of an arc's event (WP4's
    verified label, or WP5's recovery of a planted fact WP4 left unmarked)."""
    arc_events = {e for arc in world.arcs for e in arc.event_ids}
    arc_facts = {f.id for f in world.facts if f.event_id in arc_events}
    out: set[str] = set()
    for d in docs:
        for x in d.labels:
            planted = x.get("label_source") == "scripted_writer" or x.get("recovers_fact_id")
            if planted and x["fact_id"] in arc_facts:
                out.add(d.plan_id)
                break
    return out


@dataclass
class CapReport:
    share: float
    applied: bool
    reason: str | None = None
    labels_before: dict[str, int] = field(default_factory=dict)
    labels_after: dict[str, int] = field(default_factory=dict)
    documents_dropped: list[str] = field(default_factory=list)
    protected_documents: int = 0
    unmet_types: list[str] = field(default_factory=list)

    def as_json(self) -> dict[str, Any]:
        def top(counts: dict[str, int]) -> float:
            total = sum(counts.values())
            return round(max(counts.values()) / total, 4) if total else 0.0

        return {
            "share": self.share,
            "applied": self.applied,
            "reason": self.reason,
            "labels_before": dict(sorted(self.labels_before.items())),
            "labels_after": dict(sorted(self.labels_after.items())),
            "max_share_before": top(self.labels_before),
            "max_share_after": top(self.labels_after),
            "labels_dropped": sum(self.labels_before.values()) - sum(self.labels_after.values()),
            "documents_dropped": self.documents_dropped,
            "protected_documents": self.protected_documents,
            "met": not self.unmet_types,
            "unmet_types": self.unmet_types,
        }


_EPSILON = 1e-9


def _excess(counts: Counter, share: float) -> float:
    """Labels above the cap, summed over fact types."""
    total = sum(counts.values())
    return sum(max(0.0, n - share * total) for n in counts.values())


def apply_type_cap(
    docs: list[ComposedDoc], *, share: float, seed_key: str, protected: set[str]
) -> tuple[list[ComposedDoc], CapReport]:
    """Drop whole documents while some fact type is above `share` of the
    labels. Each drop is the first document, in a fixed order seeded by
    `seed_key`, that is not protected and whose removal lowers the labels
    over the cap summed over every type, so no drop makes the mix worse
    (dropping one type's documents shrinks the total and can push another
    type over). When no document lowers it, the cap is reported unmet."""
    counts: Counter = Counter()
    for d in docs:
        counts.update(d.type_counts())
    report = CapReport(
        share=share,
        applied=True,
        labels_before=dict(counts),
        protected_documents=sum(1 for d in docs if d.plan_id in protected),
    )
    order = sorted(docs, key=lambda d: hashlib.sha256(f"{seed_key}:{d.plan_id}".encode()).digest())
    kept = {d.plan_id: d for d in docs}
    while (current := _excess(counts, share)) > _EPSILON:
        victim = next(
            (
                d
                for d in order
                if d.plan_id in kept
                and d.plan_id not in protected
                and d.labels
                and _excess(counts - d.type_counts(), share) < current - _EPSILON
            ),
            None,
        )
        if victim is None:
            break
        del kept[victim.plan_id]
        counts = counts - victim.type_counts()
        report.documents_dropped.append(victim.plan_id)
    report.labels_after = dict(counts)
    total = sum(counts.values())
    report.unmet_types = sorted(t for t, n in counts.items() if total and n > share * total)
    return [d for d in docs if d.plan_id in kept], report


# --------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------


def _locators(text: str) -> list[dict[str, Any]]:
    """`[{locator, start, end}]`: each `prepare_text` locator line up to the
    next one."""
    marks: list[tuple[str, int]] = []
    pos = 0
    for line in text.splitlines(keepends=True):
        loc = parse_locator(line)
        if loc is not None:
            marks.append((loc, pos))
        pos += len(line)
    return [
        {
            "locator": loc,
            "start": start,
            "end": marks[i + 1][1] if i + 1 < len(marks) else len(text),
        }
        for i, (loc, start) in enumerate(marks)
    ]


def _op_types(noise: list[dict[str, Any]]) -> list[str]:
    return list(dict.fromkeys(str(op.get("type")) for op in noise if op.get("type")))


# --------------------------------------------------------------------------
# Training records
# --------------------------------------------------------------------------


@dataclass
class RecordContext:
    """Per-world values every record of the world shares."""

    world: World
    split: Split
    company_context: str | None
    prompt_variant: str
    created: str


def _stamp_context(
    state: wp5.DocCompleteness, doc: WrittenDoc, world: World, variant: str
) -> tuple[bool, str | None]:
    """Whether the teacher saw today's system prompt for this document
    (WP5's `teacher_key` over it matches), and with which company context:
    the world's vendor note, or none."""
    for context in (wp5.world_company_context(world), None):
        probe = SimpleNamespace(
            teacher_model=wp5.TEACHER_MODEL,
            system_prompt=build_system_prompt(company_context=context, variant=variant),
        )
        if state.teacher_key and wp5.teacher_key(doc, probe) == state.teacher_key:  # type: ignore[arg-type]
            return True, context
    return False, None


def _known_unlabeled(state: wp5.DocCompleteness, texts: dict[str, str]) -> list[wp5.TeacherFact]:
    """Teacher facts that ship unlabeled, counted exactly as WP5's
    `known_unlabeled` counts them: the labeling teacher's unlabelable
    candidates no audit label covers, and every audit fact left unlabeled."""
    if not state.attempts:
        return []
    audit_spans = [
        wp5.ScriptedSpan("fact", x.fact_id, x.doc_id, x.char_start, x.char_end, True)
        for x in (state.audit.labels if state.audit else [])
    ]
    out = [
        f
        for f in state.attempts[-1].open_facts()
        if f.unlabelable and not (f.has_span and wp5._best_overlap(f, audit_spans, texts))
    ]
    if state.audit is not None:
        out += [
            f
            for f in state.audit.facts
            if f.grounded and f.triage in ("candidate", "planted") and not f.labeled
        ]
    return out


def _teacher_facts(state: wp5.DocCompleteness) -> list[wp5.TeacherFact]:
    """Every grounded teacher fact on the shipped version that is not a
    decoy hit, both teachers."""
    facts = list(state.attempts[-1].facts) if state.attempts else []
    facts += state.audit.facts if state.audit else []
    return [
        f for f in facts if f.grounded and f.triage != "decoy_hit" and f.audit_status != "decoy_hit"
    ]


def _in_envelope(
    f: wp5.TeacherFact, index: int, envelopes_now: int, envelopes_then: dict[str, int]
) -> bool:
    """A teacher fact belongs to envelope `index` of its source when the
    source still splits the way the teacher read it; otherwise to every
    envelope (it cannot be placed, so it counts against all of them)."""
    if envelopes_then.get(f.doc_id) != envelopes_now:
        return True
    return f.envelope == index


def _fact_for(label: dict[str, Any], quote_span: tuple[int, int], locator: str) -> TrainingFact:
    fields = label.get("teacher_fields") or {}
    return TrainingFact(
        fact_type=str(label["fact_type"]),
        claim=str(label.get("claim") or ""),
        quote=label["text"],
        quote_span=quote_span,
        locator=locator,
        entities=list(fields.get("entities") or []),
        sentiment=fields.get("sentiment"),
        confidence=fields.get("confidence"),
        severity=fields.get("severity"),
        speaker=fields.get("speaker"),
        valid_from=fields.get("valid_from"),
        valid_to=fields.get("valid_to"),
        label_source=label.get("label_source"),
        claim_source=label.get("claim_source"),
    )


def _claim_teacher(label: dict[str, Any]) -> str | None:
    if label.get("claim_source") not in TEACHER_CLAIM_SOURCES:
        return None
    return label.get("claim_teacher_model") or label.get("teacher_model")


@dataclass
class RecordOutcome:
    records: list[TrainingRecord] = field(default_factory=list)
    excluded: Counter = field(default_factory=Counter)
    not_sft_ready: Counter = field(default_factory=Counter)
    notes: list[str] = field(default_factory=list)


def training_records_for(composed: ComposedDoc, rctx: RecordContext) -> RecordOutcome:
    """The document's training records, one per envelope of each source.
    A record that cannot be expressed as a production answer (a label cut
    by a segment boundary, a locator or quote not in its text, a quote over
    the forty-word rule) is left out whole, never trimmed."""
    out = RecordOutcome()
    state = composed.state
    if state is None:
        out.excluded["no_completeness_pass"] += 1
        return out
    world = rctx.world
    try:
        sources = wp5.source_inputs(composed.doc, composed.plan)
    except wp5.PreparedTextDrift as exc:
        out.excluded["prepared_text_drift"] += 1
        out.notes.append(f"{composed.plan_id}: {exc}")
        return out
    teacher_saw = composed.teacher_doc or composed.doc
    stamped, context = _stamp_context(
        state, composed.original or teacher_saw, world, rctx.prompt_variant
    )
    texts_then = {p.source_id: p.text for p in teacher_saw.prepared_sources}
    unlabeled = _known_unlabeled(state, texts_then)
    grounded = _teacher_facts(state)
    calls = [c for c in (state.attempts[-1].calls if state.attempts else [])]
    calls = [c for c in calls if c.role == "completeness"]
    answers = {(c.source_id, c.envelope): c.answer for c in calls}
    envelopes_then: dict[str, int] = {}
    for c in calls:
        envelopes_then[c.source_id] = max(envelopes_then.get(c.source_id, 0), c.envelope)
    writer = composed.doc.writer_model
    for source in sources:
        labels = [x for x in composed.labels if x["doc_id"] == source.source_id]
        envelopes = wp5._source_envelopes(source)
        n = len(envelopes)
        for index, envelope, offset in envelopes:
            record_id = f"{world.id}:{source.source_id}" + (f"#seg{index}of{n}" if n > 1 else "")
            if offset < 0:
                out.excluded["segment_not_located"] += 1
                continue
            end = offset + len(envelope.text)
            inside = [x for x in labels if offset <= x["char_start"] and x["char_end"] <= end]
            cut = [
                x
                for x in labels
                if x not in inside and x["char_start"] < end and x["char_end"] > offset
            ]
            if cut:
                out.excluded["label_cut_by_segment"] += 1
                continue
            facts: list[TrainingFact] = []
            reason = None
            for x in sorted(inside, key=lambda x: (x["char_start"], x["char_end"], x["fact_id"])):
                span = (x["char_start"] - offset, x["char_end"] - offset)
                locator = x.get("prepared_locator") or x.get("locator") or ""
                if envelope.text[span[0] : span[1]] != x["text"]:
                    reason = "quote_not_at_span"
                elif not locator or f"[{locator}]" not in envelope.text:
                    reason = "locator_not_in_text"
                elif (
                    str(x["fact_type"]) in _QUOTE_REQUIRED
                    and len(x["text"].split()) > MAX_QUOTE_WORDS
                ):
                    reason = "quote_over_40_words"
                if reason:
                    break
                facts.append(_fact_for(x, span, locator))
            if reason:
                out.excluded[reason] += 1
                continue
            in_env = [
                f
                for f in grounded
                if f.doc_id == source.source_id and _in_envelope(f, index, n, envelopes_then)
            ]
            open_env = [
                f
                for f in unlabeled
                if f.doc_id == source.source_id and _in_envelope(f, index, n, envelopes_then)
            ]
            answer = (
                answers.get((source.source_id, index))
                if envelopes_then.get(source.source_id) == n
                else None
            )
            teachers = {writer} if writer else set()
            teachers.update(t for t in (_claim_teacher(x) for x in inside) if t)
            if answer is not None:
                teachers.add(wp5.TEACHER_MODEL)
            record = TrainingRecord(
                record_id=record_id,
                corpus=TRAINING_CORPUS,
                source_id=f"{world.id}/{source.source_id}",
                kind=source.source_type.value,
                license=LICENSE,
                text=envelope.text,
                facts=facts,
                teachers=sorted(teachers),
                agreement=1.0,
                created=rctx.created,
                mode=RECORD_MODE,
                label_audit=LabelAudit(
                    status="incomplete" if open_env else "complete",
                    teacher_facts=len(in_env),
                    kept=max(0, len(in_env) - len(open_env)),
                    regated=0,
                    repaired=0,
                    withdrawn=0,
                    unresolved=len(open_env),
                    unlogged=0,
                    audited=rctx.created,
                ),
                prompt_variant=rctx.prompt_variant if stamped else None,
                system_prompt_sha256=system_prompt_sha256(
                    build_system_prompt(company_context=context, variant=rctx.prompt_variant)
                )
                if stamped
                else None,
                user_message_inputs=envelope.model_dump(mode="json", exclude={"text"}),
                teacher_answer=TeacherAnswer.model_validate(answer) if answer is not None else None,
                company_context=context if stamped else rctx.company_context,
                world_id=world.id,
                world_split=rctx.split.value,
            )
            try:
                production_sft_messages_for_record(record, company_context=record.company_context)
            except LegacyRecordError as exc:
                out.not_sft_ready[_legacy_reason(str(exc))] += 1
            except ValueError as exc:
                out.excluded["answer_invalid"] += 1
                out.notes.append(f"{record_id}: {exc}"[:300])
                continue
            out.records.append(record)
    return out


def _legacy_reason(message: str) -> str:
    if "prompt stamp" in message:
        return "no_prompt_stamp"
    if "teacher_answer" in message:
        return "no_teacher_answer"
    if "sentiment" in message:
        return "no_teacher_fact_values"
    return "legacy_record"


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------


def _jsonl(rows: list[dict[str, Any]] | list[str]) -> str:
    return "".join(
        (r if isinstance(r, str) else json.dumps(r, ensure_ascii=False)) + "\n" for r in rows
    )


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _crm_sources(world: World, world_plan: WorldDocPlan) -> tuple[list[dict], list[dict]]:
    """The code-rendered CRM account records as sources, and their
    field-derived labels (bench only: no teacher ever claims them)."""
    rows: list[dict] = []
    ids: set[str] = set()
    for record in world_plan.crm_accounts:
        source_id = crm_account_source_id(record.crm_id)
        text = crm_prepared_prefix(record.crm_id) + render_crm_row_text("account", record.row)
        ids.add(source_id)
        rows.append(
            {
                "source_id": source_id,
                "source_type": "crm",
                "kind": "crm_account",
                "date": world.window_end.isoformat(),
                "text": text,
                "plan_id": CRM_ACCOUNTS_PATH,
                "account_id": record.account_id,
                "true_account_id": record.account_id,
                "participants": [],
                "noise": [],
                "locators": _locators(text),
                "path": CRM_ACCOUNTS_PATH,
                "writer_model": None,
            }
        )
    labels = [_dump(x) for x in world_plan.labels if x.doc_id in ids]
    return rows, labels


def export_world(
    world: World,
    world_plan: WorldDocPlan,
    *,
    split: Split,
    pack: str,
    seed: int,
    world_dir: Path,
    writer_dir: Path | None = None,
    completeness_dir: Path | None = None,
    noise_dir: Path | None = None,
    held_out: bool = False,
    dry_run: bool = False,
    prompt_variant: str = wp5.PROMPT_VARIANT,
    cap_share: float = TYPE_CAP_SHARE,
    cap_min_labels: int = CAP_MIN_LABELS,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> dict[str, Any]:
    """Export one world into `world_dir` (the world root) and return its
    manifest. Stage directories default to `writer/`, `completeness/` and
    `noise/` under the world root; the last two are optional."""
    assert_not_holdout(world_dir)
    writer_dir = writer_dir or world_dir / "writer"
    completeness_dir = (
        completeness_dir if completeness_dir is not None else world_dir / "completeness"
    )
    noise_dir = noise_dir if noise_dir is not None else world_dir / "noise"
    for stage_dir in (writer_dir, completeness_dir, noise_dir):
        if stage_dir.resolve() == world_dir.resolve():
            raise ValueError(f"{stage_dir} is the export root; stages need their own directories")
    created = now().isoformat()
    stage = load_stage_outputs(writer_dir, completeness_dir, noise_dir)
    docs, excluded = compose_documents(world, world_plan, stage)

    trainable_split = split in TRAINABLE_SPLITS and not held_out
    not_trainable_reason = (
        "held-out vertical: test worlds only, never trained on"
        if held_out
        else "test world: the bench's gold, never trained on (spec section 6)"
    )
    protected = protected_documents(docs, world)
    counts: Counter = Counter()
    for d in docs:
        counts.update(d.type_counts())
    if trainable_split and sum(counts.values()) >= cap_min_labels:
        docs, cap = apply_type_cap(
            docs, share=cap_share, seed_key=f"{CAP_SEED}:{world.id}", protected=protected
        )
        for pid in cap.documents_dropped:
            excluded[pid] = "type_cap"
    else:
        cap = CapReport(
            share=cap_share,
            applied=False,
            reason=not_trainable_reason
            if not trainable_split
            else f"{sum(counts.values())} labels, fewer than {cap_min_labels}: too few for a "
            "per-type share to mean anything (whole-document drops could empty the world)",
            labels_before=dict(counts),
            labels_after=dict(counts),
            protected_documents=len(protected),
        )

    files_root = world_dir / "files"
    if files_root.exists():
        shutil.rmtree(files_root)
    written_files: list[Path] = []
    for d in docs:
        for rel, content in sorted(d.doc.files.items()):
            path = files_root / rel
            _write(path, content)
            written_files.append(path)
    written_files.append(materialize_crm_accounts(world_plan.crm_accounts, files_root))

    # sources, labels, decoys
    source_rows: list[dict[str, Any]] = []
    source_meta: list[dict[str, Any]] = []
    for d in docs:
        try:
            inputs = {s.source_id: s for s in wp5.source_inputs(d.doc, d.plan)}
        except wp5.PreparedTextDrift:
            inputs = {}
        for p in d.doc.prepared_sources:
            si = inputs.get(p.source_id)
            row_date: date = si.source_date if si else d.plan.date
            source_rows.append(
                {
                    "source_id": p.source_id,
                    "source_type": p.source_type,
                    "kind": d.plan.kind.value,
                    "date": row_date.isoformat(),
                    "text": p.text,
                    "plan_id": d.plan_id,
                    "account_id": d.filed_account_id,
                    "true_account_id": d.plan.account_id,
                    "participants": [x.name for x in si.participants] if si else [],
                    "noise": d.noise,
                    "locators": _locators(p.text),
                    "path": p.path,
                    "writer_model": d.doc.writer_model,
                    "role": d.plan.role.value,
                    "world_id": world.id,
                }
            )
            source_meta.append(
                {
                    "id": p.source_id,
                    "plan_id": d.plan_id,
                    "kind": d.plan.kind.value,
                    "date": row_date.isoformat(),
                    "path": f"files/{p.path}",
                }
            )
    crm_rows, crm_labels = _crm_sources(world, world_plan)
    source_rows.extend(crm_rows)
    source_meta.extend(
        {
            "id": r["source_id"],
            "plan_id": r["plan_id"],
            "kind": r["kind"],
            "date": r["date"],
            "path": f"files/{r['path']}",
        }
        for r in crm_rows
    )
    label_rows: list[dict[str, Any]] = []
    for d in docs:
        ops = _op_types(d.noise)
        for x in d.labels:
            # `noise`: operations that touched the span (WP6); `doc_noise`: the document's.
            label_rows.append(
                {**x, "world_id": world.id, "noise": list(x.get("noise") or []), "doc_noise": ops}
            )
    label_rows.extend({**x, "world_id": world.id, "noise": [], "doc_noise": []} for x in crm_labels)
    decoy_rows = [{**x, "world_id": world.id} for d in docs for x in d.decoys]
    texts = {r["source_id"]: r["text"] for r in source_rows}
    for row in [*label_rows, *decoy_rows]:
        text = texts.get(row["doc_id"])
        if text is None or text[row["char_start"] : row["char_end"]] != row["text"]:
            raise AssertionError(
                f"export bug: span of {row.get('fact_id') or row.get('decoy_id')} in {row['doc_id']} does not resolve"
            )
    for key in _LABEL_KEYS:
        assert all(key in row for row in label_rows), key

    _write(world_dir / "sources.jsonl", _jsonl(source_rows))
    _write(world_dir / "labels.jsonl", _jsonl(label_rows))
    _write(world_dir / "decoys.jsonl", _jsonl(decoy_rows))
    qa_gates: dict[str, Any] = {}
    qa_out = world_dir / "qa_report.json"
    if stage.qa_report_path is not None:
        shutil.copyfile(stage.qa_report_path, qa_out)
        qa = _read_json(qa_out)
        qa_gates = {
            str(row.get("gate")): row.get("result")
            for row in qa.get("gates") or []
            if isinstance(row, dict)
        }
    elif qa_out.exists():
        qa_out.unlink()
    written_files += [
        world_dir / "sources.jsonl",
        world_dir / "labels.jsonl",
        world_dir / "decoys.jsonl",
    ]
    if qa_out.exists():
        written_files.append(qa_out)

    # training records
    training: dict[str, Any] = {"written": False}
    records_path = world_dir / "training_records.jsonl"
    context = wp5.world_company_context(world)
    if not trainable_split:
        training["reason"] = not_trainable_reason
        if records_path.exists():
            records_path.unlink()
    else:
        rctx = RecordContext(
            world=world,
            split=split,
            company_context=context,
            prompt_variant=prompt_variant,
            created=created,
        )
        records: list[TrainingRecord] = []
        excluded_records: Counter = Counter()
        not_ready: Counter = Counter()
        notes: list[str] = []
        docs_without: Counter = Counter()
        for d in docs:
            if d.state is None:
                docs_without[
                    "near_duplicate_no_teacher_pass"
                    if d.plan.role == DocRole.NEAR_DUPLICATE
                    else "no_completeness_pass"
                ] += 1
                continue
            outcome = training_records_for(d, rctx)
            records.extend(outcome.records)
            excluded_records.update(outcome.excluded)
            not_ready.update(outcome.not_sft_ready)
            notes.extend(outcome.notes)
        _write(records_path, _jsonl([r.model_dump_json() for r in records]))
        written_files.append(records_path)
        claims_scripted = sum(
            1 for r in records if any(f.claim_source not in TEACHER_CLAIM_SOURCES for f in r.facts)
        )
        incomplete = sum(1 for r in records if r.label_audit and r.label_audit.status != "complete")
        training = {
            "written": True,
            "path": records_path.name,
            "records": len(records),
            "positives": sum(1 for r in records if r.facts),
            "negatives": sum(1 for r in records if not r.facts),
            "facts": sum(len(r.facts) for r in records),
            "facts_by_type": dict(
                sorted(Counter(f.fact_type for r in records for f in r.facts).items())
            ),
            "documents_without_records": dict(docs_without),
            "records_excluded_by_reason": dict(sorted(excluded_records.items())),
            "records_with_a_claim_not_from_a_teacher": claims_scripted,
            "records_with_known_unlabeled_facts": incomplete,
            "sft_ready": len(records) - sum(not_ready.values()),
            "not_sft_ready_by_reason": dict(sorted(not_ready.items())),
            "passing_provenance_and_completeness": sum(
                1
                for r in records
                if all(f.claim_source in TEACHER_CLAIM_SOURCES for f in r.facts)
                and r.label_audit is not None
                and r.label_audit.status == "complete"
            ),
            "prompt_variant": prompt_variant,
            "company_context_sha256": hashlib.sha256(context.encode("utf-8")).hexdigest(),
            "summary_and_sentiment_policy": (
                "from the teacher's own answer; when WP5 did not keep it the fields stay empty "
                "and sft_format raises LegacyRecordError (no placeholders)"
            ),
            "notes": notes[:50],
        }

    ready_reasons: list[str] = []
    if dry_run:
        ready_reasons.append("dry run: fixture writer text")
    if not trainable_split:
        ready_reasons.append(training["reason"])
    if stage.states is None:
        ready_reasons.append(
            "completeness (WP5) did not run: labels are unchecked for completeness and claims are scripted"
        )
    if stage.noised is None:
        ready_reasons.append("noise (WP6) did not run")
    if training.get("written") and not training.get("sft_ready"):
        ready_reasons.append(
            "no record carries the teacher's full answer (sft_format refuses them all)"
        )

    labels_by_type = Counter(str(x["fact_type"]) for x in label_rows)
    kind_of = {r["source_id"]: r["kind"] for r in source_rows}
    manifest = {
        "export_version": EXPORT_VERSION,
        "created": created,
        "world_id": world.id,
        "split": split.value,
        "pack": pack,
        "vertical": world.vertical,
        "seed": seed,
        "held_out": held_out,
        "dry_run": dry_run,
        "vendor": world.vendor.name,
        "training_ready": not ready_reasons,
        "training_ready_reasons": ready_reasons,
        "upstream": {
            "writer": {"documents_written": len(stage.written), **stage.versions.get("writer", {})},
            "completeness": {
                "ran": stage.states is not None,
                "pass_version": stage.versions.get("completeness"),
                "documents": len(stage.states or {}),
            },
            "noise": {
                "ran": stage.noised is not None,
                "noise_version": stage.versions.get("noise"),
                "documents": len(stage.noised or {}),
            },
        },
        "documents": {
            "exported": len(docs),
            "by_kind": dict(sorted(Counter(d.plan.kind.value for d in docs).items())),
            "by_role": dict(sorted(Counter(d.plan.role.value for d in docs).items())),
            "noised": sum(1 for d in docs if d.noise_applied),
            "noise_not_applied": {d.plan_id: d.note for d in docs if d.note},
            "excluded_by_reason": dict(sorted(Counter(excluded.values()).items())),
            "excluded": dict(sorted(excluded.items())),
        },
        "counts": {
            "sources": len(source_rows),
            "sources_by_kind": dict(sorted(Counter(r["kind"] for r in source_rows).items())),
            "labels": len(label_rows),
            "labels_by_fact_type": dict(sorted(labels_by_type.items())),
            "labels_by_label_source": dict(
                sorted(Counter(str(x.get("label_source")) for x in label_rows).items())
            ),
            "labels_by_kind": dict(
                sorted(Counter(kind_of[x["doc_id"]] for x in label_rows).items())
            ),
            "decoys": len(decoy_rows),
            "crm_rows": len(crm_rows),
            "crm_links": "WP6 link noise is recorded in noise/noise_manifest.json, not re-rendered into files/crm/accounts.csv",
        },
        "caps": {"per_type": cap.as_json()},
        "qa_gates": qa_gates,
        "training": training,
        "sources": source_meta,
        "files": {str(p.relative_to(world_dir)): _sha_file(p) for p in sorted(set(written_files))},
    }
    _write(world_dir / "manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest
