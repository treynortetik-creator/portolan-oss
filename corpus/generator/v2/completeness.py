"""WP5: the completeness pass (`01-spec/world-generator-v2-2026-09-22.md`
sections 3.3 to 3.5, and section 9's WP5 row).

For every document WP4 wrote, one blind extraction by the labeling teacher
(`moonshotai/kimi-k2`, production prompt `v6-type-precision`). The teacher
gets exactly what production would send (the `prepare_text` output, chunked
by `build_envelopes`) and never sees a scripted fact, a claim or a marker.
Each teacher fact is gated the way the silver pipeline gates it (the quote
gate, the numeral-and-name rule with content overlap off, exact-span
resolution; the claim is read with WP4 check 6's normalizations), then
matched to the document's scripted spans by code, edge punctuation trimmed.
A fact dropped only for its claim or its exact span is still triaged: the
text says something there, so as a candidate it is unlabelable and blocks.

- **confirmed**: it overlaps a scripted fact's label span (IoU >= 0.5, or
  one span inside the other); the type agreement is recorded and the
  scripted fact counts toward the teacher's recall `r`.
- **decoy hit**: it shares any characters with a decoy, verified or not (a
  decoy WP4 dropped for a wrong speaker or an unclosed marker is still decoy
  wording in the text). The fact never becomes a label and is logged as a
  teacher decoy error (`qa_report.json` `teacher_decoy_errors`).
- **restatement**: same type as a scripted fact, another span, claim
  similarity >= 0.8 against that fact's claim or the teacher's own claim
  on it; kept as a second silver locator.
- **candidate**: overlaps nothing, a possible unlabeled positive. Code
  checks it against the world state (`contradictions`) and notes any
  unknown entity or unscripted number.

Per-document rule (`decide`; section 3.4 as changed by the 2026-09-22 cost
decision): a candidate consistent with the world becomes a silver label where
it stands. Only a candidate that contradicts the world, names an invented
entity or price, or states a later date as past sends the document back once
through WP4's writer, with those sentences as "do not state"; if it survives
the rewrite, the document is dropped.

**Provenance.** Label text is always an exact span of the writer's
document; a silver label's claim is the teacher's, and so is a scripted
label's when the teacher confirmed it (`CompletedLabel.claim_source`; the
scripted claim stays as metadata). Code writes neither. Each label also keeps the
claim teacher's own fact values (`teacher_fields`) and each call the rest
of the teacher's answer (`TeacherCall.answer`), which WP7's training
records need. The audit teacher (another family) labels only where the
labeling teacher found an overlapping span (spec Build notes, WP5).

With `CompletenessContext.label_judge` set (`--label-judge MODEL`), an
open-weight judge (`portolan.training.label_judge`) then reads every shipped
document's silver labels: drop and retype are applied by code, claims are
never edited, spend goes under `label_judge`.

`run_completeness` drives a world: per-document checkpoints (resume makes
no call for a finished document), a hard dollar cap checked before every
call on the same `synthetic_v2` spend line WP4 uses, the audit phase, and
`qa_report.json` with the section 3.5 gates and the residual estimate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from corpus.generator.v2.models import Person, PersonSide, World
from corpus.generator.v2.planner import (
    DocKind,
    DocPlan,
    DocRole,
    LabelRecord,
    PlannedDecoy,
    WorldDocPlan,
)
from corpus.generator.v2.verify import (
    _DAY_MONTH_RE,
    _MONTH_DAY_RE,
    _SLASH_DATE_RE,
    FORBIDDEN_PATTERN_FAILURE,
    WriterLabel,
    coverage_extras,
    forbidden_pattern_hits,
    rule_claim,
)
from corpus.generator.v2.writer import (
    ALLOWED_WRITER_MODELS,
    CORPUS_KEY,
    RunBudget,
    WriterBudgetExceeded,
    WrittenDoc,
)
from portolan.bench.extraction import _claim_similarity
from portolan.extract.backends import Backend
from portolan.extract.numerals import _document_numbers
from portolan.extract.prompt import build_system_prompt, build_user_message
from portolan.extract.quote_gate import _fold_glyphs
from portolan.extract.schema import ExtractionResult, FactType, SourceType, Vocabulary
from portolan.extract.schema import Participant as ExtractParticipant
from portolan.ingest.locators import parse_locator
from portolan.ingest.models import RawSource
from portolan.ingest.prepare import prepare_text
from portolan.metering import InMemorySink
from portolan.pipeline import build_envelopes
from portolan.training.label_judge import (
    JUDGE_PROMPT_VERSION,
    JudgeCall,
    JudgeFact,
    JudgeLogEntry,
    judge_facts,
    retype_allowed,
)
from portolan.training.overlap_gate import overlap_gate_reason
from portolan.training.quote_span import resolve_quote_span
from portolan.training.records import teacher_answer_fields, teacher_fact_fields

__all__ = [
    "AUDIT_SHARE",
    "AUDIT_TEACHERS",
    "LABEL_SOURCE_SILVER",
    "PASS_VERSION",
    "PROMPT_VARIANT",
    "TEACHER_MODEL",
    "CompletenessContext",
    "DocCompleteness",
    "Regenerator",
    "SilverLabel",
    "SourceInput",
    "TeacherFact",
    "TeacherRunBudget",
    "WriterRegenerator",
    "build_qa_report",
    "decide",
    "pick_audit_teacher",
    "process_document",
    "openrouter_teacher_factory",
    "run_completeness",
    "select_audit_docs",
    "source_inputs",
    "spans_match",
    "teacher_fields",
    "world_company_context",
]

PASS_VERSION = "wp5-2026-09-23.13"
TEACHER_MODEL = "moonshotai/kimi-k2"
PROMPT_VARIANT = "v6-type-precision"
LABEL_SOURCE_SILVER = "silver_completeness"

# Spec section 3.5's audit: "a second teacher (another family)". Both are
# open-weight; the first one whose family differs from the labeling
# teacher's and the document writer's is used.
AUDIT_TEACHERS: tuple[str, ...] = ("qwen/qwen3-235b-a22b-2507", "deepseek/deepseek-v3.2")
# The audit teacher also labels (orchestrator, 2026-09-22), and the residual
# needs both teachers on the same documents, so it reads every document.
AUDIT_SHARE = 1.0

MATCH_IOU = 0.5  # spec section 3.3
RESTATEMENT_SIMILARITY = 0.8  # spec section 3.3; the audit's "same fact" bar
# A teacher fact is the same as one of the document's planted facts (labeled
# or not) at the bench's calibrated same-fact bar with the same type
# (`portolan.bench.extraction._CLAIM_SIMILARITY_THRESHOLD`), or a stricter
# bar across types (a planted template's type is often not the teacher's).
PLANTED_SAME_TYPE_SIMILARITY = 0.30
PLANTED_ANY_TYPE_SIMILARITY = 0.40
DECOY_REVIEW_RATE = 0.30  # spec section 4: a class hit this often goes to hand review
HALVARD_RECALL = 0.931  # spec section 3.3; pilot gate: within 8 points

# Spec section 3.5, verbatim thresholds. `max` gates pass at or below the
# threshold, `min` gates at or above it.
# The residual's 2% is replaced by the pilot gate (orchestrator, spec Build
# notes, WP5, 2026-09-22): residual at most 3% on a whole world, silver-label
# precision at least 90% per teacher on a 30-label by-eye sample, and total
# label error (residual plus estimated wrong labels) at most 5%.
GATES: tuple[tuple[str, str, float], ...] = (
    ("scripted_facts_verified", "min", 0.90),
    ("silver_share_of_labeled_facts", "max", 0.15),
    ("documents_regenerated", "max", 0.20),
    ("teacher_recall_on_scripted", "min", 0.85),
    ("residual_unlabeled_share", "max", 0.03),
    ("silver_precision_min_teacher", "min", 0.90),
    ("total_label_error", "max", 0.05),
    ("audit_new_share", "max", 0.03),
)
PRECISION_SAMPLE = 30  # silver labels per teacher, read by eye at the pilot

# USD per million tokens, for the pre-call worst-case check only (the cost
# recorded is OpenRouter's own `usage.cost`). Kimi K2 from spec section 8;
# the audit teachers at the section's audit ceiling (Kimi K2.6's price).
TEACHER_PRICES: dict[str, tuple[float, float]] = {
    TEACHER_MODEL: (0.57, 2.30),
    **{m: (0.95, 4.00) for m in AUDIT_TEACHERS},
}
# `OpenRouterClient.complete`'s default output cap, and `LLMClientBackend`'s
# strict attempt plus one validation retry: the worst case of one extract.
TEACHER_MAX_OUTPUT_TOKENS = 4096
TEACHER_REQUESTS_PER_EXTRACT = 2

_FILE_CONNECTOR_KINDS = frozenset({DocKind.CALL, DocKind.TICKET, DocKind.EMAIL})


def _family(model: str) -> str:
    info = ALLOWED_WRITER_MODELS.get(model)
    if info is not None:
        return info.family
    prefix = model.split("/", 1)[0].lower()
    return {"moonshotai": "kimi", "z-ai": "glm", "mistralai": "mistral"}.get(prefix, prefix)


def pick_audit_teacher(writer_model: str | None, teacher_model: str = TEACHER_MODEL) -> str:
    banned = {_family(teacher_model)}
    if writer_model:
        banned.add(_family(writer_model))
    for model in AUDIT_TEACHERS:
        if _family(model) not in banned:
            return model
    return AUDIT_TEACHERS[0]


# --------------------------------------------------------------------------
# Budget (one cap for teacher and regeneration calls)
# --------------------------------------------------------------------------


@dataclass
class TeacherRunBudget(RunBudget):
    """WP4's `RunBudget` (same cap, meter, corpus key and stop exception)
    with teacher prices for the worst-case check. Passing this one object
    to the regeneration writer session keeps a single cap per run."""

    def worst_case_usd(self, model: str, prompt_chars: int, max_tokens: int) -> float:
        price = TEACHER_PRICES.get(model)
        if price is None:
            return super().worst_case_usd(model, prompt_chars, max_tokens)
        list_price = (math.ceil(prompt_chars / 3.0) * price[0] + max_tokens * price[1]) / 1e6
        return list_price * self.price_safety_factor


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------

Triage = Literal["confirmed", "decoy_hit", "restatement", "planted", "candidate"]


class TeacherFact(BaseModel):
    """One fact a teacher returned, with its gate result, its exact span in
    the source's prepared text when one resolves, and its triage.

    A fact the gate drops only for its claim (numeral or name rule) or its
    quote's exact span is still *grounded*: the text says something there.
    It is triaged like a kept fact but can never become a label, so as a
    candidate it is `unlabelable` and blocks the document (it would ship
    as an unlabeled positive)."""

    model_config = ConfigDict(extra="forbid")

    teacher_model: str
    role: Literal["completeness", "audit"]
    doc_id: str
    envelope: int
    fact_type: str
    claim: str
    quote: str | None = None
    entities: list[str] = Field(default_factory=list)
    locator: str = ""
    speaker: str | None = None
    gate: str = "kept"
    char_start: int | None = None
    char_end: int | None = None
    span_text: str | None = None
    tier: str | None = None
    duplicate: bool = False
    triage: Triage | None = None
    matched_id: str | None = None
    decoy_class_id: str | None = None
    decoy_verified: bool | None = None
    type_agrees: bool | None = None
    iou: float | None = None
    similarity: float | None = None
    recovers_fact_id: str | None = None
    contradictions: list[str] = Field(default_factory=list)
    unknown_entities: list[str] = Field(default_factory=list)
    unscripted_numbers: list[str] = Field(default_factory=list)
    unlabelable: bool = False
    # Why this candidate sends the document back for a rewrite (a world
    # contradiction, an invented entity or price, a broken date).
    blockers: list[str] = Field(default_factory=list)
    audit_status: Literal["known", "decoy_hit", "new"] | None = None
    labeled: bool = False  # an audit fact that became a label
    # The teacher's own sentiment, confidence, severity, speaker and validity
    # dates (`records.teacher_fact_fields`), so an exported training answer
    # never needs placeholders. `None` on facts stored before 2026-09-23.
    answer_fields: dict[str, Any] | None = None

    @property
    def usable(self) -> bool:
        return self.gate == "kept" and not self.duplicate

    @property
    def grounded(self) -> bool:
        return self.gate != "quote_gate" and not self.duplicate

    @property
    def has_span(self) -> bool:
        return self.char_start is not None and self.char_end is not None

    @property
    def span(self) -> tuple[int, int]:
        assert self.char_start is not None and self.char_end is not None
        return self.char_start, self.char_end


ClaimSource = Literal["teacher", "teacher_restatement", "scripted"]


class SilverLabel(LabelRecord):
    """A label the completeness pass adds (spec section 3.4:
    `label_source: silver_completeness`, teacher id, exact span), with the
    teacher's claim and type. A restatement of a labeled fact
    (`restates_fact_id`) or a planted fact WP4 did not label
    (`recovers_fact_id`) reuses the scripted fact's id; a new fact gets
    `<plan_id>-C<n>`. `locator` and `prepared_locator` are both the
    `prepare_text` locator of the line the span starts in."""

    plan_id: str
    teacher_model: str
    claim: str
    claim_source: ClaimSource = "teacher"
    # The claim teacher's own fact values plus its entities (`teacher_fields`);
    # `None` when the claim is scripted or the teacher output predates them.
    teacher_fields: dict[str, Any] | None = None
    restates_fact_id: str | None = None
    recovers_fact_id: str | None = None
    type_agrees: bool | None = None  # a recovered fact: teacher type == scripted type
    # Every teacher with a grounded fact on this span (`teacher_model` wrote the claim).
    found_by: list[str] = Field(default_factory=list)
    prepared_locator: str
    span_words: int
    # The teacher's own type when the label judge retyped this label.
    teacher_fact_type: str | None = None

    @property
    def new_fact(self) -> bool:
        return self.restates_fact_id is None and self.recovers_fact_id is None


class JudgedLabel(BaseModel):
    """One silver label's label-judge outcome. `unresolved`: the judge gave
    no usable verdict (or a retype fact rule 5 refuses); the label ships
    unchanged and the document counts as not fully judged."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str
    char_start: int
    char_end: int
    teacher_model: str
    fact_id: str
    verdict: Literal["keep", "drop", "retype", "unresolved"]
    type_before: str
    type_after: str
    reason: str = ""


class DocJudge(BaseModel):
    """The label judge on one document's silver labels. `input_labels` are
    the labels as they stood before it, so a new judge model or prompt
    re-judges from the same input."""

    model_config = ConfigDict(extra="forbid")

    model: str
    prompt_version: str
    input_labels: list[SilverLabel] = Field(default_factory=list)
    labels: list[JudgedLabel] = Field(default_factory=list)
    dropped: list[SilverLabel] = Field(default_factory=list)
    calls: int = 0
    cached: int = 0
    cost_usd: float = 0.0
    errors: list[str] = Field(default_factory=list)

    @property
    def unresolved(self) -> int:
        return sum(1 for x in self.labels if x.verdict == "unresolved")


class CompletedLabel(WriterLabel):
    """A scripted label as it ships: WP4's verified span and type, with the
    claim the student learns from. The claim is the blind teacher's own
    sentence when it confirmed this span with the same type (`teacher`),
    or found the same fact only at another mention (`teacher_restatement`);
    otherwise the label stays, per spec section 3.3, with the scripted
    claim (`scripted`), and the document is not `claims_complete`. The
    scripted claim is always kept as `scripted_claim` for audits."""

    claim: str = ""
    claim_source: ClaimSource = "scripted"
    scripted_claim: str = ""
    claim_teacher_model: str | None = None
    teacher_type: str | None = None  # the teacher's type when it disagreed
    teacher_fields: dict[str, Any] | None = None  # as on `SilverLabel`


class TeacherCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plan_id: str
    role: Literal["completeness", "audit"]
    attempt: int
    model: str
    source_id: str
    envelope: int
    prompt_chars: int
    cost_usd: float = 0.0
    facts: int = 0
    # Facts the lenient parser dropped from the teacher's reply (schema
    # failures), with the first few reasons.
    rejected: int = 0
    rejected_reasons: list[str] = Field(default_factory=list)
    error: str | None = None
    at: str
    # Paid in an earlier run and reused (`teacher_key` matched); not new spend.
    reused: bool = False
    # The non-fact half of the teacher's answer for this envelope
    # (`records.teacher_answer_fields`); `None` on calls stored before 2026-09-23.
    answer: dict[str, Any] | None = None


class PassAttempt(BaseModel):
    """One teacher pass over one version of a document (the WP4 original,
    or WP5's regeneration)."""

    model_config = ConfigDict(extra="forbid")

    version: Literal["original", "regenerated"]
    writer_model: str | None
    scripted_labels: int
    facts: list[TeacherFact] = Field(default_factory=list)
    calls: list[TeacherCall] = Field(default_factory=list)
    decision: str | None = None
    reason: str | None = None
    # Planted decoys whose sentence is in this version's text, recovered ones included.
    decoys_in_text: list[str] = Field(default_factory=list)

    def usable(self) -> list[TeacherFact]:
        return [f for f in self.facts if f.usable]

    def of(self, triage: Triage) -> list[TeacherFact]:
        """Every grounded fact with this triage, gated or not."""
        return [f for f in self.facts if f.grounded and f.triage == triage]

    def open_facts(self) -> list[TeacherFact]:
        """U: the candidates, plus planted facts the text states that could
        not be labeled (both would ship as unlabeled positives)."""
        return self.of("candidate") + [f for f in self.of("planted") if f.unlabelable]


class AuditResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    facts: list[TeacherFact] = Field(default_factory=list)
    calls: list[TeacherCall] = Field(default_factory=list)
    # Labels before the audit's own: its "new" facts are measured against these.
    labeled_facts: int = 0
    labels: list[SilverLabel] = Field(default_factory=list)
    # Spans where the audit teacher typed a labeled fact differently; the
    # label keeps its type (the labeling teacher's, or the script's).
    type_disagreements: list[dict[str, Any]] = Field(default_factory=list)

    def new(self) -> list[TeacherFact]:
        return [f for f in self.facts if f.grounded and f.audit_status == "new"]


DocStatus = Literal[
    "accepted",
    "clean_negative",
    "relabeled_positive",
    "dropped",
    "needs_regeneration",
    "error",
    "in_progress",
]
FINAL_STATUSES = frozenset(
    {"accepted", "clean_negative", "relabeled_positive", "dropped", "needs_regeneration"}
)
SHIPPED_STATUSES = frozenset({"accepted", "clean_negative", "relabeled_positive"})


class DocCompleteness(BaseModel):
    """One document's WP5 checkpoint, `docs/<plan_id>.json`."""

    model_config = ConfigDict(extra="forbid")

    plan_id: str
    kind: DocKind
    fingerprint: str
    # What the teacher's first pass depends on; a later pass version reuses
    # that pass (re-gated and re-triaged) when this still matches.
    teacher_key: str = ""
    status: DocStatus = "in_progress"
    reason: str | None = None
    doc_class: Literal["positive", "negative"] = "positive"
    attempts: list[PassAttempt] = Field(default_factory=list)
    regenerated_doc: WrittenDoc | None = None
    regeneration_tried: bool = False
    do_not_state: list[str] = Field(default_factory=list)
    scripted_labels: list[CompletedLabel] = Field(default_factory=list)
    silver_labels: list[SilverLabel] = Field(default_factory=list)
    audit: AuditResult | None = None
    # A previous pass version's audit on the same teacher input, re-gated
    # instead of paid for again.
    audit_seed: AuditResult | None = None
    judge: DocJudge | None = None

    def final_doc(self, original: WrittenDoc) -> WrittenDoc:
        return self.regenerated_doc if self.regenerated_doc is not None else original

    def summary(self) -> dict[str, Any]:
        last = self.attempts[-1] if self.attempts else None
        usable = last.usable() if last else []
        confirmed = [f for f in usable if f.triage == "confirmed"]
        scripted_hit = {f.matched_id for f in confirmed}
        candidates = last.open_facts() if last else []
        n_scripted = last.scripted_labels if last else 0
        return {
            "plan_id": self.plan_id,
            "kind": self.kind.value,
            "status": self.status,
            "attempts": len(self.attempts),
            "writer_model": last.writer_model if last else None,
            "scripted_labels": n_scripted,
            "teacher_facts": len(last.facts) if last else 0,
            "teacher_facts_kept": len(usable),
            "confirmed": len(confirmed),
            "scripted_confirmed": len(scripted_hit),
            "missed_scripted": max(0, n_scripted - len(scripted_hit)),
            "restated_only": len(
                {f.matched_id for f in usable if f.triage == "restatement"} - scripted_hit
            ),
            "type_agrees": sum(1 for f in confirmed if f.type_agrees),
            "decoy_hits": len(last.of("decoy_hit")) if last else 0,
            "restatements": sum(1 for f in usable if f.triage == "restatement"),
            "planted_recovered": len({f.matched_id for f in usable if f.triage == "planted"}),
            "candidates": len(candidates),
            "unlabelable": sum(1 for f in candidates if f.unlabelable),
            "contradictions": sum(1 for f in candidates if f.contradictions),
            "silver_labels": len(self.silver_labels),
            "claims_from_teacher": sum(
                1 for x in self.scripted_labels if x.claim_source != "scripted"
            ),
            "silver_new": sum(1 for s in self.silver_labels if s.new_fact),
            "cost_usd": round(self.cost_usd(), 6),
        }

    def cost_usd(self) -> float:
        teacher = sum(c.cost_usd for a in self.attempts for c in a.calls)
        audit = sum(c.cost_usd for c in self.audit.calls) if self.audit else 0.0
        writer = self.regenerated_doc.cost_usd if self.regenerated_doc else 0.0
        return teacher + audit + writer


# --------------------------------------------------------------------------
# Inputs: what production would send the extractor
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceInput:
    source_id: str
    source_type: SourceType
    source_date: date
    participants: tuple[ExtractParticipant, ...]
    text: str  # the source's `prepare_text` output, which labels index


class PreparedTextDrift(RuntimeError):
    """Re-reading a file-connector source did not reproduce the prepared
    text WP4 stored, so its label offsets cannot be trusted."""


_CHAT_SPEAKER_RE = re.compile(r"^\[[^\]]+\]\s+([^:\n]{1,80}):", re.MULTILINE)


def source_inputs(doc: WrittenDoc, plan: DocPlan) -> list[SourceInput]:
    """Calls, tickets and emails go back through `FileConnector` and
    `prepare_text` (the production path, which also gives the participants
    and the source date) and must reproduce WP4's stored prepared text.
    Chat and prose kinds have no connector yet: their stored prepared text
    is used as is, dated by the plan (chat participants are the speakers)."""
    stored = {p.source_id: p for p in doc.prepared_sources}
    out: list[SourceInput] = []
    if plan.kind in _FILE_CONNECTOR_KINDS:
        from portolan.connectors.file_connector import FileConnector

        with tempfile.TemporaryDirectory(prefix="wp5-sources-") as tmp:
            root = Path(tmp)
            for rel, content in doc.files.items():
                target = root / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
            if plan.kind == DocKind.CALL:
                # A transcript carries no date of its own; `FileConnector`
                # reads it from the world's manifest (spec section 1's
                # `manifest.json`), else falls back to today.
                manifest = {
                    "sources": [
                        {"id": r.source_id, "date": plan.date.isoformat()}
                        for r in doc.prepared_sources
                    ]
                }
                (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            connector = FileConnector(root)
            refs = {ref.ref.get("path"): ref for ref in connector.list_sources()}
            for record in doc.prepared_sources:
                ref = refs.get(record.path)
                if ref is None:
                    raise PreparedTextDrift(f"FileConnector did not list {record.path}")
                raw = connector.fetch(ref)
                prepared = prepare_text(raw).text
                if raw.source_id != record.source_id or prepared != record.text:
                    raise PreparedTextDrift(
                        f"{record.source_id}: prepared text differs from WP4's stored text"
                    )
                out.append(
                    SourceInput(
                        source_id=raw.source_id,
                        source_type=raw.source_type,
                        source_date=raw.source_date,
                        participants=tuple(
                            ExtractParticipant(name=p.name, role=p.role, registry_id=None)
                            for p in raw.participants
                        ),
                        text=prepared,
                    )
                )
        return out
    for record in stored.values():
        source_type = SourceType(record.source_type)
        names: list[str] = []
        if plan.kind in (DocKind.SLACK, DocKind.TEAMS):
            names = list(
                dict.fromkeys(m.group(1).strip() for m in _CHAT_SPEAKER_RE.finditer(record.text))
            )
        out.append(
            SourceInput(
                source_id=record.source_id,
                source_type=source_type,
                source_date=plan.date,
                participants=tuple(ExtractParticipant(name=n) for n in names),
                text=record.text,
            )
        )
    return out


def world_company_context(world: World) -> str:
    """The world's vendor as the extractor's company context (spec open
    decision 2's recommended option), rendered by code from world fields:
    names only, no model-facing prose beyond the field labels."""
    lines = [
        f"Company: {world.vendor.name}",
        f"Vertical: {world.vertical.replace('_', ' ')}",
        "Products: " + ", ".join(p.name for p in world.vendor.products),
        "Competitors: " + ", ".join(c.name for c in world.vendor.competitors),
        "Customers: the accounts named in each source",
    ]
    if world.vendor.overview.strip():
        lines.append(world.vendor.overview.strip())
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Teacher calls and the gate
# --------------------------------------------------------------------------


class TeacherCallError(RuntimeError):
    """The teacher failed twice on one envelope; the document is left
    unfinished (status `error`) and a later run retries it."""


@dataclass
class LabelJudgeSettings:
    """The label judge (`portolan.training.label_judge`) for WP5's silver
    labels: its model, the call (None: cached replies only), a budget check
    made before every call, and the call log that is also its cache."""

    model: str
    call: JudgeCall | None
    budget_left: Callable[[], bool] = lambda: True
    log_path: Path | None = None


@dataclass
class CompletenessContext:
    """Everything the pass needs besides the document itself."""

    world: World
    world_plan: WorldDocPlan
    backend_factory: Callable[[str], Backend]
    budget: RunBudget
    teacher_model: str = TEACHER_MODEL
    company_context: str | None = None
    prompt_variant: str = PROMPT_VARIANT
    regenerator: Regenerator | None = None
    sink: InMemorySink = field(default_factory=InMemorySink)
    tenant: str = "corpus-generator-v2-wp5"
    sleep: Callable[[float], None] = time.sleep
    now: Callable[[], datetime] = lambda: datetime.now(UTC)
    call_log: Path | None = None
    # True when the teacher client records each call on the spend meter itself
    # (`OpenRouterClient(spend_meter=...)`); the budget then only counts it toward the cap.
    client_meters: bool = False
    label_judge: LabelJudgeSettings | None = None
    _system_prompt: str | None = None
    _lexicon: WorldLexicon | None = None

    @property
    def system_prompt(self) -> str:
        if self._system_prompt is None:
            self._system_prompt = build_system_prompt(
                company_context=self.company_context, variant=self.prompt_variant
            )
        return self._system_prompt

    @property
    def context_names(self) -> str:
        """The vendor's and its products' names, when the teacher gets the
        world's company context (they are in its input, not only the text)."""
        if not self.company_context:
            return ""
        vendor = self.world.vendor
        names = [vendor.name] + [n for p in vendor.products for n in (p.name, *p.aliases)]
        return "\n".join(names)

    @property
    def lexicon(self) -> WorldLexicon:
        if self._lexicon is None:
            self._lexicon = WorldLexicon.build(self.world)
        return self._lexicon

    @property
    def decoys_by_id(self) -> dict[str, PlannedDecoy]:
        return {d.id: d for d in self.world_plan.decoys}

    def log(self, record: TeacherCall) -> None:
        if self.call_log is None:
            return
        self.call_log.parent.mkdir(parents=True, exist_ok=True)
        with self.call_log.open("a", encoding="utf-8") as fh:
            fh.write(record.model_dump_json() + "\n")


def _extract(
    ctx: CompletenessContext,
    *,
    model: str,
    envelope: Any,
    record: TeacherCall,
) -> ExtractionResult:
    """One budget-checked teacher extract with one retry (the silver
    pipeline's rule). Every cent lands on the meter, failed calls too."""
    user = build_user_message(envelope)
    prompt_chars = len(ctx.system_prompt) + len(user)
    record.prompt_chars = prompt_chars
    last_error: Exception | None = None
    for attempt in range(2):
        ctx.budget.precheck(
            model,
            prompt_chars * TEACHER_REQUESTS_PER_EXTRACT,
            TEACHER_MAX_OUTPUT_TOKENS * TEACHER_REQUESTS_PER_EXTRACT,
        )
        before = len(ctx.sink.events)
        try:
            return ctx.backend_factory(model).extract(
                envelope, ctx.system_prompt, ctx.sink, ctx.tenant
            )
        except WriterBudgetExceeded:
            raise
        except Exception as exc:  # noqa: BLE001 -- retried once, then reported
            last_error = exc
            if attempt == 0:
                ctx.sleep(2.0)
        finally:
            cost = sum((e.cost_estimate or 0.0) for e in ctx.sink.events[before:])
            if ctx.client_meters:
                ctx.budget.spent_usd = round(ctx.budget.spent_usd + max(0.0, cost), 8)
            else:
                ctx.budget.record(model, cost)
            record.cost_usd = round(record.cost_usd + cost, 8)
    record.error = f"{type(last_error).__name__}: {last_error}"[:500]
    raise TeacherCallError(record.error)


_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")
MIN_SENTENCE_WORDS = 4


def _resolve(quote: str | None, text: str) -> tuple[Any, str] | None:
    """`resolve_quote_span`, then, like its stitched-quote rule, the quote's
    first sentence alone (a teacher quote that runs two sentences together
    across one it skipped resolves nowhere as a whole)."""
    resolved = resolve_quote_span(quote, text)
    if resolved is not None:
        return resolved, resolved.tier
    first = _SENTENCE_END_RE.split((quote or "").strip(), maxsplit=1)[0]
    if len(first.split()) < MIN_SENTENCE_WORDS or first == (quote or "").strip():
        return None
    resolved = resolve_quote_span(first, text)
    return (resolved, f"sentence:{resolved.tier}") if resolved is not None else None


_ORDINAL_RE = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)\b")


def numeral_extras(text: str, source_date: date) -> str:
    """WP4's `coverage_extras` (the source date, spelled dates and numbers
    as ISO dates and digits) plus each of those dates' year, month and day
    alone and bare ordinals: the numeral rule compares single tokens, so a
    claim's "February 2, 2026" needs "2" and "2026", not "2026-02-02"."""
    extras = coverage_extras(text, source_date).split()
    out = list(extras)
    for e in extras:
        if _ISO_RE.fullmatch(e):
            year, month, day = e.split("-")
            out += [year, month, day, str(int(month)), str(int(day))]
    out += _ORDINAL_RE.findall(text)
    return " ".join(out)


# Calendar words are not names; the numeral rule checks the dates they carry.
CALENDAR_WORDS = (
    "January February March April May June July August September October November "
    "December Monday Tuesday Wednesday Thursday Friday Saturday Sunday"
)


def _gate(
    fact: Any,
    *,
    envelope_text: str,
    offset: int,
    source: SourceInput,
    base: TeacherFact,
    context_names: str = "",
) -> TeacherFact:
    """The silver pipeline's gate (`run_corpus_single`): the quote gate and
    the numeral-and-name rule with content overlap off, then exact-span
    resolution, mapped from the envelope into the source's prepared text.
    The numeral-and-name rule reads the claim the way WP4's check 6 does
    (`rule_claim`, and `numeral_extras` count as covered), so "Hospital
    2's" and "2026-01-28" for "January 28" do not drop a faithful fact.
    Month and weekday names, and `context_names` (the vendor and product
    names the teacher read in its company context), count as covered."""
    coverage = "\n".join(
        [
            envelope_text,
            numeral_extras(envelope_text, source.source_date),
            CALENDAR_WORDS,
            context_names,
        ]
    )
    reason = overlap_gate_reason(
        claim=rule_claim(fact.claim), quote=fact.quote, doc_text=coverage, threshold=0.0
    )
    if reason == "quote_gate":
        base.gate = reason
        return base
    found = _resolve(fact.quote, envelope_text)
    if found is None:
        base.gate = reason or "quote_not_exact"
        found = _probe(fact.quote, envelope_text)
        if found is None:
            return base
    elif reason is not None:
        base.gate = reason
    resolved, tier = found
    if offset >= 0:
        start, end = offset + resolved.span[0], offset + resolved.span[1]
    else:
        again = resolve_quote_span(resolved.quote, source.text)
        if again is None:
            base.gate = base.gate if base.gate != "kept" else "quote_not_exact"
            return base
        start, end = again.span
    assert source.text[start:end] == resolved.quote
    base.char_start, base.char_end = start, end
    base.span_text = resolved.quote
    base.tier = tier if base.gate == "kept" else f"probe:{tier}"
    return base


def _probe(quote: str | None, text: str) -> tuple[Any, str] | None:
    """Where a dropped fact sits, for triage only: the first sentence of
    its quote (any position) that resolves exactly."""
    for sentence in _SENTENCE_END_RE.split((quote or "").strip()):
        if len(sentence.split()) < MIN_SENTENCE_WORDS:
            continue
        resolved = resolve_quote_span(sentence, text)
        if resolved is not None:
            return resolved, f"sentence:{resolved.tier}"
    return None


def _source_envelopes(source: SourceInput) -> list[tuple[int, Any, int]]:
    """`(index, envelope, offset into the source's prepared text)`, exactly
    the envelopes production builds (`build_envelopes`)."""
    raw = RawSource(
        source_id=source.source_id,
        source_type=source.source_type,
        source_date=source.source_date,
        text=source.text,
    )
    envelopes = build_envelopes(
        raw,
        source.text,
        participants=list(source.participants),
        registry_context=[],
        vocabulary=Vocabulary(),
    )
    out = []
    cursor = 0
    for index, envelope in enumerate(envelopes, start=1):
        offset = 0
        if envelope.segment is not None:
            offset = source.text.find(envelope.text, cursor)
            if offset >= 0:
                cursor = offset + 1
        out.append((index, envelope, offset))
    return out


def _mark_duplicates(facts: list[TeacherFact]) -> None:
    seen: set[tuple[Any, ...]] = set()
    for f in sorted(facts, key=lambda f: f.gate != "kept"):
        if f.gate == "quote_gate":
            continue
        where: tuple[Any, ...] = f.span if f.has_span else (_norm(f.quote or ""),)
        key = (f.doc_id, *where, f.fact_type)
        if key in seen:
            f.duplicate = True  # the same fact again from an overlapping segment
        seen.add(key)


def teacher_pass(
    ctx: CompletenessContext,
    sources: list[SourceInput],
    *,
    model: str,
    role: Literal["completeness", "audit"],
    plan_id: str,
    attempt: int,
) -> tuple[list[TeacherFact], list[TeacherCall]]:
    facts: list[TeacherFact] = []
    calls: list[TeacherCall] = []
    for source in sources:
        for index, envelope, offset in _source_envelopes(source):
            record = TeacherCall(
                plan_id=plan_id,
                role=role,
                attempt=attempt,
                model=model,
                source_id=source.source_id,
                envelope=index,
                prompt_chars=0,
                at=ctx.now().isoformat(),
            )
            try:
                result = _extract(ctx, model=model, envelope=envelope, record=record)
            except BaseException:
                calls.append(record)
                ctx.log(record)
                raise
            record.facts = len(result.facts)
            record.answer = teacher_answer_fields(result)
            record.rejected = len(result.rejected)
            record.rejected_reasons = [r.error[:200] for r in result.rejected[:5]]
            calls.append(record)
            ctx.log(record)
            for fact in result.facts:
                base = TeacherFact(
                    teacher_model=model,
                    role=role,
                    doc_id=source.source_id,
                    envelope=index,
                    fact_type=str(fact.type),
                    claim=fact.claim,
                    quote=fact.quote,
                    entities=[str(e) for e in fact.entities],
                    locator=fact.locator,
                    speaker=fact.speaker,
                    answer_fields=teacher_fact_fields(fact),
                )
                facts.append(
                    _gate(
                        fact,
                        envelope_text=envelope.text,
                        offset=offset,
                        source=source,
                        base=base,
                        context_names=ctx.context_names,
                    )
                )
    _mark_duplicates(facts)
    return facts, calls


def regate(
    ctx: CompletenessContext, stored: list[TeacherFact], sources: list[SourceInput]
) -> list[TeacherFact]:
    """The teacher's stored raw output (type, claim, quote, entities) run
    through today's gate, as if it had just been returned."""
    envelopes = {
        (source.source_id, index): (envelope, offset, source)
        for source in sources
        for index, envelope, offset in _source_envelopes(source)
    }
    facts: list[TeacherFact] = []
    for old in stored:
        envelope, offset, source = envelopes[(old.doc_id, old.envelope)]
        base = TeacherFact(
            teacher_model=old.teacher_model,
            role=old.role,
            doc_id=old.doc_id,
            envelope=old.envelope,
            fact_type=old.fact_type,
            claim=old.claim,
            quote=old.quote,
            entities=list(old.entities),
            locator=old.locator,
            speaker=old.speaker,
            answer_fields=old.answer_fields,
        )
        facts.append(
            _gate(
                SimpleNamespace(claim=old.claim, quote=old.quote),
                envelope_text=envelope.text,
                offset=offset,
                source=source,
                base=base,
                context_names=ctx.context_names,
            )
        )
    _mark_duplicates(facts)
    return facts


# --------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------


def span_iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def spans_match(a: tuple[int, int], b: tuple[int, int]) -> bool:
    """Spec section 3.3: character IoU >= 0.5, or one span containing the
    other."""
    if span_iou(a, b) >= MATCH_IOU:
        return True
    return (a[0] <= b[0] and b[1] <= a[1]) or (b[0] <= a[0] and a[1] <= b[1])


_EDGE_CHARS = frozenset(" \t\n.,;:!?\"'()[]-\u2013\u2014\u2018\u2019\u201c\u201d")


def core_span(text: str, span: tuple[int, int]) -> tuple[int, int]:
    """`span` without leading or trailing whitespace and punctuation, so a
    quote that stops before a sentence's full stop still sits inside the
    label that includes it."""
    start, end = span
    while start < end and text[start] in _EDGE_CHARS:
        start += 1
    while end > start and text[end - 1] in _EDGE_CHARS:
        end -= 1
    return (start, end) if start < end else span


@dataclass(frozen=True)
class ScriptedSpan:
    kind: Literal["fact", "decoy"]
    marker_id: str
    doc_id: str
    start: int
    end: int
    verified: bool
    fact_type: str | None = None
    claim: str | None = None
    decoy_class_id: str | None = None

    @property
    def span(self) -> tuple[int, int]:
        return self.start, self.end


_ANY_MARKER_RE = re.compile(r"⟦|⟧")


def unmatched_marker_text(raw: str, marker_id: str) -> list[str]:
    """The sentence a writer wrapped in a marker it never closed (or closed
    without opening): after `⟦ID⟧` up to its closer, the next bracket or the
    line end; or before a lone `⟦/ID⟧` back to the line start. WP4 drops such
    a slot with no `dropped_text`, yet the planted sentence is in the text."""
    out: list[str] = []
    opener, closer = f"⟦{marker_id}⟧", f"⟦/{marker_id}⟧"
    if opener in raw:
        for m in re.finditer(re.escape(opener), raw):
            rest = raw[m.end() :]
            cut = min(
                (i for i in (rest.find(closer), rest.find("\n")) if i >= 0),
                default=len(rest),
            )
            bracket = _ANY_MARKER_RE.search(rest[:cut])
            out.append(rest[: bracket.start() if bracket else cut])
    elif closer in raw:
        for m in re.finditer(re.escape(closer), raw):
            head = raw[: m.start()]
            line = head[head.rfind("\n") + 1 :]
            bracket = list(_ANY_MARKER_RE.finditer(line))
            out.append(line[bracket[-1].end() :] if bracket else line)
    cleaned = [" ".join(t.split()).strip() for t in out]
    return [t for t in cleaned if len(t.split()) >= MIN_SENTENCE_WORDS]


def _recovered_decoy_spans(
    doc: WrittenDoc,
    sources: list[SourceInput],
    decoys_by_id: dict[str, PlannedDecoy],
    known: set[str],
) -> list[ScriptedSpan]:
    """Unverified decoy spans for dropped decoy slots with no `dropped_text`,
    read from the writer's raw replies and kept only where the sentence is in
    the shipped text."""
    out: list[ScriptedSpan] = []
    for outcome in doc.slot_outcomes:
        if outcome.slot_kind != "decoy" or outcome.status != "dropped":
            continue
        if outcome.dropped_text or outcome.marker_id in known:
            continue
        found = False
        planned = decoys_by_id.get(outcome.marker_id)
        forbidden = planned.forbidden_patterns if planned else []
        for call in reversed(doc.calls):
            for sentence in unmatched_marker_text(call.raw_output or "", outcome.marker_id):
                if forbidden_pattern_hits(sentence, forbidden):
                    continue
                for source in sources:
                    resolved = resolve_quote_span(sentence, source.text)
                    if resolved is None:
                        continue
                    decoy = decoys_by_id.get(outcome.marker_id)
                    out.append(
                        ScriptedSpan(
                            "decoy",
                            outcome.marker_id,
                            source.source_id,
                            *resolved.span,
                            False,
                            decoy_class_id=decoy.decoy_class_id if decoy else None,
                        )
                    )
                    found = True
                    break
                if found:
                    break
            if found:
                break
    return out


def scripted_spans(
    doc: WrittenDoc,
    sources: list[SourceInput],
    world: World,
    decoys_by_id: dict[str, PlannedDecoy],
) -> list[ScriptedSpan]:
    """Verified label and decoy spans, plus the spans of slots WP4 dropped
    whose sentence is still in the text (their `dropped_text`, or for a
    decoy whose marker was left unclosed, the sentence after it)."""
    facts = {f.id: f for f in world.facts}
    out: list[ScriptedSpan] = []
    for label in doc.labels:
        fact = facts.get(label.fact_id)
        out.append(
            ScriptedSpan(
                "fact",
                label.fact_id,
                label.doc_id,
                label.char_start,
                label.char_end,
                True,
                fact_type=label.fact_type.value,
                claim=fact.claim if fact else None,
            )
        )
    for span in doc.decoy_spans:
        out.append(
            ScriptedSpan(
                "decoy",
                span.decoy_id,
                span.doc_id,
                span.char_start,
                span.char_end,
                True,
                decoy_class_id=span.decoy_class_id,
            )
        )
    for outcome in doc.slot_outcomes:
        if outcome.status != "dropped" or not outcome.dropped_text:
            continue
        if outcome.slot_kind == "decoy" and any(
            f.startswith(FORBIDDEN_PATTERN_FAILURE) for f in outcome.final_failures
        ):
            continue  # it says a real fact (e.g. a competitor ruled out), so it is no decoy
        for source in sources:
            resolved = resolve_quote_span(outcome.dropped_text, source.text)
            if resolved is None:
                continue
            if outcome.slot_kind == "fact":
                fact = facts.get(outcome.marker_id)
                out.append(
                    ScriptedSpan(
                        "fact",
                        outcome.marker_id,
                        source.source_id,
                        *resolved.span,
                        False,
                        fact_type=fact.fact_type.value if fact else None,
                        claim=fact.claim if fact else None,
                    )
                )
            else:
                decoy = decoys_by_id.get(outcome.marker_id)
                out.append(
                    ScriptedSpan(
                        "decoy",
                        outcome.marker_id,
                        source.source_id,
                        *resolved.span,
                        False,
                        decoy_class_id=decoy.decoy_class_id if decoy else None,
                    )
                )
            break
    known = {s.marker_id for s in out if s.kind == "decoy"}
    return out + _recovered_decoy_spans(doc, sources, decoys_by_id, known)


def _best_overlap(
    fact: TeacherFact, spans: Iterable[ScriptedSpan], texts: dict[str, str]
) -> tuple[ScriptedSpan, float] | None:
    text = texts.get(fact.doc_id, "")
    if not fact.has_span:
        # A dropped fact whose quote resolves nowhere: one text inside the
        # other, after normalizing case, glyphs and punctuation.
        quote = _norm(fact.quote or "")
        for s in spans:
            theirs = _norm(text[s.start : s.end])
            if (
                s.doc_id == fact.doc_id
                and quote
                and theirs
                and (theirs in quote or quote in theirs)
            ):
                return s, 0.0
        return None
    mine = core_span(text, fact.span)
    best: tuple[ScriptedSpan, float] | None = None
    for s in spans:
        if s.doc_id != fact.doc_id:
            continue
        theirs = core_span(text, s.span)
        if not spans_match(mine, theirs):
            continue
        iou = span_iou(mine, theirs)
        if best is None or iou > best[1]:
            best = (s, iou)
    return best


def _touching(
    fact: TeacherFact, spans: Iterable[ScriptedSpan], texts: dict[str, str]
) -> tuple[ScriptedSpan, float] | None:
    """The span sharing the most characters with `fact` (edge punctuation
    trimmed), however little: the decoy rule drops any teacher fact that
    touches a planted decoy. A spanless fact falls back to `_best_overlap`."""
    if not fact.has_span:
        return _best_overlap(fact, spans, texts)
    text = texts.get(fact.doc_id, "")
    mine = core_span(text, fact.span)
    best: tuple[ScriptedSpan, float, int] | None = None
    for s in spans:
        if s.doc_id != fact.doc_id:
            continue
        theirs = core_span(text, s.span)
        shared = min(mine[1], theirs[1]) - max(mine[0], theirs[0])
        if shared <= 0:
            continue
        if best is None or shared > best[2]:
            best = (s, span_iou(mine, theirs), shared)
    return (best[0], best[1]) if best is not None else None


def _entity_tokens(world: World, entity_ids: Iterable[str]) -> frozenset[str]:
    names: list[str] = []
    accounts = world.account_by_id()
    people = world.person_by_id()
    products = {p.id: p for p in world.vendor.products}
    competitors = {c.id: c for c in world.vendor.competitors}
    for e in entity_ids:
        if e in accounts:
            names += [accounts[e].name, *accounts[e].aliases]
        elif e in people:
            names.append(people[e].name)
        elif e in products:
            names += [products[e].name, *products[e].aliases]
        elif e in competitors:
            names += [competitors[e].name, *competitors[e].aliases]
    return frozenset(t for n in names for t in re.findall(r"[a-z0-9]+", n.lower()))


_YEAR_RE = re.compile(r"^(19|20)\d\d$")


def _claim_numbers(claim: str) -> set[str]:
    """A claim's quantities by value, from the gates' shared normalizer
    (`portolan.extract.numerals`): "03" is 3, "$5k" is 5000, "three" is 3,
    and "10%" is a percentage, not the count 10. Dates ("March 27, 2026",
    "3/27", "2026-03-27") and years are not quantities."""
    text = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", " ", claim)
    for pattern in (_MONTH_DAY_RE, _DAY_MONTH_RE, _SLASH_DATE_RE):
        text = pattern.sub(" ", text)
    return {form for form in _document_numbers(text).forms if not _YEAR_RE.match(form)}


def _within_document(claim: str, names: list[str]) -> str:
    """`claim` without the names every fact of one document shares (its
    account and the vendor), and with plural -s dropped: within a document
    those names say nothing about which fact a sentence states."""
    for name in sorted(names, key=len, reverse=True):
        claim = re.sub(re.escape(name) + r"(['’]s)?", " ", claim, flags=re.IGNORECASE)
    return " ".join(re.sub(r"(?<=[a-z]{3})s$", "", w) for w in claim.split())


def _planted_match(
    f: TeacherFact,
    dropped_fact_spans: list[ScriptedSpan],
    planted_ids: list[str],
    texts: dict[str, str],
    world: World,
    confirmed_claims: dict[str, list[str]],
    family: frozenset[str] | set[str] = frozenset(),
) -> tuple[str, str, float] | None:
    """`(fact_id, how, similarity)` when `f` states one of the document's
    planted facts: on the span of a slot WP4 dropped, or by claim (same type
    at `PLANTED_SAME_TYPE_SIMILARITY`, any type at the stricter bar), never
    when both claims carry quantities and none is shared."""
    dropped = _best_overlap(f, dropped_fact_spans, texts)
    if dropped is not None:
        return dropped[0].marker_id, "span", 1.0
    accounts = world.account_by_id()
    shared_names = [world.vendor.name] + [
        n for a in family if a in accounts for n in (accounts[a].name, *accounts[a].aliases)
    ]
    world_facts = {w.id: w for w in world.facts}
    mine = _claim_numbers(f.claim)
    best: tuple[float, bool, str] | None = None
    for fact_id in planted_ids:
        wf = world_facts[fact_id]
        theirs = _claim_numbers(wf.claim)
        if mine and theirs and not mine & theirs:
            continue
        tokens = _entity_tokens(world, wf.entities)
        others = [wf.claim, *confirmed_claims.get(fact_id, [])]
        sims = [_claim_similarity(f.claim, c, tokens) for c in others]
        if shared_names:
            local_tokens = _entity_tokens(world, [e for e in wf.entities if e not in family])
            sims += [
                _claim_similarity(
                    _within_document(f.claim, shared_names),
                    _within_document(c, shared_names),
                    local_tokens,
                )
                for c in others
            ]
        sim = max(sims)
        same = f.fact_type == wf.fact_type.value
        if sim < (PLANTED_SAME_TYPE_SIMILARITY if same else PLANTED_ANY_TYPE_SIMILARITY):
            continue
        if best is None or (sim, same) > (best[0], best[1]):
            best = (sim, same, fact_id)
    return (best[2], "claim", best[0]) if best is not None else None


def triage(
    facts: list[TeacherFact],
    spans: list[ScriptedSpan],
    *,
    world: World,
    plan: DocPlan,
    sources: list[SourceInput],
    lexicon: WorldLexicon,
) -> None:
    """Sort every grounded completeness fact (in place): confirmed on a
    label span, a decoy hit, a restatement of a labeled fact, a planted
    fact WP4 did not label (`planted`), or a candidate, which is then
    checked against the world state. Only a kept fact counts toward recall
    or can become a label; a dropped one that is a candidate, or a planted
    fact nobody labeled, is unlabelable and blocks the document."""
    world_facts = {f.id: f for f in world.facts}
    fact_spans = [s for s in spans if s.kind == "fact" and s.verified]
    dropped_fact_spans = [s for s in spans if s.kind == "fact" and not s.verified]
    decoy_spans = [s for s in spans if s.kind == "decoy"]
    grounded = sorted((f for f in facts if f.grounded), key=lambda f: not f.usable)
    texts = {s.source_id: s.text for s in sources}
    confirmed_claims: dict[str, list[str]] = {}
    rest: list[TeacherFact] = []
    for f in grounded:
        hit = _best_overlap(f, fact_spans, texts)
        if hit is not None:
            s, iou = hit
            f.triage, f.matched_id, f.iou = "confirmed", s.marker_id, round(iou, 4)
            f.type_agrees = f.fact_type == s.fact_type
            if f.usable:
                confirmed_claims.setdefault(s.marker_id, []).append(f.claim)
            continue
        hit = _touching(f, decoy_spans, texts)
        if hit is not None:
            s, iou = hit
            f.triage, f.matched_id, f.iou = "decoy_hit", s.marker_id, round(iou, 4)
            f.decoy_class_id, f.decoy_verified = s.decoy_class_id, s.verified
            continue
        rest.append(f)
    verified_ids = {s.marker_id for s in fact_spans}
    planted_ids = [i for i in dict.fromkeys([*plan.fact_ids, *verified_ids]) if i in world_facts]
    for f in rest:
        match = _planted_match(
            f,
            dropped_fact_spans,
            planted_ids,
            texts,
            world,
            confirmed_claims,
            _account_family(world, plan.account_id),
        )
        if match is not None:
            fact_id, how, sim = match
            f.matched_id, f.similarity = fact_id, round(sim, 4)
            f.type_agrees = f.fact_type == world_facts[fact_id].fact_type.value
            if fact_id in verified_ids:
                f.triage = "restatement"
            else:
                f.triage, f.recovers_fact_id = "planted", fact_id
                # A dropped claim still located the planted fact: the label
                # takes the scripted claim there, so only a spanless one blocks.
                f.unlabelable = not f.usable and not f.has_span
                f.tier = f"{f.tier or 'none'}|{how}"
            continue
        f.triage = "candidate"
        f.unlabelable = not f.usable
        f.contradictions = contradictions_for(f, plan=plan, world=world, lexicon=lexicon)
    # Flagged on every grounded fact: a writer also invents inside a
    # scripted span ("the new EHR platform, known internally as MedFlow").
    for f in grounded:
        f.unknown_entities = list(
            dict.fromkeys([*lexicon.unknown_entities(f.entities), *lexicon.unknown_names(f.claim)])
        )
        f.unscripted_numbers = unscripted_numbers(
            f.span_text or f.quote or "",
            plan=plan,
            world=world,
            source_text=texts.get(f.doc_id, ""),
        )
        if f.triage == "candidate":
            f.blockers = [
                *f.contradictions,
                *(f"invented_entity:{e}" for e in f.unknown_entities),
                *(f"invented_price:{n}" for n in invented_prices(f, plan=plan, world=world)),
                *(f"date_after_document:{d}" for d in broken_dates(f, plan=plan)),
            ]


# --------------------------------------------------------------------------
# World-state checks (code, lexical)
# --------------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-z0-9]+(?:['’][a-z]+)?")
_ISO_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
_NUMERAL_RE = re.compile(r"\d[\d,.:/%\-]*\d%?|\d%?")
_RENEW_RE = re.compile(r"\brenew", re.IGNORECASE)
# Capitalized words that are not names: acronyms and job words the teacher
# lists as entities. Only used to decide whether an entity string names
# something at all.
_GENERIC_WORDS = frozenset(
    """
    the a an and of for our their its team unit units department dept group staff
    director nursing nurse vp chief officer manager head lead coordinator executive
    engineering support sales account customer vendor hospital health system systems
    clinical pharmacy supply chain procurement finance legal it operations ops board
    january february march april may june july august september october november
    december monday tuesday wednesday thursday friday saturday sunday i
    med surg icu
    """.split()
)
_SENTENCE_START_RE = re.compile(r"(?:^|[.!?:;]\s+|\n)\s*[\"'“‘(]*([A-Za-z][\w'’\-]*)")


def _norm(text: str) -> str:
    folded = _fold_glyphs(text).lower()
    folded = re.sub(r"['’]s\b", "", folded)
    return " ".join(_WORD_RE.findall(folded))


def _contains_phrase(haystack_norm: str, phrase_norm: str) -> bool:
    if not phrase_norm:
        return False
    return (
        re.search(rf"(?<![a-z0-9]){re.escape(phrase_norm)}(?![a-z0-9])", haystack_norm) is not None
    )


@dataclass
class WorldLexicon:
    """Every name the world knows, normalized."""

    known_names: set[str]
    name_tokens: set[str]
    ids: set[str]
    account_names: dict[str, set[str]]  # normalized name or alias -> account ids
    competitor_names: set[str]
    people_by_name: dict[str, Person]

    @classmethod
    def build(cls, world: World) -> WorldLexicon:
        known: set[str] = {_norm(world.vendor.name)}
        account_names: dict[str, set[str]] = {}
        for a in world.accounts:
            for n in (a.name, *a.aliases):
                account_names.setdefault(_norm(n), set()).add(a.id)
                known.add(_norm(n))
        competitor_names: set[str] = set()
        for c in world.vendor.competitors:
            for n in (c.name, *c.aliases):
                competitor_names.add(_norm(n))
                known.add(_norm(n))
        for p in world.vendor.products:
            known.update(_norm(n) for n in (p.name, *p.aliases))
        people_by_name = {_norm(p.name): p for p in world.people}
        known.update(people_by_name)
        tokens = {t for n in known for t in n.split()}
        for p in world.people:
            tokens.update(_norm(f"{p.title} {p.role} {p.chat_handle}").split())
        for a in world.accounts:
            tokens.update(_norm(a.region).split())
        ids = (
            {a.id for a in world.accounts}
            | {p.id for p in world.people}
            | {p.id for p in world.vendor.products}
            | {c.id for c in world.vendor.competitors}
            | {world.vendor.id}
        )
        return cls(known, tokens, ids, account_names, competitor_names, people_by_name)

    def _name_tokens_of(self, entity: str) -> list[str]:
        """The words of `entity` that make it a name: capitalized in the
        teacher's own string, not an all-caps acronym of five letters or
        fewer, not a generic job or org word."""
        out = []
        for raw in re.findall(r"[A-Za-z][A-Za-z0-9'’\-]*", _fold_glyphs(entity)):
            word = re.sub(r"(['’]s|['’])$", "", raw)
            if not word[:1].isupper():
                continue
            if word.isupper() and len(word) <= 5:
                continue
            if all(part.lower() in _GENERIC_WORDS for part in word.split("-")):
                continue
            out.append(word.lower())
        return out

    def _known(self, word: str) -> bool:
        parts = word.split("-")
        return all(
            p in self.name_tokens
            or p in _GENERIC_WORDS
            or (p.endswith("s") and p[:-1] in self.name_tokens)
            for p in parts
        )

    def unknown_entities(self, entities: Iterable[str]) -> list[str]:
        out: list[str] = []
        for entity in entities:
            if entity in self.ids or _norm(entity) in self.known_names:
                continue
            words = self._name_tokens_of(entity)
            if not words or all(self._known(w) for w in words):
                continue
            out.append(entity)
        return out

    def unknown_names(self, text: str) -> list[str]:
        """Name-like words in running text (a claim or span) that the world
        does not know, skipping each sentence's first word: the teacher
        does not always list an invented product among its entities."""
        starts = {m.start(1) for m in _SENTENCE_START_RE.finditer(text)}
        out: list[str] = []
        for m in re.finditer(r"[A-Za-z][A-Za-z0-9'’\-]*", _fold_glyphs(text)):
            if m.start() in starts:
                continue
            [word] = self._name_tokens_of(m.group(0)) or [None]
            if word and not self._known(word) and m.group(0) not in out:
                out.append(m.group(0))
        return out


def _account_family(world: World, account_id: str | None) -> set[str]:
    if account_id is None:
        return set()
    accounts = world.account_by_id()
    family = {account_id}
    acct = accounts.get(account_id)
    if acct is not None and acct.parent:
        family.add(acct.parent)
        family.update(a.id for a in world.accounts if a.parent == acct.parent)
    family.update(a.id for a in world.accounts if a.parent == account_id)
    return family


def _dates_in(text: str, doc_date: date) -> set[str]:
    found = set(_ISO_RE.findall(text))
    extras = coverage_extras(text, doc_date).split()[1:]  # [0] is the doc date itself
    found.update(e for e in extras if _ISO_RE.fullmatch(e))
    return found


def contradictions_for(
    fact: TeacherFact, *, plan: DocPlan, world: World, lexicon: WorldLexicon
) -> list[str]:
    """A candidate that disagrees with the world at the document's date.
    Four lexical checks; meaning is not checked:

    - `other_account`: names an account outside the document's account
      family (parent and siblings included) and none inside it;
    - `unknown_competitor`: a competitor mention that names no world
      competitor but does name an unknown entity;
    - `person_not_at_account`: names a customer person who does not work
      at the document's account family on its date (a contact change
      is exempt, since it is about exactly that);
    - `renewal_date`: talks about a renewal with a date the world has for
      none of the family's contract dates, facts or events.
    """
    out: list[str] = []
    family = _account_family(world, plan.account_id)
    text_norm = _norm(" ".join([fact.claim, fact.span_text or "", *fact.entities]))
    if family:
        named = {
            n: ids for n, ids in lexicon.account_names.items() if _contains_phrase(text_norm, n)
        }
        inside = any(ids & family for ids in named.values())
        outside = sorted(n for n, ids in named.items() if not ids & family)
        if outside and not inside:
            out.append("other_account:" + ", ".join(outside))
    if fact.fact_type == "competitor_mention":
        known = any(_contains_phrase(text_norm, c) for c in lexicon.competitor_names)
        unknown = lexicon.unknown_entities(fact.entities)
        if not known and unknown:
            out.append("unknown_competitor:" + ", ".join(unknown))
    if fact.fact_type != "contact_change" and family:
        for name, person in lexicon.people_by_name.items():
            if person.side != PersonSide.CUSTOMER or not _contains_phrase(text_norm, name):
                continue
            tenure = person.tenure
            at_date = tenure.start <= plan.date and (tenure.end is None or plan.date <= tenure.end)
            if person.account_id not in family or not at_date:
                out.append(f"person_not_at_account:{person.name}")
    if family and _RENEW_RE.search(f"{fact.claim} {fact.span_text or ''}"):
        dates = _dates_in(f"{fact.claim} {fact.span_text or ''}", plan.date)
        allowed = {plan.date.isoformat()}
        for a in world.accounts:
            if a.id in family:
                allowed.update({a.renewal_date.isoformat(), a.contract_start.isoformat()})
        for wf in world.facts:
            if wf.account_id in family:
                allowed.update(
                    d.isoformat() for d in (wf.date, wf.valid_from, wf.valid_to) if d is not None
                )
        allowed.update(e.date.isoformat() for e in world.events if e.account_id in family)
        bad = sorted(dates - allowed)
        if bad:
            out.append("renewal_date:" + ", ".join(bad))
    return out


def unscripted_numbers(
    span_text: str, *, plan: DocPlan, world: World, source_text: str
) -> list[str]:
    """Numerals in a candidate's span that no scripted fact of this
    document carries and no world name contains (informational: the
    "invented numbers" WP4's smoke saw)."""
    facts = {f.id: f for f in world.facts}
    scripted = " ".join(
        " ".join([facts[i].claim, *facts[i].anchors]) for i in plan.fact_ids if i in facts
    )
    allowed = set(re.findall(r"\d+", scripted)) | set(
        re.findall(r"\d+", coverage_extras(scripted, plan.date))
    )
    names = " ".join(
        [a.name for a in world.accounts]
        + [p.name for p in world.vendor.products]
        + [c.name for c in world.vendor.competitors]
    )
    allowed |= set(re.findall(r"\d+", names))
    allowed |= set(re.findall(r"\d+", plan.date.isoformat()))
    text = re.sub(r"^\[[^\]]*\]\s*", "", span_text)
    return [
        m.group(0)
        for m in _NUMERAL_RE.finditer(text)
        if not set(re.findall(r"\d+", m.group(0))) <= allowed
    ]


_PRICE_RE = re.compile(
    r"[$€£]\s?(\d[\d,.]*)\s*(k|m|million|thousand)?\b|(\d[\d,.]*)\s?%"
    r"|(\d[\d,.]*)\s*(dollars|usd)\b",
    re.IGNORECASE,
)


def invented_prices(fact: TeacherFact, *, plan: DocPlan, world: World) -> list[str]:
    """Money amounts and percentages in a candidate that no world fact of
    the account family, and no account's ARR, carries."""
    text = f"{fact.claim} {fact.span_text or ''}"
    found = [m.group(0).strip() for m in _PRICE_RE.finditer(text)]
    if not found:
        return []
    family = _account_family(world, plan.account_id)
    allowed: set[str] = set()
    for wf in world.facts:
        if not family or wf.account_id in family:
            allowed |= set(re.findall(r"\d+", " ".join([wf.claim, *wf.anchors]).replace(",", "")))
    for a in world.accounts:
        if not family or a.id in family:
            arr = int(a.arr)
            allowed |= {str(arr), str(round(arr / 1000)), str(round(arr / 1_000_000))}
    return [
        amount
        for amount in dict.fromkeys(found)
        if not set(re.findall(r"\d+", amount.replace(",", ""))) <= allowed
    ]


_FUTURE_RE = re.compile(
    r"\b(will|would|by|due|expect\w*|plan\w*|schedul\w*|next|deadline|until|before|"
    r"upcoming|going to|target\w*|commit\w*|promis\w*|intend\w*|aim\w*|pending|"
    r"if|unless|could|may|might|want\w*|hope\w*|request\w*|ask\w*)\b",
    re.IGNORECASE,
)


def broken_dates(fact: TeacherFact, *, plan: DocPlan) -> list[str]:
    """Dates after the document's own date that the candidate states as
    past, with no word of a plan, a deadline or a condition."""
    text = f"{fact.claim} {fact.span_text or ''}"
    later = sorted(d for d in _dates_in(text, plan.date) if d > plan.date.isoformat())
    if not later or _FUTURE_RE.search(text):
        return []
    return later


# --------------------------------------------------------------------------
# Per-document rule (spec section 3.4, changed by the 2026-09-22 cost decision)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Decision:
    action: Literal["accept", "clean_negative", "relabel_positive", "regenerate", "drop"]
    reason: str


def _blockers(candidates: list[TeacherFact]) -> str | None:
    reasons = Counter(b.split(":", 1)[0] for c in candidates for b in c.blockers)
    return ", ".join(f"{n} {r}" for r, n in sorted(reasons.items())) or None


def decide(n_scripted: int, candidates: list[TeacherFact], *, regenerated: bool) -> Decision:
    """A candidate consistent with the world is labeled where it stands; only
    a candidate that contradicts the world, names an invented entity or
    price, or states a later date as past sends the document back (once;
    after that it is dropped). This replaces section 3.4's U threshold (cost
    decision, spec Build notes 2026-09-22). An unlabelable candidate never
    triggers a rewrite; it ships as known residual."""
    u = len(candidates)
    blocking = [c for c in candidates if c.blockers]
    if blocking:
        why = f"{len(blocking)} of U={u} need a rewrite ({_blockers(blocking)})"
        return Decision("drop" if regenerated else "regenerate", why)
    if n_scripted > 0:
        return Decision("accept", f"U={u}, all consistent with the world")
    if u == 0:
        return Decision("clean_negative", "U=0")
    if any(not c.unlabelable for c in candidates):
        return Decision("relabel_positive", f"U={u}, all consistent with the world")
    return Decision("accept", f"U={u}, none labelable: known residual")


def _prepared_locator(text: str, start: int) -> str:
    """The locator of the line `start` falls in, or of the nearest line
    above it that has one (a ticket comment's later paragraphs carry none)."""
    line_start = text.rfind("\n", 0, start) + 1
    while True:
        line_end = text.find("\n", line_start)
        found = parse_locator(text[line_start : line_end if line_end >= 0 else len(text)])
        if found is not None:
            return found
        if line_start == 0:
            return ""
        line_start = text.rfind("\n", 0, line_start - 1) + 1


def teacher_fields(fact: TeacherFact) -> dict[str, Any] | None:
    """What a label needs from the teacher fact whose claim it ships: that
    fact's own values plus its entities, or `None` when they were not kept."""
    if fact.answer_fields is None:
        return None
    return {**fact.answer_fields, "entities": list(fact.entities)}


def labelable(attempt: PassAttempt) -> list[TeacherFact]:
    """Facts a silver label can come from: kept ones, and a planted fact the
    gate dropped for its claim but whose span resolved."""
    return [
        f
        for f in attempt.facts
        if f.usable or (f.triage == "planted" and f.has_span and not f.duplicate)
    ]


def silver_labels_for(
    plan: DocPlan,
    facts: list[TeacherFact],
    sources: list[SourceInput],
    world: World,
    prefix: str = "C",
) -> list[SilverLabel]:
    text = {s.source_id: s.text for s in sources}
    world_facts = {w.id: w for w in world.facts}
    out: list[SilverLabel] = []
    chosen = sorted(
        (
            f
            for f in facts
            if f.triage in ("candidate", "planted") or (f.triage == "restatement" and f.type_agrees)
        ),
        key=lambda f: (f.doc_id, f.char_start, f.char_end, f.fact_type),
    )
    n = 0
    for f in chosen:
        start, end = f.span
        span_text = text[f.doc_id][start:end]
        assert span_text == f.span_text
        if f.triage in ("restatement", "planted"):
            fact_id = f.matched_id or ""
        else:
            n += 1
            fact_id = f"{plan.id}-{prefix}{n:02d}"
        claim, fact_type, source = f.claim, f.fact_type, "teacher"
        fields = teacher_fields(f)
        if not f.usable:
            planted = world_facts[fact_id]
            claim, fact_type, source = planted.claim, planted.fact_type.value, "scripted"
            fields = None
        locator = _prepared_locator(text[f.doc_id], start)
        out.append(
            SilverLabel(
                fact_id=fact_id,
                fact_type=fact_type,
                doc_id=f.doc_id,
                source_type=plan.kind.value,
                locator=locator,
                label_source=LABEL_SOURCE_SILVER,
                char_start=start,
                char_end=end,
                text=span_text,
                account_id=plan.account_id or "",
                date=plan.date,
                plan_id=plan.id,
                teacher_model=f.teacher_model,
                claim=claim,
                claim_source=source,
                teacher_fields=fields,
                restates_fact_id=f.matched_id if f.triage == "restatement" else None,
                recovers_fact_id=f.recovers_fact_id if f.triage == "planted" else None,
                type_agrees=f.type_agrees if f.triage == "planted" else None,
                prepared_locator=locator,
                span_words=len(span_text.split()),
            )
        )
    return out


# --------------------------------------------------------------------------
# Regeneration (through WP4's writer)
# --------------------------------------------------------------------------


class Regenerator(Protocol):
    def regenerate(
        self, plan_id: str, do_not_state: list[str], avoid_models: set[str]
    ) -> WrittenDoc | None: ...


@dataclass
class WriterRegenerator:
    """WP4's `write_document` again, with the offending sentences as "do
    not state" and the original writer avoided. The session's budget should
    be the pass's own `TeacherRunBudget` so one cap covers both."""

    world: World
    world_plan: WorldDocPlan
    pack: Any
    session: Any
    finished: dict[str, WrittenDoc] = field(default_factory=dict)

    def regenerate(
        self, plan_id: str, do_not_state: list[str], avoid_models: set[str]
    ) -> WrittenDoc | None:
        from corpus.generator.v2.renderers import render
        from corpus.generator.v2.writer import _context_for, write_document

        plans = self.world_plan.doc_by_id()
        plan = plans.get(plan_id)
        if plan is None:
            return None
        decoys = {d.id: d for d in self.world_plan.decoys}
        rendered = render(plan, self.world, self.world_plan, self.pack)
        doc = write_document(
            rendered,
            plan,
            self.world,
            self.pack,
            decoys_by_id=decoys,
            session=self.session,
            context=_context_for(plan, self.finished, plans),
            do_not_state=do_not_state,
            avoid_models=avoid_models,
        )
        if doc.status == "written" and doc.writer_model:
            self.session.pool.assign(doc.writer_model)
        return doc


# --------------------------------------------------------------------------
# One document
# --------------------------------------------------------------------------


def doc_fingerprint(doc: WrittenDoc, ctx: CompletenessContext) -> str:
    payload = {
        "pass_version": PASS_VERSION,
        "teacher": ctx.teacher_model,
        "system_prompt": hashlib.sha256(ctx.system_prompt.encode()).hexdigest(),
        "wp4": doc.input_fingerprint,
        "prepared": [hashlib.sha256(p.text.encode()).hexdigest() for p in doc.prepared_sources],
        "labels": [label.model_dump(mode="json") for label in doc.labels],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def teacher_key(doc: WrittenDoc, ctx: CompletenessContext) -> str:
    payload = {
        "teacher": ctx.teacher_model,
        "system_prompt": hashlib.sha256(ctx.system_prompt.encode()).hexdigest(),
        "prepared": [
            [p.source_id, hashlib.sha256(p.text.encode()).hexdigest()] for p in doc.prepared_sources
        ],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _decoy_ids(spans: list[ScriptedSpan]) -> list[str]:
    return sorted({s.marker_id for s in spans if s.kind == "decoy"})


def _reuse_attempt(
    ctx: CompletenessContext, old: PassAttempt, doc: WrittenDoc, plan: DocPlan
) -> PassAttempt:
    sources = source_inputs(doc, plan)
    facts = regate(ctx, old.facts, sources)
    spans = scripted_spans(doc, sources, ctx.world, ctx.decoys_by_id)
    triage(facts, spans, world=ctx.world, plan=plan, sources=sources, lexicon=ctx.lexicon)
    return PassAttempt(
        version="original",
        writer_model=doc.writer_model,
        scripted_labels=len(doc.labels),
        facts=facts,
        calls=[c.model_copy(update={"reused": True}) for c in old.calls],
        decoys_in_text=_decoy_ids(spans),
    )


def _run_attempt(
    ctx: CompletenessContext,
    doc: WrittenDoc,
    plan: DocPlan,
    version: Literal["original", "regenerated"],
    attempt_index: int,
) -> PassAttempt:
    sources = source_inputs(doc, plan)
    facts, calls = teacher_pass(
        ctx,
        sources,
        model=ctx.teacher_model,
        role="completeness",
        plan_id=plan.id,
        attempt=attempt_index,
    )
    spans = scripted_spans(doc, sources, ctx.world, ctx.decoys_by_id)
    triage(facts, spans, world=ctx.world, plan=plan, sources=sources, lexicon=ctx.lexicon)
    return PassAttempt(
        version=version,
        writer_model=doc.writer_model,
        scripted_labels=len(doc.labels),
        facts=facts,
        calls=calls,
        decoys_in_text=_decoy_ids(spans),
    )


def completed_labels(
    labels: list[WriterLabel], attempt: PassAttempt, world: World
) -> list[CompletedLabel]:
    """Each scripted label with the claim it ships with (`CompletedLabel`).
    Only facts that passed the gate lend their claim."""
    claims = {f.id: f.claim for f in world.facts}
    usable = attempt.usable()
    out: list[CompletedLabel] = []
    for label in labels:
        confirmed = [f for f in usable if f.triage == "confirmed" and f.matched_id == label.fact_id]
        agreeing = [f for f in confirmed if f.type_agrees]
        restated = [
            f
            for f in usable
            if f.triage == "restatement" and f.matched_id == label.fact_id and f.type_agrees
        ]
        base = label.model_dump(
            exclude={
                "claim",
                "claim_source",
                "scripted_claim",
                "claim_teacher_model",
                "teacher_type",
            }
        )
        scripted = claims.get(label.fact_id, "")
        if agreeing:
            best = max(agreeing, key=lambda f: f.iou or 0.0)
            extra = {"claim": best.claim, "claim_source": "teacher"}
        elif restated:
            best = max(restated, key=lambda f: f.similarity or 0.0)
            extra = {"claim": best.claim, "claim_source": "teacher_restatement"}
        else:
            best = None
            extra = {"claim": scripted, "claim_source": "scripted"}
        out.append(
            CompletedLabel(
                **base,
                **extra,
                scripted_claim=scripted,
                claim_teacher_model=best.teacher_model if best else None,
                teacher_type=confirmed[0].fact_type if confirmed and not agreeing else None,
                teacher_fields=teacher_fields(best) if best else None,
            )
        )
    return out


def _finish(
    state: DocCompleteness, doc: WrittenDoc, plan: DocPlan, decision: Decision, world: World
) -> DocCompleteness:
    attempt = state.attempts[-1]
    attempt.decision, attempt.reason = decision.action, decision.reason
    state.reason = decision.reason
    if decision.action == "drop":
        state.status = "dropped"
        state.scripted_labels, state.silver_labels = [], []
        return state
    sources = source_inputs(doc, plan)
    state.scripted_labels = completed_labels(list(doc.labels), attempt, world)
    state.silver_labels = silver_labels_for(plan, labelable(attempt), sources, world)
    state.status = {
        "accept": "accepted",
        "clean_negative": "clean_negative",
        "relabel_positive": "relabeled_positive",
    }[decision.action]
    return state


def scripted_count(doc: WrittenDoc, attempt: PassAttempt) -> int:
    """Scripted facts the document ships labeled: WP4's labels plus the
    planted facts this pass recovered (a document whose only facts WP4 left
    unmarked is a positive, not a negative)."""
    recovered = {f.matched_id for f in labelable(attempt) if f.triage == "planted"}
    return len(doc.labels) + len(recovered - {x.fact_id for x in doc.labels})


def process_document(
    ctx: CompletenessContext,
    doc: WrittenDoc,
    plan: DocPlan,
    *,
    prior: DocCompleteness | None = None,
    save: Callable[[DocCompleteness], None] = lambda _s: None,
) -> DocCompleteness:
    """Run (or resume) the pass on one written document. `save` is called
    after every paid step so a stop never pays twice for the same step."""
    fingerprint = doc_fingerprint(doc, ctx)
    key = teacher_key(doc, ctx)
    if prior is not None and prior.fingerprint == fingerprint:
        state = prior
    else:
        state = DocCompleteness(
            plan_id=plan.id, kind=plan.kind, fingerprint=fingerprint, teacher_key=key
        )
        if prior is not None and prior.teacher_key == key and prior.attempts:
            # A new pass version with the same teacher input: no new call.
            state.attempts = [_reuse_attempt(ctx, prior.attempts[0], doc, plan)]
            state.audit_seed = prior.audit or prior.audit_seed
            save(state)
    waiting = state.status == "needs_regeneration" and ctx.regenerator is not None
    if state.status in FINAL_STATUSES and not waiting:
        if state.status in SHIPPED_STATUSES and state.attempts:
            # Labels are rebuilt from the stored teacher facts (no call), so
            # a finished document always ships the current label shape.
            final = state.final_doc(doc)
            state.scripted_labels = completed_labels(
                list(final.labels), state.attempts[-1], ctx.world
            )
        return state
    state.status = "in_progress"
    state.doc_class = "positive" if doc.labels else "negative"

    if not state.attempts:
        state.attempts.append(_run_attempt(ctx, doc, plan, "original", 0))
        save(state)
    first = state.attempts[0]
    state.doc_class = "positive" if scripted_count(doc, first) else "negative"
    d0 = decide(scripted_count(doc, first), first.open_facts(), regenerated=False)
    if d0.action != "regenerate":
        return _finish(state, doc, plan, d0, ctx.world)
    first.decision, first.reason = d0.action, d0.reason

    if ctx.regenerator is None:
        state.status, state.reason = "needs_regeneration", d0.reason
        return state
    if not state.regeneration_tried:
        offending = sorted(
            {
                f.span_text or f.quote
                for f in first.of("candidate")
                if f.blockers and (f.span_text or f.quote)
            },
        )
        state.do_not_state = offending
        new = ctx.regenerator.regenerate(plan.id, offending, {doc.writer_model or ""} - {""})
        state.regenerated_doc = new
        state.regeneration_tried = True
        save(state)
    new = state.regenerated_doc
    if new is not None and new.status == "error":
        # No writer replied: retry on a later run rather than drop.
        state.regenerated_doc, state.regeneration_tried = None, False
        state.status, state.reason = "error", f"regeneration error: {new.reason}"
        return state
    if new is None or new.status != "written":
        state.status = "dropped"
        state.reason = f"{d0.reason}; regeneration " + (
            "produced nothing" if new is None else f"{new.status}: {new.reason}"
        )
        state.scripted_labels, state.silver_labels = [], []
        return state
    if len(state.attempts) < 2:
        state.attempts.append(_run_attempt(ctx, new, plan, "regenerated", 1))
        save(state)
    second = state.attempts[1]
    state.doc_class = "positive" if scripted_count(new, second) else "negative"
    d1 = decide(scripted_count(new, second), second.open_facts(), regenerated=True)
    return _finish(state, new, plan, d1, ctx.world)


# --------------------------------------------------------------------------
# Audit (spec section 3.5, last gate)
# --------------------------------------------------------------------------


def select_audit_docs(plan_ids: list[str], world_id: str, share: float = AUDIT_SHARE) -> set[str]:
    """A fixed-seed share of documents (at least one unless the share is
    zero), stable across runs."""
    if not plan_ids or share <= 0:
        return set()
    ranked = sorted(plan_ids, key=lambda p: hashlib.sha256(f"{world_id}:{p}".encode()).hexdigest())
    n = max(1, round(share * len(plan_ids)))
    return set(ranked[:n])


def audit_document(
    ctx: CompletenessContext, state: DocCompleteness, doc: WrittenDoc, plan: DocPlan
) -> AuditResult:
    """The second teacher on the shipped document. Measured first against
    the labels as they stand (`audit_status`, the section 3.5 audit gate);
    then triaged like the first teacher's facts, with every label as a
    known span: a consistent new fact, a planted fact nobody labeled, or a
    same-type restatement becomes a silver label under the audit teacher's
    name. A fact that breaks the world is not labeled (known residual)."""
    model = pick_audit_teacher(doc.writer_model, ctx.teacher_model)
    sources = source_inputs(doc, plan)
    seed = state.audit_seed
    if seed is not None and seed.model == model:
        facts = regate(ctx, seed.facts, sources)
        calls = [c.model_copy(update={"reused": True}) for c in seed.calls]
    else:
        facts, calls = teacher_pass(
            ctx, sources, model=model, role="audit", plan_id=plan.id, attempt=len(state.attempts)
        )
    labels: list[tuple[str, tuple[int, int], str, str | None]] = []
    world_facts = {f.id: f for f in ctx.world.facts}
    for label in state.scripted_labels:
        wf = world_facts.get(label.fact_id)
        labels.append(
            (
                label.doc_id,
                (label.char_start, label.char_end),
                label.fact_type.value,
                wf.claim if wf else None,
            )
        )
    for label in state.silver_labels:
        labels.append(
            (label.doc_id, (label.char_start, label.char_end), label.fact_type.value, label.claim)
        )
    script = scripted_spans(doc, sources, ctx.world, ctx.decoys_by_id)
    decoys = [x for x in script if x.kind == "decoy"]
    texts = {src.source_id: src.text for src in sources}
    label_spans = [
        ScriptedSpan("fact", str(i), d, sp[0], sp[1], True)
        for i, (d, sp, _t, _c) in enumerate(labels)
    ]
    for f in facts:
        if not f.grounded:
            continue
        if _best_overlap(f, label_spans, texts) is not None:
            f.audit_status = "known"
            continue
        if _touching(f, decoys, texts) is not None:
            f.audit_status = "decoy_hit"
            continue
        same_type = [c for _d, _sp, t, c in labels if t == f.fact_type and c]
        if any(_claim_similarity(f.claim, c) >= RESTATEMENT_SIMILARITY for c in same_type):
            f.audit_status = "known"
            continue
        f.audit_status = "new"

    known = [
        ScriptedSpan(
            "fact",
            x.fact_id,
            x.doc_id,
            x.char_start,
            x.char_end,
            True,
            fact_type=x.fact_type.value,
            claim=world_facts[x.fact_id].claim if x.fact_id in world_facts else x.claim,
        )
        for x in [*state.scripted_labels, *state.silver_labels]
    ]
    dropped = [x for x in script if x.kind == "fact" and not x.verified]
    triage(
        facts,
        [*known, *decoys, *dropped],
        world=ctx.world,
        plan=plan,
        sources=sources,
        lexicon=ctx.lexicon,
    )
    disagreements = [
        {
            "doc_id": f.doc_id,
            "span": list(f.span),
            "label_fact_id": f.matched_id,
            "label_type": next(x.fact_type for x in known if x.marker_id == f.matched_id),
            "audit_type": f.fact_type,
            "audit_claim": f.claim,
        }
        for f in facts
        if f.grounded and f.triage == "confirmed" and f.type_agrees is False and f.has_span
    ]
    # Orchestrator policy: an audit fact is labeled only where the labeling
    # teacher also found an overlapping span, and never on an off-topic or
    # hard-negative document; audit-only facts are logged and counted as
    # residual.
    kimi_spans = [
        ScriptedSpan("fact", f"k{i}", f.doc_id, f.char_start, f.char_end, True)
        for i, f in enumerate(state.attempts[-1].facts if state.attempts else [])
        if f.grounded and f.has_span and f.triage != "decoy_hit"
    ]
    may_label = plan.role not in (DocRole.OFF_TOPIC, DocRole.HARD_NEGATIVE)
    taken = [
        f
        for f in labelable(
            PassAttempt(version="original", writer_model=None, scripted_labels=0, facts=facts)
        )
        if not f.blockers
        and may_label
        and f.has_span
        and _best_overlap(f, kimi_spans, texts) is not None
    ]
    for f in taken:
        f.labeled = True
    added = silver_labels_for(plan, taken, sources, ctx.world, prefix="A")
    return AuditResult(
        model=model,
        facts=facts,
        calls=calls,
        labeled_facts=len(state.scripted_labels) + len(state.silver_labels),
        labels=added,
        type_disagreements=disagreements,
    )


def mark_found_by(state: DocCompleteness, doc: WrittenDoc) -> None:
    """`found_by` on every silver label: each teacher with a grounded fact
    (not a decoy hit) on its span."""
    texts = {p.source_id: p.text for p in doc.prepared_sources}
    teachers = [f for a in state.attempts[-1:] for f in a.facts] + (
        state.audit.facts if state.audit else []
    )
    for label in state.silver_labels:
        span = ScriptedSpan(
            "fact", label.fact_id, label.doc_id, label.char_start, label.char_end, True
        )
        label.found_by = sorted(
            {
                f.teacher_model
                for f in teachers
                if f.grounded
                and f.has_span
                and f.triage != "decoy_hit"
                and _best_overlap(f, [span], texts) is not None
            }
            | {label.teacher_model}
        )


# --------------------------------------------------------------------------
# Label judge (portolan.training.label_judge)
# --------------------------------------------------------------------------


def read_judge_cache(path: Path | None) -> dict[str, JudgeLogEntry]:
    if path is None or not path.exists():
        return {}
    entries = [
        JudgeLogEntry.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {e.key: e for e in entries if e.error is None}


def judge_document(
    ctx: CompletenessContext,
    state: DocCompleteness,
    doc: WrittenDoc,
    plan: DocPlan,
    cache: dict[str, JudgeLogEntry],
) -> DocJudge:
    """The label judge over one document's silver labels, one call per
    source (every label of that source together). Applied here: a dropped
    label moves to `DocJudge.dropped`, a retyped one changes only its type;
    claims and spans are never edited. A document whose only labels were
    silver and all dropped becomes a clean negative."""
    settings = ctx.label_judge
    assert settings is not None
    inputs = (
        list(state.judge.input_labels) if state.judge is not None else list(state.silver_labels)
    )
    sources = {s.source_id: s for s in source_inputs(doc, plan)}
    record = DocJudge(
        model=settings.model, prompt_version=JUDGE_PROMPT_VERSION, input_labels=inputs
    )
    groups: dict[str, list[int]] = {}
    for i, label in enumerate(inputs):
        groups.setdefault(label.doc_id, []).append(i)
    outcome: dict[int, JudgedLabel] = {}
    for doc_id, idxs in groups.items():
        labels = [inputs[i] for i in idxs]
        facts = [JudgeFact(x.fact_type.value, x.claim, x.text) for x in labels]
        source = sources.get(doc_id)
        result, how = judge_facts(
            source.text if source else "",
            facts,
            model=settings.model,
            call=settings.call if source else None,
            company_context=ctx.company_context,
            source_date=source.source_date.isoformat() if source else None,
            variant=ctx.prompt_variant,
            cache=cache,
            budget_left=settings.budget_left,
        )
        record.cached += how == "cached"
        record.calls += how == "called"
        record.cost_usd += result.cost_usd
        if result.error:
            record.errors.append(f"{doc_id}: {result.error}")
        if how == "called":
            entry = JudgeLogEntry.from_result(
                result, subject=f"{plan.id}/{doc_id}", facts=facts, created=ctx.now().isoformat()
            )
            if entry.error is None:
                cache[entry.key] = entry
            if settings.log_path is not None:
                settings.log_path.parent.mkdir(parents=True, exist_ok=True)
                with settings.log_path.open("a", encoding="utf-8") as fh:
                    fh.write(entry.model_dump_json() + "\n")
        for i, label, judgement in zip(idxs, labels, result.judgements, strict=True):
            verdict = "unresolved" if judgement is None else judgement.verdict
            after = label.fact_type.value
            if judgement is not None and judgement.verdict == "retype":
                if retype_allowed(judgement.fact_type, label.text):
                    after = judgement.fact_type
                else:
                    verdict = "unresolved"
            outcome[i] = JudgedLabel(
                doc_id=label.doc_id,
                char_start=label.char_start,
                char_end=label.char_end,
                teacher_model=label.teacher_model,
                fact_id=label.fact_id,
                verdict=verdict,
                type_before=label.fact_type.value,
                type_after=after,
                reason=judgement.reason if judgement is not None else (result.error or ""),
            )
    kept: list[SilverLabel] = []
    for i, label in enumerate(inputs):
        judged = outcome[i]
        record.labels.append(judged)
        if judged.verdict == "drop":
            record.dropped.append(label)
        elif judged.verdict == "retype":
            kept.append(
                label.model_copy(
                    update={
                        "fact_type": FactType(judged.type_after),
                        "teacher_fact_type": label.fact_type.value,
                    }
                )
            )
        else:
            kept.append(label)
    state.silver_labels = kept
    state.judge = record
    if not kept and not state.scripted_labels and state.status == "relabeled_positive":
        state.status = "clean_negative"
        state.reason = f"label judge dropped all {len(record.dropped)} silver label(s)"
    return record


# --------------------------------------------------------------------------
# QA report
# --------------------------------------------------------------------------


def _ratio(num: float, den: float) -> float | None:
    return round(num / den, 4) if den else None


def _gate_row(
    name: str, sense: str, threshold: float, value: float | None, note: str = ""
) -> dict[str, Any]:
    if value is None:
        passed = None
    elif sense == "min":
        passed = value >= threshold
    else:
        passed = value <= threshold
    result = "PENDING" if passed is None else ("PASS" if passed else "FAIL")
    row = {"gate": name, "sense": sense, "threshold": threshold, "value": value}
    row |= {"passed": passed, "result": result}
    if note:
        row["note"] = note
    return row


def _label_key(label: SilverLabel) -> tuple[str, str, int, int, str]:
    return (label.plan_id, label.doc_id, label.char_start, label.char_end, label.teacher_model)


def precision_sample(
    shipped: list[DocCompleteness], world_id: str, size: int = PRECISION_SAMPLE
) -> list[SilverLabel]:
    """Up to `size` silver labels per teacher, in a fixed-seed order: the
    by-eye precision sample (every labeling teacher, Kimi-only labels
    included)."""
    by_teacher: dict[str, list[SilverLabel]] = {}
    for s in shipped:
        for label in s.silver_labels:
            by_teacher.setdefault(label.teacher_model, []).append(label)
    out: list[SilverLabel] = []
    for teacher in sorted(by_teacher):
        labels = sorted(
            by_teacher[teacher],
            key=lambda x: hashlib.sha256(f"{world_id}:{_label_key(x)}".encode()).hexdigest(),
        )
        out += labels[:size]
    return out


def precision_by_teacher(
    shipped: list[DocCompleteness], reviews: list[dict[str, Any]], world_id: str
) -> dict[str, dict[str, Any]]:
    """Per teacher: labels, the sample, how many were read and judged
    correct, and the precision once the sample is fully read."""
    verdicts = {
        (r["plan_id"], r["doc_id"], r["char_start"], r["char_end"], r["teacher_model"]): bool(
            r["correct"]
        )
        for r in reviews
    }
    counts: dict[str, Counter] = {}
    for s in shipped:
        for label in s.silver_labels:
            counts.setdefault(label.teacher_model, Counter())["labels"] += 1
    for label in precision_sample(shipped, world_id):
        c = counts[label.teacher_model]
        c["sample"] += 1
        verdict = verdicts.get(_label_key(label))
        if verdict is not None:
            c["read"] += 1
            c["correct"] += verdict
    return {
        teacher: {
            "labels": c["labels"],
            "sample": c["sample"],
            "read": c["read"],
            "correct": c["correct"],
            "precision": round(c["correct"] / c["read"], 4)
            if c["read"] and c["read"] >= c["sample"]
            else None,
        }
        for teacher, c in sorted(counts.items())
    }


_FACT_TRIAGE = frozenset({"confirmed", "restatement", "planted", "candidate"})


@dataclass
class CaptureCounts:
    """Distinct facts in the shipped text, grouped by span, and who found
    each: the labeling teacher (K), the audit teacher (A), the script (S)."""

    kimi: int = 0  # n1
    audit: int = 0  # n2
    both: int = 0  # m
    scripted: int = 0
    scripted_kimi: int = 0
    scripted_audit: int = 0
    scripted_neither: int = 0
    unscripted_found: int = 0

    def add(self, other: CaptureCounts) -> None:
        for name in self.__dataclass_fields__:
            setattr(self, name, getattr(self, name) + getattr(other, name))


def capture_counts(state: DocCompleteness, doc: WrittenDoc) -> CaptureCounts:
    """Union the two teachers' grounded facts (decoy hits excluded) and the
    scripted labels into clusters of overlapping spans, per source."""
    texts = {p.source_id: p.text for p in doc.prepared_sources}
    items: list[tuple[str, str, tuple[int, int] | None]] = []
    for f in state.attempts[-1].facts if state.attempts else []:
        if f.grounded and f.triage in _FACT_TRIAGE:
            items.append(("K", f.doc_id, f.span if f.has_span else None))
    for f in state.audit.facts if state.audit else []:
        if f.grounded and f.triage in _FACT_TRIAGE:
            items.append(("A", f.doc_id, f.span if f.has_span else None))
    for x in state.scripted_labels:
        items.append(("S", x.doc_id, (x.char_start, x.char_end)))
    parent = list(range(len(items)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, (_wi, di, si) in enumerate(items):
        for j in range(i):
            _wj, dj, sj = items[j]
            if si is None or sj is None or di != dj:
                continue
            text = texts.get(di, "")
            if spans_match(core_span(text, si), core_span(text, sj)):
                parent[find(i)] = find(j)
    groups: dict[int, set[str]] = {}
    for i, (who, _d, _s) in enumerate(items):
        groups.setdefault(find(i), set()).add(who)
    c = CaptureCounts()
    for who in groups.values():
        k, a, sc = "K" in who, "A" in who, "S" in who
        c.kimi += k
        c.audit += a
        c.both += k and a
        if sc:
            c.scripted += 1
            c.scripted_kimi += k
            c.scripted_audit += a
            c.scripted_neither += not (k or a)
        elif k or a:
            c.unscripted_found += 1
    return c


def unseen_estimates(c: CaptureCounts) -> dict[str, float | None]:
    """Facts neither teacher found and the script did not label, two ways.

    - `chapman`: the capture-recapture estimate of the whole population
      from the two teachers' overlap, (n1 + 1)(n2 + 1) / (m + 1) - 1, minus
      what either found, minus scripted facts both missed (labeled anyway).
    - `recall`: each teacher's recall on the scripted facts (a marked set of
      known size), assumed to hold for the rest: the found unscripted facts
      times P(both miss) / (1 - P(both miss)).

    Both assume the teachers miss independently; they read the same prompt,
    so misses are correlated and both estimates run low."""
    out: dict[str, float | None] = {"chapman": None, "recall": None}
    if c.kimi or c.audit:
        population = (c.kimi + 1) * (c.audit + 1) / (c.both + 1) - 1
        out["chapman"] = max(0.0, population - (c.kimi + c.audit - c.both) - c.scripted_neither)
    if c.scripted:
        miss = (1 - c.scripted_kimi / c.scripted) * (1 - c.scripted_audit / c.scripted)
        out["recall"] = c.unscripted_found * miss / (1 - miss) if miss < 1 else None
    return out


def _teacher_decoy_errors(finals: list[DocCompleteness]) -> dict[str, Any]:
    """Kept teacher facts on a planted decoy span (any overlap): the decoy
    rule dropped each one, so none became a label."""
    rows = []
    for s in finals:
        facts = [*(s.attempts[-1].facts if s.attempts else []), *(s.audit.facts if s.audit else [])]
        for f in facts:
            if f.usable and f.triage == "decoy_hit":
                rows.append(
                    {
                        "plan_id": s.plan_id,
                        "doc_id": f.doc_id,
                        "span": list(f.span) if f.has_span else None,
                        "decoy_id": f.matched_id,
                        "decoy_class_id": f.decoy_class_id,
                        "decoy_verified": f.decoy_verified,
                        "teacher_model": f.teacher_model,
                        "role": f.role,
                        "fact_type": f.fact_type,
                        "claim": f.claim,
                    }
                )
    return {
        "count": len(rows),
        "by_class": dict(Counter(r["decoy_class_id"] or "unknown" for r in rows)),
        "facts": rows,
    }


def _label_judge_summary(shipped: list[DocCompleteness]) -> dict[str, Any] | None:
    judged = [(s, s.judge) for s in shipped if s.judge is not None]
    if not judged:
        return None
    verdicts = Counter(x.verdict for _s, j in judged for x in j.labels)
    return {
        "models": dict(Counter(j.model for _s, j in judged)),
        "prompt_versions": dict(Counter(j.prompt_version for _s, j in judged)),
        "documents": len(judged),
        "documents_not_fully_judged": sum(1 for _s, j in judged if j.unresolved),
        "labels_in": sum(len(j.input_labels) for _s, j in judged),
        "verdicts": dict(verdicts),
        "calls": sum(j.calls for _s, j in judged),
        "cached": sum(j.cached for _s, j in judged),
        "cost_usd": round(sum(j.cost_usd for _s, j in judged), 6),
        "dropped": [
            {
                "plan_id": s.plan_id,
                "doc_id": x.doc_id,
                "span": [x.char_start, x.char_end],
                "teacher_model": x.teacher_model,
                "fact_type": x.type_before,
                "reason": x.reason,
            }
            for s, j in judged
            for x in j.labels
            if x.verdict == "drop"
        ],
        "retyped": [
            {
                "plan_id": s.plan_id,
                "doc_id": x.doc_id,
                "span": [x.char_start, x.char_end],
                "from": x.type_before,
                "to": x.type_after,
                "reason": x.reason,
            }
            for s, j in judged
            for x in j.labels
            if x.verdict == "retype"
        ],
    }


def build_qa_report(
    states: dict[str, DocCompleteness],
    wp4_docs: dict[str, WrittenDoc],
    *,
    decoy_classes: dict[str, str],
    world_id: str,
    teacher_model: str,
    audited: set[str],
    run_spend_usd: float,
    stop_reason: str | None,
    skipped: dict[str, str],
    now: str,
    reviews: list[dict[str, Any]] | None = None,
    world_documents: int | None = None,
) -> dict[str, Any]:
    finals = [s for s in states.values() if s.status in FINAL_STATUSES]
    shipped = [s for s in finals if s.status in SHIPPED_STATUSES]

    # Gate 1: the writer's verification rate, on the final version of every
    # document the writer handled (WP4-dropped documents included).
    planted = verified = 0
    wp4_regen = 0
    for plan_id, original in wp4_docs.items():
        state = states.get(plan_id)
        doc = state.final_doc(original) if state is not None else original
        c = doc.counts()
        planted += c["facts_planted"]
        verified += c["facts_verified"]
        if any(call.call_kind == "regenerate" for call in original.calls):
            wp4_regen += 1
    wp5_regen = sum(1 for s in states.values() if s.regeneration_tried)
    regenerated_docs = {
        p for p, d in wp4_docs.items() if any(c.call_kind == "regenerate" for c in d.calls)
    } | {p for p, s in states.items() if s.regeneration_tried}

    # Teacher recall on scripted facts: last attempt of every finished document.
    scripted_seen = scripted_hit = scripted_found = confirmed = type_agrees = 0
    triage_counts: Counter = Counter()
    gate_counts: Counter = Counter()
    decoy_class_seen: Counter = Counter()
    decoy_class_hit: Counter = Counter()
    candidates_by_kind: dict[str, Counter] = {}
    for s in finals:
        if not s.attempts:
            continue
        last = s.attempts[-1]
        scripted_seen += last.scripted_labels
        usable = last.usable()
        hit_ids = {f.matched_id for f in usable if f.triage == "confirmed"}
        scripted_hit += len(hit_ids)
        scripted_found += len(hit_ids | {f.matched_id for f in usable if f.triage == "restatement"})
        confirmed += sum(1 for f in usable if f.triage == "confirmed")
        type_agrees += sum(1 for f in usable if f.triage == "confirmed" and f.type_agrees)
        gate_counts.update(f.gate if not f.duplicate else "duplicate" for f in last.facts)
        triage_counts.update(f.triage for f in usable if f.triage)
        for attempt in s.attempts:
            bucket = candidates_by_kind.setdefault(s.kind.value, Counter())
            for f in attempt.of("candidate"):
                bucket["candidates"] += 1
                bucket[f"in_{attempt.version}"] += 1
                bucket["contradiction"] += bool(f.contradictions)
                bucket["unlabelable"] += f.unlabelable
                bucket["unknown_entity"] += bool(f.unknown_entities)
                bucket["unscripted_number"] += bool(f.unscripted_numbers)
                bucket["recovers_dropped_fact"] += bool(f.recovers_fact_id)
                if attempt is s.attempts[-1] and s.status in SHIPPED_STATUSES:
                    bucket["labeled_silver"] += 1
                elif attempt is not s.attempts[-1]:
                    bucket["removed_by_regeneration"] += 1
                else:
                    bucket["in_dropped_or_unfinished_doc"] += 1
    # Decoy hit rates: every teacher pass is an exposure, so each attempt
    # counts against the version of the document it read.
    for s in finals:
        original = wp4_docs.get(s.plan_id)
        if original is None:
            continue
        for attempt in s.attempts:
            doc = original if attempt.version == "original" else s.final_doc(original)
            hits = {f.matched_id for f in attempt.of("decoy_hit")}
            in_text = (
                {d.decoy_id for d in doc.decoy_spans}
                | {
                    o.marker_id
                    for o in doc.slot_outcomes
                    if o.slot_kind == "decoy" and o.status == "dropped" and o.dropped_text
                }
                | set(attempt.decoys_in_text)
                | hits
            )
            for decoy_id in in_text:
                cls = decoy_classes.get(decoy_id, "unknown")
                decoy_class_seen[cls] += 1
                decoy_class_hit[cls] += decoy_id in hits

    r = scripted_hit / scripted_seen if scripted_seen else None
    scripted_labeled = sum(len(s.scripted_labels) for s in shipped)
    silver = sum(len(s.silver_labels) for s in shipped)
    silver_new = sum(1 for s in shipped for label in s.silver_labels if label.new_fact)
    silver_recovered = sum(
        1 for s in shipped for label in s.silver_labels if label.recovers_fact_id
    )
    labeled = scripted_labeled + silver
    recovery: Counter = Counter()
    for s in finals:
        for attempt in s.attempts:
            for f in attempt.of("planted"):
                recovery["facts"] += 1
                recovery["by_" + (f.tier or "").rsplit("|", 1)[-1]] += 1
                recovery["unlabelable"] += f.unlabelable
                recovery["type_disagrees"] += f.type_agrees is False
            recovery["candidates"] += len(attempt.of("candidate"))
    claim_sources: Counter = Counter()
    for s in shipped:
        for label in s.scripted_labels:
            key = label.claim_source
            if key == "scripted":
                key = "scripted_type_disagrees" if label.teacher_type else "scripted_teacher_missed"
            claim_sources[key] += 1
    positive_shipped = [s for s in shipped if s.scripted_labels]
    complete_docs = sum(
        1 for s in positive_shipped if all(x.claim_source != "scripted" for x in s.scripted_labels)
    )
    u_total = sum(len(s.attempts[-1].open_facts()) for s in shipped if s.attempts)
    flags: dict[str, Counter] = {}
    for s in finals:
        for attempt in s.attempts:
            for f in attempt.facts:
                if f.grounded and f.triage:
                    bucket = flags.setdefault(f.triage, Counter())
                    bucket["facts"] += 1
                    bucket["unknown_entity"] += bool(f.unknown_entities)
                    bucket["unscripted_number"] += bool(f.unscripted_numbers)
    # Found but unlabeled facts ship as-is: they are residual for certain,
    # unless the audit teacher's claim labeled the same span.
    known_unlabeled = 0
    for s in shipped:
        original = wp4_docs.get(s.plan_id)
        if not s.attempts or original is None:
            continue
        texts = {p.source_id: p.text for p in s.final_doc(original).prepared_sources}
        audit_spans = [
            ScriptedSpan("fact", x.fact_id, x.doc_id, x.char_start, x.char_end, True)
            for x in (s.audit.labels if s.audit else [])
        ]
        known_unlabeled += sum(
            1
            for f in s.attempts[-1].open_facts()
            if f.unlabelable and not (f.has_span and _best_overlap(f, audit_spans, texts))
        )
    kimi_only = u_total * (1 - r) / r if r else None
    audits = [s.audit for p, s in states.items() if p in audited and s.audit is not None]
    audit_new = sum(len(a.new()) for a in audits)
    audit_labeled = sum(a.labeled_facts for a in audits)
    # Two teachers: the residual is what neither found (estimated) plus what
    # one found that could not be labeled (known). Documents without an
    # audit fall back to the single-teacher term.
    capture = CaptureCounts()
    single_u = 0
    audit_only = 0
    for s in shipped:
        original = wp4_docs.get(s.plan_id)
        if not s.attempts or original is None:
            continue
        if s.audit is not None:
            capture.add(capture_counts(s, s.final_doc(original)))
            audit_unlabeled = [
                f
                for f in s.audit.facts
                if f.grounded and f.triage in ("candidate", "planted") and not f.labeled
            ]
            known_unlabeled += len(audit_unlabeled)
            audit_only += len(audit_unlabeled)
        else:
            single_u += len(s.attempts[-1].open_facts())
    unseen = unseen_estimates(capture)
    estimates = [v for v in unseen.values() if v is not None]
    unseen_by_both = max(estimates) if estimates else None
    single = single_u * (1 - r) / r if r else 0.0
    if audits:
        residual = known_unlabeled + unseen_by_both + single if unseen_by_both is not None else None
    else:
        residual = known_unlabeled + kimi_only if kimi_only is not None else None

    precision = precision_by_teacher(shipped, reviews or [], world_id)
    values = [t["precision"] for t in precision.values()]
    min_precision = min(values) if values and None not in values else None
    wrong = (
        sum(t["labels"] * (1 - t["precision"]) for t in precision.values())
        if None not in values
        else None
    )
    total_error = residual + wrong if residual is not None and wrong is not None else None
    whole_world = world_documents is not None and len(wp4_docs) >= world_documents
    scope = "whole world" if whole_world else f"{len(wp4_docs)} documents, not a whole world"
    gates = [
        _gate_row(*GATES[0], _ratio(verified, planted)),
        _gate_row(*GATES[1], _ratio(silver_new, labeled)),
        _gate_row(*GATES[2], _ratio(len(regenerated_docs), len(wp4_docs))),
        _gate_row(*GATES[3], round(r, 4) if r is not None else None),
        _gate_row(*GATES[4], _ratio(residual, labeled) if residual is not None else None, scope),
        _gate_row(
            *GATES[5],
            round(min_precision, 4) if min_precision is not None else None,
            "by-eye sample incomplete" if min_precision is None and precision else "",
        ),
        _gate_row(
            *GATES[6],
            _ratio(total_error, labeled) if total_error is not None else None,
            "scripted labels counted as correct; " + scope,
        ),
        _gate_row(*GATES[7], _ratio(audit_new, audit_labeled) if audits else None),
    ]
    teacher_cost = sum(c.cost_usd for s in states.values() for a in s.attempts for c in a.calls)
    audit_cost = sum(c.cost_usd for s in states.values() if s.audit for c in s.audit.calls)
    writer_cost = sum(s.regenerated_doc.cost_usd for s in states.values() if s.regenerated_doc)
    reused_cost = sum(
        c.cost_usd for s in states.values() for a in s.attempts for c in a.calls if c.reused
    )
    processed = sum(1 for s in states.values() if s.attempts)
    return {
        "generated_at": now,
        "pass_version": PASS_VERSION,
        "world_id": world_id,
        "teacher_model": teacher_model,
        "prompt_variant": PROMPT_VARIANT,
        "corpus_key": CORPUS_KEY,
        "stop_reason": stop_reason,
        "documents": {
            "wp4_documents": len(wp4_docs),
            "by_status": dict(Counter(s.status for s in states.values())),
            "skipped": skipped,
            "regenerated_by_wp4": wp4_regen,
            "regenerated_by_wp5": wp5_regen,
            "wp5_regeneration_rate": _ratio(wp5_regen, len(wp4_docs)),
        },
        "spend": {
            "run_usd": round(run_spend_usd, 6),
            "teacher_usd": round(teacher_cost, 6),
            "teacher_reused_usd": round(reused_cost, 6),
            "audit_usd": round(audit_cost, 6),
            "regeneration_writer_usd": round(writer_cost, 6),
            "teacher_calls": sum(len(a.calls) for s in states.values() for a in s.attempts),
            "audit_calls": sum(len(s.audit.calls) for s in states.values() if s.audit),
            # Every call a document took, reused or new, as a fresh run would pay.
            "cost_per_document_usd": _ratio(teacher_cost + audit_cost + writer_cost, processed),
        },
        "scripted": {
            "planted": planted,
            "verified": verified,
            "labeled_in_shipped_docs": scripted_labeled,
        },
        "teacher": {
            "gate": dict(gate_counts),
            "triage": dict(triage_counts),
            "scripted_seen": scripted_seen,
            "scripted_confirmed": scripted_hit,
            "recall_r": round(r, 4) if r is not None else None,
            # Not a gate: a scripted fact the teacher found only at another
            # mention (a restatement) also counts here.
            "recall_found_anywhere": _ratio(scripted_found, scripted_seen),
            "recall_gap_vs_halvard_points": round((HALVARD_RECALL - r) * 100, 1)
            if r is not None
            else None,
            "type_agreement": _ratio(type_agrees, confirmed),
        },
        "silver": {
            "labels": silver,
            "new_facts": silver_new,
            "recovered_planted": silver_recovered,
            "restatements": silver - silver_new - silver_recovered,
            "note": "the silver-share gate counts new facts only",
            "by_teacher": dict(
                Counter(label.teacher_model for s in shipped for label in s.silver_labels)
            ),
            "found_by_both": sum(
                1 for s in shipped for label in s.silver_labels if len(label.found_by) > 1
            ),
        },
        "planted_recovery": dict(recovery),
        "scripted_label_claims": {
            **{
                k: claim_sources[k]
                for k in (
                    "teacher",
                    "teacher_restatement",
                    "scripted_teacher_missed",
                    "scripted_type_disagrees",
                )
            },
            "documents_with_scripted_labels": len(positive_shipped),
            "documents_claims_complete": complete_docs,
        },
        "candidates_by_kind": {k: dict(v) for k, v in sorted(candidates_by_kind.items())},
        "flags_by_triage": {k: dict(v) for k, v in sorted(flags.items())},
        "decoys": {
            cls: {
                "in_text": decoy_class_seen[cls],
                "hit": decoy_class_hit[cls],
                "rate": _ratio(decoy_class_hit[cls], decoy_class_seen[cls]),
                "hand_review": (decoy_class_hit[cls] / decoy_class_seen[cls]) > DECOY_REVIEW_RATE,
            }
            for cls in sorted(decoy_class_seen)
        },
        "teacher_decoy_errors": _teacher_decoy_errors(finals),
        "label_judge": _label_judge_summary(shipped),
        "residual": {
            "known_unlabeled": known_unlabeled,
            "audit_only_unlabeled": audit_only,
            "capture": capture.__dict__,
            "unseen_by_both": {
                k: round(v, 3) if v is not None else None for k, v in unseen.items()
            },
            "unseen_used": round(unseen_by_both, 3) if unseen_by_both is not None else None,
            "estimated_unlabeled": round(residual, 3) if residual is not None else None,
            "labeled_facts": labeled,
            "share": _ratio(residual, labeled) if residual is not None else None,
            "kimi_only": {
                "u_total": u_total,
                "r": round(r, 4) if r is not None else None,
                "estimated_missed": round(kimi_only, 3) if kimi_only is not None else None,
            },
            "note": "known unlabeled + facts neither teacher found (the larger of the "
            "capture-recapture and recall estimates) over shipped documents",
        },
        "audit": {
            "documents": sorted(p for p in audited if states.get(p) and states[p].audit),
            "models": dict(Counter(a.model for a in audits)),
            "facts_kept": sum(sum(1 for f in a.facts if f.usable) for a in audits),
            "facts_grounded": sum(sum(1 for f in a.facts if f.grounded) for a in audits),
            "new_facts": audit_new,
            "labeled_facts": audit_labeled,
            "labels_added": sum(len(a.labels) for a in audits),
            "labels_added_new_facts": sum(1 for a in audits for x in a.labels if x.new_fact),
            "not_labeled_breaks_world": sum(
                1 for a in audits for f in a.facts if f.grounded and f.blockers
            ),
            "type_disagreements": sum(len(a.type_disagreements) for a in audits),
        },
        "precision": {
            "by_teacher": precision,
            "estimated_wrong_labels": round(wrong, 3) if wrong is not None else None,
            "total_label_error": round(total_error, 3) if total_error is not None else None,
            "note": "by-eye sample in precision_sample.jsonl; verdicts in precision_review.jsonl",
        },
        "gates": gates,
        "accepted": all(g["passed"] is True for g in gates),
        "per_document": [states[p].summary() for p in sorted(states)],
    }


# --------------------------------------------------------------------------
# run_completeness
# --------------------------------------------------------------------------


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def load_written_docs(in_dir: Path) -> dict[str, WrittenDoc]:
    out: dict[str, WrittenDoc] = {}
    for path in sorted((in_dir / "docs").glob("*.json")):
        doc = WrittenDoc.model_validate_json(path.read_text(encoding="utf-8"))
        out[doc.plan_id] = doc
    return out


def _load_state(path: Path) -> DocCompleteness | None:
    if not path.exists():
        return None
    try:
        return DocCompleteness.model_validate_json(path.read_text(encoding="utf-8"))
    except ValueError:
        return None


def run_completeness(
    ctx: CompletenessContext,
    wp4_docs: dict[str, WrittenDoc],
    *,
    out_dir: Path,
    doc_ids: list[str] | None = None,
    audit_share: float = AUDIT_SHARE,
    fingerprint_ok: Callable[[WrittenDoc, DocPlan], bool] | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """The pass over one world's WP4 output. Writes `docs/<plan_id>.json`
    checkpoints, `labels.jsonl` (scripted plus silver, shipped documents
    only), `silver_labels.jsonl`, `decoys.jsonl`, `teacher_facts.jsonl`,
    the shipped documents' files, and `qa_report.json`."""
    from portolan.training.pipeline import assert_not_holdout

    assert_not_holdout(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plans = ctx.world_plan.doc_by_id()
    wanted = set(doc_ids) if doc_ids is not None else None
    selected = {p: d for p, d in wp4_docs.items() if wanted is None or p in wanted}
    states: dict[str, DocCompleteness] = {}
    skipped: dict[str, str] = {}
    stop_reason: str | None = None

    def save(state: DocCompleteness) -> None:
        _atomic_write(out_dir / "docs" / f"{state.plan_id}.json", state.model_dump_json(indent=2))

    def order(d: WrittenDoc) -> tuple[date, str]:
        plan = plans.get(d.plan_id)
        return (plan.date if plan else date.max, d.plan_id)

    for doc in sorted(selected.values(), key=order):
        plan = plans.get(doc.plan_id)
        if plan is None:
            skipped[doc.plan_id] = "not in the world plan"
            continue
        if doc.status != "written":
            skipped[doc.plan_id] = f"wp4 status {doc.status}"
            continue
        if fingerprint_ok is not None and not fingerprint_ok(doc, plan):
            skipped[doc.plan_id] = "stale_input: WP4 fingerprint does not match this world"
            continue
        checkpoint = out_dir / "docs" / f"{doc.plan_id}.json"
        try:
            state = process_document(ctx, doc, plan, prior=_load_state(checkpoint), save=save)
        except WriterBudgetExceeded as exc:
            stop_reason = str(exc)
            partial = _load_state(checkpoint)
            if partial is not None:
                states[doc.plan_id] = partial
            break
        except (TeacherCallError, PreparedTextDrift) as exc:
            state = _load_state(checkpoint) or DocCompleteness(
                plan_id=doc.plan_id, kind=plan.kind, fingerprint=doc_fingerprint(doc, ctx)
            )
            state.status, state.reason = "error", str(exc)[:500]
        save(state)
        states[doc.plan_id] = state
        if progress is not None:
            s = state.summary()
            progress(
                f"{doc.plan_id} {state.status} scripted={s['scripted_labels']} "
                f"kept={s['teacher_facts_kept']} confirmed={s['scripted_confirmed']} "
                f"candidates={s['candidates']} silver={s['silver_labels']} "
                f"run=${ctx.budget.spent_usd:.4f}"
            )

    audited: set[str] = set()
    if stop_reason is None:
        shipped_ids = sorted(p for p, s in states.items() if s.status in SHIPPED_STATUSES)
        audited = select_audit_docs(shipped_ids, ctx.world.id, audit_share)
        for plan_id in sorted(audited):
            state = states[plan_id]
            if state.audit is not None:
                continue
            doc = state.final_doc(wp4_docs[plan_id])
            try:
                state.audit = audit_document(ctx, state, doc, plans[plan_id])
                state.silver_labels = [*state.silver_labels, *state.audit.labels]
                mark_found_by(state, doc)
                if state.status == "clean_negative" and state.audit.labels:
                    state.status = "relabeled_positive"
                    state.reason = f"audit labeled {len(state.audit.labels)} fact(s)"
            except WriterBudgetExceeded as exc:
                stop_reason = str(exc)
                break
            except TeacherCallError as exc:
                if progress is not None:
                    progress(f"{plan_id} audit failed: {exc}")
                continue
            save(state)
            if progress is not None:
                progress(
                    f"{plan_id} audit {state.audit.model}: new={len(state.audit.new())} "
                    f"run=${ctx.budget.spent_usd:.4f}"
                )

    if stop_reason is None and ctx.label_judge is not None:
        settings = ctx.label_judge
        cache = read_judge_cache(settings.log_path)
        for plan_id in sorted(states):
            state = states[plan_id]
            if state.status not in SHIPPED_STATUSES:
                continue
            done = state.judge
            if (
                done is not None
                and done.model == settings.model
                and done.prompt_version == JUDGE_PROMPT_VERSION
                and done.unresolved == 0
            ):
                continue
            if done is None and not state.silver_labels:
                continue
            judged = judge_document(
                ctx, state, state.final_doc(wp4_docs[plan_id]), plans[plan_id], cache
            )
            save(state)
            if progress is not None:
                verdicts = Counter(x.verdict for x in judged.labels)
                progress(f"{plan_id} label judge {judged.model}: {dict(verdicts)}")

    shipped = [s for s in states.values() if s.status in SHIPPED_STATUSES]
    labels = [
        label.model_dump_json()
        for s in sorted(shipped, key=lambda s: s.plan_id)
        for label in [*s.scripted_labels, *s.silver_labels]
    ]
    _atomic_write(out_dir / "labels.jsonl", "".join(line + "\n" for line in labels))
    _atomic_write(
        out_dir / "silver_labels.jsonl",
        "".join(
            label.model_dump_json() + "\n"
            for s in sorted(shipped, key=lambda s: s.plan_id)
            for label in s.silver_labels
        ),
    )
    decoy_lines = []
    teacher_lines = []
    for s in sorted(states.values(), key=lambda s: s.plan_id):
        original = wp4_docs.get(s.plan_id)
        if original is not None and s.status in SHIPPED_STATUSES:
            final = s.final_doc(original)
            decoy_lines += [d.model_dump_json() for d in final.decoy_spans]
            for rel, content in final.files.items():
                _atomic_write(out_dir / "files" / rel, content)
        for i, attempt in enumerate(s.attempts):
            for f in attempt.facts:
                row = {"plan_id": s.plan_id, "attempt": i, "version": attempt.version}
                teacher_lines.append(json.dumps({**row, **f.model_dump(mode="json")}))
        if s.audit is not None:
            for f in s.audit.facts:
                row = {"plan_id": s.plan_id, "attempt": "audit", "version": "final"}
                teacher_lines.append(json.dumps({**row, **f.model_dump(mode="json")}))
    _atomic_write(out_dir / "decoys.jsonl", "".join(line + "\n" for line in decoy_lines))
    _atomic_write(out_dir / "teacher_facts.jsonl", "".join(line + "\n" for line in teacher_lines))
    in_scope = {
        p: d
        for p, d in selected.items()
        if p in plans and not skipped.get(p, "").startswith("stale_input")
    }
    review_path = out_dir / "precision_review.jsonl"
    reviews = (
        [json.loads(line) for line in review_path.read_text().splitlines() if line.strip()]
        if review_path.exists()
        else []
    )
    _atomic_write(
        out_dir / "precision_sample.jsonl",
        "".join(
            x.model_dump_json() + "\n"
            for x in precision_sample(
                [s for s in states.values() if s.status in SHIPPED_STATUSES], ctx.world.id
            )
        ),
    )
    from corpus.generator.v2.writer import WriterConfig, writer_doc_plans

    report = build_qa_report(
        states,
        in_scope,
        reviews=reviews,
        world_documents=len(writer_doc_plans(ctx.world_plan, WriterConfig())),
        decoy_classes={d.id: d.decoy_class_id for d in ctx.world_plan.decoys},
        world_id=ctx.world.id,
        teacher_model=ctx.teacher_model,
        audited=audited,
        run_spend_usd=ctx.budget.spent_usd,
        stop_reason=stop_reason,
        skipped=skipped,
        now=ctx.now().isoformat(),
    )
    _atomic_write(out_dir / "qa_report.json", json.dumps(report, indent=2))
    return report


# --------------------------------------------------------------------------
# CLI (`python -m corpus.generator.v2.completeness run ...`)
# --------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]


def openrouter_teacher_factory(api_key: str, meter: Any) -> Callable[[str], Backend]:
    """The teacher and audit models through one `OpenRouterClient` that
    meters every call under `synthetic_v2` (the context then sets
    `client_meters=True`)."""
    import httpx

    from portolan.extract.backends import LLMClientBackend
    from portolan.llm.openrouter import OpenRouterClient

    client = OpenRouterClient(
        spend_key=CORPUS_KEY,
        spend_meter=meter,
        api_key=api_key,
        default_model=TEACHER_MODEL,
        sink=InMemorySink(),
        tenant="corpus-generator-v2-wp5",
        timeout=httpx.Timeout(30.0, read=180.0),
        max_retries=1,
    )
    for model in (TEACHER_MODEL, *AUDIT_TEACHERS):
        try:
            client.warm_json_strategy(model)
        except Exception as exc:  # noqa: BLE001 -- warming is best-effort
            print(f"warm_json_strategy({model}) failed, continuing: {type(exc).__name__}")
    return lambda model: LLMClientBackend(client, model=model, surface="extract")


def main(argv: list[str] | None = None) -> int:
    import os

    from corpus.generator.prose import _env_or_dotenv
    from corpus.generator.v2.models import Split
    from corpus.generator.v2.packs.schema import load_pack
    from corpus.generator.v2.planner import TYPE_CAP_SHARE, plan_world
    from corpus.generator.v2.renderers import render
    from corpus.generator.v2.simulator import simulate_world
    from corpus.generator.v2.writer import (
        DEFAULT_ROTATION,
        OpenRouterWriterBackend,
        WriterConfig,
        WriterPool,
        WriterSession,
        input_fingerprint,
        writer_doc_plans,
    )
    from portolan.training.pipeline import assert_not_holdout
    from portolan.training.spend import SpendMeter, default_spend_path

    parser = argparse.ArgumentParser(prog="python -m corpus.generator.v2.completeness")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run the completeness pass over one world's WP4 output")
    run.add_argument("--in", dest="in_dir", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--pack", default="medical_devices")
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--months", type=int, default=8)
    run.add_argument("--split", default="train")
    run.add_argument("--doc-ids", default="")
    run.add_argument("--budget-usd", type=float, default=1.00)
    run.add_argument("--audit-share", type=float, default=AUDIT_SHARE)
    run.add_argument("--no-regenerate", action="store_true")
    run.add_argument("--company-context", choices=("world", "none"), default="world")
    run.add_argument("--spend-path", type=Path, default=default_spend_path(REPO_ROOT / "var"))
    run.add_argument("--env-root", type=Path, default=REPO_ROOT)
    run.add_argument(
        "--skip-fingerprint-check",
        action="store_true",
        help="accept documents WP4 wrote under an earlier prompt version of the same world",
    )
    run.add_argument(
        "--no-type-cap",
        action="store_true",
        help="plan the world without the plan-time per-type cap, as documents written "
        "before 2026-09-23 were planned",
    )
    run.add_argument(
        "--label-judge",
        default=None,
        metavar="MODEL",
        help="judge every shipped document's silver labels with this open-weight model "
        "(portolan.training.label_judge, e.g. deepseek/deepseek-v4.1-flash); off by default",
    )
    run.add_argument(
        "--judge-budget-usd",
        type=float,
        default=1.00,
        help="cap on this run's growth of the label_judge spend key",
    )
    args = parser.parse_args(argv)

    assert_not_holdout(args.in_dir)
    assert_not_holdout(args.out)
    api_key = _env_or_dotenv(args.env_root, "OPENROUTER_API_KEY")
    if not api_key:
        print("error: OPENROUTER_API_KEY is not set", file=sys.stderr)
        return 2
    os.environ["OPENROUTER_API_KEY"] = api_key  # the writer backend reads it from here

    pack = load_pack(REPO_ROOT / "corpus" / "generator" / "v2" / "packs" / f"{args.pack}.yaml")
    world = simulate_world(pack, seed=args.seed, months=args.months, split=Split(args.split))
    world_plan = plan_world(world, pack, type_cap=None if args.no_type_cap else TYPE_CAP_SHARE)
    decoys = {d.id: d for d in world_plan.decoys}
    wp4_docs = load_written_docs(args.in_dir)

    def fingerprint_ok(doc: WrittenDoc, plan: DocPlan) -> bool:
        return input_fingerprint(render(plan, world, world_plan, pack), plan, world, decoys) == (
            doc.input_fingerprint
        )

    meter = SpendMeter(args.spend_path)
    budget = TeacherRunBudget(cap_usd=args.budget_usd, meter=meter)
    teacher_factory = openrouter_teacher_factory(api_key, meter)

    regenerator: Regenerator | None = None
    if not args.no_regenerate:
        config = WriterConfig(models=DEFAULT_ROTATION)
        pool = WriterPool(config.models, world_doc_total=len(writer_doc_plans(world_plan, config)))
        for d in wp4_docs.values():
            if d.status == "written" and d.writer_model:
                pool.assign(d.writer_model)
        session = WriterSession(
            backend=OpenRouterWriterBackend(),
            pool=pool,
            budget=budget,
            config=config,
            call_log=args.out / "writer_calls.jsonl",
        )
        regenerator = WriterRegenerator(world, world_plan, pack, session, finished=wp4_docs)

    judge: LabelJudgeSettings | None = None
    if args.label_judge:
        from portolan.training.label_judge import (
            LABEL_JUDGE_SPEND_KEY,
            judge_model_allowed,
            openrouter_judge_call,
        )

        if not judge_model_allowed(args.label_judge):
            print(
                f"error: --label-judge {args.label_judge} is not on the allowlist", file=sys.stderr
            )
            return 2
        judge_start = meter.corpus_usd(LABEL_JUDGE_SPEND_KEY)
        judge = LabelJudgeSettings(
            model=args.label_judge,
            call=openrouter_judge_call(api_key, meter),
            budget_left=lambda: (
                meter.corpus_usd(LABEL_JUDGE_SPEND_KEY) - judge_start < args.judge_budget_usd
            ),
            log_path=args.out / "label_judge.jsonl",
        )

    ctx = CompletenessContext(
        world=world,
        world_plan=world_plan,
        backend_factory=teacher_factory,
        budget=budget,
        company_context=world_company_context(world) if args.company_context == "world" else None,
        regenerator=regenerator,
        call_log=args.out / "teacher_calls.jsonl",
        client_meters=True,
        label_judge=judge,
    )
    doc_ids = [d for d in args.doc_ids.split(",") if d] or None
    print(f"world {world.id}: completeness pass over {len(wp4_docs)} WP4 documents", flush=True)
    report = run_completeness(
        ctx,
        wp4_docs,
        out_dir=args.out,
        doc_ids=doc_ids,
        audit_share=args.audit_share,
        fingerprint_ok=None if args.skip_fingerprint_check else fingerprint_ok,
        progress=lambda line: print(line, flush=True),
    )
    print(
        json.dumps(
            {k: report[k] for k in ("stop_reason", "spend", "residual", "gates", "accepted")},
            indent=2,
        )
    )
    return 4 if report["stop_reason"] else 0


if __name__ == "__main__":
    sys.exit(main())
