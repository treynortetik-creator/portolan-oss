"""Tests for world generator v2 work package 5, the completeness pass
(`01-spec/world-generator-v2-2026-09-22.md` sections 3.3 to 3.5, and section
9's WP5 row). Every model call is stubbed; nothing here touches the network.

Documents come from WP4's own writer driven by the obedient stub writer
(`tests/v2_writer_fakes`), so spans, labels and prepared text are the real
WP4 output shapes. The teacher is a stub `Backend` that returns facts
quoting the prepared text.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from corpus.generator.prose import PAID_BUDGET_STOP_MARKER
from corpus.generator.v2.completeness import (
    AUDIT_TEACHERS,
    LABEL_SOURCE_SILVER,
    PROMPT_VARIANT,
    TEACHER_MODEL,
    CompletenessContext,
    LabelJudgeSettings,
    PassAttempt,
    PreparedTextDrift,
    ScriptedSpan,
    SourceInput,
    TeacherFact,
    TeacherRunBudget,
    WorldLexicon,
    WriterRegenerator,
    _account_family,
    _best_overlap,
    _claim_numbers,
    _gate,
    broken_dates,
    completed_labels,
    contradictions_for,
    core_span,
    decide,
    invented_prices,
    pick_audit_teacher,
    process_document,
    run_completeness,
    scripted_spans,
    select_audit_docs,
    source_inputs,
    span_iou,
    spans_match,
    teacher_pass,
    unmatched_marker_text,
    unscripted_numbers,
    world_company_context,
)
from corpus.generator.v2.models import PersonSide, Split
from corpus.generator.v2.packs.schema import VerticalPack, load_pack
from corpus.generator.v2.planner import DocKind, DocPlan, DocRole, WorldDocPlan, plan_world
from corpus.generator.v2.renderers import render
from corpus.generator.v2.simulator import simulate_world
from corpus.generator.v2.writer import (
    CORPUS_KEY,
    DEFAULT_ROTATION,
    RunBudget,
    WriterBudgetExceeded,
    WriterConfig,
    WriterPool,
    WriterSession,
    WrittenDoc,
    build_cast,
    slot_specs_for,
    write_document,
)
from portolan.extract.prompt import build_system_prompt, build_user_message
from portolan.extract.schema import Coverage, ExtractionResult, Fact, SourceType
from portolan.metering import metered
from portolan.training.spend import SpendMeter
from tests.v2_writer_fakes import ScriptedBackend, render_document, repair_reply

REPO_ROOT = Path(__file__).resolve().parents[1]
PACK_PATH = REPO_ROOT / "corpus" / "generator" / "v2" / "packs" / "medical_devices.yaml"
OTHER_WRITER = "deepseek/deepseek-v4-flash"


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pack() -> VerticalPack:
    return load_pack(PACK_PATH)


@pytest.fixture(scope="module")
def world(pack: VerticalPack):
    return simulate_world(pack, seed=42, months=8, split=Split.TRAIN)


@pytest.fixture(scope="module")
def world_plan(world, pack) -> WorldDocPlan:
    # Uncapped: the documents these tests pick keep every planted fact (the
    # plan-time per-type cap is WP3's, tested there).
    return plan_world(world, pack, type_cap=None)


class DecoyIndex(dict):
    """Decoys by id, plus the `WorldDocPlan` they came from (`render` takes
    the world plan; the writer takes the decoy index)."""

    world_plan: WorldDocPlan


@pytest.fixture(scope="module")
def decoys(world_plan):
    index = DecoyIndex({d.id: d for d in world_plan.decoys})
    index.world_plan = world_plan
    return index


def _first(
    world_plan: WorldDocPlan,
    kinds,
    *,
    facts: int = 0,
    decoy: bool | None = None,
    role: DocRole = DocRole.EVENT,
) -> DocPlan:
    for d in sorted(world_plan.doc_plans, key=lambda d: (d.date, d.id)):
        if d.kind not in kinds or d.role != role or len(d.fact_ids) < facts:
            continue
        if decoy is not None and bool(d.decoy_ids) != decoy:
            continue
        return d
    raise LookupError(kinds)


def _compact(text: str, user: str) -> str:
    """The obedient writer pads its closing line with filler to the planned
    length; keep 60% of it (over WP4's 30% truncation floor) so one call
    stays one envelope and one teacher call."""
    m = re.search(r"^LENGTH: about (\d+) words", user, re.MULTILINE)
    extra = len(text.split()) - int(0.6 * int(m.group(1))) if m else 0
    if extra <= 0:
        return text
    return re.sub(
        r"closing words((?: fixture)+)",
        lambda g: "closing words" + " fixture" * max(0, len(g.group(1).split()) - extra),
        text,
    )


def compact_obedient(model: str, system: str, user: str, n: int) -> str:
    if "REWRITE THESE UNITS" in user:
        return repair_reply(user)
    return _compact(render_document(system, user), user)


def _write(plan, world, pack, decoys, reply=compact_obedient) -> WrittenDoc:
    config = WriterConfig(models=DEFAULT_ROTATION)
    session = WriterSession(
        backend=ScriptedBackend(reply),
        pool=WriterPool(config.models, world_doc_total=300),
        budget=RunBudget(cap_usd=5.0),
        config=config,
    )
    doc = write_document(
        render(plan, world, decoys.world_plan, pack),
        plan,
        world,
        pack,
        decoys_by_id=decoys,
        session=session,
    )
    assert doc.status == "written", doc.reason
    return doc


@pytest.fixture(scope="module")
def call_plan(world_plan) -> DocPlan:
    return _first(world_plan, {DocKind.CALL}, facts=3, decoy=True)


@pytest.fixture(scope="module")
def call_doc(call_plan, world, pack, decoys) -> WrittenDoc:
    return _write(call_plan, world, pack, decoys)


@pytest.fixture(scope="module")
def email_plan(world_plan) -> DocPlan:
    return _first(world_plan, {DocKind.EMAIL}, facts=1)


@pytest.fixture(scope="module")
def email_doc(email_plan, world, pack, decoys) -> WrittenDoc:
    return _write(email_plan, world, pack, decoys)


@pytest.fixture(scope="module")
def negative_plan(world_plan) -> DocPlan:
    return _first(world_plan, set(DocKind), decoy=True, role=DocRole.HARD_NEGATIVE)


@pytest.fixture(scope="module")
def negative_doc(negative_plan, world, pack, decoys) -> WrittenDoc:
    return _write(negative_plan, world, pack, decoys)


# --------------------------------------------------------------------------
# Stub teacher and regenerator
# --------------------------------------------------------------------------

Row = dict[str, Any]


@dataclass
class TeacherCallSeen:
    model: str
    system: str
    user: str
    envelope: Any


@dataclass
class FakeTeacher:
    """`rows(model, envelope, n)` gives the facts for call `n` (0-based,
    counting every call to every model). A row is `{type, claim, quote,
    entities?}`; rows whose quote is not in the envelope are still sent, so
    the gate can be tested."""

    rows: Callable[[str, Any, int], list[Row]]
    cost: float = 0.002
    fail_models: set[str] = field(default_factory=set)
    seen: list[TeacherCallSeen] = field(default_factory=list)

    def factory(self, model: str):
        teacher = self

        class _Backend:
            def extract(self, envelope, system_prompt, sink, tenant):  # noqa: ANN001
                n = len(teacher.seen)
                teacher.seen.append(
                    TeacherCallSeen(model, system_prompt, build_user_message(envelope), envelope)
                )
                with metered(sink, tenant, "extract", model) as usage:
                    usage.cost_override = teacher.cost
                    if model in teacher.fail_models:
                        raise RuntimeError("status 503 (fake)")
                    rows = teacher.rows(model, envelope, n)
                facts = [
                    Fact(
                        id=f"f{i}",
                        type=r["type"],
                        claim=r["claim"],
                        quote=r.get("quote"),
                        entities=r.get("entities", []),
                        locator=r.get("locator", "L1"),
                        sentiment=0,
                        confidence=0.9,
                    )
                    for i, r in enumerate(rows)
                ]
                return ExtractionResult(
                    source_id=envelope.source_id,
                    summary="fixture",
                    facts=facts,
                    coverage=Coverage(segments_read=1, segments_total=1),
                )

        return _Backend()


def quoting(*rows: Row, until: int | None = None, audit: list[Row] | None = None):
    """Rows whose quote appears in the envelope; only for the first `until`
    completeness calls when given. `audit` rows go to any non-Kimi model."""

    def fn(model: str, envelope: Any, n: int) -> list[Row]:
        pool = list(rows) if model == TEACHER_MODEL else list(audit or [])
        if model == TEACHER_MODEL and until is not None and n >= until:
            return []
        return [r for r in pool if r.get("quote") and r["quote"] in envelope.text]

    return fn


@dataclass
class FakeRegenerator:
    make: Callable[[str], WrittenDoc | None]
    raise_budget: bool = False
    seen: list[tuple[str, list[str], set[str]]] = field(default_factory=list)

    def regenerate(self, plan_id: str, do_not_state: list[str], avoid_models: set[str]):
        self.seen.append((plan_id, list(do_not_state), set(avoid_models)))
        if self.raise_budget:
            raise WriterBudgetExceeded(f"{PAID_BUDGET_STOP_MARKER}: fake stop")
        return self.make(plan_id)


def _ctx(
    world, world_plan, teacher: FakeTeacher, *, cap: float = 5.0, meter=None, regenerator=None
):
    return CompletenessContext(
        world=world,
        world_plan=world_plan,
        backend_factory=teacher.factory,
        budget=TeacherRunBudget(cap_usd=cap, meter=meter),
        company_context=world_company_context(world),
        regenerator=regenerator,
        sleep=lambda _s: None,
    )


def _label(doc: WrittenDoc, index: int = 0):
    return doc.labels[index]


def _slot_speaker(plan: DocPlan, marker_id: str, world, pack, decoys) -> str:
    specs = slot_specs_for(render(plan, world, decoys.world_plan, pack), plan, world, decoys)
    spec = next(s for s in specs if s.marker_id == marker_id)
    return world.person_by_id()[spec.speaker_person_id].name


def _misattributing_writer(marker_id: str, wrong: str, *, repair: bool):
    """The obedient writer, except `marker_id` is always said by `wrong`
    (check 4 fails); its repair replies fix nothing when `repair` is off."""

    def reply(model, system, user, n):  # noqa: ANN001
        if "REWRITE THESE UNITS" in user:
            return repair_reply(user) if repair else "nothing useful\n"
        return _compact(
            render_document(system, user, mutate={marker_id: lambda item, span: (span, wrong)}),
            user,
        )

    return reply


# --------------------------------------------------------------------------
# Matching primitives
# --------------------------------------------------------------------------


def test_spans_match_by_iou_or_containment() -> None:
    assert span_iou((0, 10), (5, 15)) == pytest.approx(5 / 15)
    assert not spans_match((0, 10), (5, 15))
    assert spans_match((0, 10), (2, 11))  # IoU 8/11
    assert spans_match((0, 10), (3, 6))  # teacher span inside the label
    assert spans_match((3, 6), (0, 10))  # label inside the teacher span
    assert not spans_match((0, 10), (10, 20))


def test_decide_rewrites_only_for_world_breaking_candidates() -> None:
    """The cost decision (spec Build notes, WP5, 2026-09-22): consistent
    candidates are labeled where they stand, however many there are."""

    def cands(n: int, blocking: int = 0, unlabelable: int = 0) -> list[TeacherFact]:
        return [
            TeacherFact(
                teacher_model=TEACHER_MODEL,
                role="completeness",
                doc_id="d",
                envelope=1,
                fact_type="request",
                claim=f"c{i}",
                blockers=["invented_entity:MedFlow"] if i < blocking else [],
                unlabelable=i >= n - unlabelable,
            )
            for i in range(n)
        ]

    assert decide(3, cands(1), regenerated=False).action == "accept"
    assert decide(3, cands(9), regenerated=False).action == "accept"
    assert decide(3, cands(9, blocking=1), regenerated=False).action == "regenerate"
    assert decide(3, cands(9, blocking=1), regenerated=True).action == "drop"
    assert decide(0, cands(0), regenerated=False).action == "clean_negative"
    assert decide(0, cands(3), regenerated=False).action == "relabel_positive"
    assert decide(0, cands(1, blocking=1), regenerated=False).action == "regenerate"
    # Found but unlabelable: shipped as known residual, never a rewrite.
    assert decide(3, cands(2, unlabelable=2), regenerated=False).action == "accept"
    assert decide(0, cands(1, unlabelable=1), regenerated=False).action == "accept"


# --------------------------------------------------------------------------
# Inputs and blindness
# --------------------------------------------------------------------------


def test_source_inputs_reproduce_wp4_text_and_date_the_call(call_doc, call_plan) -> None:
    [src] = source_inputs(call_doc, call_plan)
    assert src.text == call_doc.prepared_sources[0].text
    assert src.source_type == SourceType.CALL
    assert src.source_date == call_plan.date
    assert src.participants  # the speakers, as FileConnector reads them
    tampered = call_doc.model_copy(deep=True)
    tampered.prepared_sources[0].text += "x"
    with pytest.raises(PreparedTextDrift):
        source_inputs(tampered, call_plan)


def test_teacher_is_blind_to_the_script(world, world_plan, call_doc, call_plan) -> None:
    teacher = FakeTeacher(quoting())
    ctx = _ctx(world, world_plan, teacher)
    state = process_document(ctx, call_doc, call_plan)
    assert state.status == "accepted"
    assert len(teacher.seen) == 1
    seen = teacher.seen[0]
    assert seen.model == TEACHER_MODEL
    assert seen.system == build_system_prompt(
        company_context=world_company_context(world), variant=PROMPT_VARIANT
    )
    assert call_doc.prepared_sources[0].text in seen.user
    claims = [f.claim for f in world.facts if f.id in call_plan.fact_ids]
    for text in (seen.system, seen.user):
        assert "⟦" not in text and "⟧" not in text
        for fact_id in [*call_plan.fact_ids, *call_plan.decoy_ids]:
            assert fact_id not in text
        for claim in claims:
            assert claim not in text


# --------------------------------------------------------------------------
# Gate and triage
# --------------------------------------------------------------------------


def test_confirmed_fact_records_type_agreement_and_recall(
    world, world_plan, call_doc, call_plan
) -> None:
    first, second = _label(call_doc, 0), _label(call_doc, 1)
    line = next(
        line for line in call_doc.prepared_sources[0].text.split("\n") if second.text in line
    )
    teacher = FakeTeacher(
        quoting(
            # Part of the first label, same type.
            {
                "type": first.fact_type.value,
                "claim": "usage fixture",
                "quote": first.text[:40],
                "entities": ["MedFlow"],  # invented inside a scripted span
            },
            # The whole line around the second label, another type.
            {"type": "request", "claim": "request fixture", "quote": line.split(": ", 1)[1]},
        )
    )
    state = process_document(_ctx(world, world_plan, teacher), call_doc, call_plan)
    facts = state.attempts[0].usable()
    assert [(f.triage, f.matched_id, f.type_agrees) for f in facts] == [
        ("confirmed", first.fact_id, True),
        ("confirmed", second.fact_id, False),
    ]
    assert facts[0].unknown_entities == ["MedFlow"]  # flagged, not blocking
    s = state.summary()
    assert s["scripted_confirmed"] == 2 and s["missed_scripted"] == len(call_doc.labels) - 2
    assert state.status == "accepted" and not state.silver_labels


def test_gate_drops_unsupported_quotes_names_and_numbers(
    world, world_plan, call_doc, call_plan
) -> None:
    rows = [
        {"type": "request", "claim": "asked for a thing", "quote": "words nobody wrote down"},
        {"type": "request", "claim": "Zebulon Quartz asked", "quote": "fixture closing words"},
        {"type": "request", "claim": "asked for 97 units", "quote": "fixture closing words"},
    ]
    teacher = FakeTeacher(lambda _m, _e, _n: rows)
    state = process_document(_ctx(world, world_plan, teacher), call_doc, call_plan)
    facts = state.attempts[0].facts
    assert [f.gate for f in facts] == ["quote_gate", "overlap_gate_name", "overlap_gate_numeral"]
    assert state.attempts[0].usable() == []
    # The text does say something at that line: a grounded, unlabelable
    # candidate (the second is the same span and type again).
    assert not facts[0].grounded and facts[2].duplicate
    [c] = state.attempts[0].of("candidate")
    assert c is facts[1] and c.unlabelable and c.tier == "probe:exact"
    assert state.status == "needs_regeneration" and state.silver_labels == []


def test_dropped_fact_on_a_script_span_is_no_candidate(
    world, world_plan, call_doc, call_plan
) -> None:
    label = _label(call_doc)
    rows = [
        # Claim with a number the text lacks, on the label's own span.
        {"type": label.fact_type.value, "claim": "used in 97 units", "quote": label.text},
        # Two sentences the text never has together; the second part is
        # the label's words, so fuzzy containment finds the label.
        {
            "type": label.fact_type.value,
            "claim": "usage fixture",
            "quote": f"{label.text} and more words here",
        },
    ]
    teacher = FakeTeacher(lambda _m, _e, _n: rows)
    state = process_document(_ctx(world, world_plan, teacher), call_doc, call_plan)
    a, b = state.attempts[0].facts
    assert (a.gate, a.triage, a.matched_id) == ("overlap_gate_numeral", "confirmed", label.fact_id)
    assert b.gate == "quote_not_exact" and not b.has_span
    assert (b.triage, b.matched_id) == ("confirmed", label.fact_id)
    assert state.summary()["scripted_confirmed"] == 0  # recall counts kept facts only
    assert state.status == "accepted"


def _gate_one(
    text: str,
    claim: str,
    quote: str,
    source_date: date = date(2026, 3, 1),
    context_names: str = "",
):
    src = SourceInput("doc-1", SourceType.DOC, source_date, (), text)
    base = TeacherFact(
        teacher_model=TEACHER_MODEL,
        role="completeness",
        doc_id="doc-1",
        envelope=1,
        fact_type="complaint",
        claim=claim,
        quote=quote,
    )
    fact = SimpleNamespace(claim=claim, quote=quote)
    return _gate(
        fact, envelope_text=text, offset=0, source=src, base=base, context_names=context_names
    )


def test_gate_reads_claims_like_check_6() -> None:
    # Kimi writes "Hospital 2's" and an ISO date for "February 2"; neither
    # may fail the numeral rule when the text supports it.
    text = "[L1] Marrowbrook Community Hospital 2 said the pump failed on February 2."
    kept = _gate_one(
        text,
        "Marrowbrook Community Hospital 2's pump failed on 2026-02-02",
        "the pump failed on February 2",
    )
    assert kept.gate == "kept" and kept.tier == "exact"
    invented = _gate_one(text, "The pump failed on 2026-02-09", "the pump failed on February 2")
    assert invented.gate == "overlap_gate_numeral"
    ordinal = "[L1] As of this QBR on February 2nd, the pump failed."
    spelled = _gate_one(ordinal, "The pump failed at the February 2, 2026 QBR", "the pump failed")
    assert spelled.gate == "kept"
    counted = _gate_one(ordinal, "Five pumps, 5 in all, failed", "the pump failed")
    assert counted.gate == "overlap_gate_numeral"


def test_gate_takes_calendar_words_and_context_names() -> None:
    # Kimi dates "by the 9th" with its month, and names the vendor and full
    # product names it read in the company context.
    text = "[L1] We will keep the T3 integration stable by the 9th."
    claim = "Ryecrest MedTech committed to keep the Alderline T3 integration stable by April 9"
    quote = "We will keep the T3 integration stable by the 9th"
    assert _gate_one(text, claim, quote).gate == "overlap_gate_name"
    kept = _gate_one(text, claim, quote, context_names="Ryecrest MedTech\nAlderline T3")
    assert kept.gate == "kept"
    rival = "Knollwell Biomedical committed to keep the integration stable by April 9"
    assert _gate_one(text, rival, quote, context_names="Ryecrest MedTech").gate == (
        "overlap_gate_name"
    )


def test_gate_falls_back_to_the_quotes_first_sentence() -> None:
    text = (
        "[L1] The pump failed twice last week. We are escalating it today. "
        "The team will send a full report by Friday."
    )
    joined = "The pump failed twice last week. The team will send a full report by Friday."
    f = _gate_one(text, "The pump failed twice and a report is due", joined)
    assert f.gate == "kept" and f.tier == "sentence:exact"
    assert f.span_text == "The pump failed twice last week."
    assert text[f.char_start : f.char_end] == f.span_text
    short = _gate_one(text, "It failed", "It failed. The team will send a full report by Friday.")
    assert short.gate == "quote_not_exact"


def test_matching_ignores_edge_punctuation() -> None:
    text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. We are switching vendors."
    decoy_start = text.index("We are")
    decoy = ScriptedSpan("decoy", "D1", "doc-1", decoy_start, len(text), True, decoy_class_id="x")
    teacher = TeacherFact(
        teacher_model=TEACHER_MODEL,
        role="completeness",
        doc_id="doc-1",
        envelope=1,
        fact_type="risk",
        claim="switching vendors",
        char_start=0,
        char_end=len(text) - 1,  # stops before the full stop
    )
    assert not spans_match(teacher.span, decoy.span)
    assert core_span(text, decoy.span) == (decoy_start, len(text) - 1)
    hit = _best_overlap(teacher, [decoy], {"doc-1": text})
    assert hit is not None and hit[0].marker_id == "D1"


def test_decoy_hits_count_verified_and_dropped_decoys(
    world, world_plan, pack, decoys, call_doc, call_plan
) -> None:
    [decoy] = call_doc.decoy_spans
    teacher = FakeTeacher(quoting({"type": "risk", "claim": "a risk", "quote": decoy.text}))
    state = process_document(_ctx(world, world_plan, teacher), call_doc, call_plan)
    [f] = state.attempts[0].usable()
    assert (f.triage, f.matched_id, f.decoy_verified) == ("decoy_hit", decoy.decoy_id, True)
    assert f.decoy_class_id == decoy.decoy_class_id

    # A decoy WP4 drops for the wrong speaker is still decoy wording.
    cast, _a, _p = build_cast(call_plan, world, [])
    speaker = _slot_speaker(call_plan, decoy.decoy_id, world, pack, decoys)
    wrong = next(p.name for p in cast if p.name != speaker)
    dropped_doc = _write(
        call_plan, world, pack, decoys, _misattributing_writer(decoy.decoy_id, wrong, repair=True)
    )
    outcome = next(o for o in dropped_doc.slot_outcomes if o.marker_id == decoy.decoy_id)
    assert outcome.status == "dropped" and outcome.dropped_text
    teacher = FakeTeacher(
        quoting({"type": "risk", "claim": "a risk", "quote": outcome.dropped_text})
    )
    state = process_document(_ctx(world, world_plan, teacher), dropped_doc, call_plan)
    [f] = state.attempts[0].usable()
    assert (f.triage, f.matched_id, f.decoy_verified) == ("decoy_hit", decoy.decoy_id, False)


def test_shipped_labels_carry_the_teachers_claim(
    world, world_plan, call_doc, call_plan, tmp_path
) -> None:
    """The student learns the production teacher's claim style: a confirmed
    scripted label ships the blind teacher's claim; the scripted claim stays
    as metadata; a label the teacher missed (or typed differently) keeps the
    scripted claim."""
    agree, disagree, missed = call_doc.labels[:3]
    teacher = FakeTeacher(
        quoting(
            {"type": agree.fact_type.value, "claim": "teacher words one", "quote": agree.text},
            {"type": "request", "claim": "teacher words two", "quote": disagree.text},
        )
    )
    run_completeness(
        _ctx(world, world_plan, teacher), {call_plan.id: call_doc}, out_dir=tmp_path, audit_share=0
    )
    rows = [json.loads(line) for line in (tmp_path / "labels.jsonl").read_text().splitlines()]
    by_id = {r["fact_id"]: r for r in rows}
    claims = {f.id: f.claim for f in world.facts}
    a, d, m = by_id[agree.fact_id], by_id[disagree.fact_id], by_id[missed.fact_id]
    assert (a["claim"], a["claim_source"]) == ("teacher words one", "teacher")
    assert a["scripted_claim"] == claims[agree.fact_id]
    assert a["claim_teacher_model"] == TEACHER_MODEL
    assert (d["claim"], d["claim_source"], d["teacher_type"]) == (
        claims[disagree.fact_id],
        "scripted",
        "request",
    )
    assert (m["claim"], m["claim_source"], m["teacher_type"]) == (
        claims[missed.fact_id],
        "scripted",
        None,
    )
    report = json.loads((tmp_path / "qa_report.json").read_text())
    assert report["scripted_label_claims"] == {
        "teacher": 1,
        "teacher_restatement": 0,
        "scripted_teacher_missed": len(call_doc.labels) - 2,
        "scripted_type_disagrees": 1,
        "documents_with_scripted_labels": 1,
        "documents_claims_complete": 0,
    }


def test_label_found_only_at_another_mention_takes_that_claim(world, call_doc) -> None:
    label = call_doc.labels[0]
    restated = TeacherFact(
        teacher_model=TEACHER_MODEL,
        role="completeness",
        doc_id=label.doc_id,
        envelope=1,
        fact_type=label.fact_type.value,
        claim="teacher words at the other mention",
        triage="restatement",
        matched_id=label.fact_id,
        similarity=0.9,
        type_agrees=True,
    )
    attempt = PassAttempt(
        version="original", writer_model=None, scripted_labels=1, facts=[restated]
    )
    [done] = completed_labels([label], attempt, world)
    assert (done.claim, done.claim_source) == (
        "teacher words at the other mention",
        "teacher_restatement",
    )
    assert done.char_start == label.char_start and done.fact_type == label.fact_type


def test_restatement_is_a_second_locator_not_a_candidate(
    world, world_plan, call_doc, call_plan
) -> None:
    label = _label(call_doc, 2)
    # The label's own words, so the claim's names are in the document.
    claim = " ".join(label.text.split()[4:])
    teacher = FakeTeacher(
        quoting(
            {"type": label.fact_type.value, "claim": claim, "quote": label.text},
            {"type": label.fact_type.value, "claim": claim, "quote": "fixture reply words"},
        )
    )
    state = process_document(_ctx(world, world_plan, teacher), call_doc, call_plan)
    restated = [f for f in state.attempts[0].usable() if f.triage == "restatement"]
    assert len(restated) == 1 and restated[0].matched_id == label.fact_id
    assert restated[0].similarity == pytest.approx(1.0)
    assert state.status == "accepted"
    [silver] = state.silver_labels
    assert (silver.fact_id, silver.restates_fact_id) == (label.fact_id, label.fact_id)
    assert silver.text == "fixture reply words"


def test_candidate_becomes_a_silver_label_with_an_exact_span(
    world, world_plan, call_doc, call_plan, tmp_path
) -> None:
    claim = "The customer requested fixture closing words"
    teacher = FakeTeacher(
        quoting({"type": "request", "claim": claim, "quote": "fixture closing words"})
    )
    ctx = _ctx(world, world_plan, teacher)
    report = run_completeness(ctx, {call_plan.id: call_doc}, out_dir=tmp_path)
    state = process_document(ctx, call_doc, call_plan)
    assert state.status == "accepted"
    [silver] = state.silver_labels
    text = call_doc.prepared_sources[0].text
    assert text[silver.char_start : silver.char_end] == silver.text == "fixture closing words"
    assert silver.label_source == LABEL_SOURCE_SILVER
    assert silver.teacher_model == TEACHER_MODEL and silver.claim == claim
    assert silver.fact_id == f"{call_plan.id}-C01" and silver.restates_fact_id is None
    line = next(line for line in text.split("\n") if "fixture closing words" in line)
    assert line.startswith(f"[{silver.locator}]")
    lines = (tmp_path / "labels.jsonl").read_text().splitlines()
    assert len(lines) == len(call_doc.labels) + 1
    assert {k: report["silver"][k] for k in ("labels", "new_facts", "recovered_planted")} == {
        "labels": 1,
        "new_facts": 1,
        "recovered_planted": 0,
    }
    assert (tmp_path / "files").exists() and (tmp_path / "qa_report.json").exists()


def test_fact_on_a_dropped_slot_is_a_recovered_label(
    world, world_plan, pack, decoys, call_plan
) -> None:
    target = call_plan.fact_ids[0]
    cast, _a, _p = build_cast(call_plan, world, [])
    speaker = _slot_speaker(call_plan, target, world, pack, decoys)
    wrong = next(p.name for p in cast if p.name != speaker)
    doc = _write(
        call_plan, world, pack, decoys, _misattributing_writer(target, wrong, repair=False)
    )
    outcome = next(o for o in doc.slot_outcomes if o.marker_id == target)
    assert outcome.status == "dropped" and outcome.dropped_text
    teacher = FakeTeacher(
        quoting({"type": "usage", "claim": "usage fixture", "quote": outcome.dropped_text})
    )
    state = process_document(_ctx(world, world_plan, teacher), doc, call_plan)
    [f] = state.attempts[0].usable()
    assert (f.triage, f.recovers_fact_id, f.tier) == ("planted", target, "exact|span")
    assert state.attempts[0].open_facts() == [] and state.status == "accepted"
    [silver] = state.silver_labels
    assert (silver.fact_id, silver.recovers_fact_id, silver.claim) == (
        target,
        target,
        "usage fixture",
    )
    assert not silver.new_fact


def _unmarked(doc: WrittenDoc, fact_id: str) -> WrittenDoc:
    """`doc` as if the writer had stated `fact_id` without its markers: the
    words stay in the text, WP4 has no label for it."""
    out = doc.model_copy(deep=True)
    out.labels = [x for x in out.labels if x.fact_id != fact_id]
    for o in out.slot_outcomes:
        if o.marker_id == fact_id:
            o.status, o.final_failures, o.dropped_text = "dropped", ["c1:missing_marker"], None
    return out


def test_unlabeled_planted_fact_is_recovered_by_its_claim(
    world, world_plan, call_doc, call_plan, tmp_path
) -> None:
    label = call_doc.labels[0]
    planted = next(f for f in world.facts if f.id == label.fact_id)
    doc = _unmarked(call_doc, label.fact_id)
    claim = planted.claim  # the teacher reads the planted fact back
    teacher = FakeTeacher(
        quoting({"type": planted.fact_type.value, "claim": claim, "quote": label.text})
    )
    report = run_completeness(
        _ctx(world, world_plan, teacher), {call_plan.id: doc}, out_dir=tmp_path, audit_share=0
    )
    state = process_document(_ctx(world, world_plan, teacher), doc, call_plan)
    [f] = state.attempts[0].usable()
    assert (f.triage, f.matched_id, f.type_agrees) == ("planted", label.fact_id, True)
    assert f.tier == "exact|claim" and f.similarity >= 0.30
    assert state.status == "accepted" and state.attempts[0].open_facts() == []
    [silver] = state.silver_labels
    assert (silver.fact_id, silver.recovers_fact_id) == (label.fact_id, label.fact_id)
    assert report["planted_recovery"] == {
        "facts": 1,
        "by_claim": 1,
        "unlabelable": 0,
        "type_disagrees": 0,
        "candidates": 0,
    }
    assert report["silver"]["recovered_planted"] == 1 and report["silver"]["new_facts"] == 0
    assert report["residual"]["kimi_only"]["u_total"] == 0


def test_document_whose_facts_are_all_unmarked_is_a_positive(
    world, world_plan, call_doc, call_plan
) -> None:
    doc = call_doc
    for label in call_doc.labels:
        doc = _unmarked(doc, label.fact_id)
    assert doc.labels == []
    first = call_doc.labels[0]
    planted = next(f for f in world.facts if f.id == first.fact_id)
    teacher = FakeTeacher(
        quoting({"type": planted.fact_type.value, "claim": planted.claim, "quote": first.text})
    )
    state = process_document(_ctx(world, world_plan, teacher), doc, call_plan)
    assert (state.status, state.doc_class) == ("accepted", "positive")
    assert [x.recovers_fact_id for x in state.silver_labels] == [first.fact_id]


def test_planted_match_needs_a_shared_quantity(world, world_plan, call_doc, call_plan) -> None:
    label = call_doc.labels[0]
    planted = next(f for f in world.facts if f.id == label.fact_id)
    assert "10" in planted.claim
    doc = _unmarked(call_doc, label.fact_id)
    planted_values = {int(n) for n in re.findall(r"\d+", planted.claim)}
    # A count the quote states, unequal by value to every planted quantity: "03" is the
    # planted 3, and a percentage the text never states fails the gate's numeral rule.
    other_number = next(
        n for n in re.findall(r"\b\d{1,3}\b", label.text) if int(n) not in planted_values
    )
    claim = re.sub(r"\d+%?", other_number, planted.claim)  # every quantity changed
    teacher = FakeTeacher(
        quoting({"type": planted.fact_type.value, "claim": claim, "quote": label.text})
    )
    state = process_document(_ctx(world, world_plan, teacher), doc, call_plan)
    [f] = state.attempts[0].usable()
    assert f.triage == "candidate"


def test_claim_quantities_compare_by_value_and_kind() -> None:
    assert _claim_numbers("a 10% discount for 03 more units at $5k across three sites") == {
        "10%",
        "3",
        "5000",
    }
    assert _claim_numbers("a 10 percent discount") != _claim_numbers("10 more units")
    assert _claim_numbers("$48k renewal") == _claim_numbers("$48,000 renewal")
    assert _claim_numbers("signed March 27, 2026 (2026-03-27, 3/27) for fiscal 2026") == set()


def test_planted_fact_with_a_dropped_claim_keeps_the_scripted_claim(
    world, world_plan, call_doc, call_plan
) -> None:
    label = call_doc.labels[0]
    doc = _unmarked(call_doc, label.fact_id)
    planted = next(f for f in world.facts if f.id == label.fact_id)
    claim = f"Zebulon Quartz said {planted.claim}"  # a name the text lacks
    teacher = FakeTeacher(
        quoting({"type": planted.fact_type.value, "claim": claim, "quote": label.text})
    )
    state = process_document(_ctx(world, world_plan, teacher), doc, call_plan)
    [f] = state.attempts[0].of("planted")
    assert f.gate == "overlap_gate_name" and f.has_span and not f.unlabelable
    assert state.status == "accepted"
    [silver] = state.silver_labels
    assert (silver.claim, silver.claim_source, silver.fact_type) == (
        planted.claim,
        "scripted",
        planted.fact_type,
    )
    assert (silver.char_start, silver.char_end) == (label.char_start, label.char_end)


def test_planted_fact_without_a_span_is_known_residual(
    world, world_plan, call_doc, call_plan
) -> None:
    label = call_doc.labels[0]
    doc = _unmarked(call_doc, label.fact_id)
    planted = next(f for f in world.facts if f.id == label.fact_id)
    teacher = FakeTeacher(
        lambda _m, _e, _n: [
            {
                "type": planted.fact_type.value,
                "claim": planted.claim,
                "quote": f"{label.text} and more words here",  # resolves nowhere exactly
            }
        ]
    )
    state = process_document(_ctx(world, world_plan, teacher), doc, call_plan)
    [f] = state.attempts[0].of("planted")
    assert f.gate == "quote_not_exact" and not f.has_span and f.unlabelable
    assert state.attempts[0].open_facts() == [f]
    assert state.status == "accepted" and state.silver_labels == []
    assert "known residual" not in (state.reason or "")  # the document has scripted labels


# --------------------------------------------------------------------------
# Per-document rules with regeneration
# --------------------------------------------------------------------------

# A candidate that names an entity the world does not have: a rewrite.
BREAKING = {
    "type": "request",
    "claim": "asked for fixture closing words",
    "quote": "fixture closing words",
    "entities": ["Zyphora Systems"],
}
# A candidate consistent with the world: a silver label where it stands.
CONSISTENT = {
    "type": "complaint",
    "claim": "said fixture reply words",
    "quote": "fixture reply words",
}


def test_consistent_candidates_are_labeled_without_a_rewrite(
    world, world_plan, call_doc, call_plan
) -> None:
    also = {"type": "praise", "claim": "said fixture opening words", "quote": "fixture opening"}
    teacher = FakeTeacher(quoting(CONSISTENT, also))
    regen = FakeRegenerator(lambda _p: call_doc)
    state = process_document(
        _ctx(world, world_plan, teacher, regenerator=regen), call_doc, call_plan
    )
    assert regen.seen == [] and state.status == "accepted" and len(state.attempts) == 1
    assert sorted(x.text for x in state.silver_labels) == ["fixture opening", "fixture reply words"]
    assert all(x.new_fact and x.label_source == LABEL_SOURCE_SILVER for x in state.silver_labels)


def test_world_breaking_candidate_regenerates_with_do_not_state(
    world, world_plan, call_doc, call_plan
) -> None:
    teacher = FakeTeacher(quoting(BREAKING, CONSISTENT, until=1))
    regen = FakeRegenerator(lambda _p: call_doc.model_copy(update={"writer_model": OTHER_WRITER}))
    state = process_document(
        _ctx(world, world_plan, teacher, regenerator=regen), call_doc, call_plan
    )
    [breaking] = [f for f in state.attempts[0].of("candidate") if f.blockers]
    assert breaking.blockers == ["invented_entity:Zyphora Systems"]
    # Only the breaking sentence is named; the consistent one would have been labeled.
    assert regen.seen == [(call_plan.id, ["fixture closing words"], {call_doc.writer_model})]
    assert state.status == "accepted" and len(state.attempts) == 2
    assert state.attempts[0].decision == "regenerate"
    assert state.attempts[1].writer_model == OTHER_WRITER
    assert state.silver_labels == []


def test_still_breaking_after_regeneration_is_dropped(
    world, world_plan, call_doc, call_plan, tmp_path
) -> None:
    teacher = FakeTeacher(quoting(BREAKING, CONSISTENT))
    regen = FakeRegenerator(lambda _p: call_doc.model_copy(update={"writer_model": OTHER_WRITER}))
    ctx = _ctx(world, world_plan, teacher, regenerator=regen)
    report = run_completeness(ctx, {call_plan.id: call_doc}, out_dir=tmp_path)
    assert report["documents"]["by_status"] == {"dropped": 1}
    assert report["documents"]["wp5_regeneration_rate"] == 1.0
    assert (tmp_path / "labels.jsonl").read_text() == ""
    assert report["candidates_by_kind"]["call"]["removed_by_regeneration"] == 2
    assert report["candidates_by_kind"]["call"]["in_dropped_or_unfinished_doc"] == 2


def test_regeneration_that_wp4_drops_drops_the_document(
    world, world_plan, call_doc, call_plan
) -> None:
    teacher = FakeTeacher(quoting(BREAKING))
    broken = call_doc.model_copy(update={"status": "dropped", "reason": "fake"})
    regen = FakeRegenerator(lambda _p: broken)
    state = process_document(
        _ctx(world, world_plan, teacher, regenerator=regen), call_doc, call_plan
    )
    assert state.status == "dropped" and "regeneration dropped" in (state.reason or "")
    assert len(teacher.seen) == 1


def test_regeneration_with_no_writer_reply_is_retried_later(
    world, world_plan, call_doc, call_plan
) -> None:
    teacher = FakeTeacher(quoting(BREAKING, until=1))
    silent = call_doc.model_copy(update={"status": "error", "reason": "no writer replied"})
    state = process_document(
        _ctx(world, world_plan, teacher, regenerator=FakeRegenerator(lambda _p: silent)),
        call_doc,
        call_plan,
    )
    assert state.status == "error" and not state.regeneration_tried
    working = FakeRegenerator(lambda _p: call_doc.model_copy(update={"writer_model": OTHER_WRITER}))
    state = process_document(
        _ctx(world, world_plan, teacher, regenerator=working), call_doc, call_plan, prior=state
    )
    assert state.status == "accepted" and len(teacher.seen) == 2


def test_without_a_regenerator_the_document_waits(world, world_plan, call_doc, call_plan) -> None:
    teacher = FakeTeacher(quoting(BREAKING, until=1))
    state = process_document(_ctx(world, world_plan, teacher), call_doc, call_plan)
    assert state.status == "needs_regeneration"
    # A later run with a writer picks it up without paying for the first pass.
    regen = FakeRegenerator(lambda _p: call_doc.model_copy(update={"writer_model": OTHER_WRITER}))
    later = process_document(
        _ctx(world, world_plan, teacher, regenerator=regen), call_doc, call_plan, prior=state
    )
    assert later.status == "accepted" and len(teacher.seen) == 2 and len(regen.seen) == 1


def test_negative_document_rules(world, world_plan, negative_doc, negative_plan) -> None:
    assert not negative_doc.labels
    clean = process_document(
        _ctx(world, world_plan, FakeTeacher(quoting())), negative_doc, negative_plan
    )
    assert clean.status == "clean_negative" and clean.doc_class == "negative"

    consistent = {**BREAKING, "entities": []}
    regen = FakeRegenerator(
        lambda _p: negative_doc.model_copy(update={"writer_model": OTHER_WRITER})
    )
    relabeled = process_document(
        _ctx(world, world_plan, FakeTeacher(quoting(consistent)), regenerator=regen),
        negative_doc,
        negative_plan,
    )
    assert relabeled.status == "relabeled_positive" and len(relabeled.silver_labels) == 1
    assert regen.seen == []

    dropped = process_document(
        _ctx(world, world_plan, FakeTeacher(quoting(BREAKING)), regenerator=regen),
        negative_doc,
        negative_plan,
    )
    assert dropped.status == "dropped" and len(regen.seen) == 1


def test_contradiction_blocks_silver_and_forces_regeneration(
    world, world_plan, call_doc, call_plan
) -> None:
    family = _account_family(world, call_plan.account_id)
    other = next(a for a in world.accounts if a.id not in family)
    row = {
        "type": "request",
        "claim": "asked for fixture closing words",
        "quote": "fixture closing words",
        "entities": [other.name],
    }
    teacher = FakeTeacher(quoting(row, until=1))
    regen = FakeRegenerator(lambda _p: call_doc.model_copy(update={"writer_model": OTHER_WRITER}))
    state = process_document(
        _ctx(world, world_plan, teacher, regenerator=regen), call_doc, call_plan
    )
    [c] = state.attempts[0].of("candidate")
    assert c.contradictions and c.contradictions[0].startswith("other_account:")
    assert state.attempts[0].decision == "regenerate"
    assert state.status == "accepted" and state.silver_labels == []


# --------------------------------------------------------------------------
# World-state checks
# --------------------------------------------------------------------------


def _fact(claim: str, *, fact_type: str = "request", span: str = "", entities=()) -> TeacherFact:
    return TeacherFact(
        teacher_model=TEACHER_MODEL,
        role="completeness",
        doc_id="d",
        envelope=1,
        fact_type=fact_type,
        claim=claim,
        span_text=span,
        entities=list(entities),
    )


def test_contradiction_checks(world, call_plan) -> None:
    lex = WorldLexicon.build(world)
    family = _account_family(world, call_plan.account_id)
    mine = world.account_by_id()[call_plan.account_id]
    other = next(a for a in world.accounts if a.id not in family)

    def check(f: TeacherFact, plan: DocPlan = call_plan) -> list[str]:
        return contradictions_for(f, plan=plan, world=world, lexicon=lex)

    assert check(_fact(f"{other.name} asked for a demo"))[0].startswith("other_account:")
    assert check(_fact(f"{mine.name} asked for a demo")) == []
    assert check(_fact(f"{other.name} and {mine.name} compared notes")) == []

    competitor = world.vendor.competitors[0].name
    assert check(
        _fact(
            "evaluating Zyphora Systems",
            fact_type="competitor_mention",
            entities=["Zyphora Systems"],
        )
    )[0].startswith("unknown_competitor:")
    assert (
        check(
            _fact(
                f"evaluating {competitor}",
                fact_type="competitor_mention",
                entities=[competitor, "Zyphora Systems"],
            )
        )
        == []
    )

    stranger = next(
        p for p in world.people if p.side == PersonSide.CUSTOMER and p.account_id not in family
    )
    assert f"person_not_at_account:{stranger.name}" in check(_fact(f"{stranger.name} asked"))
    vendor_person = next(p for p in world.people if p.side == PersonSide.VENDOR)
    assert check(_fact(f"{vendor_person.name} promised a call")) == []
    leaver = next(p for p in world.people if p.side == PersonSide.CUSTOMER and p.tenure.end)
    later = call_plan.model_copy(
        update={"account_id": leaver.account_id, "date": leaver.tenure.end + timedelta(days=5)}
    )
    assert f"person_not_at_account:{leaver.name}" in check(_fact(f"{leaver.name} asked"), later)
    assert check(_fact(f"{leaver.name} left", fact_type="contact_change"), later) == []

    assert check(_fact(f"renewal moved to {mine.renewal_date.isoformat()}")) == []
    assert check(_fact("renewal moved to 2031-01-01"))[0] == "renewal_date:2031-01-01"
    assert check(_fact("the renewal", span="we renew on 12/31/2033")) == ["renewal_date:2033-12-31"]


def test_invented_prices_and_broken_dates(world, call_plan) -> None:
    family = _account_family(world, call_plan.account_id)
    known = next(
        n for f in world.facts if f.account_id in family for n in re.findall(r"(\d+)%", f.claim)
    )
    unknown = next(
        str(n)
        for n in range(97, 1, -1)
        if not any(str(n) in f.claim for f in world.facts if f.account_id in family)
    )
    assert invented_prices(_fact(f"asked for {known}% off"), plan=call_plan, world=world) == []
    assert invented_prices(
        _fact(f"asked for {unknown}% off", fact_type="pricing"), plan=call_plan, world=world
    ) == [f"{unknown}%"]
    assert invented_prices(_fact("asked for a discount"), plan=call_plan, world=world) == []

    later = (call_plan.date + timedelta(days=12)).isoformat()
    assert broken_dates(_fact(f"the pumps went live on {later}"), plan=call_plan) == [later]
    assert broken_dates(_fact(f"the pumps will go live on {later}"), plan=call_plan) == []
    earlier = (call_plan.date - timedelta(days=3)).isoformat()
    assert broken_dates(_fact(f"the pumps went live on {earlier}"), plan=call_plan) == []


def test_unknown_entities_and_unscripted_numbers_are_informational(world, call_plan) -> None:
    lex = WorldLexicon.build(world)
    product = world.vendor.products[0].name
    person = next(p for p in world.people if p.side == PersonSide.CUSTOMER)
    assert lex.unknown_entities(
        ["MedFlow", product, person.name, "ICU", "nursing team", "acct-nowhere", "Boston"]
    ) == ["MedFlow", "Boston"]
    # The teacher does not always list an invented product as an entity.
    claim = f"Zyrtex asked whether {product} works with the new MedFlow EHR by April. It does."
    assert lex.unknown_names(claim) == ["MedFlow"]
    nums = unscripted_numbers(
        "[00:01:02] We hit 98% compliance and a T2 phase-out by 2027, per the 2 sites",
        plan=call_plan,
        world=world,
        source_text="",
    )
    assert "98%" in nums and "2027" in nums


# --------------------------------------------------------------------------
# Segments and multi-source documents
# --------------------------------------------------------------------------


def test_segmented_source_maps_spans_and_marks_overlap_duplicates(world, world_plan) -> None:
    lines = [f"[L{i}] fixture line {i} with some filler words to pad it out" for i in range(1, 700)]
    text = "\n".join(lines)
    src = SourceInput("long-doc", SourceType.DOC, date(2026, 3, 1), (), text)
    late = "fixture line 650 with some filler words"

    def rows(_m, env, _n):  # noqa: ANN001
        # Every tenth line: a segment starts with the previous one's tail,
        # so a line in that overlap is quoted twice.
        body = [line.split("] ", 1)[1] for line in env.text.split("\n")]
        quotes = [b.replace(" to pad it out", "") for b in body if b.split()[2].endswith("0")]
        return [
            {"type": "request", "claim": "asked about fixture line", "quote": q} for q in quotes
        ]

    ctx = _ctx(world, world_plan, FakeTeacher(rows))
    facts, calls = teacher_pass(
        ctx, [src], model=TEACHER_MODEL, role="completeness", plan_id="p", attempt=0
    )
    assert len(calls) > 1
    kept = [f for f in facts if f.gate == "kept"]
    for f in kept:
        assert text[f.char_start : f.char_end] == f.span_text
    lates = [f for f in kept if f.span_text == late]
    assert lates[0].char_start == text.index(late) and not lates[0].duplicate
    assert all(f.duplicate for f in lates[1:])
    assert any(f.duplicate for f in kept)  # the overlap really repeated a line
    assert len({(f.char_start, f.char_end) for f in kept if not f.duplicate}) == sum(
        1 for f in kept if not f.duplicate
    )


def test_email_runs_the_teacher_once_per_message(world, world_plan, email_doc, email_plan) -> None:
    label = _label(email_doc)
    teacher = FakeTeacher(
        quoting({"type": label.fact_type.value, "claim": "commitment fixture", "quote": label.text})
    )
    state = process_document(_ctx(world, world_plan, teacher), email_doc, email_plan)
    assert len(teacher.seen) == len(email_doc.prepared_sources) > 1
    [f] = state.attempts[0].usable()
    assert (f.doc_id, f.triage, f.matched_id) == (label.doc_id, "confirmed", label.fact_id)


# --------------------------------------------------------------------------
# Audit
# --------------------------------------------------------------------------


def test_audit_selection_is_stable_and_at_least_one() -> None:
    ids = [f"S-doc-{i:05d}" for i in range(100)]
    picked = select_audit_docs(ids, "world-x", 0.10)
    assert len(picked) == 10 and picked == select_audit_docs(list(reversed(ids)), "world-x", 0.10)
    assert picked != select_audit_docs(ids, "world-y", 0.10)
    assert len(select_audit_docs(ids[:3], "world-x", 0.10)) == 1
    assert select_audit_docs(ids, "world-x") == set(ids)  # the audit reads every document
    assert select_audit_docs([], "world-x") == set()


def test_audit_teacher_is_another_family() -> None:
    assert pick_audit_teacher("qwen/qwen3-32b") == "deepseek/deepseek-v3.2"
    assert pick_audit_teacher("deepseek/deepseek-v4-flash") == "qwen/qwen3-235b-a22b-2507"
    assert pick_audit_teacher("mistralai/mistral-small-3.2-24b-instruct") == AUDIT_TEACHERS[0]
    assert all("moonshot" not in m and "kimi" not in m for m in AUDIT_TEACHERS)


def _two_teacher_run(world, world_plan, call_doc, call_plan, tmp_path):
    """Kimi finds labels 0-2, X, and W (its claim names someone the text
    lacks, so it cannot be labeled). The audit finds labels 1-3 (typing
    label 1 differently), X, W with a clean claim, Y (audit only), and Z
    (audit only, names an entity the world lacks)."""
    labels = call_doc.labels
    x = {
        "type": "request",
        "claim": "asked for fixture closing words",
        "quote": "fixture closing words",
    }
    kimi = [
        {"type": lab.fact_type.value, "claim": "fixture", "quote": lab.text} for lab in labels[:3]
    ] + [
        x,
        {"type": "praise", "claim": "flagged a fixture lead 97 times", "quote": "fixture lead"},
    ]
    audit = [
        {"type": "other" if i == 1 else lab.fact_type.value, "claim": "fixture", "quote": lab.text}
        for i, lab in enumerate(labels[:4])
        if i >= 1
    ] + [
        x,
        {"type": "praise", "claim": "flagged a fixture lead", "quote": "fixture lead"},
        {"type": "complaint", "claim": "said fixture reply words", "quote": "fixture reply words"},
        {
            "type": "request",
            "claim": "asked for fixture opening words",
            "quote": "fixture opening words",
            "entities": ["Zyphora Systems"],
        },
    ]
    teacher = FakeTeacher(quoting(*kimi, audit=audit))
    report = run_completeness(
        _ctx(world, world_plan, teacher), {call_plan.id: call_doc}, out_dir=tmp_path
    )
    return teacher, report


def test_audit_labels_only_where_kimi_found_the_span(
    world, world_plan, call_doc, call_plan, tmp_path
) -> None:
    teacher, report = _two_teacher_run(world, world_plan, call_doc, call_plan, tmp_path)
    audit_model = pick_audit_teacher(call_doc.writer_model)
    assert [s.model for s in teacher.seen if s.model != TEACHER_MODEL] == [audit_model]
    rows = [json.loads(line) for line in (tmp_path / "labels.jsonl").read_text().splitlines()]
    silver = {r["text"]: r for r in rows if r["label_source"] == LABEL_SOURCE_SILVER}
    # W: Kimi found the span, its claim failed the gate, the audit's claim labels it.
    assert set(silver) == {"fixture closing words", "fixture lead"}
    rescued = silver["fixture lead"]
    assert (rescued["teacher_model"], rescued["fact_id"]) == (audit_model, f"{call_plan.id}-A01")
    assert rescued["claim"] == "flagged a fixture lead"
    assert sorted(rescued["found_by"]) == sorted([TEACHER_MODEL, audit_model])
    assert silver["fixture closing words"]["teacher_model"] == TEACHER_MODEL
    # Y and Z are the audit's alone: logged, counted as residual, not labeled.
    state = json.loads((tmp_path / "docs" / f"{call_plan.id}.json").read_text())
    unlabeled = {
        f["span_text"]
        for f in state["audit"]["facts"]
        if f["triage"] in ("candidate", "planted") and not f["labeled"]
    }
    assert unlabeled == {"fixture reply words", "fixture opening words"}
    [disagreement] = state["audit"]["type_disagreements"]
    assert disagreement["label_fact_id"] == call_doc.labels[1].fact_id
    assert (disagreement["label_type"], disagreement["audit_type"]) == (
        call_doc.labels[1].fact_type.value,
        "other",
    )
    assert report["audit"]["labels_added"] == 1 and report["audit"]["not_labeled_breaks_world"] == 1
    assert report["silver"]["by_teacher"] == {TEACHER_MODEL: 1, audit_model: 1}
    assert report["residual"]["audit_only_unlabeled"] == 2


def test_residual_counts_what_neither_teacher_found(
    world, world_plan, call_doc, call_plan, tmp_path
) -> None:
    _teacher, report = _two_teacher_run(world, world_plan, call_doc, call_plan, tmp_path)
    n = len(call_doc.labels)
    capture = report["residual"]["capture"]
    assert capture == {
        "kimi": 5,  # labels 0-2, X, W
        "audit": 7,  # labels 1-3, X, W, Y, Z
        "both": 4,  # labels 1-2, X, W
        "scripted": n,
        "scripted_kimi": 3,
        "scripted_audit": 3,
        "scripted_neither": n - 4,
        "unscripted_found": 4,  # X, W, Y, Z
    }
    chapman = max(0.0, 6 * 8 / 5 - 1 - (5 + 7 - 4) - (n - 4))
    miss = (1 - 3 / n) ** 2
    recall = 4 * miss / (1 - miss)
    unseen = report["residual"]["unseen_by_both"]
    assert unseen["chapman"] == pytest.approx(round(chapman, 3))
    assert unseen["recall"] == pytest.approx(round(recall, 3))
    known = 2  # Y and Z (W is labeled by the audit's claim)
    residual = known + max(chapman, recall)
    labeled = n + 2
    assert report["residual"]["known_unlabeled"] == known
    assert report["residual"]["estimated_unlabeled"] == pytest.approx(round(residual, 3))
    gate = next(g for g in report["gates"] if g["gate"] == "residual_unlabeled_share")
    assert gate["value"] == pytest.approx(round(residual / labeled, 4))
    assert gate["threshold"] == 0.03 and gate["result"] == (
        "PASS" if residual / labeled <= 0.03 else "FAIL"
    )
    assert gate["note"].endswith("not a whole world")


def test_pilot_gate_precision_and_total_error(
    world, world_plan, call_doc, call_plan, tmp_path
) -> None:
    teacher, report = _two_teacher_run(world, world_plan, call_doc, call_plan, tmp_path)
    gates = {g["gate"]: g for g in report["gates"]}
    assert gates["silver_precision_min_teacher"]["result"] == "PENDING"
    assert gates["total_label_error"]["result"] == "PENDING"
    sample = [
        json.loads(line) for line in (tmp_path / "precision_sample.jsonl").read_text().splitlines()
    ]
    assert sorted(x["teacher_model"] for x in sample) == sorted(
        [TEACHER_MODEL, pick_audit_teacher(call_doc.writer_model)]
    )
    reviews = [
        {k: x[k] for k in ("plan_id", "doc_id", "char_start", "char_end", "teacher_model")}
        | {"correct": x["teacher_model"] == TEACHER_MODEL}
        for x in sample
    ]
    (tmp_path / "precision_review.jsonl").write_text("".join(json.dumps(r) + "\n" for r in reviews))
    calls = len(teacher.seen)
    again = run_completeness(
        _ctx(world, world_plan, teacher), {call_plan.id: call_doc}, out_dir=tmp_path
    )
    assert len(teacher.seen) == calls
    by_teacher = again["precision"]["by_teacher"]
    assert by_teacher[TEACHER_MODEL]["precision"] == 1.0
    assert by_teacher[pick_audit_teacher(call_doc.writer_model)]["precision"] == 0.0
    gates = {g["gate"]: g for g in again["gates"]}
    assert (
        gates["silver_precision_min_teacher"]["value"],
        gates["silver_precision_min_teacher"]["result"],
    ) == (0.0, "FAIL")
    residual = again["residual"]["estimated_unlabeled"]
    labeled = again["residual"]["labeled_facts"]
    assert again["precision"]["estimated_wrong_labels"] == 1.0
    assert gates["total_label_error"]["value"] == pytest.approx(round((residual + 1) / labeled, 4))
    assert gates["total_label_error"]["result"] == "FAIL"


# --------------------------------------------------------------------------
# QA report
# --------------------------------------------------------------------------


def test_qa_report_gates_and_residual(
    world, world_plan, call_doc, call_plan, email_doc, email_plan, tmp_path
) -> None:
    confirmed = [
        {"type": lab.fact_type.value, "claim": "fixture", "quote": lab.text}
        for lab in call_doc.labels[:2]
    ] + [
        {
            "type": email_doc.labels[0].fact_type.value,
            "claim": "fixture",
            "quote": email_doc.labels[0].text,
        }
    ]
    candidate = {
        "type": "request",
        "claim": "asked for fixture closing words",
        "quote": "fixture closing words",
    }
    teacher = FakeTeacher(
        lambda m, env, n: [
            r
            for r in [*confirmed, candidate]
            if r["quote"] in env.text and (r is not candidate or env.source_id == call_plan.id)
        ]
    )
    report = run_completeness(
        _ctx(world, world_plan, teacher),
        {call_plan.id: call_doc, email_plan.id: email_doc},
        out_dir=tmp_path,
        audit_share=0.0,
    )
    scripted = len(call_doc.labels) + len(email_doc.labels)
    r = 3 / scripted
    assert report["teacher"]["recall_r"] == pytest.approx(round(r, 4))
    labeled = scripted + 1
    residual = 1 * (1 - r) / r
    assert report["residual"]["kimi_only"]["u_total"] == 1
    assert report["residual"]["estimated_unlabeled"] == pytest.approx(round(residual, 3))
    gates = {g["gate"]: g for g in report["gates"]}
    assert gates["residual_unlabeled_share"]["value"] == pytest.approx(round(residual / labeled, 4))
    assert gates["silver_share_of_labeled_facts"]["value"] == pytest.approx(round(1 / labeled, 4))
    assert gates["teacher_recall_on_scripted"]["passed"] is (r >= 0.85)
    assert gates["scripted_facts_verified"]["value"] == 1.0
    assert gates["documents_regenerated"]["value"] == 0.0
    assert gates["audit_new_share"]["passed"] is None  # nothing audited
    assert report["accepted"] is False
    [decoy] = call_doc.decoy_spans
    assert report["decoys"][decoy.decoy_class_id] == {
        "in_text": 1,
        "hit": 0,
        "rate": 0.0,
        "hand_review": False,
    }


# --------------------------------------------------------------------------
# Spend, budget, resume, errors
# --------------------------------------------------------------------------


def test_spend_lands_on_synthetic_v2_per_model(
    world, world_plan, call_doc, call_plan, tmp_path
) -> None:
    meter = SpendMeter(tmp_path / "spend.json")
    teacher = FakeTeacher(quoting(), cost=0.0025)
    run_completeness(
        _ctx(world, world_plan, teacher, meter=meter),
        {call_plan.id: call_doc},
        out_dir=tmp_path / "out",
        audit_share=1.0,
    )
    snap = meter.snapshot()
    assert snap["per_corpus_usd"][CORPUS_KEY] == pytest.approx(0.0025 * len(teacher.seen))
    assert snap["per_model_usd"][TEACHER_MODEL] == pytest.approx(0.0025)
    assert CORPUS_KEY == "synthetic_v2"


def test_a_client_that_meters_its_calls_is_not_recorded_twice(
    world, world_plan, call_doc, call_plan, tmp_path
) -> None:
    meter = SpendMeter(tmp_path / "spend.json")
    teacher = FakeTeacher(quoting(), cost=0.0025)
    ctx = _ctx(world, world_plan, teacher, meter=meter)
    ctx.client_meters = True
    run_completeness(ctx, {call_plan.id: call_doc}, out_dir=tmp_path / "out", audit_share=1.0)
    assert meter.corpus_usd(CORPUS_KEY) == 0
    assert ctx.budget.spent_usd == pytest.approx(0.0025 * len(teacher.seen))


def test_budget_stop_is_checked_before_the_call(
    world, world_plan, call_doc, call_plan, tmp_path
) -> None:
    teacher = FakeTeacher(quoting())
    report = run_completeness(
        _ctx(world, world_plan, teacher, cap=0.0001), {call_plan.id: call_doc}, out_dir=tmp_path
    )
    assert teacher.seen == []
    assert PAID_BUDGET_STOP_MARKER in (report["stop_reason"] or "")
    assert (tmp_path / "qa_report.json").exists()


def test_resume_makes_no_call_for_a_finished_document(
    world, world_plan, call_doc, call_plan, negative_doc, negative_plan, tmp_path
) -> None:
    teacher = FakeTeacher(quoting(BREAKING, until=1))
    docs = {call_plan.id: call_doc, negative_plan.id: negative_doc}
    regen = FakeRegenerator(lambda p: docs[p].model_copy(update={"writer_model": OTHER_WRITER}))
    first = run_completeness(
        _ctx(world, world_plan, teacher, regenerator=regen), docs, out_dir=tmp_path
    )
    calls = len(teacher.seen)
    labels = (tmp_path / "labels.jsonl").read_text()
    second = run_completeness(
        _ctx(world, world_plan, teacher, regenerator=regen), docs, out_dir=tmp_path
    )
    assert len(teacher.seen) == calls and len(regen.seen) == 1
    assert (tmp_path / "labels.jsonl").read_text() == labels
    assert first["per_document"] == second["per_document"]


def test_new_pass_version_reuses_the_teachers_first_pass(
    world, world_plan, call_doc, call_plan, tmp_path, monkeypatch
) -> None:
    from corpus.generator.v2 import completeness

    teacher = FakeTeacher(quoting(CONSISTENT))
    first = run_completeness(
        _ctx(world, world_plan, teacher), {call_plan.id: call_doc}, out_dir=tmp_path, audit_share=0
    )
    assert len(teacher.seen) == 1 and first["spend"]["teacher_reused_usd"] == 0
    monkeypatch.setattr(completeness, "PASS_VERSION", "wp5-test-next")
    second = run_completeness(
        _ctx(world, world_plan, teacher), {call_plan.id: call_doc}, out_dir=tmp_path, audit_share=0
    )
    assert len(teacher.seen) == 1  # no new call: the stored output was re-gated
    assert second["spend"]["run_usd"] == 0
    assert second["spend"]["teacher_reused_usd"] == pytest.approx(teacher.cost)
    assert second["silver"]["labels"] == first["silver"]["labels"] == 1
    state = json.loads((tmp_path / "docs" / f"{call_plan.id}.json").read_text())
    assert state["attempts"][0]["calls"][0]["reused"] is True
    # A different teacher input (another prompt) pays again.
    ctx = _ctx(world, world_plan, teacher)
    ctx.company_context = None
    run_completeness(ctx, {call_plan.id: call_doc}, out_dir=tmp_path, audit_share=0)
    assert len(teacher.seen) == 2


def test_new_pass_version_reuses_the_audit_too(
    world, world_plan, call_doc, call_plan, tmp_path, monkeypatch
) -> None:
    from corpus.generator.v2 import completeness

    teacher = FakeTeacher(quoting(audit=[CONSISTENT]))
    first = run_completeness(
        _ctx(world, world_plan, teacher), {call_plan.id: call_doc}, out_dir=tmp_path
    )
    assert len(teacher.seen) == 2 and first["audit"]["facts_grounded"] == 1
    monkeypatch.setattr(completeness, "PASS_VERSION", "wp5-test-next")
    second = run_completeness(
        _ctx(world, world_plan, teacher), {call_plan.id: call_doc}, out_dir=tmp_path
    )
    assert len(teacher.seen) == 2 and second["spend"]["run_usd"] == 0
    assert second["audit"]["facts_grounded"] == 1
    assert second["residual"]["audit_only_unlabeled"] == 1


def test_no_audit_labels_on_hard_negatives(
    world, world_plan, negative_doc, negative_plan, tmp_path
) -> None:
    also = {"type": "praise", "claim": "said fixture opening words", "quote": "fixture opening"}
    teacher = FakeTeacher(quoting(audit=[also]))
    report = run_completeness(
        _ctx(world, world_plan, teacher), {negative_plan.id: negative_doc}, out_dir=tmp_path
    )
    assert report["documents"]["by_status"] == {"clean_negative": 1}
    assert report["audit"]["labels_added"] == 0
    assert report["residual"]["audit_only_unlabeled"] == 1


def test_resume_after_a_stop_does_not_repay_finished_steps(
    world, world_plan, call_doc, call_plan, tmp_path
) -> None:
    teacher = FakeTeacher(quoting(BREAKING, until=1))
    stopping = FakeRegenerator(lambda _p: None, raise_budget=True)
    report = run_completeness(
        _ctx(world, world_plan, teacher, regenerator=stopping),
        {call_plan.id: call_doc},
        out_dir=tmp_path,
    )
    assert report["stop_reason"] and report["documents"]["by_status"] == {"in_progress": 1}
    assert len(teacher.seen) == 1
    working = FakeRegenerator(lambda _p: call_doc.model_copy(update={"writer_model": OTHER_WRITER}))
    report = run_completeness(
        _ctx(world, world_plan, teacher, regenerator=working),
        {call_plan.id: call_doc},
        out_dir=tmp_path,
        audit_share=0.0,
    )
    assert len(teacher.seen) == 2  # only the regenerated version was extracted
    assert report["documents"]["by_status"] == {"accepted": 1}


def test_teacher_failure_retries_once_then_errors_and_reruns(
    world, world_plan, call_doc, call_plan, tmp_path
) -> None:
    meter = SpendMeter(tmp_path / "spend.json")
    teacher = FakeTeacher(quoting(), fail_models={TEACHER_MODEL})
    report = run_completeness(
        _ctx(world, world_plan, teacher, meter=meter),
        {call_plan.id: call_doc},
        out_dir=tmp_path / "o",
    )
    assert len(teacher.seen) == 2 and report["documents"]["by_status"] == {"error": 1}
    assert meter.corpus_usd(CORPUS_KEY) == pytest.approx(2 * teacher.cost)
    teacher.fail_models.clear()
    report = run_completeness(
        _ctx(world, world_plan, teacher, meter=meter),
        {call_plan.id: call_doc},
        out_dir=tmp_path / "o",
    )
    assert report["documents"]["by_status"] == {"accepted": 1}


def test_holdout_output_is_refused(world, world_plan, call_doc, call_plan, tmp_path) -> None:
    with pytest.raises(RuntimeError, match="hold-out"):
        run_completeness(
            _ctx(world, world_plan, FakeTeacher(quoting())),
            {call_plan.id: call_doc},
            out_dir=tmp_path / "corpus" / "out" / "wp5",
        )


def test_stale_wp4_input_is_skipped(world, world_plan, call_doc, call_plan, tmp_path) -> None:
    teacher = FakeTeacher(quoting())
    report = run_completeness(
        _ctx(world, world_plan, teacher),
        {call_plan.id: call_doc},
        out_dir=tmp_path,
        fingerprint_ok=lambda _d, _p: False,
    )
    assert teacher.seen == [] and report["documents"]["skipped"][call_plan.id].startswith("stale")


# --------------------------------------------------------------------------
# Regeneration through WP4's writer
# --------------------------------------------------------------------------


def test_writer_regenerator_sends_do_not_state_and_avoids_the_writer(
    world, world_plan, pack, call_plan
) -> None:
    backend = ScriptedBackend(compact_obedient)
    config = WriterConfig(models=DEFAULT_ROTATION)
    session = WriterSession(
        backend=backend,
        pool=WriterPool(config.models, world_doc_total=300),
        budget=TeacherRunBudget(cap_usd=5.0),
        config=config,
    )
    regen = WriterRegenerator(world, world_plan, pack, session)
    avoided = DEFAULT_ROTATION[0]
    doc = regen.regenerate(call_plan.id, ["We hit 98% compliance last quarter."], {avoided})
    assert doc is not None and doc.status == "written"
    assert backend.calls[0].model != avoided and doc.writer_model != avoided
    assert "DO NOT STATE" in backend.calls[0].user
    assert '- "We hit 98% compliance last quarter."' in backend.calls[0].user
    assert session.pool.doc_counts[doc.writer_model] == 1
    # One cap: the writer's calls land on the pass's own budget.
    assert session.budget.spent_usd == pytest.approx(backend.cost * len(backend.calls))


def test_wp4_prompt_is_unchanged_without_do_not_state(
    world, world_plan, pack, decoys, call_plan
) -> None:
    backend = ScriptedBackend(compact_obedient)
    config = WriterConfig(models=DEFAULT_ROTATION)
    session = WriterSession(
        backend=backend,
        pool=WriterPool(config.models, world_doc_total=300),
        budget=RunBudget(cap_usd=5.0),
        config=config,
    )
    write_document(
        render(call_plan, world, decoys.world_plan, pack),
        call_plan,
        world,
        pack,
        decoys_by_id=decoys,
        session=session,
    )
    assert all("DO NOT STATE" not in c.user for c in backend.calls)


def test_call_log_records_the_fact_count(world, world_plan, call_doc, call_plan, tmp_path) -> None:
    label = _label(call_doc)
    teacher = FakeTeacher(quoting({"type": "usage", "claim": "usage fixture", "quote": label.text}))
    ctx = _ctx(world, world_plan, teacher)
    ctx.call_log = tmp_path / "teacher_calls.jsonl"
    process_document(ctx, call_doc, call_plan)
    [row] = [json.loads(line) for line in ctx.call_log.read_text().splitlines()]
    assert (row["facts"], row["rejected"], row["cost_usd"]) == (1, 0, teacher.cost)


# --------------------------------------------------------------------------
# The decoy rule and the label judge (`01-spec/label-judge-2026-09-22.md`)
# --------------------------------------------------------------------------

JUDGE = "deepseek/deepseek-v4.1-flash"
CLOSING = {
    "type": "request",
    "claim": "asked for fixture closing words",
    "quote": "fixture closing words",
}


def _answers(rows: list[dict[str, Any]]):
    """A stub judge: `rows` gives answers by claim; any other fact is kept.
    Every call's user message is kept in `.calls`."""
    calls: list[str] = []

    def call(model: str, system: str, user: str):
        calls.append(user)
        facts = json.loads(user.split("## Facts to check\n", 1)[1])
        out = []
        for f in facts:
            row = next((r for r in rows if r["claim"] == f["claim"]), {})
            out.append(
                {
                    "id": f["id"],
                    "reason": row.get("reason", "stated"),
                    "stated_as_true": True,
                    "durable": row.get("durable", True),
                    "claim_supported": True,
                    "repeats": None,
                    "best_type": row.get("best_type", f["fact_type"]),
                }
            )
        return {"facts": out}, 0.0002

    call.calls = calls  # type: ignore[attr-defined]
    return call


def test_decoy_rule_drops_a_fact_that_only_touches_a_decoy(
    world, world_plan, call_doc, call_plan
) -> None:
    [decoy] = call_doc.decoy_spans
    text = next(p.text for p in call_doc.prepared_sources if p.source_id == decoy.doc_id)
    last_two = " ".join(decoy.text.split()[-2:])
    start = decoy.char_end - len(last_two)
    end = text.index("fixture closing words", decoy.char_end) + len("fixture closing words")
    assert not spans_match((start, end), (decoy.char_start, decoy.char_end))
    teacher = FakeTeacher(quoting({**CLOSING, "quote": text[start:end]}))
    state = process_document(_ctx(world, world_plan, teacher), call_doc, call_plan)
    [f] = state.attempts[0].usable()
    assert (f.triage, f.matched_id) == ("decoy_hit", decoy.decoy_id)
    assert state.silver_labels == []
    assert decoy.decoy_id in state.attempts[0].decoys_in_text


def test_unclosed_decoy_marker_is_still_a_decoy(world, world_plan, call_doc, call_plan) -> None:
    [decoy] = call_doc.decoy_spans
    closer = f"⟦/{decoy.decoy_id}⟧"
    first = call_doc.calls[0]
    assert closer in first.raw_output
    outcomes = [
        o.model_copy(update={"status": "dropped", "dropped_text": None})
        if o.marker_id == decoy.decoy_id
        else o
        for o in call_doc.slot_outcomes
    ]
    broken = call_doc.model_copy(
        update={
            "calls": [
                first.model_copy(update={"raw_output": first.raw_output.replace(closer, "")})
            ],
            "slot_outcomes": outcomes,
            "decoy_spans": [],
        }
    )
    decoys_by_id = {d.id: d for d in world_plan.decoys}
    sources = source_inputs(broken, call_plan)
    [recovered] = [
        s for s in scripted_spans(broken, sources, world, decoys_by_id) if s.kind == "decoy"
    ]
    assert (recovered.marker_id, recovered.verified) == (decoy.decoy_id, False)
    assert recovered.start == decoy.char_start
    teacher = FakeTeacher(quoting({"type": "risk", "claim": "a risk", "quote": decoy.text}))
    state = process_document(_ctx(world, world_plan, teacher), broken, call_plan)
    [f] = state.attempts[0].usable()
    assert (f.triage, f.decoy_verified) == ("decoy_hit", False)


def test_unmatched_marker_text_reads_either_half() -> None:
    raw = "Ann: hi\nBo: ⟦D00001⟧ I heard another hospital switched vendors last month.\nAnn: ok"
    assert unmatched_marker_text(raw, "D00001") == [
        "I heard another hospital switched vendors last month."
    ]
    closer_only = "Bo: rumor has it their rollout stalled badly⟦/D00002⟧ tail"
    assert unmatched_marker_text(closer_only, "D00002") == [
        "Bo: rumor has it their rollout stalled badly"
    ]
    assert unmatched_marker_text("⟦D00003⟧ too short\n", "D00003") == []
    assert unmatched_marker_text("no markers here", "D00004") == []


def test_label_judge_drops_and_retypes_silver_labels(
    world, world_plan, call_doc, call_plan, tmp_path
) -> None:
    drop = {
        "type": "complaint",
        "claim": "said fixture reply words",
        "quote": "fixture reply words",
    }
    retype = {
        "type": "praise",
        "claim": "said fixture opening words",
        "quote": "fixture opening words",
    }
    judge = _answers(
        [
            {"claim": drop["claim"], "durable": False, "reason": "small talk"},
            {"claim": retype["claim"], "best_type": "other"},
        ]
    )
    ctx = _ctx(world, world_plan, FakeTeacher(quoting(CLOSING, drop, retype)))
    ctx.label_judge = LabelJudgeSettings(
        model=JUDGE, call=judge, log_path=tmp_path / "label_judge.jsonl"
    )
    report = run_completeness(ctx, {call_plan.id: call_doc}, out_dir=tmp_path)
    [sent] = judge.calls
    claims = sorted(f["claim"] for f in json.loads(sent.split("## Facts to check\n", 1)[1]))
    assert claims == sorted(r["claim"] for r in (CLOSING, drop, retype))  # silver labels only
    silver = [
        json.loads(line) for line in (tmp_path / "silver_labels.jsonl").read_text().splitlines()
    ]
    assert {r["text"]: (r["fact_type"], r["claim"], r["teacher_fact_type"]) for r in silver} == {
        "fixture closing words": ("request", CLOSING["claim"], None),
        "fixture opening words": ("other", retype["claim"], "praise"),
    }
    assert "fixture reply words" not in (tmp_path / "labels.jsonl").read_text()
    section = report["label_judge"]
    assert section["verdicts"] == {"keep": 1, "drop": 1, "retype": 1}
    assert [d["reason"] for d in section["dropped"]] == ["small talk"]
    assert section["documents_not_fully_judged"] == 0
    assert len((tmp_path / "label_judge.jsonl").read_text().splitlines()) == 1

    # A rerun on the same output pays nothing: the document is already judged.
    again = run_completeness(ctx, {call_plan.id: call_doc}, out_dir=tmp_path)
    assert len(judge.calls) == 1 and again["label_judge"]["verdicts"]["drop"] == 1


def test_label_judge_that_drops_every_silver_label_leaves_a_clean_negative(
    world, world_plan, negative_doc, negative_plan, tmp_path
) -> None:
    consistent = {**BREAKING, "entities": []}
    ctx = _ctx(world, world_plan, FakeTeacher(quoting(consistent)))
    ctx.label_judge = LabelJudgeSettings(
        model=JUDGE, call=_answers([{"claim": consistent["claim"], "durable": False}])
    )
    run_completeness(ctx, {negative_plan.id: negative_doc}, out_dir=tmp_path)
    state = json.loads((tmp_path / "docs" / f"{negative_plan.id}.json").read_text())
    assert state["status"] == "clean_negative" and state["silver_labels"] == []
    assert len(state["judge"]["dropped"]) == 1


def test_label_judge_without_a_verdict_leaves_the_label_and_says_so(
    world, world_plan, call_doc, call_plan, tmp_path
) -> None:
    ctx = _ctx(world, world_plan, FakeTeacher(quoting(CLOSING)))
    ctx.label_judge = LabelJudgeSettings(model=JUDGE, call=_answers([]), budget_left=lambda: False)
    report = run_completeness(ctx, {call_plan.id: call_doc}, out_dir=tmp_path)
    assert report["label_judge"]["verdicts"] == {"unresolved": 1}
    assert report["label_judge"]["documents_not_fully_judged"] == 1
    assert "fixture closing words" in (tmp_path / "silver_labels.jsonl").read_text()


def test_a_decoy_that_rules_out_a_competitor_is_labeled_as_a_fact(
    world, world_plan, pack, decoys
) -> None:
    """Decisions log, 2026-09-23: a negated competitor statement is a
    `competitor_mention` fact. A negation decoy WP4 dropped for saying one
    is no decoy wording, so the teacher's fact on it becomes a label."""
    plan, did = next(
        (d, x)
        for d in sorted(world_plan.doc_plans, key=lambda d: (d.date, d.id))
        if d.kind == DocKind.CALL and d.fact_ids
        for x in d.decoy_ids
        if decoys[x].decoy_class_id == "negation"
    )
    sentence = "honestly we are not looking at another pump vendor right now"

    def writer(model, system, user, n):  # noqa: ANN001
        if "REWRITE THESE UNITS" in user:
            return repair_reply(user)
        return _compact(
            render_document(system, user, mutate={did: lambda item, _old: (sentence, None)}),
            user,
        )

    doc = _write(plan, world, pack, decoys, reply=writer)
    outcome = next(o for o in doc.slot_outcomes if o.marker_id == did)
    assert outcome.status == "dropped" and sentence in (outcome.dropped_text or "")
    spans = scripted_spans(doc, source_inputs(doc, plan), world, decoys)
    assert did not in {s.marker_id for s in spans if s.kind == "decoy"}

    claim = "The customer is not looking at another pump vendor"
    teacher = FakeTeacher(
        quoting({"type": "competitor_mention", "claim": claim, "quote": sentence})
    )
    state = process_document(_ctx(world, world_plan, teacher), doc, plan)
    [fact] = [f for f in state.attempts[0].usable() if f.quote == sentence]
    assert fact.triage == "candidate"
    [label] = [x for x in state.silver_labels if x.claim == claim]
    assert label.fact_type.value == "competitor_mention" and label.text == sentence
