"""Tests for world generator v2 work package 7, the exporter
(`corpus/generator/v2/export.py`; `01-spec/world-generator-v2-2026-09-22.md`
section 1 "Output per world", section 6, section 9's WP7 row and the WP1
build note's per-type cap). Every model call is stubbed.

Documents come from WP4's writer driven by the obedient stub writer, WP5's
completeness pass runs with a stub teacher that returns a full production
answer (summary, sentiment, confidence), and WP6's noise stage runs its code
operations. The exporter's outputs are then read back the way their
consumers read them: WP9's `load_world`, the SFT formatter, and the
production extractor's parser.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from corpus.generator.v2 import export as wp7
from corpus.generator.v2.completeness import (
    SHIPPED_STATUSES,
    TEACHER_MODEL,
    CompletenessContext,
    TeacherRunBudget,
    regate,
    run_completeness,
    source_inputs,
    world_company_context,
)
from corpus.generator.v2.models import Split
from corpus.generator.v2.noise import (
    OP_ASR,
    OP_MISFILED,
    NoiseConfig,
    apply_noise,
    noise_completed_world,
    write_outputs,
)
from corpus.generator.v2.packs.schema import VerticalPack, load_pack
from corpus.generator.v2.planner import DocKind, DocPlan, DocRole, WorldDocPlan, plan_world
from corpus.generator.v2.renderers import render
from corpus.generator.v2.simulator import simulate_world
from corpus.generator.v2.writer import (
    DEFAULT_ROTATION,
    RunBudget,
    WriterConfig,
    WriterPool,
    WriterSession,
    WrittenDoc,
    write_document,
)
from portolan.bench.world_v2 import load_world
from portolan.extract.backends import LLMClientBackend
from portolan.extract.prompt import (
    DEFAULT_PROMPT_VARIANT,
    build_system_prompt,
    build_user_message,
    system_prompt_sha256,
)
from portolan.extract.schema import Coverage, ExtractionResult, Fact
from portolan.llm.fake import FakeLLMClient
from portolan.metering import InMemorySink, metered
from portolan.training.records import read_records
from portolan.training.sft_format import (
    LegacyRecordError,
    envelope_for_record,
    production_sft_messages_for_record,
)
from tests.v2_writer_fakes import ScriptedBackend, render_document, repair_reply

REPO_ROOT = Path(__file__).resolve().parents[1]
PACK_PATH = REPO_ROOT / "corpus" / "generator" / "v2" / "packs" / "medical_devices.yaml"
NOW = datetime(2026, 9, 23, 12, 0, tzinfo=UTC)
# Lowercase fixture words the stub writer puts in every call: a quote with
# no name or number, so the stub teacher can state an unscripted fact that
# WP5 labels as silver without any world check objecting.
SILVER_QUOTE = "fixture reply words"


# --------------------------------------------------------------------------
# Fixtures: a small world written by WP4's stub writer
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pack() -> VerticalPack:
    return load_pack(PACK_PATH)


@pytest.fixture(scope="module")
def world(pack):
    return simulate_world(pack, seed=42, months=8, split=Split.TRAIN)


@pytest.fixture(scope="module")
def world_plan(world, pack) -> WorldDocPlan:
    # Uncapped: the documents these tests pick keep every planted fact type
    # (the plan-time cap is WP3's, tested there; the export-time cap below).
    return plan_world(world, pack, type_cap=None)


def _compact(text: str, user: str) -> str:
    """Keep the stub writer's padding short, so a call is one envelope."""
    m = re.search(r"^LENGTH: about (\d+) words", user, re.MULTILINE)
    extra = len(text.split()) - int(0.6 * int(m.group(1))) if m else 0
    if extra <= 0:
        return text
    return re.sub(
        r"closing words((?: fixture)+)",
        lambda g: "closing words" + " fixture" * max(0, len(g.group(1).split()) - extra),
        text,
    )


def _writer(model: str, system: str, user: str, n: int) -> str:
    if "REWRITE THESE UNITS" in user:
        return repair_reply(user)
    return _compact(render_document(system, user), user)


def _first(
    world_plan: WorldDocPlan, kinds, *, facts: int = 0, role: DocRole = DocRole.EVENT
) -> DocPlan:
    for d in sorted(world_plan.doc_plans, key=lambda d: (d.date, d.id)):
        if d.kind in kinds and d.role == role and len(d.fact_ids) >= facts:
            return d
    raise LookupError(kinds)


@pytest.fixture(scope="module")
def plans(world_plan) -> dict[str, DocPlan]:
    chosen = [
        _first(world_plan, {DocKind.CALL}, facts=3),
        _first(world_plan, {DocKind.EMAIL}, facts=1),
        _first(world_plan, set(DocKind), role=DocRole.HARD_NEGATIVE),
    ]
    return {p.id: p for p in chosen}


@pytest.fixture(scope="module")
def written(plans, world, world_plan, pack) -> dict[str, WrittenDoc]:
    decoys = {d.id: d for d in world_plan.decoys}
    out: dict[str, WrittenDoc] = {}
    for plan in plans.values():
        config = WriterConfig(models=DEFAULT_ROTATION)
        doc = write_document(
            render(plan, world, world_plan, pack),
            plan,
            world,
            pack,
            decoys_by_id=decoys,
            session=WriterSession(
                backend=ScriptedBackend(_writer),
                pool=WriterPool(config.models, world_doc_total=300),
                budget=RunBudget(cap_usd=5.0),
                config=config,
            ),
        )
        assert doc.status == "written", (plan.id, doc.reason)
        out[plan.id] = doc
    return out


# --------------------------------------------------------------------------
# Stub teacher: confirms every scripted span with a full production answer
# --------------------------------------------------------------------------


@dataclass
class StubTeacher:
    """The labeling teacher states every WP4 label in the envelope (same
    type, the span's words as its claim) plus, when `silver` is on, one
    unscripted fact on `SILVER_QUOTE` in calls. The audit teacher finds
    nothing. Every answer carries a summary, sentiment and confidence."""

    labels: dict[str, list[tuple[str, str]]]  # source id -> [(type, span text)]
    silver: bool = False
    seen_users: list[str] = field(default_factory=list)

    def factory(self, model: str):
        teacher = self

        class _Backend:
            def extract(self, envelope, system_prompt, sink, tenant):  # noqa: ANN001
                teacher.seen_users.append(build_user_message(envelope))
                with metered(sink, tenant, "extract", model) as usage:
                    usage.cost_override = 0.0
                facts: list[Fact] = []
                if model == TEACHER_MODEL:
                    for i, (fact_type, text) in enumerate(
                        teacher.labels.get(envelope.source_id, [])
                    ):
                        if text in envelope.text:
                            facts.append(
                                Fact(
                                    id=f"f{i}",
                                    type=fact_type,
                                    claim=text,
                                    quote=text,
                                    locator="L1",
                                    entities=["Alderline"] if "Alderline" in text else [],
                                    severity="medium" if fact_type == "risk" else None,
                                    sentiment=-1 if fact_type in ("risk", "complaint") else 0,
                                    confidence=0.85,
                                )
                            )
                    if teacher.silver and SILVER_QUOTE in envelope.text:
                        facts.append(
                            Fact(
                                id="s1",
                                type="other",
                                claim="Someone said the fixture reply words.",
                                quote=SILVER_QUOTE,
                                locator="L1",
                                sentiment=0,
                                confidence=0.6,
                            )
                        )
                return ExtractionResult(
                    source_id=envelope.source_id,
                    summary=f"stub summary of {envelope.source_id}",
                    facts=facts,
                    coverage=Coverage(segments_read=1, segments_total=1),
                )

        return _Backend()


def _teacher(written: dict[str, WrittenDoc], *, silver: bool = False) -> StubTeacher:
    labels: dict[str, list[tuple[str, str]]] = {}
    for doc in written.values():
        for x in doc.labels:
            labels.setdefault(x.doc_id, []).append((x.fact_type.value, x.text))
    return StubTeacher(labels, silver=silver)


def _stage_world(
    wdir: Path,
    written: dict[str, WrittenDoc],
    world,
    world_plan,
    *,
    silver: bool = False,
    noise_ops: frozenset[str] | None = None,
    noise_after_completeness: bool = False,
) -> StubTeacher:
    """WP4's documents under `writer/`, WP5's pass under `completeness/`, and
    (when `noise_ops`) WP6's code noise under `noise/`: over WP4's output, or
    over WP5's shipped documents and labels as the runner orders it."""
    for doc in written.values():
        path = wdir / "writer" / "docs" / f"{doc.plan_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(doc.model_dump_json(), encoding="utf-8")
    teacher = _teacher(written, silver=silver)
    ctx = CompletenessContext(
        world=world,
        world_plan=world_plan,
        backend_factory=teacher.factory,
        budget=TeacherRunBudget(cap_usd=5.0),
        company_context=world_company_context(world),
        sleep=lambda _s: None,
    )
    run_completeness(ctx, dict(written), out_dir=wdir / "completeness")
    if noise_ops is not None:
        config = NoiseConfig(seed=3, force=noise_ops)
        if noise_after_completeness:
            states = wp7.load_stage_outputs(wdir / "writer", wdir / "completeness").states or {}
            run = noise_completed_world(
                world,
                world_plan,
                dict(written),
                states,
                shipped=SHIPPED_STATUSES,
                config=config,
            )
        else:
            run = apply_noise(world, world_plan, dict(written), config=config)
        write_outputs(run, wdir / "noise")
    return teacher


def _export(wdir: Path, world, world_plan, *, split: Split = Split.TRAIN, **kw) -> dict[str, Any]:
    return wp7.export_world(
        world,
        world_plan,
        split=split,
        pack="medical_devices",
        seed=42,
        world_dir=wdir,
        now=lambda: NOW,
        **kw,
    )


@pytest.fixture(scope="module")
def exported(tmp_path_factory, written, world, world_plan):
    wdir = tmp_path_factory.mktemp("wp7") / "train" / "world-medical_devices-42"
    teacher = _stage_world(wdir, written, world, world_plan, silver=True)
    manifest = _export(wdir, world, world_plan)
    return wdir, manifest, teacher


# --------------------------------------------------------------------------
# WP5 keeps the teacher's whole answer
# --------------------------------------------------------------------------


def test_wp5_keeps_the_teacher_answer_and_fact_values(
    exported, world, world_plan, plans, written
) -> None:
    wdir, _, _ = exported
    call_id = next(p for p, plan in plans.items() if plan.kind == DocKind.CALL)
    state = json.loads((wdir / "completeness" / "docs" / f"{call_id}.json").read_text())
    [call] = [c for c in state["attempts"][-1]["calls"] if c["role"] == "completeness"]
    assert call["answer"]["summary"] == f"stub summary of {call_id}"
    assert call["answer"]["coverage"] == {"segments_read": 1, "segments_total": 1}
    scripted = state["scripted_labels"]
    assert scripted and all(x["claim_source"] == "teacher" for x in scripted)
    assert all(x["teacher_fields"]["confidence"] == 0.85 for x in scripted)
    risk = [x for x in scripted if x["fact_type"] == "risk"]
    assert risk and risk[0]["teacher_fields"]["severity"] == "medium"
    [silver] = [x for x in state["silver_labels"] if x["text"] == SILVER_QUOTE]
    assert silver["teacher_fields"]["confidence"] == 0.6
    # A stored pass re-gated later (a new pass version) keeps the values.
    from corpus.generator.v2.completeness import TeacherFact

    stored = [TeacherFact.model_validate(f) for f in state["attempts"][-1]["facts"]]
    ctx = CompletenessContext(
        world=world,
        world_plan=world_plan,
        backend_factory=lambda m: None,
        budget=TeacherRunBudget(cap_usd=1.0),
        company_context=world_company_context(world),
    )
    again = regate(ctx, stored, source_inputs(written[call_id], plans[call_id]))
    assert [f.answer_fields for f in again] == [f.answer_fields for f in stored]
    assert all(f.answer_fields and f.answer_fields["confidence"] for f in again)


# --------------------------------------------------------------------------
# The world export
# --------------------------------------------------------------------------


def test_export_writes_a_world_the_bench_reads_with_no_excluded_label(exported) -> None:
    wdir, manifest, _ = exported
    for name in ("sources.jsonl", "labels.jsonl", "decoys.jsonl", "qa_report.json"):
        assert (wdir / name).exists(), name
    world = load_world(wdir)
    assert world.input_format == "sources.jsonl"
    assert (
        world.excluded_labels == {} and world.excluded_decoys == {} and world.skipped_sources == {}
    )
    labels = [json.loads(x) for x in (wdir / "labels.jsonl").read_text().splitlines()]
    assert len(world.labels) == len(labels) == manifest["counts"]["labels"] > 0
    sources = {x["label_source"] for x in labels}
    assert {"scripted_code", "scripted_writer", "silver_completeness"} <= sources
    assert all(x["world_id"] == "world-medical_devices-42" for x in labels)
    assert manifest["world_id"] == "world-medical_devices-42" and manifest["split"] == "train"
    assert (manifest["pack"], manifest["seed"]) == ("medical_devices", 42)
    assert manifest["counts"]["labels_by_fact_type"]
    assert manifest["counts"]["labels_by_label_source"]["silver_completeness"] >= 1


def test_manifest_hashes_every_file_it_wrote(exported) -> None:
    wdir, manifest, _ = exported
    files = manifest["files"]
    for rel in ("sources.jsonl", "labels.jsonl", "decoys.jsonl", "training_records.jsonl"):
        assert rel in files
    assert "files/crm/accounts.csv" in files
    assert any(k.startswith("files/") and k != "files/crm/accounts.csv" for k in files)
    for rel, digest in files.items():
        assert hashlib.sha256((wdir / rel).read_bytes()).hexdigest() == digest, rel
    on_disk = {str(p.relative_to(wdir)) for p in (wdir / "files").rglob("*") if p.is_file()}
    assert on_disk == {k for k in files if k.startswith("files/")}


def test_qa_report_is_copied_unchanged(exported) -> None:
    wdir, manifest, _ = exported
    assert (wdir / "qa_report.json").read_bytes() == (
        wdir / "completeness" / "qa_report.json"
    ).read_bytes()
    assert set(manifest["qa_gates"]) >= {"scripted_facts_verified", "residual_unlabeled_share"}


def test_training_records_round_trip_through_the_sft_formatter_and_production_parser(
    exported, world
) -> None:
    wdir, manifest, teacher = exported
    records = list(read_records(wdir / "training_records.jsonl"))
    assert len(records) == manifest["training"]["records"] > 0
    assert manifest["training"]["sft_ready"] == len(records)
    assert any(r.facts for r in records) and any(not r.facts for r in records)
    context = world_company_context(world)
    system_prompt = build_system_prompt(company_context=context, variant=DEFAULT_PROMPT_VARIANT)
    for r in records:
        assert (r.corpus, r.world_id, r.world_split) == ("synthetic_v2", world.id, "train")
        assert r.company_context == context
        assert r.system_prompt_sha256 == system_prompt_sha256(system_prompt)
        assert r.source_id.startswith(f"{world.id}/")
        assert r.label_audit is not None and r.label_audit.status == "complete"
        system, user, assistant = production_sft_messages_for_record(
            r, company_context=r.company_context
        )
        assert system["content"] == system_prompt
        # The user message is the one the teacher was sent for this envelope.
        assert user["content"] in teacher.seen_users
        client = FakeLLMClient([assistant["content"]])
        parsed = LLMClientBackend(client).extract(
            envelope_for_record(r), system_prompt, InMemorySink(), "test"
        )
        assert parsed.rejected == [] and parsed.coercions == []
        assert [(f.type, f.quote, f.locator) for f in parsed.facts] == [
            (x.fact_type, x.quote, x.locator) for x in r.facts
        ]
        assert parsed.summary == r.teacher_answer.summary
        for f in r.facts:
            assert r.text[f.quote_span[0] : f.quote_span[1]] == f.quote
            assert f.claim_source in ("teacher", "teacher_restatement")
            assert f.sentiment is not None and f.confidence is not None
        assert TEACHER_MODEL in r.teachers and any(t in DEFAULT_ROTATION for t in r.teachers)


def test_every_label_of_a_trainable_document_is_a_fact_of_its_record(exported) -> None:
    wdir, _, _ = exported
    records = list(read_records(wdir / "training_records.jsonl"))
    labels = [json.loads(x) for x in (wdir / "labels.jsonl").read_text().splitlines()]
    by_source: dict[str, list[str]] = {}
    for x in labels:
        if x["label_source"] != "scripted_code":
            by_source.setdefault(x["doc_id"], []).append(x["text"])
    for r in records:
        source = r.source_id.split("/", 1)[1]
        assert sorted(f.quote for f in r.facts) == sorted(by_source.get(source, []))


def test_without_the_teacher_answer_records_follow_sft_formats_choice(
    tmp_path, written, world, world_plan
) -> None:
    """WP5 checkpoints written before the answer was kept: the record is
    exported without placeholders and the formatter refuses it."""
    wdir = tmp_path / "train" / "world-medical_devices-42"
    _stage_world(wdir, written, world, world_plan)
    for path in (wdir / "completeness" / "docs").glob("*.json"):
        state = json.loads(path.read_text())
        for attempt in state["attempts"]:
            for call in attempt["calls"]:
                call["answer"] = None
        for x in state["scripted_labels"] + state["silver_labels"]:
            x["teacher_fields"] = None
        path.write_text(json.dumps(state))
    manifest = _export(wdir, world, world_plan)
    records = list(read_records(wdir / "training_records.jsonl"))
    assert records and manifest["training"]["sft_ready"] == 0
    assert manifest["training"]["not_sft_ready_by_reason"] == {"no_teacher_answer": len(records)}
    assert any("teacher's full answer" in x for x in manifest["training_ready_reasons"])
    for r in records:
        assert r.teacher_answer is None
        assert all(f.sentiment is None and f.confidence is None for f in r.facts)
        with pytest.raises(LegacyRecordError):
            production_sft_messages_for_record(r, company_context=r.company_context)


def test_a_scripted_claim_is_exported_marked_and_a_known_unlabeled_fact_marks_the_record(
    tmp_path, written, world, world_plan
) -> None:
    wdir = tmp_path / "train" / "world-medical_devices-42"
    _stage_world(wdir, written, world, world_plan)
    call_path = next((wdir / "completeness" / "docs").glob("S-call-*.json"))
    state = json.loads(call_path.read_text())
    state["scripted_labels"][0]["claim_source"] = "scripted"
    call_path.write_text(json.dumps(state))
    email_path = next((wdir / "completeness" / "docs").glob("S-email-*.json"))
    state = json.loads(email_path.read_text())
    fact = state["attempts"][-1]["facts"][0]
    fact.update(triage="candidate", unlabelable=True)
    email_path.write_text(json.dumps(state))
    manifest = _export(wdir, world, world_plan)
    records = {r.record_id: r for r in read_records(wdir / "training_records.jsonl")}
    call = next(r for r in records.values() if "S-call-" in r.record_id)
    assert any(f.claim_source == "scripted" for f in call.facts)
    assert manifest["training"]["records_with_a_claim_not_from_a_teacher"] == 1
    source = fact["doc_id"]
    incomplete = records[f"{world.id}:{source}"]
    assert incomplete.label_audit.status == "incomplete" and incomplete.label_audit.unresolved == 1
    assert manifest["training"]["records_with_known_unlabeled_facts"] == 1


def test_documents_wp5_did_not_ship_are_left_out(tmp_path, written, world, world_plan) -> None:
    wdir = tmp_path / "train" / "world-medical_devices-42"
    _stage_world(wdir, written, world, world_plan)
    path = next((wdir / "completeness" / "docs").glob("S-email-*.json"))
    state = json.loads(path.read_text())
    state["status"] = "dropped"
    path.write_text(json.dumps(state))
    manifest = _export(wdir, world, world_plan)
    assert manifest["documents"]["excluded"] == {state["plan_id"]: "completeness_dropped"}
    sources = [json.loads(x) for x in (wdir / "sources.jsonl").read_text().splitlines()]
    assert not any(x["plan_id"] == state["plan_id"] for x in sources)
    assert not any(
        state["plan_id"] in r.record_id for r in read_records(wdir / "training_records.jsonl")
    )


def test_without_wp5_the_export_serves_the_bench_but_writes_no_training_record(
    tmp_path, written, world, world_plan
) -> None:
    wdir = tmp_path / "train" / "world-medical_devices-42"
    for doc in written.values():
        path = wdir / "writer" / "docs" / f"{doc.plan_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(doc.model_dump_json())
    manifest = _export(wdir, world, world_plan)
    assert load_world(wdir).excluded_labels == {}
    assert manifest["training"]["records"] == 0
    assert manifest["training"]["documents_without_records"]["no_completeness_pass"] == len(written)
    assert not manifest["training_ready"]
    assert any("completeness (WP5) did not run" in x for x in manifest["training_ready_reasons"])


# --------------------------------------------------------------------------
# Splits: test worlds and the held-out vertical never train
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("split", "held_out"), [(Split.TEST, False), (Split.TEST, True), (Split.TRAIN, True)]
)
def test_test_and_held_out_worlds_write_no_training_records_and_are_not_capped(
    tmp_path, written, world, world_plan, split, held_out
) -> None:
    wdir = tmp_path / split.value / "world-medical_devices-42"
    _stage_world(wdir, written, world, world_plan)
    (wdir / "training_records.jsonl").write_text("stale\n")
    manifest = _export(wdir, world, world_plan, split=split, held_out=held_out)
    assert not (wdir / "training_records.jsonl").exists()
    assert manifest["training"]["written"] is False
    assert manifest["caps"]["per_type"]["applied"] is False
    assert (
        manifest["caps"]["per_type"]["labels_before"]
        == manifest["caps"]["per_type"]["labels_after"]
    )
    assert load_world(wdir).excluded_labels == {}
    assert not manifest["training_ready"]


def test_a_validation_world_stamps_its_split_on_every_record(
    tmp_path, written, world, world_plan
) -> None:
    wdir = tmp_path / "validation" / "world-medical_devices-42"
    _stage_world(wdir, written, world, world_plan)
    _export(wdir, world, world_plan, split=Split.VALIDATION)
    records = list(read_records(wdir / "training_records.jsonl"))
    assert records and {r.world_split for r in records} == {"validation"}


# --------------------------------------------------------------------------
# Noise (WP6) composed onto WP5's labels
# --------------------------------------------------------------------------


def test_noise_is_applied_when_every_label_comes_out_of_wp6(
    tmp_path, written, world, world_plan
) -> None:
    wdir = tmp_path / "train" / "world-medical_devices-42"
    _stage_world(wdir, written, world, world_plan, noise_ops=frozenset({OP_MISFILED, OP_ASR}))
    manifest = _export(wdir, world, world_plan)
    assert manifest["documents"]["noised"] == len(written)
    assert manifest["documents"]["noise_not_applied"] == {}
    loaded = load_world(wdir)
    assert loaded.excluded_labels == {}
    misfiled = [s for s in loaded.sources.values() if "misfiled" in s.noise]
    assert misfiled
    rows = [json.loads(x) for x in (wdir / "sources.jsonl").read_text().splitlines()]
    moved = [r for r in rows if r["source_id"] in {s.source_id for s in misfiled}]
    assert all(r["account_id"] != r["true_account_id"] for r in moved)
    labels = [json.loads(x) for x in (wdir / "labels.jsonl").read_text().splitlines()]
    assert any(x.get("remapped") for x in labels)
    assert all(x.get("claim_source") for x in labels if x["label_source"] == "scripted_writer")
    records = list(read_records(wdir / "training_records.jsonl"))
    assert records
    noised_text = {r["source_id"]: r["text"] for r in rows}
    for r in records:
        source = r.source_id.split("/", 1)[1]
        assert r.text in noised_text[source]
        production_sft_messages_for_record(r, company_context=r.company_context)


def test_noise_is_not_applied_over_silver_labels_wp6_never_saw(
    tmp_path, written, world, world_plan
) -> None:
    wdir = tmp_path / "train" / "world-medical_devices-42"
    _stage_world(wdir, written, world, world_plan, silver=True, noise_ops=frozenset({OP_MISFILED}))
    manifest = _export(wdir, world, world_plan)
    notes = manifest["documents"]["noise_not_applied"]
    assert notes and all("silver" in v for v in notes.values())
    assert load_world(wdir).excluded_labels == {}


def test_noise_after_completeness_carries_every_silver_label(
    tmp_path, written, world, world_plan
) -> None:
    """The runner's order (WP5, then WP6 on WP5's shipped documents): every
    WP5 label, silver ones included, moves with the noised text, so every
    shipped document can be noised."""
    wdir = tmp_path / "train" / "world-medical_devices-42"
    _stage_world(
        wdir,
        written,
        world,
        world_plan,
        silver=True,
        noise_ops=frozenset({OP_MISFILED, OP_ASR}),
        noise_after_completeness=True,
    )
    manifest = _export(wdir, world, world_plan)
    assert manifest["documents"]["noise_not_applied"] == {}
    assert manifest["documents"]["noised"] == manifest["documents"]["exported"] == len(written)
    assert load_world(wdir).excluded_labels == {}
    labels = [json.loads(x) for x in (wdir / "labels.jsonl").read_text().splitlines()]
    silver = [x for x in labels if x["label_source"] == "silver_completeness"]
    assert silver and all(x["claim"] and x["teacher_model"] for x in silver)
    assert any(x.get("remapped") for x in silver)
    assert all("slot_id" not in x or x["label_source"] != "silver_completeness" for x in silver)
    noised = json.loads((wdir / "noise" / "docs" / f"{silver[0]['plan_id']}.json").read_text())
    assert noised["noised_from_sha256"]
    rows = {
        r["source_id"]: r["text"]
        for r in (json.loads(x) for x in (wdir / "sources.jsonl").read_text().splitlines())
    }
    for x in silver:
        assert rows[x["doc_id"]][x["char_start"] : x["char_end"]] == x["text"]
    records = list(read_records(wdir / "training_records.jsonl"))
    assert records
    for r in records:
        assert r.text in rows[r.source_id.split("/", 1)[1]]
        production_sft_messages_for_record(r, company_context=r.company_context)


# --------------------------------------------------------------------------
# The per-type cap (WP1 build note)
# --------------------------------------------------------------------------


def _doc(plan: DocPlan, *types: str, protected_fact: str | None = None) -> wp7.ComposedDoc:
    labels = [
        {"fact_id": f"{plan.id}-{i}", "fact_type": t, "label_source": "silver_completeness"}
        for i, t in enumerate(types)
    ]
    if protected_fact is not None:
        labels[0].update(fact_id=protected_fact, label_source="scripted_writer")
    return wp7.ComposedDoc(plan=plan, doc=None, labels=labels, decoys=[])  # type: ignore[arg-type]


def _world_of(world_plan: WorldDocPlan, spec: list[tuple[str, ...]]) -> list[wp7.ComposedDoc]:
    plans = sorted(world_plan.doc_plans, key=lambda d: d.id)
    return [_doc(plans[i], *types) for i, types in enumerate(spec)]


TYPES = (
    "usage", "decision", "praise", "pricing", "question", "commitment",
    "competitor_mention", "complaint", "other", "contact_change", "request",
)  # fmt: skip


def test_cap_drops_whole_documents_until_no_type_is_over_12_percent(world_plan) -> None:
    spec = [("risk",)] * 10 + [(t,) for t in TYPES for _ in range(6)]
    docs = _world_of(world_plan, spec)
    kept, report = wp7.apply_type_cap(docs, share=0.12, seed_key="s", protected=set())
    assert report.labels_before["risk"] == 10
    assert report.labels_after["risk"] == 9  # 9 of 75 is 12.0%
    assert len(report.documents_dropped) == 1 and not report.unmet_types
    assert report.as_json()["max_share_after"] <= 0.12
    assert len(kept) == len(docs) - 1


def test_cap_is_seeded_and_never_drops_a_protected_document(world_plan) -> None:
    spec = [("risk",)] * 12 + [(t,) for t in TYPES for _ in range(6)]
    docs = _world_of(world_plan, spec)
    one = wp7.apply_type_cap(docs, share=0.12, seed_key="a", protected=set())[1]
    two = wp7.apply_type_cap(docs, share=0.12, seed_key="a", protected=set())[1]
    assert one.documents_dropped == two.documents_dropped
    others = {
        tuple(
            wp7.apply_type_cap(docs, share=0.12, seed_key=k, protected=set())[1].documents_dropped
        )
        for k in "bcdefgh"
    }
    assert len(others) > 1
    protected = {d.plan_id for d in docs if d.labels[0]["fact_type"] == "risk"}
    kept, report = wp7.apply_type_cap(docs, share=0.12, seed_key="a", protected=protected)
    assert report.documents_dropped == [] and report.unmet_types == ["risk"]
    assert len(kept) == len(docs)


def test_cap_never_makes_the_mix_worse(world_plan) -> None:
    """Dropping a document shrinks the total and can push another type over;
    a drop is only taken when the labels over the cap go down."""
    import random

    for seed in range(30):
        rng = random.Random(seed)
        spec = [
            tuple(rng.choice(("risk", "risk", *TYPES)) for _ in range(rng.randint(1, 4)))
            for _ in range(rng.randint(3, 30))
        ]
        docs = _world_of(world_plan, spec)
        before = wp7._excess(
            sum((d.type_counts() for d in docs), start=__import__("collections").Counter()), 0.12
        )
        kept, report = wp7.apply_type_cap(docs, share=0.12, seed_key=str(seed), protected=set())
        after = wp7._excess(
            sum((d.type_counts() for d in kept), start=__import__("collections").Counter()), 0.12
        )
        assert after <= before + 1e-9
        assert (after < before) or not report.documents_dropped


def test_protected_documents_hold_a_verified_fact_of_an_arc_event(world, world_plan) -> None:
    arc_events = {e for arc in world.arcs for e in arc.event_ids}
    arc_fact = next(f.id for f in world.facts if f.event_id in arc_events)
    other_fact = next(f.id for f in world.facts if f.event_id not in arc_events)
    plans = sorted(world_plan.doc_plans, key=lambda d: d.id)
    docs = [
        _doc(plans[0], "risk", protected_fact=arc_fact),
        _doc(plans[1], "risk", protected_fact=other_fact),
        _doc(plans[2], "risk"),
    ]
    recovered = _doc(plans[3], "risk")
    recovered.labels[0].update(fact_id=arc_fact, recovers_fact_id=arc_fact)
    assert wp7.protected_documents([*docs, recovered], world) == {plans[0].id, plans[3].id}


def test_a_small_export_is_not_capped(exported) -> None:
    """Dropping every labeled document meets any cap, so a few documents are
    never capped; the manifest says why."""
    _, manifest, _ = exported
    cap = manifest["caps"]["per_type"]
    assert sum(cap["labels_before"].values()) < wp7.CAP_MIN_LABELS
    assert cap["applied"] is False and "fewer than" in cap["reason"]
    assert "type_cap" not in manifest["documents"]["excluded_by_reason"]


def test_export_reports_the_cap_it_applied(tmp_path, written, world, world_plan) -> None:
    wdir = tmp_path / "train" / "world-medical_devices-42"
    _stage_world(wdir, written, world, world_plan, silver=True)
    manifest = _export(wdir, world, world_plan, cap_min_labels=0)
    cap = manifest["caps"]["per_type"]
    assert cap["applied"] is True and cap["share"] == 0.12
    assert cap["documents_dropped"]
    assert (
        sum(cap["labels_before"].values()) - sum(cap["labels_after"].values())
        == cap["labels_dropped"]
    )
    assert set(cap["documents_dropped"]) == {
        k for k, v in manifest["documents"]["excluded"].items() if v == "type_cap"
    }
    sources = {json.loads(x)["plan_id"] for x in (wdir / "sources.jsonl").read_text().splitlines()}
    assert not sources & set(cap["documents_dropped"])
    assert load_world(wdir).excluded_labels == {}
