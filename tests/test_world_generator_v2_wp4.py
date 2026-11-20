"""Tests for world generator v2 work package 4, the writer
(`01-spec/world-generator-v2-2026-09-22.md` sections 3.1 and 3.2, and
section 9's WP4 row). Every model call is stubbed (`tests/v2_writer_fakes`);
nothing here touches the network.

Covers: marker parsing with decoys; each writer output protocol; each of
the eight verification checks; targeted repair on the same writer;
regeneration on a different writer; dropping a fact that never verifies;
provenance (every word in a written unit comes from a writer reply); the
open-weight allowlist; the 25% cap; the budget stop and spend lines; and
resume without redoing finished documents.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from corpus.generator.markers import (
    normalize_marker_glyphs,
    strip_all_markers,
)
from corpus.generator.prose import PAID_BUDGET_STOP_MARKER
from corpus.generator.v2.assembly import (
    assemble,
    locate_markers,
    parse_draft,
    prepare_sources,
)
from corpus.generator.v2.models import Split
from corpus.generator.v2.packs.schema import VerticalPack, load_pack
from corpus.generator.v2.planner import (
    DocKind,
    DocPlan,
    DocRole,
    LabelRecord,
    WorldDocPlan,
    plan_world,
)
from corpus.generator.v2.renderers import render
from corpus.generator.v2.simulator import simulate_world
from corpus.generator.v2.verify import (
    build_slot_specs,
    coverage_extras,
    is_claim_copy,
    required_terms_for,
    verify_document,
)
from corpus.generator.v2.writer import (
    ALLOWED_WRITER_MODELS,
    CORPUS_KEY,
    DEFAULT_ROTATION,
    RunBudget,
    WriterBudgetExceeded,
    WriterCapExhausted,
    WriterConfig,
    WriterPool,
    WriterSession,
    build_cast,
    run_world,
    write_document,
    writer_doc_plans,
)
from portolan.connectors.file_connector import FileConnector
from portolan.training.spend import SpendMeter
from tests.v2_writer_fakes import (
    ScriptedBackend,
    obedient,
    parse_prompt,
    render_document,
    repair_reply,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PACK_PATH = REPO_ROOT / "corpus" / "generator" / "v2" / "packs" / "medical_devices.yaml"


@pytest.fixture(scope="module")
def pack() -> VerticalPack:
    return load_pack(PACK_PATH)


@pytest.fixture(scope="module")
def world(pack: VerticalPack):
    return simulate_world(pack, seed=42, months=8, split=Split.TRAIN)


@pytest.fixture(scope="module")
def world_plan(world, pack) -> WorldDocPlan:
    return plan_world(world, pack)


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
    facts: int = 1,
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


def _session(
    backend, world_plan=None, *, cap_usd: float = 5.0, meter=None, total: int = 300, models=None
):
    config = WriterConfig(models=tuple(models or DEFAULT_ROTATION))
    return WriterSession(
        backend=backend,
        pool=WriterPool(config.models, world_doc_total=total),
        budget=RunBudget(cap_usd=cap_usd, meter=meter),
        config=config,
    )


def _write(plan, world, pack, decoys, backend, **kw):
    session = _session(backend, **kw)
    doc = write_document(
        render(plan, world, decoys.world_plan, pack),
        plan,
        world,
        pack,
        decoys_by_id=decoys,
        session=session,
    )
    return doc, session


# --------------------------------------------------------------------------
# Markers
# --------------------------------------------------------------------------


def test_marker_glyph_variants_are_normalized_and_stripped() -> None:
    text = "a [[F00041]]one[[/F00041]] b 〚D00002〛two〚/D00002〛 c ⟦ F00043 ⟧three⟦ /F00043 ⟧"
    norm = normalize_marker_glyphs(text)
    assert "⟦F00041⟧one⟦/F00041⟧" in norm
    assert "⟦D00002⟧two⟦/D00002⟧" in norm
    assert "⟦F00043⟧three⟦/F00043⟧" in norm
    assert strip_all_markers(norm) == "a one b two c three"
    assert normalize_marker_glyphs("keep [[this]] as is") == "keep [[this]] as is"
    # Live 2026-09-22: an angle bracket typed next to the closer.
    assert normalize_marker_glyphs("tasks.⟩⟧/F00406⟧ x") == "tasks.⟦/F00406⟧ x"
    assert normalize_marker_glyphs("a ⟨b⟩ c") == "a ⟨b⟩ c"


def test_locate_markers_offsets_unbalanced_and_overlap() -> None:
    clean, spans, unbalanced, overlap = locate_markers("x ⟦F1⟧ab⟦/F1⟧ y ⟦D2⟧cd⟦/D2⟧ ⟦F3⟧open")
    assert clean == "x ab y cd open"
    assert [(s.marker_id, clean[s.start : s.end]) for s in spans] == [("F1", "ab"), ("D2", "cd")]
    assert unbalanced == ["F3"]
    _clean, _spans, _unb, overlap = locate_markers("⟦F1⟧a ⟦D2⟧b⟦/F1⟧ c⟦/D2⟧")
    assert set(overlap) == {"F1", "D2"}


def test_live_glyph_mistakes_are_repaired_structurally() -> None:
    # Seen live 2026-09-22: qwen3 wrote `⟧/F…⟧` for the closer; others drop
    # the closer's slash or leave a lone bracket.
    clean, spans, unbalanced, _o = locate_markers(
        normalize_marker_glyphs("a ⟦F00041⟧one⟧/F00041⟧ b ⟦F00042⟧two⟦F00042⟧ c ⟧ d")
    )
    assert clean == "a one b two c d"
    assert [s.marker_id for s in spans] == ["F00041", "F00042"]
    assert unbalanced == []
    clean, spans, unbalanced, _o = locate_markers(
        "issues. ✦D00004But we hope ⟦F00041⟧not F00999⟦/F00041⟧"
    )
    assert clean == "issues. But we hope not"
    assert unbalanced == ["D00004", "F00999"] and [s.marker_id for s in spans] == ["F00041"]
    # Whitespace just inside a marker is trimmed off the span, and the
    # double space a removed marker leaves is collapsed.
    clean, spans, _u, _o = locate_markers("said ⟦F00041⟧ the fact ⟦/F00041⟧ then")
    assert clean == "said the fact then"
    assert clean[spans[0].start : spans[0].end] == "the fact"


# --------------------------------------------------------------------------
# Output protocols
# --------------------------------------------------------------------------


def test_parse_lines_resolves_speakers_and_flags_invented_people(world, world_plan) -> None:
    plan = _first(world_plan, {DocKind.CALL}, facts=2)
    cast, _author, people = build_cast(plan, world, [])
    a, b = cast[0].name, cast[1].name
    raw = f"```\nHere is the call:\n[00:00:01] {a}: hello\n**{b}:** hi there\ncontinued words\n```"
    draft = parse_draft(DocKind.CALL, raw, people, want_subject=False)
    assert [u.author_name for u in draft.units] == [a, b]
    assert draft.units[1].text == "hi there continued words"
    assert not draft.fatal
    typo = a[:3] + a[2] + a[3:]  # one doubled letter, as a live writer did
    fixed = parse_draft(DocKind.CALL, f"{typo}: hello", people, want_subject=False)
    assert [u.author_name for u in fixed.units] == [a] and not fixed.fatal
    moved = parse_draft(
        DocKind.CALL, f"{a}: hi\n⟦D00004⟧{b}: if it ran late⟦/D00004⟧", people, want_subject=False
    )
    assert [u.author_name for u in moved.units] == [a, b]
    assert moved.units[1].text == "⟦D00004⟧if it ran late⟦/D00004⟧"
    bad = parse_draft(
        DocKind.CALL, f"{a}: hello\nZelda Quartermaine: I am new", people, want_subject=False
    )
    assert any("unknown speaker" in f for f in bad.fatal)


def test_parse_blocks_subject_and_prose_paragraphs(world, world_plan) -> None:
    plan = _first(world_plan, {DocKind.EMAIL})
    cast, _author, people = build_cast(plan, world, [])
    raw = f"Subject: pumps\n### {cast[0].name}\nHi,\n\npara two\n### {cast[1].name}\nthanks"
    draft = parse_draft(DocKind.EMAIL, raw, people, want_subject=True)
    assert draft.subject == "pumps"
    assert [u.text for u in draft.units] == ["Hi,\n\npara two", "thanks"]
    inside = parse_draft(
        DocKind.TICKET, f"### {cast[0].name}\nSubject: pumps\n\nbody", people, want_subject=False
    )
    assert inside.subject is None and inside.units[0].text == "body"
    missing = parse_draft(DocKind.EMAIL, f"### {cast[0].name}\nHi", people, want_subject=True)
    assert "requested Subject line missing" in missing.fatal
    prose = parse_draft(
        DocKind.CRM_NOTE,
        "# Internal note\n\nHere is the note:\n\nline one\n\nline two\n",
        people,
        want_subject=False,
    )
    assert [u.text for u in prose.units] == ["line one", "line two"]


# --------------------------------------------------------------------------
# Happy path, every writer kind
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kinds, facts, role",
    [
        ({DocKind.CALL}, 2, DocRole.EVENT),
        ({DocKind.EMAIL}, 2, DocRole.EVENT),
        ({DocKind.TICKET}, 2, DocRole.EVENT),
        ({DocKind.SLACK, DocKind.TEAMS}, 1, DocRole.EVENT),
        ({DocKind.SLACK, DocKind.TEAMS}, 0, DocRole.ROUTINE),
        ({DocKind.CRM_NOTE}, 2, DocRole.EVENT),
        ({DocKind.MEETING_NOTES}, 2, DocRole.EVENT),
        ({DocKind.DOC}, 2, DocRole.EVENT),
        (set(DocKind), 0, DocRole.HARD_NEGATIVE),
        (set(DocKind), 0, DocRole.OFF_TOPIC),
    ],
)
def test_obedient_writer_verifies_every_slot(
    world, world_plan, pack, decoys, kinds, facts, role
) -> None:
    plan = _first(world_plan, kinds, facts=facts, role=role)
    backend = ScriptedBackend(obedient)
    doc, _session = _write(plan, world, pack, decoys, backend)
    assert doc.status == "written", doc.fatal
    assert doc.writer_model in ALLOWED_WRITER_MODELS
    assert len(backend.calls) == 1
    assert all(o.status == "verified" for o in doc.slot_outcomes)
    assert len(doc.labels) == len(plan.fact_ids)
    assert len(doc.decoy_spans) == len(plan.decoy_ids)
    prepared = {p.source_id: p.text for p in doc.prepared_sources}
    for label in doc.labels:
        assert isinstance(label, LabelRecord)
        assert label.label_source == "scripted_writer"
        assert label.writer_model == doc.writer_model
        assert prepared[label.doc_id][label.char_start : label.char_end] == label.text
    for span in doc.decoy_spans:
        assert prepared[span.doc_id][span.char_start : span.char_end] == span.text
    for content in doc.files.values():
        assert "⟦" not in content and "⟧" not in content and "[SLOT" not in content


def test_written_files_load_through_file_connector(
    world, world_plan, pack, decoys, tmp_path
) -> None:
    for kinds in ({DocKind.CALL}, {DocKind.EMAIL}, {DocKind.TICKET}):
        plan = _first(world_plan, kinds, facts=1)
        doc, _s = _write(plan, world, pack, decoys, ScriptedBackend(obedient))
        for rel, content in doc.files.items():
            (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
            (tmp_path / rel).write_text(content, encoding="utf-8")
    refs = list(FileConnector(tmp_path).list_sources())
    assert {r.source_type.value for r in refs} == {"call", "email", "ticket"}


def test_email_quote_block_is_stripped_before_labels(world, world_plan, pack, decoys) -> None:
    plan = next(
        d
        for d in sorted(world_plan.doc_plans, key=lambda d: (d.date, d.id))
        if d.kind == DocKind.EMAIL
        and d.role == DocRole.EVENT
        and d.fact_ids
        and render(d, world, world_plan, pack).quote_ops
    )
    doc, _s = _write(plan, world, pack, decoys, ScriptedBackend(obedient))
    quoted = [c for c in doc.files.values() if "wrote:\n> " in c]
    assert quoted, "the skeleton asks for quoted replies"
    for p in doc.prepared_sources:
        assert "wrote:" not in p.text and "\n> " not in p.text


# --------------------------------------------------------------------------
# The eight checks
# --------------------------------------------------------------------------


def _verify_raw(plan, world, pack, decoys, raw):
    rendered = render(plan, world, decoys.world_plan, pack)
    specs = build_slot_specs(rendered, plan, world, decoys)
    _cast, _author, people = build_cast(plan, world, specs)
    draft = parse_draft(plan.kind, raw, people, want_subject=False)
    assembled = assemble(plan=plan, rendered=rendered, world=world, draft=draft, people=people)
    return verify_document(assembled, prepare_sources(assembled), specs, world), specs


def _call_prompt(plan, world, pack, decoys):
    backend = ScriptedBackend(obedient)
    _write(plan, world, pack, decoys, backend)
    return backend.calls[0].system, backend.calls[0].user


def _span_anchor(world, pack, decoys, plan, fid):
    """The first anchor the span itself must carry (a product, competitor or
    person); the account is never one (the speaker says we or you)."""
    specs = build_slot_specs(render(plan, world, decoys.world_plan, pack), plan, world, decoys)
    return next(sp for sp in specs if sp.marker_id == fid).anchors[0]


def _codes(verification, marker_id):
    r = next(r for r in verification.results if r.spec.marker_id == marker_id)
    return {f.split(" ")[0] for f in r.failures}


@pytest.fixture(scope="module")
def call_plan(world_plan):
    return _first(world_plan, {DocKind.CALL}, facts=2)


def test_check1_missing_and_duplicate_markers(world, pack, decoys, call_plan) -> None:
    system, user = _call_prompt(call_plan, world, pack, decoys)
    fid = call_plan.fact_ids[0]
    raw = render_document(system, user, mutate={fid: lambda item, span: ("", None)})
    v, _ = _verify_raw(call_plan, world, pack, decoys, raw)
    assert "c1:missing_marker" in _codes(v, fid)
    good = render_document(system, user)
    line = next(line for line in good.split("\n") if f"⟦{fid}⟧" in line)
    v, _ = _verify_raw(call_plan, world, pack, decoys, good + line + "\n")
    assert any(c.startswith("c1:marker_count=2") for c in _codes(v, fid))


def test_check1_fact_stated_twice_unmarked(world, pack, decoys, call_plan) -> None:
    system, user = _call_prompt(call_plan, world, pack, decoys)
    fid = call_plan.fact_ids[0]
    raw = render_document(system, user)
    line = next(line for line in raw.split("\n") if f"⟦{fid}⟧" in line)
    speaker, _, text = line.partition(": ")
    copy = strip_all_markers(text).replace("fixture lead ", "").replace(" tail", "")
    v, _ = _verify_raw(call_plan, world, pack, decoys, raw + f"{speaker}: {copy}.\n")
    assert "c1:stated_twice" in _codes(v, fid)


def test_degenerate_repetition_is_fatal(world, pack, decoys, call_plan) -> None:
    system, user = _call_prompt(call_plan, world, pack, decoys)
    raw = render_document(system, user)
    speaker = raw.split(":", 1)[0]
    raw += "".join(f"{speaker}: the same looping sentence again\n" for _ in range(5))
    v, _ = _verify_raw(call_plan, world, pack, decoys, raw)
    assert any("degenerate repetition" in f for f in v.fatal)


def test_repair_that_pastes_another_unit_is_rejected(world, pack, decoys, call_plan) -> None:
    from corpus.generator.v2.writer import apply_repairs
    from corpus.generator.v2.writer_prompt import RepairTarget

    rendered = render(call_plan, world, decoys.world_plan, pack)
    specs = build_slot_specs(rendered, call_plan, world, decoys)
    _cast, _author, people = build_cast(call_plan, world, specs)
    system, user = _call_prompt(call_plan, world, pack, decoys)
    draft = parse_draft(call_plan.kind, render_document(system, user), people, want_subject=False)
    donor = "Everyone on the floor said the new screens are much easier to read now."
    draft.units[0].text += ". " + donor
    target = RepairTarget(3, draft.units[2].author_name, ["state it"])
    before = draft.units[2].text
    new, notes = apply_repairs(draft, [target], {"U3": f"Sure. {donor}"}, call_plan.kind, world)
    assert new.units[2].text == before
    assert any("pasted another unit" in n for n in notes)
    ok, _notes = apply_repairs(draft, [target], {"U3": "A fresh sentence."}, call_plan.kind, world)
    assert ok.units[2].text == "A fresh sentence."


def test_prose_prompt_carries_no_sign_off_hint(world, world_plan, pack, decoys) -> None:
    plan = _first(world_plan, {DocKind.CRM_NOTE}, facts=1)
    _system, user = _call_prompt(plan, world, pack, decoys)
    assert "signs off" not in user and " register" not in user


def test_check2_span_of_names_only_states_nothing(world, pack, decoys, call_plan) -> None:
    system, user = _call_prompt(call_plan, world, pack, decoys)
    fid = call_plan.fact_ids[0]
    names = ", ".join(next(f for f in world.facts if f.id == fid).anchors)
    raw = render_document(system, user, mutate={fid: lambda item, span: (names, None)})
    v, _ = _verify_raw(call_plan, world, pack, decoys, raw)
    assert "c2:span_states_nothing" in _codes(v, fid)


def test_check2_long_and_crossing_spans(world, pack, decoys, call_plan) -> None:
    system, user = _call_prompt(call_plan, world, pack, decoys)
    fid = call_plan.fact_ids[0]
    raw = render_document(
        system, user, mutate={fid: lambda item, span: (span + " word" * 60, None)}
    )
    v, _ = _verify_raw(call_plan, world, pack, decoys, raw)
    assert "c2:span_too_long" in _codes(v, fid)
    crossing = render_document(system, user).replace(f"⟦/{fid}⟧", "", 1)
    crossing = crossing.replace("fixture closing words", f"fixture closing⟦/{fid}⟧ words")
    v, _ = _verify_raw(call_plan, world, pack, decoys, crossing)
    assert "c1:unbalanced_marker" in _codes(v, fid)


def test_check3_anchor_and_reference_resolution(world, world_plan, pack, decoys, call_plan) -> None:
    system, user = _call_prompt(call_plan, world, pack, decoys)
    fid = call_plan.fact_ids[0]
    anchor = _span_anchor(world, pack, decoys, call_plan, fid)
    raw = render_document(
        system, user, mutate={fid: lambda item, span: (span.replace(anchor, "our place"), None)}
    )
    v, _ = _verify_raw(call_plan, world, pack, decoys, raw)
    assert "c3:anchor_missing" in _codes(v, fid)
    # A non-explicit reference passes when the full name is named earlier,
    # and fails (c3:reference_unresolvable) when it is not.
    ref_plan, ref_spec = None, None
    for d in sorted(world_plan.doc_plans, key=lambda d: (d.date, d.id)):
        if d.kind not in (DocKind.EMAIL, DocKind.TICKET) or not d.fact_ids:
            continue
        specs = build_slot_specs(render(d, world, decoys.world_plan, pack), d, world, decoys)
        ref_spec = next((sp for sp in specs if sp.is_fact and sp.reference_anchor), None)
        if ref_spec is not None:
            ref_plan = d
            break
    assert ref_plan is not None and ref_spec is not None
    doc, _s = _write(ref_plan, world, pack, decoys, ScriptedBackend(obedient))
    ref_labels = [label for label in doc.labels if label.fact_id == ref_spec.marker_id]
    assert ref_labels and ref_labels[0].reference_mode != "explicit"
    system, user = _call_prompt(ref_plan, world, pack, decoys)
    raw = render_document(system, user)
    raw = raw.replace(f"fixture context about {ref_spec.reference_anchor} first", "fixture")
    raw = raw.replace(ref_spec.reference_anchor, "someone")
    v, _ = _verify_raw(ref_plan, world, pack, decoys, raw)
    assert "c3:reference_unresolvable" in _codes(v, ref_spec.marker_id)


def test_check4_wrong_speaker(world, pack, decoys, call_plan) -> None:
    system, user = _call_prompt(call_plan, world, pack, decoys)
    fid = call_plan.fact_ids[0]
    fact = next(f for f in world.facts if f.id == fid)
    cast, _a, _p = build_cast(call_plan, world, [])
    other = next(p.name for p in cast if p.id != fact.speaker)
    raw = render_document(system, user, mutate={fid: lambda item, span: (span, other)})
    v, _ = _verify_raw(call_plan, world, pack, decoys, raw)
    assert "c4:wrong_speaker" in _codes(v, fid)


def test_check5_forbidden_words_and_script_copy(world, pack, decoys, call_plan) -> None:
    system, user = _call_prompt(call_plan, world, pack, decoys)
    fid = call_plan.fact_ids[0]
    fact = next(f for f in world.facts if f.id == fid)
    other_account = next(
        a.name for a in world.accounts if a.id != fact.account_id and a.name not in fact.anchors[0]
    )
    raw = render_document(
        system, user, mutate={fid: lambda item, span: (f"{span} {other_account}", None)}
    )
    v, _ = _verify_raw(call_plan, world, pack, decoys, raw)
    assert "c5:forbidden_word" in _codes(v, fid)
    raw = render_document(
        system, user, mutate={fid: lambda item, span: (f"{span} [SLOT color]", None)}
    )
    v, _ = _verify_raw(call_plan, world, pack, decoys, raw)
    assert "c5:forbidden_word" in _codes(v, fid)
    raw = render_document(system, user, mutate={fid: lambda item, span: (fact.claim, None)})
    v, _ = _verify_raw(call_plan, world, pack, decoys, raw)
    assert "c5:copies_the_script_wording" in _codes(v, fid)
    assert is_claim_copy(fact.claim, fact.claim)
    assert not is_claim_copy("we run it in the ICU", fact.claim)
    # Seen live 2026-09-22: the claim minus its ISO date is still a copy;
    # a real paraphrase with the same names is not.
    from corpus.generator.v2.verify import entity_words_for

    ew = entity_words_for(world)
    claim = (
        "Tangleview Community Hospital 1 flagged that Harborlight Health Systems's "
        "2026-01-28 commitment on Wayfinder Link is now overdue."
    )
    near = (
        "Tangleview Community Hospital 1 flagged that Harborlight Health Systems's "
        "commitment on Wayfinder Link is now overdue."
    )
    para = (
        "we've raised a red flag with Harborlight Health Systems because the Wayfinder "
        "Link commitment they made to Tangleview Community Hospital 1 has come and gone"
    )
    assert is_claim_copy(near, claim, ew)
    assert not is_claim_copy(para, claim, ew)


def test_check6_numeral_and_name_rule(world, pack, decoys, call_plan) -> None:
    system, user = _call_prompt(call_plan, world, pack, decoys)
    fid = next(
        f.id
        for f in world.facts
        if f.id in call_plan.fact_ids
        and required_terms_for(f.claim, f.anchors, call_plan.date, f.date)
    )
    fact = next(f for f in world.facts if f.id == fid)
    terms = required_terms_for(fact.claim, fact.anchors, call_plan.date, fact.date)
    raw = render_document(system, user)
    for term in terms:
        raw = raw.replace(term, "thing")
    v, _ = _verify_raw(call_plan, world, pack, decoys, raw)
    assert _codes(v, fid) & {"c6:name_not_in_text", "c6:numeral_not_in_text"}


def test_check6_coverage_extras_are_normalizations_only() -> None:
    from datetime import date

    extras = coverage_extras(
        "see you on January 20th, about fifteen units, 3/4", date(2026, 1, 5)
    ).split()
    assert {"2026-01-05", "2026-01-20", "15", "2026-03-04"} <= set(extras)


def test_check7_locator_patterns(world, world_plan, pack, decoys) -> None:
    from corpus.generator.v2.verify import locator_pattern_for

    email = _first(world_plan, {DocKind.EMAIL})
    doc, _s = _write(email, world, pack, decoys, ScriptedBackend(obedient))
    pattern = locator_pattern_for(DocKind.EMAIL, world)
    assert all(pattern.match(label.locator) for label in doc.labels)
    assert not pattern.match("S-email-00001-m1@halvardmedical.com#p1")
    assert locator_pattern_for(DocKind.TICKET, world).match("S-ticket-routine-00001#3")
    assert locator_pattern_for(DocKind.CRM_NOTE, world).match("S-crm_note-00002#p1")


def test_check8_span_in_a_stripped_signature_does_not_resolve(
    world, world_plan, pack, decoys
) -> None:
    plan = _first(world_plan, {DocKind.EMAIL}, facts=1)
    system, user = _call_prompt(plan, world, pack, decoys)
    fid = plan.fact_ids[0]
    raw = render_document(system, user)
    # Move the fact's paragraph under an RFC 3676 signature delimiter, which
    # prepare_text strips: the span exists in the file, not in the text sent.
    lines = raw.split("\n")
    i = next(k for k, line in enumerate(lines) if f"⟦{fid}⟧" in line)
    lines[i : i + 1] = ["-- ", lines[i]]
    v, _ = _verify_raw(plan, world, pack, decoys, "\n".join(lines))
    assert "c8:span_not_in_prepared_text" in _codes(v, fid)


# --------------------------------------------------------------------------
# Repair, regeneration, drop
# --------------------------------------------------------------------------


def test_targeted_repair_fixes_a_missing_anchor_on_the_same_writer(
    world, pack, decoys, call_plan
) -> None:
    fid = call_plan.fact_ids[0]
    anchor = _span_anchor(world, pack, decoys, call_plan, fid)

    def reply(model, system, user, n):
        if "REWRITE THESE UNITS" in user:
            return repair_reply(user)
        return render_document(
            system, user, mutate={fid: lambda item, span: (span.replace(anchor, "us"), None)}
        )

    backend = ScriptedBackend(reply)
    doc, _s = _write(call_plan, world, pack, decoys, backend)
    assert [c.call_kind for c in doc.calls] == ["initial", "repair"]
    assert doc.calls[0].model == doc.calls[1].model == doc.writer_model
    outcome = next(o for o in doc.slot_outcomes if o.marker_id == fid)
    assert outcome.status == "repaired"
    assert any(f.startswith("c3:anchor_missing") for f in outcome.first_failures)
    assert fid in {label.fact_id for label in doc.labels}
    repair_user = backend.calls[1].user
    assert "=== U" in repair_user and fid in repair_user


def test_regeneration_uses_a_different_writer(world, pack, decoys, call_plan) -> None:
    fid = call_plan.fact_ids[0]

    def reply(model, system, user, n):
        if "REWRITE THESE UNITS" in user:
            return "=== U1 ===\nnothing useful\n"
        if n == 0:
            return render_document(system, user, mutate={fid: lambda item, span: ("", None)})
        return render_document(system, user)

    backend = ScriptedBackend(reply)
    doc, _s = _write(call_plan, world, pack, decoys, backend)
    kinds = [c.call_kind for c in doc.calls]
    assert kinds == ["initial", "repair", "regenerate"]
    assert doc.calls[2].model != doc.calls[0].model
    assert doc.writer_model == doc.calls[2].model
    assert "AN EARLIER DRAFT FAILED THESE CHECKS" in backend.calls[2].user
    assert all(o.status == "regenerated" for o in doc.slot_outcomes if o.slot_kind == "fact")


def test_fact_that_never_verifies_is_dropped_and_logged(world, pack, decoys, call_plan) -> None:
    fid = call_plan.fact_ids[0]
    anchor = _span_anchor(world, pack, decoys, call_plan, fid)

    def broken(item, span):
        return span.replace(anchor, "our site"), None

    def reply(model, system, user, n):
        if "REWRITE THESE UNITS" in user:
            return repair_reply(user, fix=lambda marker, span: span.replace(anchor, "our site"))
        return render_document(system, user, mutate={fid: broken})

    doc, _s = _write(call_plan, world, pack, decoys, ScriptedBackend(reply))
    assert doc.status == "written"
    assert len(doc.calls) == 3
    outcome = next(o for o in doc.slot_outcomes if o.marker_id == fid)
    assert outcome.status == "dropped"
    assert outcome.dropped_text and "our site" in outcome.dropped_text
    assert fid not in {label.fact_id for label in doc.labels}
    others = [o for o in doc.slot_outcomes if o.marker_id != fid and o.slot_kind == "fact"]
    assert others and all(o.status != "dropped" for o in others)


def test_script_wording_is_never_labeled_and_is_noted_for_wp5(
    world, pack, decoys, call_plan
) -> None:
    # The copy rule used to drop the whole document (provenance, while the
    # claim templates were Claude-written). The templates are open-weight
    # since WP2, so the document ships; the copying span is never a label,
    # and the note tells WP5 there is an unlabeled restatement.
    fid = call_plan.fact_ids[0]
    claim = next(f for f in world.facts if f.id == fid).claim

    def reply(model, system, user, n):
        if "REWRITE THESE UNITS" in user:
            return "=== U1 ===\nnothing useful\n"
        return render_document(system, user, mutate={fid: lambda item, span: (claim, None)})

    doc, _s = _write(call_plan, world, pack, decoys, ScriptedBackend(reply))
    assert doc.status == "written"
    assert fid not in {label.fact_id for label in doc.labels}
    outcome = next(o for o in doc.slot_outcomes if o.marker_id == fid)
    assert outcome.status == "dropped" and any(
        f.startswith("c5:copies_the_script_wording") for f in outcome.final_failures
    )
    assert any("repeats the script wording" in n and fid in n for n in doc.notes)


def test_failing_decoy_is_dropped_without_extra_calls(world, world_plan, pack, decoys) -> None:
    plan = _first(world_plan, {DocKind.CALL, DocKind.TICKET, DocKind.EMAIL}, facts=1, decoy=True)
    did = plan.decoy_ids[0]
    backend = ScriptedBackend(
        lambda m, s, u, n: render_document(s, u, mutate={did: lambda item, span: ("", None)})
    )
    doc, _s = _write(plan, world, pack, decoys, backend)
    assert len(backend.calls) == 1
    outcome = next(o for o in doc.slot_outcomes if o.marker_id == did)
    assert outcome.status == "dropped" and not doc.decoy_spans


def test_structurally_broken_document_is_dropped_never_filled(
    world, pack, decoys, call_plan
) -> None:
    backend = ScriptedBackend(lambda m, s, u, n: "Zelda Quartermaine: hello\nOrvil Pike: hi")
    doc, _s = _write(call_plan, world, pack, decoys, backend)
    assert doc.status == "dropped"
    assert not doc.files and not doc.labels
    assert [c.call_kind for c in doc.calls] == ["initial", "regenerate"]


def test_provenance_every_unit_word_comes_from_a_writer_reply(
    world, pack, decoys, call_plan
) -> None:
    fid = call_plan.fact_ids[0]
    anchor = _span_anchor(world, pack, decoys, call_plan, fid)

    def reply(model, system, user, n):
        if "REWRITE THESE UNITS" in user:
            return repair_reply(user)
        return render_document(
            system, user, mutate={fid: lambda item, span: (span.replace(anchor, "x"), None)}
        )

    doc, _s = _write(call_plan, world, pack, decoys, ScriptedBackend(reply))
    replies = " ".join(strip_all_markers(normalize_marker_glyphs(c.raw_output)) for c in doc.calls)
    reply_words = set(re.findall(r"\S+", replies))
    text = next(iter(doc.files.values()))
    # Above the first turn is WP3's code header (Date, Call type, Account,
    # Attendees): structure, not writer text.
    turns = text[re.search(r"^\[\d{2}:\d{2}:\d{2}\]", text, re.MULTILINE).start() :]
    for line in turns.strip().split("\n"):
        body = re.sub(r"^\[\d{2}:\d{2}:\d{2}\] [^:]+: ", "", line)
        assert set(body.split()) <= reply_words, line


# --------------------------------------------------------------------------
# Allowlist, cap, rotation, budget, spend, resume
# --------------------------------------------------------------------------


def test_allowlist_refuses_nemotron_free_and_unknown_ids() -> None:
    for bad in (
        "nvidia/nemotron-3.5-lightning",
        "qwen/qwen3-32b:free",
        "anthropic/claude-haiku-4.5",
    ):
        with pytest.raises(ValueError):
            WriterPool((bad,), world_doc_total=10)
    assert all(m in ALLOWED_WRITER_MODELS for m in DEFAULT_ROTATION)
    assert len({ALLOWED_WRITER_MODELS[m].family for m in ALLOWED_WRITER_MODELS}) >= 5


def test_cap_blocks_a_writer_at_25_percent() -> None:
    pool = WriterPool(DEFAULT_ROTATION[:2], world_doc_total=4)
    assert pool.cap == 1
    first = pool.pick()
    pool.assign(first)
    second = pool.pick()
    assert second != first
    pool.assign(second)
    with pytest.raises(WriterCapExhausted):
        pool.pick()


def test_run_world_respects_the_cap_and_records_each_writer(
    world, world_plan, pack, tmp_path
) -> None:
    plans = [d for d in writer_doc_plans(world_plan, WriterConfig()) if d.fact_ids][:4]
    backend = ScriptedBackend(obedient)
    session = _session(backend, total=4, models=DEFAULT_ROTATION[:4])
    report = run_world(
        world, world_plan, pack, session=session, out_dir=tmp_path, doc_ids=[p.id for p in plans]
    )
    assert report["writer_cap_documents"] == 1
    assert sorted(report["writer_docs"].values()) == [1, 1, 1, 1]
    labels = [json.loads(line) for line in (tmp_path / "labels.jsonl").read_text().splitlines()]
    assert labels and all(label["writer_model"] in DEFAULT_ROTATION for label in labels)


def test_failed_call_cools_the_model_and_switches(world, pack, decoys, call_plan) -> None:
    backend = ScriptedBackend(obedient, fail_models={DEFAULT_ROTATION[0]})
    doc, session = _write(call_plan, world, pack, decoys, backend)
    assert doc.status == "written"
    assert doc.calls[0].error and doc.calls[0].model == DEFAULT_ROTATION[0]
    assert doc.writer_model != DEFAULT_ROTATION[0]
    assert session.pool.rotation.is_cooling(DEFAULT_ROTATION[0])


def test_budget_precheck_stops_before_the_call(world, pack, decoys, call_plan) -> None:
    backend = ScriptedBackend(obedient)
    with pytest.raises(WriterBudgetExceeded) as exc:
        _write(call_plan, world, pack, decoys, backend, cap_usd=0.0)
    assert PAID_BUDGET_STOP_MARKER in str(exc.value)
    assert backend.calls == []


def test_spend_lines_land_on_the_meter_under_synthetic_v2(
    world, world_plan, pack, tmp_path
) -> None:
    meter = SpendMeter(tmp_path / "spend.json")
    plan = _first(world_plan, {DocKind.TICKET})
    backend = ScriptedBackend(obedient, cost=0.0025)
    session = _session(backend, meter=meter)
    run_world(world, world_plan, pack, session=session, out_dir=tmp_path / "out", doc_ids=[plan.id])
    snap = meter.snapshot()
    assert snap["per_corpus_usd"] == {CORPUS_KEY: 0.0025}
    assert snap["per_model_usd"] == {DEFAULT_ROTATION[0]: 0.0025}


def test_run_world_stops_cleanly_at_the_cap(world, world_plan, pack, tmp_path) -> None:
    plans = [d for d in writer_doc_plans(world_plan, WriterConfig()) if d.fact_ids][:12]
    # Each fake call costs exactly its list-price worst case, the most a real
    # call at list price can cost; the cap must hold anyway.
    backend = ScriptedBackend(obedient)
    session = _session(backend, cap_usd=0.003)

    def priced(*, model, system, user, max_tokens, temperature):
        reply = ScriptedBackend.complete(
            backend,
            model=model,
            system=system,
            user=user,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        cost = session.budget.worst_case_usd(model, len(system) + len(user), max_tokens)
        return type(reply)(reply.text, cost / session.budget.price_safety_factor, 0, 0)

    session.backend = type("Priced", (), {"complete": staticmethod(priced)})()
    report = run_world(
        world, world_plan, pack, session=session, out_dir=tmp_path, doc_ids=[p.id for p in plans]
    )
    assert report["stop_reason"] and PAID_BUDGET_STOP_MARKER in report["stop_reason"]
    assert session.budget.spent_usd <= 0.003
    assert 1 <= len(backend.calls) < len(plans)
    assert not (tmp_path / "docs" / f"{plans[-1].id}.json").exists()


def test_resume_skips_finished_documents(world, world_plan, pack, tmp_path) -> None:
    ids = [d.id for d in writer_doc_plans(world_plan, WriterConfig()) if d.fact_ids][:3]
    first = ScriptedBackend(obedient)
    run_world(world, world_plan, pack, session=_session(first), out_dir=tmp_path, doc_ids=ids)
    assert len(first.calls) == 3
    second = ScriptedBackend(obedient)
    report = run_world(
        world, world_plan, pack, session=_session(second), out_dir=tmp_path, doc_ids=ids
    )
    assert second.calls == []
    assert report["documents"]["resumed"] == 3
    # A changed input (the fingerprint) is rewritten, not reused.
    doc_path = tmp_path / "docs" / f"{ids[0]}.json"
    data = json.loads(doc_path.read_text())
    data["input_fingerprint"] = "stale"
    doc_path.write_text(json.dumps(data))
    third = ScriptedBackend(obedient)
    run_world(world, world_plan, pack, session=_session(third), out_dir=tmp_path, doc_ids=ids)
    assert len(third.calls) == 1


def test_event_cluster_context_reaches_the_later_document(
    world, world_plan, pack, tmp_path
) -> None:
    by_id = world_plan.doc_by_id()
    pair = None
    for d in writer_doc_plans(world_plan, WriterConfig()):
        for other in d.links.get("same_event", []):
            o = by_id[other]
            if d.fact_ids and (o.date, o.id) > (d.date, d.id) and o.role != DocRole.NEAR_DUPLICATE:
                pair = (d, o)
                break
        if pair:
            break
    assert pair is not None
    backend = ScriptedBackend(obedient)
    run_world(
        world,
        world_plan,
        pack,
        session=_session(backend),
        out_dir=tmp_path,
        doc_ids=[pair[0].id, pair[1].id],
    )
    later_user = backend.calls[-1].user
    assert "STORY SO FAR" in later_user


def _negation_decoy_plan(world_plan, decoys) -> tuple[DocPlan, str]:
    for d in sorted(world_plan.doc_plans, key=lambda d: (d.date, d.id)):
        if d.kind in {DocKind.CALL, DocKind.TICKET, DocKind.EMAIL} and d.fact_ids:
            for did in d.decoy_ids:
                if decoys[did].decoy_class_id == "negation":
                    return d, did
    raise LookupError("no negation decoy in an event call, ticket or email")


def test_negation_decoys_never_rule_out_a_competitor(world, world_plan, pack, decoys) -> None:
    """Decisions log, 2026-09-23: a negated competitor statement is a
    `competitor_mention` fact, so the negation decoy class is told not to
    make one, and check 5 drops a decoy span that does."""
    plan, did = _negation_decoy_plan(world_plan, decoys)
    assert decoys[did].forbidden_patterns == ["competitor_evaluation"]
    assert decoys[did].near_type.value != "competitor_mention"

    backend = ScriptedBackend(obedient)
    doc, _s = _write(plan, world, pack, decoys, backend)
    user = backend.calls[0].user
    item = user.split(f"{did}")[1].split("\n\n")[0]
    assert "must not name a competitor or any other vendor" in item
    assert "sounds like a request" in item
    assert any(d.decoy_id == did for d in doc.decoy_spans)

    for span in (
        "fixture honestly we are not looking at another pump vendor right now noted",
        "fixture we looked at them last spring but stuck with what we have noted",
    ):
        mutated = ScriptedBackend(
            lambda m, s, u, n, span=span: render_document(
                s, u, mutate={did: lambda item, _old: (span, None)}
            )
        )
        doc, _s = _write(plan, world, pack, decoys, mutated)
        outcome = next(o for o in doc.slot_outcomes if o.marker_id == did)
        assert outcome.status == "dropped"
        assert "c5:forbidden_pattern (competitor_evaluation)" in outcome.final_failures
        assert not any(d.decoy_id == did for d in doc.decoy_spans)


def test_near_duplicates_are_left_to_wp6(world, world_plan, pack, decoys) -> None:
    plan = next(d for d in world_plan.doc_plans if d.role == DocRole.NEAR_DUPLICATE)
    backend = ScriptedBackend(obedient)
    doc, _s = _write(plan, world, pack, decoys, backend)
    assert doc.status == "skipped" and backend.calls == []


# --------------------------------------------------------------------------
# Voice: a labeled span must sound like the person saying it (checks c9 to
# c13, c4 speaker side, c6 account). The bad spans below are verbatim labels
# from the first WP4 smoke (`var/data/synth_v2/smoke/`, seed 42), rejected by
# the 2026-09-22 review.
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# The first WP4 smoke's world, frozen: the voice tests below replay its
# verbatim spans, and the merged WP1/WP3 name pools no longer generate these
# names from seed 42. Only the entities those spans touch.
# --------------------------------------------------------------------------


def _first_smoke_world():
    from datetime import date

    from corpus.generator.v2.models import (
        Account,
        AccountKind,
        AccountTier,
        Employment,
        Event,
        EventType,
        Fact,
        Person,
        Persona,
        PersonSide,
        Product,
        Vendor,
        World,
    )
    from portolan.extract.schema import FactType

    persona = Persona(speech_register="operations", formality="neutral")

    def account(aid, name, kind, parent=None, alias=None):
        return Account(
            id=aid,
            crm_id=aid.upper(),
            name=name,
            kind=kind,
            parent=parent,
            tier=AccountTier.MID_MARKET,
            arr=100000.0,
            contract_start=date(2025, 6, 1),
            renewal_date=date(2026, 6, 1),
            region="Midwest",
            aliases=[alias] if alias else [],
        )

    def person(name, side, account_id, title):
        first, last = name.lower().split()
        return Person(
            id=f"person-{first}-{last}",
            name=name,
            side=side,
            account_id=account_id,
            role=title,
            title=title,
            email=f"{first[0]}{last}@example.test",
            chat_handle=f"@{first}{last[:3]}",
            tenure=Employment(start=date(2025, 1, 1)),
            persona=persona,
        )

    cust, vend = PersonSide.CUSTOMER, PersonSide.VENDOR
    accounts = [
        account("acct-marrowbrook", "Marrowbrook Health System", AccountKind.SYSTEM,
                alias="Marrowbrook"),
        account("acct-marrowbrook-2", "Marrowbrook Community Hospital 2", AccountKind.FACILITY,
                "acct-marrowbrook", "Marrowbrook"),
        account("acct-tangleview", "Tangleview Health Network", AccountKind.SYSTEM,
                alias="Tangleview"),
        account("acct-tangleview-1", "Tangleview Community Hospital 1", AccountKind.FACILITY,
                "acct-tangleview", "Tangleview"),
        account("acct-bramblemont", "Bramblemont Health Network", AccountKind.SYSTEM,
                alias="Bramblemont"),
        account("acct-bramblemont-2", "Bramblemont Community Hospital 2", AccountKind.FACILITY,
                "acct-bramblemont", "Bramblemont"),
    ]  # fmt: skip
    people = [
        person("Igka Kuzyeva", cust, "acct-marrowbrook-2", "VP Supply Chain"),
        person("Kawan Gorbova", cust, "acct-marrowbrook-2", "ICU Nurse Manager"),
        person("Callyn Petrvich", cust, "acct-tangleview-1", "Director of Nursing"),
        person("Boqi Goldmore", cust, "acct-tangleview-1", "Director of Nursing"),
        person("Haisa Farouk", cust, "acct-bramblemont-2", "Biomed Engineering Lead"),
        person("Leian Habib", vend, None, "Product Manager"),
        person("Cardro Lin", vend, None, "Support Engineer"),
    ]
    by_name = {p.name: p.id for p in people}
    # (id, type, account, speaker, claim, product, date); claims verbatim
    facts_src = [
        ("F00041", FactType.USAGE, "acct-marrowbrook-2", "Igka Kuzyeva",
         "Marrowbrook Community Hospital 2 runs Wayfinder T3 across its ICU and med-surg "
         "units as reported at the 2026-02-02 QBR.", "Wayfinder T3", date(2026, 2, 2)),
        ("F00042", FactType.PRAISE, "acct-marrowbrook-2", "Igka Kuzyeva",
         "Marrowbrook Community Hospital 2's nursing director praised Wayfinder T3 uptime "
         "at the 2026-02-02 QBR.", "Wayfinder T3", date(2026, 2, 2)),
        ("F00043", FactType.QUESTION, "acct-marrowbrook-2", "Igka Kuzyeva",
         "Marrowbrook Community Hospital 2 asked whether Wayfinder T3 supports a planned EHR "
         "migration; unanswered as of 2026-02-02.", "Wayfinder T3", date(2026, 2, 2)),
        ("F00268", FactType.COMMITMENT, "acct-tangleview-1", "Cardro Lin",
         "Harborlight Health Systems owes Tangleview Community Hospital 1 the Wayfinder Link "
         "fix committed for 2026-01-28.", "Wayfinder Link", date(2026, 1, 28)),
        ("F00269", FactType.RISK, "acct-tangleview-1", "Callyn Petrvich",
         "Tangleview Community Hospital 1 flagged that Harborlight Health Systems's "
         "2026-01-28 commitment on Wayfinder Link is now overdue.", "Wayfinder Link",
         date(2026, 1, 28)),
        ("F00178", FactType.PRICING, "acct-tangleview", "Boqi Goldmore",
         "Tangleview Health Network's CFO pushed back on the proposed Wayfinder Care renewal "
         "price as of 2026-03-24.", "Wayfinder Care", date(2026, 3, 24)),
        ("F00199", FactType.USAGE, "acct-bramblemont-2", "Haisa Farouk",
         "Harborlight Health Systems shipped a Wayfinder Care firmware update that Bramblemont "
         "Community Hospital 2 installed as of 2026-02-02.", "Wayfinder Care",
         date(2026, 2, 2)),
    ]  # fmt: skip
    products = {
        name: Product(
            id=f"product-{name.lower().replace(' ', '-')}",
            name=name,
            aliases=aliases,
            kind="device",
        )
        for name, aliases in (
            ("Wayfinder", []),
            ("Wayfinder T2", ["T2"]),
            ("Wayfinder T3", ["T3"]),
            ("Wayfinder Link", []),
            ("Wayfinder Care", []),
        )
    }
    names = {a.id: a.name for a in accounts}
    events, facts = [], []
    for n, (fid, ftype, aid, speaker, claim, product, on) in enumerate(facts_src, start=1):
        events.append(Event(id=f"E{n:05d}", date=on, type=EventType.QBR, account_id=aid))
        facts.append(
            Fact(
                id=fid,
                fact_type=ftype,
                claim=claim,
                entities=[aid, products[product].id],
                speaker=by_name[speaker],
                date=on,
                account_id=aid,
                event_id=f"E{n:05d}",
                anchors=[names[aid], product],
            )
        )
    vendor = Vendor(
        id="vendor-harborlight",
        name="Harborlight Health Systems",
        vertical="medical_devices",
        domain="harborlight.com",
        message_id_domain="harborlight.com",
        products=list(products.values()),
    )
    return World(
        id="world-first-smoke",
        seed=42,
        vertical="medical_devices",
        split="train",
        window_start=date(2026, 1, 1),
        window_end=date(2026, 8, 31),
        vendor=vendor,
        accounts=accounts,
        people=people,
        events=events,
        facts=facts,
        arcs=[],
    )


@pytest.fixture(scope="module")
def smoke_world():
    return _first_smoke_world()


def _person(world, name):
    return next(p for p in world.people if p.name == name)


def _voice(world, *, kind, speaker, text, account, fact_id=None):
    from corpus.generator.v2.verify import entity_words_for
    from corpus.generator.v2.voice import voice_failures

    facts = {f.id: f for f in world.facts}
    claim = facts[fact_id].claim if fact_id else None
    return {
        f.split(" ")[0]
        for f in voice_failures(
            text=text,
            kind=kind,
            world=world,
            speaker=_person(world, speaker) if speaker else None,
            account_ids=[account],
            claim=claim,
            entity_words=entity_words_for(world),
            other_names=tuple(c.name for c in world.vendor.competitors),
        )
    }


SMOKE_BAD_SPANS = [
    # (fact, kind, speaker, account, verbatim smoke span, codes it must fail)
    (
        "F00041",
        DocKind.CALL,
        "Igka Kuzyeva",
        "acct-marrowbrook-2",
        "Let me confirm: Marrowbrook Community Hospital 2 is currently using Wayfinder T3 in the "
        "ICU and med-surg units, as we discussed during the QBR on 2026-02-02.",
        {"c9:iso_date", "c11:own_org_named", "c12:claim_shaped"},
    ),
    (
        "F00042",
        DocKind.CALL,
        "Igka Kuzyeva",
        "acct-marrowbrook-2",
        "The nursing director at Marrowbrook Community Hospital 2 highlighted during the QBR how "
        "the uptime of Wayfinder T3 has reduced alert fatigue in our ICU, which is a significant "
        "improvement.",
        {"c11:own_org_named", "c12:claim_shaped"},
    ),
    (
        "F00043",
        DocKind.CALL,
        "Igka Kuzyeva",
        "acct-marrowbrook-2",
        "Marrowbrook Community Hospital 2 has asked if Wayfinder T3 supports their planned EHR "
        "migration. We’re aware of the query, but no detailed timeline has been provided yet.",
        {"c11:own_org_named", "c12:claim_shaped", "c13:pronoun_side"},
    ),
    (
        "F00268",
        DocKind.TICKET,
        "Cardro Lin",
        "acct-tangleview-1",
        "Harborlight Health Systems owes Tangleview Community Hospital 1 the Wayfinder Link fix "
        "that was promised by 2026-01-28.",
        {"c9:iso_date", "c10:both_parties_named", "c11:own_org_named", "c12:claim_shaped"},
    ),
    (
        "F00269",
        DocKind.TICKET,
        "Callyn Petrvich",
        "acct-tangleview-1",
        "Tangleview Community Hospital 1 has noted that Harborlight Health Systems's Wayfinder "
        "Link fix is behind schedule.",
        {"c10:both_parties_named", "c11:own_org_named", "c12:claim_shaped"},
    ),
    (
        # A customer (ICU nurse manager) about her own hospital's plans.
        "F00043",
        DocKind.CALL,
        "Kawan Gorbova",
        "acct-marrowbrook-2",
        "Well, we’ve had a query about whether the T3 supports their upcoming EHR migration.",
        {"c13:pronoun_side"},
    ),
    (
        # A vendor product manager claiming the customer's ICU.
        "F00042",
        DocKind.CALL,
        "Leian Habib",
        "acct-marrowbrook-2",
        "The uptime of the T3 has reduced alert fatigue in our ICU.",
        {"c13:pronoun_side"},
    ),
    (
        # The dropped CRM note's span: a vendor note naming the vendor.
        "F00199",
        DocKind.CRM_NOTE,
        None,
        "acct-bramblemont-2",
        "Harborlight Health Systems shipped a Wayfinder Care firmware update that Bramblemont "
        "Community Hospital 2 installed as of 2026-02-02.",
        {"c9:iso_date", "c10:both_parties_named", "c11:own_org_named", "c12:claim_shaped"},
    ),
]


@pytest.mark.parametrize(
    "fact_id, kind, speaker, account, text, codes",
    SMOKE_BAD_SPANS,
    ids=[f"{row[0]}-{row[2] or 'note'}" for row in SMOKE_BAD_SPANS],
)
def test_voice_checks_reject_the_first_smoke_spans(
    smoke_world, fact_id, kind, speaker, account, text, codes
) -> None:
    got = _voice(
        smoke_world, kind=kind, speaker=speaker, text=text, account=account, fact_id=fact_id
    )
    assert codes <= got, got


NATURAL_SPANS = [
    # Hand-written test fixtures (never corpus text): the same facts said
    # the way these people would say them.
    ("F00041", DocKind.CALL, "Igka Kuzyeva", "acct-marrowbrook-2",
     "we have the T3 running in the ICU and on med-surg for a while now"),
    ("F00042", DocKind.CALL, "Igka Kuzyeva", "acct-marrowbrook-2",
     "honestly the uptime on the T3 has been great, our nursing director brought it up at the QBR"),
    ("F00043", DocKind.CALL, "Igka Kuzyeva", "acct-marrowbrook-2",
     "we still have not heard back on whether the T3 will play nice with our EHR move"),
    ("F00268", DocKind.TICKET, "Cardro Lin", "acct-tangleview-1",
     "I know we still owe you the Link fix, we promised it for today and we are behind."),
    ("F00269", DocKind.TICKET, "Callyn Petrvich", "acct-tangleview-1",
     "honestly that Link fix you promised us is late now and people here are getting antsy"),
    ("F00199", DocKind.CRM_NOTE, None, "acct-bramblemont-2",
     "Bramblemont got the Care firmware update installed today, no issues"),
]  # fmt: skip


@pytest.mark.parametrize("fact_id, kind, speaker, account, text", NATURAL_SPANS)
def test_voice_checks_pass_natural_speech(
    smoke_world, fact_id, kind, speaker, account, text
) -> None:
    from corpus.generator.v2.verify import entity_words_for
    from corpus.generator.v2.voice import claim_overlap

    assert (
        _voice(smoke_world, kind=kind, speaker=speaker, text=text, account=account, fact_id=fact_id)
        == set()
    )
    claim = next(f.claim for f in smoke_world.facts if f.id == fact_id)
    assert claim_overlap(text, claim, entity_words_for(smoke_world))[0] < 0.5


def test_check9_iso_date_is_fine_in_a_formal_document(smoke_world) -> None:
    text = "Harborlight Health Systems will ship the Care update on 2026-02-02."
    assert "c9:iso_date" not in _voice(
        smoke_world, kind=DocKind.DOC, speaker=None, text=text, account="acct-bramblemont-2"
    )
    assert "c9:iso_date" in _voice(
        smoke_world,
        kind=DocKind.EMAIL,
        speaker="Cardro Lin",
        text=text,
        account="acct-tangleview-1",
    )


def test_check11_allows_a_first_person_frame(smoke_world) -> None:
    ok = "here at Marrowbrook we run the T3 on every ICU bed now"
    bad = "Marrowbrook runs the T3 on every ICU bed now"
    kw = {"kind": DocKind.CALL, "speaker": "Igka Kuzyeva", "account": "acct-marrowbrook-2"}
    assert "c11:own_org_named" not in _voice(smoke_world, text=ok, **kw)
    assert "c11:own_org_named" in _voice(smoke_world, text=bad, **kw)
    # The vendor's staff never name the vendor; naming the customer is fine.
    vendor = {"kind": DocKind.CALL, "speaker": "Leian Habib", "account": "acct-marrowbrook-2"}
    assert "c11:own_org_named" in _voice(
        smoke_world, text="Harborlight will send the patch", **vendor
    )
    assert _voice(smoke_world, text="Marrowbrook will get the patch from us", **vendor) == set()


def test_natural_date_phrases_resolve_back_through_check6() -> None:
    from datetime import date, timedelta

    from corpus.generator.v2.voice import natural_date_options

    doc_date = date(2026, 2, 2)
    for delta in range(-70, 71):
        fact_date = doc_date + timedelta(days=delta)
        options = natural_date_options(fact_date, doc_date)
        assert options and not any(re.search(r"\d{4}-\d", o) for o in options)
        for option in options:
            extras = coverage_extras(f"yeah we sorted that {option}, thanks", doc_date).split()
            assert fact_date.isoformat() in extras, (delta, option)


def _customer_fact_spec(world, pack, decoys, plan):
    from corpus.generator.v2.writer import slot_specs_for

    specs = slot_specs_for(render(plan, world, decoys.world_plan, pack), plan, world, decoys)
    return next(sp for sp in specs if sp.is_fact and sp.expected_side == "customer")


def test_prompt_gives_meaning_not_a_claim_sentence(world, pack, decoys, call_plan) -> None:
    system, user = _call_prompt(call_plan, world, pack, decoys)
    facts = {f.id: f for f in world.facts}
    for fid in call_plan.fact_ids:
        assert facts[fid].claim not in user
        assert facts[fid].claim not in system
    assert not re.search(r"\d{4}-\d{2}-\d{2}", user)
    assert "[the customer]" in user and "Meaning (notes, never words to reuse)" in user
    assert "VOICE" in system and "never names their own hospital" in system
    spec = _customer_fact_spec(world, pack, decoys, call_plan)
    speaker = next(p for p in world.people if p.id == spec.speaker_person_id)
    assert f"Point of view: {speaker.name.split()[0]} works for the customer" in user
    account = next(a.name for a in world.accounts if a.id == spec.account_id)
    # Never asked inside the marked sentence. A call's code header already
    # names the account, so the writer is not asked to say it at all.
    for line in user.split("\n"):
        if "Inside the marked sentence, name:" in line:
            assert account not in line
    assert (
        account
        in render(call_plan, world, decoys.world_plan, pack)
        .files[next(iter(render(call_plan, world, decoys.world_plan, pack).files))]
        .split("[SLOT", 1)[0]
    )
    assert f'- "{account}"' not in user
    # An email message is its own source with no header: the account is
    # named once in the message, outside the marked sentence.
    email = _first(decoys.world_plan, {DocKind.EMAIL}, facts=1)
    _sys, email_user = _call_prompt(email, world, pack, decoys)
    email_account = next(a.name for a in world.accounts if a.id == email.account_id)
    assert "Name once in the same message, outside the marked sentence" in email_user
    assert f'"{email_account}"' in email_user


def test_account_is_not_a_span_anchor_but_must_be_in_the_source(
    world, world_plan, pack, decoys
) -> None:
    plan = _first(world_plan, {DocKind.EMAIL}, facts=1)
    system, user = _call_prompt(plan, world, pack, decoys)
    raw = render_document(system, user)
    v, specs = _verify_raw(plan, world, pack, decoys, raw)
    fid = next(sp.marker_id for sp in specs if sp.is_fact)
    fact = next(f for f in world.facts if f.id == fid)
    account = next(a.name for a in world.accounts if a.id == fact.account_id)
    assert all(account not in sp.anchors for sp in specs)
    assert all(r.passed for r in v.results), [r.failures for r in v.results]
    v, _ = _verify_raw(plan, world, pack, decoys, raw.replace(account, "the hospital"))
    assert "c6:account_not_in_text" in _codes(v, fid)


def _call_with_nicknamed_first_fact(world, world_plan, pack, decoys):
    """A call whose first fact has one span anchor, a product with a
    nickname: the plan, the fact id, the anchor and the nickname."""
    for plan in sorted(world_plan.doc_plans, key=lambda d: (d.date, d.id)):
        if plan.kind != DocKind.CALL or plan.role != DocRole.EVENT or not plan.fact_ids:
            continue
        specs = build_slot_specs(render(plan, world, world_plan, pack), plan, world, decoys)
        first = next(sp for sp in specs if sp.is_fact)
        if (
            len(first.anchors) == 1
            and first.forms_for(first.anchors[0])
            and not first.reference_anchor
        ):
            return (
                plan,
                first.marker_id,
                first.anchors[0],
                first.forms_for(first.anchors[0])[0].text,
            )
    raise LookupError("no call with a nicknamed first anchor")


def test_two_turn_facts_and_nicknames_resolve(world, world_plan, pack, decoys) -> None:
    plan, fid, full, nick = _call_with_nicknamed_first_fact(world, world_plan, pack, decoys)
    system, user = _call_prompt(plan, world, pack, decoys)
    raw = render_document(system, user)
    line = next(line for line in raw.split("\n") if f"⟦{fid}⟧" in line)
    who, _, _text = line.partition(": ")
    other = next(name for name in parse_prompt(user)[0] if name != who)
    opener = "fixture words first."

    def with_turns(before: str, span: str) -> str:
        new = f"{other}: {before}\n{who}: {opener} ⟦{fid}⟧{span}⟦/{fid}⟧ tail"
        return raw.replace(line, new)

    def c3(text: str) -> set[str]:
        v, _ = _verify_raw(plan, world, pack, decoys, text)
        return {c for c in _codes(v, fid) if c.startswith("c3")}

    # The product named in the turn just before; the marked answer refers back.
    assert c3(with_turns(f"fixture question about the {full} today?", "yes it runs fine")) == set()
    # A nickname with the full name said earlier.
    assert c3(with_turns(f"fixture question about the {full}?", f"the {nick} runs fine")) == set()
    # A nickname with the full name never said before it.
    assert "c3:reference_unresolvable" in c3(raw.replace(full, nick))
    # Named earlier in the same turn but not in the marked sentence: the
    # label would miss part of the fact (seen live 2026-09-22, "Chenping
    # Clayland ... has left." then a marked "We have not yet identified a
    # replacement champion").
    same_turn = raw.replace(
        line, f"{who}: {opener} fixture note on the {full} first. ⟦{fid}⟧it runs fine⟦/{fid}⟧ tail"
    )
    assert "c3:anchor_missing" in c3(same_turn)
    # Nothing in the span and nothing in the turn before.
    assert "c3:anchor_missing" in c3(with_turns("fixture small talk", "it runs fine"))


def test_check4_speaker_side(world, pack, decoys, call_plan) -> None:
    spec = _customer_fact_spec(world, pack, decoys, call_plan)
    system, user = _call_prompt(call_plan, world, pack, decoys)
    v, _ = _verify_raw(call_plan, world, pack, decoys, render_document(system, user))
    assert "c4:speaker_side" not in _codes(v, spec.marker_id)
    cast, _a, _p = build_cast(call_plan, world, [])
    vendor = next(p.name for p in cast if p.side.value == "vendor")
    raw = render_document(system, user, mutate={spec.marker_id: lambda item, span: (span, vendor)})
    v, _ = _verify_raw(call_plan, world, pack, decoys, raw)
    assert {"c4:speaker_side", "c4:wrong_speaker"} <= _codes(v, spec.marker_id)


def test_departed_and_unplanned_speakers(world, pack, decoys, call_plan) -> None:
    import dataclasses
    from datetime import timedelta

    from corpus.generator.v2.writer import _replace_departed_speaker, departed_before
    from corpus.generator.v2.writer_prompt import PromptContext, build_initial_prompt

    people = world.person_by_id()
    spec = _customer_fact_spec(world, pack, decoys, call_plan)
    gone = people[spec.speaker_person_id]
    # A speaker who already left is replaced by a colleague still there.
    new = _replace_departed_speaker(
        spec, call_plan, world, {gone.id: call_plan.date - timedelta(days=1)}
    )
    assert new.speaker_person_id not in (None, gone.id)
    assert people[new.speaker_person_id].side == gone.side
    # A contact-change fact's subject counts as departed after its date.
    change = next(
        f
        for f in world.facts
        if f.fact_type.value == "contact_change"
        and any(e in people and e != f.speaker for e in f.entities)
    )
    subject = next(e for e in change.entities if e in people and e != change.speaker)
    left_on = min(
        f.date
        for f in world.facts
        if f.fact_type.value == "contact_change" and subject in f.entities
    )
    assert subject in departed_before(world, left_on + timedelta(days=1))
    assert subject not in departed_before(world, left_on)
    # With no speaker, the prompt names the side instead of a person.
    unplanned = dataclasses.replace(spec, speaker_person_id=None)
    cast, author, _p = build_cast(call_plan, world, [unplanned])
    ctx = PromptContext(call_plan, world, pack, [unplanned], cast, author, want_subject=False)
    _system, user = build_initial_prompt(ctx)
    assert f"{spec.marker_id} said by one of the customer people" in user


def test_system_account_speaks_through_its_facilities(world, world_plan) -> None:
    from corpus.generator.v2.writer import account_contacts

    system = next(a for a in world.accounts if a.kind.value == "system")
    assert not [p for p in world.people if p.account_id == system.id]
    contacts = account_contacts(world, system.id)
    children = {a.id for a in world.accounts if a.parent == system.id}
    assert contacts and all(p.account_id in children for p in contacts)
    plan = next(
        (d for d in world_plan.doc_plans if d.account_id == system.id and d.kind == DocKind.CALL),
        None,
    )
    if plan is not None:
        cast, _a, _p = build_cast(plan, world, [])
        assert any(p.side.value == "customer" for p in cast)


def test_changed_checks_invalidate_the_resume_fingerprint(
    world, pack, decoys, call_plan, monkeypatch
) -> None:
    from corpus.generator.v2 import writer

    rendered = render(call_plan, world, decoys.world_plan, pack)
    before = writer.input_fingerprint(rendered, call_plan, world, decoys)
    monkeypatch.setattr(writer, "CHECKS_VERSION", "a-changed-check")
    assert writer.input_fingerprint(rendered, call_plan, world, decoys) != before


def test_a_rewrite_that_drops_removes_the_old_files(world, world_plan, pack, tmp_path) -> None:
    plan = next(d for d in writer_doc_plans(world_plan, WriterConfig()) if d.kind == DocKind.CALL)
    run_world(
        world,
        world_plan,
        pack,
        session=_session(ScriptedBackend(obedient)),
        out_dir=tmp_path,
        doc_ids=[plan.id],
    )
    doc_path = tmp_path / "docs" / f"{plan.id}.json"
    files = list(json.loads(doc_path.read_text())["files"])
    assert files and all((tmp_path / "files" / rel).exists() for rel in files)
    data = json.loads(doc_path.read_text())
    data["input_fingerprint"] = "stale"
    doc_path.write_text(json.dumps(data))
    broken = ScriptedBackend(lambda m, s, u, n: "Zelda Quartermaine: hello\nOrvil Pike: hi")
    run_world(
        world, world_plan, pack, session=_session(broken), out_dir=tmp_path, doc_ids=[plan.id]
    )
    assert json.loads(doc_path.read_text())["status"] == "dropped"
    assert not any((tmp_path / "files" / rel).exists() for rel in files)


def test_check12_needs_a_third_person_party_or_a_pasted_template(smoke_world) -> None:
    # Live 2026-09-22 (writer text, used here as a fixture): faithful, in the
    # speaker's voice, keeps the content words, names its own side with a
    # first-person frame. Must pass.
    faithful = (
        "The way our CFO framed it today, the proposed Wayfinder Care renewal price just "
        "doesn't line up with what we're comfortable committing to this cycle, and we need to "
        "see some movement before we can treat that as a done deal over here at Tangleview "
        "Health Network."
    )
    kw = {"kind": DocKind.EMAIL, "speaker": "Boqi Goldmore", "account": "acct-tangleview"}
    assert _voice(smoke_world, text=faithful, fact_id="F00178", **kw) == set()
    # Live 2026-09-22: the meaning note pasted verbatim. Must fail.
    pasted = "We run Wayfinder T3 across ICU and med-surg units reported at QBR."
    got = _voice(
        smoke_world,
        kind=DocKind.CALL,
        speaker="Igka Kuzyeva",
        account="acct-marrowbrook-2",
        text=pasted,
        fact_id="F00041",
    )
    assert "c12:claim_shaped" in got


def test_required_terms_find_whole_numbers_and_trim_leading_words() -> None:
    from datetime import date

    from corpus.generator.v2.verify import required_terms_for, rule_claim

    on = date(2026, 2, 7)
    # WP2 template text: "20" once resolved to the token "2026-02-07" and asked
    # the writer for an ISO date.
    claim = (
        "After the 2026-02-07 incident, Willowmere North Medical Center adopted a stricter "
        "Alderline T3 use policy across their network of 20 days."
    )
    terms = required_terms_for(
        claim, ["Alderline T3"], on, on, names=["Willowmere North Medical Center"]
    )
    assert terms == ["Willowmere North Medical Center", "20"]
    qbr = "The QBR on 2026-04-01 featured a glowing endorsement of Alderline T2 from Bogka Guo."
    assert required_terms_for(qbr, ["Alderline T2", "Bogka Guo"], on, on) == ["QBR"]
    # A product with a lowercase suffix: the text names the product.
    assert rule_claim("took over Alderline T2-related work and med-surg units") == (
        "took over Alderline T2 related work and med-surg units"
    )


def test_check12_in_a_vendor_note_only_the_vendor_name_counts(smoke_world) -> None:
    # A rep's CRM note names the customer in the third person as a matter of
    # course; the same sentence naming the vendor reads like the claim.
    kw = {"kind": DocKind.CRM_NOTE, "speaker": None, "account": "acct-bramblemont-2"}
    note = (
        "Bramblemont Community Hospital 2 installed the Wayfinder Care firmware update we shipped"
    )
    assert _voice(smoke_world, text=note, fact_id="F00199", **kw) == set()
    vendor_named = note.replace("we shipped", "Harborlight Health Systems shipped")
    assert "c12:claim_shaped" in _voice(smoke_world, text=vendor_named, fact_id="F00199", **kw)


# Live spans from the merged smoke (2026-09-22, seed-42 world on main's WP3),
# hand-sorted into pastes of the meaning note and natural restatements.
MERGED_SMOKE_PASTES = [
    ("F00249", "Callyn Petrin shared an internal report on January 28 showing Knollwell "
     "Biomedical outperforming you in certain Alderline Link use cases"),
    ("F00202", "- positive feedback from us highlighted how Alderline Link release simplified "
     "care workflows yesterday"),
    ("F00390", "- We will now require 6 pre-approval for any changes to Care configuration, per "
     "new policy today."),
    ("F00391", "Calford informed you of policy change that will impact 3 of training needs for "
     "Care."),
]  # fmt: skip
MERGED_SMOKE_NATURAL = [
    ("F00250", "I almost said we were satisfied with performance, but no, we're actually "
     "considering Knollwell Biomedical for Alderline Link replacement due to ongoing issues"),
    ("F00050", "We experienced a delay of 4 days for go-live of T2 due to setup issues, which "
     "was frustrating for everyone."),
    ("F00054", "We had multiple reports from our team about T3 system lockups on March 4, and we "
     "escalated those to engineering already."),
    ("F00267", "We've reassigned our team today to meet the pending commitment on Alderline Care "
     "within 10 days."),
]  # fmt: skip


def test_c5_and_c12_reject_pastes_and_keep_natural_restatements(world) -> None:
    from corpus.generator.v2.verify import entity_words_for
    from corpus.generator.v2.voice import is_claim_shaped

    ew = entity_words_for(world)
    claims = {f.id: f.claim for f in world.facts}
    for fid, text in MERGED_SMOKE_PASTES:
        assert is_claim_shaped(text, claims[fid], ew, names_party=False) is not None, fid
    for fid, text in MERGED_SMOKE_NATURAL:
        assert is_claim_shaped(text, claims[fid], ew, names_party=False) is None, fid
        assert not is_claim_copy(text, claims[fid], ew), fid


# --------------------------------------------------------------------------
# Check 14: content nobody scripted. Real sentences from the WP4 smokes
# (2026-09-22), the kind WP5's blind pass counted as unlabeled facts.
# --------------------------------------------------------------------------

UNSCRIPTED_SMOKE_SENTENCES = [
    ("promise", "I'll send out the meeting invite."),
    ("promise", "And we'll send over the revised pricing breakdown by end of day today."),
    ("number", "We're rolling back the update on the 30th, and we'll push a patch by the 1st."),
    ("name", "T3 has API hooks for major EHR systems, including Epic and Cerner."),
    ("number", "Our team's already confirmed version 2.8.1 is loaded on all units."),
    ("name", "We're moving everything onto MedFlow next spring."),
    ("date", "The link in the main menu points to a version dated January 28."),
]
NEUTRAL_COLOR = [
    "Morning, Leian. It's been a bit chilly, but we're managing. Thanks for asking.",
    "Let me share my screen so you can see where that setting is.",
    "Can you hear me okay? My audio keeps cutting out.",
    "The alarm settings are under the Admin menu, second tab.",
    "Thanks everyone, have a good weekend.",
]


@pytest.mark.parametrize("kind, sentence", UNSCRIPTED_SMOKE_SENTENCES)
def test_check14_flags_unscripted_content(world, kind, sentence) -> None:
    from corpus.generator.v2.voice import unscripted_signals

    [signal] = unscripted_signals([sentence], world, allowed_numbers=set())
    assert kind in signal.split(":", 1)[0].split("+"), signal


def test_check14_passes_neutral_color(world) -> None:
    from corpus.generator.v2.voice import unscripted_signals

    assert unscripted_signals(NEUTRAL_COLOR, world, allowed_numbers=set()) == []
    # A number a scripted claim carries is not an invention.
    assert unscripted_signals(["That is 15% of staff."], world, allowed_numbers={"15"}) == []


def test_unscripted_content_triggers_one_regeneration_with_do_not_state(
    world, pack, decoys, call_plan
) -> None:
    extra = "I'll send the full report by Friday. We'll push a patch by the 1st."

    def reply(model, system, user, n):
        if "REWRITE THESE UNITS" in user:
            return repair_reply(user)
        text = render_document(system, user)
        if n == 0:
            first_line = text.split("\n", 1)[0]
            text = text.replace(first_line, f"{first_line} {extra}", 1)
        return text

    backend = ScriptedBackend(reply)
    doc, _s = _write(call_plan, world, pack, decoys, backend)
    assert [c.call_kind for c in doc.calls] == ["initial", "regenerate"]
    assert (
        "do not state it" in backend.calls[1].user and "report by Friday" in backend.calls[1].user
    )
    assert doc.final_call_kind == "regenerate" and doc.status == "written"
    assert not any(n.startswith("unscripted") for n in doc.notes)


def test_a_fact_told_again_in_other_words_is_stated_twice(world, pack, decoys, call_plan) -> None:
    system, user = _call_prompt(call_plan, world, pack, decoys)
    fid = call_plan.fact_ids[0]
    raw = render_document(system, user)
    line = next(line for line in raw.split("\n") if f"⟦{fid}⟧" in line)
    span = line.split(f"⟦{fid}⟧", 1)[1].split(f"⟦/{fid}⟧", 1)[0]
    who = line.split(": ", 1)[0]
    # The same content words in another order, unmarked, in a later turn.
    retold = " ".join(reversed(span.split())) + "."
    v, _ = _verify_raw(call_plan, world, pack, decoys, raw + f"{who}: again {retold}\n")
    r = next(r for r in v.results if r.spec.marker_id == fid)
    assert "c1:stated_twice" in _codes(v, fid) and r.repeat_units
    # A question about it is not a second telling.
    v, _ = _verify_raw(call_plan, world, pack, decoys, raw + f"{who}: was it {retold[:-1]}?\n")
    assert "c1:stated_twice" not in _codes(v, fid)


def test_a_truncated_reply_never_ships(world, pack, decoys, call_plan) -> None:
    # Live 2026-09-22: a regeneration stopped after 67 tokens and a 35-word
    # email, ending mid-sentence, outranked a longer draft and shipped.
    backend = ScriptedBackend(lambda m, s, u, n: render_document(s, u).split("\n", 1)[0] + "\n")
    doc, _s = _write(call_plan, world, pack, decoys, backend)
    assert doc.status == "dropped" and "truncated" in (doc.reason or "")
    assert not doc.files and not doc.labels


def test_a_markerless_draft_never_ships_or_outranks_labeled_facts(
    world, pack, decoys, call_plan
) -> None:
    # Live 2026-09-22: the first draft verified 6 of 7 facts; the
    # regeneration carried no markers at all and still won, shipping seven
    # scripted facts unlabeled.
    fid = call_plan.fact_ids[0]
    extra = "I'll send the full report by Friday. We'll push a patch by the 1st."

    def reply(model, system, user, n):
        if "REWRITE THESE UNITS" in user:
            return "=== U1 ===\nnothing useful\n"
        text = render_document(system, user, mutate={fid: lambda item, span: ("", None)})
        if n == 0:
            first_line = text.split("\n", 1)[0]
            return text.replace(first_line, f"{first_line} {extra}", 1)
        return re.sub(r"⟦/?[FD]\d+⟧", "", text)

    doc, _s = _write(call_plan, world, pack, decoys, ScriptedBackend(reply))
    assert [c.call_kind for c in doc.calls] == ["initial", "repair", "regenerate"]
    assert doc.final_call_kind == "initial" and doc.status == "written"
    assert len(doc.labels) == len(call_plan.fact_ids) - 1
    # Both drafts markerless: the document is dropped, never shipped.
    bare = ScriptedBackend(lambda m, s, u, n: re.sub(r"⟦/?[FD]\d+⟧", "", render_document(s, u)))
    doc, _s = _write(call_plan, world, pack, decoys, bare)
    assert doc.status == "dropped" and "no markers" in (doc.reason or "")
