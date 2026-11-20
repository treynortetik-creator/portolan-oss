"""Tests for world generator v2 work package 6, the noise stage
(`01-spec/world-generator-v2-2026-09-22.md` section 5, section 9's WP6 row).
Every model call is stubbed (`tests/v2_writer_fakes`, `tests/v2_noise_fakes`).

The core is the property test: over many seeds and every operation (and
every partial variant), each surviving label's text is exactly the new
prepared text at its offsets, sits inside one locator of one of its own
document's sources, keeps its text when no operation touched it, and every
label that did not survive is logged with a reason. The noised files are
prepared again from scratch and must give the same text. Also covered:
round trips for every kind, determinism, the spec's rates over a batch,
each operation's label policy, provenance of code-made words, and the
near-duplicate writer path (verification, supersession, regeneration,
budget, resume).
"""

from __future__ import annotations

import json
import random
import re
from datetime import date
from difflib import SequenceMatcher
from pathlib import Path

import pytest

from corpus.generator.v2.models import Split
from corpus.generator.v2.noise import (
    _LINK_FIELDS,
    ASR_WER_BAND,
    DOC_OP_ORDER,
    HOMOPHONES,
    NOISE_RATES,
    OP_ASR,
    OP_FORWARD,
    OP_MISFILED,
    OP_MISSING_LINK,
    OP_NEAR_DUPLICATE,
    OP_PARTIAL,
    OP_STALE_LINK,
    PARTIAL_HTML,
    PARTIAL_MISSING_COMMENT,
    PARTIAL_MOJIBAKE,
    PARTIAL_TRUNCATED,
    STALE_LINK_RATE,
    NoiseConfig,
    anchors_survive,
    apply_noise,
    eligible_ops,
    link_noise,
    mishear,
    mojibake,
    noise_document,
    plan_noise,
    select_docs,
    write_outputs,
)
from corpus.generator.v2.noise_neardup import (
    NearDupConfig,
    added_sentences,
    choose_change,
    value_present,
    write_near_duplicates,
)
from corpus.generator.v2.noise_text import (
    assign_original_keys,
    key_index,
    locate,
    parse_written,
    render_files,
)
from corpus.generator.v2.packs.schema import VerticalPack, load_pack
from corpus.generator.v2.planner import DocKind, DocPlan, DocRole, WorldDocPlan, plan_world
from corpus.generator.v2.renderers import render
from corpus.generator.v2.simulator import simulate_world
from corpus.generator.v2.verify import entity_words_for
from corpus.generator.v2.writer import (
    ALLOWED_WRITER_MODELS,
    CORPUS_KEY,
    DEFAULT_ROTATION,
    RunBudget,
    WriterConfig,
    WriterPool,
    WriterSession,
    WrittenDoc,
    write_document,
)
from portolan.training.spend import SpendMeter
from tests.v2_noise_fakes import neardup_reply, rich
from tests.v2_writer_fakes import ScriptedBackend

REPO_ROOT = Path(__file__).resolve().parents[1]
PACK_PATH = REPO_ROOT / "corpus" / "generator" / "v2" / "packs" / "medical_devices.yaml"

SEEDS = range(20)
DROP_REASONS = {"span_cut", "span_removed", "garbled_anchor_lost"}


@pytest.fixture(scope="module")
def pack() -> VerticalPack:
    return load_pack(PACK_PATH)


@pytest.fixture(scope="module")
def world(pack):
    return simulate_world(pack, seed=42, months=8, split=Split.TRAIN)


@pytest.fixture(scope="module")
def world_plan(world, pack) -> WorldDocPlan:
    return plan_world(world, pack)


@pytest.fixture(scope="module")
def decoys(world_plan):
    return {d.id: d for d in world_plan.decoys}


def _session(backend, *, cap_usd: float = 5.0, meter=None, total: int = 300) -> WriterSession:
    config = WriterConfig(models=DEFAULT_ROTATION)
    return WriterSession(
        backend=backend,
        pool=WriterPool(config.models, world_doc_total=total),
        budget=RunBudget(cap_usd=cap_usd, meter=meter),
        config=config,
    )


def _best(world_plan: WorldDocPlan, kind: DocKind, role: DocRole = DocRole.EVENT) -> DocPlan:
    return sorted(
        (d for d in world_plan.doc_plans if d.kind == kind and d.role == role),
        key=lambda d: (-min(len(d.fact_ids), 3), -len(d.decoy_ids), d.date, d.id),
    )[0]


@pytest.fixture(scope="module")
def fixture_docs(world, world_plan, pack, decoys):
    """One WP4-written event document per kind (stub writer), plus a hard
    negative (decoys only) and an off-topic document."""
    plans = [_best(world_plan, kind) for kind in DocKind]
    plans.append(_best(world_plan, DocKind.TICKET, DocRole.HARD_NEGATIVE))
    plans.append(_best(world_plan, DocKind.TEAMS, DocRole.OFF_TOPIC))
    docs: dict[str, WrittenDoc] = {}
    for plan in plans:
        doc = write_document(
            render(plan, world, world_plan, pack),
            plan,
            world,
            pack,
            decoys_by_id=decoys,
            session=_session(ScriptedBackend(rich)),
        )
        assert doc.status == "written", (plan.id, doc.fatal, doc.slot_outcomes)
        docs[plan.id] = doc
    return docs, {p.id: p for p in plans}


def _own(fixture_docs, world_plan) -> dict[str, WrittenDoc]:
    return dict(fixture_docs[0])


def _prepared(doc) -> dict[str, str]:
    return {p.source_id: p.text for p in doc.prepared_sources}


def _regions(doc, plan: DocPlan) -> dict[str, list[tuple[str, int, int]]]:
    """Prepare the noised document's own files again from scratch: the text
    must come back identical. Returns its locator regions."""
    text = parse_written(
        plan_id=doc.plan_id,
        kind=doc.kind,
        doc_date=plan.date,
        files=dict(doc.files),
        prepared=[(p.source_id, p.source_type, p.path) for p in doc.prepared_sources],
        prepared_text=_prepared(doc),
    )
    located = locate(text)
    assert [p.text for p in located.prepared] == [p.text for p in doc.prepared_sources]
    return {
        p.source_id: [(span.locator, span.start, span.end) for span in p.locators]
        for p in located.prepared
    }


def _check_invariants(before: WrittenDoc, after, plan: DocPlan) -> None:
    """The properties every noised document must hold."""
    prepared = _prepared(after)
    regions = _regions(after, plan)
    for x in [*after.labels, *after.decoy_spans]:
        assert x.doc_id in prepared, "a label points outside its own document"
        text = prepared[x.doc_id]
        assert 0 <= x.char_start < x.char_end <= len(text)
        assert text[x.char_start : x.char_end] == x.text
        inside = [loc for loc, s, e in regions[x.doc_id] if s <= x.char_start and x.char_end <= e]
        assert len(inside) == 1, ("span crosses a locator boundary", x)
        if before.kind != DocKind.CRM_NOTE:
            assert x.prepared_locator == inside[0]
    # Accounting: each input label survives once as a primary or is logged.
    primaries = [x for x in after.labels if x.copy_of is None]
    fact_drops = [d for d in after.label_drops if d.slot_kind == "fact"]
    assert len(primaries) + len(fact_drops) == len(before.labels)
    decoys = [x for x in after.decoy_spans if x.copy_of is None]
    decoy_drops = [d for d in after.label_drops if d.slot_kind == "decoy"]
    assert len(decoys) + len(decoy_drops) == len(before.decoy_spans)
    for d in after.label_drops:
        assert d.reason in DROP_REASONS and d.noise_op, d
    # Untouched spans keep their exact text; facts keep the true account.
    old = {(x.fact_id, x.slot_id): x for x in before.labels}
    for x in primaries:
        orig = old[(x.fact_id, x.slot_id)]
        if not x.noise:
            assert x.text == orig.text
        assert x.account_id == orig.account_id
    for x in after.labels:
        if x.copy_of:
            source = x.copy_of.split(":")[0]
            assert source in {y.doc_id for y in before.labels} and source != x.doc_id


# --------------------------------------------------------------------------
# Round trip
# --------------------------------------------------------------------------


def test_every_kind_round_trips_exactly(fixture_docs) -> None:
    docs, plans = fixture_docs
    kinds = {d.kind for d in docs.values()}
    assert {DocKind.CALL, DocKind.EMAIL, DocKind.TICKET, DocKind.SLACK, DocKind.TEAMS} <= kinds
    for doc in docs.values():
        plan = plans[doc.plan_id]
        text = parse_written(
            plan_id=doc.plan_id,
            kind=doc.kind,
            doc_date=plan.date,
            files=dict(doc.files),
            prepared=[(p.source_id, p.source_type, p.path) for p in doc.prepared_sources],
            prepared_text=_prepared(doc),
        )
        assert render_files(text) == doc.files
        located = locate(text)
        assert located.mismatches == 0
        assert [p.text for p in located.prepared] == [p.text for p in doc.prepared_sources]
        assign_original_keys(text, located)
        index = key_index(text)
        for x in [*doc.labels, *doc.decoy_spans]:
            assert all((x.doc_id, k) in index for k in range(x.char_start, x.char_end))


def test_no_operation_leaves_the_document_byte_identical(fixture_docs, world) -> None:
    docs, plans = fixture_docs
    for doc in docs.values():
        out = noise_document(doc, plans[doc.plan_id], world, [], config=NoiseConfig(seed=1))
        assert out.files == doc.files
        assert [x.text for x in out.labels] == [x.text for x in doc.labels]
        assert out.noise == []


# --------------------------------------------------------------------------
# The property test
# --------------------------------------------------------------------------


def _op_sets(doc: WrittenDoc, plan: DocPlan) -> list[tuple[list[str], str | None]]:
    ops = eligible_ops(doc, plan)
    sets: list[tuple[list[str], str | None]] = [([op], None) for op in ops if op != OP_PARTIAL]
    for variant in ("truncated", "missing_comment", "mojibake", "html_leftovers"):
        sets.append(([OP_PARTIAL], variant))
    sets.append((ops, None))
    return sets


def test_property_spans_stay_exact_over_seeds_and_operations(fixture_docs, world) -> None:
    docs, plans = fixture_docs
    entity_words = entity_words_for(world)
    applied: set[str] = set()
    runs = 0
    for doc in docs.values():
        plan = plans[doc.plan_id]
        for ops, variant in _op_sets(doc, plan):
            for seed in SEEDS:
                config = NoiseConfig(
                    seed=seed, partial_variants={doc.plan_id: variant} if variant else {}
                )
                out = noise_document(
                    doc, plan, world, ops, config=config, entity_words=entity_words
                )
                assert not [n for n in out.noise_notes if n.startswith("not noised")]
                assert not [n for n in out.noise_notes if "could not be placed" in n]
                _check_invariants(doc, out, plan)
                runs += 1
                for op in out.noise:
                    applied.add(op.type + ":" + str(op.params.get("variant", "")))
    assert runs > 1000
    assert {
        "asr:",
        "misfiled:",
        "forwarded_or_quoted:forward",
        "forwarded_or_quoted:quote",
        "partial_or_garbled:truncated",
        "partial_or_garbled:missing_comment",
        "partial_or_garbled:mojibake",
        "partial_or_garbled:html_leftovers",
    } <= applied


def test_noise_is_deterministic_for_a_fixed_seed(fixture_docs, world) -> None:
    docs, plans = fixture_docs
    changed = 0
    for doc in docs.values():
        plan = plans[doc.plan_id]
        ops = eligible_ops(doc, plan)
        a = noise_document(doc, plan, world, ops, config=NoiseConfig(seed=5))
        b = noise_document(doc, plan, world, ops, config=NoiseConfig(seed=5))
        c = noise_document(doc, plan, world, ops, config=NoiseConfig(seed=6))
        assert a.model_dump_json() == b.model_dump_json()
        changed += a.model_dump_json() != c.model_dump_json()
    assert changed >= len(docs) // 2


# --------------------------------------------------------------------------
# Rates
# --------------------------------------------------------------------------


def _stub_docs(world_plan: WorldDocPlan) -> dict[str, WrittenDoc]:
    docs = {}
    for plan in world_plan.doc_plans:
        if plan.role == DocRole.NEAR_DUPLICATE:
            continue
        files = (
            {f"email/{plan.id}-m1.txt": "", f"email/{plan.id}-m2.txt": ""}
            if plan.kind == DocKind.EMAIL
            else {f"x/{plan.id}": ""}
        )
        docs[plan.id] = WrittenDoc(
            plan_id=plan.id, kind=plan.kind, status="written", input_fingerprint="x", files=files
        )
    return docs


def test_selection_rates_land_in_the_spec_bands(world_plan) -> None:
    plans = world_plan.doc_by_id()
    docs = _stub_docs(world_plan)
    eligible = {
        OP_MISFILED: [d for d in docs.values() if plans[d.plan_id].account_id],
        OP_ASR: [d for d in docs.values() if d.kind == DocKind.CALL],
        OP_FORWARD: [d for d in docs.values() if d.kind == DocKind.EMAIL],
        OP_PARTIAL: list(docs.values()),
    }
    for seed in range(8):
        chosen = plan_noise(docs, plans, NoiseConfig(seed=seed))
        for op, rate in NOISE_RATES.items():
            n = len(eligible[op])
            got = sum(1 for d in eligible[op] if op in chosen[d.plan_id])
            assert got == int(rate * n + 0.5), (op, got, n)
            assert abs(got / n - rate) <= 0.5 / n + 1e-9
        for doc_id, ops in chosen.items():
            if OP_ASR in ops:
                assert docs[doc_id].kind == DocKind.CALL
            if OP_FORWARD in ops:
                assert docs[doc_id].kind == DocKind.EMAIL
            if OP_MISFILED in ops:
                assert plans[doc_id].account_id


def test_selection_is_order_independent_and_seeded() -> None:
    ids = [f"S-doc-{i:05d}" for i in range(200)]
    a = select_docs(ids, 0.2, 3, OP_FORWARD)
    b = select_docs(list(reversed(ids)), 0.2, 3, OP_FORWARD)
    assert a == b and len(a) == 40
    assert select_docs(ids, 0.2, 4, OP_FORWARD) != a


def test_asr_word_error_rate_stays_in_band(fixture_docs, world) -> None:
    docs, plans = fixture_docs
    call = next(d for d in docs.values() if d.kind == DocKind.CALL)
    lo, hi = ASR_WER_BAND
    for seed in range(20):
        out = noise_document(
            call, plans[call.plan_id], world, [OP_ASR], config=NoiseConfig(seed=seed)
        )
        (op,) = [o for o in out.noise if o.type == OP_ASR]
        assert lo <= op.params["wer"] <= hi, op.params
        assert op.params["errors"] == sum(op.params["by_error"].values())


def test_stale_or_missing_links_keep_the_spec_share(world_plan) -> None:
    ops, overrides = link_noise(world_plan, NoiseConfig(seed=11))
    missing_in = [n for p in world_plan.doc_plans for n in p.noise_ops if n.startswith("missing_")]
    wired = sum(len(getattr(r, f)) for r in world_plan.crm_accounts for f in _LINK_FIELDS)
    # Every link WP3 left unwired is still noise, now missing or stale; the
    # combined share is WP3's, the spec's "about 10%".
    assert len(ops) == len(missing_in)
    assert 0.07 <= len(ops) / (wired + len(ops)) <= 0.15
    stale = [op for op in ops if op.type == OP_STALE_LINK]
    target = int(STALE_LINK_RATE * len(missing_in) + 0.5)
    assert target - 2 <= len(stale) <= target
    for op in stale:
        p = op.params
        links = overrides[p["record_id"]][p["field"]]
        assert p["stale_target"] != p["true_target"]
        assert p["stale_target"] in links and p["true_target"] not in links
        plans = world_plan.doc_by_id()
        assert plans[p["stale_target"]].account_id != plans[p["true_target"]].account_id
        assert len(links) == len(set(links))
    for op in ops:
        if op.type == OP_MISSING_LINK:
            record = overrides[op.params["record_id"]]
            assert all(op.params["doc_id"] not in ids for ids in record.values())


# --------------------------------------------------------------------------
# Each operation's label policy
# --------------------------------------------------------------------------

_NUM_RE = re.compile(r"\d[\d,:/.\-]*|\b(?:not|no|never)\b|n[’']t\b", re.I)


def test_asr_never_changes_numerals_or_negations_and_flags_its_spans(fixture_docs, world) -> None:
    docs, plans = fixture_docs
    call = next(d for d in docs.values() if d.kind == DocKind.CALL)
    before_text = next(iter(call.files.values()))
    old = {x.fact_id: x.text for x in call.labels}
    touched = 0
    for seed in SEEDS:
        out = noise_document(
            call, plans[call.plan_id], world, [OP_ASR], config=NoiseConfig(seed=seed)
        )
        after_text = next(iter(out.files.values()))
        assert sorted(_NUM_RE.findall(before_text)) == sorted(_NUM_RE.findall(after_text))
        for label in out.labels:
            assert label.asr_in_span == (OP_ASR in label.noise)
            assert label.asr_in_span == (label.text != old[label.fact_id])
            touched += label.asr_in_span
    assert touched > 0


def test_forward_copies_labels_and_quote_changes_nothing_visible(fixture_docs, world) -> None:
    docs, plans = fixture_docs
    email = next(d for d in docs.values() if d.kind == DocKind.EMAIL)
    extra = {"noise", "remapped", "asr_in_span", "copy_of", "supersedes"}
    seen = set()
    for seed in range(40):
        out = noise_document(
            email, plans[email.plan_id], world, [OP_FORWARD], config=NoiseConfig(seed=seed)
        )
        (op,) = out.noise
        seen.add(op.params["variant"])
        dst = op.params["into_message"]
        if op.params["variant"] == "quote":
            assert [p.text for p in out.prepared_sources] == [
                p.text for p in email.prepared_sources
            ]
            assert [x.model_dump(exclude=extra) for x in out.labels] == [
                x.model_dump() for x in email.labels
            ]
            assert "wrote:\n>" in out.files[f"email/{dst}.txt"]
        else:
            src = op.params["forwarded_message"]
            assert "---------- Forwarded message ---------" in out.files[f"email/{dst}.txt"]
            originals = [x for x in email.labels if x.doc_id == src]
            copies = [x for x in out.labels if x.copy_of]
            assert len(copies) == len(originals)
            for c in copies:
                assert c.doc_id == dst and c.text in {x.text for x in originals}
                assert c.locator == c.prepared_locator and c.locator.startswith(dst)
    assert seen == {"forward", "quote"}


def test_truncation_and_missing_comment_drop_only_cut_spans(fixture_docs, world) -> None:
    docs, plans = fixture_docs
    reasons: set[tuple[str, str | None]] = set()
    for doc in docs.values():
        plan = plans[doc.plan_id]
        for variant in ("truncated", "missing_comment"):
            for seed in range(20):
                config = NoiseConfig(seed=seed, partial_variants={doc.plan_id: variant})
                out = noise_document(doc, plan, world, [OP_PARTIAL], config=config)
                reasons.update((d.reason, d.noise_op) for d in out.label_drops)
    assert ("span_removed", PARTIAL_TRUNCATED) in reasons
    assert ("span_removed", PARTIAL_MISSING_COMMENT) in reasons
    assert {r for r, _op in reasons} <= {"span_removed", "span_cut"}


def test_garbled_span_is_kept_only_if_its_anchors_survive(fixture_docs, world) -> None:
    docs, plans = fixture_docs
    ticket = next(d for d in docs.values() if d.kind == DocKind.TICKET)
    plan = plans[ticket.plan_id]
    ids = {x.fact_id for x in ticket.labels}
    # The fake's spans say "it’s"; make it an anchor, as a name with an
    # apostrophe would be. Mojibake and HTML entities both rewrite "’".
    strict = world.model_copy(
        update={
            "facts": [
                f.model_copy(update={"anchors": [*f.anchors, "it’s"]}) if f.id in ids else f
                for f in world.facts
            ]
        }
    )
    label = ticket.labels[0]
    assert anchors_survive(label.text, label.text, label.fact_id, strict)
    garbled = label.text.replace("it’s", "it&#39;s")
    assert garbled != label.text
    assert not anchors_survive(label.text, garbled, label.fact_id, strict)
    assert anchors_survive(label.text, garbled, label.fact_id, world)
    before = {x.slot_id: x.text for x in ticket.labels}
    outcomes: dict[str, set[str]] = {"strict": set(), "plain": set()}
    for variant in ("mojibake", "html_leftovers"):
        for seed in range(30):
            config = NoiseConfig(seed=seed, partial_variants={ticket.plan_id: variant})
            for name, w in (("strict", strict), ("plain", world)):
                out = noise_document(ticket, plan, w, [OP_PARTIAL], config=config)
                _check_invariants(ticket, out, plan)
                outcomes[name] |= {d.reason for d in out.label_drops}
                for x in out.labels:
                    if set(x.noise) & {PARTIAL_MOJIBAKE, PARTIAL_HTML}:
                        outcomes[name].add("kept_garbled")
                        assert anchors_survive(before[x.slot_id], x.text, x.fact_id, w)
    assert "garbled_anchor_lost" in outcomes["strict"]
    assert outcomes["plain"] == {"kept_garbled"}


def test_misfiled_keeps_true_account_and_changes_only_metadata(fixture_docs, world) -> None:
    docs, plans = fixture_docs
    ticket = next(d for d in docs.values() if d.kind == DocKind.TICKET)
    plan = plans[ticket.plan_id]
    out = noise_document(ticket, plan, world, [OP_MISFILED], config=NoiseConfig(seed=2))
    (op,) = out.noise
    assert op.type == OP_MISFILED
    assert out.true_account_id == plan.account_id != out.filed_account_id
    assert [p.text for p in out.prepared_sources] == [p.text for p in ticket.prepared_sources]
    payload = json.loads(next(iter(out.files.values())))
    assert payload["account"] == world.account_by_id()[out.filed_account_id].name
    assert all(x.account_id == plan.account_id for x in out.labels)
    assert "ticket_account_field" in op.params["surfaces"]


def test_mishear_and_mojibake_helpers() -> None:
    outs = {mishear("Corvane", random.Random(i)) for i in range(40)}
    assert "Carvane" in outs and "Corvane" not in outs
    assert all(o and o[0] == "C" for o in outs)
    assert mojibake("’") == "â€™" and mojibake("é") == "Ã©" and mojibake("a") == "a"
    assert set(HOMOPHONES["their"]) == {"there", "they're"}


_NOISE_WORDS = {"inaudible", "crosstalk", "p", "br", "amp", "quot", "forwarded", "message", "wrote"}


def test_code_noise_adds_no_new_words_beyond_noise_vocabulary(fixture_docs, world) -> None:
    """Provenance: code noise moves, drops, re-encodes or mis-transcribes
    writer words. Any word it adds is a noise tag, an HTML leftover, a
    homophone, a header word, or a misheard name close to a real one."""
    docs, plans = fixture_docs
    entity = set(entity_words_for(world))
    homophones = {
        piece
        for w in [*HOMOPHONES, *(x for group in HOMOPHONES.values() for x in group)]
        for piece in re.findall(r"[a-z]+", w)
    }

    def words(files: dict[str, str]) -> set[str]:
        # JSON files escape a newline as the two characters `\n`.
        return {
            w.lower()
            for content in files.values()
            for w in re.findall(r"[A-Za-z]+", content.replace("\\n", " "))
        }

    for doc in docs.values():
        plan = plans[doc.plan_id]
        before = words(doc.files)
        for seed in range(10):
            config = NoiseConfig(seed=seed, force=frozenset(DOC_OP_ORDER))
            out = noise_document(doc, plan, world, eligible_ops(doc, plan), config=config)
            after = words(out.files)
            for w in after - before:
                if w in _NOISE_WORDS or w in homophones:
                    continue
                assert any(SequenceMatcher(None, w, e).ratio() >= 0.7 for e in entity), (
                    doc.plan_id,
                    w,
                )


def test_world_run_writes_outputs_and_manifest(fixture_docs, world, world_plan, tmp_path) -> None:
    own = _own(fixture_docs, world_plan)
    run = apply_noise(
        world, world_plan, own, config=NoiseConfig(seed=3, force=frozenset(DOC_OP_ORDER))
    )
    write_outputs(run, tmp_path)
    labels = (tmp_path / "labels.jsonl").read_text().splitlines()
    assert len(labels) == sum(len(d.labels) for d in run.docs.values())
    manifest = json.loads((tmp_path / "noise_manifest.json").read_text())
    assert set(manifest["docs"]) == set(own)
    assert any(link["type"] == "missing_link" for link in manifest["links"])
    assert any(link["type"] == "stale_link" for link in manifest["links"])
    report = json.loads((tmp_path / "noise_report.json").read_text())
    assert report["documents_written"] == len(own)
    for doc_id in own:
        saved = json.loads((tmp_path / "docs" / f"{doc_id}.json").read_text())
        assert saved["noise"], doc_id


# --------------------------------------------------------------------------
# Near-duplicates (stubbed writer)
# --------------------------------------------------------------------------


def _neardup_backend(**kw) -> ScriptedBackend:
    def reply(model, system, user, n):  # noqa: ANN001, ANN202
        return neardup_reply(user, **kw)

    return ScriptedBackend(reply)


def test_choose_change_needs_the_value_in_the_span(world) -> None:
    fact = next(f for f in world.facts if re.search(r"\d{4}-\d{2}-\d{2}", f.claim))
    iso = re.search(r"\d{4}-\d{2}-\d{2}", fact.claim).group(0)
    assert choose_change(fact, "nothing here", fact.date, random.Random(0)) is None
    change = choose_change(fact, f"the fix lands {iso}", fact.date, random.Random(0))
    assert change is not None and change.value_kind == "date"
    assert change.new_value != iso and change.new_value in change.new_claim
    assert change.new_claim.replace(change.new_value, iso) == fact.claim
    assert value_present("percent", "15%", "we hit fifteen percent", fact.date)
    assert not value_present("percent", "25%", "we hit fifteen percent", fact.date)


def test_near_duplicate_is_labeled_and_supersedes(
    fixture_docs, world, world_plan, pack, tmp_path
) -> None:
    docs = _own(fixture_docs, world_plan)
    plans = world_plan.doc_by_id()
    meter = SpendMeter(tmp_path / "spend.json")
    backend = _neardup_backend()
    result = write_near_duplicates(
        world,
        world_plan,
        pack,
        docs,
        session=_session(backend, meter=meter),
        config=NearDupConfig(seed=1, limit=2),
        out_dir=tmp_path,
    )
    written = [d for d in result.docs.values() if d.status == "written"]
    assert len(written) == 2, [(d.plan_id, d.reason) for d in result.docs.values()]
    for nd in written:
        (op,) = [o for o in nd.noise if o.type == OP_NEAR_DUPLICATE]
        new_id, old_id = op.params["new_fact_id"], op.params["changed_fact_id"]
        prepared = _prepared(nd)
        changed = [x for x in nd.labels if x.fact_id == new_id]
        assert len(changed) == 1 and changed[0].supersedes == old_id
        for x in nd.labels:
            assert prepared[x.doc_id][x.char_start : x.char_end] == x.text
            assert x.label_source == "near_duplicate_writer"
            assert x.writer_model in ALLOWED_WRITER_MODELS
        fact = next(f for f in result.new_facts if f.id == new_id)
        assert fact.supersedes == old_id and op.params["new_value"] in fact.claim
        kind, new, old = op.params["value_kind"], op.params["new_value"], op.params["old_value"]
        on = date.fromisoformat(op.params["doc_date"])
        assert value_present(kind, new, changed[0].text, on)
        assert not value_present(kind, old, changed[0].text, on)
        assert op.params["jaccard_5gram"] > 0.3
        assert nd.kind == plans[op.params["shadow_id"]].kind  # the original's own kind
        # every word the writer units hold comes from a writer reply
        reply_words = set(re.findall(r"[A-Za-z]+", " ".join(c.raw_output for c in nd.calls)))
        units = parse_written(
            plan_id=nd.plan_id,
            kind=nd.kind,
            doc_date=on,
            files=dict(nd.files),
            prepared=[(p.source_id, p.source_type, p.path) for p in nd.prepared_sources],
            prepared_text=_prepared(nd),
        ).live_units(role="writer")
        assert units
        for u in units:
            assert set(re.findall(r"[A-Za-z]+", u.text)) <= reply_words, u.text
    assert {s["new_fact_id"] for s in result.supersessions} == {f.id for f in result.new_facts}
    assert meter.corpus_usd(CORPUS_KEY) == pytest.approx(
        sum(d.cost_usd for d in result.docs.values())
    )
    assert result.report["calls"] == len(backend.calls) == 2
    # resume: a finished near-duplicate on disk is reused, not rewritten
    (tmp_path / "docs").mkdir(exist_ok=True)
    for d in result.docs.values():
        (tmp_path / "docs" / f"{d.plan_id}.json").write_text(d.model_dump_json())
    again = _neardup_backend()
    second = write_near_duplicates(
        world,
        world_plan,
        pack,
        docs,
        session=_session(again),
        config=NearDupConfig(seed=1, limit=2),
        out_dir=tmp_path,
    )
    assert again.calls == []
    assert {k: d.status for k, d in second.docs.items()} == {
        k: d.status for k, d in result.docs.items()
    }


def test_near_duplicate_keeping_the_old_value_regenerates_then_drops(
    fixture_docs, world, world_plan, pack
) -> None:
    docs = _own(fixture_docs, world_plan)
    backend = _neardup_backend(keep_old=True)
    result = write_near_duplicates(
        world,
        world_plan,
        pack,
        docs,
        session=_session(backend),
        config=NearDupConfig(seed=1, limit=1),
    )
    (nd,) = result.docs.values()
    assert nd.status == "dropped" and "changed statement" in (nd.reason or "")
    assert [c.call_kind for c in nd.calls] == ["initial", "regenerate"]
    assert nd.calls[0].model != nd.calls[1].model
    assert result.new_facts == [] and result.supersessions == []
    failures = [f for o in nd.slot_outcomes for f in o.final_failures]
    assert any(f.startswith("n2:") for f in failures)
    assert any(f.startswith("n1:") for f in failures)


_LIVE_ADDED = (
    "Harborlight Health Systems remains fully committed to meeting the original January 28 "
    "deadline for the broader rollout, though the current focus is on resolving the outstanding "
    "integration challenges for Tangleview Community Hospital 1."
)


def test_added_sentence_is_caught() -> None:
    original = (
        "### Cardro Lin\nWe committed to delivering the fix by 2026-01-28, and despite our best "
        "efforts, we are behind schedule. ⟦F00268s⟧Harborlight Health Systems owes Tangleview "
        "Community Hospital 1 the Wayfinder Link fix that was promised by 2026-01-28.⟦/F00268s⟧"
    )
    revised = (
        "We committed to delivering the fix by 2026-02-16, and despite our best efforts, we are "
        "behind schedule. Harborlight Health Systems owes Tangleview Community Hospital 1 the "
        "Wayfinder Link fix that was promised by 2026-02-16."
    )
    assert added_sentences(original, revised) == []
    added = added_sentences(original, revised + " " + _LIVE_ADDED)
    assert len(added) == 1 and added[0].startswith("harborlight health systems remains")


def test_near_duplicate_with_an_added_sentence_is_dropped(
    fixture_docs, world, world_plan, pack
) -> None:
    docs = _own(fixture_docs, world_plan)
    backend = _neardup_backend(add_sentence=_LIVE_ADDED)
    result = write_near_duplicates(
        world,
        world_plan,
        pack,
        docs,
        session=_session(backend),
        config=NearDupConfig(seed=1, limit=1),
    )
    (nd,) = result.docs.values()
    assert nd.status == "dropped" and "n3:added_sentence" in (nd.reason or "")
    assert [c.call_kind for c in nd.calls] == ["initial", "regenerate"]
    assert "n3:added_sentence" in backend.calls[1].user  # the regeneration was told why
    assert result.new_facts == []


def test_near_duplicate_budget_stops_before_the_call(fixture_docs, world, world_plan, pack) -> None:
    docs = _own(fixture_docs, world_plan)
    backend = _neardup_backend()
    result = write_near_duplicates(
        world,
        world_plan,
        pack,
        docs,
        session=_session(backend, cap_usd=0.0),
        config=NearDupConfig(seed=1),
    )
    assert backend.calls == []
    assert result.report["stop_reason"] and result.docs == {}


def test_apply_noise_marks_the_original_of_a_near_duplicate(
    fixture_docs, world, world_plan, pack
) -> None:
    own = _own(fixture_docs, world_plan)
    session = _session(_neardup_backend())

    def near(written):  # noqa: ANN001, ANN202
        return write_near_duplicates(
            world,
            world_plan,
            pack,
            written,
            session=session,
            config=NearDupConfig(seed=1, limit=1),
        )

    run = apply_noise(world, world_plan, own, config=NoiseConfig(seed=3), near_duplicates=near)
    nds = [
        d
        for d in run.docs.values()
        if d.status == "written" and any(o.params.get("role") == "near_duplicate" for o in d.noise)
    ]
    assert nds
    shadow = nds[0].noise[0].params["shadow_id"]
    assert any(
        o.type == OP_NEAR_DUPLICATE and o.params["role"] == "original"
        for o in run.docs[shadow].noise
    )
    assert run.manifest["supersessions"] and run.manifest["new_facts"]
