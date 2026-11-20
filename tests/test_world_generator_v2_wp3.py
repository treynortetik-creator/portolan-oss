"""Tests for world generator v2 work package 3 (`01-spec/world-generator-
v2-2026-09-22.md` section 2): the doc planner (`corpus.generator.v2.
planner`) and the code renderers (`corpus.generator.v2.renderers`), plus
the WP1 name and simulator fixes the WP3 review made
(`01-spec/reviews/review-v2-wp3-2026-09-22.md`).

Each defect in the review has a regression test here; the test names are
listed in the review.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from collections import Counter
from datetime import date
from pathlib import Path

import pytest
import yaml

from corpus.generator.manifest import LOCATOR_PATTERNS
from corpus.generator.v2 import names
from corpus.generator.v2.models import PersonSide, RenewalStatus, Split
from corpus.generator.v2.packs.schema import VerticalPack, load_pack
from corpus.generator.v2.planner import (
    BUNDLE_RULES,
    CRM_ACCOUNT_FIELDS,
    DocKind,
    DocRole,
    TargetGroup,
    WorldDocPlan,
    plan_world,
    render_crm_row_text,
    scaled_targets,
)
from corpus.generator.v2.renderers import (
    STRUCTURAL_VOCABULARY,
    RenderedDoc,
    SlotKind,
    materialize_crm_accounts,
    materialize_rendered_doc,
    render,
    render_chat_directories,
)
from corpus.generator.v2.simulator import simulate_world
from portolan.connectors.file_connector import FileConnector
from portolan.connectors.file_connector import (
    _render_crm_row_text as _file_connector_render_crm_row_text,
)
from portolan.extract.schema import FactType, SourceType
from portolan.ingest.prepare import prepare_text

REPO_ROOT = Path(__file__).resolve().parents[1]
MEDICAL_DEVICES_PACK_PATH = (
    REPO_ROOT / "corpus" / "generator" / "v2" / "packs" / "medical_devices.yaml"
)

_SLOT_RE = re.compile(r"\[SLOT [^\]]*\]")
_QUOTE_RE = re.compile(r"\[QUOTE [^\]]*\]")


@pytest.fixture(scope="module")
def medical_devices_pack() -> VerticalPack:
    return load_pack(MEDICAL_DEVICES_PACK_PATH)


@pytest.fixture(scope="module")
def small_world(medical_devices_pack: VerticalPack):
    return simulate_world(medical_devices_pack, seed=42, months=8, split=Split.TRAIN)


@pytest.fixture(scope="module")
def small_plan(small_world, medical_devices_pack: VerticalPack) -> WorldDocPlan:
    return plan_world(small_world, medical_devices_pack)


@pytest.fixture(scope="module")
def rendered(small_plan: WorldDocPlan, small_world, medical_devices_pack) -> list[RenderedDoc]:
    return [render(d, small_world, small_plan, medical_devices_pack) for d in small_plan.doc_plans]


def _fill_placeholders(docs: list[RenderedDoc]) -> tuple[dict[str, str], dict[str, str]]:
    """Stand-in for WP4: every slot becomes one unique non-English token,
    every quote placeholder the quoted message's filled body with `> `
    before each line -- enough to run the real connector and
    `prepare_text` over the skeletons."""
    files: dict[str, str] = {}
    token_by_slot: dict[str, str] = {}
    for doc in docs:
        for path, text in doc.files.items():
            for i, slot in enumerate(doc.slots):
                token = f"zq{abs(hash(slot.slot_id)) % 10**8:08d}x{i}"
                token_by_slot[slot.slot_id] = token
                text = text.replace(slot.placeholder_text, token)
            files[path] = text
    for doc in docs:
        for op in doc.quote_ops:
            quoted = files[op.quoted_file].split("\n\n", 1)[1]
            quoted = quoted.split("\n\nOn ", 1)[0]
            block = "\n".join(f"> {line}" for line in quoted.splitlines())
            files[op.file] = files[op.file].replace(op.placeholder_text, block)
    return files, token_by_slot


# --------------------------------------------------------------------------
# Every scripted fact lands in exactly one plan.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("seed", "months"), [(1, 8), (7, 6), (42, 8), (3, 12)])
def test_every_scripted_fact_lands_in_exactly_one_plan(
    medical_devices_pack: VerticalPack, seed: int, months: int
) -> None:
    world = simulate_world(medical_devices_pack, seed=seed, months=months, split=Split.TRAIN)
    for plan in (
        plan_world(world, medical_devices_pack, type_cap=None),
        plan_world(world, medical_devices_pack),
    ):
        placements: Counter[str] = Counter(fid for doc in plan.doc_plans for fid in doc.fact_ids)
        capped = set(plan.type_cap.capped_fact_ids) if plan.type_cap else set()
        assert set(placements) == {f.id for f in world.facts} - capped
        assert not (set(placements) & capped)
        assert not {fid: n for fid, n in placements.items() if n != 1}
        for doc in plan.doc_plans:
            if doc.fact_ids:
                assert doc.role == DocRole.EVENT and not doc.is_negative


@pytest.mark.parametrize("seed", [1, 3, 42])
def test_the_type_cap_leaves_facts_out_at_plan_time(
    medical_devices_pack: VerticalPack, seed: int
) -> None:
    """WP7 dropped 55 of 222 written documents for the 12% per-type cap;
    the planner now leaves the excess facts unscripted before any writer is
    paid, and never an arc event's fact."""
    world = simulate_world(medical_devices_pack, seed=seed, split=Split.TRAIN)
    uncapped = plan_world(world, medical_devices_pack, type_cap=None)
    plan = plan_world(world, medical_devices_pack)
    assert plan == plan_world(world, medical_devices_pack)
    cap = plan.type_cap
    assert cap is not None and cap.applied and not cap.unmet_types
    facts = {f.id: f for f in world.facts}
    planted = Counter(facts[fid].fact_type.value for doc in plan.doc_plans for fid in doc.fact_ids)
    assert dict(planted) == cap.facts_after
    total = sum(planted.values())
    assert max(planted.values()) <= 0.12 * total
    before = Counter(
        facts[fid].fact_type.value for doc in uncapped.doc_plans for fid in doc.fact_ids
    )
    assert max(before.values()) > 0.12 * sum(before.values())  # the cap had work to do
    assert len(cap.capped_fact_ids) == sum(before.values()) - total > 0
    arc_events = {e for arc in world.arcs for e in arc.event_ids}
    assert not [fid for fid in cap.capped_fact_ids if facts[fid].event_id in arc_events]
    # The same documents are planned; only their scripts shrink.
    assert [d.id for d in plan.doc_plans] == [d.id for d in uncapped.doc_plans]
    by_id = {d.id: d for d in plan.doc_plans}
    for doc in uncapped.doc_plans:
        kept = by_id[doc.id].fact_ids
        assert kept == [f for f in doc.fact_ids if f not in set(cap.capped_fact_ids)]
    for doc_id in cap.documents_emptied:
        assert by_id[doc_id].role == DocRole.MENTION and by_id[doc_id].is_negative


def test_the_type_cap_skips_test_worlds(medical_devices_pack: VerticalPack) -> None:
    world = simulate_world(medical_devices_pack, seed=201, split=Split.TEST)
    plan = plan_world(world, medical_devices_pack)
    assert plan.type_cap is not None and not plan.type_cap.applied
    assert "gold" in (plan.type_cap.reason or "")
    placed = {fid for doc in plan.doc_plans for fid in doc.fact_ids}
    assert placed == {f.id for f in world.facts}


def test_crm_field_facts_are_not_world_facts(small_plan: WorldDocPlan, small_world) -> None:
    world_fact_ids = {f.id for f in small_world.facts}
    crm_fact_ids = {ff.id for record in small_plan.crm_accounts for ff in record.field_facts}
    assert crm_fact_ids and not (world_fact_ids & crm_fact_ids)
    doc_fact_ids = {fid for doc in small_plan.doc_plans for fid in doc.fact_ids}
    assert not (crm_fact_ids & doc_fact_ids)


# --------------------------------------------------------------------------
# Links.
# --------------------------------------------------------------------------

_LINK_FIELDS = (
    "linked_calls",
    "linked_email_threads",
    "linked_tickets",
    "linked_chat_threads",
    "linked_docs",
)


def test_crm_account_links_are_symmetric(small_plan: WorldDocPlan) -> None:
    doc_by_id = small_plan.doc_by_id()
    checked = 0
    for record in small_plan.crm_accounts:
        for field in _LINK_FIELDS:
            for doc_id in getattr(record, field):
                checked += 1
                assert doc_by_id[doc_id].links.get("crm_account") == [record.id]
    for doc in small_plan.doc_plans:
        record_ids = doc.links.get("crm_account")
        if not record_ids:
            continue
        record = next(r for r in small_plan.crm_accounts if r.id == record_ids[0])
        assert doc.id in {d for f in _LINK_FIELDS for d in getattr(record, f)}
        assert record.account_id == doc.account_id
    assert checked


def test_missing_link_noise_never_leaves_a_dangling_half(small_plan: WorldDocPlan) -> None:
    doc_by_id = small_plan.doc_by_id()
    doc_side = {op for d in small_plan.doc_plans for op in d.noise_ops if op.startswith("missing_")}
    record_side = {
        op for r in small_plan.crm_accounts for op in r.noise_ops if op.startswith("missing_")
    }
    assert doc_side and doc_side == record_side
    for op in doc_side:
        _, doc_id, record_id = op.split(":")
        assert doc_by_id[doc_id].links.get("crm_account") != [record_id]


def test_same_event_links_are_symmetric_and_resolve(small_plan: WorldDocPlan) -> None:
    doc_by_id = small_plan.doc_by_id()
    checked = 0
    for doc in small_plan.doc_plans:
        for other_id in doc.links.get("same_event", []):
            checked += 1
            other = doc_by_id[other_id]
            assert doc.id in other.links.get("same_event", [])
            assert set(doc.event_ids) & set(other.event_ids)
            assert other.role in (DocRole.EVENT, DocRole.MENTION)
    assert checked


def test_every_link_target_exists_and_negatives_are_linked_only_by_design(
    small_plan: WorldDocPlan, rendered: list[RenderedDoc]
) -> None:
    """Link integrity: every id a plan or a rendered document points at is
    a real document or CRM record; off-topic and near-duplicate documents
    are never CRM-linked; `shadows` points at a fact-carrying document."""
    doc_by_id = small_plan.doc_by_id()
    record_ids = {r.id for r in small_plan.crm_accounts}
    for doc in small_plan.doc_plans:
        for key, targets in doc.links.items():
            for target in targets:
                assert target in doc_by_id or target in record_ids, (doc.id, key, target)
        if doc.role in (DocRole.OFF_TOPIC, DocRole.NEAR_DUPLICATE):
            assert "crm_account" not in doc.links
        if doc.role == DocRole.NEAR_DUPLICATE:
            assert doc_by_id[doc.links["shadows"][0]].fact_ids
    for doc in rendered:
        for path, text in doc.files.items():
            if path.startswith("tickets/"):
                payload = json.loads(text)
                for target in [*payload["linked_issues"], *payload["related_calls"]]:
                    assert target in doc_by_id


# --------------------------------------------------------------------------
# Defect 5: volumes. Bundled documents land within 0.7x to 1.4x of the
# spec's per-world counts, scaled to the world's fact count.
# --------------------------------------------------------------------------

_LOWER_RATIO = 0.7
_UPPER_RATIO = 1.4


@pytest.mark.parametrize(("seed", "months"), [(1, 8), (7, 8), (42, 8), (5, 6), (9, 12)])
def test_per_kind_volumes_within_spec_ranges(
    medical_devices_pack: VerticalPack, seed: int, months: int
) -> None:
    world = simulate_world(medical_devices_pack, seed=seed, months=months, split=Split.TRAIN)
    plan = plan_world(world, medical_devices_pack)
    counts = plan.counts_by_target()
    targets = scaled_targets(world)
    for group, target in targets.items():
        assert _LOWER_RATIO * target <= counts[group] <= _UPPER_RATIO * target, (
            f"{group.value}: {counts[group]} documents against a target of {target}"
        )


def test_bundling_puts_several_facts_in_calls(small_plan: WorldDocPlan) -> None:
    """The spec's "scripted facts per doc" column is 5.0 for calls; one
    document per event gave about 2."""
    calls = [d for d in small_plan.doc_plans if d.kind == DocKind.CALL and d.fact_ids]
    assert sum(len(d.fact_ids) for d in calls) / len(calls) >= 4.0
    for doc in small_plan.doc_plans:
        if doc.fact_ids:
            group = TargetGroup.CHAT if doc.kind in (DocKind.SLACK, DocKind.TEAMS) else None
            rule = BUNDLE_RULES[group or TargetGroup(doc.kind.value)]
            assert len(doc.fact_ids) <= rule.max_facts or rule.max_span_days == 0


def test_routine_docs_respect_the_per_leaf_cap(
    small_plan: WorldDocPlan, medical_devices_pack: VerticalPack
) -> None:
    leaves = {leaf for node in medical_devices_pack.topic_tree for leaf in node.leaves()}
    routine = Counter(d.topic_leaf for d in small_plan.doc_plans if d.role == DocRole.ROUTINE)
    assert routine
    for leaf, count in routine.items():
        assert leaf in leaves and count <= 5


# --------------------------------------------------------------------------
# Defects 1 and 2: CRM records carry no generator ids, and field types
# follow the annotation table.
# --------------------------------------------------------------------------

_GENERATOR_ID_RE = re.compile(r"\b(?:F\d{5}|E\d{5}|D\d{5}|ARC\d{4}|CF-[\w-]+)\b")


def test_crm_records_carry_no_generator_ids(small_plan: WorldDocPlan) -> None:
    assert not any("fact_id" in column for column in CRM_ACCOUNT_FIELDS)
    for record in small_plan.crm_accounts:
        text = render_crm_row_text("account", record.row)
        assert not _GENERATOR_ID_RE.search(text), text
    for label in small_plan.labels:
        assert not _GENERATOR_ID_RE.search(label.text)


def test_no_rendered_file_carries_a_generator_id(rendered: list[RenderedDoc]) -> None:
    """Slot tags carry fact ids for WP4 and are replaced; nothing else in a
    file may."""
    for doc in rendered:
        for path, text in doc.files.items():
            outside = _QUOTE_RE.sub("", _SLOT_RE.sub("", text))
            assert not _GENERATOR_ID_RE.search(outside), (path, _GENERATOR_ID_RE.search(outside))


def test_crm_field_types_follow_the_annotation_table(small_plan: WorldDocPlan) -> None:
    stage_types = {
        RenewalStatus.AT_RISK.value: FactType.RISK,
        RenewalStatus.RENEWED.value: FactType.DECISION,
        RenewalStatus.CHURNED.value: FactType.DECISION,
    }
    seen_fields: set[str] = set()
    for record in small_plan.crm_accounts:
        for ff in record.field_facts:
            key = "+".join(ff.fields)
            seen_fields.add(key)
            if key == "arr":
                assert ff.fact_type == FactType.OTHER and int(record.row["arr"]) > 0
            elif key == "parent_crm_id":
                assert ff.fact_type == FactType.OTHER
            elif key == "renewal_date":
                assert ff.fact_type == FactType.DECISION
            elif key == "renewal_stage":
                assert ff.fact_type == stage_types[record.row["renewal_stage"]]
            elif key == "owner+owner_previous":
                assert ff.fact_type == FactType.CONTACT_CHANGE
            elif key == "competitors":
                assert ff.fact_type == FactType.COMPETITOR_MENTION
            elif key == "status":
                assert record.row["status"] in ("churned", "expanded")
                assert ff.fact_type == FactType.DECISION
            else:
                pytest.fail(f"unexpected labeled CRM field {key}")
        labeled = {f for ff in record.field_facts for f in ff.fields}
        for descriptor in ("crm_id", "name", "kind", "domain", "tier", "region"):
            assert descriptor not in labeled
        if record.row["renewal_stage"] == RenewalStatus.IN_PROGRESS.value:
            assert "renewal_stage" not in labeled
    assert {"arr", "renewal_date", "competitors"} <= seen_fields


def test_render_crm_row_text_matches_file_connector() -> None:
    row = {"crm_id": "A-1010", "name": "Test Health", "blank": "", "arr": "500000"}
    assert render_crm_row_text("account", row) == _file_connector_render_crm_row_text(
        "account", row
    )


def test_crm_labels_resolve_in_prepared_text(small_plan: WorldDocPlan, tmp_path: Path) -> None:
    """Spec 3.2 check 8: spans index `prepare_text`'s output, the text
    production sends, through a real `FileConnector` round trip."""
    materialize_crm_accounts(small_plan.crm_accounts, tmp_path)
    conn = FileConnector(tmp_path)
    refs = {ref.source_id: ref for ref in conn.list_sources() if ref.source_type == SourceType.CRM}
    checked = 0
    for record in small_plan.crm_accounts:
        prepared = prepare_text(conn.fetch(refs[record.id])).text
        for label in (lb for lb in small_plan.labels if lb.doc_id == record.id):
            assert prepared[label.char_start : label.char_end] == label.text
            values = [
                record.row[f]
                for ff in record.field_facts
                if ff.id == label.fact_id
                for f in ff.fields
            ]
            assert label.text.startswith(values[0]) and label.text.endswith(values[-1])
            checked += 1
    assert checked == len(small_plan.labels)


# --------------------------------------------------------------------------
# Defect 3: names. No digit in any name, prefix-level split disjointness,
# exam and prompt names blocked.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [1, 2, 3, 42, 99])
def test_no_generated_name_ends_in_a_digit(medical_devices_pack: VerticalPack, seed: int) -> None:
    for split in Split:
        world = simulate_world(medical_devices_pack, seed=seed, months=8, split=split)
        strings = [
            world.vendor.name,
            world.vendor.domain,
            *(a.name for a in world.accounts),
            *(a.domain for a in world.accounts),
            *(p.name for p in world.people),
            *(p.email for p in world.people),
            # Product *line* names only: a pack's tier codes ("T2", "T3")
            # are model numbers, as in the exam's own "Tidewater T2".
            *(p.name for p in world.vendor.products if not p.aliases),
            *(c.name for c in world.vendor.competitors),
        ]
        assert not [s for s in strings if re.search(r"\d", s)]
        account_names = [a.name for a in world.accounts]
        assert len(account_names) == len(set(account_names))
        person_names = [p.name for p in world.people]
        assert len(person_names) == len(set(person_names))


def test_vendor_and_competitor_prefixes_are_split_disjoint(
    medical_devices_pack: VerticalPack,
) -> None:
    """The prefix is the vendor's short name, alias and email domain; it
    used to cross splits (19 of 20 vendor prefixes)."""
    for fn in (names.vendor_prefixes_for_split, names.competitor_prefixes_for_split):
        pools = {split: set(fn(split)) for split in Split}
        assert all(pools.values())
        assert not (pools[Split.TRAIN] & pools[Split.VALIDATION])
        assert not (pools[Split.TRAIN] & pools[Split.TEST])
        assert not (pools[Split.VALIDATION] & pools[Split.TEST])

    def _surface(split: Split, seeds: range) -> set[str]:
        out: set[str] = set()
        for seed in seeds:
            w = simulate_world(medical_devices_pack, seed=seed, months=6, split=split)
            out |= {w.vendor.name.split()[0], w.vendor.domain}
            out |= {alias for c in w.vendor.competitors for alias in c.aliases}
            out |= {p.name for p in w.vendor.products}
        return out

    assert not (_surface(Split.TRAIN, range(1, 5)) & _surface(Split.TEST, range(1, 5)))
    train_vendors = {
        simulate_world(medical_devices_pack, seed=s, months=6, split=Split.TRAIN).vendor.domain
        for s in range(1, 13)
    }
    assert len(train_vendors) == 12  # every world has its own vendor


def test_name_pools_exclude_exam_and_prompt_names() -> None:
    truth = yaml.safe_load((REPO_ROOT / "corpus" / "truth-table.yaml").read_text())
    exam: set[str] = {truth["meta"]["supplier"]["name"].split()[0].lower()}
    for key in ("products", "competitors"):
        for item in truth["meta"][key]:
            exam |= {w.lower() for w in re.findall(r"[A-Za-z]{4,}", item["name"])}
    generic = {"health", "medical", "systems", "service", "contract", "link"}
    for account in truth["accounts"]:
        exam.add(account["name"].split()[0].lower())
    exam -= generic
    pools = [
        *names.FIRST_NAMES,
        *names.LAST_NAMES,
        *names.ACCOUNT_STEMS,
        *names.VENDOR_PREFIXES,
        *names.COMPETITOR_PREFIXES,
        *names.PRODUCT_LINE_NAMES,
    ]
    hits = {p for p in pools if re.sub(r"\d", "", p).lower() in exam}
    assert not hits, sorted(hits)
    prompt_names = {"fernbridge", "oakhollow", "cedarvale", "ashgrove"}
    assert not {p for p in pools if p.lower() in prompt_names}
    assert not [p for p in pools if re.search(r"\d", p)]


def test_every_fact_has_a_person_speaker_employed_on_its_date(small_world) -> None:
    """Top-up events used to land on system accounts (no contacts), so 16%
    of facts had speaker "customer"; others were spoken by someone who had
    left or not yet joined."""
    people = small_world.person_by_id()
    for fact in small_world.facts:
        speaker = people[fact.speaker]
        assert speaker.tenure.start <= fact.date
        assert speaker.tenure.end is None or speaker.tenure.end >= fact.date


# --------------------------------------------------------------------------
# Defect 4 and provenance: no code-authored words in a rendered file.
# --------------------------------------------------------------------------


def _world_vocabulary(world, plan: WorldDocPlan, pack: VerticalPack) -> set[str]:
    strings = [
        world.vendor.name,
        world.vendor.domain,
        *(p.name for p in world.vendor.products),
        *(c.name for c in world.vendor.competitors),
        *(a.name for a in world.accounts),
        *(a.domain for a in world.accounts),
        *(a.region for a in world.accounts),
        *(a.kind.value for a in world.accounts),
        *(a.status.value for a in world.accounts),
        *(s.value for s in RenewalStatus),
        *(f"{p.name} {p.title} {p.email} {p.chat_handle} {p.role}" for p in world.people),
        # initials_only signatures
        *("".join(part[0] for part in p.name.replace("-", " ").split()) for p in world.people),
        *(r.title for r in pack.roles),
        *(d.channel or "" for d in plan.doc_plans),
        *(d.subtype or "" for d in plan.doc_plans),
        *CRM_ACCOUNT_FIELDS,
    ]
    return {w.lower() for s in strings for w in re.findall(r"[A-Za-z]+", s)}


_ID_TOKENS_RE = re.compile(
    r"\b(?:[UC][0-9A-F]{8}|S-[a-z_]+-\d{5}(?:-m\d+)?|S-crm-A\d+|[AC]-\d{4})\b"
)


def test_rendered_files_hold_no_code_authored_words(
    small_world, small_plan: WorldDocPlan, rendered: list[RenderedDoc], medical_devices_pack
) -> None:
    allowed = STRUCTURAL_VOCABULARY | _world_vocabulary(
        small_world, small_plan, medical_devices_pack
    )
    files = {path: text for doc in rendered for path, text in doc.files.items()}
    files.update(render_chat_directories(small_world, small_plan))
    offenders: Counter[str] = Counter()
    for text in files.values():
        stripped = _ID_TOKENS_RE.sub(" ", _QUOTE_RE.sub(" ", _SLOT_RE.sub(" ", text)))
        for word in re.findall(r"[A-Za-z]{2,}", stripped):
            if word.lower() not in allowed:
                offenders[word] += 1
    assert not offenders, offenders.most_common(10)


def test_subject_lines_are_writer_slots(rendered: list[RenderedDoc]) -> None:
    subject_re = re.compile(r"^Subject: (?:RE: )?(\[SLOT \S+-subject subject\])$", re.MULTILINE)
    for doc in rendered:
        subject_slots = [s for s in doc.slots if s.kind == SlotKind.SUBJECT]
        for path, text in doc.files.items():
            if path.startswith("email/"):
                match = subject_re.search(text)
                assert match and match.group(1) == subject_slots[0].placeholder_text
            elif path.startswith(("tickets/", "crm/activity_notes/")):
                assert json.loads(text)["subject"] == subject_slots[0].placeholder_text
        if doc.kind in (DocKind.EMAIL, DocKind.TICKET, DocKind.CRM_NOTE, DocKind.DOC):
            assert len(subject_slots) == 1
            assert set(subject_slots[0].files) == set(doc.files)


def test_every_placeholder_is_unique_and_present(rendered: list[RenderedDoc]) -> None:
    seen: set[str] = set()
    for doc in rendered:
        for slot in doc.slots:
            assert slot.placeholder_text not in seen
            seen.add(slot.placeholder_text)
            for path in slot.files:
                assert doc.files[path].count(slot.placeholder_text) == 1
        tags = {m for text in doc.files.values() for m in _SLOT_RE.findall(text)}
        assert tags == {s.placeholder_text for s in doc.slots}


def test_negative_roles_render_as_ordinary_kinds(
    small_plan: WorldDocPlan, rendered: list[RenderedDoc]
) -> None:
    """A path or id naming the role would hand the student the label."""
    for doc in small_plan.doc_plans:
        assert doc.id.startswith(f"S-{doc.kind.value}-")
    for doc in rendered:
        for path in doc.files:
            assert not re.search(r"negative|off.topic|duplicate|routine|mention", path)
    roles = Counter(d.role for d in small_plan.doc_plans)
    assert roles[DocRole.HARD_NEGATIVE] and roles[DocRole.OFF_TOPIC]
    assert roles[DocRole.NEAR_DUPLICATE]


# --------------------------------------------------------------------------
# Speakers, participants and dates (review findings).
# --------------------------------------------------------------------------


def test_every_fact_speaker_is_a_participant_and_the_slot_speaker(
    small_world, small_plan: WorldDocPlan, rendered: list[RenderedDoc]
) -> None:
    facts = {f.id: f for f in small_world.facts}
    doc_by_id = small_plan.doc_by_id()
    for doc in rendered:
        plan = doc_by_id[doc.plan_id]
        assert plan.participants
        for slot in doc.slots:
            if slot.kind != SlotKind.FACT:
                continue
            fact = facts[slot.fact_id]
            assert slot.speaker_person_id == fact.speaker
            if plan.kind != DocKind.MEETING_NOTES:  # internal: speakers are reported
                assert fact.speaker in plan.participants


def test_customer_participants_belong_to_the_account_and_are_employed(
    small_world, small_plan: WorldDocPlan
) -> None:
    """Routine documents used to draw a random person from the whole world
    (12 to 18 per world were another account's contact)."""
    people = small_world.person_by_id()
    for plan in small_plan.doc_plans:
        for pid in plan.participants:
            person = people[pid]
            if person.side == PersonSide.CUSTOMER:
                assert person.account_id == plan.account_id, (plan.id, pid)
            if plan.role != DocRole.EVENT:
                assert person.tenure.start <= plan.date
                assert person.tenure.end is None or person.tenure.end >= plan.date
        if plan.role == DocRole.OFF_TOPIC:
            assert all(people[pid].side == PersonSide.VENDOR for pid in plan.participants)


def test_email_fact_message_is_from_the_fact_speaker(
    small_world, rendered: list[RenderedDoc]
) -> None:
    """WP4 check 4 compares the fact's speaker with the message's `From`;
    the first build sent every thread from the plan's first contact."""
    people = small_world.person_by_id()
    facts = {f.id: f for f in small_world.facts}
    checked = 0
    for doc in rendered:
        if doc.kind != DocKind.EMAIL:
            continue
        for slot in doc.slots:
            if slot.kind == SlotKind.FACT:
                text = doc.files[slot.files[0]]
                sender = re.search(r"^From: (.*)$", text, re.MULTILINE).group(1)
                speaker = people[facts[slot.fact_id].speaker]
                assert sender == f"{speaker.name} <{speaker.email}>"
                checked += 1
    assert checked


def test_units_are_dated_after_their_fact_and_inside_the_window(
    small_world, small_plan: WorldDocPlan, rendered: list[RenderedDoc]
) -> None:
    facts = {f.id: f for f in small_world.facts}
    people = small_world.person_by_id()
    lo, hi = small_world.window_start, small_world.window_end
    for plan in small_plan.doc_plans:
        assert lo <= plan.date <= hi
    for doc in rendered:
        for slot in doc.slots:
            assert slot.unit_date is not None and lo <= slot.unit_date <= hi
            if slot.kind == SlotKind.FACT:
                assert slot.unit_date >= facts[slot.fact_id].date
            if slot.speaker_person_id:
                person = people[slot.speaker_person_id]
                assert person.tenure.start <= slot.unit_date or slot.kind == SlotKind.COLOR
        if doc.kind == DocKind.EMAIL:
            stamps = [
                re.search(r"^Date: (\S+ \S+)", text, re.MULTILINE).group(1)
                for text in doc.files.values()
            ]
            assert stamps == sorted(stamps)
            assert all(date.fromisoformat(s[:10]) <= hi for s in stamps)
        if doc.kind == DocKind.TICKET:
            payload = json.loads(next(iter(doc.files.values())))
            created = [c["created"] for c in payload["comments"]]
            assert created == sorted(created) and created[-1] <= hi.isoformat()


def test_no_two_documents_share_a_file_path(rendered: list[RenderedDoc]) -> None:
    paths = Counter(path for doc in rendered for path in doc.files)
    assert not [p for p, n in paths.items() if n > 1]


def test_skeletons_have_the_spec_shape(rendered: list[RenderedDoc]) -> None:
    """Spec section 2: calls 1,000 to 4,000 words, tickets 3 to 9
    comments, email threads 2 to 7 messages."""
    for doc in rendered:
        if doc.kind == DocKind.CALL:
            words = sum(s.target_words or 0 for s in doc.slots)
            assert 1000 <= words <= 4400
        elif doc.kind == DocKind.TICKET:
            comments = json.loads(next(iter(doc.files.values())))["comments"]
            assert 3 <= len(comments) <= 9
        elif doc.kind == DocKind.EMAIL:
            assert 2 <= len(doc.files) <= 7


def test_slot_locators_match_manifest_patterns(rendered: list[RenderedDoc]) -> None:
    patterns = {
        DocKind.CALL: LOCATOR_PATTERNS["call"],
        DocKind.TICKET: LOCATOR_PATTERNS["ticket"],
        DocKind.SLACK: LOCATOR_PATTERNS["slack"],
        DocKind.TEAMS: LOCATOR_PATTERNS["teams"],
        DocKind.CRM_NOTE: LOCATOR_PATTERNS["crm"],
    }
    for doc in rendered:
        pattern = patterns.get(doc.kind)
        for slot in doc.slots:
            if slot.kind == SlotKind.SUBJECT:
                assert slot.locator is None
            elif pattern is not None:
                assert pattern.match(slot.locator), (doc.plan_id, slot.locator)


def test_filled_skeletons_resolve_to_their_locators_after_prepare_text(
    small_plan: WorldDocPlan, small_world, rendered: list[RenderedDoc], tmp_path: Path
) -> None:
    """Simulates WP4 (each slot -> one token, each quote -> the quoted
    body) and runs the real connector and `prepare_text`: every fact and
    decoy token must sit inside the locator span its slot names -- the
    paragraph numbering of an email after signature and quote stripping,
    the call timestamp, the ticket comment."""
    files, token_by_slot = _fill_placeholders(rendered)
    for path, text in files.items():
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_text(text, encoding="utf-8")
    materialize_crm_accounts(small_plan.crm_accounts, tmp_path)
    conn = FileConnector(tmp_path)
    prepared = {}
    for ref in conn.list_sources():
        raw = conn.fetch(ref)
        prepared[ref.source_id] = prepare_text(raw)
    checked = 0
    for doc in rendered:
        if doc.kind not in (DocKind.CALL, DocKind.TICKET, DocKind.EMAIL):
            continue
        for slot in doc.slots:
            if slot.kind not in (SlotKind.FACT, SlotKind.DECOY):
                continue
            source_id = Path(slot.files[0]).stem
            pt = prepared[source_id]
            token = token_by_slot[slot.slot_id]
            assert pt.text.count(token) == 1, (slot.slot_id, source_id)
            at = pt.text.index(token)
            span = next(s for s in pt.locators if s.start <= at < s.end)
            assert span.locator == slot.locator, (slot.slot_id, span.locator, slot.locator)
            checked += 1
    assert checked > 100
    # Quoted copies of earlier messages were stripped by prepare_text.
    for doc in rendered:
        for op in doc.quote_ops:
            assert "> " not in prepared[Path(op.file).stem].text


def test_rendered_call_ticket_email_and_crm_load_through_file_connector(
    small_plan: WorldDocPlan, rendered: list[RenderedDoc], tmp_path: Path
) -> None:
    for doc in rendered:
        if doc.kind in (DocKind.CALL, DocKind.TICKET, DocKind.EMAIL):
            materialize_rendered_doc(doc, tmp_path)
    materialize_crm_accounts(small_plan.crm_accounts, tmp_path)
    conn = FileConnector(tmp_path)
    refs = list(conn.list_sources())
    by_type = Counter(ref.source_type for ref in refs)
    plans = small_plan.counts_by_kind()
    assert by_type[SourceType.CALL] == plans[DocKind.CALL]
    assert by_type[SourceType.TICKET] == plans[DocKind.TICKET]
    assert by_type[SourceType.EMAIL] == sum(
        len(d.files) for d in rendered if d.kind == DocKind.EMAIL
    )
    assert by_type[SourceType.CRM] == len(small_plan.crm_accounts)
    for ref in refs:
        raw = conn.fetch(ref)
        assert raw.text.strip()
        if ref.source_type == SourceType.EMAIL:
            assert raw.participants and all(p.email for p in raw.participants)
        if ref.source_type in (SourceType.CALL, SourceType.TICKET):
            assert raw.participants


def test_slack_and_teams_skeletons_are_valid_envelopes(
    small_world, small_plan: WorldDocPlan, rendered: list[RenderedDoc]
) -> None:
    directory = render_chat_directories(small_world, small_plan)
    user_ids = {u["id"] for u in json.loads(directory["slack/users.json"])}
    kinds = set()
    for doc in rendered:
        if doc.kind not in (DocKind.SLACK, DocKind.TEAMS):
            continue
        kinds.add(doc.kind)
        messages = json.loads(next(iter(doc.files.values())))
        assert messages
        if doc.kind == DocKind.SLACK:
            assert all({"type", "user", "text", "ts", "thread_ts"} <= set(m) for m in messages)
            assert {m["user"] for m in messages} <= user_ids
            ts = {m["ts"] for m in messages}
            assert all(m["thread_ts"] in ts for m in messages if m["thread_ts"])
        else:
            keys = {"id", "from", "body", "createdDateTime", "replyToId"}
            assert all(keys <= set(m) for m in messages)
    assert kinds == {DocKind.SLACK, DocKind.TEAMS}


# --------------------------------------------------------------------------
# Determinism.
# --------------------------------------------------------------------------


def test_plan_world_is_deterministic_for_a_seed(medical_devices_pack: VerticalPack) -> None:
    world_a = simulate_world(medical_devices_pack, seed=99, months=8, split=Split.TRAIN)
    world_b = simulate_world(medical_devices_pack, seed=99, months=8, split=Split.TRAIN)
    plan_a = plan_world(world_a, medical_devices_pack)
    plan_b = plan_world(world_b, medical_devices_pack)
    assert plan_a.model_dump(mode="json") == plan_b.model_dump(mode="json")


def test_render_is_deterministic_for_a_plan(
    small_plan: WorldDocPlan, small_world, medical_devices_pack: VerticalPack
) -> None:
    for plan in small_plan.doc_plans[:40]:
        a = render(plan, small_world, small_plan, medical_devices_pack)
        b = render(plan, small_world, small_plan, medical_devices_pack)
        assert a.model_dump() == b.model_dump()


_DIGEST_SCRIPT = """
import hashlib, json
from pathlib import Path
from corpus.generator.v2.models import Split
from corpus.generator.v2.packs.schema import load_pack
from corpus.generator.v2.planner import plan_world
from corpus.generator.v2.renderers import render
from corpus.generator.v2.simulator import simulate_world
pack = load_pack(Path({pack!r}))
w = simulate_world(pack, seed=7, months=6, split=Split.VALIDATION)
p = plan_world(w, pack)
h = hashlib.sha256(json.dumps(p.model_dump(mode="json"), sort_keys=True).encode())
for d in p.doc_plans:
    h.update(json.dumps(render(d, w, p, pack).model_dump(mode="json"), sort_keys=True).encode())
print(h.hexdigest())
"""


def test_world_plan_and_render_are_identical_across_hash_seeds() -> None:
    code = _DIGEST_SCRIPT.format(pack=str(MEDICAL_DEVICES_PACK_PATH))
    digests = set()
    for hash_seed in ("0", "1"):
        env = {**os.environ, "PYTHONHASHSEED": hash_seed}
        result = subprocess.run(
            [sys.executable, "-c", code],
            env=env,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        digests.add(result.stdout.strip())
    assert len(digests) == 1
