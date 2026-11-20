"""Tests for world generator v2 work package 2 (`01-spec/world-generator-
v2-2026-09-22.md` section 4, WP2 row): the seven new vertical packs plus
the eighth, held-out vertical (`lab_and_imaging_services.yaml` -- the
spec's own example: "an eighth held-out vertical (for example lab and
imaging services)").

`tests/test_world_generator_v2.py` (WP1) already covers the schema's
general validation machinery against `medical_devices.yaml`; this file
covers what's specific to WP2's task list: every new pack validates and
simulates cleanly across seeds/splits, no name-pool overlap between the
held-out pack and the rest, and the senior-living pack's hard exclusion
guard (the maintainer, 2026-09-22: no resident fall detection/prevention,
resident monitoring, video monitoring, or any product in that category).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from corpus.generator.v2.models import Split
from corpus.generator.v2.packs.schema import (
    MIN_CLAIM_VARIANTS_PER_FACT_TYPE,
    VerticalPack,
    load_pack,
)
from corpus.generator.v2.simulator import FACT_TYPE_FLOOR, simulate_world
from portolan.extract.schema import FactType

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKS_DIR = REPO_ROOT / "corpus" / "generator" / "v2" / "packs"

# The seven new verticals (spec section 4, verbatim list) plus the eighth,
# held-out one.
WP2_TRAINED_PACK_FILES = [
    "senior_living_tech.yaml",
    "pharmacy_services.yaml",
    "ehr_addons.yaml",
    "healthcare_staffing.yaml",
    "medical_supplies.yaml",
    "telehealth.yaml",
]
HELD_OUT_PACK_FILE = "lab_and_imaging_services.yaml"
WP2_ALL_PACK_FILES = [*WP2_TRAINED_PACK_FILES, HELD_OUT_PACK_FILE]

# Every pack this repo ships, WP1's reference included -- used for the
# held-out name-pool disjointness check ("the others" means every other
# pack, not just WP2's own six).
ALL_PACK_FILES = ["medical_devices.yaml", *WP2_ALL_PACK_FILES]

SENIOR_LIVING_PACK_FILE = "senior_living_tech.yaml"
# the maintainer, 2026-09-22: the senior-living pack must not include resident
# fall detection, fall prevention, resident monitoring, video monitoring
# of residents, or any product in that category (a category deliberately
# out of scope). Word-boundary, case-insensitive; `monitor\w*` catches
# "monitoring"/"monitors"/"monitored" etc.
FORBIDDEN_SENIOR_LIVING_PATTERN = re.compile(
    r"\b(fall|monitor\w*|camera|video|sensor)\b", re.IGNORECASE
)


@pytest.fixture(scope="module")
def wp2_packs() -> dict[str, VerticalPack]:
    return {f: load_pack(PACKS_DIR / f) for f in WP2_ALL_PACK_FILES}


# --------------------------------------------------------------------------
# Every pack validates against the schema.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("pack_file", WP2_ALL_PACK_FILES)
def test_wp2_pack_loads_and_validates(pack_file: str) -> None:
    pack = load_pack(PACKS_DIR / pack_file)
    assert pack.id == pack_file.removesuffix(".yaml")
    assert len(pack.products) >= 1
    assert len(pack.roles) >= 3
    assert len(pack.persona_bank) >= 8
    assert len(pack.decoy_classes) >= 7


@pytest.mark.parametrize("pack_file", WP2_ALL_PACK_FILES)
def test_wp2_pack_meets_directive_minimum_per_fact_type(pack_file: str) -> None:
    pack = load_pack(PACKS_DIR / pack_file)
    for fact_type in FactType:
        assert len(pack.fact_directives[fact_type]) >= 12, (
            f"{pack_file}: {fact_type.value} has fewer than 12 directives"
        )


@pytest.mark.parametrize("pack_file", WP2_ALL_PACK_FILES)
def test_wp2_pack_includes_commitment_decision_confusable_decoy(pack_file: str) -> None:
    """Task list: decoy classes must cover "type-confusable pairs such as
    commitment versus decision" in addition to the schema's own minimum
    seven near-miss classes."""
    pack = load_pack(PACKS_DIR / pack_file)
    confusable = [d for d in pack.decoy_classes if "confusable" in d.id]
    assert confusable, f"{pack_file}: no commitment/decision confusable-pair decoy class"
    assert any(d.near_type in (FactType.COMMITMENT, FactType.DECISION) for d in confusable)


def test_only_lab_and_imaging_services_is_held_out(wp2_packs: dict[str, VerticalPack]) -> None:
    held_out_ids = {f for f, p in wp2_packs.items() if p.held_out}
    assert held_out_ids == {HELD_OUT_PACK_FILE}


def test_medical_devices_pack_defaults_held_out_false() -> None:
    """The `held_out` field WP2 added to the schema must not change
    WP1's existing reference pack's validated shape."""
    pack = load_pack(PACKS_DIR / "medical_devices.yaml")
    assert pack.held_out is False


# --------------------------------------------------------------------------
# Every pack simulates cleanly, three seeds per split, per-type floors met.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("pack_file", WP2_ALL_PACK_FILES)
@pytest.mark.parametrize("split", [Split.TRAIN, Split.VALIDATION, Split.TEST])
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_wp2_pack_simulates_cleanly_and_meets_fact_floor(
    pack_file: str, split: Split, seed: int
) -> None:
    pack = load_pack(PACKS_DIR / pack_file)
    world = simulate_world(pack, seed=seed, months=8, split=split)

    event_ids = {e.id for e in world.events}
    for fact in world.facts:
        assert fact.event_id in event_ids

    counts = world.facts_by_type()
    short = {ft.value: len(facts) for ft, facts in counts.items() if len(facts) < FACT_TYPE_FLOOR}
    assert not short, f"{pack_file} {split.value} seed={seed}: below the floor: {short}"


# --------------------------------------------------------------------------
# No name overlap between the held-out pack's name pools and the others.
# --------------------------------------------------------------------------


def _account_suffix_pool(pack: VerticalPack) -> set[str]:
    refs = pack.name_pool_refs
    return (
        set(refs.account_suffixes_system)
        | set(refs.account_suffixes_facility)
        | set(refs.account_suffixes_standalone)
    )


def test_held_out_pack_name_pools_disjoint_from_every_other_pack() -> None:
    packs = {f: load_pack(PACKS_DIR / f) for f in ALL_PACK_FILES}
    held_out_pool = _account_suffix_pool(packs[HELD_OUT_PACK_FILE])
    assert held_out_pool, "held-out pack has an empty account-suffix pool"

    for other_file, other_pack in packs.items():
        if other_file == HELD_OUT_PACK_FILE:
            continue
        overlap = held_out_pool & _account_suffix_pool(other_pack)
        assert not overlap, f"held-out pack shares name-pool entries with {other_file}: {overlap}"


def test_held_out_pack_product_kinds_disjoint_from_trained_verticals() -> None:
    """A held-out vertical whose product archetypes literally repeat a
    trained vertical's `kind` labels would not test generalization past
    the training verticals (spec section 4). Product `kind`s should be
    genuinely new vocabulary, not shared with any of the seven trained
    packs (WP1's `medical_devices.yaml` included)."""
    trained_files = ["medical_devices.yaml", *WP2_TRAINED_PACK_FILES]
    trained_kinds: set[str] = set()
    for f in trained_files:
        pack = load_pack(PACKS_DIR / f)
        trained_kinds |= {p.kind for p in pack.products}

    held_out_pack = load_pack(PACKS_DIR / HELD_OUT_PACK_FILE)
    held_out_kinds = {p.kind for p in held_out_pack.products}
    assert not (held_out_kinds & trained_kinds)


# --------------------------------------------------------------------------
# Senior-living exclusion guard (the maintainer, 2026-09-22).
# --------------------------------------------------------------------------


def test_senior_living_pack_excludes_monitoring_category() -> None:
    """Scans the senior-living pack's `products` and `event_templates`
    text for fall/monitor*/camera/video/sensor and fails if any appear.
    Reads the raw YAML mapping directly (not just the validated model) so
    the check also covers fields the schema doesn't otherwise assert on."""
    raw = yaml.safe_load((PACKS_DIR / SENIOR_LIVING_PACK_FILE).read_text(encoding="utf-8"))

    hits: list[str] = []
    for product in raw["products"]:
        haystack = " ".join(
            [
                product.get("kind", ""),
                product.get("name_template", ""),
                *product.get("tiers", []),
                *product.get("known_issue_pool", []),
            ]
        )
        if FORBIDDEN_SENIOR_LIVING_PATTERN.search(haystack):
            hits.append(f"product {product.get('kind')!r}: {haystack!r}")

    for event in raw["event_templates"]:
        for fact_type, templates in event["claim_templates"].items():
            for template in templates:
                if FORBIDDEN_SENIOR_LIVING_PATTERN.search(template):
                    hits.append(f"event {event['event_type']!r} ({fact_type}): {template!r}")

    assert not hits, f"senior-living pack contains excluded-category terms: {hits}"


def test_senior_living_pack_declares_the_exclusion() -> None:
    """The schema-level half of the guard: `excluded_product_kinds` names
    the excluded category, and (via the schema's own validator) none of
    those kinds actually appear among the pack's products."""
    pack = load_pack(PACKS_DIR / SENIOR_LIVING_PACK_FILE)
    excluded_text = " ".join(pack.excluded_product_kinds).lower()
    assert "fall" in excluded_text
    assert "monitor" in excluded_text
    present_kinds = {p.kind for p in pack.products}
    assert not (present_kinds & set(pack.excluded_product_kinds))


# --------------------------------------------------------------------------
# 15-or-more claim-template variants per fact type (coordinator review of
# 7f779a3, 2026-09-22: spec section 4, "Each fact type has 15 or more event
# variants per vertical" -- every pack was at 1-5 before this fix).
# --------------------------------------------------------------------------


def _unique_variant_count(pack: VerticalPack, fact_type: FactType) -> int:
    seen: set[str] = set()
    for template in pack.event_templates:
        for claim in template.claim_templates.get(fact_type, []):
            seen.add(re.sub(r"\s+", " ", claim.strip().lower()))
    return len(seen)


@pytest.mark.parametrize("pack_file", ALL_PACK_FILES)
def test_pack_meets_claim_variant_floor_per_fact_type(pack_file: str) -> None:
    pack = load_pack(PACKS_DIR / pack_file)
    for fact_type in FactType:
        count = _unique_variant_count(pack, fact_type)
        assert count >= MIN_CLAIM_VARIANTS_PER_FACT_TYPE, (
            f"{pack_file}: {fact_type.value} has only {count} unique claim "
            f"variants (need {MIN_CLAIM_VARIANTS_PER_FACT_TYPE})"
        )


def test_schema_rejects_a_pack_below_the_claim_variant_floor(tmp_path: Path) -> None:
    """The validator itself, not just the shipped packs: take a real,
    otherwise-valid pack and collapse one fact type's claim templates down
    to a single variant -- must fail to load."""
    raw = yaml.safe_load((PACKS_DIR / "medical_devices.yaml").read_text())
    trimmed_one = False
    for template in raw["event_templates"]:
        if "complaint" in template["claim_templates"]:
            template["claim_templates"]["complaint"] = template["claim_templates"]["complaint"][:1]
            trimmed_one = True
        elif trimmed_one:
            template["claim_templates"].pop("complaint", None)
    assert trimmed_one, "fixture assumption broken: no event_template carries 'complaint'"

    bad_path = tmp_path / "bad_pack.yaml"
    bad_path.write_text(yaml.safe_dump(raw, sort_keys=False))
    with pytest.raises(ValidationError, match="below the 15-variant floor"):
        load_pack(bad_path)


# --------------------------------------------------------------------------
# Claim-template / decoy-example provenance (coordinator review of
# 7f779a3: this text becomes training-data-facing, so it must be recorded
# as coming from an open-weight model, never Claude).
# --------------------------------------------------------------------------


@pytest.mark.parametrize("pack_file", ALL_PACK_FILES)
def test_pack_records_claim_templates_source(pack_file: str) -> None:
    pack = load_pack(PACKS_DIR / pack_file)
    assert pack.claim_templates_source.model
    assert "claude" not in pack.claim_templates_source.model.lower()
    assert "anthropic" not in pack.claim_templates_source.model.lower()
    # ISO 8601 date, parses cleanly.
    from datetime import date as _date

    _date.fromisoformat(pack.claim_templates_source.generated)


# Every pack's negation example before 2026-09-23 (model-written, as they
# stood in the packs), plus the production prompt's own negated competitor
# case: each is a `competitor_mention` fact (decisions log, 2026-09-23).
_COMPETITOR_NEGATIONS = [
    "We looked at MedFlow's platform last spring but stuck with our current vendor since "
    "the integration was already dialed in.",
    "We evaluated CareShift’s nurse placement tool last quarter, but ultimately passed "
    "on it since their EHR integration didn’t align with our current workflows.",
    "We considered HealthScope Solutions' PACS offering last year, but we decided to stick "
    "with our current platform.",
    "We looked into AuroraMed’s infusion pumps last quarter, but their closed ecosystem "
    "didn’t play nice with our EHR, so we stuck with our current vendor’s modules.",
    "We considered OptiPharm’s platform last quarter but decided against it due to "
    "integration complexities.",
    "We looked at HealthTrack’s inventory system but chose to stick with our current "
    "vendor for better integration.",
    "We considered switching to CareLink’s solution for behavioral health.",
    "We evaluated GreenVue CareSync a couple of months ago.",
    "We're not looking at other suppliers",
    "we aren't using the other scheduling platform anymore",
]


@pytest.mark.parametrize("pack_file", ALL_PACK_FILES)
def test_negation_decoys_carry_no_competitor_statement(pack_file: str) -> None:
    from corpus.generator.v2.verify import forbidden_pattern_hits

    pack = load_pack(PACKS_DIR / pack_file)
    [negation] = [c for c in pack.decoy_classes if c.id.split("_")[0] == "negation"]
    assert negation.near_type != FactType.COMPETITOR_MENTION
    assert negation.forbidden_patterns == ["competitor_evaluation"]
    # Its text was rewritten by the open-weight model, never by hand.
    source = negation.text_source
    assert source is not None and source.raw_response_sha256
    assert source.model == pack.claim_templates_source.model
    assert "claude" not in source.model.lower()
    for text in (negation.description, negation.example):
        assert not forbidden_pattern_hits(text, negation.forbidden_patterns), text
    for text in _COMPETITOR_NEGATIONS:
        assert forbidden_pattern_hits(text, negation.forbidden_patterns), text


def test_pack_rejects_missing_claim_templates_source(tmp_path: Path) -> None:
    raw = yaml.safe_load((PACKS_DIR / "medical_devices.yaml").read_text())
    raw.pop("claim_templates_source", None)
    bad_path = tmp_path / "bad_pack.yaml"
    bad_path.write_text(yaml.safe_dump(raw, sort_keys=False))
    with pytest.raises(ValidationError, match="claim_templates_source"):
        load_pack(bad_path)
