"""Tests for world generator v2 work package 1 (`01-spec/world-generator-
v2-2026-09-22.md`): the data model invariant, the simulator's determinism
and per-type floors, split-disjoint name pools, pack validation, and the
`truth-table.yaml` compat export round-tripping through `portolan.bench.
truth`.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from corpus.generator.v2 import names
from corpus.generator.v2.models import (
    AccountKind,
    AccountTier,
    Employment,
    Event,
    EventType,
    Fact,
    Person,
    Persona,
    PersonSide,
    ReferenceMode,
    Split,
    Vendor,
    World,
)
from corpus.generator.v2.packs.schema import VerticalPack, load_pack
from corpus.generator.v2.simulator import FACT_TYPE_FLOOR, simulate_world
from portolan.bench.truth import TruthTable
from portolan.extract.schema import FactType

REPO_ROOT = Path(__file__).resolve().parents[1]
MEDICAL_DEVICES_PACK_PATH = (
    REPO_ROOT / "corpus" / "generator" / "v2" / "packs" / "medical_devices.yaml"
)


@pytest.fixture(scope="module")
def medical_devices_pack() -> VerticalPack:
    return load_pack(MEDICAL_DEVICES_PACK_PATH)


def _minimal_world_kwargs() -> dict:
    """Enough to build a one-account, one-event `World` by hand, for the
    invariant tests below -- no simulator involved."""
    vendor = Vendor(
        id="vendor-test",
        name="Test Vendor",
        vertical="medical_devices",
        domain="testvendor.com",
        message_id_domain="testvendor.com",
    )
    from corpus.generator.v2.models import Account, AccountStatus

    account = Account(
        id="acct-1",
        crm_id="A-1",
        name="Test Account",
        kind=AccountKind.STANDALONE,
        tier=AccountTier.MID_MARKET,
        arr=100_000.0,
        contract_start=date(2025, 1, 1),
        renewal_date=date(2027, 1, 1),
        region="Midwest",
        status=AccountStatus.ACTIVE,
    )
    person = Person(
        id="person-1",
        name="Jamie Rivera",
        side=PersonSide.CUSTOMER,
        account_id="acct-1",
        role="Director of Nursing",
        title="Director of Nursing",
        email="jrivera@testaccount.org",
        chat_handle="@jrivera",
        tenure=Employment(start=date(2024, 1, 1), end=None),
        persona=Persona(speech_register="clinical", formality="neutral"),
    )
    event = Event(
        id="E00001",
        date=date(2026, 2, 1),
        type=EventType.INCIDENT,
        account_id="acct-1",
    )
    return {
        "id": "world-test",
        "seed": 1,
        "vertical": "medical_devices",
        "split": Split.TRAIN,
        "window_start": date(2026, 1, 1),
        "window_end": date(2026, 6, 1),
        "vendor": vendor,
        "accounts": [account],
        "people": [person],
        "events": [event],
        "arcs": [],
    }


# --------------------------------------------------------------------------
# Model invariant: every fact must trace to a real event id.
# --------------------------------------------------------------------------


def test_world_rejects_fact_with_unknown_event_id() -> None:
    kwargs = _minimal_world_kwargs()
    orphan_fact = Fact(
        id="F00001",
        fact_type=FactType.COMPLAINT,
        claim="Test Account reported a device malfunction as of 2026-02-01.",
        speaker="person-1",
        date=date(2026, 2, 1),
        account_id="acct-1",
        event_id="E-does-not-exist",
    )
    with pytest.raises(ValidationError, match="invariant violated"):
        World(**kwargs, facts=[orphan_fact])


def test_world_accepts_fact_with_real_event_id() -> None:
    kwargs = _minimal_world_kwargs()
    fact = Fact(
        id="F00001",
        fact_type=FactType.COMPLAINT,
        claim="Test Account reported a device malfunction as of 2026-02-01.",
        speaker="person-1",
        date=date(2026, 2, 1),
        account_id="acct-1",
        event_id="E00001",
    )
    world = World(**kwargs, facts=[fact])
    assert world.facts[0].event_id == "E00001"


def test_fact_rejects_blank_event_id() -> None:
    with pytest.raises(ValidationError):
        Fact(
            id="F00001",
            fact_type=FactType.COMPLAINT,
            claim="x",
            speaker="person-1",
            date=date(2026, 2, 1),
            account_id="acct-1",
            event_id="",
        )


def test_person_side_account_id_invariant() -> None:
    """Customer-side people need an account; vendor-side people must not
    carry a customer's account id."""
    with pytest.raises(ValidationError):
        Person(
            id="p1",
            name="A B",
            side=PersonSide.CUSTOMER,
            account_id=None,
            role="x",
            title="x",
            email="a@b.com",
            chat_handle="@ab",
            tenure=Employment(start=date(2025, 1, 1)),
            persona=Persona(speech_register="x", formality="x"),
        )
    with pytest.raises(ValidationError):
        Person(
            id="p2",
            name="A B",
            side=PersonSide.VENDOR,
            account_id="acct-1",
            role="x",
            title="x",
            email="a@b.com",
            chat_handle="@ab",
            tenure=Employment(start=date(2025, 1, 1)),
            persona=Persona(speech_register="x", formality="x"),
        )


# --------------------------------------------------------------------------
# Pack validation
# --------------------------------------------------------------------------


def test_medical_devices_pack_loads_and_validates(medical_devices_pack: VerticalPack) -> None:
    assert medical_devices_pack.id == "medical_devices"
    assert {t.event_type for t in medical_devices_pack.event_templates} == set(EventType)
    assert {a.arc_type for a in medical_devices_pack.arcs} == set(
        __import__("corpus.generator.v2.models", fromlist=["ArcType"]).ArcType
    )
    for fact_type in FactType:
        assert len(medical_devices_pack.fact_directives[fact_type]) >= 12


def test_pack_rejects_missing_fact_type_directives(tmp_path: Path) -> None:
    raw = yaml.safe_load(MEDICAL_DEVICES_PACK_PATH.read_text())
    del raw["fact_directives"]["complaint"]
    bad_path = tmp_path / "bad_pack.yaml"
    bad_path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValidationError, match="missing type"):
        load_pack(bad_path)


def test_pack_rejects_event_template_not_covering_every_event_type(tmp_path: Path) -> None:
    raw = yaml.safe_load(MEDICAL_DEVICES_PACK_PATH.read_text())
    raw["event_templates"] = [raw["event_templates"][0]]  # drop coverage of the other 13 types
    bad_path = tmp_path / "bad_pack.yaml"
    bad_path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValidationError, match="event_templates missing"):
        load_pack(bad_path)


def test_pack_rejects_unreachable_fact_type(tmp_path: Path) -> None:
    """If no event template names a given fact type, that type's 25-per-
    world floor can never be met -- the pack should fail to load, not
    fail silently at simulation time."""
    raw = yaml.safe_load(MEDICAL_DEVICES_PACK_PATH.read_text())
    for template in raw["event_templates"]:
        template["claim_templates"].pop("other", None)
    bad_path = tmp_path / "bad_pack.yaml"
    bad_path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValidationError, match="no event_template emits fact type"):
        load_pack(bad_path)


# --------------------------------------------------------------------------
# Simulator: determinism, per-type floors, no orphan facts
# --------------------------------------------------------------------------


def test_simulate_world_is_deterministic_for_a_seed(medical_devices_pack: VerticalPack) -> None:
    w1 = simulate_world(medical_devices_pack, seed=42, months=9, split=Split.TRAIN)
    w2 = simulate_world(medical_devices_pack, seed=42, months=9, split=Split.TRAIN)
    assert w1.model_dump(mode="json") == w2.model_dump(mode="json")


def test_simulate_world_different_seeds_differ(medical_devices_pack: VerticalPack) -> None:
    w1 = simulate_world(medical_devices_pack, seed=1, months=8, split=Split.TRAIN)
    w2 = simulate_world(medical_devices_pack, seed=2, months=8, split=Split.TRAIN)
    names_1 = [a.name for a in w1.accounts]
    names_2 = [a.name for a in w2.accounts]
    assert w1.vendor.name != w2.vendor.name or names_1 != names_2


@pytest.mark.parametrize("seed", [1, 5, 42])
def test_every_fact_type_meets_the_floor(medical_devices_pack: VerticalPack, seed: int) -> None:
    world = simulate_world(medical_devices_pack, seed=seed, months=8, split=Split.TRAIN)
    counts = world.facts_by_type()
    short = {ft.value: len(facts) for ft, facts in counts.items() if len(facts) < FACT_TYPE_FLOOR}
    assert not short, f"fact types below the {FACT_TYPE_FLOOR}-per-world floor: {short}"


@pytest.mark.parametrize("seed", [1, 5, 42])
def test_no_fact_without_a_matching_event(medical_devices_pack: VerticalPack, seed: int) -> None:
    world = simulate_world(medical_devices_pack, seed=seed, months=8, split=Split.TRAIN)
    event_ids = {e.id for e in world.events}
    for fact in world.facts:
        assert fact.event_id in event_ids


def test_simulate_world_rejects_out_of_range_months(medical_devices_pack: VerticalPack) -> None:
    with pytest.raises(ValueError, match="months must be 6-12"):
        simulate_world(medical_devices_pack, seed=1, months=3, split=Split.TRAIN)
    with pytest.raises(ValueError, match="months must be 6-12"):
        simulate_world(medical_devices_pack, seed=1, months=13, split=Split.TRAIN)


def test_arcs_reference_real_events(medical_devices_pack: VerticalPack) -> None:
    world = simulate_world(medical_devices_pack, seed=3, months=7, split=Split.VALIDATION)
    event_ids = {e.id for e in world.events}
    for arc in world.arcs:
        for event_id in arc.event_ids:
            assert event_id in event_ids


def test_snapshots_cover_the_window_weekly(medical_devices_pack: VerticalPack) -> None:
    world = simulate_world(medical_devices_pack, seed=9, months=6, split=Split.TEST)
    assert world.snapshots[0].date == world.window_start
    assert world.snapshots[-1].date <= world.window_end
    for a, b in zip(world.snapshots, world.snapshots[1:], strict=False):
        assert (b.date - a.date).days == 7


# --------------------------------------------------------------------------
# Split-disjoint name pools
# --------------------------------------------------------------------------


def test_name_pool_sizes_meet_spec_minimums() -> None:
    assert len(names.FIRST_NAMES) >= 400
    assert len(names.LAST_NAMES) >= 600


def test_name_pools_disjoint_across_splits() -> None:
    for pool, salt in [
        (names.FIRST_NAMES, "first_name"),
        (names.LAST_NAMES, "last_name"),
        (names.ACCOUNT_STEMS, "account_stem"),
        (names.VENDOR_NAMES, "vendor_name"),
        (names.COMPETITOR_NAMES, "competitor_name"),
    ]:
        train = set(names.pool_for_split(pool, Split.TRAIN, salt))
        validation = set(names.pool_for_split(pool, Split.VALIDATION, salt))
        test = set(names.pool_for_split(pool, Split.TEST, salt))
        assert train, salt
        assert validation, salt
        assert test, salt
        assert not (train & validation), f"{salt}: train/validation overlap"
        assert not (train & test), f"{salt}: train/test overlap"
        assert not (validation & test), f"{salt}: validation/test overlap"


@pytest.mark.parametrize("seed_a,seed_b", [(1, 2), (10, 99)])
def test_simulated_worlds_never_share_names_across_splits(
    medical_devices_pack: VerticalPack, seed_a: int, seed_b: int
) -> None:
    """Two seeds per split (per this work package's test list), checking
    the full assembled names a simulated world actually uses -- not just
    the raw pools."""
    seeds = (seed_a, seed_b)
    train_worlds = [
        simulate_world(medical_devices_pack, seed=s, months=8, split=Split.TRAIN) for s in seeds
    ]
    test_worlds = [
        simulate_world(medical_devices_pack, seed=s, months=8, split=Split.TEST) for s in seeds
    ]

    def _names_in(world: World) -> set[str]:
        out = {world.vendor.name}
        out.update(a.name for a in world.accounts)
        out.update(p.name for p in world.people)
        return out

    train_names: set[str] = set()
    for w in train_worlds:
        train_names |= _names_in(w)
    test_names: set[str] = set()
    for w in test_worlds:
        test_names |= _names_in(w)

    assert not (train_names & test_names)


def _world_name_strings(world: World) -> list[str]:
    """Every generated name a document can carry: vendor, domain, products,
    competitors, accounts and their aliases and domains, people with their
    emails and chat handles."""
    vendor = world.vendor
    out = [vendor.name, vendor.domain]
    out += [x for p in vendor.products for x in (p.name, *p.aliases)]
    out += [x for c in vendor.competitors for x in (c.name, *c.aliases)]
    out += [x for a in world.accounts for x in (a.name, a.domain, *a.aliases)]
    out += [x for p in world.people for x in (p.name, p.email, p.chat_handle)]
    return out


def _blocklist_of(*phrases: str) -> dict[str, dict[int, set[int]]]:
    from portolan.training import contamination as C

    out: dict[str, dict[int, set[int]]] = {}
    for phrase in phrases:
        words = C.normalize_name_words(phrase)
        out.setdefault("erb_test", {}).setdefault(len(words), set()).add(
            C.word_sequence_hash(words)
        )
    return out


def test_name_guard_redraws_every_name_the_contamination_gate_would_block(
    medical_devices_pack: VerticalPack,
) -> None:
    """WP7 saw a generated facility name ("Parkview") block records. With a
    guard, a world redraws each blocked vendor, product line, competitor,
    account stem or qualifier, and person, and the gate finds none of them."""
    import numpy as np

    from portolan.training import contamination as C

    plain = simulate_world(medical_devices_pack, seed=42, split=Split.TRAIN)
    facility = next(a for a in plain.accounts if a.kind == AccountKind.FACILITY and a.parent)
    qualifier = facility.name.split()[1]
    person = next(p for p in plain.people if p.side == PersonSide.CUSTOMER)
    blocked = [
        plain.vendor.name.split()[0],
        plain.vendor.products[-1].name,
        plain.vendor.competitors[0].aliases[0],
        plain.accounts[-1].aliases[0],
        qualifier,
        person.name,
        plain.people[0].chat_handle,
    ]
    blocklist = _blocklist_of(*blocked)
    guard = names.NameGuard(blocklist, max_name_words=4)
    for name in blocked:
        assert guard.blocked(name), name
    assert not guard.blocked("an unrelated phrase")

    guarded = simulate_world(medical_devices_pack, seed=42, split=Split.TRAIN, name_guard=guard)
    assert guarded == simulate_world(
        medical_devices_pack, seed=42, split=Split.TRAIN, name_guard=guard
    )
    assert guarded != plain
    strings = _world_name_strings(guarded)
    assert [s for s in strings if guard.blocked(s)] == []
    # The same answer the builder's gate gives, on the same words.
    empty = np.zeros((0, C.ContaminationParams().minhash_permutations), dtype=np.uint64)
    refs = C.ContaminationReferences(
        empty, [], np.array([], dtype=np.uint64), blocklist, C.ContaminationParams()
    )
    assert [s for s in strings if C.check_text(s, refs)] == []
    assert [s for s in _world_name_strings(plain) if C.check_text(s, refs)]
    # A guard that blocks nothing the world uses leaves the world as it was.
    idle = names.NameGuard(_blocklist_of("Zzyzx Quorvath"), max_name_words=4)
    assert simulate_world(medical_devices_pack, seed=42, split=Split.TRAIN, name_guard=idle) == (
        plain
    )


def test_name_guard_loads_the_builders_blocklist(tmp_path: Path) -> None:
    import numpy as np

    from portolan.training import contamination as C

    assert names.NameGuard.from_dir(tmp_path) is None
    params = C.ContaminationParams()
    C.save_references(
        tmp_path,
        np.zeros((0, params.minhash_permutations), dtype=np.uint64),
        [],
        np.array([], dtype=np.uint64),
        _blocklist_of("Parkview"),
        params,
        {},
        source_fingerprints={},
    )
    guard = names.NameGuard.from_dir(tmp_path)
    assert guard is not None and guard.fingerprint
    assert guard.blocked("Marrowbrook Parkview Hospital") and not guard.blocked("Park View")


# --------------------------------------------------------------------------
# Compat export: World.to_truth_table() round-trips through
# portolan.bench.truth.TruthTable.
# --------------------------------------------------------------------------


def test_to_truth_table_validates_directly(medical_devices_pack: VerticalPack) -> None:
    world = simulate_world(medical_devices_pack, seed=11, months=8, split=Split.TEST)
    table = TruthTable.model_validate(world.to_truth_table())
    assert len(table.accounts) == len(world.accounts)
    assert len(table.facts) == len(world.facts)
    fact_types = {f.type for f in table.facts}
    assert fact_types <= {ft.value for ft in FactType}


def test_to_truth_table_round_trips_through_yaml_file(
    medical_devices_pack: VerticalPack, tmp_path: Path
) -> None:
    world = simulate_world(medical_devices_pack, seed=12, months=8, split=Split.TEST)
    truth_table_dict = world.to_truth_table()
    out_path = tmp_path / "truth-table.yaml"
    out_path.write_text(yaml.safe_dump(truth_table_dict, sort_keys=False))

    loaded = TruthTable.load(out_path)
    customer_people = [p for p in world.people if p.side == PersonSide.CUSTOMER]
    assert len(loaded.facts) == len(world.facts)
    assert len(loaded.contacts) == len(customer_people)
    assert loaded.meta["seed"] == world.seed
    assert loaded.meta["vertical"] == world.vertical


def test_to_truth_table_every_fact_source_points_at_a_real_event(
    medical_devices_pack: VerticalPack,
) -> None:
    world = simulate_world(medical_devices_pack, seed=13, months=8, split=Split.VALIDATION)
    event_ids = {e.id for e in world.events}
    table = TruthTable.model_validate(world.to_truth_table())
    for fact in table.facts:
        assert len(fact.sources) == 1
        source_event_id = fact.sources[0].id.removeprefix("E-")
        assert source_event_id in event_ids


# --------------------------------------------------------------------------
# ReferenceMode / determinism sanity
# --------------------------------------------------------------------------


def test_reference_modes_are_a_valid_closed_set(medical_devices_pack: VerticalPack) -> None:
    world = simulate_world(medical_devices_pack, seed=21, months=8, split=Split.TRAIN)
    modes = {f.reference_mode for f in world.facts}
    assert modes <= set(ReferenceMode)
    assert ReferenceMode.EXPLICIT in modes  # the majority mode, per spec ~75%
