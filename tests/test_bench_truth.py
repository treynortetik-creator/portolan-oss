"""Tests for `portolan.bench.truth`: load the truth table + manifest and
cross-check them against each other."""

from __future__ import annotations

from pathlib import Path

from portolan.bench.truth import Manifest, Truth, TruthTable

REPO_ROOT = Path(__file__).resolve().parents[1]
TRUTH_TABLE_PATH = REPO_ROOT / "corpus" / "truth-table.yaml"
MANIFEST_PATH = REPO_ROOT / "corpus" / "out" / "manifest.json"


def _load_truth() -> Truth:
    return Truth.load(TRUTH_TABLE_PATH, MANIFEST_PATH)


def test_truth_table_loads_and_has_38_planted_facts_and_4_distractors() -> None:
    # v1.1 (2026-09-13) added 35 facts (F37..F71, F40b) over v0's 38 -> 73.
    table = TruthTable.load(TRUTH_TABLE_PATH)
    planted = [f for f in table.facts if not f.is_distractor]
    distractors = [f for f in table.facts if f.is_distractor]
    assert len(planted) == 73
    assert len(distractors) == 4
    assert {f.id for f in distractors} == {"D01", "D02", "D03", "D04"}


def test_manifest_loads() -> None:
    manifest = Manifest.load(MANIFEST_PATH)
    assert manifest.sources
    assert all(src.id for src in manifest.sources)


def test_every_placement_source_exists() -> None:
    """Every fact's `sources[].id` in the truth table must correspond to a
    real manifest source -- a typo here would silently zero out that
    fact's recall opportunity."""
    truth = _load_truth()
    assert truth.missing_source_refs() == []


def test_truth_and_manifest_cross_check_counts() -> None:
    truth = _load_truth()
    assert len(truth.planted_fact_ids()) == 73
    assert set(truth.distractor_fact_ids()) == {"D01", "D02", "D03", "D04"}

    # Every non-distractor fact_id in the truth table shows up in at least
    # one manifest placement, and vice versa.
    truth_fact_ids = set(truth.planted_fact_ids())
    manifest_fact_ids = {pf.fact_id for pf in truth.all_planted()}
    assert truth_fact_ids == manifest_fact_ids

    # The 4 distractor sources are exactly D01..D04's sources, per
    # manifest.json's own is_distractor/distractor_id flags.
    distractor_sources = truth.distractor_source_ids()
    assert len(distractor_sources) == 4
    for src_id in distractor_sources:
        src = truth.manifest.source(src_id)
        assert src.is_distractor
        assert src.distractor_id in {"D01", "D02", "D03", "D04"}


def test_cross_source_facts_produce_one_placement_per_source() -> None:
    """F02, F11, F11b, F19, F34 are each planted in two different sources
    (`cross_source_dedup` tests) -- `planted()` must return them
    independently per source, not collapsed."""
    truth = _load_truth()
    for fact_id in ("F02", "F11", "F11b", "F19", "F34"):
        fact_def = truth.fact(fact_id)
        source_ids = {ref.id for ref in fact_def.sources}
        assert len(source_ids) == 2
        for source_id in source_ids:
            planted_here = [pf for pf in truth.planted(source_id) if pf.fact_id == fact_id]
            assert len(planted_here) == 1


def test_within_source_duplicate_placement_collapses_to_one_planted_fact() -> None:
    """F04 is planted twice inside the same call (`within_source_dedup`) --
    `planted()` should surface it as one PlantedFact with two acceptable
    locators, not two separate requirements."""
    truth = _load_truth()
    f04_entries = [pf for pf in truth.planted("S-call-011") if pf.fact_id == "F04"]
    assert len(f04_entries) == 1
    assert len(f04_entries[0].locators) == 2


def test_entity_name_and_alias_matching() -> None:
    truth = _load_truth()
    assert truth.entities_overlap(["ashgrove-medical"], ["Ashgrove Medical Center"])
    assert truth.entities_overlap(["ashgrove-medical"], ["AMC"])
    assert truth.entities_overlap(["tidewater-t2"], ["the T2"])
    assert not truth.entities_overlap(["ashgrove-medical"], ["Northlake General Hospital"])
    # No entities listed on the planted fact -> trivially satisfied.
    assert truth.entities_overlap([], ["anything"])


def test_identify_entity_is_unambiguous_reverse_lookup() -> None:
    truth = _load_truth()
    assert truth.identify_entity("AMC") == "ashgrove-medical"
    assert truth.identify_entity("Ashgrove Medical Center") == "ashgrove-medical"
    assert truth.identify_entity("Some Unknown Company") is None


def test_entities_overlap_accepts_registry_id_shaped_guesses() -> None:
    """Diagnosis (var/bench/diag/diagnosis.md): with no real registry
    supplied, a model asked to "resolve entities to registry ids" (fact
    rule 6) sometimes invents its own slug for an entity's name rather than
    using the plain mention string -- `northlake-health-system` or
    `account:northlake` for the account the truth table itself calls
    `northlake` (canonical name "Northlake Health System"). Dashes,
    underscores and colons must line up with the space-separated canonical
    name/alias, not just an exact string match."""
    truth = _load_truth()
    assert truth.entities_overlap(["northlake"], ["northlake-health-system"])
    assert truth.entities_overlap(["northlake"], ["account:northlake"])
    assert truth.entities_overlap(["larkspur-memorial"], ["larkspur_memorial_hospital"])
    # Still correctly rejects an unrelated entity spelled the same way.
    assert not truth.entities_overlap(["northlake"], ["ashgrove-medical-center"])


def test_entities_overlap_accepts_descriptive_mention_containing_the_name() -> None:
    """A produced mention doesn't have to be *exactly* the canonical name or
    alias -- a longer, descriptive mention that names the entity along with
    other words (a registry-context-style annotation, or the model
    combining a role with a name) should still count, as long as the name
    appears as whole words, not as a substring of some other word."""
    truth = _load_truth()
    assert truth.entities_overlap(
        ["derek-hollis"], ["Northlake's Director of Clinical Engineering, Derek Hollis"]
    )
    assert truth.entities_overlap(["northlake"], ["Northlake Health System (account)"])
    # Word-boundary guard: "AMC" must not match inside an unrelated word.
    assert not truth.entities_overlap(["ashgrove-medical"], ["AMCity Consulting Group"])


def test_entities_overlap_contact_names_match_interchangeably_with_ids() -> None:
    """The truth table's contact names (no separate `aliases` field, unlike
    accounts) must still match their id, and vice versa -- "registry ids,
    canonical names, aliases and the truth table's contact names
    interchangeably", per the fix instructions."""
    truth = _load_truth()
    assert truth.entities_overlap(["felix-arden"], ["Felix Arden"])
    assert truth.entities_overlap(["felix-arden"], ["felix-arden"])
    assert truth.entities_overlap(["felix-arden"], ["felix_arden"])
