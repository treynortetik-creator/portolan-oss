"""Tests for `portolan.registry.owners` (`01-spec/actions-and-owners-
2026-09-19.md`, Phase 1).

Run with this module's own test database:
    createdb portolan_test_owners
    DATABASE_URL=postgresql://localhost/portolan_test_owners \
        uv run pytest tests/test_registry_owners.py -v
"""

from __future__ import annotations

import csv
from datetime import date

from portolan.bench.truth import Truth
from portolan.extract.schema import FactType
from portolan.llm.fake import FakeLLMClient
from portolan.merge.facts import merge_fact
from portolan.registry import Registry
from portolan.registry.owners import (
    OwnerCandidate,
    account_note_slugs,
    assign_owners_from_corpus,
    backfill_action_owners,
    confirm_owner_proposal,
    evidence_from_claims,
    evidence_from_corpus,
    is_vendor_side,
    is_vendor_side_crm_id,
    load_internal_people_index,
    propose_owners,
    rank_candidates,
    write_owner_crm_id_column,
)
from portolan.search.embed import FakeEmbedder
from tests.merge_helpers import (
    ASHGROVE,
    ASHGROVE_MEDICAL,
    SAM_OKAFOR,
    build_action,
    build_fact,
    loaded_registry_fixture,  # noqa: F401
    make_source_row,
)

# ---------------------------------------------------------------------
# rank_candidates -- pure, no DB.
# ---------------------------------------------------------------------


def test_rank_candidates_orders_by_source_count_then_recency_then_commitments():
    a = OwnerCandidate("a", "A", source_count=2, last_seen=date(2026, 8, 1), commitments=0)
    b = OwnerCandidate("b", "B", source_count=3, last_seen=date(2026, 1, 1), commitments=0)
    c = OwnerCandidate("c", "C", source_count=3, last_seen=date(2026, 8, 1), commitments=1)
    d = OwnerCandidate("d", "D", source_count=3, last_seen=date(2026, 8, 1), commitments=5)

    ranked = rank_candidates([a, b, c, d])

    assert [r.person_crm_id for r in ranked] == ["d", "c", "b", "a"]


def test_rank_candidates_never_seen_sorts_last_among_ties():
    seen = OwnerCandidate("seen", "Seen", source_count=1, last_seen=date(2026, 1, 1), commitments=0)
    unseen = OwnerCandidate("unseen", "Unseen", source_count=1, last_seen=None, commitments=0)

    ranked = rank_candidates([unseen, seen])

    assert [r.person_crm_id for r in ranked] == ["seen", "unseen"]


# ---------------------------------------------------------------------
# The synthetic-world (corpus) path.
# ---------------------------------------------------------------------


def _truth() -> Truth:
    return Truth.load()


def test_evidence_from_corpus_counts_distinct_sources_not_placements():
    t = _truth()
    evidence = evidence_from_corpus(t.table, t.manifest)
    ashgrove_medical = evidence.get("ashgrove-medical") or []
    assert ashgrove_medical, "expected direct evidence for ashgrove-medical in the real corpus"
    # every candidate's source_count must be <= manifest sources actually
    # naming this account -- a sanity bound, not an exact count (which
    # would just restate the fixture back at itself).
    named_sources = {s.id for s in t.manifest.sources if "ashgrove-medical" in s.accounts}
    for candidate in ashgrove_medical:
        assert candidate.source_count <= len(named_sources)
        assert candidate.source_count >= 1


def test_evidence_from_corpus_never_treats_a_committing_person_as_an_account():
    """F09's `entities` list is `[ashgrove, sam-okafor]` -- the person who
    committed is named alongside the account. `evidence_from_corpus` must
    never mistake "sam-okafor" for an account id it should be gathering
    evidence FOR."""
    t = _truth()
    evidence = evidence_from_corpus(t.table, t.manifest)
    assert "sam-okafor" not in evidence
    assert "priya-natarajan" not in evidence


def test_assign_owners_from_corpus_assigns_every_account_exactly_once():
    t = _truth()
    results = assign_owners_from_corpus(t.table, t.manifest)
    assert len(results) == len(t.table.accounts) == 20
    for account in t.table.accounts:
        assert account.id in results
        r = results[account.id]
        assert r.owner_crm_id  # never blank
        assert r.basis in (
            "own evidence",
            f"territory fallback ({account.region})",
        ) or r.basis.startswith("inherited from parent (")


def test_assign_owners_from_corpus_is_deterministic():
    t = _truth()
    first = assign_owners_from_corpus(t.table, t.manifest)
    second = assign_owners_from_corpus(t.table, t.manifest)
    assert {k: v.owner_crm_id for k, v in first.items()} == {
        k: v.owner_crm_id for k, v in second.items()
    }


def test_assign_owners_from_corpus_facility_inherits_parent_with_no_own_evidence():
    """Ashgrove Riverside Hospital (ashgrove-medical's sibling facility)
    has no direct manifest evidence of its own in the real corpus, so it
    must read the same owner as its parent, Ashgrove Health."""
    t = _truth()
    results = assign_owners_from_corpus(t.table, t.manifest)
    riverside = results["ashgrove-riverside"]
    parent = results["ashgrove"]
    assert riverside.owner_crm_id == parent.owner_crm_id
    assert riverside.basis.startswith("inherited from parent")


def test_assign_owners_from_corpus_system_under_capital_partner_is_independent():
    """Larkspur Regional Health (a `system`) has a `capital_partner`
    parent (Pinecrest) -- it must be resolved on its OWN evidence, never
    inherit Pinecrest's owner, per the spec's "the parent SYSTEM's owner"
    wording (only a facility inherits)."""
    t = _truth()
    results = assign_owners_from_corpus(t.table, t.manifest)
    larkspur = results["larkspur"]
    assert larkspur.basis == "own evidence"


def test_write_owner_crm_id_column_is_additive_and_idempotent(tmp_path):
    t = _truth()
    results = assign_owners_from_corpus(t.table, t.manifest)

    csv_path = tmp_path / "accounts.csv"
    csv_path.write_text(
        "crm_id,name,kind,parent_crm_id,domain,arr,region,status\n"
        "A-1010,Ashgrove Health,system,,ashgrovehealth.org,2350000,Midwest,\n"
    )

    write_owner_crm_id_column(csv_path, results)
    write_owner_crm_id_column(csv_path, results)  # idempotent: no duplicate column

    with csv_path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1
    assert rows[0]["owner_crm_id"] == results["ashgrove"].owner_crm_id
    with csv_path.open() as fh:
        header = fh.readline().strip().split(",")
    assert header.count("owner_crm_id") == 1


# ---------------------------------------------------------------------
# The real-customer (live graph) path.
# ---------------------------------------------------------------------


def test_is_vendor_side():
    assert is_vendor_side("internal_person") is True
    assert is_vendor_side("contact") is False
    assert is_vendor_side("account") is False


def test_load_internal_people_index_covers_the_five_reps(loaded_registry: Registry, db):
    people = load_internal_people_index(db)
    assert people.by_key.get("Sam Okafor") == SAM_OKAFOR
    assert people.name_by_id.get(SAM_OKAFOR) == "Sam Okafor"
    assert people.by_key.get("s.okafor@halvardmedical.com") == SAM_OKAFOR
    assert len(people.name_by_id) == 5


def _merge_ashgrove_medical_complaint(db, loaded_registry, *, speaker: str, locator: str):
    embedder = FakeEmbedder()
    llm = FakeLLMClient([])
    source = make_source_row(db, f"s-owner-test-{locator}", source_date=date(2026, 8, 20))
    fact = build_fact(
        "F-owner-1",
        FactType.COMPLAINT,
        "Ashgrove Medical Center received a late Tidewater T2 delivery.",
        ["Ashgrove Medical Center", "Tidewater T2"],
        locator=locator,
        quote="late delivery",
        speaker=speaker,
    )
    outcome = merge_fact(db, llm, embedder, loaded_registry, fact, source)
    assert outcome.status == "new"
    return outcome


def test_evidence_from_claims_counts_the_speaker_of_an_account_anchored_signal_note(
    loaded_registry: Registry, db
):
    _merge_ashgrove_medical_complaint(db, loaded_registry, speaker="Sam Okafor", locator="p1")

    slugs = account_note_slugs(db)
    account_slug = slugs[ASHGROVE_MEDICAL]
    people = load_internal_people_index(db)

    candidates = evidence_from_claims(db, account_slug, people)

    assert [c.person_crm_id for c in candidates] == [SAM_OKAFOR]
    assert candidates[0].source_count == 1
    assert candidates[0].last_seen == date(2026, 8, 20)


def test_evidence_from_claims_counts_commitments_only_for_the_commitment_fact_type(
    loaded_registry: Registry, db
):
    embedder = FakeEmbedder()
    llm = FakeLLMClient([])
    source = make_source_row(db, "s-owner-commit-1", source_date=date(2026, 8, 20))
    fact = build_fact(
        "F-owner-commit",
        FactType.COMMITMENT,
        "Sam Okafor committed to a written root-cause summary for Ashgrove Medical Center.",
        ["Ashgrove Medical Center"],
        locator="p2",
        quote="I'll send a written root-cause summary",
        speaker="Sam Okafor",
    )
    outcome = merge_fact(db, llm, embedder, loaded_registry, fact, source)
    assert outcome.status == "new"

    slugs = account_note_slugs(db)
    people = load_internal_people_index(db)
    candidates = evidence_from_claims(db, slugs[ASHGROVE_MEDICAL], people)

    assert candidates[0].person_crm_id == SAM_OKAFOR
    assert candidates[0].commitments == 1


def test_evidence_from_claims_matches_a_participant_by_email_when_name_differs(
    loaded_registry: Registry, db
):
    """`sources.participants` carries `{name, role, email}` -- a
    participant matched by email (not the exact display name) still
    counts as the same vendor rep."""
    _merge_ashgrove_medical_complaint(db, loaded_registry, speaker="Rob Tallis", locator="p3")
    with db.cursor() as cur:
        cur.execute(
            "UPDATE sources SET participants = %s::jsonb WHERE slug = %s",
            (
                '[{"name": "S. Okafor", "role": null, "email": "s.okafor@halvardmedical.com"}]',
                "s-owner-test-p3",
            ),
        )
    db.commit()

    slugs = account_note_slugs(db)
    people = load_internal_people_index(db)
    candidates = evidence_from_claims(db, slugs[ASHGROVE_MEDICAL], people)

    assert [c.person_crm_id for c in candidates] == [SAM_OKAFOR]


def test_propose_owners_skips_accounts_with_no_evidence(loaded_registry: Registry, db):
    proposals = propose_owners(db)
    assert proposals == []  # no claims merged yet in this test's own DB

    _merge_ashgrove_medical_complaint(db, loaded_registry, speaker="Sam Okafor", locator="p4")
    proposals = propose_owners(db)

    assert len(proposals) == 1
    assert proposals[0].account_crm_id == ASHGROVE_MEDICAL
    assert proposals[0].candidates[0].person_crm_id == SAM_OKAFOR


def test_confirm_owner_proposal_sets_the_field(loaded_registry: Registry, db):
    ok = confirm_owner_proposal(db, account_crm_id=ASHGROVE_MEDICAL, owner_crm_id=SAM_OKAFOR)
    assert ok is True

    with db.cursor() as cur:
        cur.execute(
            "SELECT owner.crm_id FROM registry_entities a "
            "JOIN registry_entities owner ON owner.id = a.owner_entity_id "
            "WHERE a.crm_id = %s",
            (ASHGROVE_MEDICAL,),
        )
        row = cur.fetchone()
    assert row == (SAM_OKAFOR,)


def test_owner_survives_a_note_re_render_and_a_merge(loaded_registry: Registry, db):
    """ "Owner is a registry fact, not note text ... a re-ingest or merge
    never clobbers it" (spec, Phase 1). Confirm an owner, then force the
    account note through two more real writes -- `ensure_entity_note`'s
    own re-render path and a full `merge_fact` call that adds a new claim
    to it -- and check `owner_entity_id` is untouched by either, since
    neither one is capable of writing to `registry_entities` at all."""
    assert (
        confirm_owner_proposal(db, account_crm_id=ASHGROVE_MEDICAL, owner_crm_id=SAM_OKAFOR) is True
    )

    from portolan.notes.persist import ensure_entity_note

    note_before = ensure_entity_note(db, loaded_registry, ASHGROVE_MEDICAL)
    # Re-render the note (bumps `notes.version`, touches only the notes
    # table) -- a no-op scenario `ensure_entity_note` itself already
    # exercises on every call after the first, but explicit here.
    note_after_rerender = ensure_entity_note(db, loaded_registry, ASHGROVE_MEDICAL)
    assert note_after_rerender.slug == note_before.slug

    _merge_ashgrove_medical_complaint(
        db, loaded_registry, speaker="Rob Tallis", locator="p-rerender"
    )

    with db.cursor() as cur:
        cur.execute(
            "SELECT owner.crm_id FROM registry_entities a "
            "JOIN registry_entities owner ON owner.id = a.owner_entity_id "
            "WHERE a.crm_id = %s",
            (ASHGROVE_MEDICAL,),
        )
        assert cur.fetchone() == (SAM_OKAFOR,)


def test_confirm_owner_proposal_refuses_a_customer_contact(loaded_registry: Registry, db):
    """The alias-conflict-style guard: a payload naming a real registry
    entity that is NOT vendor-side is refused, not silently written."""
    with db.cursor() as cur:
        cur.execute("SELECT crm_id FROM registry_entities WHERE kind = 'contact' LIMIT 1")
        customer_contact_crm_id = cur.fetchone()[0]

    ok = confirm_owner_proposal(
        db, account_crm_id=ASHGROVE_MEDICAL, owner_crm_id=customer_contact_crm_id
    )

    assert ok is False
    assert is_vendor_side_crm_id(db, customer_contact_crm_id) is False
    with db.cursor() as cur:
        cur.execute(
            "SELECT owner_entity_id FROM registry_entities WHERE crm_id = %s",
            (ASHGROVE_MEDICAL,),
        )
        assert cur.fetchone()[0] is None


def test_backfill_action_owners_sets_null_owners_from_the_account(loaded_registry: Registry, db):
    source = make_source_row(db, "s-backfill-1", source_date=date(2026, 8, 20))
    action = build_action(
        "Send Ashgrove Health a written delivery root-cause summary",
        owner_mention=None,
        due=date(2026, 8, 28),
        locator="p9",
    )
    from portolan.merge.actions import merge_action

    outcome = merge_action(db, loaded_registry, action, source, entity_id=ASHGROVE)
    with db.cursor() as cur:
        cur.execute("SELECT owner_entity_id FROM actions WHERE id = %s", (outcome.action_id,))
        assert cur.fetchone()[0] is None  # no owner set yet -- Ashgrove has none either

    assert confirm_owner_proposal(db, account_crm_id=ASHGROVE, owner_crm_id=SAM_OKAFOR) is True

    updated = backfill_action_owners(db)

    assert updated == 1
    with db.cursor() as cur:
        cur.execute(
            "SELECT owner.crm_id FROM actions a "
            "JOIN registry_entities owner ON owner.id = a.owner_entity_id "
            "WHERE a.id = %s",
            (outcome.action_id,),
        )
        assert cur.fetchone() == (SAM_OKAFOR,)

    # Idempotent: a second run touches nothing (owner_entity_id is no
    # longer null).
    assert backfill_action_owners(db) == 0
