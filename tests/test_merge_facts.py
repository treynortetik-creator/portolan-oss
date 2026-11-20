"""Tests for `portolan.merge.merge_fact` (`01-spec/build-contracts.md`
"Merge"), against the real registry and `corpus/truth-table.yaml` facts.

Run with this module's own test database:
    createdb portolan_test_merge
    DATABASE_URL=postgresql://localhost/portolan_test_merge \
        uv run pytest tests/test_merge_facts.py -v
"""

from __future__ import annotations

from datetime import date

from portolan.extract.schema import FactType
from portolan.llm.fake import FakeLLMClient
from portolan.merge.facts import merge_fact
from portolan.notes import read_note
from portolan.registry import Registry, add_alias
from portolan.search.embed import FakeEmbedder
from tests.merge_helpers import (
    ASHGROVE,
    ASHGROVE_MEDICAL,
    COBALT_RIDGE,
    PINECREST,
    build_fact,
    loaded_registry_fixture,  # noqa: F401 - registers the `loaded_registry` fixture
    make_source_row,
)


def _embedder() -> FakeEmbedder:
    return FakeEmbedder()


# --------------------------------------------------------------------
# F02: same fact, ticket then call -> one signal note, two citations,
# exactly one compare call (the gate skips the call on the first, brand
# new claim; a near-duplicate second claim gets exactly one).
# --------------------------------------------------------------------


def test_f02_ticket_then_call_yields_one_signal_note_two_citations(loaded_registry: Registry, db):
    embedder = _embedder()
    llm = FakeLLMClient([{"verdict": "same", "reason": "same delivery, retold on the call"}])

    source1 = make_source_row(
        db, "s-ticket-031", source_type="ticket", source_date=date(2026, 8, 5)
    )
    fact1 = build_fact(
        "F02",
        FactType.COMPLAINT,
        "Larkspur Memorial Hospital reported a Tidewater T2 consumable delivery "
        "arriving four days late on 2026-08-04.",
        ["Larkspur Memorial Hospital", "Tidewater T2"],
        locator="T-031#1",
        quote="4 days late",
        sentiment=-1,
    )
    outcome1 = merge_fact(db, llm, embedder, loaded_registry, fact1, source1)
    assert outcome1.status == "new"
    assert outcome1.compare_calls == 0  # brand new note, nothing to compare against

    source2 = make_source_row(db, "s-call-009", source_type="call", source_date=date(2026, 8, 5))
    fact2 = build_fact(
        "F02b",
        FactType.COMPLAINT,
        "Larkspur Memorial Hospital had that late order last week for the "
        "Tidewater T2 consumables.",
        ["Larkspur Memorial Hospital", "Tidewater T2"],
        locator="00:04:12",
        quote="that late order last week",
        sentiment=-1,
    )
    outcome2 = merge_fact(db, llm, embedder, loaded_registry, fact2, source2)

    assert outcome2.status == "same"
    assert outcome2.compare_calls == 1
    assert outcome2.note_slug == outcome1.note_slug
    assert outcome2.note_slug == "signal-complaint-larkspur-memorial-hospital"

    # Review finding G4: a `same` verdict is ONE claim with a second
    # citation, never a second claim carrying the other source's words
    # under this source's citation.
    note, claims = read_note(db, outcome1.note_slug)
    assert len(claims) == 1
    claim = claims[0]
    assert outcome2.claim_id == claim.id == outcome1.claim_id
    assert claim.source_slug == "s-ticket-031"
    assert claim.locator == "T-031#1"
    assert len(claim.citations) == 1
    extra = claim.citations[0]
    assert extra.source_slug == "s-call-009"
    assert extra.locator == "00:04:12"
    # Each citation's locator points into its own source: the primary
    # citation's locator is the ticket's comment id, the extra citation's
    # is the call's timestamp -- never the other source's locator under
    # this source's slug (the exact provenance failure G4 fixes).
    assert claim.locator != extra.locator
    # The account link is written immediately (a different note than the
    # home note it's linked from); the product link is only *requested*
    # here -- `merge_extraction` folds it into the home note's one
    # end-of-source write (see `MergeOutcome.related_links`'s docstring),
    # which calling `merge_fact` directly, as this test does, bypasses.
    assert note.related == ["larkspur-memorial-hospital"]
    assert outcome1.related_links == ["tidewater-t2"]
    assert outcome2.related_links == ["tidewater-t2"]


def test_gate_skips_the_compare_call_when_nothing_similar_exists(loaded_registry: Registry, db):
    embedder = _embedder()
    llm = FakeLLMClient([])  # any call raises IndexError -- proves zero calls happen

    source1 = make_source_row(db, "s-call-005", source_type="call", source_date=date(2026, 6, 9))
    fact1 = build_fact(
        "F12",
        FactType.PRICING,
        "Larkspur Regional Health expects a 12 percent volume discount on "
        "Tidewater T2 consumables at the 2027 renewal.",
        ["Larkspur Regional Health", "Tidewater T2"],
        quote="we'd be looking for something around twelve points off",
    )
    outcome1 = merge_fact(db, llm, embedder, loaded_registry, fact1, source1)
    assert outcome1.status == "new"
    assert outcome1.compare_calls == 0

    # A second, completely unrelated fact type on the SAME account must
    # also skip the gate -- it lands on a different signal note (pricing
    # vs usage) with no existing claims to compare against.
    fact2 = build_fact(
        "F_unrelated",
        FactType.USAGE,
        "Larkspur Regional Health runs the Tidewater T2 fleet across two campuses.",
        ["Larkspur Regional Health", "Tidewater T2"],
    )
    outcome2 = merge_fact(db, llm, embedder, loaded_registry, fact2, source1)
    assert outcome2.status == "new"
    assert outcome2.compare_calls == 0
    assert outcome2.note_slug != outcome1.note_slug


# --------------------------------------------------------------------
# F11 -> F11b: two clocks, supersession.
# --------------------------------------------------------------------


def test_f11_then_f11b_supersedes_with_both_clocks(loaded_registry: Registry, db):
    embedder = _embedder()
    llm = FakeLLMClient([{"verdict": "contradicts", "reason": "the renewal date moved"}])

    source1 = make_source_row(
        db, "s-crm-opp-a1010", source_type="crm", source_date=date(2026, 4, 14)
    )
    fact1 = build_fact(
        "F11",
        FactType.DECISION,
        "Ashgrove Health's HalvardCare contract renews on 2027-03-31.",
        ["Ashgrove Health", "HalvardCare"],
        valid_from=date(2026, 4, 14),
    )
    outcome1 = merge_fact(db, llm, embedder, loaded_registry, fact1, source1)
    assert outcome1.status == "new"

    source2 = make_source_row(db, "s-call-013", source_type="call", source_date=date(2026, 8, 20))
    fact2 = build_fact(
        "F11b",
        FactType.DECISION,
        "Ashgrove Health's HalvardCare renewal was moved to 2027-06-30 at "
        "Ashgrove's request on 2026-08-20.",
        ["Ashgrove Health", "HalvardCare"],
        valid_from=date(2026, 8, 20),
    )
    outcome2 = merge_fact(db, llm, embedder, loaded_registry, fact2, source2)

    assert outcome2.status == "contradicts"
    assert outcome2.compare_calls == 1
    assert outcome2.note_slug == outcome1.note_slug

    _note, claims = read_note(db, outcome1.note_slug)
    old_claim = next(c for c in claims if c.id == outcome1.claim_id)
    new_claim = next(c for c in claims if c.id == outcome2.claim_id)

    # World time: the old claim's own valid_from is untouched.
    assert old_claim.valid_from == date(2026, 4, 14)
    # Record time: it now carries the supersession.
    assert old_claim.superseded_at is not None
    assert old_claim.superseded_by == new_claim.id
    assert new_claim.superseded_at is None


# --------------------------------------------------------------------
# Review finding 2a (blocker): superseding an already-superseded claim
# must not silently overwrite the first correction. Tested directly
# against `_supersede`, not through `merge_fact` end to end -- finding
# 2b's fix (below, `test_merge_gate.py`) means a superseded claim never
# reaches the gate's candidate pool any more in the live pipeline, so the
# guard this test proves is defense in depth against anything that still
# calls `_supersede` twice for the same old claim (a race between two
# concurrent writers, a retried/re-driven source, ...), not a path the
# ordinary single-threaded merge flow can reach post-fix.
# --------------------------------------------------------------------


def test_supersede_is_idempotent_and_queues_a_review_row_on_a_second_call(
    loaded_registry: Registry, db
):
    from portolan.merge.facts import _supersede

    source = make_source_row(
        db, "s-crm-opp-a1010-idem", source_type="crm", source_date=date(2026, 4, 14)
    )
    fact = build_fact(
        "F11-idem",
        FactType.DECISION,
        "Ashgrove Health's HalvardCare contract renews on 2027-03-31.",
        ["Ashgrove Health", "HalvardCare"],
        valid_from=date(2026, 4, 14),
    )
    embedder = _embedder()
    outcome = merge_fact(db, FakeLLMClient([]), embedder, loaded_registry, fact, source)
    assert outcome.status == "new"
    old_claim_id = outcome.claim_id

    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO claims (note_slug, fact_type, claim_text, source_slug, locator, "
            "recorded) VALUES (%s, 'decision', 'first correction', %s, 'test#1', now()) "
            "RETURNING id",
            (outcome.note_slug, source.slug),
        )
        first_new_id = cur.fetchone()[0]
        cur.execute(
            "INSERT INTO claims (note_slug, fact_type, claim_text, source_slug, locator, "
            "recorded) VALUES (%s, 'decision', 'second correction', %s, 'test#2', now()) "
            "RETURNING id",
            (outcome.note_slug, source.slug),
        )
        second_new_id = cur.fetchone()[0]

    blocked_first = _supersede(db, old_claim_id, first_new_id, date(2027, 6, 30))
    assert blocked_first is False

    with db.cursor() as cur:
        cur.execute(
            "SELECT superseded_by, valid_to, superseded_at FROM claims WHERE id = %s",
            (old_claim_id,),
        )
        first_superseded_by, first_valid_to, first_superseded_at = cur.fetchone()
    assert first_superseded_by == first_new_id
    assert first_valid_to == date(2027, 6, 30)
    assert first_superseded_at is not None

    # The second call must not touch any of the three fields the first
    # call set -- the whole point of the guard.
    blocked_second = _supersede(db, old_claim_id, second_new_id, date(2028, 1, 1))
    assert blocked_second is True

    with db.cursor() as cur:
        cur.execute(
            "SELECT superseded_by, valid_to, superseded_at FROM claims WHERE id = %s",
            (old_claim_id,),
        )
        superseded_by, valid_to, superseded_at = cur.fetchone()
    assert superseded_by == first_new_id
    assert valid_to == date(2027, 6, 30)
    assert superseded_at == first_superseded_at

    with db.cursor() as cur:
        cur.execute(
            "SELECT kind, payload->>'attempted_new_claim_id', payload->>'reason' "
            "FROM review_queue WHERE payload->>'old_claim_id' = %s "
            "ORDER BY id DESC LIMIT 1",
            (str(old_claim_id),),
        )
        row = cur.fetchone()
    assert row is not None
    kind, attempted_new_claim_id, reason = row
    assert kind == "other"
    assert int(attempted_new_claim_id) == second_new_id
    assert "already superseded" in reason


# --------------------------------------------------------------------
# F15 -> F15b: neither fact carries an explicit valid_from. Item 6
# (e2e-run follow-up wave) removed the old `source.source_date` fallback
# entirely -- `portolan_e2e` had 11 claims marked superseded this way
# against the corpus's 2 planted reversals, none of which turned out to
# be a real contradiction on inspection (see `_explicit_supersession_
# date`'s docstring) -- so an undated `contradicts` verdict like this one
# now downgrades to `extends` (both claims stay live) with a
# `review_queue` row, rather than silently superseding via a comparison
# that was really just comparing two sources' dates, not the facts'.
# --------------------------------------------------------------------


def test_f15_then_f15b_downgrades_undated_contradiction_to_extends(loaded_registry: Registry, db):
    embedder = _embedder()
    llm = FakeLLMClient(
        [{"verdict": "contradicts", "reason": "now evaluating a competitor after all"}]
    )

    source1 = make_source_row(db, "s-call-004", source_type="call", source_date=date(2026, 6, 17))
    fact1 = build_fact(
        "F15",
        FactType.COMPETITOR_MENTION,
        "Northlake Health System said on 2026-06-17 that it was not evaluating other pump vendors.",
        ["Northlake Health System"],
        quote="we're not shopping around",
    )
    outcome1 = merge_fact(db, llm, embedder, loaded_registry, fact1, source1)
    assert outcome1.status == "new"

    source2 = make_source_row(db, "s-call-011", source_type="call", source_date=date(2026, 8, 14))
    fact2 = build_fact(
        "F15b",
        FactType.COMPETITOR_MENTION,
        "Northlake Health System began evaluating Corvane Systems pumps for "
        "Northlake General's new tower in 2026-08.",
        ["Northlake Health System", "Northlake General Hospital", "Corvane Systems"],
        quote="we've had Corvane in to demo for the tower",
    )
    outcome2 = merge_fact(db, llm, embedder, loaded_registry, fact2, source2)

    # The model's own verdict was "contradicts", but with neither fact
    # explicitly dated, the merge reports what it actually did to the
    # graph -- added a second, still-live claim -- not what the model said.
    assert outcome2.status == "extends"
    _note, claims = read_note(db, outcome1.note_slug)
    old_claim = next(c for c in claims if c.id == outcome1.claim_id)
    new_claim = next(c for c in claims if c.id == outcome2.claim_id)
    assert old_claim.superseded_at is None
    assert old_claim.superseded_by is None
    assert old_claim.valid_from is None
    assert new_claim.valid_from is None

    with db.cursor() as cur:
        cur.execute(
            "SELECT kind, payload->>'new_claim_id', payload->>'reason' FROM review_queue "
            "WHERE payload->>'old_claim_id' = %s",
            (str(old_claim.id),),
        )
        row = cur.fetchone()
    assert row is not None
    kind, new_claim_id, reason = row
    assert kind == "other"
    assert int(new_claim_id) == new_claim.id
    assert "downgraded to extends" in reason


# --------------------------------------------------------------------
# F03: "Larkspur" resolves, per the registry's tiers, to the Larkspur
# system account -- NOT Larkspur Memorial, the facility the truth table's
# planted fact is actually about (phrasing note: mentions "Larkspur" only,
# not "Memorial"). This asserts what actually happens rather than forcing
# a particular "correct" resolution: the home note is Cobalt Ridge (the
# fact's first, unambiguous account entity), and "Larkspur" is linked into
# the claim text as the *system* account, a real but coarser entity than
# the facility a human reading the source call would infer.
# --------------------------------------------------------------------


def test_f03_larkspur_mention_resolves_to_the_system_not_the_facility(
    loaded_registry: Registry, db
):
    embedder = _embedder()
    llm = FakeLLMClient([])

    source = make_source_row(db, "s-email-052", source_type="email", source_date=date(2026, 8, 7))
    fact = build_fact(
        "F03",
        FactType.COMPLAINT,
        "Cobalt Ridge Medical Center had a Tidewater T2 delivery arrive three "
        "days late on 2026-08-06 and had to borrow consumables from Larkspur.",
        ["Cobalt Ridge Medical Center", "Larkspur", "Tidewater T2"],
        quote="delivery arrive three days late",
    )
    outcome = merge_fact(db, llm, embedder, loaded_registry, fact, source)

    assert outcome.status == "new"
    assert outcome.note_slug == "signal-complaint-cobalt-ridge-medical-center"

    _note, claims = read_note(db, outcome.note_slug)
    claim_text = claims[0].claim_text
    # Resolved to the Larkspur *system* account (alias "Larkspur" only
    # belongs to it, not to Larkspur Memorial) -- the facility's slug must
    # NOT appear.
    assert "[[larkspur-regional-health]]" in claim_text
    assert "larkspur-memorial-hospital" not in claim_text


# --------------------------------------------------------------------
# F30: a contact_change fact arriving from an address not in the CRM
# (Maria Chen's personal Gmail). The fact's `entities` mention her by
# name, not by that address, and the registry has exactly one "Maria
# Chen" -- the real CRM contact -- so this resolves cleanly by name
# despite the source being unrecognised. Asserting what happens, per the
# build instructions.
# --------------------------------------------------------------------


def test_f30_personal_email_source_resolves_the_contact_by_name(loaded_registry: Registry, db):
    embedder = _embedder()
    llm = FakeLLMClient([])

    source = make_source_row(
        db,
        "s-email-058",
        source_type="email",
        source_date=date(2026, 8, 24),
    )
    fact = build_fact(
        "F30",
        FactType.CONTACT_CHANGE,
        "Maria Chen moved from Director of Procurement to Director of Clinical "
        "Operations at Ashgrove Health in 2026-08.",
        ["Ashgrove Health", "Maria Chen"],
    )
    outcome = merge_fact(db, llm, embedder, loaded_registry, fact, source)

    assert outcome.status == "new"
    note, _claims = read_note(db, outcome.note_slug)
    assert note.crm_id == "C-2001"  # the real CRM Maria Chen, resolved by name
    assert note.type.value == "contact"


# --------------------------------------------------------------------
# CompareVerdict case normalization (fix wave 2026-09-12, killed-run-
# against-portolan_demo2 investigation): found running this pipeline end
# to end against a real `amazon/nova-lite-v1` compare call, which came
# back `{"verdict": "SAME", ...}` -- schema-valid JSON, but not one of
# `CompareVerdict.verdict`'s exact lowercase `Literal` members, so a
# strict `model_validate` raised. Reproduced directly: `merge_fact` ->
# `merge_extraction` -> `_retry_stage` (three attempts, all doomed -- the
# model says the same thing every time) -> the whole source dead-lettered
# and `failed`, zero notes/claims, for a compare call that actually
# answered the real question correctly. `portolan.merge.models.
# CompareVerdict` now normalizes a string verdict's case before the
# `Literal` check runs.
# --------------------------------------------------------------------


def test_compare_verdict_accepts_uppercase_and_mixed_case_from_a_real_model():
    from portolan.merge.models import CompareVerdict

    for raw in ("SAME", "Same", " same ", "EXTENDS", "Contradicts"):
        verdict = CompareVerdict.model_validate({"verdict": raw, "reason": "because"})
        assert verdict.verdict == raw.strip().lower()


def test_compare_verdict_still_rejects_a_genuinely_unknown_verdict():
    import pytest
    from pydantic import ValidationError

    from portolan.merge.models import CompareVerdict

    with pytest.raises(ValidationError):
        CompareVerdict.model_validate({"verdict": "maybe", "reason": "because"})


# --------------------------------------------------------------------
# Company-context wave, 2026-09-17: `claims.company_context_updated_at`
# provenance stamp (migration 0021_claims_context_stamp.sql).
# --------------------------------------------------------------------


def test_merge_fact_stamps_company_context_updated_at_on_a_new_claim(loaded_registry: Registry, db):
    from datetime import UTC, datetime

    stamp = datetime(2026, 9, 15, 12, 0, tzinfo=UTC)
    source = make_source_row(db, "s-ticket-999", source_type="ticket", source_date=date(2026, 8, 5))
    fact = build_fact(
        "F-CTX",
        FactType.COMPLAINT,
        "Larkspur Memorial Hospital's Tidewater T2 order arrived late.",
        ["Larkspur Memorial Hospital", "Tidewater T2"],
        locator="T-999#1",
        quote="arrived late",
        sentiment=-1,
    )
    outcome = merge_fact(
        db,
        FakeLLMClient([]),
        _embedder(),
        loaded_registry,
        fact,
        source,
        company_context_updated_at=stamp,
    )
    assert outcome.status == "new"
    _note, claims = read_note(db, outcome.note_slug)
    assert claims[0].company_context_updated_at == stamp


def test_merge_fact_leaves_the_stamp_null_when_not_given(loaded_registry: Registry, db):
    source = make_source_row(db, "s-ticket-998", source_type="ticket", source_date=date(2026, 8, 5))
    fact = build_fact(
        "F-CTX2",
        FactType.COMPLAINT,
        "Larkspur Memorial Hospital's Tidewater T2 order arrived late again.",
        ["Larkspur Memorial Hospital", "Tidewater T2"],
        locator="T-998#1",
        quote="arrived late again",
        sentiment=-1,
    )
    outcome = merge_fact(db, FakeLLMClient([]), _embedder(), loaded_registry, fact, source)
    _note, claims = read_note(db, outcome.note_slug)
    assert claims[0].company_context_updated_at is None


# --------------------------------------------------------------------
# Bug hunt 2026-09-18, finding 3: an ambiguous mention that matches an
# account AND its own child used to drop the fact entirely (`attach_to`
# fell through to `None` because `parents` was `{None, "A-1010"}`, never
# a single shared value -- `Registry.resolve`'s own fix, tested directly
# in `test_registry.py`). Here through the full `merge_fact` path: the
# fact must still land somewhere (spec sec 3, "facts are never dropped"),
# on the shared ancestor's signal note, flagged ambiguous in the claim
# text -- the shipped corpus's own "Ashgrove" case (real aliases: only
# `ashgrove-health` carries it; this test adds it to the child too, the
# way the bug hunt's own alias-seeding repro did).
# --------------------------------------------------------------------


def test_ashgrove_ambiguous_parent_child_mention_attaches_to_the_parent(
    loaded_registry: Registry, db
):
    add_alias(db, ASHGROVE_MEDICAL, "Ashgrove", "human")
    registry = Registry.load(db)

    source = make_source_row(db, "s-call-bug4", source_type="call", source_date=date(2026, 8, 20))
    fact = build_fact(
        "F-ASHGROVE-AMBIG",
        FactType.COMPLAINT,
        "Ashgrove reported a delayed shipment on the last call.",
        ["Ashgrove"],
        locator="00:02:00",
        quote="the shipment was late again",
        sentiment=-1,
    )
    outcome = merge_fact(db, FakeLLMClient([]), _embedder(), registry, fact, source)

    # Not dropped, not parked unresolved -- landed on the shared parent
    # (Ashgrove Health, A-1010)'s own complaint signal note.
    assert outcome.status == "new"
    assert outcome.note_slug == "signal-complaint-ashgrove-health"
    assert outcome.claim_id is not None

    _note, claims = read_note(db, outcome.note_slug)
    claim = next(c for c in claims if c.id == outcome.claim_id)
    assert "(ambiguous: Ashgrove)" in claim.claim_text

    # An ambiguity that resolved to a shared parent is not the same thing
    # as one that couldn't resolve at all -- no `ambiguous_entity` review
    # row for THIS mention (it has a home; nothing to review).
    with db.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM review_queue WHERE kind = 'ambiguous_entity' "
            "AND source_slug = %s",
            (source.slug,),
        )
        assert cur.fetchone()[0] == 0


# --------------------------------------------------------------------
# Bug hunt 2026-09-18, finding 7: `_queue_ambiguous` had no dedup, unlike
# its sibling `registry.queue_unknown` -- reprocessing the same source
# (a retry, or `redrive()`) inserted one review_queue row per attempt for
# the SAME still-unresolvable mention. Also covers the review-queue-
# usability requirement (task guidance): the row must always carry the
# fact's claim, quote and locator, not just the bare mention/candidates.
# --------------------------------------------------------------------


def test_truly_ambiguous_mention_with_no_shared_parent_dedups_and_carries_the_fact(
    loaded_registry: Registry, db
):
    # Pinecrest (A-1001) and Cobalt Ridge (A-1020) are both top-level
    # accounts with no parent relationship to each other -- a mention
    # matching both has nothing to attach to.
    add_alias(db, PINECREST, "Wrenfield Group", "human")
    add_alias(db, COBALT_RIDGE, "Wrenfield Group", "human")
    registry = Registry.load(db)

    source = make_source_row(db, "s-call-bug7", source_type="call", source_date=date(2026, 8, 20))
    fact = build_fact(
        "F-WRENFIELD-AMBIG",
        FactType.COMPLAINT,
        "Wrenfield Group asked about the outage on the call.",
        ["Wrenfield Group"],
        locator="00:03:00",
        quote="what happened with the outage",
        sentiment=-1,
    )

    outcome1 = merge_fact(db, FakeLLMClient([]), _embedder(), registry, fact, source)
    assert outcome1.status == "queued_unknown"
    outcome2 = merge_fact(db, FakeLLMClient([]), _embedder(), registry, fact, source)
    assert outcome2.status == "queued_unknown"

    with db.cursor() as cur:
        cur.execute(
            "SELECT payload FROM review_queue WHERE kind = 'ambiguous_entity' AND source_slug = %s",
            (source.slug,),
        )
        rows = cur.fetchall()

    assert len(rows) == 1, "a second merge of the same source must not duplicate the row"
    payload = rows[0][0]
    assert payload["mention"] == "Wrenfield Group"
    assert sorted(payload["candidates"]) == sorted([PINECREST, COBALT_RIDGE])
    assert payload["claim"] == fact.claim
    assert payload["quote"] == fact.quote
    assert payload["locator"] == fact.locator
    assert payload["fact_type"] == fact.type.value


# --------------------------------------------------------------------
# Bug hunt 2026-09-18, finding 10: a `contradicts` verdict's review row
# (both the idempotent-supersession-blocked path and the downgraded-to-
# extends path) carried no `source_slug`, so a human couldn't trace it
# back to the call/ticket/email it came from.
# --------------------------------------------------------------------


def test_downgraded_contradiction_review_row_carries_source_slug(loaded_registry: Registry, db):
    llm = FakeLLMClient([{"verdict": "contradicts", "reason": "seems to disagree"}])
    embedder = _embedder()

    source1 = make_source_row(db, "s-call-bug10a", source_type="call", source_date=date(2026, 8, 1))
    fact1 = build_fact(
        "F-BUG10A",
        FactType.DECISION,
        "Ashgrove Health decided to expand the pilot program.",
        ["Ashgrove Health"],
    )
    outcome1 = merge_fact(db, llm, embedder, loaded_registry, fact1, source1)
    assert outcome1.status == "new"

    source2 = make_source_row(db, "s-call-bug10b", source_type="call", source_date=date(2026, 8, 2))
    fact2 = build_fact(
        "F-BUG10B",
        FactType.DECISION,
        "Ashgrove Health decided to pause the pilot program instead.",
        ["Ashgrove Health"],
    )
    # Neither fact carries an explicit valid_from -- item 6's downgrade
    # rule fires, both claims stay live, and a review row records the
    # model's own `contradicts` call.
    outcome2 = merge_fact(db, llm, embedder, loaded_registry, fact2, source2)
    assert outcome2.status == "extends"

    with db.cursor() as cur:
        cur.execute(
            "SELECT source_slug, payload->>'reason' FROM review_queue "
            "WHERE kind = 'other' AND payload->>'new_claim_id' = %s",
            (str(outcome2.claim_id),),
        )
        row = cur.fetchone()
    assert row is not None
    row_source_slug, reason = row
    assert row_source_slug == source2.slug
    assert "downgraded to extends" in reason


# --------------------------------------------------------------------
# Bug hunt 2026-09-18, finding 2: `claims.fact_type` carried a DB-level
# CHECK against only the closed 12-name base list, but a customer
# extension (extraction-schema-v0.md, `CustomerExtension.extra_types`) is
# a real, paid onboarding feature that adds open-ended per-customer type
# names -- no single CHECK could enumerate them for every future
# customer. Migration 0022 drops the constraint (Python-side validation,
# `Fact._validate_against_effective_schema`, is the real guard). This
# inserts directly (bypassing `merge_fact`, which never builds an
# extension-typed `Fact` today -- see the fix report on why the
# extraction-side `effective_schema` wiring into
# `ExtractionResult.parse_lenient` is a separate, out-of-scope fix) to
# prove the DB itself no longer rejects it -- this is exactly the
# `CheckViolation` the bug hunt's own repro hit against the un-migrated
# schema.
# --------------------------------------------------------------------


def test_extension_fact_type_no_longer_violates_the_claims_check(loaded_registry: Registry, db):
    from portolan.notes import ensure_entity_note

    anchor_note = ensure_entity_note(db, loaded_registry, ASHGROVE)
    source = make_source_row(
        db, "s-ticket-bug2", source_type="ticket", source_date=date(2026, 8, 5)
    )
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO claims (note_slug, fact_type, claim_text, source_slug, locator, recorded) "
            "VALUES (%s, %s, %s, %s, %s, now()) RETURNING id",
            (
                anchor_note.slug,
                "clinical_incident",
                "A customer-extension-typed fact.",
                source.slug,
                "T-BUG2#1",
            ),
        )
        new_id = cur.fetchone()[0]
    db.commit()

    with db.cursor() as cur:
        cur.execute("SELECT fact_type FROM claims WHERE id = %s", (new_id,))
        assert cur.fetchone()[0] == "clinical_incident"
