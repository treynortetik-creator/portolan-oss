"""Tests for `portolan.report.grounding`, revised contract (`01-spec/
actions-and-owners-2026-09-19.md`, "Phase 2, revised contract", "The
gate"). Uses a fake support judge throughout (`FakeSupportJudge`) so no
test needs a real model call; `tests/test_report_researcher_tools.py`
covers the finding tools, and the one real Haiku judge run lives only in
the real-run report handed to the coordinator, never in this suite.

The ten crafted fabrications from `01-spec/reviews/review-action-
researcher-2026-09-19.md` are kept below, adapted to the revised
contract's byte/referential/typed-judge design (several are now caught
mechanically, before the judge is ever asked -- the new gate has no
fuzzy-match or fragment-vote shortcut left to exploit at all):

1. negation inversion               -> test_attack_1_negation...
2. cross-account entity swap        -> test_attack_2_cross_account...
3. fragment-vote-invisible fabrication -> test_attack_3_verb_only...
4. union-enabled causal fabrication -> test_attack_4_union...
5. numeral licensed card-wide       -> test_attack_5_numeral...
6. empty evidence                   -> test_attack_6_empty_evidence...
7. fake due/confidence/reason       -> test_attack_7_fake_due... /
                                        test_attack_7b_fake_confidence...
8. superseded claim still grounds   -> test_attack_8_superseded...
9. invented claim id                -> test_attack_9_invented_claim_id
10. owner is a customer contact     -> test_attack_10_customer_owner...
"""

from __future__ import annotations

from datetime import date, timedelta

from portolan.extract.schema import FactType
from portolan.llm.fake import FakeLLMClient
from portolan.merge.facts import merge_fact
from portolan.notes import NoteRow, NoteStatus, NoteType, write_note
from portolan.notes.persist import ensure_entity_note
from portolan.registry import Registry
from portolan.report.grounding import mechanical_checks, run_grounding_gate
from portolan.report.researcher import ActionCard, EvidenceItem, InferenceSentence, RefSet
from portolan.report.researcher_tools import record_finding
from portolan.search.embed import FakeEmbedder
from portolan.util import clock as clock_module
from tests.merge_helpers import (
    ASHGROVE_MEDICAL,
    COBALT_RIDGE,
    SAM_OKAFOR,
    build_fact,
    loaded_registry_fixture,  # noqa: F401
    make_source_row,
)

CUSTOMER_CONTACT = "C-2001"  # Maria Chen, a customer contact -- never vendor-side.
PRIYA = "priya-natarajan"

RUN_ID = "test-run-1"
CANDIDATE_SLUG = "test-candidate"


class FakeSupportJudge:
    """A `SupportJudgeFn` a test fully controls: `not_supported` names
    sentences (by exact text) that should fail, `error_for` names ones
    that should simulate a judge error, everything else answers
    `default`. Records every call for assertions that the judge is
    actually invoked (never bypassed) and what it was asked to weigh."""

    def __init__(
        self,
        *,
        not_supported: set[str] | None = None,
        error_for: set[str] | None = None,
        default: str = "supported",
    ) -> None:
        self.not_supported = not_supported or set()
        self.error_for = error_for or set()
        self.default = default
        self.calls: list[tuple[str, str]] = []

    def __call__(self, *, sentence: str, reference_text: str) -> str | None:
        self.calls.append((sentence, reference_text))
        key = sentence.strip()
        if key in self.error_for:
            return None
        if key in self.not_supported:
            return "not_supported"
        return self.default


def _merge(
    db,
    registry: Registry,
    *,
    claim: str,
    quote: str,
    locator: str,
    fact_type=FactType.COMPLAINT,
    entities=None,
) -> EvidenceItem:
    """Merge one fact, then read back the STORED `claim_text`/`quote`/
    `source_slug`/`locator` verbatim -- the exact byte-for-byte shape
    `mechanical_checks` compares an `EvidenceItem` against, so a test
    fixture and the gate never quietly disagree about what "verbatim"
    means (e.g. `[[wikilink]]` rewriting)."""
    embedder = FakeEmbedder()
    # A callable, not an empty list: a second claim merged onto a note that
    # already carries a similar one triggers `merge_fact`'s own compare-
    # against-existing-claim call, which an empty canned-response list has
    # no answer for. "extends" (distinct fact, not a duplicate/contradiction)
    # is right for every claim this module merges -- each is its own new
    # complaint/decision, never a restatement of the one before it.
    llm = FakeLLMClient(lambda **_kwargs: {"verdict": "extends", "reason": "a distinct fact"})
    source = make_source_row(db, f"s-ground-{locator}", source_date=date(2026, 8, 20))
    fact = build_fact(
        f"F-ground-{locator}",
        fact_type,
        claim,
        entities or ["Ashgrove Medical Center", "Tidewater T2"],
        locator=locator,
        quote=quote,
        speaker="Rob Tallis",
    )
    outcome = merge_fact(db, llm, embedder, registry, fact, source)
    assert outcome.status in ("new", "extends")
    with db.cursor() as cur:
        cur.execute(
            "SELECT claim_text, quote, source_slug, locator FROM claims WHERE id = %s",
            (outcome.claim_id,),
        )
        stored_text, stored_quote, stored_source, stored_locator = cur.fetchone()
    return EvidenceItem(
        claim_id=outcome.claim_id,
        text=stored_text,
        quote=stored_quote,
        source_slug=stored_source,
        locator=stored_locator,
    )


_WHAT = "Follow up on the Tidewater T2 delivery at Ashgrove Medical Center."
_DEFAULT_DUE = date(2026, 9, 13)
_DEFAULT_DUE_RULE = "next_report"


def _good_card(
    *,
    account: str,
    owner: str,
    evidence: list[EvidenceItem],
    verdict="action",
    due=_DEFAULT_DUE,
    due_rule=_DEFAULT_DUE_RULE,
    findings=None,
    inference=None,
) -> ActionCard:
    return ActionCard(
        run_id=RUN_ID,
        candidate_slug=CANDIDATE_SLUG,
        brief_version="researcher-brief-v4",
        account=account,
        verdict=verdict,
        what=_WHAT,
        evidence=evidence,
        findings=findings or [],
        inference=inference or [],
        owner=owner,
        due=due,
        due_rule=due_rule,
        theme=None,
        confidence="medium",
        confidence_refs=RefSet(evidence_refs=[0] if evidence else [], finding_refs=[]),
    )


def test_a_good_action_card_passes(loaded_registry: Registry, db):
    account_slug = ensure_entity_note(db, loaded_registry, ASHGROVE_MEDICAL).slug
    item = _merge(
        db,
        loaded_registry,
        claim="Ashgrove Medical Center received a late Tidewater T2 delivery.",
        quote="late delivery",
        locator="p1",
    )
    card = _good_card(
        account=account_slug,
        owner=SAM_OKAFOR,
        evidence=[item],
        due=date(2026, 9, 20),
        due_rule="next_report",
    )
    judge = FakeSupportJudge()
    result = run_grounding_gate(
        db, card, judge=judge, as_of=date(2026, 9, 6), default_owner_crm_id=SAM_OKAFOR
    )
    assert result.passed, result.reason
    assert judge.calls  # the judge was actually asked about `what`


def test_a_trigger_stale_card_with_a_finding_passes(loaded_registry: Registry, db):
    """The Ashgrove-shaped acceptance case: no `due`/owner override
    needed for a `trigger_stale` verdict, just a finding/evidence item
    that says why the trigger is stale."""
    account_slug = ensure_entity_note(db, loaded_registry, ASHGROVE_MEDICAL).slug
    item = _merge(
        db,
        loaded_registry,
        claim="Ashgrove Medical Center's renewal moved to 2027-06-30.",
        quote="pushed the renewal to 2027-06-30",
        locator="p2",
    )
    finding_id = record_finding(
        db,
        run_id=RUN_ID,
        candidate_slug=CANDIDATE_SLUG,
        tool="renewal",
        arguments={"account_slug": account_slug},
        result={"close_date": "2027-06-30"},
    )
    card = _good_card(
        account=account_slug,
        owner=None,
        evidence=[item],
        verdict="trigger_stale",
        findings=[finding_id],
    )
    card.what = "The renewal-risk trigger for Ashgrove Medical Center is stale."
    judge = FakeSupportJudge()
    result = run_grounding_gate(
        db, card, judge=judge, as_of=date(2026, 9, 6), default_owner_crm_id=SAM_OKAFOR
    )
    assert result.passed, result.reason


# ---------------------------------------------------------------------
# The ten crafted fabrications, adapted.
# ---------------------------------------------------------------------


def test_attack_1_negation_inversion_is_dropped_when_the_judge_says_not_supported(
    loaded_registry: Registry, db
):
    """B1: "Ashgrove Medical Center confirmed it will cancel the Tidewater
    T2 contract" cited against a claim that says it will NOT cancel. Byte-
    verbatim evidence means the model can no longer paraphrase the claim
    at all -- the only way an inverted MEANING reaches the card is in
    `what`/`inference`, which is exactly what the typed judge exists to
    catch. This proves the gate actually calls the judge and honors a
    `not_supported` verdict, rather than any mechanical shortcut."""
    account_slug = ensure_entity_note(db, loaded_registry, ASHGROVE_MEDICAL).slug
    item = _merge(
        db,
        loaded_registry,
        claim="Ashgrove Medical Center said it will not cancel the Tidewater T2 contract.",
        quote="will not cancel",
        locator="p3",
    )
    card = _good_card(account=account_slug, owner=SAM_OKAFOR, evidence=[item])
    card.what = "Ashgrove Medical Center confirmed it will cancel the Tidewater T2 contract."
    judge = FakeSupportJudge(not_supported={card.what})
    result = run_grounding_gate(
        db, card, judge=judge, as_of=date(2026, 9, 6), default_owner_crm_id=SAM_OKAFOR
    )
    assert not result.passed
    assert result.reason == "judge_not_supported:what"


def test_attack_2_cross_account_entity_swap_is_dropped_mechanically(loaded_registry: Registry, db):
    """B1's second crafted sentence named a DIFFERENT account than the
    claim it cited. With `_claim_in_scope` enforced on every evidence
    item, an evidence item belonging to a different account (and no
    shared theme) is dropped before the judge is ever asked."""
    account_slug = ensure_entity_note(db, loaded_registry, ASHGROVE_MEDICAL).slug
    other_account = ensure_entity_note(db, loaded_registry, "A-1020").slug  # Cobalt Ridge
    item = _merge(
        db,
        loaded_registry,
        claim=(
            "Cobalt Ridge Medical Center cancelled its Tidewater T2 contract after "
            "repeated late deliveries."
        ),
        quote="cancelled its Tidewater T2 contract",
        locator="p4",
        entities=["Cobalt Ridge Medical Center", "Tidewater T2"],
    )
    # Force the claim onto the OTHER account's own scope (it already is,
    # via merge_fact); cite it from a card about `account_slug` instead.
    assert other_account != account_slug
    card = _good_card(account=account_slug, owner=SAM_OKAFOR, evidence=[item])
    judge = FakeSupportJudge()
    result = run_grounding_gate(
        db, card, judge=judge, as_of=date(2026, 9, 6), default_owner_crm_id=SAM_OKAFOR
    )
    assert not result.passed
    assert result.reason.startswith("claim_out_of_scope:")
    assert not judge.calls  # never reached the judge -- caught mechanically


def test_attack_3_verb_only_fabrication_is_dropped_when_the_judge_says_not_supported(
    loaded_registry: Registry, db
):
    """B2: a fabrication introducing no new proper noun or number used to
    be invisible to the fragment vote. That mechanism is gone entirely --
    there is no mechanical acceptance path left for sentence support at
    all, only the judge. Proven by a fake judge that answers
    `not_supported` for the fabricated sentence with no special-casing of
    "does it introduce a new name/number" anywhere in the gate itself."""
    account_slug = ensure_entity_note(db, loaded_registry, ASHGROVE_MEDICAL).slug
    item = _merge(
        db,
        loaded_registry,
        claim="Ashgrove Medical Center received a late Tidewater T2 delivery.",
        quote="late delivery",
        locator="p5",
    )
    card = _good_card(account=account_slug, owner=SAM_OKAFOR, evidence=[item])
    card.what = (
        "Ashgrove Medical Center has decided to terminate the Tidewater T2 contract at "
        "renewal and has already told its board."
    )
    judge = FakeSupportJudge(not_supported={card.what})
    result = run_grounding_gate(
        db, card, judge=judge, as_of=date(2026, 9, 6), default_owner_crm_id=SAM_OKAFOR
    )
    assert not result.passed
    assert result.reason == "judge_not_supported:what"


def test_attack_4_union_backed_causal_fabrication_is_dropped_when_the_judge_says_not_supported(
    loaded_registry: Registry, db
):
    """B3: a sentence citing two claims that each fail alone used to have
    its fuzzy score inflated by concatenation. The revised contract makes
    a union of EXPLICIT references the deliberately correct design (spec:
    "this is the one place a union is right") -- proven safe here because
    the union is still judged for real support: two genuinely unrelated
    evidence items, concatenated, still get `not_supported` from a judge
    that actually reads them."""
    account_slug = ensure_entity_note(db, loaded_registry, ASHGROVE_MEDICAL).slug
    item_a = _merge(
        db,
        loaded_registry,
        claim="Ashgrove Medical Center reported a Tidewater T2 refrigeration issue.",
        quote="refrigeration issue",
        locator="p6a",
    )
    item_b = _merge(
        db,
        loaded_registry,
        claim="Ashgrove Medical Center asked about Tidewater T2 pricing.",
        quote="asked about pricing",
        locator="p6b",
    )
    sentence = (
        "Corvane, Meridian and Bayline approached Ashgrove Medical Center with unsolicited "
        "competitive proposals because of the Tidewater T2 refrigeration outage."
    )
    card = _good_card(
        account=account_slug,
        owner=SAM_OKAFOR,
        evidence=[item_a, item_b],
        inference=[InferenceSentence(sentence=sentence, evidence_refs=[0, 1], finding_refs=[])],
    )
    judge = FakeSupportJudge(not_supported={sentence})
    result = run_grounding_gate(
        db, card, judge=judge, as_of=date(2026, 9, 6), default_owner_crm_id=SAM_OKAFOR
    )
    assert not result.passed
    assert result.reason == "judge_not_supported:inference:0"
    # both evidence items' text reached the judge, concatenated -- proving
    # this IS the union path, not two independent (and therefore
    # incomparable) calls.
    _sentence, reference_text = judge.calls[-1]
    assert "refrigeration issue" in reference_text
    assert "pricing" in reference_text


def test_attack_5_numeral_licensed_from_an_unreferenced_item_is_dropped_mechanically(
    loaded_registry: Registry, db
):
    """B4: a numeral licensed card-wide let a sentence citing claim B use
    a number that only appears in claim A. Numerals are now scoped to a
    sentence's OWN `evidence_refs`/`finding_refs` -- the numeral "12" in
    an inference sentence that references only item B (which never
    mentions it) is dropped before the judge is ever asked."""
    account_slug = ensure_entity_note(db, loaded_registry, ASHGROVE_MEDICAL).slug
    item_a = _merge(
        db,
        loaded_registry,
        claim="Ashgrove Medical Center reported 12 late Tidewater T2 deliveries.",
        quote="12 late deliveries",
        locator="p7a",
    )
    item_b = _merge(
        db,
        loaded_registry,
        claim="Ashgrove Medical Center asked about Tidewater T2 renewal terms.",
        quote="renewal terms",
        locator="p7b",
    )
    sentence = "Ashgrove Medical Center faces a 12% churn risk on Tidewater T2."
    card = _good_card(
        account=account_slug,
        owner=SAM_OKAFOR,
        evidence=[item_a, item_b],
        inference=[InferenceSentence(sentence=sentence, evidence_refs=[1], finding_refs=[])],
    )
    judge = FakeSupportJudge()
    result = run_grounding_gate(
        db, card, judge=judge, as_of=date(2026, 9, 6), default_owner_crm_id=SAM_OKAFOR
    )
    assert not result.passed
    assert result.reason.startswith("numeral_not_supported:inference:0:")
    assert not judge.calls


# ---------------------------------------------------------------------
# B1 (review-action-researcher-2026-09-20.md): a finding's own ID, not
# its loaded RESULT, was what `_numeral_source_text` licensed --
# `grounding.py:207`, `parts.append(json.dumps(card.findings[j]))`, and
# `card.findings` is `list[int]`. Two failures from one line: a
# fabricated count that happens to equal the cited finding's id passed
# the numeral gate, and a TRUE count copied out of the finding's own
# result was rejected because the id-as-text never contained it -- the
# real Ashgrove run's own bug ("...versus 4 commitment claims recorded
# before that date" deleted by the model rather than fixed).
# ---------------------------------------------------------------------


def test_b1_a_fabricated_count_matching_the_finding_id_is_no_longer_licensed(
    loaded_registry: Registry, db
):
    account_slug = ensure_entity_note(db, loaded_registry, ASHGROVE_MEDICAL).slug
    item = _merge(
        db,
        loaded_registry,
        claim="Ashgrove Medical Center received a late Tidewater T2 delivery.",
        quote="late delivery",
        locator="p-b1a",
    )
    finding_id = record_finding(
        db,
        run_id=RUN_ID,
        candidate_slug=CANDIDATE_SLUG,
        tool="count_by",
        arguments={"fact_type": "complaint"},
        result={"groups": [], "total": 0},
    )
    # The fabricated numeral is the finding's OWN id -- on c4cc2c6 this
    # passed, because `_numeral_source_text` licensed `json.dumps(finding
    # _id)` (the bare id) as the finding's contributed text, and that
    # string IS exactly this numeral. The finding's real result (`total:
    # 0`) says no such thing.
    sentence = (
        f"Ashgrove Medical Center reported {finding_id} late Tidewater T2 deliveries this window."
    )
    card = _good_card(
        account=account_slug,
        owner=SAM_OKAFOR,
        evidence=[item],
        findings=[finding_id],
        inference=[
            InferenceSentence(sentence=sentence, evidence_refs=[], finding_refs=[0]),
        ],
    )
    errors = mechanical_checks(db, card, default_owner_crm_id=SAM_OKAFOR, as_of=date(2026, 9, 6))
    assert any(e.startswith("numeral_not_supported:inference:0:") for e in errors), errors


def test_b1_a_true_numeral_from_the_findings_own_result_is_licensed(loaded_registry: Registry, db):
    account_slug = ensure_entity_note(db, loaded_registry, ASHGROVE_MEDICAL).slug
    item = _merge(
        db,
        loaded_registry,
        claim="Ashgrove Medical Center received a late Tidewater T2 delivery.",
        quote="late delivery",
        locator="p-b1b",
    )
    # A large, id-improbable number so this assertion can never
    # accidentally pass because the finding's OWN id happens to match --
    # the point is that the number comes from the RESULT, not the id.
    finding_id = record_finding(
        db,
        run_id=RUN_ID,
        candidate_slug=CANDIDATE_SLUG,
        tool="absence_check",
        arguments={"account_slug": account_slug},
        result={"count": 0, "count_before": 500, "had_matches_before": True},
    )
    assert finding_id != 500
    sentence = "500 commitment claims were recorded before that date."
    card = _good_card(
        account=account_slug,
        owner=SAM_OKAFOR,
        evidence=[item],
        findings=[finding_id],
        inference=[
            InferenceSentence(sentence=sentence, evidence_refs=[], finding_refs=[0]),
        ],
    )
    errors = mechanical_checks(db, card, default_owner_crm_id=SAM_OKAFOR, as_of=date(2026, 9, 6))
    assert not any(e.startswith("numeral_not_supported:inference:0:") for e in errors), errors


def test_attack_6_empty_evidence_and_findings_is_dropped(loaded_registry: Registry, db):
    """Should-fix 3: `evidence=[]` used to pass because nothing required
    it to be non-empty. `mechanical_checks` now requires at least one
    evidence item OR at least one finding on every verdict."""
    account_slug = ensure_entity_note(db, loaded_registry, ASHGROVE_MEDICAL).slug
    card = _good_card(account=account_slug, owner=SAM_OKAFOR, evidence=[])
    errors = mechanical_checks(db, card, default_owner_crm_id=SAM_OKAFOR, as_of=date(2026, 9, 6))
    assert "no_evidence_or_findings" in errors


def test_an_action_verdict_with_findings_only_and_no_evidence_is_dropped(
    loaded_registry: Registry, db
):
    """Coordinator fix wave, 2026-09-19: "an `action` verdict requires at
    least one evidence item (a copied claim). The action must point at
    the claim that motivates it." A `no_action`/`trigger_stale` card may
    still rely on findings alone (`test_a_trigger_stale_card_with_a_
    finding_passes` above); an `action` card may not."""
    account_slug = ensure_entity_note(db, loaded_registry, ASHGROVE_MEDICAL).slug
    finding_id = record_finding(
        db,
        run_id=RUN_ID,
        candidate_slug=CANDIDATE_SLUG,
        tool="absence_check",
        arguments={"account_slug": account_slug},
        result={"count": 0, "claim_ids": []},
    )
    card = _good_card(
        account=account_slug, owner=SAM_OKAFOR, evidence=[], verdict="action", findings=[finding_id]
    )
    errors = mechanical_checks(db, card, default_owner_crm_id=SAM_OKAFOR, as_of=date(2026, 9, 6))
    assert "action_needs_evidence" in errors


def test_attack_7_fake_far_future_due_with_no_renewal_evidence_is_dropped(
    loaded_registry: Registry, db
):
    """Should-fix 4: `due=2099-01-01`, `due_rule="renewal minus 30 days
    (renewal 2099-01-31)"` used to pass with nothing checking it. The due
    rule is now a closed set, mechanically re-derived: `renewal_minus_
    days:N` requires the exact renewal date (`due + N days`) to appear
    verbatim in the card's own evidence."""
    account_slug = ensure_entity_note(db, loaded_registry, ASHGROVE_MEDICAL).slug
    item = _merge(
        db,
        loaded_registry,
        claim="Ashgrove Medical Center received a late Tidewater T2 delivery.",
        quote="late delivery",
        locator="p8",
    )
    card = _good_card(
        account=account_slug,
        owner=SAM_OKAFOR,
        evidence=[item],
        due=date(2099, 1, 1),
        due_rule="renewal_minus_days:30",
    )
    errors = mechanical_checks(db, card, default_owner_crm_id=SAM_OKAFOR, as_of=date(2026, 9, 6))
    assert "renewal_date_not_in_evidence" in errors


def test_attack_7b_fake_confidence_backing_is_dropped_as_an_out_of_range_ref(
    loaded_registry: Registry, db
):
    """Should-fix 4's other half: a fabricated `confidence_reason` string
    ("Corroborated by four independent sources.") has no place to exist
    on the revised card at all -- `confidence_refs` is indexes, not
    prose. The only way to "fake" backing now is an out-of-range index,
    which is mechanically rejected."""
    account_slug = ensure_entity_note(db, loaded_registry, ASHGROVE_MEDICAL).slug
    item = _merge(
        db,
        loaded_registry,
        claim="Ashgrove Medical Center received a late Tidewater T2 delivery.",
        quote="late delivery",
        locator="p9",
    )
    card = _good_card(account=account_slug, owner=SAM_OKAFOR, evidence=[item])
    card.confidence_refs = RefSet(evidence_refs=[7], finding_refs=[])  # out of range
    errors = mechanical_checks(db, card, default_owner_crm_id=SAM_OKAFOR, as_of=date(2026, 9, 6))
    assert any(e.startswith("ref_out_of_range:confidence:") for e in errors)


def test_attack_8_superseded_claim_does_not_ground_a_card(loaded_registry: Registry, db):
    """Should-fix 1: a superseded claim still grounded a card. `_load_
    claim` requires `superseded_at IS NULL`; an evidence item citing a
    since-superseded claim id is `claim_not_found`."""
    account_slug = ensure_entity_note(db, loaded_registry, ASHGROVE_MEDICAL).slug
    item = _merge(
        db,
        loaded_registry,
        claim="Ashgrove Medical Center received a late Tidewater T2 delivery.",
        quote="late delivery",
        locator="p10",
    )
    with db.cursor() as cur:
        cur.execute("UPDATE claims SET superseded_at = now() WHERE id = %s", (item.claim_id,))
    db.commit()
    card = _good_card(account=account_slug, owner=SAM_OKAFOR, evidence=[item])
    errors = mechanical_checks(db, card, default_owner_crm_id=SAM_OKAFOR, as_of=date(2026, 9, 6))
    assert f"claim_not_found:{item.claim_id}" in errors


def test_attack_9_invented_claim_id_is_dropped(loaded_registry: Registry, db):
    """Verified-clean case, kept: a claim id that was never real."""
    account_slug = ensure_entity_note(db, loaded_registry, ASHGROVE_MEDICAL).slug
    fake_item = EvidenceItem(
        claim_id=999_999, text="invented", quote=None, source_slug="s-x", locator="p1"
    )
    card = _good_card(account=account_slug, owner=SAM_OKAFOR, evidence=[fake_item])
    errors = mechanical_checks(db, card, default_owner_crm_id=SAM_OKAFOR, as_of=date(2026, 9, 6))
    assert "claim_not_found:999999" in errors


def test_attack_10_customer_contact_owner_is_dropped(loaded_registry: Registry, db):
    """Verified-clean case, kept: a customer contact can never own an
    action."""
    account_slug = ensure_entity_note(db, loaded_registry, ASHGROVE_MEDICAL).slug
    item = _merge(
        db,
        loaded_registry,
        claim="Ashgrove Medical Center received a late Tidewater T2 delivery.",
        quote="late delivery",
        locator="p11",
    )
    card = _good_card(account=account_slug, owner=CUSTOMER_CONTACT, evidence=[item])
    errors = mechanical_checks(db, card, default_owner_crm_id=SAM_OKAFOR, as_of=date(2026, 9, 6))
    assert f"owner_not_vendor_side:{CUSTOMER_CONTACT}" in errors


# ---------------------------------------------------------------------
# A few more mechanical checks not among the ten, but load-bearing for
# the revised contract's own new machinery.
# ---------------------------------------------------------------------


def test_evidence_must_be_byte_verbatim(loaded_registry: Registry, db):
    account_slug = ensure_entity_note(db, loaded_registry, ASHGROVE_MEDICAL).slug
    item = _merge(
        db,
        loaded_registry,
        claim="Ashgrove Medical Center received a late Tidewater T2 delivery.",
        quote="late delivery",
        locator="p12",
    )
    tampered = item.model_copy(update={"text": item.text + " (retyped slightly)"})
    card = _good_card(account=account_slug, owner=SAM_OKAFOR, evidence=[tampered])
    errors = mechanical_checks(db, card, default_owner_crm_id=SAM_OKAFOR, as_of=date(2026, 9, 6))
    assert f"evidence_not_verbatim:{item.claim_id}" in errors


def test_a_finding_from_a_different_run_is_not_accepted(loaded_registry: Registry, db):
    account_slug = ensure_entity_note(db, loaded_registry, ASHGROVE_MEDICAL).slug
    item = _merge(
        db,
        loaded_registry,
        claim="Ashgrove Medical Center received a late Tidewater T2 delivery.",
        quote="late delivery",
        locator="p13",
    )
    other_run_finding_id = record_finding(
        db,
        run_id="a-different-run",
        candidate_slug=CANDIDATE_SLUG,
        tool="timeline",
        arguments={},
        result={"entries": []},
    )
    card = _good_card(
        account=account_slug, owner=SAM_OKAFOR, evidence=[item], findings=[other_run_finding_id]
    )
    errors = mechanical_checks(db, card, default_owner_crm_id=SAM_OKAFOR, as_of=date(2026, 9, 6))
    assert f"finding_not_found:{other_run_finding_id}" in errors


def test_a_judge_error_drops_the_card(loaded_registry: Registry, db):
    account_slug = ensure_entity_note(db, loaded_registry, ASHGROVE_MEDICAL).slug
    item = _merge(
        db,
        loaded_registry,
        claim="Ashgrove Medical Center received a late Tidewater T2 delivery.",
        quote="late delivery",
        locator="p14",
    )
    card = _good_card(account=account_slug, owner=SAM_OKAFOR, evidence=[item])
    judge = FakeSupportJudge(error_for={card.what})
    result = run_grounding_gate(
        db, card, judge=judge, as_of=date(2026, 9, 6), default_owner_crm_id=SAM_OKAFOR
    )
    assert not result.passed
    assert result.reason == "judge_error:what"


def test_a_different_owner_needs_justification(loaded_registry: Registry, db):
    account_slug = ensure_entity_note(db, loaded_registry, ASHGROVE_MEDICAL).slug
    item = _merge(
        db,
        loaded_registry,
        claim="Ashgrove Medical Center received a late Tidewater T2 delivery.",
        quote="late delivery",
        locator="p15",
    )
    card = _good_card(account=account_slug, owner=PRIYA, evidence=[item])
    errors = mechanical_checks(db, card, default_owner_crm_id=SAM_OKAFOR, as_of=date(2026, 9, 6))
    assert "owner_not_justified" in errors


def test_a_different_owner_named_in_evidence_is_accepted(loaded_registry: Registry, db):
    account_slug = ensure_entity_note(db, loaded_registry, ASHGROVE_MEDICAL).slug
    item = _merge(
        db,
        loaded_registry,
        claim="Priya Natarajan is now handling the Tidewater T2 delivery escalation.",
        quote="Priya Natarajan is now handling",
        locator="p16",
    )
    card = _good_card(account=account_slug, owner=PRIYA, evidence=[item])
    errors = mechanical_checks(db, card, default_owner_crm_id=SAM_OKAFOR, as_of=date(2026, 9, 6))
    assert "owner_not_justified" not in errors


def test_next_report_due_outside_the_window_is_dropped(loaded_registry: Registry, db):
    account_slug = ensure_entity_note(db, loaded_registry, ASHGROVE_MEDICAL).slug
    item = _merge(
        db,
        loaded_registry,
        claim="Ashgrove Medical Center received a late Tidewater T2 delivery.",
        quote="late delivery",
        locator="p17",
    )
    card = _good_card(
        account=account_slug,
        owner=SAM_OKAFOR,
        evidence=[item],
        due=date(2026, 12, 1),
        due_rule="next_report",
    )
    errors = mechanical_checks(db, card, default_owner_crm_id=SAM_OKAFOR, as_of=date(2026, 9, 6))
    assert "due_outside_next_report_window" in errors


# ---------------------------------------------------------------------
# S1 (review-action-researcher-2026-09-20.md, should-fix 1): `due` was
# effectively unbounded for `renewal_minus_days:N` (N itself had no
# range, and nothing tied `due` to anything real) and `commitment_date`
# (no upper bound at all -- only a past commitment date is meant to be
# allowed, "that is how an overdue item shows on /actions").
# ---------------------------------------------------------------------


def test_renewal_minus_days_with_a_negative_n_is_dropped(loaded_registry: Registry, db):
    """Crafted repro, verbatim from the review: `renewal_minus_days:-30`
    -- due one month AFTER the renewal it claims to count down to."""
    account_slug = ensure_entity_note(db, loaded_registry, ASHGROVE_MEDICAL).slug
    due = date(2026, 9, 20)
    renewal_date = due + timedelta(days=-30)  # 2026-08-21, "before" due
    item = _merge(
        db,
        loaded_registry,
        claim=f"Ashgrove Medical Center's Tidewater T2 renews {renewal_date.isoformat()}.",
        quote=f"renews {renewal_date.isoformat()}",
        locator="p-s1a",
    )
    card = _good_card(
        account=account_slug,
        owner=SAM_OKAFOR,
        evidence=[item],
        due=due,
        due_rule="renewal_minus_days:-30",
    )
    errors = mechanical_checks(db, card, default_owner_crm_id=SAM_OKAFOR, as_of=date(2026, 9, 6))
    assert "renewal_minus_days_out_of_range:-30" in errors


def test_renewal_minus_days_over_365_is_dropped(loaded_registry: Registry, db):
    """Crafted repro, verbatim from the review: `renewal_minus_days:910`
    -- a renewal-date match nearly three years out."""
    account_slug = ensure_entity_note(db, loaded_registry, ASHGROVE_MEDICAL).slug
    due = date(2025, 1, 1)
    renewal_date = due + timedelta(days=910)
    item = _merge(
        db,
        loaded_registry,
        claim=f"Ashgrove Medical Center's Tidewater T2 renews {renewal_date.isoformat()}.",
        quote=f"renews {renewal_date.isoformat()}",
        locator="p-s1b",
    )
    card = _good_card(
        account=account_slug,
        owner=SAM_OKAFOR,
        evidence=[item],
        due=due,
        due_rule="renewal_minus_days:910",
    )
    errors = mechanical_checks(db, card, default_owner_crm_id=SAM_OKAFOR, as_of=date(2026, 9, 6))
    assert "renewal_minus_days_out_of_range:910" in errors


def test_renewal_minus_days_due_far_before_as_of_is_dropped(loaded_registry: Registry, db):
    """`N` in range, the renewal date genuinely in evidence, but `due`
    itself sits long before `as_of` -- still no reason to believe this
    due date is real (nothing else ties it to a near-term check-in)."""
    account_slug = ensure_entity_note(db, loaded_registry, ASHGROVE_MEDICAL).slug
    as_of = date(2026, 9, 6)
    due = as_of - timedelta(days=200)
    renewal_date = due + timedelta(days=30)
    item = _merge(
        db,
        loaded_registry,
        claim=f"Ashgrove Medical Center's Tidewater T2 renews {renewal_date.isoformat()}.",
        quote=f"renews {renewal_date.isoformat()}",
        locator="p-s1c",
    )
    card = _good_card(
        account=account_slug,
        owner=SAM_OKAFOR,
        evidence=[item],
        due=due,
        due_rule="renewal_minus_days:30",
    )
    errors = mechanical_checks(db, card, default_owner_crm_id=SAM_OKAFOR, as_of=as_of)
    assert "due_outside_renewal_window" in errors


def test_commitment_date_far_in_the_future_is_dropped(loaded_registry: Registry, db):
    """Crafted repro, verbatim from the review: evidence naming an
    unrelated date ("The Ashgrove site visit happened on 2026-11-30 last
    year per the log.") let `due=2026-11-30` through with zero errors --
    nothing bounded how far into the future a `commitment_date` due
    could sit. Only the upper bound is new; a PAST commitment date must
    stay allowed (that is how an overdue item shows on `/actions`)."""
    account_slug = ensure_entity_note(db, loaded_registry, ASHGROVE_MEDICAL).slug
    as_of = date(2026, 9, 6)
    due = date(2026, 11, 30)
    item = _merge(
        db,
        loaded_registry,
        claim="The Ashgrove site visit happened on 2026-11-30 last year per the log.",
        quote="2026-11-30",
        locator="p-s1d",
    )
    card = _good_card(
        account=account_slug,
        owner=SAM_OKAFOR,
        evidence=[item],
        due=due,
        due_rule="commitment_date",
    )
    errors = mechanical_checks(db, card, default_owner_crm_id=SAM_OKAFOR, as_of=as_of)
    assert "commitment_date_too_far_in_future" in errors


def test_commitment_date_in_the_past_is_still_allowed(loaded_registry: Registry, db):
    """The other half of should-fix 1: a PAST commitment date must clear
    the gate cleanly -- that is exactly how an overdue commitment shows
    up as overdue on `/actions`."""
    account_slug = ensure_entity_note(db, loaded_registry, ASHGROVE_MEDICAL).slug
    as_of = date(2026, 9, 6)
    due = date(2026, 8, 28)
    item = _merge(
        db,
        loaded_registry,
        claim="Sam Okafor committed to a written root-cause summary due 2026-08-28.",
        quote="due 2026-08-28",
        locator="p-s1e",
    )
    card = _good_card(
        account=account_slug,
        owner=SAM_OKAFOR,
        evidence=[item],
        due=due,
        due_rule="commitment_date",
    )
    errors = mechanical_checks(db, card, default_owner_crm_id=SAM_OKAFOR, as_of=as_of)
    assert not any(e.startswith("commitment_date") for e in errors), errors


# ---------------------------------------------------------------------
# S3 (review-action-researcher-2026-09-20.md, should-fix 3): scope is an
# OR against account-or-theme (`_claim_in_scope`), so a
# card about account A whose only evidence is a THEME-member claim
# belonging to a DIFFERENT account B was mechanically clean, and
# `confirm_action_card` then cited B's own `source_slug`/`locator` on an
# action filed against A -- "every answer traces to the original file"
# traced to the wrong customer's file.
# ---------------------------------------------------------------------


def test_a_theme_member_claim_from_another_account_is_not_own_account_evidence(
    loaded_registry: Registry, db
):
    """Crafted repro, verbatim from the review: a card about Ashgrove
    Medical Center names a theme, and its only evidence is a Cobalt
    Ridge claim that is a member of that same theme -- in scope (an OR,
    by design), mechanically clean under the OLD rule, but Cobalt
    Ridge's own claim is not evidence ABOUT Ashgrove."""
    ashgrove_slug = ensure_entity_note(db, loaded_registry, ASHGROVE_MEDICAL).slug
    ensure_entity_note(db, loaded_registry, COBALT_RIDGE)

    now = clock_module.now()
    theme_note = NoteRow(
        slug="theme-late-deliveries-s3",
        type=NoteType.THEME,
        title="Late deliveries",
        description="Deliveries arriving late.",
        status=NoteStatus.LIVING,
        created=now,
        updated=now,
    )
    write_note(db, theme_note, [], changed_by="test", reason="fixture theme")

    cobalt_item = _merge(
        db,
        loaded_registry,
        claim="Cobalt Ridge Medical Center received a late Tidewater T2 delivery.",
        quote="late delivery",
        locator="p-s3",
        entities=["Cobalt Ridge Medical Center", "Tidewater T2"],
    )
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO theme_members (theme_slug, claim_id, added) VALUES (%s, %s, %s)",
            (theme_note.slug, cobalt_item.claim_id, date(2026, 8, 26)),
        )
    db.commit()

    card = _good_card(
        account=ashgrove_slug,
        owner=SAM_OKAFOR,
        evidence=[cobalt_item],
    )
    card.theme = theme_note.slug
    errors = mechanical_checks(db, card, default_owner_crm_id=SAM_OKAFOR, as_of=date(2026, 9, 6))
    assert f"claim_out_of_scope:{cobalt_item.claim_id}" not in errors, (
        "the claim should be IN SCOPE via the shared theme (an OR, by design) -- "
        "the point of this test is the separate no_own_account_evidence check"
    )
    assert "no_own_account_evidence" in errors


# ---------------------------------------------------------------------
# S4 (review-action-researcher-2026-09-20.md, should-fix 4): `what` had
# no minimum length -- crafted `what=""` and `what="   "` passed the
# full gate (the judge was handed an empty sentence), and `confirm_
# action_card` substituted "Follow up", producing an action with no real
# description that the report printed as confirmed.
# ---------------------------------------------------------------------


def test_what_empty_string_is_dropped(loaded_registry: Registry, db):
    account_slug = ensure_entity_note(db, loaded_registry, ASHGROVE_MEDICAL).slug
    item = _merge(
        db,
        loaded_registry,
        claim="Ashgrove Medical Center received a late Tidewater T2 delivery.",
        quote="late delivery",
        locator="p-s4a",
    )
    card = _good_card(account=account_slug, owner=SAM_OKAFOR, evidence=[item])
    card.what = ""
    errors = mechanical_checks(db, card, default_owner_crm_id=SAM_OKAFOR, as_of=date(2026, 9, 6))
    assert "what_too_short" in errors


def test_what_whitespace_only_is_dropped(loaded_registry: Registry, db):
    account_slug = ensure_entity_note(db, loaded_registry, ASHGROVE_MEDICAL).slug
    item = _merge(
        db,
        loaded_registry,
        claim="Ashgrove Medical Center received a late Tidewater T2 delivery.",
        quote="late delivery",
        locator="p-s4b",
    )
    card = _good_card(account=account_slug, owner=SAM_OKAFOR, evidence=[item])
    card.what = "                    "  # 20 chars, all whitespace
    errors = mechanical_checks(db, card, default_owner_crm_id=SAM_OKAFOR, as_of=date(2026, 9, 6))
    assert "what_too_short" in errors
