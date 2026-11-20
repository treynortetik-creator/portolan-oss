"""Tests for `portolan.ingest.filter.customer_filter` against the eight
named corpus sources (build instructions, `corpus/truth-table.yaml`
`expectations.filter` block), plus the junk rules and seeded sampling
mechanics. `FakeRegistry` (tests/ingest_helpers.py) stands in for
`portolan.registry.Registry`, seeded from the truth table.
"""

from __future__ import annotations

from datetime import date

import pytest
from ingest_helpers import build_fake_registry, load_call, load_email

from portolan.extract.schema import SourceType
from portolan.ingest.filter import FilterDecision, apply_filter_decision, customer_filter
from portolan.ingest.models import Participant, RawSource
from portolan.ingest.prepare import prepare_text

# A seed for which `_seeded_sample("S-email-061", 0.02, SAMPLING_SEED)` is
# True — found by brute-force search over small integers. Any seed with
# that property works; this one is pinned so the test is deterministic
# and doesn't re-search on every run.
SAMPLING_SEED = 84


@pytest.fixture(scope="module")
def registry():
    return build_fake_registry()


def _filtered(raw, registry, **kwargs) -> FilterDecision:
    prepared = prepare_text(raw)
    return customer_filter(raw, prepared, registry, **kwargs)


# --------------------------------------------------------------------
# The eight named sources
# --------------------------------------------------------------------


def test_out_of_office_email_is_junk_auto_reply(registry):
    raw = load_email("S-email-040")
    decision = _filtered(raw, registry)
    assert decision.include is False
    assert decision.reason == "junk:auto_reply"
    assert decision.sampled is False


def test_internal_ops_email_with_no_customer_or_product_is_excluded(registry):
    # F07 / S-email-061: mentions "delivery drivers" (an internal staffing
    # matter) but no customer or product. The customer filter is allowed
    # to miss this -- that's the entire point of sampling the excluded
    # stream (spec sec 5.2: "The filter is measured, not trusted").
    raw = load_email("S-email-061")
    decision = _filtered(raw, registry, seed="no-sample-please-0001")
    assert decision.include is False
    assert decision.reason == "excluded"
    assert decision.sampled is False


def test_internal_ops_email_gets_sampled_with_the_right_seed(registry):
    # Same source, seed chosen so the deterministic sample catches it —
    # this is how the truth table's `filter.sample_audit_should_find:
    # [F07]` gets exercised.
    raw = load_email("S-email-061")
    decision = _filtered(raw, registry, sample_rate=0.02, seed=SAMPLING_SEED)
    assert decision.include is False
    assert decision.reason == "excluded_sampled"
    assert decision.sampled is True


def test_internal_email_mentioning_product_is_included_by_registry_scan(registry):
    # F08 / S-email-062: internal-only correspondence, but it names
    # "Tidewater consumables" -- the registry scan should catch it even
    # though no customer participant is on the thread.
    raw = load_email("S-email-062")
    decision = _filtered(raw, registry)
    assert decision.include is True
    assert decision.reason == "registry_scan"
    assert "tidewater-t2" in decision.matched_entities


def test_personal_gmail_email_is_included_by_registry_scan_of_account_name(registry):
    # S-email-058: sent from Maria Chen's personal gmail address, which is
    # deliberately NOT in the registry (truth table `note: not_in_crm`).
    # participant_domain can't fire (gmail.com is unknown, and by_email
    # for this address returns None). The body names "Ashgrove Health"
    # (the account), not "Maria Chen" in full -- only "Maria" appears, in
    # the sign-off, and that alone isn't a registered alias. So this is
    # included by registry_scan matching the ACCOUNT name, not the
    # contact's name.
    raw = load_email("S-email-058")
    decision = _filtered(raw, registry)
    assert decision.include is True
    assert decision.reason == "registry_scan"
    assert "ashgrove" in decision.matched_entities
    assert "maria-chen" not in decision.matched_entities


def test_call_with_a_registered_contact_is_included_by_participant_domain(registry):
    # S-call-003: Ashgrove's Maria Chen (a registered contact) on the
    # call with Sam Okafor (internal).
    raw = load_call("S-call-003")
    decision = _filtered(raw, registry)
    assert decision.include is True
    assert decision.reason == "participant_domain"
    assert "maria-chen" in decision.matched_entities


def test_quoted_email_thread_is_filtered_on_its_stripped_text(registry):
    # S-email-021-thread: after quote-stripping, only a short reply to
    # St. Aldric's Owen Castellane remains. Owen is a registered contact,
    # so this is included -- exercising that the filter runs on
    # PreparedText, not raw text.
    raw = load_email("S-email-021-thread")
    decision = _filtered(raw, registry)
    assert decision.include is True
    assert decision.reason == "participant_domain"


# --------------------------------------------------------------------
# Junk rules beyond the OOO case above
# --------------------------------------------------------------------


def test_calendar_invite_subject_is_junk():
    raw = RawSource(
        source_id="S-email-invite",
        source_type=SourceType.EMAIL,
        source_date=date(2026, 8, 1),
        text="Please find attached the meeting details.",
        metadata={"subject": "Invitation: QBR sync @ Thu Aug 20"},
    )
    decision = _filtered(raw, build_fake_registry())
    assert decision.include is False
    assert decision.reason == "junk:calendar_invite"


def test_newsletter_subject_is_junk():
    raw = RawSource(
        source_id="S-email-newsletter",
        source_type=SourceType.EMAIL,
        source_date=date(2026, 8, 1),
        text="This week in healthcare operations...",
        metadata={"subject": "Healthcare Ops Weekly Newsletter"},
    )
    decision = _filtered(raw, build_fake_registry())
    assert decision.include is False
    assert decision.reason == "junk:newsletter"


def test_noreply_sender_is_junk():
    raw = RawSource(
        source_id="S-email-noreply",
        source_type=SourceType.EMAIL,
        source_date=date(2026, 8, 1),
        text="Your invoice is ready.",
        metadata={"subject": "Invoice ready", "from_email": "noreply@billing-system.example"},
    )
    decision = _filtered(raw, build_fake_registry())
    assert decision.include is False
    assert decision.reason == "junk:noreply"


def test_junk_rule_wins_even_over_a_registry_hit():
    # Junk rules run first: an auto-reply that happens to mention a
    # customer name is still junk, not registry_scan.
    raw = RawSource(
        source_id="S-email-ooo-with-name",
        source_type=SourceType.EMAIL,
        source_date=date(2026, 8, 1),
        text="I am out of the office. For anything regarding Ashgrove Health, contact my manager.",
        metadata={"subject": "Automatic reply: Following up"},
    )
    decision = _filtered(raw, build_fake_registry())
    assert decision.reason == "junk:auto_reply"


# --------------------------------------------------------------------
# account_hints
# --------------------------------------------------------------------


def test_account_hint_includes_even_with_no_other_signal(registry):
    raw = RawSource(
        source_id="S-doc-hinted",
        source_type=SourceType.DOC,
        source_date=date(2026, 8, 1),
        text="A generic document with no names or product mentions at all.",
        account_hints=["A-1010"],
    )
    decision = _filtered(raw, registry)
    assert decision.include is True
    assert decision.reason == "account_hint"
    assert decision.matched_entities == ["A-1010"]


# --------------------------------------------------------------------
# Sampling mechanics
# --------------------------------------------------------------------


def test_sampling_is_deterministic_given_a_fixed_seed(registry):
    raw = load_email("S-email-061")
    first = _filtered(raw, registry, seed=SAMPLING_SEED)
    second = _filtered(raw, registry, seed=SAMPLING_SEED)
    assert first.sampled == second.sampled == True  # noqa: E712 - clarity over brevity here
    assert first.reason == second.reason == "excluded_sampled"


def test_different_seeds_can_change_the_sampled_flag(registry):
    raw = load_email("S-email-061")
    sampled = _filtered(raw, registry, seed=SAMPLING_SEED)
    not_sampled = _filtered(raw, registry, seed="a-seed-that-does-not-hit")
    assert sampled.sampled is True
    assert not_sampled.sampled is False


def test_sample_rate_zero_never_samples(registry):
    raw = load_email("S-email-061")
    decision = _filtered(raw, registry, sample_rate=0.0, seed=SAMPLING_SEED)
    assert decision.sampled is False
    assert decision.reason == "excluded"


# --------------------------------------------------------------------
# All-internal carve-out for the signal-word rule
# --------------------------------------------------------------------


def test_signal_word_alone_does_not_include_an_all_internal_thread(registry):
    # Directly exercises the carve-out: "delivery" (a signal word) with
    # zero customer/product signal, all participants internal.
    raw = RawSource(
        source_id="S-email-internal-signal-word",
        source_type=SourceType.EMAIL,
        source_date=date(2026, 8, 1),
        text=(
            "Delivery routes were reshuffled internally this week; nothing customer-facing changed."
        ),
        participants=[
            Participant(name="Sam Okafor", email="s.okafor@halvardmedical.com"),
            Participant(name="Priya Natarajan", email="p.natarajan@halvardmedical.com"),
        ],
    )
    decision = _filtered(raw, registry, seed="no-sample-0002")
    assert decision.include is False
    assert decision.reason == "excluded"


def test_out_of_office_mid_body_with_a_normal_subject_is_not_junk(registry):
    # F9: the exact planted failure -- a customer contact's real message
    # happens to contain "out of the office" mid-body, several lines down,
    # with an ordinary subject line. Must not be misfiled as junk with no
    # audit path.
    raw = RawSource(
        source_id="S-email-real-ooo-phrase",
        source_type=SourceType.EMAIL,
        source_date=date(2026, 8, 1),
        text=(
            "Hi team,\n\n"
            "Quick update on the Springfield rollout -- everything's on track.\n\n"
            "I'll be out of the office next week, but can you ship the pumps Monday "
            "regardless? Ashgrove Health is counting on the delivery.\n"
        ),
        participants=[Participant(name="Maria Chen", email="mchen@ashgrovehealth.org")],
        metadata={"subject": "Springfield rollout update"},
    )
    decision = _filtered(raw, registry)
    assert decision.reason != "junk:auto_reply"
    assert decision.include is True  # Ashgrove Health + registered contact


def test_out_of_office_in_first_two_lines_is_still_junk():
    raw = RawSource(
        source_id="S-email-real-ooo-top",
        source_type=SourceType.EMAIL,
        source_date=date(2026, 8, 1),
        text="I'm out of the office until Monday.\n\nWill respond when I'm back.\n",
        metadata={"subject": "Re: quick question"},
    )
    decision = _filtered(raw, build_fake_registry())
    assert decision.reason == "junk:auto_reply"


def test_junk_exclusion_can_land_in_the_sample():
    # F9: junk exclusions used to skip sampling entirely. Brute-forced seed
    # for which this specific source_id lands in a 2% sample.
    raw = RawSource(
        source_id="S-email-noreply",
        source_type=SourceType.EMAIL,
        source_date=date(2026, 8, 1),
        text="Your invoice is ready.",
        metadata={"subject": "Invoice ready", "from_email": "noreply@billing-system.example"},
    )
    seed = next(
        s
        for s in range(200)
        if _filtered(raw, build_fake_registry(), sample_rate=0.02, seed=s).sampled
    )
    decision = _filtered(raw, build_fake_registry(), sample_rate=0.02, seed=seed)
    assert decision.reason == "junk:noreply"
    assert decision.sampled is True


def test_signal_word_includes_when_a_participant_is_unplaced(registry):
    # Same signal word, but one participant isn't a known internal
    # person (an unrecognised external address) -- the carve-out no
    # longer applies, so the signal word is decisive.
    raw = RawSource(
        source_id="S-email-external-signal-word",
        source_type=SourceType.EMAIL,
        source_date=date(2026, 8, 1),
        text="Following up on the late delivery we discussed last week.",
        participants=[
            Participant(name="Sam Okafor", email="s.okafor@halvardmedical.com"),
            Participant(name="Someone Unknown", email="someone@unregistered-domain.example"),
        ],
    )
    decision = _filtered(raw, registry)
    assert decision.include is True
    assert decision.reason == "signal_word"
    # SIGNAL_WORDS is checked in a fixed order and "delivery" precedes
    # "late" in that list, so it's the one reported even though "late"
    # also appears in the text.
    assert decision.matched_entities == ["delivery"]


# --------------------------------------------------------------------
# apply_filter_decision (F16): a focused sources-row writer, not a
# second place that records to coverage_ledger.
# --------------------------------------------------------------------


def _insert_bare_source(cur, slug: str) -> None:
    cur.execute(
        """
        INSERT INTO sources (slug, source_type, extraction_status, schema_version)
        VALUES (%s, 'email', 'pending', '0.1')
        """,
        (slug,),
    )


def test_apply_filter_decision_marks_excluded_source_filtered(db):
    with db.cursor() as cur:
        _insert_bare_source(cur, "S-email-apply-excluded")

    apply_filter_decision(
        db,
        "S-email-apply-excluded",
        FilterDecision(include=False, reason="excluded_sampled", sampled=True),
    )

    with db.cursor() as cur:
        cur.execute(
            "SELECT extraction_status, filter_reason FROM sources WHERE slug = %s",
            ("S-email-apply-excluded",),
        )
        row = cur.fetchone()
    assert row == ("filtered", "excluded_sampled")


def test_apply_filter_decision_marks_included_source_pending_with_no_reason(db):
    with db.cursor() as cur:
        _insert_bare_source(cur, "S-email-apply-included")

    apply_filter_decision(
        db,
        "S-email-apply-included",
        FilterDecision(include=True, reason="registry_scan", matched_entities=["ashgrove"]),
    )

    with db.cursor() as cur:
        cur.execute(
            "SELECT extraction_status, filter_reason FROM sources WHERE slug = %s",
            ("S-email-apply-included",),
        )
        row = cur.fetchone()
    assert row == ("pending", None)


def test_apply_filter_decision_does_not_touch_the_coverage_ledger(db):
    # F16: exactly one place in the codebase records a filter decision to
    # coverage_ledger -- portolan.pipeline, via portolan.ingest.ledger.
    # apply_filter_decision (the sources-row writer) must not also write
    # it, or a source run through both would be double-counted.
    with db.cursor() as cur:
        _insert_bare_source(cur, "S-email-apply-no-ledger")
        cur.execute("SELECT count(*) FROM coverage_ledger")
        (before,) = cur.fetchone()

    apply_filter_decision(
        db,
        "S-email-apply-no-ledger",
        FilterDecision(include=False, reason="junk:noreply"),
    )

    with db.cursor() as cur:
        cur.execute("SELECT count(*) FROM coverage_ledger")
        (after,) = cur.fetchone()
    assert after == before
