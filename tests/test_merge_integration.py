"""End-to-end test for `portolan.merge.merge_extraction`: three sources,
run through the real registry, and the resulting graph checked against
`portolan.lint.report.run_lint` (`01-spec/build-contracts.md` "Merge" +
"so what you write passes the lint").

Run with this module's own test database:
    createdb portolan_test_merge
    DATABASE_URL=postgresql://localhost/portolan_test_merge \
        uv run pytest tests/test_merge_integration.py -v
"""

from __future__ import annotations

from datetime import date

import pytest

from portolan.extract.schema import (
    Action,
    Coverage,
    EntityKind,
    EntityMention,
    ExtractionResult,
    Fact,
    FactType,
)
from portolan.lint.report import run_lint
from portolan.llm.fake import FakeLLMClient
from portolan.merge.pipeline import merge_extraction
from portolan.registry import Registry
from portolan.search.embed import FakeEmbedder
from tests.merge_helpers import (
    ASHGROVE_MEDICAL,
    loaded_registry_fixture,  # noqa: F401
    make_source_row,
)
from tests.test_lint import FakeStore


def test_three_sources_merge_to_a_lint_clean_graph(loaded_registry: Registry, db):
    embedder = FakeEmbedder()
    # Exactly one candidate ever crosses the gate in this scenario (F02's
    # ticket vs. call retelling) -- the FakeLLMClient's list form raises
    # IndexError on any *extra* call, which is itself part of the
    # assertion: only one compare call happens across all three sources.
    llm = FakeLLMClient([{"verdict": "same", "reason": "same delivery, retold on the call"}])

    larkspur_memorial = EntityMention(
        mention="Larkspur Memorial Hospital",
        kind=EntityKind.ACCOUNT,
        registry_id="A-1003",
        confidence=0.95,
    )
    tidewater_t2 = EntityMention(
        mention="Tidewater T2", kind=EntityKind.PRODUCT, registry_id="tidewater-t2", confidence=0.9
    )

    # --- Source 1: the ticket half of F02 --------------------------------
    source1 = make_source_row(
        db, "s-ticket-031", source_type="ticket", source_date=date(2026, 8, 5)
    )
    result1 = ExtractionResult(
        source_id=source1.slug,
        summary="Larkspur Memorial opened a ticket about a late Tidewater T2 delivery.",
        entities=[larkspur_memorial, tidewater_t2],
        facts=[
            Fact(
                id="F02",
                type=FactType.COMPLAINT,
                claim=(
                    "Larkspur Memorial Hospital reported a Tidewater T2 consumable "
                    "delivery arriving four days late on 2026-08-04."
                ),
                entities=["Larkspur Memorial Hospital", "Tidewater T2"],
                locator="T-031#1",
                quote="4 days late",
                sentiment=-1,
                confidence=0.9,
            )
        ],
        coverage=Coverage(segments_read=1, segments_total=1),
    )
    report1 = merge_extraction(db, llm, embedder, loaded_registry, result1, source1)
    assert report1.compare_calls == 0

    # --- Source 2: the call retelling F02, plus praise, usage, and an
    #     action -----------------------------------------------------------
    source2 = make_source_row(db, "s-call-009", source_type="call", source_date=date(2026, 8, 5))
    result2 = ExtractionResult(
        source_id=source2.slug,
        summary="Follow-up call with Larkspur Memorial's biomed lead.",
        entities=[larkspur_memorial, tidewater_t2],
        facts=[
            Fact(
                id="F02b",
                type=FactType.COMPLAINT,
                claim=(
                    "Larkspur Memorial Hospital had that late order last week for "
                    "the Tidewater T2 consumables."
                ),
                entities=["Larkspur Memorial Hospital", "Tidewater T2"],
                locator="00:04:12",
                quote="that late order last week",
                sentiment=-1,
                confidence=0.85,
            ),
            Fact(
                id="F10",
                type=FactType.PRAISE,
                claim=(
                    "Larkspur Memorial Hospital's biomed lead said on 2026-08-05 that "
                    "Halvard's field engineer response was \"the fastest we've had "
                    'from any vendor".'
                ),
                entities=["Larkspur Memorial Hospital", "Tidewater T2"],
                locator="00:14:00",
                quote="the fastest we've had from any vendor",
                sentiment=2,
                confidence=0.9,
            ),
            Fact(
                id="F26",
                type=FactType.USAGE,
                claim=(
                    "Larkspur Memorial Hospital has forty Tidewater T2 pumps and is "
                    "trialling four Tidewater T3 units as of 2026-08-05."
                ),
                entities=["Larkspur Memorial Hospital", "Tidewater T2", "Tidewater T3"],
                locator="00:20:00",
                sentiment=0,
                confidence=0.9,
            ),
        ],
        actions=[
            Action(
                what="Send Larkspur Memorial a written delivery root-cause summary",
                owner_mention="Sam Okafor",
                due=date(2026, 8, 28),
                locator="00:31:10",
            )
        ],
        coverage=Coverage(segments_read=1, segments_total=1),
    )
    report2 = merge_extraction(db, llm, embedder, loaded_registry, result2, source2)
    assert report2.compare_calls == 1
    assert report2.outcome_counts.get("same") == 1
    assert report2.actions_new == 1

    # --- Source 3: a CRM export fact with no signal wrapper (`other`,
    #     routed straight to the account note) ---------------------------
    source3 = make_source_row(db, "s-crm-a1002", source_type="crm", source_date=date(2026, 3, 2))
    result3 = ExtractionResult(
        source_id=source3.slug,
        summary="CRM hierarchy export.",
        entities=[
            EntityMention(
                mention="Larkspur Regional Health",
                kind=EntityKind.ACCOUNT,
                registry_id="A-1002",
                confidence=0.95,
            )
        ],
        facts=[
            Fact(
                id="F36",
                type=FactType.OTHER,
                claim="Larkspur Regional Health is owned by Pinecrest Health Partners.",
                entities=["Larkspur Regional Health", "Pinecrest Health Partners"],
                locator="accounts:row 2",
                sentiment=0,
                confidence=1.0,
            )
        ],
        coverage=Coverage(segments_read=1, segments_total=1),
    )
    report3 = merge_extraction(db, llm, embedder, loaded_registry, result3, source3)
    assert report3.compare_calls == 0

    total_compare_calls = report1.compare_calls + report2.compare_calls + report3.compare_calls
    assert total_compare_calls == 1

    store = FakeStore({"originals/s-ticket-031", "originals/s-call-009", "originals/s-crm-a1002"})
    lint_report = run_lint(db, store)
    assert lint_report.errors() == [], "\n".join(f.message for f in lint_report.errors())


def test_one_note_version_per_source_not_per_claim(loaded_registry: Registry, db):
    """`build-contracts.md` Merge section: "write_note for every touched
    note (one version per source, not per claim)". Three distinct usage
    facts about the same account in one call must land on the same signal
    note (`signal-usage-...`) and produce exactly one new `note_versions`
    row for it, not three."""
    embedder = FakeEmbedder()
    # Whatever verdict comes back doesn't matter for this test (all three
    # outcomes land a claim on the same note either way) -- a callable
    # response tolerates however many compare calls the gate happens to
    # trigger between these three worded-differently usage facts.
    llm = FakeLLMClient(lambda **_kwargs: {"verdict": "extends", "reason": "more usage detail"})

    ashgrove_medical = EntityMention(
        mention="Ashgrove Medical Center",
        kind=EntityKind.ACCOUNT,
        registry_id="A-1011",
        confidence=0.95,
    )
    source = make_source_row(db, "s-call-020", source_type="call", source_date=date(2026, 8, 10))
    result = ExtractionResult(
        source_id=source.slug,
        summary="Ashgrove Medical Center usage review call.",
        entities=[ashgrove_medical],
        facts=[
            Fact(
                id="U1",
                type=FactType.USAGE,
                claim="Ashgrove Medical Center runs eleven Tidewater T3 pumps across two ICUs.",
                entities=["Ashgrove Medical Center", "Tidewater T3"],
                locator="00:01:00",
                sentiment=0,
                confidence=0.9,
            ),
            Fact(
                id="U2",
                type=FactType.USAGE,
                claim=(
                    "Ashgrove Medical Center's outpatient pharmacy stocks Tidewater T2 "
                    "consumables for weekly infusion clinics."
                ),
                entities=["Ashgrove Medical Center", "Tidewater T2"],
                locator="00:05:00",
                sentiment=0,
                confidence=0.9,
            ),
            Fact(
                id="U3",
                type=FactType.USAGE,
                claim=(
                    "Ashgrove Medical Center trains new nursing staff on the Tidewater T3 "
                    "pump twice per quarter."
                ),
                entities=["Ashgrove Medical Center", "Tidewater T3"],
                locator="00:09:00",
                sentiment=0,
                confidence=0.9,
            ),
        ],
        coverage=Coverage(segments_read=1, segments_total=1),
    )
    report = merge_extraction(db, llm, embedder, loaded_registry, result, source)

    usage_note_slugs = {p.note_slug for p in report.placements}
    assert usage_note_slugs == {"signal-usage-ashgrove-medical-center"}
    assert report.claims_inserted == 3

    from portolan.notes import read_note

    note, claims = read_note(db, "signal-usage-ashgrove-medical-center")
    assert len(claims) == 3
    # Created once (version 1, empty), then written exactly once more for
    # this entire source's three facts (version 2) -- not version 4.
    assert note.version == 2

    with db.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM note_versions WHERE slug = %s",
            ("signal-usage-ashgrove-medical-center",),
        )
        version_rows = cur.fetchone()[0]
    assert version_rows == 1


# --------------------------------------------------------------------
# Item 5 (e2e-run follow-up wave): orphan entity notes. `portolan_e2e`
# had 3 entity notes (`corvane-systems`, `northlake-general-hospital`,
# `tidewater`) with zero inbound `[[link]]`s -- each was the ONLY entity
# `select_home` resolved for its fact (F18's documented "entity" fallback
# for a product/competitor with no account, or the "other"/`contact_
# change` "account" fallback with nothing more specific resolved), so it
# became its OWN home note, and a claim never wikilinks its own home note
# (`merge_fact`'s mention loop) -- with nothing else in either fact to
# link from, the note got no inbound edge at all. `merge_fact` now wraps
# both fallback homes in a signal note (`portolan.merge.facts.
# _ensure_signal_note`'s docstring), the same way an ordinary
# account-anchored fact already is, so the entity's own mention -- no
# longer equal to the (now different) home note -- gets wikilinked in,
# and `_ensure_signal_note`'s own backlink covers the signal note itself.
# --------------------------------------------------------------------


def test_f18_style_competitor_only_fact_leaves_no_orphan(loaded_registry: Registry, db):
    """F18 itself, verbatim from `corpus/truth-table.yaml`: a competitor's
    own recall, no customer/account named at all."""
    from tests.test_lint import FakeStore

    embedder = FakeEmbedder()
    llm = FakeLLMClient([])

    source = make_source_row(db, "s-rss-004", source_type="rss", source_date=date(2026, 8, 26))
    result = ExtractionResult(
        source_id=source.slug,
        summary="Industry news: Corvane Systems recall.",
        entities=[
            EntityMention(
                mention="Corvane Systems",
                kind=EntityKind.COMPETITOR,
                registry_id="corvane",
                confidence=0.9,
            )
        ],
        facts=[
            Fact(
                id="F18",
                type=FactType.COMPETITOR_MENTION,
                claim=(
                    "Corvane Systems issued a voluntary recall of its C-40 pump line "
                    "on 2026-08-26, per an industry news feed."
                ),
                entities=["Corvane Systems"],
                locator="00:00:01",
                quote="Corvane Systems recalls C-40 infusion pump line",
                sentiment=0,
                confidence=0.9,
            )
        ],
        coverage=Coverage(segments_read=1, segments_total=1),
    )
    report = merge_extraction(db, llm, embedder, loaded_registry, result, source)

    # The claim no longer homes directly on the competitor's own note --
    # it homes on a signal note wrapping it, exactly like an
    # account-anchored fact does.
    assert report.placements[0].note_slug == "signal-competitor_mention-corvane-systems"

    from portolan.notes import read_note

    _signal_note, signal_claims = read_note(db, "signal-competitor_mention-corvane-systems")
    assert "[[corvane-systems]]" in signal_claims[0].claim_text

    competitor_note, _claims = read_note(db, "corvane-systems")
    assert "signal-competitor_mention-corvane-systems" in competitor_note.related

    store = FakeStore({"originals/s-rss-004"})
    lint_report = run_lint(db, store)
    orphan_slugs = {f.note_slug for f in lint_report.for_check("orphan")}
    assert "corvane-systems" not in orphan_slugs
    assert "signal-competitor_mention-corvane-systems" not in orphan_slugs
    assert lint_report.errors() == [], "\n".join(f.message for f in lint_report.errors())


def test_other_type_bare_account_fact_leaves_no_orphan(loaded_registry: Registry, db):
    """An `other`-type CRM fact naming only an account, no product/
    competitor/contact -- the second self-referential fallback
    (`portolan.merge.resolve.select_home`'s `other` bucket)."""
    from tests.test_lint import FakeStore

    embedder = FakeEmbedder()
    llm = FakeLLMClient([])

    source = make_source_row(db, "s-crm-a1001", source_type="crm", source_date=date(2026, 3, 2))
    result = ExtractionResult(
        source_id=source.slug,
        summary="CRM top-level parent export.",
        entities=[
            EntityMention(
                mention="Pinecrest Health Partners",
                kind=EntityKind.ACCOUNT,
                registry_id="A-1001",
                confidence=0.95,
            )
        ],
        facts=[
            Fact(
                id="F-pinecrest",
                type=FactType.OTHER,
                claim="Pinecrest Health Partners is a capital partner with no direct ARR.",
                entities=["Pinecrest Health Partners"],
                locator="accounts:row 1",
                sentiment=0,
                confidence=1.0,
            )
        ],
        coverage=Coverage(segments_read=1, segments_total=1),
    )
    report = merge_extraction(db, llm, embedder, loaded_registry, result, source)

    assert report.placements[0].note_slug == "signal-other-pinecrest-health-partners"

    from portolan.notes import read_note

    account_note, _claims = read_note(db, "pinecrest-health-partners")
    assert "signal-other-pinecrest-health-partners" in account_note.related

    store = FakeStore({"originals/s-crm-a1001"})
    lint_report = run_lint(db, store)
    orphan_slugs = {f.note_slug for f in lint_report.for_check("orphan")}
    assert "pinecrest-health-partners" not in orphan_slugs
    assert "signal-other-pinecrest-health-partners" not in orphan_slugs
    assert lint_report.errors() == [], "\n".join(f.message for f in lint_report.errors())


# --------------------------------------------------------------------
# Bug hunt 2026-09-18, finding 1 (P0): "A mid-source merge failure commits
# claims that no note ever renders". Two facts on the same signal note;
# fact 1 gates `new` (no LLM call, would have committed immediately
# pre-fix); fact 2 gates onto a candidate and the compare call raises
# (an LLM stub standing in for a Bedrock throttle/timeout). Before the
# fix: fact 1's claim, note, and links were already committed and
# permanent -- a cited claim nothing renders, links to, or records, and
# `_retry_stage`'s three retries never clean it up (each one re-runs from
# fact 1). After the fix: the whole source rolls back to nothing, the
# real exception reaches the caller so `_retry_stage`/dead-letter see it,
# and the source is retryable from a clean, zero-footprint state.
# --------------------------------------------------------------------


def test_mid_source_merge_failure_leaves_zero_footprint_and_is_retryable(
    loaded_registry: Registry, db
):
    def _raise_on_compare(**_kwargs):
        raise RuntimeError("ThrottlingException: rate exceeded")

    embedder = FakeEmbedder()
    source = make_source_row(db, "s-call-bug1", source_type="call", source_date=date(2026, 8, 20))
    ashgrove_medical = EntityMention(
        mention="Ashgrove Medical Center",
        kind=EntityKind.ACCOUNT,
        registry_id=ASHGROVE_MEDICAL,
        confidence=0.95,
    )
    fact1 = Fact(
        id="F-BUG1A",
        type=FactType.COMPLAINT,
        claim=(
            "Ashgrove Medical Center reported three late deliveries of the "
            "X-series pump in August 2026."
        ),
        entities=["Ashgrove Medical Center"],
        locator="00:01:00",
        quote="three late deliveries",
        sentiment=-1,
        confidence=0.9,
    )
    fact2 = Fact(
        id="F-BUG1B",
        type=FactType.COMPLAINT,
        claim=(
            "Ashgrove Medical Center had three late deliveries of the "
            "X-series pump again in August 2026."
        ),
        entities=["Ashgrove Medical Center"],
        locator="00:05:00",
        quote="three late deliveries again",
        sentiment=-1,
        confidence=0.9,
    )
    result = ExtractionResult(
        source_id=source.slug,
        summary="Ashgrove Medical Center complaint call.",
        entities=[ashgrove_medical],
        facts=[fact1, fact2],
        coverage=Coverage(segments_read=1, segments_total=1),
    )

    with pytest.raises(RuntimeError, match="ThrottlingException"):
        merge_extraction(
            db, FakeLLMClient(_raise_on_compare), embedder, loaded_registry, result, source
        )

    # Zero footprint: nothing from fact 1 (which would have gated `new`
    # and committed immediately, pre-fix) survives the rollback either.
    with db.cursor() as cur:
        cur.execute("SELECT count(*) FROM claims")
        assert cur.fetchone()[0] == 0
        cur.execute("SELECT count(*) FROM notes")
        assert cur.fetchone()[0] == 0
        cur.execute("SELECT count(*) FROM links")
        assert cur.fetchone()[0] == 0
        cur.execute("SELECT count(*) FROM changelog")
        assert cur.fetchone()[0] == 0
        cur.execute("SELECT summary FROM sources WHERE slug = %s", (source.slug,))
        assert cur.fetchone()[0] is None

    # Retryable: the connection is left in a clean, usable state, and a
    # second attempt (the LLM no longer throttling, matching
    # `_retry_stage`'s real re-run-from-fact-1 behaviour) merges cleanly.
    llm_retry = FakeLLMClient(lambda **_kwargs: {"verdict": "same", "reason": "same complaint"})
    report = merge_extraction(db, llm_retry, embedder, loaded_registry, result, source)
    assert report.claims_inserted == 1
    assert report.citations_inserted == 1

    from portolan.notes import read_note

    _note, claims = read_note(db, "signal-complaint-ashgrove-medical-center")
    assert len(claims) == 1
    assert len(claims[0].citations) == 1
