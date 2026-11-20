"""Tests for `portolan.report.researcher_tools` (`01-spec/actions-and-
owners-2026-09-19.md`, "Phase 2, revised contract", "Tools"): the four
finding tools (`absence_check`, `count_by`, `timeline`, `renewal`),
called directly against a migrated Postgres connection the same way
`tests/test_mcp_server_tools.py` calls `portolan.mcp_server.tools`
functions directly.
"""

from __future__ import annotations

from datetime import date

from portolan.extract.schema import FactType
from portolan.llm.fake import FakeLLMClient
from portolan.merge.facts import merge_fact
from portolan.notes.persist import ensure_entity_note
from portolan.registry import Registry
from portolan.report.accounts import account_arr_by_note_slug
from portolan.report.researcher_tools import (
    AccountLookups,
    absence_check,
    count_by,
    record_finding,
    renewal,
    timeline,
)
from portolan.search.embed import FakeEmbedder
from tests.merge_helpers import (
    ASHGROVE_MEDICAL,
    build_fact,
    loaded_registry_fixture,  # noqa: F401
    make_source_row,
)

RUN_ID = "tools-test-run"
CANDIDATE_SLUG = "test-candidate"


def _merge(
    db,
    registry,
    *,
    claim,
    quote,
    locator,
    fact_type=FactType.COMPLAINT,
    recorded=None,
    entities=None,
):
    embedder = FakeEmbedder()
    llm = FakeLLMClient(lambda **_kwargs: {"verdict": "extends", "reason": "distinct"})
    source = make_source_row(db, f"s-tools-{locator}", source_date=recorded or date(2026, 8, 20))
    fact = build_fact(
        f"F-tools-{locator}",
        fact_type,
        claim,
        entities or ["Ashgrove Medical Center", "Tidewater T2"],
        locator=locator,
        quote=quote,
        speaker="Rob Tallis",
        valid_from=recorded,
    )
    outcome = merge_fact(db, llm, embedder, registry, fact, source)
    assert outcome.status in ("new", "extends")
    return outcome.claim_id


def _lookups(db) -> AccountLookups:
    from portolan.report.researcher_tools import _account_parent_slugs

    return AccountLookups(
        arr_by_note_slug=account_arr_by_note_slug(db, {"A-1010": 2_350_000.0}),
        region_by_note_slug={},
        branch_by_note_slug={"ashgrove-medical-center": "dayton"},
        parent_of=_account_parent_slugs(db),
    )


def test_absence_check_returns_zero_when_nothing_matches(loaded_registry: Registry, db):
    account_slug = ensure_entity_note(db, loaded_registry, ASHGROVE_MEDICAL).slug
    finding_id, result = absence_check(
        db,
        run_id=RUN_ID,
        candidate_slug=CANDIDATE_SLUG,
        account_slug=account_slug,
        theme_slug=None,
        fact_types=["commitment"],
        after_date=date(2026, 1, 1),
    )
    assert result["count"] == 0
    assert result["latest_date"] is None
    assert result["claim_ids"] == []
    assert result["count_before"] == 0
    assert result["had_matches_before"] is False
    assert finding_id > 0


def test_absence_check_counts_matches_after_the_date(loaded_registry: Registry, db):
    account_slug = ensure_entity_note(db, loaded_registry, ASHGROVE_MEDICAL).slug
    claim_id = _merge(
        db,
        loaded_registry,
        claim="Sam Okafor promised a written root-cause summary.",
        quote="promised a written summary",
        locator="a1",
        fact_type=FactType.COMMITMENT,
        recorded=date(2026, 8, 28),
    )
    _finding_id, result = absence_check(
        db,
        run_id=RUN_ID,
        candidate_slug=CANDIDATE_SLUG,
        account_slug=account_slug,
        theme_slug=None,
        fact_types=["commitment"],
        after_date=date(2026, 8, 1),
    )
    assert result["count"] == 1
    assert result["claim_ids"] == [claim_id]
    assert result["latest_date"] == "2026-08-28"


def test_count_by_requires_a_population_filter(loaded_registry: Registry, db):
    try:
        count_by(
            db,
            run_id=RUN_ID,
            candidate_slug=CANDIDATE_SLUG,
            lookups=_lookups(db),
            group_by="account",
        )
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_count_by_groups_and_includes_arr_for_account(loaded_registry: Registry, db):
    ensure_entity_note(db, loaded_registry, ASHGROVE_MEDICAL)
    ensure_entity_note(db, loaded_registry, "A-1010")  # the parent system, for roll-up
    _merge(
        db,
        loaded_registry,
        claim="Ashgrove Medical Center received a late Tidewater T2 delivery.",
        quote="late delivery",
        locator="c1",
        recorded=date(2026, 8, 20),
    )
    finding_id, result = count_by(
        db,
        run_id=RUN_ID,
        candidate_slug=CANDIDATE_SLUG,
        lookups=_lookups(db),
        fact_type="complaint",
        group_by="account",
    )
    assert result["total"] == 1
    assert finding_id > 0
    groups = {g["group"]: g for g in result["groups"]}
    assert "ashgrove-health" in groups  # rolled up to the top-level system
    assert groups["ashgrove-health"]["arr"] == 2_350_000.0


def test_count_by_group_by_branch(loaded_registry: Registry, db):
    ensure_entity_note(db, loaded_registry, ASHGROVE_MEDICAL)
    _merge(
        db,
        loaded_registry,
        claim="Ashgrove Medical Center received a late Tidewater T2 delivery.",
        quote="late delivery",
        locator="c2",
        recorded=date(2026, 8, 20),
    )
    _finding_id, result = count_by(
        db,
        run_id=RUN_ID,
        candidate_slug=CANDIDATE_SLUG,
        lookups=_lookups(db),
        fact_type="complaint",
        group_by="branch",
    )
    groups = {g["group"]: g["count"] for g in result["groups"]}
    assert groups.get("dayton") == 1


def test_timeline_orders_claims_and_records_a_finding(loaded_registry: Registry, db):
    account_slug = ensure_entity_note(db, loaded_registry, ASHGROVE_MEDICAL).slug
    _merge(
        db,
        loaded_registry,
        claim="Ashgrove Medical Center received a late Tidewater T2 delivery.",
        quote="late delivery",
        locator="t1",
        recorded=date(2026, 8, 10),
    )
    _merge(
        db,
        loaded_registry,
        claim="Ashgrove Medical Center asked about pricing.",
        quote="asked about pricing",
        locator="t2",
        fact_type=FactType.PRICING,
        recorded=date(2026, 8, 20),
    )
    finding_id, result = timeline(
        db, run_id=RUN_ID, candidate_slug=CANDIDATE_SLUG, account_slug=account_slug, theme_slug=None
    )
    assert result["count"] == 2
    assert [e["date"] for e in result["entries"]] == ["2026-08-10", "2026-08-20"]
    assert finding_id > 0


def test_renewal_finds_a_claim_mentioning_the_renewal_date(loaded_registry: Registry, db):
    account_slug = ensure_entity_note(db, loaded_registry, "A-1010").slug  # Ashgrove Health
    _merge(
        db,
        loaded_registry,
        claim="Ashgrove Health asked to push the renewal to 2027-06-30.",
        quote="pushed the renewal to 2027-06-30",
        locator="r1",
        fact_type=FactType.DECISION,
        recorded=date(2026, 8, 20),
        entities=["Ashgrove Health"],
    )
    finding_id, result = renewal(
        db,
        run_id=RUN_ID,
        candidate_slug=CANDIDATE_SLUG,
        account_slug=account_slug,
        connector=None,
        lookups=_lookups(db),
    )
    assert finding_id > 0
    assert len(result["renewal_claims"]) == 1
    assert "2027-06-30" in result["renewal_claims"][0]["dates_mentioned"]


def test_record_finding_round_trips(loaded_registry: Registry, db):
    finding_id = record_finding(
        db,
        run_id=RUN_ID,
        candidate_slug=CANDIDATE_SLUG,
        tool="timeline",
        arguments={"account_slug": "ashgrove-medical-center"},
        result={"entries": [], "count": 0},
    )
    with db.cursor() as cur:
        cur.execute(
            "SELECT run_id, candidate_slug, tool FROM researcher_findings WHERE id = %s",
            (finding_id,),
        )
        row = cur.fetchone()
    assert row == (RUN_ID, CANDIDATE_SLUG, "timeline")
