"""Tests for `portolan.report.candidates` (`01-spec/actions-and-owners-
2026-09-19.md`, Phase 2, "Candidates by code, not by the model").

Run with this module's own test database:
    createdb portolan_test_candidates
    DATABASE_URL=postgresql://localhost/portolan_test_candidates \
        uv run pytest tests/test_report_candidates.py -v
"""

from __future__ import annotations

import csv
from datetime import date, timedelta
from pathlib import Path

from portolan.extract.schema import FactType
from portolan.llm.fake import FakeLLMClient
from portolan.merge.actions import merge_action
from portolan.merge.facts import merge_fact
from portolan.notes import NoteRow, NoteStatus, NoteType, write_note
from portolan.registry import Registry
from portolan.report.candidates import (
    PRICING_COMPETITOR_ARR_FLOOR,
    RENEWAL_HORIZON_DAYS,
    _load_open_renewals,
    find_candidates,
)
from portolan.report.models import ActionRow, ActionsFeed, DiffFeed, QuietAccounts, ThemePriority
from portolan.search.embed import FakeEmbedder
from tests.merge_helpers import (
    ASHGROVE,
    ASHGROVE_MEDICAL,
    build_action,
    build_fact,
    loaded_registry_fixture,  # noqa: F401
    make_source_row,
)

SINCE = date(2026, 8, 24)
AS_OF = date(2026, 8, 30)


def _empty_diff(quiet_30: list[str] | None = None) -> DiffFeed:
    return DiffFeed(
        since=SINCE,
        as_of=AS_OF,
        quiet_accounts=QuietAccounts(as_of=AS_OF, quiet_30=quiet_30 or []),
    )


def _empty_actions() -> ActionsFeed:
    return ActionsFeed()


def test_theme_rising_candidates_pick_the_top_level_account_with_the_most_claims(
    loaded_registry: Registry, db
):
    embedder = FakeEmbedder()
    llm = FakeLLMClient([])
    from portolan.util import clock

    now = clock.now()
    theme_note = NoteRow(
        slug="theme-late-deliveries",
        type=NoteType.THEME,
        title="Late deliveries",
        description="Deliveries arriving late.",
        status=NoteStatus.LIVING,
        created=now,
        updated=now,
    )
    write_note(db, theme_note, [], changed_by="test", reason="fixture theme")

    source = make_source_row(db, "s-cand-theme-1", source_date=date(2026, 8, 26))
    fact = build_fact(
        "F-cand-1",
        FactType.COMPLAINT,
        "Ashgrove Medical Center received a late Tidewater T2 delivery.",
        ["Ashgrove Medical Center", "Tidewater T2"],
        locator="p1",
        quote="late delivery",
        speaker="Rob Tallis",
    )
    outcome = merge_fact(db, llm, embedder, loaded_registry, fact, source)
    assert outcome.status == "new"

    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO theme_members (theme_slug, claim_id, added) VALUES (%s, %s, %s)",
            (theme_note.slug, outcome.claim_id, date(2026, 8, 26)),
        )
    db.commit()

    with db.cursor() as cur:
        cur.execute(
            "SELECT note_slug FROM registry_entities WHERE crm_id = %s", (ASHGROVE_MEDICAL,)
        )
        account_slug = cur.fetchone()[0]

    priority = ThemePriority(
        theme_slug=theme_note.slug,
        theme_name="Late deliveries",
        movement="rising",
        account_slugs=[account_slug],
        account_count=1,
        arr=1_600_000.0,
    )

    candidates = find_candidates(
        db,
        since=SINCE,
        as_of=AS_OF,
        diff=_empty_diff(),
        priorities=[priority],
        questions=[],
        actions=_empty_actions(),
    )

    theme_candidates = [c for c in candidates if c.rule == "theme_rising"]
    assert len(theme_candidates) == 1
    assert theme_candidates[0].account_slug == account_slug
    assert theme_candidates[0].theme_slug == theme_note.slug
    assert outcome.claim_id in theme_candidates[0].claim_ids


def test_commitment_overdue_candidates_come_from_the_overdue_actions_feed(
    loaded_registry: Registry, db
):
    source = make_source_row(db, "s-cand-commit-1", source_date=date(2026, 8, 20))
    action = build_action(
        "Send Ashgrove Health a written delivery root-cause summary",
        owner_mention=None,
        due=date(2020, 1, 1),
        locator="p1",
    )
    outcome = merge_action(db, loaded_registry, action, source, entity_id=ASHGROVE)

    actions = ActionsFeed(
        overdue=[
            ActionRow(
                id=outcome.action_id,
                note_slug=outcome.note_slug,
                what=action.what,
                owner=None,
                owner_crm_id=None,
                due=action.due,
                status="open",
            )
        ]
    )

    candidates = find_candidates(
        db,
        since=SINCE,
        as_of=AS_OF,
        diff=_empty_diff(),
        priorities=[],
        questions=[],
        actions=actions,
    )

    commitment_candidates = [c for c in candidates if c.rule == "commitment_overdue"]
    assert len(commitment_candidates) == 1
    assert commitment_candidates[0].slug == f"commitment-overdue:{outcome.action_id}"


def test_pricing_or_competitor_candidates_respect_the_arr_floor(loaded_registry: Registry, db):
    embedder = FakeEmbedder()
    llm = FakeLLMClient([])
    source = make_source_row(db, "s-cand-pricing-1", source_date=date(2026, 8, 26))
    fact = build_fact(
        "F-cand-2",
        FactType.PRICING,
        "Ashgrove Health asked about a volume discount on Tidewater T2.",
        ["Ashgrove Health", "Tidewater T2"],
        locator="p2",
        quote="volume discount",
        speaker="Helen Ozoro",
    )
    outcome = merge_fact(db, llm, embedder, loaded_registry, fact, source)
    assert outcome.status == "new"

    with db.cursor() as cur:
        cur.execute("SELECT note_slug FROM registry_entities WHERE crm_id = %s", (ASHGROVE,))
        account_slug = cur.fetchone()[0]

    above_floor = find_candidates(
        db,
        since=SINCE,
        as_of=AS_OF,
        diff=_empty_diff(),
        priorities=[],
        questions=[],
        actions=_empty_actions(),
        arr_by_note_slug={account_slug: PRICING_COMPETITOR_ARR_FLOOR},
    )
    assert any(c.rule == "pricing_or_competitor" for c in above_floor)

    below_floor = find_candidates(
        db,
        since=SINCE,
        as_of=AS_OF,
        diff=_empty_diff(),
        priorities=[],
        questions=[],
        actions=_empty_actions(),
        arr_by_note_slug={account_slug: PRICING_COMPETITOR_ARR_FLOOR - 1},
    )
    assert not any(c.rule == "pricing_or_competitor" for c in below_floor)


def test_renewal_risk_candidates_need_both_a_horizon_renewal_and_a_trigger(
    loaded_registry: Registry, db, tmp_path
):
    from portolan.notes.persist import ensure_entity_note

    account_slug = ensure_entity_note(db, loaded_registry, ASHGROVE).slug

    root = tmp_path / "corpus_out"
    (root / "crm").mkdir(parents=True)
    opp_csv = root / "crm" / "opportunities.csv"
    with opp_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "opportunity_id",
                "account_crm_id",
                "name",
                "stage",
                "close_date",
                "amount",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "opportunity_id": "O-9001",
                "account_crm_id": ASHGROVE,
                "name": "Ashgrove renewal",
                "stage": "negotiation",
                "close_date": (AS_OF + __import__("datetime").timedelta(days=30)).isoformat(),
                "amount": "2000000",
            }
        )

    class _Connector:
        def __init__(self, root: Path) -> None:
            self.root = root

    # No quiet window, no complaint -- no candidate.
    no_trigger = find_candidates(
        db,
        since=SINCE,
        as_of=AS_OF,
        diff=_empty_diff(quiet_30=[]),
        priorities=[],
        questions=[],
        actions=_empty_actions(),
        connector=_Connector(root),
    )
    assert not any(c.rule == "renewal_risk" for c in no_trigger)

    # Quiet -- now a candidate.
    with_trigger = find_candidates(
        db,
        since=SINCE,
        as_of=AS_OF,
        diff=_empty_diff(quiet_30=[account_slug]),
        priorities=[],
        questions=[],
        actions=_empty_actions(),
        connector=_Connector(root),
    )
    renewal_candidates = [c for c in with_trigger if c.rule == "renewal_risk"]
    assert len(renewal_candidates) == 1
    assert renewal_candidates[0].account_slug == account_slug


def test_load_open_renewals_excludes_closed_stages_and_out_of_horizon_dates(tmp_path):
    opp_csv = tmp_path / "opportunities.csv"
    with opp_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["opportunity_id", "account_crm_id", "name", "stage", "close_date"]
        )
        writer.writeheader()
        writer.writerow(
            {
                "opportunity_id": "O-1",
                "account_crm_id": "A-1",
                "name": "in horizon",
                "stage": "negotiation",
                "close_date": "2026-09-15",
            }
        )
        writer.writerow(
            {
                "opportunity_id": "O-2",
                "account_crm_id": "A-2",
                "name": "closed lost",
                "stage": "closed_lost",
                "close_date": "2026-09-15",
            }
        )
        writer.writerow(
            {
                "opportunity_id": "O-3",
                "account_crm_id": "A-3",
                "name": "too far out",
                "stage": "negotiation",
                "close_date": "2030-01-01",
            }
        )

    renewals = _load_open_renewals(
        tmp_path, as_of=date(2026, 9, 1), horizon_days=RENEWAL_HORIZON_DAYS
    )

    assert set(renewals.keys()) == {"A-1"}


def test_load_open_renewals_reads_the_latest_dated_snapshot_not_the_first_file(tmp_path):
    """Should-fix 5 (`01-spec/reviews/review-action-researcher-2026-09-
    20.md`): confirmed on the corpus, 2026-09-19 -- `_load_open_renewals`
    used to read only the first, un-dated `crm/opportunities.csv`, while
    `researcher_tools._load_opportunity` already read every `opportunities
    *.csv` snapshot and took the row with the latest `export_date`.
    Crafted here as the same shape: a stale, un-dated-looking export
    (`export_date=2026-04-15`) naming a close date OUTSIDE the horizon,
    and a later snapshot (`export_date=2026-08-21`, the customer having
    moved the renewal) naming one INSIDE it -- the fix must see the
    account via the LATER file, not silently drop it because the first
    file it reads (alphabetically or otherwise) says otherwise."""
    account = "A-1010"
    as_of = date(2026, 8, 30)
    stale_close = (as_of + timedelta(days=RENEWAL_HORIZON_DAYS + 60)).isoformat()
    fresh_close = (as_of + timedelta(days=30)).isoformat()
    fieldnames = ["opportunity_id", "account_crm_id", "name", "stage", "close_date", "export_date"]

    with (tmp_path / "opportunities.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "opportunity_id": "O-1010",
                "account_crm_id": account,
                "name": "stale export",
                "stage": "negotiation",
                "close_date": stale_close,
                "export_date": "2026-04-15",
            }
        )
    with (tmp_path / "opportunities-2026-08-21.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "opportunity_id": "O-1010",
                "account_crm_id": account,
                "name": "fresh export, renewal moved",
                "stage": "negotiation",
                "close_date": fresh_close,
                "export_date": "2026-08-21",
            }
        )

    renewals = _load_open_renewals(tmp_path, as_of=as_of, horizon_days=RENEWAL_HORIZON_DAYS)

    assert account in renewals, (
        "the account was dropped -- _load_open_renewals is still reading the stale, "
        "out-of-horizon snapshot instead of the latest export_date"
    )
    close_date, _name = renewals[account]
    assert close_date.isoformat() == fresh_close


def test_find_candidates_is_deterministically_ordered(loaded_registry: Registry, db):
    first = find_candidates(
        db,
        since=SINCE,
        as_of=AS_OF,
        diff=_empty_diff(),
        priorities=[],
        questions=[],
        actions=_empty_actions(),
    )
    second = find_candidates(
        db,
        since=SINCE,
        as_of=AS_OF,
        diff=_empty_diff(),
        priorities=[],
        questions=[],
        actions=_empty_actions(),
    )
    assert [c.slug for c in first] == [c.slug for c in second]


def test_account_for_an_action_note_ignores_a_non_account_note_linking_to_it(db) -> None:
    """An overdue action's account is the account (or contact's parent)
    that links to it. A signal note linking to the same action, whose link
    row the planner may reach first, must not hide it."""
    from datetime import UTC, datetime

    from portolan.report.candidates import _account_slug_for_note

    now = datetime.now(UTC)

    def _note(slug: str, note_type: NoteType, synthesis: str, crm_id: str | None = None):
        write_note(
            db,
            NoteRow(
                slug=slug,
                type=note_type,
                title=slug,
                description="Test fixture.",
                status=NoteStatus.LIVING,
                created=now,
                updated=now,
                synthesis=synthesis,
                crm_id=crm_id,
            ),
            [],
            changed_by="test",
            reason="account-for-action fixture",
        )

    _note("action-900-send-the-summary", NoteType.ACTION, "Send the summary.")
    _note(
        "signal-complaint-cand-account",
        NoteType.SIGNAL,
        "Linked to [[action-900-send-the-summary]].",
    )
    _note(
        "cand-account",
        NoteType.ACCOUNT,
        "Open action [[action-900-send-the-summary]].",
        crm_id="TEST-CAND-1",
    )
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO registry_entities (kind, canonical_name, crm_id, note_slug) "
            "VALUES ('account', 'Cand Account', 'TEST-CAND-1', 'cand-account')"
        )
    db.commit()

    # Force the plan a large graph gets (nested loops from `links`, in
    # physical order), where the signal note's link row comes first.
    with db.cursor() as cur:
        cur.execute("SET enable_hashjoin = off")
        cur.execute("SET enable_mergejoin = off")
    assert _account_slug_for_note(db, "action-900-send-the-summary") == "cand-account"
