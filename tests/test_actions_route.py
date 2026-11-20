"""`/actions` (`01-spec/actions-and-owners-2026-09-19.md`, Phase 1, build
item 6): grouped by owner, overdue by the world clock, Done/Dropped.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import psycopg
import pytest
from fastapi.testclient import TestClient

from portolan.config import Settings
from portolan.extract.schema import FactType
from portolan.llm.fake import FakeLLMClient
from portolan.merge.actions import merge_action
from portolan.merge.facts import merge_fact
from portolan.notes.persist import ensure_entity_note
from portolan.registry import Registry
from portolan.registry.owners import confirm_owner_proposal
from portolan.report.researcher import CURRENT_BRIEF_VERSION, confirm_action_card
from portolan.search.embed import FakeEmbedder
from portolan.store import LocalStore
from portolan.web.app import create_app
from portolan.web.deps import get_conn, get_embedder_dep, get_store_dep
from tests.merge_helpers import (
    ASHGROVE,
    ASHGROVE_MEDICAL,
    SAM_OKAFOR,
    build_action,
    build_fact,
    loaded_registry_fixture,  # noqa: F401
    make_source_row,
)

# Pinned for the same reason as `tests/test_report_action_why.py`'s own
# CARD_AS_OF: a fixed `next_report` due date checked against the wall clock
# fails from 2026-10-01 on (bug hunt 2026-09-22).
CARD_AS_OF = date(2026, 9, 20)
ORG_KEY = "test-org-key-for-actions-route"
SESSION_SECRET = "test-session-secret-for-actions-route-not-a-real-secret"


def _settings(db_url: str, store_root: Path) -> Settings:
    return Settings(
        database_url=db_url,
        tenant="test",
        store_root=store_root,
        llm_provider="fake",
        embed_provider="fake",
        org_key=ORG_KEY,
        session_secret=SESSION_SECRET,
        cookie_secure=False,
    )


@pytest.fixture()
def actions_client(db: psycopg.Connection, db_url: str, tmp_path: Path):
    store_root = tmp_path / "store"
    store_root.mkdir()
    settings = _settings(db_url, store_root)
    app = create_app(settings)
    app.dependency_overrides[get_conn] = lambda: db
    app.dependency_overrides[get_embedder_dep] = lambda: FakeEmbedder()
    app.dependency_overrides[get_store_dep] = lambda: LocalStore(store_root)
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def auth_cookies(actions_client: TestClient):
    r = actions_client.post(
        "/login", data={"org_key": ORG_KEY, "editor": "Test Editor"}, follow_redirects=False
    )
    assert r.status_code == 303
    return r.cookies


def test_actions_page_shows_the_spyglass_empty_state_when_nothing_is_open(
    actions_client: TestClient, auth_cookies
) -> None:
    resp = actions_client.get("/actions", cookies=auth_cookies)
    assert resp.status_code == 200
    assert "No open actions" in resp.text


def test_actions_page_groups_by_owner_and_flags_overdue_by_the_world_clock(
    actions_client: TestClient, db: psycopg.Connection, auth_cookies, loaded_registry: Registry
) -> None:
    assert confirm_owner_proposal(db, account_crm_id=ASHGROVE, owner_crm_id=SAM_OKAFOR) is True
    source = make_source_row(db, "s-actions-page-1", source_date=date(2026, 8, 20))
    overdue_action = build_action(
        "Send Ashgrove Health a written delivery root-cause summary",
        owner_mention=None,
        due=date(2020, 1, 1),  # long in the past
        locator="p1",
    )
    outcome = merge_action(db, loaded_registry, overdue_action, source, entity_id=ASHGROVE)

    resp = actions_client.get("/actions", cookies=auth_cookies)

    assert resp.status_code == 200
    assert "Sam Okafor" in resp.text
    assert "Ashgrove Health" in resp.text
    assert "overdue" in resp.text
    assert f"/notes/{outcome.note_slug}" in resp.text


def test_marking_an_action_done_writes_an_edit_record_and_removes_it_from_open(
    actions_client: TestClient, db: psycopg.Connection, auth_cookies, loaded_registry: Registry
) -> None:
    source = make_source_row(db, "s-actions-page-2", source_date=date(2026, 8, 20))
    action = build_action(
        "Send Ashgrove Health a written delivery root-cause summary",
        owner_mention=None,
        due=date(2026, 8, 28),
        locator="p2",
    )
    outcome = merge_action(db, loaded_registry, action, source, entity_id=ASHGROVE)

    resp = actions_client.post(
        f"/actions/{outcome.action_id}/done", cookies=auth_cookies, follow_redirects=False
    )
    assert resp.status_code == 303

    with db.cursor() as cur:
        cur.execute("SELECT status FROM actions WHERE id = %s", (outcome.action_id,))
        assert cur.fetchone() == ("done",)
        cur.execute(
            "SELECT editor, before_text, after_text, reason FROM edit_records WHERE note_slug = %s",
            (outcome.note_slug,),
        )
        edit_row = cur.fetchone()
    assert edit_row == ("Test Editor", "open", "done", "marked done")

    page = actions_client.get("/actions", cookies=auth_cookies)
    assert outcome.note_slug not in page.text.split("Closed this week")[0]

    # It shows up in "closed this week" (the world-clock today falls in
    # the same ISO week as the edit_records timestamp just written).
    assert "Closed this week" in page.text
    assert "done" in page.text.lower()


def test_marking_an_action_dropped_a_second_time_is_a_no_op(
    actions_client: TestClient, db: psycopg.Connection, auth_cookies, loaded_registry: Registry
) -> None:
    source = make_source_row(db, "s-actions-page-3", source_date=date(2026, 8, 20))
    action = build_action(
        "Schedule a quarterly business review with Ashgrove Health",
        owner_mention=None,
        due=date(2026, 8, 28),
        locator="p3",
    )
    outcome = merge_action(db, loaded_registry, action, source, entity_id=ASHGROVE)

    first = actions_client.post(
        f"/actions/{outcome.action_id}/dropped", cookies=auth_cookies, follow_redirects=False
    )
    assert first.status_code == 303
    second = actions_client.post(
        f"/actions/{outcome.action_id}/dropped", cookies=auth_cookies, follow_redirects=False
    )
    assert second.status_code == 303

    with db.cursor() as cur:
        cur.execute("SELECT count(*) FROM edit_records WHERE note_slug = %s", (outcome.note_slug,))
        assert cur.fetchone()[0] == 1


def test_actions_page_shows_a_one_line_why_under_a_researched_action(
    actions_client: TestClient, db: psycopg.Connection, auth_cookies, loaded_registry: Registry
) -> None:
    """Follow-up build, 2026-09-20: an action from a confirmed researched
    card gets a one-line "why" under it -- the first inference sentence,
    or the first evidence claim's text when the card has none. An action
    from the extraction pipeline (no card) shows no such line."""
    embedder = FakeEmbedder()
    llm = FakeLLMClient([])
    source = make_source_row(db, "s-actions-why", source_date=date(2026, 8, 20))
    fact = build_fact(
        "F-actions-why",
        FactType.COMPLAINT,
        "Ashgrove Medical Center received a late Tidewater T2 delivery.",
        ["Ashgrove Medical Center", "Tidewater T2"],
        locator="p1",
        quote="late delivery",
        speaker="Rob Tallis",
    )
    outcome = merge_fact(db, llm, embedder, loaded_registry, fact, source)
    with db.cursor() as cur:
        cur.execute("SELECT claim_text FROM claims WHERE id = %s", (outcome.claim_id,))
        (claim_text,) = cur.fetchone()
    account_slug = ensure_entity_note(db, loaded_registry, ASHGROVE_MEDICAL).slug

    card = {
        "run_id": "test-run",
        "candidate_slug": "test-candidate",
        "brief_version": CURRENT_BRIEF_VERSION,
        "account": account_slug,
        "verdict": "action",
        "what": "Follow up on the late delivery.",
        "evidence": [
            {
                "claim_id": outcome.claim_id,
                "text": claim_text,
                "quote": "late delivery",
                "source_slug": source.slug,
                "locator": "p1",
            }
        ],
        "findings": [],
        "inference": [],
        "owner": SAM_OKAFOR,
        "due": "2026-10-01",
        "due_rule": "next_report",
        "theme": None,
        "confidence": "medium",
        "confidence_refs": {"evidence_refs": [0], "finding_refs": []},
    }
    action_id = confirm_action_card(db, card=card, editor="Test Editor", as_of=CARD_AS_OF)
    assert action_id is not None

    resp = actions_client.get("/actions", cookies=auth_cookies)

    assert resp.status_code == 200
    assert claim_text in resp.text


_DISTINCT_ACTIONS = [
    "Send the signed pricing addendum to procurement",
    "Book onsite training for the night-shift nursing staff",
    "Confirm the replacement pump serial numbers with biomed",
    "Schedule the quarterly business review with the CFO",
    "Share the carrier root-cause report with facilities",
    "Draft the renewal quote with the multi-year discount",
    "Escalate the firmware bug to the engineering on-call",
    "Collect the signed data processing agreement from legal",
    "Arrange a demo of the new dashboard for the charge nurses",
    "Update the shipping contacts after the loading dock move",
    "Return the loaner units that arrived damaged in transit",
]


def _count_queries(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    seen: list[str] = []
    original = psycopg.Cursor.execute

    def counting_execute(self, query, *args, **kwargs):
        seen.append(str(query))
        return original(self, query, *args, **kwargs)

    monkeypatch.setattr(psycopg.Cursor, "execute", counting_execute)
    return seen


def test_actions_page_query_count_does_not_grow_with_the_number_of_actions(
    actions_client: TestClient,
    db: psycopg.Connection,
    auth_cookies,
    loaded_registry: Registry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bug hunt 2026-09-22: `actions_feed` looked up each action's account
    with one or two queries per row (N+1), on `/actions` and on every
    account note page. At ten times the demo's actions that is hundreds of
    round trips per page view."""
    source = make_source_row(db, "s-actions-n-plus-one", source_date=date(2026, 8, 20))
    accounts = [ASHGROVE, ASHGROVE_MEDICAL, "A-1001", "A-1002", "A-1020", "A-1030"]

    def add(i: int) -> None:
        action = build_action(_DISTINCT_ACTIONS[i], due=date(2026, 10, 1 + i), locator=f"p{i}")
        merge_action(db, loaded_registry, action, source, entity_id=accounts[i % len(accounts)])

    add(0)
    seen = _count_queries(monkeypatch)
    assert actions_client.get("/actions", cookies=auth_cookies).status_code == 200
    with_one = len(seen)

    monkeypatch.undo()
    for i in range(1, len(_DISTINCT_ACTIONS)):
        add(i)
    with db.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM actions WHERE status = 'open'")
        assert cur.fetchone()[0] == len(_DISTINCT_ACTIONS)
    seen = _count_queries(monkeypatch)
    page = actions_client.get("/actions", cookies=auth_cookies)
    with_eleven = len(seen)

    assert page.status_code == 200
    assert with_eleven == with_one


def test_a_done_and_a_dropped_that_both_read_open_leave_one_transition(
    db: psycopg.Connection, loaded_registry: Registry
) -> None:
    """Bug hunt 2026-09-22: `_set_status` reads the status, then
    `record_action_status_change` updated it unconditionally. Two clicks
    (Done, then Dropped from a second tab) that both read `open` before
    either committed flipped the action twice and wrote two edit records;
    the second write must not apply once the first has closed it."""
    from portolan.web.queries import record_action_status_change

    source = make_source_row(db, "s-actions-race", source_date=date(2026, 8, 20))
    action = build_action("Send the loaner return label to receiving", locator="p9")
    outcome = merge_action(db, loaded_registry, action, source, entity_id=ASHGROVE)

    for to_status in ("done", "dropped"):
        record_action_status_change(
            db,
            action_id=outcome.action_id,
            note_slug=outcome.note_slug,
            from_status="open",
            to_status=to_status,
            editor="Test Editor",
            reason=f"marked {to_status}",
        )

    with db.cursor() as cur:
        cur.execute("SELECT status FROM actions WHERE id = %s", (outcome.action_id,))
        assert cur.fetchone() == ("done",)
        cur.execute(
            "SELECT after_text FROM edit_records WHERE note_slug = %s", (outcome.note_slug,)
        )
        assert cur.fetchall() == [("done",)]
