"""`/review/{id}/resolve` and `/dismiss` for `kind='action_card'` rows,
revised contract (`01-spec/actions-and-owners-2026-09-19.md`, "Phase 2,
revised contract", "Review page", "Keep from the first build: ... the
confirm-route state machine").
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import psycopg
import pytest
from fastapi.testclient import TestClient

from portolan.config import Settings
from portolan.extract.schema import FactType
from portolan.llm.fake import FakeLLMClient
from portolan.merge.facts import merge_fact
from portolan.notes.persist import ensure_entity_note
from portolan.registry import Registry
from portolan.search.embed import FakeEmbedder
from portolan.store import LocalStore
from portolan.util import clock
from portolan.web.app import create_app
from portolan.web.deps import get_conn, get_embedder_dep, get_llm_dep, get_store_dep
from tests.merge_helpers import (
    ASHGROVE_MEDICAL,
    SAM_OKAFOR,
    build_fact,
    loaded_registry_fixture,  # noqa: F401
    make_source_row,
)

ORG_KEY = "test-org-key-for-action-card-review"
SESSION_SECRET = "test-session-secret-for-action-card-review-not-a-real-secret"
CUSTOMER_CONTACT = "C-2001"  # Maria Chen -- not vendor-side; used to force a failed confirm.


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
def review_client(db: psycopg.Connection, db_url: str, tmp_path: Path):
    store_root = tmp_path / "store"
    store_root.mkdir()
    settings = _settings(db_url, store_root)
    app = create_app(settings)
    app.dependency_overrides[get_conn] = lambda: db
    app.dependency_overrides[get_embedder_dep] = lambda: FakeEmbedder()
    app.dependency_overrides[get_llm_dep] = lambda: FakeLLMClient([])
    app.dependency_overrides[get_store_dep] = lambda: LocalStore(store_root)
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def auth_cookies(review_client: TestClient):
    r = review_client.post(
        "/login", data={"org_key": ORG_KEY, "editor": "Test Editor"}, follow_redirects=False
    )
    assert r.status_code == 303
    return r.cookies


def _queue_action_card(
    db,
    *,
    account_slug: str,
    claim_id: int,
    claim_text: str,
    quote: str | None,
    source_slug: str,
    owner: str = SAM_OKAFOR,
    week: str = "2026-W36",
    due: str | None = None,
) -> int:
    # A week out by default: a card whose `next_report` due date is still
    # ahead of today, the state a freshly queued card is in. A fixed date
    # here goes stale on the calendar (bug hunt 2026-09-22).
    if due is None:
        due = (clock.today() + timedelta(days=7)).isoformat()
    card = {
        "run_id": "test-run",
        "candidate_slug": "test-candidate",
        "brief_version": "researcher-brief-v4",
        "account": account_slug,
        "verdict": "action",
        "what": "Follow up on the late delivery.",
        "evidence": [
            {
                "claim_id": claim_id,
                "text": claim_text,
                "quote": quote,
                "source_slug": source_slug,
                "locator": "p1",
            }
        ],
        "findings": [],
        "inference": [],
        "owner": owner,
        "due": due,
        "due_rule": "next_report",
        "theme": None,
        "confidence": "medium",
        "confidence_refs": {"evidence_refs": [0], "finding_refs": []},
        "trail": {},
    }
    payload = {"card": card, "week": week}
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO review_queue (kind, payload) VALUES ('action_card', %s::jsonb) "
            "RETURNING id",
            (json.dumps(payload),),
        )
        row_id = cur.fetchone()[0]
    db.commit()
    return row_id


def _merge_complaint(db, registry: Registry) -> tuple[int, str, str, str | None]:
    embedder = FakeEmbedder()
    llm = FakeLLMClient([])
    source = make_source_row(db, "s-review-action-card", source_date=date(2026, 8, 20))
    fact = build_fact(
        "F-review-action-card",
        FactType.COMPLAINT,
        "Ashgrove Medical Center received a late Tidewater T2 delivery.",
        ["Ashgrove Medical Center", "Tidewater T2"],
        locator="p1",
        quote="late delivery",
        speaker="Rob Tallis",
    )
    outcome = merge_fact(db, llm, embedder, registry, fact, source)
    assert outcome.status == "new"
    with db.cursor() as cur:
        cur.execute("SELECT claim_text, quote FROM claims WHERE id = %s", (outcome.claim_id,))
        claim_text, quote = cur.fetchone()
    return outcome.claim_id, source.slug, claim_text, quote


def test_review_page_renders_an_action_card(
    review_client: TestClient, db: psycopg.Connection, auth_cookies, loaded_registry: Registry
):
    account_slug = ensure_entity_note(db, loaded_registry, ASHGROVE_MEDICAL).slug
    claim_id, source_slug, claim_text, quote = _merge_complaint(db, loaded_registry)
    _queue_action_card(
        db,
        account_slug=account_slug,
        claim_id=claim_id,
        claim_text=claim_text,
        quote=quote,
        source_slug=source_slug,
    )

    resp = review_client.get("/review", cookies=auth_cookies)

    assert resp.status_code == 200
    assert "Follow up on the late delivery." in resp.text
    assert "Proposed action" in resp.text


def test_confirming_an_action_card_creates_an_open_action(
    review_client: TestClient, db: psycopg.Connection, auth_cookies, loaded_registry: Registry
):
    account_slug = ensure_entity_note(db, loaded_registry, ASHGROVE_MEDICAL).slug
    claim_id, source_slug, claim_text, quote = _merge_complaint(db, loaded_registry)
    row_id = _queue_action_card(
        db,
        account_slug=account_slug,
        claim_id=claim_id,
        claim_text=claim_text,
        quote=quote,
        source_slug=source_slug,
    )

    resp = review_client.post(
        f"/review/{row_id}/resolve", cookies=auth_cookies, follow_redirects=False
    )

    assert resp.status_code == 303
    assert "action_confirmed" in resp.headers["location"]

    with db.cursor() as cur:
        cur.execute("SELECT status FROM review_queue WHERE id = %s", (row_id,))
        assert cur.fetchone() == ("resolved",)
        cur.execute("SELECT what, status, due FROM actions ORDER BY id DESC LIMIT 1")
        what, status, due = cur.fetchone()
    assert what == "Follow up on the late delivery."
    assert status == "open"
    assert due == clock.today() + timedelta(days=7)


def test_confirming_with_an_edited_owner_and_due_uses_the_override(
    review_client: TestClient, db: psycopg.Connection, auth_cookies, loaded_registry: Registry
):
    account_slug = ensure_entity_note(db, loaded_registry, ASHGROVE_MEDICAL).slug
    claim_id, source_slug, claim_text, quote = _merge_complaint(db, loaded_registry)
    row_id = _queue_action_card(
        db,
        account_slug=account_slug,
        claim_id=claim_id,
        claim_text=claim_text,
        quote=quote,
        source_slug=source_slug,
    )
    reviewer_due = clock.today() + timedelta(days=30)

    resp = review_client.post(
        f"/review/{row_id}/resolve",
        data={"owner_crm_id": "priya-natarajan", "due": reviewer_due.isoformat()},
        cookies=auth_cookies,
        follow_redirects=False,
    )

    assert resp.status_code == 303
    with db.cursor() as cur:
        cur.execute(
            "SELECT a.due, owner.crm_id FROM actions a "
            "JOIN registry_entities owner ON owner.id = a.owner_entity_id "
            "ORDER BY a.id DESC LIMIT 1"
        )
        due, owner_crm_id = cur.fetchone()
    assert due == reviewer_due
    assert owner_crm_id == "priya-natarajan"


def test_dismissing_an_action_card_creates_nothing(
    review_client: TestClient, db: psycopg.Connection, auth_cookies, loaded_registry: Registry
):
    account_slug = ensure_entity_note(db, loaded_registry, ASHGROVE_MEDICAL).slug
    claim_id, source_slug, claim_text, quote = _merge_complaint(db, loaded_registry)
    row_id = _queue_action_card(
        db,
        account_slug=account_slug,
        claim_id=claim_id,
        claim_text=claim_text,
        quote=quote,
        source_slug=source_slug,
    )

    with db.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM actions")
        (before,) = cur.fetchone()

    resp = review_client.post(
        f"/review/{row_id}/dismiss", cookies=auth_cookies, follow_redirects=False
    )

    assert resp.status_code == 303
    with db.cursor() as cur:
        cur.execute("SELECT status FROM review_queue WHERE id = %s", (row_id,))
        assert cur.fetchone() == ("dismissed",)
        cur.execute("SELECT COUNT(*) FROM actions")
        (after,) = cur.fetchone()
    assert after == before


# ---------------------------------------------------------------------
# B5: the confirm-route state machine.
# ---------------------------------------------------------------------


def test_b5_confirming_the_same_row_twice_only_creates_one_action(
    review_client: TestClient, db: psycopg.Connection, auth_cookies, loaded_registry: Registry
):
    account_slug = ensure_entity_note(db, loaded_registry, ASHGROVE_MEDICAL).slug
    claim_id, source_slug, claim_text, quote = _merge_complaint(db, loaded_registry)
    row_id = _queue_action_card(
        db,
        account_slug=account_slug,
        claim_id=claim_id,
        claim_text=claim_text,
        quote=quote,
        source_slug=source_slug,
    )

    first = review_client.post(
        f"/review/{row_id}/resolve", cookies=auth_cookies, follow_redirects=False
    )
    second = review_client.post(
        f"/review/{row_id}/resolve", cookies=auth_cookies, follow_redirects=False
    )

    assert "action_confirmed" in first.headers["location"]
    assert "already_handled" in second.headers["location"]
    with db.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM actions WHERE what = 'Follow up on the late delivery.'")
        (count,) = cur.fetchone()
    assert count == 1


def test_b5_dismissing_then_resolving_the_same_row_never_creates_an_action(
    review_client: TestClient, db: psycopg.Connection, auth_cookies, loaded_registry: Registry
):
    account_slug = ensure_entity_note(db, loaded_registry, ASHGROVE_MEDICAL).slug
    claim_id, source_slug, claim_text, quote = _merge_complaint(db, loaded_registry)
    row_id = _queue_action_card(
        db,
        account_slug=account_slug,
        claim_id=claim_id,
        claim_text=claim_text,
        quote=quote,
        source_slug=source_slug,
    )

    dismiss_resp = review_client.post(
        f"/review/{row_id}/dismiss", cookies=auth_cookies, follow_redirects=False
    )
    resolve_resp = review_client.post(
        f"/review/{row_id}/resolve", cookies=auth_cookies, follow_redirects=False
    )

    assert dismiss_resp.status_code == 303
    assert "already_handled" in resolve_resp.headers["location"]
    with db.cursor() as cur:
        cur.execute("SELECT status FROM review_queue WHERE id = %s", (row_id,))
        assert cur.fetchone() == ("dismissed",)
        cur.execute("SELECT COUNT(*) FROM actions")
        (count,) = cur.fetchone()
    assert count == 0


def test_should_fix_2_a_failed_confirm_leaves_the_row_open_and_never_reports_confirmed(
    review_client: TestClient, db: psycopg.Connection, auth_cookies, loaded_registry: Registry
):
    account_slug = ensure_entity_note(db, loaded_registry, ASHGROVE_MEDICAL).slug
    claim_id, source_slug, claim_text, quote = _merge_complaint(db, loaded_registry)
    row_id = _queue_action_card(
        db,
        account_slug=account_slug,
        claim_id=claim_id,
        claim_text=claim_text,
        quote=quote,
        source_slug=source_slug,
        owner=CUSTOMER_CONTACT,  # not vendor-side -- confirm_action_card refuses
    )

    resp = review_client.post(
        f"/review/{row_id}/resolve", cookies=auth_cookies, follow_redirects=False
    )

    assert "action_confirm_failed" in resp.headers["location"]
    with db.cursor() as cur:
        cur.execute("SELECT status FROM review_queue WHERE id = %s", (row_id,))
        assert cur.fetchone() == ("open",)
        cur.execute("SELECT COUNT(*) FROM actions")
        (count,) = cur.fetchone()
    assert count == 0

    from portolan.report.researcher import action_card_report_lines

    lines = action_card_report_lines(db, week="2026-W36")
    assert not any("(confirmed)" in line for line in lines)
    assert any("(proposed" in line for line in lines)


def test_b3_confirm_refuses_a_card_whose_evidence_claim_was_superseded_since_queuing(
    review_client: TestClient, db: psycopg.Connection, auth_cookies, loaded_registry: Registry
):
    """B3 (`01-spec/reviews/review-action-researcher-2026-09-20.md`): the
    queue is weekly; the gap between "gated" and "confirmed" is days.
    Crafted repro, verbatim from the review: queue a card that passes
    the gate, then a human corrects the fact it cites (`UPDATE claims
    SET superseded_at = now()`), then Confirm -- before this fix, the
    route created the action anyway, showing the reviewer a payload copy
    of the now-superseded text with nothing marking it struck through."""
    account_slug = ensure_entity_note(db, loaded_registry, ASHGROVE_MEDICAL).slug
    claim_id, source_slug, claim_text, quote = _merge_complaint(db, loaded_registry)
    row_id = _queue_action_card(
        db,
        account_slug=account_slug,
        claim_id=claim_id,
        claim_text=claim_text,
        quote=quote,
        source_slug=source_slug,
    )

    with db.cursor() as cur:
        cur.execute("UPDATE claims SET superseded_at = now() WHERE id = %s", (claim_id,))
    db.commit()

    resp = review_client.post(
        f"/review/{row_id}/resolve", cookies=auth_cookies, follow_redirects=False
    )

    assert "action_confirm_failed" in resp.headers["location"]
    with db.cursor() as cur:
        cur.execute("SELECT status FROM review_queue WHERE id = %s", (row_id,))
        assert cur.fetchone() == ("open",)
        cur.execute("SELECT COUNT(*) FROM actions")
        (count,) = cur.fetchone()
    assert count == 0


def test_should_fix_6_dismissing_with_a_reason_records_it(
    review_client: TestClient, db: psycopg.Connection, auth_cookies, loaded_registry: Registry
):
    account_slug = ensure_entity_note(db, loaded_registry, ASHGROVE_MEDICAL).slug
    claim_id, source_slug, claim_text, quote = _merge_complaint(db, loaded_registry)
    row_id = _queue_action_card(
        db,
        account_slug=account_slug,
        claim_id=claim_id,
        claim_text=claim_text,
        quote=quote,
        source_slug=source_slug,
    )

    review_client.post(
        f"/review/{row_id}/dismiss",
        data={"reason": "already handled by phone"},
        cookies=auth_cookies,
        follow_redirects=False,
    )

    with db.cursor() as cur:
        cur.execute(
            "SELECT payload ->> 'dismiss_reason' FROM review_queue WHERE id = %s", (row_id,)
        )
        (reason,) = cur.fetchone()
    assert reason == "already handled by phone"


def test_should_fix_7_confirm_with_override_writes_an_edit_record_and_updates_the_payload(
    review_client: TestClient, db: psycopg.Connection, auth_cookies, loaded_registry: Registry
):
    account_slug = ensure_entity_note(db, loaded_registry, ASHGROVE_MEDICAL).slug
    claim_id, source_slug, claim_text, quote = _merge_complaint(db, loaded_registry)
    row_id = _queue_action_card(
        db,
        account_slug=account_slug,
        claim_id=claim_id,
        claim_text=claim_text,
        quote=quote,
        source_slug=source_slug,
    )
    reviewer_due = (clock.today() + timedelta(days=30)).isoformat()

    review_client.post(
        f"/review/{row_id}/resolve",
        data={"owner_crm_id": "priya-natarajan", "due": reviewer_due},
        cookies=auth_cookies,
        follow_redirects=False,
    )

    with db.cursor() as cur:
        cur.execute(
            "SELECT before_text, after_text, reason FROM edit_records ORDER BY id DESC LIMIT 1"
        )
        before_text, after_text, reason = cur.fetchone()
        cur.execute(
            "SELECT payload -> 'card' ->> 'owner', payload -> 'card' ->> 'due' "
            "FROM review_queue WHERE id = %s",
            (row_id,),
        )
        payload_owner, payload_due = cur.fetchone()
    assert "sam-okafor" in before_text
    assert "priya-natarajan" in after_text
    assert reviewer_due in after_text
    assert reason == "owner/due overridden at confirm time"
    assert payload_owner == "priya-natarajan"
    assert payload_due == reviewer_due


# ---------------------------------------------------------------------
# Stale due dates (bug hunt 2026-09-22, orchestrator spec). Live row 208
# carries due=2026-09-20 (next_report); from 2026-09-21 Confirm returned
# nothing and the page blamed the owner or the account.
# ---------------------------------------------------------------------


def _queue_stale_card(db, loaded_registry: Registry) -> tuple[int, int]:
    account_slug = ensure_entity_note(db, loaded_registry, ASHGROVE_MEDICAL).slug
    claim_id, source_slug, claim_text, quote = _merge_complaint(db, loaded_registry)
    row_id = _queue_action_card(
        db,
        account_slug=account_slug,
        claim_id=claim_id,
        claim_text=claim_text,
        quote=quote,
        source_slug=source_slug,
        due=(clock.today() - timedelta(days=2)).isoformat(),
    )
    return row_id, claim_id


def _row_status_and_action_count(db, row_id: int) -> tuple[str, int]:
    with db.cursor() as cur:
        cur.execute("SELECT status FROM review_queue WHERE id = %s", (row_id,))
        (status,) = cur.fetchone()
        cur.execute("SELECT COUNT(*) FROM actions")
        (count,) = cur.fetchone()
    return status, count


def test_confirm_refuses_a_stale_proposed_due_and_tells_the_reviewer_to_set_a_new_one(
    review_client: TestClient, db: psycopg.Connection, auth_cookies, loaded_registry: Registry
):
    row_id, _claim_id = _queue_stale_card(db, loaded_registry)
    stale = (clock.today() - timedelta(days=2)).isoformat()

    # The browser posts the card's own (pre-filled) due back unchanged.
    resp = review_client.post(
        f"/review/{row_id}/resolve",
        data={"owner_crm_id": SAM_OKAFOR, "due": stale},
        cookies=auth_cookies,
        follow_redirects=False,
    )

    assert "notice=action_due_stale" in resp.headers["location"]
    assert _row_status_and_action_count(db, row_id) == ("open", 0)
    page = review_client.get(resp.headers["location"], cookies=auth_cookies)
    assert "Set a new due date on the card and confirm again." in page.text


def test_confirm_with_a_reviewer_set_due_confirms_a_stale_card(
    review_client: TestClient, db: psycopg.Connection, auth_cookies, loaded_registry: Registry
):
    row_id, _claim_id = _queue_stale_card(db, loaded_registry)
    new_due = clock.today() + timedelta(days=10)

    resp = review_client.post(
        f"/review/{row_id}/resolve",
        data={"owner_crm_id": SAM_OKAFOR, "due": new_due.isoformat()},
        cookies=auth_cookies,
        follow_redirects=False,
    )

    assert "notice=action_confirmed" in resp.headers["location"]
    assert _row_status_and_action_count(db, row_id) == ("resolved", 1)
    with db.cursor() as cur:
        cur.execute("SELECT what, due FROM actions")
        assert cur.fetchone() == ("Follow up on the late delivery.", new_due)


@pytest.mark.parametrize("days_from_today", [-1, 366])
def test_confirm_refuses_a_reviewer_due_in_the_past_or_more_than_a_year_out(
    review_client: TestClient,
    db: psycopg.Connection,
    auth_cookies,
    loaded_registry: Registry,
    days_from_today: int,
):
    row_id, _claim_id = _queue_stale_card(db, loaded_registry)
    bad_due = clock.today() + timedelta(days=days_from_today)

    resp = review_client.post(
        f"/review/{row_id}/resolve",
        data={"owner_crm_id": SAM_OKAFOR, "due": bad_due.isoformat()},
        cookies=auth_cookies,
        follow_redirects=False,
    )

    assert "notice=action_due_invalid" in resp.headers["location"]
    assert _row_status_and_action_count(db, row_id) == ("open", 0)


def test_a_reviewer_set_due_does_not_loosen_the_evidence_recheck(
    review_client: TestClient, db: psycopg.Connection, auth_cookies, loaded_registry: Registry
):
    row_id, claim_id = _queue_stale_card(db, loaded_registry)
    with db.cursor() as cur:
        cur.execute("UPDATE claims SET superseded_at = now() WHERE id = %s", (claim_id,))
    db.commit()
    new_due = (clock.today() + timedelta(days=10)).isoformat()

    resp = review_client.post(
        f"/review/{row_id}/resolve",
        data={"owner_crm_id": SAM_OKAFOR, "due": new_due},
        cookies=auth_cookies,
        follow_redirects=False,
    )

    assert "notice=action_confirm_failed" in resp.headers["location"]
    assert _row_status_and_action_count(db, row_id) == ("open", 0)


@pytest.mark.parametrize("verdict", ["no_action", "trigger_stale"])
def test_confirming_a_no_action_card_records_the_conclusion_and_creates_nothing(
    review_client: TestClient,
    db: psycopg.Connection,
    auth_cookies,
    loaded_registry: Registry,
    verdict: str,
):
    """Bug hunt 2026-09-22: the card says "Confirming records that
    conclusion" for a `no_action`/`trigger_stale` verdict, but Confirm
    always failed (`confirm_action_card` refuses any verdict but
    `action`), showed the owner/account failure notice, and left the row
    open for good -- only Dismiss could clear it."""
    account_slug = ensure_entity_note(db, loaded_registry, ASHGROVE_MEDICAL).slug
    claim_id, source_slug, claim_text, quote = _merge_complaint(db, loaded_registry)
    row_id = _queue_action_card(
        db,
        account_slug=account_slug,
        claim_id=claim_id,
        claim_text=claim_text,
        quote=quote,
        source_slug=source_slug,
    )
    with db.cursor() as cur:
        cur.execute(
            "UPDATE review_queue "
            "SET payload = jsonb_set(payload, '{card,verdict}', to_jsonb(%s::text)) "
            "WHERE id = %s",
            (verdict, row_id),
        )
    db.commit()

    resp = review_client.post(
        f"/review/{row_id}/resolve", cookies=auth_cookies, follow_redirects=False
    )

    assert "notice=action_card_concluded" in resp.headers["location"]
    assert _row_status_and_action_count(db, row_id) == ("resolved", 0)
    page = review_client.get(resp.headers["location"], cookies=auth_cookies)
    assert "no action was created" in page.text


def test_a_double_clicked_confirm_that_read_the_row_before_the_first_committed_creates_one_action(
    review_client: TestClient,
    db: psycopg.Connection,
    auth_cookies,
    loaded_registry: Registry,
    monkeypatch: pytest.MonkeyPatch,
):
    """Bug hunt 2026-09-22: B5's "already handled" guard read the row's
    status BEFORE the transaction, with no lock. Two Confirm POSTs from a
    double click run on two pool connections; the second can read
    `open` before the first commits and then create a second action.
    That interleaving is reproduced by handing the second request the
    stale `open` snapshot it would have read."""
    from portolan.web.routes import review as review_routes

    account_slug = ensure_entity_note(db, loaded_registry, ASHGROVE_MEDICAL).slug
    claim_id, source_slug, claim_text, quote = _merge_complaint(db, loaded_registry)
    row_id = _queue_action_card(
        db,
        account_slug=account_slug,
        claim_id=claim_id,
        claim_text=claim_text,
        quote=quote,
        source_slug=source_slug,
    )
    stale_snapshot = review_routes._get_row(db, row_id)

    first = review_client.post(
        f"/review/{row_id}/resolve", cookies=auth_cookies, follow_redirects=False
    )
    monkeypatch.setattr(review_routes, "_get_row", lambda conn, rid: dict(stale_snapshot))
    second = review_client.post(
        f"/review/{row_id}/resolve", cookies=auth_cookies, follow_redirects=False
    )

    assert "action_confirmed" in first.headers["location"]
    assert "already_handled" in second.headers["location"]
    assert _row_status_and_action_count(db, row_id) == ("resolved", 1)
