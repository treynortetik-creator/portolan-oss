"""`/notes/{action-slug}` and `/actions`: the "Why" block (follow-up to
`01-spec/actions-and-owners-2026-09-19.md`'s Phase 2, 2026-09-20 build
instructions): a card-confirmed action renders its evidence/findings/
inference instead of the boilerplate "is an open action." synthesis, and
`/actions` shows a one-line why under it. An action from the extraction
pipeline (no card) keeps today's rendering, unchanged.
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
from portolan.report.researcher import CURRENT_BRIEF_VERSION, confirm_action_card
from portolan.report.researcher_tools import record_finding
from portolan.search.embed import FakeEmbedder
from portolan.store import LocalStore
from portolan.web.app import create_app
from portolan.web.deps import get_conn, get_embedder_dep, get_store_dep
from tests.merge_helpers import (
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
ORG_KEY = "test-org-key-for-action-why-notes"
SESSION_SECRET = "test-session-secret-for-action-why-notes-not-a-real-secret"


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
def why_client(db: psycopg.Connection, db_url: str, tmp_path: Path):
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
def auth_cookies(why_client: TestClient):
    r = why_client.post(
        "/login", data={"org_key": ORG_KEY, "editor": "Test Editor"}, follow_redirects=False
    )
    assert r.status_code == 303
    return r.cookies


def _merge_complaint(db, registry: Registry) -> tuple[int, str, str, str | None]:
    embedder = FakeEmbedder()
    llm = FakeLLMClient([])
    source = make_source_row(db, "s-notes-why", source_date=date(2026, 8, 20))
    fact = build_fact(
        "F-notes-why",
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
        cur.execute(
            "SELECT claim_text, quote, source_slug FROM claims WHERE id = %s", (outcome.claim_id,)
        )
        claim_text, quote, source_slug = cur.fetchone()
    return outcome.claim_id, source_slug, claim_text, quote


def test_a_confirmed_researched_action_renders_the_why_block(
    why_client: TestClient, db: psycopg.Connection, auth_cookies, loaded_registry: Registry
):
    claim_id, source_slug, claim_text, quote = _merge_complaint(db, loaded_registry)
    account_note = ensure_entity_note(db, loaded_registry, ASHGROVE_MEDICAL)
    finding_id = record_finding(
        db,
        run_id="test-run",
        candidate_slug="test-candidate",
        tool="absence_check",
        arguments={
            "account_slug": account_note.slug,
            "theme_slug": None,
            "fact_types": ["commitment"],
            "after_date": "2026-08-22",
        },
        result={
            "count": 0,
            "latest_date": None,
            "claim_ids": [],
            "count_before": 1,
            "had_matches_before": True,
            "claim_ids_before": [claim_id],
        },
    )
    card = {
        "run_id": "test-run",
        "candidate_slug": "test-candidate",
        "brief_version": CURRENT_BRIEF_VERSION,
        "account": account_note.slug,
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
        "findings": [finding_id],
        "inference": [
            {
                "sentence": "No commitment claims after 2026-08-22.",
                "evidence_refs": [],
                "finding_refs": [0],
            }
        ],
        "owner": SAM_OKAFOR,
        "due": "2026-10-01",
        "due_rule": "next_report",
        "theme": None,
        "confidence": "medium",
        "confidence_refs": {"evidence_refs": [0], "finding_refs": [0]},
    }

    action_id = confirm_action_card(db, card=card, editor="Test Editor", as_of=CARD_AS_OF)
    assert action_id is not None
    with db.cursor() as cur:
        cur.execute("SELECT note_slug FROM actions WHERE id = %s", (action_id,))
        (note_slug,) = cur.fetchone()

    resp = why_client.get(f"/notes/{note_slug}", cookies=auth_cookies)

    assert resp.status_code == 200
    body = resp.text
    assert "is an open action." not in body
    assert "Why" in body
    assert "No commitment claims after 2026-08-22." in body
    # `claim_text` itself carries `[[ashgrove-medical-center]]`-style
    # wikilinks, resolved to real `<a>` tags by `linkify_wikilinks` in the
    # Evidence block -- the surrounding plain words survive that pass.
    assert "received a late" in body
    assert f"/originals/{source_slug}?at=p1" in body
    assert "before." in body  # the finding's plain_text line
    assert "confirmed by Test Editor" in body
    assert "confidence medium" in body

    # The Claims card lists the evidence claim, linking it via its OWN
    # note (the signal note it was originally extracted onto) -- not a
    # re-extracted copy on the action's own note.
    assert f'id="claim-{claim_id}"' in body
    with db.cursor() as cur:
        cur.execute("SELECT note_slug FROM claims WHERE id = %s", (claim_id,))
        (claim_home_slug,) = cur.fetchone()
    assert f"/notes/{claim_home_slug}/edit?claim_id={claim_id}" in body


def test_a_pipeline_action_with_no_card_keeps_todays_rendering(
    why_client: TestClient, db: psycopg.Connection, auth_cookies, loaded_registry: Registry
):
    source = make_source_row(db, "s-notes-why-pipeline", source_date=date(2026, 8, 20))
    action = build_action(
        "Send Ashgrove Medical a written delivery root-cause summary.",
        owner_mention=None,
        due=date(2026, 9, 1),
        locator="p1",
    )
    outcome = merge_action(db, loaded_registry, action, source, entity_id=ASHGROVE_MEDICAL)

    resp = why_client.get(f"/notes/{outcome.note_slug}", cookies=auth_cookies)

    assert resp.status_code == 200
    assert "is an open action." in resp.text
    assert '<p class="eyebrow">Why</p>' not in resp.text


def test_the_atlas_note_pane_shows_the_why_and_the_evidence_claims(
    why_client: TestClient, db: psycopg.Connection, auth_cookies, loaded_registry: Registry
):
    """Bug hunt 2026-09-22: the Atlas pane for a card-confirmed action
    still rendered the boilerplate "is an open action." synthesis and no
    Claims card at all (the action note owns no claims), so the one
    place a reader browses actions from showed no reason and no source."""
    claim_id, source_slug, claim_text, quote = _merge_complaint(db, loaded_registry)
    account_note = ensure_entity_note(db, loaded_registry, ASHGROVE_MEDICAL)
    card = {
        "run_id": "test-run",
        "candidate_slug": "test-candidate",
        "brief_version": CURRENT_BRIEF_VERSION,
        "account": account_note.slug,
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
        "owner": SAM_OKAFOR,
        "due": "2026-10-01",
        "due_rule": "next_report",
        "theme": None,
        "confidence": "medium",
        "confidence_refs": {"evidence_refs": [0], "finding_refs": []},
    }
    action_id = confirm_action_card(db, card=card, editor="Test Editor", as_of=CARD_AS_OF)
    assert action_id is not None
    with db.cursor() as cur:
        cur.execute("SELECT note_slug FROM actions WHERE id = %s", (action_id,))
        (note_slug,) = cur.fetchone()
        cur.execute("SELECT note_slug FROM claims WHERE id = %s", (claim_id,))
        (claim_home_slug,) = cur.fetchone()

    resp = why_client.get(f"/atlas/{note_slug}", cookies=auth_cookies)

    assert resp.status_code == 200
    body = resp.text
    assert '<p class="eyebrow">Why</p>' in body
    assert "is an open action." not in body
    assert "confirmed by Test Editor" in body
    assert f"/originals/{source_slug}?at=p1" in body
    assert f'id="claim-{claim_id}"' in body
    assert f"/notes/{claim_home_slug}/edit?claim_id={claim_id}" in body


def _confirm_card_with_a_finding(db: psycopg.Connection, registry: Registry) -> dict[str, object]:
    claim_id, source_slug, claim_text, quote = _merge_complaint(db, registry)
    account_note = ensure_entity_note(db, registry, ASHGROVE_MEDICAL)
    finding_id = record_finding(
        db,
        run_id="test-run",
        candidate_slug="test-candidate",
        tool="absence_check",
        arguments={
            "account_slug": account_note.slug,
            "theme_slug": None,
            "fact_types": ["commitment"],
            "after_date": "2026-08-22",
        },
        result={
            "count": 0,
            "latest_date": None,
            "claim_ids": [],
            "count_before": 1,
            "had_matches_before": True,
            "claim_ids_before": [claim_id],
        },
    )
    card = {
        "run_id": "test-run",
        "candidate_slug": "test-candidate",
        "brief_version": CURRENT_BRIEF_VERSION,
        "account": account_note.slug,
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
        "findings": [finding_id],
        "inference": [
            {
                "sentence": "No commitment claims after 2026-08-22.",
                "evidence_refs": [],
                "finding_refs": [0],
            }
        ],
        "owner": SAM_OKAFOR,
        "due": "2026-10-01",
        "due_rule": "next_report",
        "theme": None,
        "confidence": "medium",
        "confidence_refs": {"evidence_refs": [0], "finding_refs": [0]},
    }
    action_id = confirm_action_card(db, card=card, editor="Test Editor", as_of=CARD_AS_OF)
    assert action_id is not None
    with db.cursor() as cur:
        cur.execute("SELECT note_slug FROM actions WHERE id = %s", (action_id,))
        (note_slug,) = cur.fetchone()
    return {
        "note_slug": note_slug,
        "account_slug": account_note.slug,
        "claim_id": claim_id,
        "source_slug": source_slug,
    }


def test_the_md_download_of_a_confirmed_action_carries_its_why(
    why_client: TestClient, db: psycopg.Connection, auth_cookies, loaded_registry: Registry
):
    """Demo script 2026-09-22 ("do not click the Download .md on an action
    note"): the download used the old rendering, the boilerplate synthesis
    with no why and no claims."""
    made = _confirm_card_with_a_finding(db, loaded_registry)
    slug, source = made["note_slug"], made["source_slug"]

    resp = why_client.get(f"/notes/{slug}.md", cookies=auth_cookies)

    assert resp.status_code == 200
    assert resp.headers["content-disposition"] == f'attachment; filename="{slug}.md"'
    md = resp.text
    assert "is an open action." not in md
    assert "- **What:** Follow up on the late delivery." in md
    assert "- **Owner:** [Sam Okafor](" in md
    assert "- **Due:** 2026-10-01" in md
    assert "## Why\n\nNo commitment claims after 2026-08-22." in md
    assert "## Evidence" in md
    cite = f"[source: {source} @ p1](http://testserver/originals/{source}?at=p1)"
    assert cite in md
    assert '(quote: "late delivery")' in md
    assert "## What the system checked" in md
    assert "before." in md
    assert "confirmed by Test Editor" in md
    assert "confidence medium" in md
    # The evidence claim itself, in the Claims section, with its source.
    claims_section = md.split("## Claims", 1)[1]
    assert "complaint" in claims_section
    assert cite in claims_section
    assert "[[" not in md


def test_the_atlas_account_pane_lists_its_open_actions(
    why_client: TestClient, db: psycopg.Connection, auth_cookies, loaded_registry: Registry
):
    """An account opened in Atlas shows its open actions (owner, due, the
    one-line why), each linking to the action's own Atlas pane."""
    made = _confirm_card_with_a_finding(db, loaded_registry)

    resp = why_client.get(f"/atlas/{made['account_slug']}", cookies=auth_cookies)

    assert resp.status_code == 200
    pane = resp.text.split('class="atlas-main"', 1)[1]
    assert "Open actions" in pane
    assert f'href="/atlas/{made["note_slug"]}">Follow up on the late delivery.</a>' in pane
    assert "Sam Okafor" in pane
    assert "No commitment claims after 2026-08-22." in pane
