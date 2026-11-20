"""`/review/{id}/resolve` and `/review/{id}/dismiss` for `owner_proposal`
rows (`01-spec/actions-and-owners-2026-09-19.md`, Phase 1, build item 3).

Uses the same lightweight `db`-fixture + `TestClient` pattern as
`tests/test_review_pagination.py` (see that module's own docstring for
why: raw-SQL seeded rows, not a real pipeline run).
"""

from __future__ import annotations

import json
from pathlib import Path

import psycopg
import pytest
from fastapi.testclient import TestClient

from portolan.config import Settings
from portolan.registry import Registry
from portolan.registry.owners import confirm_owner_proposal
from portolan.search.embed import FakeEmbedder
from portolan.store import LocalStore
from portolan.web.app import create_app
from portolan.web.deps import get_conn, get_embedder_dep, get_store_dep
from tests.merge_helpers import (
    ASHGROVE,
    SAM_OKAFOR,
    loaded_registry_fixture,  # noqa: F401
)

ORG_KEY = "test-org-key-for-owner-proposal-route"
SESSION_SECRET = "test-session-secret-for-owner-proposal-route-not-a-real-secret"


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
def owner_client(db: psycopg.Connection, db_url: str, tmp_path: Path):
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
def auth_cookies(owner_client: TestClient):
    r = owner_client.post(
        "/login", data={"org_key": ORG_KEY, "editor": "Test Editor"}, follow_redirects=False
    )
    assert r.status_code == 303
    return r.cookies


def _seed_owner_proposal(conn: psycopg.Connection) -> int:
    payload = json.dumps(
        {
            "account_crm_id": ASHGROVE,
            "account_name": "Ashgrove Health",
            "candidates": [
                {
                    "crm_id": SAM_OKAFOR,
                    "name": "Sam Okafor",
                    "source_count": 3,
                    "last_seen": "2026-08-24",
                    "commitments": 1,
                }
            ],
        }
    )
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO review_queue (kind, payload) VALUES ('owner_proposal', %s::jsonb) "
            "RETURNING id",
            (payload,),
        )
        row_id = cur.fetchone()[0]
    conn.commit()
    return row_id


def test_confirm_owner_proposal_sets_the_registry_owner_and_shows_a_notice(
    owner_client: TestClient, db: psycopg.Connection, auth_cookies, loaded_registry: Registry
) -> None:
    row_id = _seed_owner_proposal(db)

    resp = owner_client.post(
        f"/review/{row_id}/resolve", cookies=auth_cookies, follow_redirects=False
    )

    assert resp.status_code == 303
    assert "notice=owner_confirmed" in resp.headers["location"]

    with db.cursor() as cur:
        cur.execute(
            "SELECT owner.crm_id FROM registry_entities a "
            "JOIN registry_entities owner ON owner.id = a.owner_entity_id "
            "WHERE a.crm_id = %s",
            (ASHGROVE,),
        )
        assert cur.fetchone() == (SAM_OKAFOR,)
    with db.cursor() as cur:
        cur.execute("SELECT status FROM review_queue WHERE id = %s", (row_id,))
        assert cur.fetchone() == ("resolved",)


def test_dismiss_owner_proposal_leaves_the_account_unowned(
    owner_client: TestClient, db: psycopg.Connection, auth_cookies, loaded_registry: Registry
) -> None:
    row_id = _seed_owner_proposal(db)

    resp = owner_client.post(
        f"/review/{row_id}/dismiss", cookies=auth_cookies, follow_redirects=False
    )

    assert resp.status_code == 303
    with db.cursor() as cur:
        cur.execute("SELECT owner_entity_id FROM registry_entities WHERE crm_id = %s", (ASHGROVE,))
        assert cur.fetchone()[0] is None
    with db.cursor() as cur:
        cur.execute("SELECT status FROM review_queue WHERE id = %s", (row_id,))
        assert cur.fetchone() == ("dismissed",)


def test_confirm_refuses_a_payload_whose_candidate_is_no_longer_vendor_side(
    owner_client: TestClient, db: psycopg.Connection, auth_cookies, loaded_registry: Registry
) -> None:
    """A hand-edited/corrupted proposal naming a customer contact as the
    top candidate must never set the owner field -- the alias-conflict
    style guard `confirm_owner_proposal` already enforces, exercised here
    through the real HTTP route."""
    with db.cursor() as cur:
        cur.execute("SELECT crm_id FROM registry_entities WHERE kind = 'contact' LIMIT 1")
        customer_contact_crm_id = cur.fetchone()[0]
    payload = json.dumps(
        {
            "account_crm_id": ASHGROVE,
            "account_name": "Ashgrove Health",
            "candidates": [{"crm_id": customer_contact_crm_id, "name": "Not vendor-side"}],
        }
    )
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO review_queue (kind, payload) VALUES ('owner_proposal', %s::jsonb) "
            "RETURNING id",
            (payload,),
        )
        row_id = cur.fetchone()[0]
    db.commit()

    resp = owner_client.post(
        f"/review/{row_id}/resolve", cookies=auth_cookies, follow_redirects=False
    )

    assert resp.status_code == 303
    assert "notice=owner_confirm_failed" in resp.headers["location"]
    with db.cursor() as cur:
        cur.execute("SELECT owner_entity_id FROM registry_entities WHERE crm_id = %s", (ASHGROVE,))
        assert cur.fetchone()[0] is None


def test_confirm_owner_proposal_helper_used_directly_is_also_covered(
    db: psycopg.Connection, loaded_registry: Registry
) -> None:
    """Belt-and-suspenders: the same helper the route calls, called
    directly (mirrors `tests/test_review_routes.py`'s own direct-call
    style for `_keep_quote_unsupported_fact`)."""
    assert confirm_owner_proposal(db, account_crm_id=ASHGROVE, owner_crm_id=SAM_OKAFOR) is True
