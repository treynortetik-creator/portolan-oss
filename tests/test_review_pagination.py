"""Regression tests for the `/review` P0 fix and the alias-conflict fix
(`01-spec/reviews/bughunt-2026-09-18-web-pages.md`, findings 1 and 4).

Uses the lightweight, per-test `db` fixture (`tests/conftest.py`: a
freshly migrated, EMPTY schema, no pipeline run) rather than
`test_web.py`'s session-scoped `demo_conn`/pipeline fixture -- these
tests seed hundreds to thousands of `review_queue`/`registry_entities`
rows directly with raw SQL, which would be both slow (through the real
pipeline) and would pollute the shared demo graph other `test_web.py`
tests assert exact counts against.
"""

from __future__ import annotations

import json
from pathlib import Path

import psycopg
import pytest
from fastapi.testclient import TestClient

from portolan.config import Settings
from portolan.search.embed import FakeEmbedder
from portolan.store import LocalStore
from portolan.web.app import create_app
from portolan.web.deps import get_conn, get_embedder_dep, get_store_dep
from portolan.web.routes.review import REVIEW_PAGE_SIZE

ORG_KEY = "test-org-key-for-review-pagination"
SESSION_SECRET = "test-session-secret-for-review-pagination-not-a-real-secret"


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


def _seed_open_rows(conn: psycopg.Connection, count: int, *, kind: str = "unknown_entity") -> None:
    """`count` open `unknown_entity` rows, oldest first, each carrying its
    own claim/quote/locator already on the payload -- `_needs_evidence_
    backfill` (`portolan.web.routes.review`) treats a payload with a
    `claim` as already backfilled, so seeding it here keeps these rows
    from triggering a per-row database lookup the perf test isn't
    measuring."""
    with conn.cursor() as cur:
        for i in range(count):
            payload = json.dumps(
                {
                    "mention": f"Test Co {i:04d}",
                    "context": {
                        "claim": f"Test Co {i:04d} mentioned something in a call.",
                        "quote": "something",
                        "locator": f"loc-{i}",
                        "fact_type": "other",
                    },
                }
            )
            cur.execute(
                "INSERT INTO review_queue (kind, payload, created) VALUES "
                "(%s, %s::jsonb, now() - (%s || ' seconds')::interval)",
                (kind, payload, count - i),
            )


def _seed_registry_entities(conn: psycopg.Connection, count: int) -> None:
    with conn.cursor() as cur:
        for i in range(count):
            cur.execute(
                "INSERT INTO registry_entities (kind, canonical_name, crm_id, active) "
                "VALUES ('account', %s, %s, true)",
                (f"Registry Account {i:05d}", f"A-{i:05d}"),
            )


# --------------------------------------------------------------------
# P0 (finding 1): page size and speed at CRM-import scale.
# --------------------------------------------------------------------


def test_review_page_stays_small_with_600_open_rows_and_2000_registry_entities(
    review_client: TestClient, db: psycopg.Connection, auth_cookies
) -> None:
    """Bug hunt 2026-09-18 finding 1: before the fix, `/review` rendered
    the ENTIRE registry as `<option>`s inside EVERY open card's alias
    form -- measured at 549 rows / 2,037 entities: 223,726,913 bytes,
    9.71s, 1,102,558 `<option>` tags. Reproduces at the same order of
    magnitude (600 open rows, 2,000 registry entities, every row showing
    the alias picker) and asserts the response stays under 2 MB with one
    page of cards -- this is the fix's own regression test; it fails
    against the pre-fix code (223 MB, 600 cards) and passes against the
    paginated, on-demand-registry version. No wall-clock assert: size and
    card count already fail on the old code, and a timing bar only fails
    under load.
    """
    _seed_open_rows(db, 600)
    _seed_registry_entities(db, 2000)

    r = review_client.get("/review", cookies=auth_cookies)

    assert r.status_code == 200
    assert len(r.content) < 2 * 1024 * 1024, f"page was {len(r.content):,} bytes"
    # Paginated: only REVIEW_PAGE_SIZE cards on the first page, not all 600.
    assert r.text.count('class="card review-card"') == REVIEW_PAGE_SIZE
    # The registry itself never ships in the page -- no real `<option ...>`
    # markup (an attributed tag, e.g. `<option value=...>`; a bare
    # "<option>" can still appear in this app's own prose comments about
    # the fix, which isn't what this asserts against) and none of the
    # 2,000 seeded canonical names appear.
    assert "<option " not in r.text
    assert '<select name="entity_id"' not in r.text
    assert "Registry Account" not in r.text


def test_review_registry_endpoint_returns_bounded_json_matches(
    review_client: TestClient, db: psycopg.Connection, auth_cookies
) -> None:
    """The on-demand replacement for the old per-card `<option>` list:
    `GET /review/registry?q=` returns at most 20 matches as JSON."""
    _seed_registry_entities(db, 2000)

    r = review_client.get(
        "/review/registry", params={"q": "Registry Account 001"}, cookies=auth_cookies
    )
    assert r.status_code == 200
    matches = r.json()
    assert 0 < len(matches) <= 20
    for m in matches:
        assert set(m) == {"crm_id", "canonical_name", "kind", "description"}
        assert "Registry Account 001" in m["canonical_name"]


def test_review_registry_endpoint_requires_a_session(review_client: TestClient) -> None:
    r = review_client.get("/review/registry", params={"q": "x"})
    assert r.status_code == 401


# --------------------------------------------------------------------
# Pagination correctness: 25/page, oldest first, kind filter preserved.
# --------------------------------------------------------------------


def test_review_pagination_covers_every_row_exactly_once_oldest_first(
    review_client: TestClient, db: psycopg.Connection, auth_cookies
) -> None:
    import re

    _seed_open_rows(db, 63)  # not a multiple of REVIEW_PAGE_SIZE

    seen_mentions: list[str] = []
    total_pages = None
    page = 1
    while True:
        r = review_client.get("/review", params={"page": page}, cookies=auth_cookies)
        assert r.status_code == 200
        # Each row's own mention appears more than once per card (headline,
        # why-it-matters, technical fold) -- dedupe consecutive repeats by
        # taking only each card's FIRST mention, in document order.
        page_mentions = list(dict.fromkeys(re.findall(r"Test Co (\d{4})", r.text)))
        seen_mentions.extend(page_mentions)
        m = re.search(r"page \d+ of (\d+)", r.text)
        assert m is not None
        total_pages = int(m.group(1))
        if page >= total_pages:
            break
        page += 1

    assert total_pages == 3  # ceil(63 / 25)
    # Oldest-first order preserved end to end, no row skipped or repeated.
    assert seen_mentions == [f"{i:04d}" for i in range(63)]


def test_review_kind_filter_is_preserved_across_pages(
    review_client: TestClient, db: psycopg.Connection, auth_cookies
) -> None:
    """Build guidance: "paginate the queue ... with the kind filter
    preserved across pages" -- filtering to one kind must narrow the
    SAME paginated list, not just hide cards on whatever page happened
    to load first."""
    _seed_open_rows(db, 30, kind="unknown_entity")
    _seed_open_rows(db, 5, kind="failed_consolidation")

    r = review_client.get("/review", params={"kind": "Unknown name"}, cookies=auth_cookies)
    assert r.status_code == 200
    assert "Tidy-up failed" not in r.text or 'href="/review?kind=' in r.text  # filter chip present
    assert r.text.count('class="card review-card"') == REVIEW_PAGE_SIZE

    r2 = review_client.get(
        "/review", params={"kind": "Unknown name", "page": 2}, cookies=auth_cookies
    )
    assert r2.status_code == 200
    # Page 2 of the FILTERED set: 30 unknown_entity rows total, 5 left.
    assert r2.text.count('class="card review-card"') == 5
    assert "failed_consolidation" not in r2.text.lower() or "Tidy-up" not in r2.text


# --------------------------------------------------------------------
# Finding 4: add-alias must not create a second, conflicting alias.
# --------------------------------------------------------------------


def _one(conn: psycopg.Connection, sql: str, params: tuple = ()):
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchone()


def test_add_alias_twice_on_the_same_row_does_not_create_a_second_alias(
    review_client: TestClient, db: psycopg.Connection, auth_cookies
) -> None:
    """Bug hunt finding 4's own repro: `POST /review/1/add-alias
    entity_id=A-1` succeeds; a SECOND `POST` on the SAME row_id (a
    double submit) used to succeed too, mapping the mention to a second,
    conflicting entity. The row is already `resolved` after the first
    call, so the second must be refused with no state change."""
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO review_queue (kind, payload) VALUES ('unknown_entity', %s) RETURNING id",
            ('{"mention": "Lindenhurst", "context": {}}',),
        )
        (row_id,) = cur.fetchone()
        cur.execute(
            "INSERT INTO registry_entities (kind, canonical_name, crm_id) "
            "VALUES ('account', 'Lindenhurst Medical', 'A-1001') RETURNING id"
        )
        cur.execute(
            "INSERT INTO registry_entities (kind, canonical_name, crm_id) "
            "VALUES ('account', 'Lindenhurst Regional', 'A-1002') RETURNING id"
        )

    r1 = review_client.post(
        f"/review/{row_id}/add-alias",
        data={"entity_id": "A-1001"},
        cookies=auth_cookies,
        follow_redirects=False,
    )
    assert r1.status_code == 303
    assert _one(db, "SELECT status FROM review_queue WHERE id = %s", (row_id,))[0] == "resolved"

    r2 = review_client.post(
        f"/review/{row_id}/add-alias",
        data={"entity_id": "A-1002"},
        cookies=auth_cookies,
        follow_redirects=False,
    )
    assert r2.status_code == 303
    assert "notice=already_handled" in r2.headers["location"]

    aliases = _one(
        db,
        "SELECT count(*) FROM registry_aliases WHERE lower(alias) = lower('Lindenhurst')",
    )[0]
    assert aliases == 1


def test_add_alias_for_a_mention_already_aliased_to_a_different_entity_is_refused(
    review_client: TestClient, db: psycopg.Connection, auth_cookies
) -> None:
    """A SECOND open row citing the same mention (not a double submit on
    the same row_id) must also be refused if a human already mapped that
    mention to a different entity -- finding 4's "or a mention that
    already has an alias"."""
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO registry_entities (kind, canonical_name, crm_id) "
            "VALUES ('account', 'Lindenhurst Medical', 'A-1001') RETURNING id"
        )
        (dbid_1,) = cur.fetchone()
        cur.execute(
            "INSERT INTO registry_entities (kind, canonical_name, crm_id) "
            "VALUES ('account', 'Lindenhurst Regional', 'A-1002') RETURNING id"
        )
        cur.execute(
            "INSERT INTO registry_aliases (alias, entity_id, source) VALUES (%s, %s, 'human')",
            ("Lindenhurst", dbid_1),
        )
        cur.execute(
            "INSERT INTO review_queue (kind, payload) VALUES ('unknown_entity', %s) RETURNING id",
            ('{"mention": "Lindenhurst", "context": {}}',),
        )
        (row_id,) = cur.fetchone()

    r = review_client.post(
        f"/review/{row_id}/add-alias",
        data={"entity_id": "A-1002"},
        cookies=auth_cookies,
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert "notice=alias_conflict" in r.headers["location"]
    # Nothing changed: the row is still open, and there's still exactly
    # one alias for this mention (the original one, to A-1001).
    assert _one(db, "SELECT status FROM review_queue WHERE id = %s", (row_id,))[0] == "open"
    aliases = _one(
        db,
        "SELECT count(*) FROM registry_aliases WHERE lower(alias) = lower('Lindenhurst')",
    )[0]
    assert aliases == 1

    # The picked-again-correctly case (same entity as the existing alias)
    # is NOT a conflict and must still succeed -- "a mention keeps its
    # first mapping" is about a DIFFERENT target, not the same one twice.
    r2 = review_client.post(
        f"/review/{row_id}/add-alias",
        data={"entity_id": "A-1001"},
        cookies=auth_cookies,
        follow_redirects=False,
    )
    assert r2.status_code == 303
    assert "notice=alias_conflict" not in r2.headers["location"]
    assert _one(db, "SELECT status FROM review_queue WHERE id = %s", (row_id,))[0] == "resolved"
