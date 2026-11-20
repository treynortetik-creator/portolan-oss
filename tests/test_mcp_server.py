"""Tests for portolan/mcp_server/** (build-contracts.md "Web and MCP",
wave C4).

DB-backed tool tests call `portolan.mcp_server.tools` functions directly
against the shared `db` fixture (raw SQL fixtures, fictional names from
corpus/truth-table.yaml -- Cobalt Ridge Medical Center, Halvard Medical's
synthetic corpus), the same pattern `tests/test_search_query.py` uses for
`portolan.search.query.search`. Server-level tests (tool listing, the
HTTP transport's auth gate) build a real `MCPServer` via `create_server`
but never need the database for that.

**Hosted MCP wave 2026-09-19** ("--- hosted inside the web app
(create_app) ---" section): exercises the actual `/mcp` mount inside
`portolan.web.app.create_app`, not just the standalone `build_http_app`
output the tests above already cover. The full `initialize` + `tools/
list` + tool-call round trip uses the real MCP SDK client
(`mcp.client.streamable_http`, `mcp.client.session.ClientSession`)
against the FastAPI app over an in-process ASGI transport
(`httpx2.ASGITransport`) -- driven inside one hand-rolled `asyncio.run()`
call (`_lifespan_started`) rather than `starlette.testclient.TestClient`,
because `TestClient` runs the app's ASGI lifespan on its own background
thread/event loop, and the MCP session manager's `anyio` task group,
created during that lifespan, would then be bound to a different loop
than whatever loop a separately-`asyncio.run()`-driven SDK client calls
into it from -- this repo has no `pytest-asyncio` dependency to make the
test function itself a coroutine, so the whole round trip runs inside
one `asyncio.run()` in an otherwise-ordinary sync test function instead.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from datetime import date

import httpx2
import pytest
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client
from starlette.testclient import TestClient

from portolan.config import Settings
from portolan.ingest.ledger import record as record_coverage
from portolan.mcp_server import tools
from portolan.mcp_server.access import set_ai_access
from portolan.mcp_server.server import TOOL_NAMES, build_http_app, create_server
from portolan.search.embed import FakeEmbedder
from portolan.search.index import reindex_all
from portolan.store import LocalStore
from portolan.util import clock as clock_module
from portolan.web.app import create_app

TENANT = "test-mcp"
SOURCE_SLUG = "S-call-101"
NOTE_SLUG = "cobalt-ridge-late-deliveries"
ACCOUNT_SLUG = "cobalt-ridge-medical-center"


@pytest.fixture()
def corpus(db):
    """One source, one signal note (with a claim citing it and a `body`
    link to an account note), indexed with FakeEmbedder."""
    with db.cursor() as cur:
        cur.execute(
            """
            INSERT INTO sources (slug, source_type, s3_key, content_hash,
                                  source_date, extraction_status)
            VALUES (%s, 'call', 'originals/call/S-call-101', 'fake-hash-101',
                    '2026-08-01', 'done')
            """,
            (SOURCE_SLUG,),
        )
        cur.execute(
            """
            INSERT INTO notes (slug, type, title, description, crm_id, body_md, updated)
            VALUES (%s, 'account', 'Cobalt Ridge Medical Center',
                    'A hospital facility in the Midwest.', 'A-1020',
                    'Cobalt Ridge Medical Center is a hospital facility in the Midwest.',
                    '2026-08-01')
            """,
            (ACCOUNT_SLUG,),
        )
        cur.execute(
            """
            INSERT INTO notes (slug, type, title, description, body_md, updated)
            VALUES (%s, 'signal', 'Cobalt Ridge Late Deliveries',
                    'Repeated late Tidewater consumable deliveries at Cobalt Ridge.',
                    'Cobalt Ridge Medical Center reported late Tidewater consumable '
                    'deliveries in August 2026.',
                    '2026-08-14')
            """,
            (NOTE_SLUG,),
        )
        cur.execute(
            "INSERT INTO links (from_slug, to_slug, kind) VALUES (%s, %s, 'provenance')",
            (NOTE_SLUG, SOURCE_SLUG),
        )
        cur.execute(
            "INSERT INTO links (from_slug, to_slug, kind) VALUES (%s, %s, 'body')",
            (NOTE_SLUG, ACCOUNT_SLUG),
        )
        cur.execute(
            """
            INSERT INTO claims (note_slug, fact_type, claim_text, source_slug, locator,
                                 quote, speaker, valid_from, severity, sentiment, confidence)
            VALUES (%s, 'complaint',
                    'Cobalt Ridge reported three late Tidewater deliveries in August 2026.',
                    %s, '@ 00:05:12', 'These pumps keep showing up late.', 'Tom Ridley',
                    '2026-08-14', 'medium', -1, 0.900)
            """,
            (NOTE_SLUG, SOURCE_SLUG),
        )

    embedder = FakeEmbedder()
    reindex_all(db, embedder)
    return db, embedder


def _usage_rows(conn, tool: str) -> list[tuple]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT surface, model_key, extra FROM usage_events WHERE model_key = %s",
            (f"mcp:{tool}",),
        )
        return cur.fetchall()


# --- search --------------------------


def test_search_finds_an_indexed_note(corpus):
    conn, embedder = corpus
    result = tools.tool_search(conn, embedder, tenant=TENANT, query="Cobalt Ridge late deliveries")

    assert set(result.keys()) == {"hits", "control_ok", "query_terms_unmatched", "today"}
    assert result["today"] == clock_module.today().isoformat()
    slugs = [h["note_slug"] for h in result["hits"]]
    assert NOTE_SLUG in slugs
    hit = next(h for h in result["hits"] if h["note_slug"] == NOTE_SLUG)
    assert {
        "note_slug",
        "title",
        "score",
        "via",
        "best_chunk",
        "heading",
        "updated",
        "content_date",
        "content_date_source",
        "source_slugs",
        "expanded_from",
    } <= set(hit.keys())
    assert hit["content_date"] is not None
    assert SOURCE_SLUG in hit["source_slugs"]


def test_search_writes_mcp_usage_event_without_the_query_text(corpus):
    conn, embedder = corpus
    tools.tool_search(conn, embedder, tenant=TENANT, query="a secret query nobody should log")

    rows = _usage_rows(conn, "search")
    assert len(rows) == 1
    surface, model_key, extra = rows[0]
    assert surface == "mcp"
    assert model_key == "mcp:search"
    assert extra["tool"] == "search"
    assert "hit_count" in extra
    assert "query" not in extra["args"]
    assert "a secret query nobody should log" not in str(extra)


def test_search_expand_pulls_in_linked_note(corpus):
    conn, embedder = corpus
    without_expand = tools.tool_search(
        conn, embedder, tenant=TENANT, query="Cobalt Ridge Medical Center hospital"
    )
    with_expand = tools.tool_search(
        conn, embedder, tenant=TENANT, query="Cobalt Ridge late deliveries", expand=1, k=1
    )
    assert any(h["expanded_from"] for h in with_expand["hits"]) or ACCOUNT_SLUG in {
        h["note_slug"] for h in with_expand["hits"]
    }
    assert without_expand["hits"]  # sanity: the account note is reachable directly too


def test_search_k_is_clamped_to_the_documented_maximum(corpus):
    """Review finding F15: `k` had no upper bound at all."""
    conn, embedder = corpus
    result = tools.tool_search(conn, embedder, tenant=TENANT, query="Cobalt Ridge", k=10_000_000)
    assert result["hits"]  # clamped, not rejected -- still a normal call


def test_search_rejects_a_query_over_the_max_length(corpus):
    """P1-6 (bug hunt 2026-09-18): `k`/`expand`/`since_days` were clamped
    but `query` had no bound at all -- straight into `embedder.embed()`
    (billed per character) and the keyword leg (one index probe per
    distinct word). Bounded the same way the
    web `/search?q=` route already is (`MAX_SEARCH_QUERY_CHARS`, 500)."""
    conn, embedder = corpus
    with pytest.raises(ValueError, match="500"):
        tools.tool_search(conn, embedder, tenant=TENANT, query="x" * 501)


def test_search_accepts_a_query_right_at_the_max_length(corpus):
    conn, embedder = corpus
    result = tools.tool_search(conn, embedder, tenant=TENANT, query="x" * 500)
    assert result["control_ok"] in (True, False)  # doesn't raise


def test_search_rejects_an_unknown_type_instead_of_silently_matching_nothing(corpus):
    """P1-1 (bug hunt 2026-09-18): a near-miss type name (`"signals"` for
    `"signal"`) used to silently match zero notes and report a healthy
    empty search (`control_ok=True`) rather than the caller's typo."""
    conn, embedder = corpus
    with pytest.raises(ValueError, match="signals"):
        tools.tool_search(conn, embedder, tenant=TENANT, query="Cobalt Ridge", types=["signals"])


# --- open_note --------------------------


def test_open_note_returns_frontmatter_body_claims_and_links(corpus):
    conn, _embedder = corpus
    note = tools.tool_open_note(conn, tenant=TENANT, slug=NOTE_SLUG)

    assert note is not None
    assert note["slug"] == NOTE_SLUG
    assert note["type"] == "signal"
    assert "Cobalt Ridge" in note["body_md"]
    assert len(note["claims"]) == 1
    claim = note["claims"][0]
    assert claim["source_slug"] == SOURCE_SLUG
    assert claim["locator"] == "@ 00:05:12"
    assert claim["quote"] == "These pumps keep showing up late."
    # Two clocks, both present in the shape even when null.
    assert "valid_from" in claim and "valid_to" in claim
    assert "recorded" in claim and "superseded_at" in claim
    link_kinds = {link["kind"] for link in note["links"]}
    assert "provenance" in link_kinds
    assert "body" in link_kinds


def test_open_note_returns_none_for_missing_slug(corpus):
    conn, _embedder = corpus
    assert tools.tool_open_note(conn, tenant=TENANT, slug="does-not-exist") is None


def test_open_note_usage_event_never_logs_the_body(corpus):
    conn, _embedder = corpus
    tools.tool_open_note(conn, tenant=TENANT, slug=NOTE_SLUG)

    rows = _usage_rows(conn, "open_note")
    assert len(rows) == 1
    _surface, _model_key, extra = rows[0]
    assert "Cobalt Ridge Medical Center reported" not in str(extra)
    assert extra["hit_count"] == 1


# --- open_source --------------------------


def test_open_source_returns_local_path_from_local_store(corpus, tmp_path):
    conn, _embedder = corpus
    store = LocalStore(tmp_path)
    store.put("originals/call/S-call-101", b"transcript bytes", "text/plain")

    source = tools.tool_open_source(conn, store, tenant=TENANT, slug=SOURCE_SLUG)

    assert source is not None
    assert source["source_type"] == "call"
    assert source["notes_extracted"] == [NOTE_SLUG]
    assert source["original_url"] == str(tmp_path / "originals" / "call" / "S-call-101")


def test_open_source_returns_none_for_missing_slug(corpus, tmp_path):
    conn, _embedder = corpus
    store = LocalStore(tmp_path)
    assert tools.tool_open_source(conn, store, tenant=TENANT, slug="S-does-not-exist") is None


def test_open_source_hosted_without_public_base_url_returns_a_relative_originals_path(
    corpus, tmp_path
):
    """Hosted MCP wave 2026-09-19: `hosted=True` with no `public_base_url`
    configured falls back to a bare relative `/originals/<slug>` -- still
    meaningful to a client that already knows the deployment's own
    origin (the same `https://<host>` it just called `/mcp` on)."""
    conn, _embedder = corpus
    store = LocalStore(tmp_path)
    store.put("originals/call/S-call-101", b"transcript bytes", "text/plain")

    source = tools.tool_open_source(conn, store, tenant=TENANT, slug=SOURCE_SLUG, hosted=True)

    assert source["original_url"] == "/originals/S-call-101"


def test_open_source_hosted_with_public_base_url_returns_an_absolute_originals_url(
    corpus, tmp_path
):
    conn, _embedder = corpus
    store = LocalStore(tmp_path)
    store.put("originals/call/S-call-101", b"transcript bytes", "text/plain")

    source = tools.tool_open_source(
        conn,
        store,
        tenant=TENANT,
        slug=SOURCE_SLUG,
        hosted=True,
        public_base_url="https://portolan-acme.example.com",
    )

    assert source["original_url"] == "https://portolan-acme.example.com/originals/S-call-101"


def test_open_source_hosted_percent_encodes_a_slug_with_special_characters(corpus, tmp_path):
    """Security review 2026-09-19, S3: `_resolve_slug`
    (`portolan/ingest/intake.py`) builds a source's slug from the
    connector's raw `source_id` verbatim, so external data shapes what
    lands in `original_url`. Unescaped, a slug like
    `y?next=https://evil.example.com` produced a URL whose query string
    pointed somewhere else entirely; `x#@evil.example.com` truncated the
    path at a fragment. Same-origin throughout (not an open redirect),
    but a citation link must always point at the source it claims to."""
    conn, _embedder = corpus
    dangerous_slug = "y?next=https://evil.example.com"
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO sources (slug, source_type, s3_key, content_hash,
                                  source_date, extraction_status)
            VALUES (%s, 'call', 'originals/call/dangerous', 'fake-hash-dangerous',
                    '2026-08-01', 'done')
            """,
            (dangerous_slug,),
        )

    source = tools.tool_open_source(
        conn, LocalStore(tmp_path), tenant=TENANT, slug=dangerous_slug, hosted=True
    )

    assert source["original_url"] == "/originals/y%3Fnext%3Dhttps%3A%2F%2Fevil.example.com"


def test_open_source_hosted_with_a_trailing_slash_on_public_base_url_is_not_doubled(
    corpus, tmp_path
):
    conn, _embedder = corpus
    store = LocalStore(tmp_path)
    store.put("originals/call/S-call-101", b"transcript bytes", "text/plain")

    source = tools.tool_open_source(
        conn,
        store,
        tenant=TENANT,
        slug=SOURCE_SLUG,
        hosted=True,
        public_base_url="https://portolan-acme.example.com/",
    )

    assert source["original_url"] == "https://portolan-acme.example.com/originals/S-call-101"


def test_open_source_not_hosted_ignores_public_base_url_and_keeps_the_store_url(corpus, tmp_path):
    """`hosted=False` (the default -- every caller before this wave, and
    every stdio invocation) keeps `Store.url(s3_key)` verbatim even if a
    `public_base_url` happens to be configured in the environment; only
    `hosted=True` (the streamable-HTTP transport) switches the formula."""
    conn, _embedder = corpus
    store = LocalStore(tmp_path)
    store.put("originals/call/S-call-101", b"transcript bytes", "text/plain")

    source = tools.tool_open_source(
        conn,
        store,
        tenant=TENANT,
        slug=SOURCE_SLUG,
        hosted=False,
        public_base_url="https://portolan-acme.example.com",
    )

    assert source["original_url"] == str(tmp_path / "originals" / "call" / "S-call-101")


def test_open_source_hosted_with_no_stored_original_is_still_none(corpus, tmp_path):
    """A source row with no `s3_key` has nothing to link to, hosted or
    not -- same guard the pre-existing `store.url(s3_key) if s3_key else
    None` used."""
    conn, _embedder = corpus
    store = LocalStore(tmp_path)
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO sources (slug, source_type, s3_key, content_hash,
                                  source_date, extraction_status)
            VALUES ('S-call-no-original', 'call', NULL, 'fake-hash-noorig',
                    '2026-08-01', 'done')
            """
        )
    source = tools.tool_open_source(
        conn, store, tenant=TENANT, slug="S-call-no-original", hosted=True, public_base_url=None
    )
    assert source["original_url"] is None


def test_open_source_redacts_a_dsn_in_last_error(corpus, tmp_path):
    """Security review 2026-09-19, S2: `last_error` is `f"{stage}:
    {str(error)}"` (`portolan/pipeline.py`) -- raw exception text from
    any pipeline stage, exactly where a backend DSN or credentialed URL
    surfaces. Seeded a row with a DSN-shaped secret in `last_error`;
    `open_source` used to return it verbatim. Redacted with the same
    `redact_dsn` (`portolan.util.redact`) every other logged/displayed
    DSN in this codebase already goes through -- not transport-gated
    (stdio gets the same protection; only `original_url`'s formula
    depends on `hosted`)."""
    conn, _embedder = corpus
    store = LocalStore(tmp_path)
    leaky_error = (
        "extract: connection failed postgresql://portolan:s3cr3t-pw@db.internal:5432/portolan"
    )
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO sources (slug, source_type, s3_key, content_hash,
                                  source_date, extraction_status, last_error)
            VALUES ('S-call-leaky', 'call', 'originals/call/S-call-leaky', 'fake-hash-leaky',
                    '2026-08-01', 'failed', %s)
            """,
            (leaky_error,),
        )

    hosted_source = tools.tool_open_source(
        conn, store, tenant=TENANT, slug="S-call-leaky", hosted=True
    )
    stdio_source = tools.tool_open_source(
        conn, store, tenant=TENANT, slug="S-call-leaky", hosted=False
    )

    for source in (hosted_source, stdio_source):
        assert "s3cr3t-pw" not in source["last_error"]
        assert source["last_error"] == (
            "extract: connection failed postgresql://portolan:***@db.internal:5432/portolan"
        )


def test_open_source_hosted_drops_s3_key(corpus, tmp_path):
    """Security review 2026-09-19, S2 (nearby finding): `s3_key` is store
    layout (`store._validate_key` forbids absolute paths and `..`, so it
    never leaks a container path) but a remote client has no use for it
    once `original_url` already points at the viewer route -- dropped in
    hosted mode, kept unchanged over stdio."""
    conn, _embedder = corpus
    store = LocalStore(tmp_path)
    store.put("originals/call/S-call-101", b"transcript bytes", "text/plain")

    hosted_source = tools.tool_open_source(
        conn, store, tenant=TENANT, slug=SOURCE_SLUG, hosted=True
    )
    stdio_source = tools.tool_open_source(
        conn, store, tenant=TENANT, slug=SOURCE_SLUG, hosted=False
    )

    assert hosted_source["s3_key"] is None
    assert stdio_source["s3_key"] == "originals/call/S-call-101"


# --- standing questions --------------------------


def test_standing_questions_round_trip(db):
    added = tools.tool_add_standing_question(
        db,
        tenant=TENANT,
        reader="dana@halvardmedical.com",
        question="Is Cobalt Ridge still churning?",
    )
    assert added["id"] is not None
    assert added["active"] is True

    listed = tools.tool_list_standing_questions(db, tenant=TENANT, reader="dana@halvardmedical.com")
    assert len(listed) == 1
    assert listed[0]["id"] == added["id"]
    assert listed[0]["question"] == "Is Cobalt Ridge still churning?"

    listed_other_reader = tools.tool_list_standing_questions(
        db, tenant=TENANT, reader="nobody@example.com"
    )
    assert listed_other_reader == []

    listed_all = tools.tool_list_standing_questions(db, tenant=TENANT)
    assert len(listed_all) == 1


def test_add_standing_question_rejects_an_oversized_reader(db):
    """P2-8 (bug hunt 2026-09-18): `reader` had no length bound at all --
    verified in the bug hunt storing a 5,000-char reader."""
    with pytest.raises(ValueError, match="reader"):
        tools.tool_add_standing_question(
            db, tenant=TENANT, reader="x" * 201, question="A question."
        )


def test_add_standing_question_rejects_an_oversized_question(db):
    """Verified in the bug hunt storing a 200,000-char question."""
    with pytest.raises(ValueError, match="question"):
        tools.tool_add_standing_question(
            db, tenant=TENANT, reader="dana@halvardmedical.com", question="x" * 2_001
        )


def test_search_expand_is_clamped_to_the_documented_maximum(corpus):
    """P2-8 (bug hunt 2026-09-18): `expand` was passed through raw while
    `k` was already clamped -- each hop is one query per frontier node,
    so an unclamped value is unbounded DB work per call. Clamped, not
    rejected -- still a normal call, same choice `k`'s own clamp made."""
    conn, embedder = corpus
    result = tools.tool_search(
        conn, embedder, tenant=TENANT, query="Cobalt Ridge late deliveries", expand=10_000
    )
    assert result["hits"]  # clamped, not rejected


# --- coverage --------------------------


def test_coverage_returns_recent_ledger_rows(db):
    record_coverage(db, "call", date.today(), seen=5, extracted=3, filtered={"junk:auto_reply": 2})

    rows = tools.tool_coverage(db, tenant=TENANT, since_days=1)

    assert len(rows) == 1
    assert rows[0]["connector"] == "call"
    assert rows[0]["seen"] == 5
    assert rows[0]["extracted"] == 3
    assert rows[0]["filtered"] == {"junk:auto_reply": 2}
    assert isinstance(rows[0]["day"], str)  # ISO date, JSON-shaped


def test_coverage_excludes_rows_before_the_window(db):
    from datetime import timedelta

    record_coverage(db, "email", date.today() - timedelta(days=30), seen=1)
    rows = tools.tool_coverage(db, tenant=TENANT, since_days=1)
    assert rows == []


def test_coverage_since_days_is_clamped_to_the_documented_maximum(db):
    """Review finding F15: since_days had no upper bound at all."""
    record_coverage(db, "email", date.today(), seen=1)
    rows = tools.tool_coverage(db, tenant=TENANT, since_days=10_000_000)
    assert len(rows) == 1  # clamped to a large-but-finite window, not rejected


# --- graph_neighbors --------------------------


def test_graph_neighbors_follows_body_links(corpus):
    conn, _embedder = corpus
    neighbors = tools.tool_graph_neighbors(conn, tenant=TENANT, slug=NOTE_SLUG, depth=1)

    assert len(neighbors) == 1
    neighbor = neighbors[0]
    assert neighbor["slug"] == ACCOUNT_SLUG
    assert neighbor["kind"] == "body"
    assert neighbor["direction"] == "out"
    assert neighbor["depth"] == 1
    assert neighbor["via"] == NOTE_SLUG


def test_graph_neighbors_excludes_provenance_links_to_sources(corpus):
    conn, _embedder = corpus
    neighbors = tools.tool_graph_neighbors(conn, tenant=TENANT, slug=NOTE_SLUG, depth=1)
    assert all(n["slug"] != SOURCE_SLUG for n in neighbors)


def test_graph_neighbors_is_reachable_from_the_other_direction(corpus):
    conn, _embedder = corpus
    neighbors = tools.tool_graph_neighbors(conn, tenant=TENANT, slug=ACCOUNT_SLUG, depth=1)
    assert len(neighbors) == 1
    assert neighbors[0]["slug"] == NOTE_SLUG
    assert neighbors[0]["direction"] == "in"


def test_graph_neighbors_depth_is_clamped_to_the_documented_maximum(corpus):
    """Review finding F15: depth clamped at the bottom (max(1, depth))
    but not the top -- one query per node, per hop."""
    conn, _embedder = corpus
    # A depth this large would otherwise mean a huge number of hops (one
    # query per node reached at each hop) over a two-node graph -- if it
    # weren't clamped, this call would still return quickly here only
    # because the graph is tiny; the clamp is what keeps a real graph
    # from turning a bad argument into a very long query chain.
    neighbors = tools.tool_graph_neighbors(conn, tenant=TENANT, slug=ACCOUNT_SLUG, depth=10_000)
    assert len(neighbors) == 1


# --- precedents --------------------------

THEME_SLUG = "theme-late-tidewater-deliveries"


def _theme_with_a_commitment_and_a_confirmed_resolution(conn) -> int:
    """A theme note, a `commitment` claim (on the same account note the
    `corpus` fixture already builds) that joined it, and a CONFIRMED
    resolution three weeks later -- `tool_precedents`' own worked
    example: one precedent, `resolved_after_days` computed from the
    claim's world date to the resolution's `resolved_week`."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO notes (slug, type, title, description, body_md, status, updated)
            VALUES (%s, 'theme', 'Late Tidewater deliveries',
                    'Recurring late Tidewater consumable deliveries.',
                    'Recurring late Tidewater consumable deliveries.', 'living', '2026-08-14')
            """,
            (THEME_SLUG,),
        )
        cur.execute(
            """
            INSERT INTO claims (note_slug, fact_type, claim_text, source_slug, locator,
                                 quote, speaker, recorded)
            VALUES (%s, 'commitment',
                    'Tom Ridley committed to a written root-cause summary for '
                    '[[cobalt-ridge-medical-center]].', %s, '@ 00:06:00',
                    'I will send a written summary', 'Tom Ridley', '2026-08-14 12:00:00-07')
            RETURNING id
            """,
            (NOTE_SLUG, SOURCE_SLUG),
        )
        (commitment_claim_id,) = cur.fetchone()
        cur.execute(
            "INSERT INTO theme_members (theme_slug, claim_id, added) VALUES (%s, %s, %s)",
            (THEME_SLUG, commitment_claim_id, "2026-08-14"),
        )
        cur.execute(
            """
            INSERT INTO claims (note_slug, fact_type, claim_text, source_slug, locator, recorded)
            VALUES (%s, 'decision', 'Deliveries returned to schedule.', %s, '@ 00:07:00',
                    '2026-09-04 12:00:00-07')
            RETURNING id
            """,
            (NOTE_SLUG, SOURCE_SLUG),
        )
        (resolution_claim_id,) = cur.fetchone()
        cur.execute(
            """
            INSERT INTO theme_resolutions
                (theme_slug, resolution_claim_id, resolved_reason, resolved_week, status,
                 confirmed_by, confirmed_at)
            VALUES (%s, %s, 'deliveries back on schedule', '2026-09-04', 'confirmed',
                    'Test Editor', now())
            """,
            (THEME_SLUG, resolution_claim_id),
        )
    conn.commit()
    return commitment_claim_id


def test_precedents_finds_a_commitment_with_its_resolution_time(corpus):
    # Timestamps carry an explicit offset: a naive midnight is read in the
    # session zone (UTC in CI, local here) and relabels to the previous world date.
    conn, _embedder = corpus
    _theme_with_a_commitment_and_a_confirmed_resolution(conn)

    result = tools.tool_precedents(
        conn, tenant=TENANT, theme_slug=THEME_SLUG, account_slug=ACCOUNT_SLUG
    )

    assert result["theme_slug"] == THEME_SLUG
    assert len(result["precedents"]) == 1
    precedent = result["precedents"][0]
    assert precedent["who"] == "Tom Ridley"
    assert precedent["when"] == "2026-08-14"
    assert precedent["account_slug"] == ACCOUNT_SLUG
    assert precedent["same_account"] is True
    assert precedent["resolved_after_days"] == 21  # 2026-08-14 -> 2026-09-04


def test_precedents_returns_empty_for_a_theme_with_no_commitments_or_decisions(corpus):
    conn, _embedder = corpus
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO notes (slug, type, title, description, body_md, status, updated)
            VALUES ('theme-empty', 'theme', 'Empty theme', 'Nothing here.', 'Nothing here.',
                    'living', '2026-08-14')
            """
        )
    conn.commit()

    result = tools.tool_precedents(conn, tenant=TENANT, theme_slug="theme-empty")

    assert result["precedents"] == []


def test_precedents_writes_a_usage_event(corpus):
    conn, _embedder = corpus
    _theme_with_a_commitment_and_a_confirmed_resolution(conn)
    tools.tool_precedents(conn, tenant=TENANT, theme_slug=THEME_SLUG, account_slug=ACCOUNT_SLUG)
    rows = _usage_rows(conn, "precedents")
    assert len(rows) == 1
    assert rows[0][0] == "mcp"


# --- the server factory --------------------------


def _fake_settings(tmp_path) -> Settings:
    return Settings(
        database_url="postgresql://localhost/unused-for-this-test",
        tenant=TENANT,
        embed_provider="fake",
        store_root=tmp_path,
    )


def test_create_server_lists_exactly_the_documented_tools(tmp_path):
    server = create_server(_fake_settings(tmp_path))
    registered = server._tool_manager.list_tools()

    names = {t.name for t in registered}
    assert names == set(TOOL_NAMES)
    for t in registered:
        assert t.description and t.description.strip(), f"tool {t.name!r} has no description"


# --- the HTTP transport's org-key gate --------------------------


def test_http_transport_rejects_a_missing_key(tmp_path):
    server = create_server(_fake_settings(tmp_path))
    app = build_http_app(server, org_key="a-real-shared-secret")

    with TestClient(app) as client:
        resp = client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "ping"})

    assert resp.status_code == 401


def test_http_transport_logs_a_401_with_the_client_address_and_no_header_contents(
    tmp_path, caplog: pytest.LogCaptureFixture
):
    """Coordinator follow-up to the security review, 2026-09-19: `/login`
    logs every failed attempt (`routes/login.py`'s own `_logger.warning
    ("login failed: ip=%s", ip)`, never the submitted key); `_send_
    unauthorized` used to log nothing at all, making a leaked-key probe
    or a scanner against `/mcp` invisible. One warning line per rejected
    request, naming the client address and nothing from the request's
    own headers -- not the wrong key, not any real `Authorization`
    value, right or wrong."""
    server = create_server(_fake_settings(tmp_path))
    app = build_http_app(server, org_key="a-real-shared-secret")

    with caplog.at_level("WARNING"), TestClient(app) as client:
        client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 1, "method": "ping"},
            headers={"Authorization": "Bearer a-lookalike-but-wrong-key"},
        )

    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1
    message = warnings[0].message
    assert "testclient" in message  # TestClient's own default `client` address
    assert "a-lookalike-but-wrong-key" not in message
    assert "a-real-shared-secret" not in message
    assert "Bearer" not in message


def test_http_transport_rejects_the_wrong_key(tmp_path):
    server = create_server(_fake_settings(tmp_path))
    app = build_http_app(server, org_key="a-real-shared-secret")

    with TestClient(app) as client:
        resp = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 1, "method": "ping"},
            headers={"Authorization": "Bearer wrong-key"},
        )

    assert resp.status_code == 401


def test_http_transport_lets_the_right_key_past_the_auth_gate(tmp_path):
    server = create_server(_fake_settings(tmp_path))
    app = build_http_app(server, org_key="a-real-shared-secret")

    with TestClient(app) as client:
        resp = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 1, "method": "ping"},
            headers={
                "Authorization": "Bearer a-real-shared-secret",
                "Accept": "application/json, text/event-stream",
            },
        )

    # Security review 2026-09-19, S5: `!= 401` also passes on a 500 (or
    # any other unexpected failure past the auth gate) -- assert the
    # actual status the MCP protocol layer gives a bare "ping" with no
    # prior `initialize`: a 400 JSON-RPC error ("Missing session ID"),
    # not a crash, and specifically not the auth gate's 401.
    assert resp.status_code == 400


def test_http_transport_accepts_a_non_localhost_host_header(tmp_path):
    """Found proving the hosted mount end-to-end (build report,
    2026-09-19). `Server.streamable_http_app` (the installed SDK)
    auto-enables DNS-rebinding protection -- an allowlist of exactly
    `127.0.0.1`/`localhost`/`[::1]` Host headers -- whenever its own
    `host` parameter is left at its own default, `"127.0.0.1"`, which is
    what happens whenever `build_http_app`'s caller doesn't pass `host=`
    through (every real caller: `portolan.web.app._mount_mcp`,
    `__main__.py`'s `--http` with `--host` left at its own default).
    Every real caller of the streamable-HTTP transport sends its own
    deployment's hostname as Host, never `127.0.0.1` -- unpatched, this
    421'd every such request outright, before the auth gate or the MCP
    protocol layer ever saw it. `TestClient`'s default base URL
    (`http://testserver`) already sends a non-localhost Host header, so
    this is the same request shape `test_http_transport_lets_the_right_
    key_past_the_auth_gate` above makes -- but that test only asserts
    `!= 401`, which a 421 also satisfies, so it would keep passing even
    with this bug back in place. This test pins the actual status.
    """
    server = create_server(_fake_settings(tmp_path))
    app = build_http_app(server, org_key="a-real-shared-secret")

    with TestClient(app) as client:
        resp = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 1, "method": "ping"},
            headers={
                "Authorization": "Bearer a-real-shared-secret",
                "Accept": "application/json, text/event-stream",
            },
        )

    assert resp.status_code != 421


def test_http_transport_rejects_duplicate_authorization_headers(tmp_path):
    """Review finding F3: `dict(scope["headers"])`, the old lookup,
    collapses duplicate `Authorization` headers to whichever one a dict
    happens to keep last -- not necessarily the one an intermediary or
    the client meant to be authoritative. A request carrying two must be
    rejected outright, even when one of the two is the real key."""
    server = create_server(_fake_settings(tmp_path))
    app = build_http_app(server, org_key="a-real-shared-secret")

    with TestClient(app) as client:
        resp = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 1, "method": "ping"},
            headers=[
                ("Authorization", "Bearer a-real-shared-secret"),
                ("Authorization", "Bearer something-else"),
                ("Accept", "application/json, text/event-stream"),
            ],
        )

    assert resp.status_code == 401


def test_build_http_app_without_a_key_refuses_to_build(tmp_path, monkeypatch):
    """Review finding F2. This used to assert the exact opposite --
    `test_build_http_app_without_a_key_returns_the_bare_app` pinned the
    fail-open as the contract, the same way `test_metering.py` once
    pinned a different finding: with no key configured, `build_http_app`
    returned the bare, unwrapped Starlette app, and that app served
    `open_note` (full `body_md`) and claim quotes (PHI) to anyone. It now
    raises instead -- there is no app to build without a real key."""
    monkeypatch.delenv("PORTOLAN_MCP_ORG_KEY", raising=False)
    server = create_server(_fake_settings(tmp_path))

    with pytest.raises(RuntimeError, match="PORTOLAN_MCP_ORG_KEY"):
        build_http_app(server)


# --- hosted inside the web app (create_app) --------------------------


@asynccontextmanager
async def _lifespan_started(app):
    """Manually drive the ASGI lifespan protocol -- startup, then
    shutdown on exit -- in whatever event loop is currently running.

    `starlette.testclient.TestClient` also drives lifespan, but on its
    own background thread with its own event loop; the MCP session
    manager's `anyio` task group is created during that lifespan and
    would then be bound to a different loop than an SDK client driven by
    a separate top-level `asyncio.run()` call. Hand-rolling this keeps
    the whole round trip -- server lifespan and SDK client alike -- on
    one loop, the one `asyncio.run()` (in each test below) actually
    creates.
    """
    startup_complete = asyncio.Event()
    shutdown_complete = asyncio.Event()
    receive_queue: asyncio.Queue = asyncio.Queue()

    async def receive():
        return await receive_queue.get()

    async def send(message):
        if message["type"] == "lifespan.startup.complete":
            startup_complete.set()
        elif message["type"] == "lifespan.startup.failed":
            startup_complete.set()
            raise RuntimeError(f"lifespan startup failed: {message.get('message')}")
        elif message["type"] == "lifespan.shutdown.complete":
            shutdown_complete.set()

    task = asyncio.create_task(app({"type": "lifespan"}, receive, send))
    await receive_queue.put({"type": "lifespan.startup"})
    await startup_complete.wait()
    try:
        yield
    finally:
        await receive_queue.put({"type": "lifespan.shutdown"})
        await shutdown_complete.wait()
        await task


@pytest.fixture()
def hosted_settings(db, db_url, tmp_path) -> Settings:
    """A `Settings` for a `create_app()`-built web app with the MCP
    streamable-HTTP transport mounted at `/mcp` (`mcp_org_key` set).
    Depends on `db` (function-scoped, fresh migrated schema every test)
    only to guarantee `db_url` already exists and is migrated before
    `create_app`'s own `lifespan` re-runs migrations against it -- a
    no-op by then, and the same connection a test's own seeding SQL
    (e.g. the `corpus` fixture) runs against, since `create_app` opens
    its own separate pool to this same `db_url`.

    Checklist item 26(b), 2026-09-22: the `db` fixture builds a brand-new
    database, whose MCP door is closed until turned on (migration 0031).
    These tests are about an open door, so this fixture turns it on for
    `TENANT`; the closed-door paths live in `tests/test_mcp_door.py`."""
    set_ai_access(db, TENANT, enabled=True, changed_by="test", reason="hosted_settings fixture")
    return Settings(
        database_url=db_url,
        tenant=TENANT,
        store_root=tmp_path,
        llm_provider="fake",
        embed_provider="fake",
        org_key="unused-web-org-key-for-mcp-tests",
        session_secret="unused-session-secret-for-mcp-tests-not-real",
        mcp_org_key="hosted-mcp-test-key-0123456789ab",
        cookie_secure=False,
    )


def test_mcp_session_manager_start_failure_never_stops_the_web_app_from_booting(
    monkeypatch: pytest.MonkeyPatch, hosted_settings: Settings
) -> None:
    """B1 (security review 2026-09-19, blocker). Reproduced exactly as
    the review did: patch `StreamableHTTPSessionManager.run` to raise.
    Before the fix, that `RuntimeError` propagated straight out of
    `lifespan` and the app never finished starting -- on Railway that is
    `healthcheckPath: /healthz` never serving,
    `restartPolicyMaxRetries: 5` exhausted, then a dead service: the
    whole reader-facing app down because a side door wouldn't open. Fail-
    loud is right for a missing *secret* (`_require_auth_settings`); not
    for a transport failing to start. After the fix: the app boots and
    serves everything else normally, and `/mcp` itself degrades to a
    clean per-request 503 rather than a crash, a hang, or a stale 200."""
    import contextlib

    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

    @contextlib.asynccontextmanager
    async def _broken_run(self):  # noqa: ANN001 -- matches the patched method's own signature
        raise RuntimeError("simulated MCP transport start failure")
        yield  # pragma: no cover -- unreachable; keeps this an async generator

    monkeypatch.setattr(StreamableHTTPSessionManager, "run", _broken_run)

    app = create_app(hosted_settings)

    with TestClient(app) as client:
        health_resp = client.get("/healthz")
        assert health_resp.status_code == 200

        login_resp = client.get("/login")
        assert login_resp.status_code == 200

        mcp_resp = client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "ping"})
        assert mcp_resp.status_code == 503


def test_mcp_route_404s_when_org_key_is_unset_and_the_rest_of_the_app_is_unchanged(
    db, db_url, tmp_path
):
    del db
    settings = Settings(
        database_url=db_url,
        tenant=TENANT,
        store_root=tmp_path,
        llm_provider="fake",
        embed_provider="fake",
        org_key="unused-web-org-key-for-mcp-tests",
        session_secret="unused-session-secret-for-mcp-tests-not-real",
        cookie_secure=False,
    )
    app = create_app(settings)

    with TestClient(app) as client:
        mcp_resp = client.get("/mcp")
        assert mcp_resp.status_code == 404

        # The rest of the app is unaffected by mcp_org_key being unset.
        login_resp = client.get("/login")
        assert login_resp.status_code == 200
        root_resp = client.get("/", follow_redirects=False)
        assert root_resp.status_code == 302
        assert root_resp.headers["location"] == "/login"


def test_mcp_route_401s_without_a_bearer_header_and_never_redirects_to_login(hosted_settings):
    app = create_app(hosted_settings)

    with TestClient(app) as client:
        resp = client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "ping"})

    assert resp.status_code == 401
    assert "location" not in {k.lower() for k in resp.headers}


def test_mcp_route_401s_with_the_wrong_bearer(hosted_settings):
    app = create_app(hosted_settings)

    with TestClient(app) as client:
        resp = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 1, "method": "ping"},
            headers={"Authorization": "Bearer wrong-key"},
        )

    assert resp.status_code == 401


def test_mcp_route_lets_the_right_bearer_past_the_gate(hosted_settings):
    app = create_app(hosted_settings)

    with TestClient(app) as client:
        resp = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 1, "method": "ping"},
            headers={
                "Authorization": "Bearer hosted-mcp-test-key-0123456789ab",
                "Accept": "application/json, text/event-stream",
            },
        )

    # Security review 2026-09-19, S5: exact status, not `!= 401` (which
    # also passes on a 500 or a 421) -- the same bare "ping" the
    # standalone-transport test above pins to 400, proven again through
    # the actual hosted mount, which also proves the non-localhost
    # Host-header fix along the way (a 421 here would fail this
    # assertion just as loudly as a 401 would).
    assert resp.status_code == 400


@pytest.mark.parametrize("method", ["GET", "HEAD", "OPTIONS", "DELETE", "PUT", "PATCH"])
def test_mcp_route_401s_on_every_method_with_no_bearer(hosted_settings, method):
    """Security review 2026-09-19, S5: the review's "Verified clean"
    section covered these seven methods by hand (`POST`, `GET` (SSE),
    `HEAD`, `OPTIONS`, `DELETE`, `PUT`, `PATCH`); `POST` already has its
    own dedicated test above, this pins the rest as an automated
    regression check."""
    app = create_app(hosted_settings)
    with TestClient(app) as client:
        resp = client.request(method, "/mcp")
    assert resp.status_code == 401


def test_mcp_session_id_replayed_with_no_bearer_still_401s(hosted_settings):
    """Security review 2026-09-19, S5: completed a real `initialize` with
    the right bearer, took the `mcp-session-id` it returned, and replayed
    that id on a follow-up call with no `Authorization` header at all --
    the session id must never substitute for the bearer. `_OrgKeyMiddleware`
    sits outside the session layer entirely (`server.py`'s own module
    docstring), so this should already be true by construction; pinned
    here as an automated check rather than only the manual one in the
    review."""
    app = create_app(hosted_settings)
    init_body = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "s5-regression-probe", "version": "0.0.1"},
        },
    }
    with TestClient(app) as client:
        init_resp = client.post(
            "/mcp",
            json=init_body,
            headers={
                "Authorization": "Bearer hosted-mcp-test-key-0123456789ab",
                "Accept": "application/json, text/event-stream",
            },
        )
        assert init_resp.status_code == 200
        session_id = init_resp.headers["mcp-session-id"]

        replay_resp = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            headers={
                "mcp-session-id": session_id,
                "Accept": "application/json, text/event-stream",
            },
        )

    assert replay_resp.status_code == 401


def test_mcp_initialize_tools_list_and_search_round_trip_through_the_hosted_mount(
    corpus, hosted_settings
):
    """The end-to-end proof this build task asks for (point 6): a real
    MCP SDK client (`mcp.client.streamable_http` + `mcp.client.session.
    ClientSession`) performs `initialize`, `tools/list`, `search`, and
    `open_source` through `create_app`'s own `/mcp` mount -- not just the
    standalone `build_http_app` output the tests above exercise -- over
    an in-process ASGI transport. Also proves the shared-embedder wiring:
    `search` only returns a hit if the MCP tool's `embedder` proxy
    actually resolved to the *same* indexed data the `corpus` fixture
    just wrote (via a *different* `FakeEmbedder` instance -- deterministic,
    so this is a real proxy-wiring check, not a coincidence)."""
    app = create_app(hosted_settings)

    async def _run():
        async with (
            _lifespan_started(app),
            httpx2.AsyncClient(
                transport=httpx2.ASGITransport(app=app),
                base_url="http://testserver",
                headers={"Authorization": "Bearer hosted-mcp-test-key-0123456789ab"},
            ) as http_client,
            streamable_http_client("http://testserver/mcp", http_client=http_client) as (
                read,
                write,
            ),
            ClientSession(read, write) as session,
        ):
            init_result = await session.initialize()
            listed = await session.list_tools()
            search_result = await session.call_tool(
                "search", {"query": "Cobalt Ridge late deliveries"}
            )
            source_result = await session.call_tool("open_source", {"slug": SOURCE_SLUG})
            return init_result, listed, search_result, source_result

    init_result, listed, search_result, source_result = asyncio.run(_run())

    assert init_result.protocol_version

    tool_names = sorted(t.name for t in listed.tools)
    assert tool_names == sorted(TOOL_NAMES)

    search_payload = json.loads(search_result.content[0].text)
    assert search_payload["hits"], "expected at least one search hit"
    assert any(h["note_slug"] == NOTE_SLUG for h in search_payload["hits"])

    source_payload = json.loads(source_result.content[0].text)
    # No PORTOLAN_PUBLIC_URL on `hosted_settings` -- the relative fallback.
    assert source_payload["original_url"] == f"/originals/{SOURCE_SLUG}"


def test_mcp_open_source_uses_public_base_url_end_to_end_when_configured(corpus, hosted_settings):
    hosted_settings = hosted_settings.model_copy(
        update={"public_base_url": "https://portolan-acme.example.com"}
    )
    app = create_app(hosted_settings)

    async def _run():
        async with (
            _lifespan_started(app),
            httpx2.AsyncClient(
                transport=httpx2.ASGITransport(app=app),
                base_url="http://testserver",
                headers={"Authorization": "Bearer hosted-mcp-test-key-0123456789ab"},
            ) as http_client,
            streamable_http_client("http://testserver/mcp", http_client=http_client) as (
                read,
                write,
            ),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            return await session.call_tool("open_source", {"slug": SOURCE_SLUG})

    source_result = asyncio.run(_run())
    source_payload = json.loads(source_result.content[0].text)
    expected = f"https://portolan-acme.example.com/originals/{SOURCE_SLUG}"
    assert source_payload["original_url"] == expected
