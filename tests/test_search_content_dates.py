"""A note's content date (migration 0035) drives `search(since=)`, and recency when
`search_recency_content_date` is on (off by default).

`notes.content_date` is the newest `recorded` date among the note's live claims (not
superseded, not archived), else the note's last write, marked by `content_date_source`.
The case that motivated it: a graph loaded in one sitting, where every note's
`updated` is the load time (01-spec/bench-graph-retrieval-2026-09-23.md on
feature/graph-retrieval: `since=` on `updated` cut recall from 74.4% to 28.6%).

DB-backed, FakeEmbedder. Each note's `updated` is today's write, as after a backfill.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from portolan.mcp_server.tools import tool_search
from portolan.search import query as query_module
from portolan.search.embed import FakeEmbedder
from portolan.search.index import reindex_all
from portolan.search.query import search

SOURCE_OLD = "S-call-old"
SOURCE_NEW = "S-call-new"
OLD = datetime(2026, 1, 10, 7, tzinfo=UTC)
NEW = datetime(2026, 9, 10, 7, tzinfo=UTC)

# Same words, so text relevance ties and only the date can separate them.
BODY = "Kestrel Medical Group raised infusion pump downtime at the quarterly review."


def _note(cur, slug: str, title: str, note_type: str = "signal", crm_id: str | None = None):
    cur.execute(
        "INSERT INTO notes (slug, type, title, description, body_md, crm_id) "
        "VALUES (%s, %s, %s, '', %s, %s)",
        (slug, note_type, title, BODY, crm_id),
    )


def _claim(cur, note_slug: str, source_slug: str, recorded: datetime) -> int:
    cur.execute(
        """
        INSERT INTO claims (note_slug, fact_type, claim_text, source_slug, locator, recorded)
        VALUES (%s, 'complaint', 'Infusion pump downtime.', %s, 'L1', %s)
        RETURNING id
        """,
        (note_slug, source_slug, recorded),
    )
    return cur.fetchone()[0]


def _content(conn, slug: str) -> tuple[datetime | None, datetime, str, datetime]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT content_date_claims, content_date, content_date_source, updated "
            "FROM notes WHERE slug = %s",
            (slug,),
        )
        return cur.fetchone()


@pytest.fixture()
def graph(db):
    """Three notes written today: one whose only claim was recorded 2026-01-10, one
    recorded 2026-09-10, and one account page with no claims."""
    with db.cursor() as cur:
        for slug, day in ((SOURCE_OLD, "2026-01-10"), (SOURCE_NEW, "2026-09-10")):
            cur.execute(
                "INSERT INTO sources (slug, source_type, source_date, extraction_status) "
                "VALUES (%s, 'call', %s, 'done')",
                (slug, day),
            )
        _note(cur, "downtime-january", "Pump downtime complaint")
        _note(cur, "downtime-september", "Pump downtime complaint")
        _note(cur, "kestrel-medical-group", "Kestrel Medical Group", "account", "A-1")
        _claim(cur, "downtime-january", SOURCE_OLD, OLD)
        _claim(cur, "downtime-september", SOURCE_NEW, NEW)
    embedder = FakeEmbedder()
    reindex_all(db, embedder)
    return db, embedder


def test_content_date_is_the_newest_live_claim_and_says_where_it_came_from(graph):
    conn, _ = graph
    claims_date, content_date, source, _updated = _content(conn, "downtime-january")
    assert claims_date == OLD
    assert content_date == OLD
    assert source == "claims_recorded"

    _claims, content_date, source, updated = _content(conn, "kestrel-medical-group")
    assert source == "note_updated"
    assert content_date == updated


def test_superseding_or_archiving_a_claim_moves_the_content_date(graph):
    conn, _ = graph
    with conn.cursor() as cur:
        newer = _claim(cur, "downtime-january", SOURCE_NEW, NEW)
    assert _content(conn, "downtime-january")[1] == NEW

    with conn.cursor() as cur:
        cur.execute("UPDATE claims SET superseded_at = now() WHERE id = %s", (newer,))
    assert _content(conn, "downtime-january")[1] == OLD

    with conn.cursor() as cur:
        cur.execute("UPDATE claims SET archived = true WHERE note_slug = 'downtime-january'")
    claims_date, content_date, source, updated = _content(conn, "downtime-january")
    assert claims_date is None
    assert source == "note_updated"
    assert content_date == updated


def test_a_content_date_refresh_does_not_count_as_a_write_of_the_note(graph):
    conn, _ = graph
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE notes SET updated = '2026-02-01T00:00:00Z' WHERE slug = 'downtime-january'"
        )
        _claim(cur, "downtime-january", SOURCE_NEW, NEW)
    _claims, content_date, _source, updated = _content(conn, "downtime-january")
    assert content_date == NEW
    assert updated == datetime(2026, 2, 1, tzinfo=UTC)

    # Any other change to the row is still a write.
    with conn.cursor() as cur:
        cur.execute("UPDATE notes SET title = 'Renamed' WHERE slug = 'downtime-january'")
    assert _content(conn, "downtime-january")[3] > datetime(2026, 2, 1, tzinfo=UTC)


def test_since_reads_the_claim_date_not_the_write_time(graph):
    conn, embedder = graph
    result = search(conn, embedder, "infusion pump downtime", k=10, since=date(2026, 9, 1))
    slugs = {h.note_slug for h in result.hits}
    assert "downtime-september" in slugs
    assert "downtime-january" not in slugs
    # No claims: the write-time fallback, and today's write is after the cutoff.
    assert "kestrel-medical-group" in slugs


def test_since_is_a_calendar_day_in_the_deployment_zone(graph, monkeypatch):
    """A claim recorded at midnight Phoenix on 2026-09-10 is 07:00 UTC; `since` that
    day keeps it and `since` the next day does not, whatever the session TimeZone."""
    conn, embedder = graph
    monkeypatch.setenv("PORTOLAN_TIMEZONE", "America/Phoenix")
    on_day = search(conn, embedder, "infusion pump downtime", k=10, since=date(2026, 9, 10))
    next_day = search(conn, embedder, "infusion pump downtime", k=10, since=date(2026, 9, 11))
    assert "downtime-september" in {h.note_slug for h in on_day.hits}
    assert "downtime-september" not in {h.note_slug for h in next_day.hits}


def _written_in_january(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE notes SET updated = '2026-01-01T19:00:00Z' WHERE slug = 'downtime-september'"
        )


def test_recency_on_ranks_the_newer_claim_first_although_it_was_written_earlier(graph):
    conn, embedder = graph
    _written_in_january(conn)
    result = search(conn, embedder, "pump downtime complaint", k=10, recency_content_date=True)
    slugs = [h.note_slug for h in result.hits]
    assert slugs.index("downtime-september") < slugs.index("downtime-january")
    hit = {h.note_slug: h for h in result.hits}["downtime-september"]
    assert hit.content_date == date(2026, 9, 10)
    assert hit.content_date_source == "claims_recorded"
    assert hit.updated == date(2026, 1, 1)


def test_recency_off_by_default_ranks_by_last_write_as_before(graph):
    """The setting defaults off (`search_recency_content_date`): recency reads
    `notes.updated`, so the note written in January ranks below today's write."""
    conn, embedder = graph
    _written_in_january(conn)
    default = search(conn, embedder, "pump downtime complaint", k=10)
    off = search(conn, embedder, "pump downtime complaint", k=10, recency_content_date=False)
    slugs = [h.note_slug for h in default.hits]
    assert slugs == [h.note_slug for h in off.hits]
    assert slugs.index("downtime-january") < slugs.index("downtime-september")


def test_the_setting_turns_content_date_recency_on(graph, monkeypatch):
    from portolan import config

    conn, embedder = graph
    _written_in_january(conn)
    monkeypatch.setattr(config.settings, "search_recency_content_date", True)
    slugs = [h.note_slug for h in search(conn, embedder, "pump downtime complaint", k=10).hits]
    assert slugs.index("downtime-september") < slugs.index("downtime-january")


def test_recency_on_gives_a_note_without_claims_the_floor_not_its_write_time(graph):
    conn, _ = graph
    meta = query_module._load_notes_meta(conn, ["kestrel-medical-group", "downtime-september"])
    now = datetime(2026, 9, 25, tzinfo=UTC)
    on = {"now": now, "content_date": True}
    assert query_module._note_recency(meta["kestrel-medical-group"], **on) == pytest.approx(
        query_module.RECENCY_FLOOR
    )
    assert query_module._note_recency(meta["downtime-september"], **on) > 0.99
    # Off: today's write gives the claimless note the full weight, as before.
    assert (
        query_module._note_recency(meta["kestrel-medical-group"], now=now, content_date=False)
        > 0.99
    )


def test_mcp_search_reports_the_content_date_and_filters_on_it(graph):
    conn, embedder = graph
    out = tool_search(
        conn, embedder, tenant="dev", query="infusion pump downtime", since=date(2026, 9, 1)
    )
    by_slug = {h["note_slug"]: h for h in out["hits"]}
    assert "downtime-january" not in by_slug
    assert by_slug["downtime-september"]["content_date"] == "2026-09-10"
    assert by_slug["downtime-september"]["content_date_source"] == "claims_recorded"
    assert by_slug["kestrel-medical-group"]["content_date_source"] == "note_updated"
