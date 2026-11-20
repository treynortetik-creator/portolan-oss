"""Slash- and dot-joined words are separate keyword words (migration 0035).

Postgres's `english` parser reads "Link/Epic" as one file-path token and "Link.Epic" as
one host token, so before 0035 a note holding either never matched "epic" in the keyword
leg (01-spec/gbrain-borrow-knowledge-2026-09-23.md, gap 3). `portolan_search_text` splits
them on both the index side (`notes.search`, `chunks.search`) and the query side.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from portolan import db as db_module
from portolan.db import MIGRATIONS_DIR
from portolan.search.embed import FakeEmbedder
from portolan.search.index import reindex_all
from portolan.search.query import _fts_candidates, search

JIRA_BODY = "Ticket moved to Link/Epic PLAT-88 after triage; see Board.Roadmap for the date."


def _lexemes(conn, sql: str, text: str) -> set[str]:
    with conn.cursor() as cur:
        cur.execute(f"SELECT lexeme FROM unnest({sql})", (text,))
        return {row[0] for row in cur.fetchall()}


@pytest.fixture()
def jira_graph(db):
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO notes (slug, type, title, description, body_md) "
            "VALUES ('plat-88', 'signal', 'Triage outcome', '', %s)",
            (JIRA_BODY,),
        )
        cur.execute(
            "INSERT INTO notes (slug, type, title, description, body_md) "
            "VALUES ('unrelated', 'signal', 'Quarterly review', '', "
            "'Pricing came up at the quarterly review.')"
        )
    embedder = FakeEmbedder()
    reindex_all(db, embedder)
    return db, embedder


def test_normalizer_splits_slash_backslash_and_letter_dots_but_not_decimals(db):
    words = _lexemes(
        db,
        "to_tsvector('english', portolan_search_text(%s))",
        r"Link/Epic Jira\Board Board.Roadmap v2.1 costs 3.5 percent",
    )
    assert {"link", "epic", "jira", "board", "roadmap", "v2.1", "3.5"} <= words
    assert not {w for w in words if "/" in w or "\\" in w}


def test_note_with_slash_joined_word_is_found_by_its_second_half(jira_graph):
    conn, embedder = jira_graph
    ranked, chunks = _fts_candidates(conn, "epic", 10)
    assert ranked == ["plat-88"]
    assert "Link/Epic" in chunks["plat-88"][1]

    hit = {h.note_slug: h for h in search(conn, embedder, "epic", k=5).hits}["plat-88"]
    assert "fts" in hit.via


def test_note_with_dot_joined_word_is_found_by_either_half(jira_graph):
    conn, _ = jira_graph
    assert _fts_candidates(conn, "roadmap", 10)[0] == ["plat-88"]


def test_query_typed_with_a_slash_matches_the_split_words(jira_graph):
    conn, _ = jira_graph
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO notes (slug, type, title, description, body_md) "
            "VALUES ('spaced', 'signal', 'Spaced', '', 'The link to the epic was added.')"
        )
    ranked, _chunks = _fts_candidates(conn, "Link/Epic", 10)
    assert set(ranked) == {"plat-88", "spaced"}


def test_migration_0035_reindexes_rows_written_before_it(db, tmp_path: Path):
    """Rows already in the table when 0035 runs are rebuilt by the migration itself,
    and their content date is filled from their claims, without touching `updated`."""
    before = tmp_path / "migrations"
    before.mkdir()
    for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
        if path.name < "0035":
            shutil.copy(path, before / path.name)
    with db.cursor() as cur:
        cur.execute("DROP SCHEMA public CASCADE")
        cur.execute("CREATE SCHEMA public")
    db_module.apply_migrations(db, before)
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO sources (slug, source_type, source_date, extraction_status) "
            "VALUES ('S-1', 'ticket', '2026-03-02', 'done')"
        )
        cur.execute(
            "INSERT INTO notes (slug, type, title, description, body_md, updated) "
            "VALUES ('plat-88', 'signal', 'Triage outcome', '', %s, '2026-09-20T00:00:00Z')",
            (JIRA_BODY,),
        )
        cur.execute("INSERT INTO chunks (note_slug, text) VALUES ('plat-88', %s)", (JIRA_BODY,))
        cur.execute(
            "INSERT INTO claims (note_slug, fact_type, claim_text, source_slug, locator, "
            "recorded) VALUES ('plat-88', 'other', 'Moved to an epic.', 'S-1', 'L1', "
            "'2026-03-02T07:00:00Z')"
        )
        cur.execute("SELECT count(*) FROM notes WHERE search @@ 'epic'::tsquery")
        assert cur.fetchone()[0] == 0

    applied = db_module.apply_migrations(db)
    assert applied[0] == "0035_content_dates_and_split_words.sql"

    with db.cursor() as cur:
        cur.execute("SELECT count(*) FROM notes WHERE search @@ 'epic'::tsquery")
        assert cur.fetchone()[0] == 1
        cur.execute("SELECT count(*) FROM chunks WHERE search @@ 'epic'::tsquery")
        assert cur.fetchone()[0] == 1
        cur.execute(
            "SELECT content_date_source, content_date = '2026-03-02T07:00:00Z', "
            "updated = '2026-09-20T00:00:00Z' FROM notes WHERE slug = 'plat-88'"
        )
        assert cur.fetchone() == ("claims_recorded", True, True)
        cur.execute("SELECT portolan_refresh_content_dates()")
        assert cur.fetchone()[0] == 0
