"""Tests for portolan/search/index.py.

DB-backed: uses the shared `db` fixture (tests/conftest.py), pointed at
DATABASE_URL=postgresql://localhost/portolan_test_search per the build
instructions. Uses FakeEmbedder throughout -- no network.
"""

from __future__ import annotations

import psycopg
import pytest

from portolan.search.embed import FakeEmbedder
from portolan.search.index import index_note, index_pending, reindex_all


def _insert_note(
    cur, slug: str, body_md: str, title: str = "A Note", ntype: str = "signal"
) -> None:
    cur.execute(
        """
        INSERT INTO notes (slug, type, title, description, body_md)
        VALUES (%s, %s, %s, 'A test note.', %s)
        """,
        (slug, ntype, title, body_md),
    )


def test_index_note_writes_chunks_and_state(db):
    embedder = FakeEmbedder()
    with db.cursor() as cur:
        _insert_note(
            cur, "ashgrove-medical-center", "Ashgrove had a late delivery.\n\nA second paragraph."
        )

    written = index_note(db, embedder, "ashgrove-medical-center")
    assert written > 0

    with db.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM chunks WHERE note_slug = %s", ("ashgrove-medical-center",)
        )
        assert cur.fetchone()[0] == written

        cur.execute(
            "SELECT body_hash FROM note_index_state WHERE note_slug = %s",
            ("ashgrove-medical-center",),
        )
        row = cur.fetchone()
        assert row is not None and row[0]

        # Every chunk got a real embedding, not nulls.
        cur.execute(
            "SELECT embedding IS NOT NULL FROM chunks WHERE note_slug = %s",
            ("ashgrove-medical-center",),
        )
        assert all(r[0] for r in cur.fetchall())


def test_index_note_chunk_zero_is_the_title_chunk(db):
    embedder = FakeEmbedder()
    with db.cursor() as cur:
        cur.execute(
            """
            INSERT INTO notes (slug, type, title, description, body_md)
            VALUES ('larkspur-memorial', 'signal', 'Larkspur Memorial Delays',
                    'A summary.', 'Body text here.')
            """
        )
    index_note(db, embedder, "larkspur-memorial")
    with db.cursor() as cur:
        cur.execute(
            "SELECT text FROM chunks WHERE note_slug = %s ORDER BY id LIMIT 1",
            ("larkspur-memorial",),
        )
        first_text = cur.fetchone()[0]
    assert "Larkspur Memorial Delays" in first_text
    assert "A summary." in first_text


def test_index_note_is_idempotent_second_call_writes_zero_chunks(db):
    embedder = FakeEmbedder()
    with db.cursor() as cur:
        _insert_note(
            cur, "cobalt-ridge", "Cobalt Ridge borrowed consumables from a neighbouring facility."
        )

    first = index_note(db, embedder, "cobalt-ridge")
    second = index_note(db, embedder, "cobalt-ridge")
    assert first > 0
    assert second == 0

    with db.cursor() as cur:
        cur.execute("SELECT count(*) FROM chunks WHERE note_slug = %s", ("cobalt-ridge",))
        assert cur.fetchone()[0] == first


def test_index_note_reindexes_when_body_changes(db):
    embedder = FakeEmbedder()
    with db.cursor() as cur:
        _insert_note(cur, "meadowbrook", "Original body about a delayed order.")
    index_note(db, embedder, "meadowbrook")

    with db.cursor() as cur:
        cur.execute(
            "UPDATE notes SET body_md = %s WHERE slug = %s",
            ("A completely different body now.", "meadowbrook"),
        )

    written = index_note(db, embedder, "meadowbrook")
    assert written > 0

    with db.cursor() as cur:
        cur.execute("SELECT text FROM chunks WHERE note_slug = %s ORDER BY id", ("meadowbrook",))
        texts = " ".join(r[0] for r in cur.fetchall())
    assert "completely different" in texts
    assert "Original body" not in texts


def test_index_note_raises_for_unknown_slug(db):
    embedder = FakeEmbedder()
    with pytest.raises(ValueError):
        index_note(db, embedder, "does-not-exist")


def test_reindex_all_covers_every_note_and_reports_counts(db):
    embedder = FakeEmbedder()
    with db.cursor() as cur:
        _insert_note(cur, "note-a", "Body A about late deliveries.")
        _insert_note(cur, "note-b", "Body B about a portal password issue.")

    report = reindex_all(db, embedder)
    assert report.notes_considered == 2
    assert report.notes_reindexed == 2
    assert report.notes_skipped == 0
    assert report.chunks_written > 0
    assert report.skipped_lock is False

    # Second full pass: both notes are unchanged now.
    report2 = reindex_all(db, embedder)
    assert report2.notes_considered == 2
    assert report2.notes_reindexed == 0
    assert report2.notes_skipped == 2
    assert report2.chunks_written == 0


def test_reindex_all_single_flight_skips_when_lock_already_held(db_url):
    embedder = FakeEmbedder()
    holder = psycopg.connect(db_url)
    holder.autocommit = True
    try:
        with holder.cursor() as cur:
            cur.execute("SELECT pg_try_advisory_lock(785412309)")
            assert cur.fetchone()[0] is True

        waiter = psycopg.connect(db_url)
        waiter.autocommit = True
        try:
            report = reindex_all(waiter, embedder)
            assert report.skipped_lock is True
            assert report.notes_considered == 0
        finally:
            waiter.close()
    finally:
        with holder.cursor() as cur:
            cur.execute("SELECT pg_advisory_unlock(785412309)")
        holder.close()


def test_index_pending_only_touches_new_or_changed_notes(db):
    embedder = FakeEmbedder()
    with db.cursor() as cur:
        _insert_note(cur, "note-a", "Body A about late deliveries.")
        _insert_note(cur, "note-b", "Body B about a portal password issue.")

    first = index_pending(db, embedder)
    assert first.notes_considered == 2
    assert first.notes_reindexed == 2

    # Nothing changed -- index_pending's SQL filter should find zero candidates.
    second = index_pending(db, embedder)
    assert second.notes_considered == 0
    assert second.notes_reindexed == 0
    assert second.chunks_written == 0

    with db.cursor() as cur:
        cur.execute(
            "UPDATE notes SET body_md = %s WHERE slug = 'note-a'",
            ("Body A, now about something else entirely.",),
        )
        _insert_note(cur, "note-c", "A brand new note.")

    third = index_pending(db, embedder)
    assert third.notes_considered == 2  # note-a (changed) + note-c (new)
    assert third.notes_reindexed == 2

    with db.cursor() as cur:
        cur.execute("SELECT text FROM chunks WHERE note_slug = 'note-a'")
        assert any("something else entirely" in r[0] for r in cur.fetchall())
        cur.execute("SELECT note_slug FROM note_index_state ORDER BY note_slug")
        assert [r[0] for r in cur.fetchall()] == ["note-a", "note-b", "note-c"]


def test_a_title_change_through_write_note_refreshes_the_title_chunk(db):
    """`write_note` renders the title and description into `body_md`'s
    frontmatter, so hashing `body_md` alone still catches a title change."""
    from datetime import UTC, datetime

    from portolan.notes import NoteRow, NoteStatus, NoteType, read_note, write_note

    now = datetime.now(UTC)
    write_note(
        db,
        NoteRow(
            slug="title-change-probe",
            type=NoteType.SIGNAL,
            title="Old Title",
            description="Probe note.",
            status=NoteStatus.LIVING,
            created=now,
            updated=now,
        ),
        [],
        changed_by="test",
        reason="title change probe",
    )
    embedder = FakeEmbedder()
    index_note(db, embedder, "title-change-probe")

    note, claims = read_note(db, "title-change-probe")
    note.title = "New Title"
    write_note(db, note, claims, changed_by="test", reason="rename")

    assert index_note(db, embedder, "title-change-probe") > 0
    with db.cursor() as cur:
        cur.execute(
            "SELECT text FROM chunks WHERE note_slug = 'title-change-probe' ORDER BY id LIMIT 1"
        )
        (title_chunk,) = cur.fetchone()
    assert title_chunk.startswith("New Title")
