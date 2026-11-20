"""Schema tests for portolan/db/migrations/0001_init.sql.

Covers: migrations are idempotent, every required table exists, pgvector
is installed, the two clocks on a claim can be set, a nearest-neighbour
vector query and a full-text query both find what they should, an invalid
usage_events.surface is rejected by its check constraint, and deleting a
source that a claim still points at is refused (spec sec 8/11: nothing
that would remove a claim or a source is allowed to cascade).
"""

from __future__ import annotations

import psycopg
import pytest

from portolan import db as db_module

REQUIRED_TABLES = {
    "sources",
    "notes",
    "note_versions",
    "claims",
    "links",
    "registry_entities",
    "registry_aliases",
    "review_queue",
    "chunks",
    "coverage_ledger",
    "usage_events",
    "edit_records",
    "actions",
    "changelog",
}


def test_migrations_are_idempotent(db):
    # The `db` fixture already applied migrations once when it rebuilt the
    # schema. Applying again must be a no-op, not an error.
    first = db_module.apply_migrations(db)
    assert first == []

    second = db_module.apply_migrations(db)
    assert second == []


def test_each_migration_file_is_idempotent_when_reapplied_directly(db):
    # Review finding F7: `apply_migrations` (above) skips a file it has
    # already recorded in `schema_migrations`, so re-calling it never
    # actually re-executes a file's SQL a second time and would not have
    # caught F7 -- 0001_init.sql's own header claimed "every statement
    # uses IF NOT EXISTS / OR REPLACE so this file can be re-applied
    # safely", which was false for its FK add (`ALTER TABLE claims ADD
    # CONSTRAINT ...`, no `IF NOT EXISTS` form in Postgres), verified by
    # applying the file twice directly with psql. This test does the same
    # thing every migration file, bypassing the ledger entirely: run each
    # file's raw SQL a second time against the already-migrated schema and
    # confirm none of them raises.
    for path in sorted(db_module.MIGRATIONS_DIR.glob("*.sql")):
        sql = path.read_text()
        with db.cursor() as cur:
            cur.execute(sql)


def test_all_required_tables_exist(db):
    with db.cursor() as cur:
        cur.execute(
            """
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public'
            """
        )
        existing = {row[0] for row in cur.fetchall()}

    missing = REQUIRED_TABLES - existing
    assert not missing, f"missing tables: {missing}"


def test_vector_extension_is_present(db):
    with db.cursor() as cur:
        cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        row = cur.fetchone()
    assert row is not None, "pgvector extension is not installed in this database"


def _insert_source(cur, slug: str = "mercy-qbr-2026-08-14") -> str:
    cur.execute(
        """
        INSERT INTO sources (slug, source_type, s3_key, content_hash,
                              source_date, extraction_status, schema_version)
        VALUES (%s, 'call', 's3://bucket/mercy-qbr.mp4', 'abc123',
                '2026-08-14', 'done', '0.1')
        """,
        (slug,),
    )
    return slug


def _insert_note(cur, slug: str = "mercy-health-springfield") -> str:
    cur.execute(
        """
        INSERT INTO notes (slug, type, title, description, crm_id, body_md)
        VALUES (%s, 'account', 'Ashgrove Medical Center',
                'A hospital account in the Midwest region.',
                'crm-00118', 'Ashgrove received three late deliveries in 2026-08.')
        """,
        (slug,),
    )
    return slug


def test_insert_source_note_claim_link_chunk_round_trip(db):
    with db.cursor() as cur:
        source_slug = _insert_source(cur)
        note_slug = _insert_note(cur)

        # A claim with both clocks set: world time (valid_from/valid_to)
        # and record time (recorded/superseded_at) -- spec sec 2.
        cur.execute(
            """
            INSERT INTO claims (note_slug, fact_type, claim_text, source_slug,
                                 locator, quote, speaker, valid_from, valid_to,
                                 recorded, superseded_at, sentiment, confidence)
            VALUES (%s, 'complaint',
                    'Ashgrove received three late Tidewater T2 deliveries in 2026-08.',
                    %s, '00:14:32', 'we had three late deliveries last month and nobody called us',
                    'contact:maria-chen', '2026-08-01', '2026-08-31',
                    '2026-08-14 10:00:00+00', '2026-09-01 09:00:00+00', -2, 0.9)
            RETURNING id
            """,
            (note_slug, source_slug),
        )
        claim_id = cur.fetchone()[0]
        assert claim_id is not None

        cur.execute(
            """
            INSERT INTO links (from_slug, to_slug, kind)
            VALUES (%s, %s, 'provenance')
            """,
            (note_slug, source_slug),
        )

        embedding = [0.001] * 1024
        cur.execute(
            """
            INSERT INTO chunks (note_slug, heading, text, token_count, embedding)
            VALUES (%s, 'Deliveries', 'Ashgrove received three late deliveries in 2026-08.', 12, %s)
            RETURNING id
            """,
            (note_slug, embedding),
        )
        chunk_id = cur.fetchone()[0]
        assert chunk_id is not None


def test_nearest_neighbour_query_finds_the_inserted_chunk(db):
    with db.cursor() as cur:
        note_slug = _insert_note(cur)
        # Cosine distance only sees direction, not magnitude, so the decoy
        # must point a genuinely different way, not just be scaled -- an
        # orthogonal vector (all weight on the last dimension instead of
        # the first) guarantees it is farther from `target` than `target`
        # is from itself.
        target = [1.0] + [0.0] * 1023
        decoy = [0.0] * 1023 + [1.0]

        cur.execute(
            "INSERT INTO chunks (note_slug, text, embedding) "
            "VALUES (%s, 'target chunk', %s) RETURNING id",
            (note_slug, target),
        )
        target_id = cur.fetchone()[0]
        cur.execute(
            "INSERT INTO chunks (note_slug, text, embedding) VALUES (%s, 'decoy chunk', %s)",
            (note_slug, decoy),
        )

        cur.execute(
            "SELECT id FROM chunks ORDER BY embedding <=> %s::vector LIMIT 1",
            (target,),
        )
        nearest_id = cur.fetchone()[0]

    assert nearest_id == target_id


def test_full_text_query_finds_the_note(db):
    with db.cursor() as cur:
        note_slug = _insert_note(cur)

        cur.execute(
            "SELECT slug FROM notes WHERE search @@ plainto_tsquery('english', 'late deliveries')"
        )
        rows = [row[0] for row in cur.fetchall()]

    assert note_slug in rows


def test_usage_events_rejects_invalid_surface(db):
    with pytest.raises(psycopg.errors.CheckViolation), db.cursor() as cur:
        cur.execute(
            """
                INSERT INTO usage_events (tenant, surface, model_key, input_tokens, output_tokens)
                VALUES ('acme', 'not-a-real-surface', 'nova-lite', 100, 50)
                """
        )


def test_deleting_a_source_referenced_by_a_claim_is_refused(db):
    with db.cursor() as cur:
        source_slug = _insert_source(cur)
        note_slug = _insert_note(cur)
        cur.execute(
            """
            INSERT INTO claims (note_slug, fact_type, claim_text, source_slug, locator)
            VALUES (%s, 'other', 'A durable fact that fits none of the base types.', %s, '00:01:00')
            """,
            (note_slug, source_slug),
        )

    with pytest.raises(psycopg.errors.ForeignKeyViolation), db.cursor() as cur:
        cur.execute("DELETE FROM sources WHERE slug = %s", (source_slug,))
