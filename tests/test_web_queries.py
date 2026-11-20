"""Tests for portolan/web/queries.py -- small, shared read-queries.

Security review 2026-09-19 (S2, "nearby finding"): `not_ingested_sources`
returns `last_error` verbatim, the same raw-exception-text leak
`portolan.mcp_server.tools.tool_open_source` had -- fixed there with
`portolan.util.redact.redact_dsn`; fixed here the same way.
"""

from __future__ import annotations

import psycopg
import pytest

from portolan.web.queries import not_ingested_sources

pytestmark = pytest.mark.usefixtures("db")


def test_not_ingested_sources_redacts_a_dsn_in_last_error(db: psycopg.Connection) -> None:
    leaky_error = (
        "extract: connection failed postgresql://portolan:s3cr3t-pw@db.internal:5432/portolan"
    )
    with db.cursor() as cur:
        cur.execute(
            """
            INSERT INTO sources (slug, source_type, s3_key, content_hash,
                                  source_date, extraction_status, attempts, last_error)
            VALUES ('S-call-leaky-coverage', 'call', 'originals/call/S-call-leaky-coverage',
                    'fake-hash-leaky-coverage', '2026-08-01', 'failed', 1, %s)
            """,
            (leaky_error,),
        )

    [row] = [r for r in not_ingested_sources(db) if r["slug"] == "S-call-leaky-coverage"]

    assert "s3cr3t-pw" not in row["last_error"]
    assert row["last_error"] == (
        "extract: connection failed postgresql://portolan:***@db.internal:5432/portolan"
    )


def test_not_ingested_sources_handles_a_null_last_error(db: psycopg.Connection) -> None:
    with db.cursor() as cur:
        cur.execute(
            """
            INSERT INTO sources (slug, source_type, s3_key, content_hash,
                                  source_date, extraction_status, attempts, last_error)
            VALUES ('S-call-no-error', 'call', 'originals/call/S-call-no-error',
                    'fake-hash-no-error', '2026-08-01', 'partial', 1, NULL)
            """
        )

    [row] = [r for r in not_ingested_sources(db) if r["slug"] == "S-call-no-error"]

    assert row["last_error"] is None
