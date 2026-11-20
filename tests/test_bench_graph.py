"""Tests for `portolan.bench.graph`. DB-backed: uses the `db` fixture from
`tests/conftest.py`, which rebuilds the schema from migrations against
`DATABASE_URL` (falls back to `postgresql://localhost/portolan_test` if
unset). Per build-contracts.md's "own test database" rule, run this file
with:

    DATABASE_URL=postgresql://localhost/portolan_test_bench uv run pytest tests/test_bench_graph.py

(`createdb portolan_test_bench` once, first). Never against the shared
`portolan_test` database while other agents' tests may be running.
"""

from __future__ import annotations

from pathlib import Path

import psycopg
import psycopg.types.json
import pytest

from portolan.bench.graph import score_graph
from portolan.bench.truth import Truth

REPO_ROOT = Path(__file__).resolve().parents[1]
TRUTH_TABLE_PATH = REPO_ROOT / "corpus" / "truth-table.yaml"
MANIFEST_PATH = REPO_ROOT / "corpus" / "out" / "manifest.json"


@pytest.fixture()
def truth() -> Truth:
    return Truth.load(TRUTH_TABLE_PATH, MANIFEST_PATH)


def _insert_sources(conn: psycopg.Connection, truth: Truth) -> None:
    overrides = {
        "S-email-061": ("filtered", "internal_only_no_customer_or_product"),
        "S-email-040": ("filtered", "junk_auto_reply_or_calendar_invite"),
        "S-email-021-thread": ("filtered", "quoted_reply_thread_no_new_content"),
    }
    with conn.cursor() as cur:
        for src in truth.manifest.sources:
            status, reason = overrides.get(src.id, ("done", None))
            cur.execute(
                "INSERT INTO sources (slug, source_type, source_date, "
                "extraction_status, filter_reason) VALUES (%s,%s,%s,%s,%s)",
                (src.id, src.type, src.date, status, reason),
            )


def _insert_claims_for_fact(
    conn: psycopg.Connection, truth: Truth, note_slug: str, fact_id: str
) -> list[int]:
    fact_def = truth.fact(fact_id)
    placements = [p for p in truth.all_planted() if p.fact_id == fact_id]
    ids: list[int] = []
    with conn.cursor() as cur:
        for placement in placements:
            cur.execute(
                "INSERT INTO claims (note_slug, fact_type, claim_text, source_slug, locator, "
                "valid_from, valid_to) VALUES (%s,%s,%s,%s,%s,%s,%s) RETURNING id",
                (
                    note_slug,
                    fact_def.type,
                    fact_def.claim,
                    placement.source_id,
                    placement.locator,
                    fact_def.valid_from,
                    fact_def.valid_to,
                ),
            )
            ids.append(cur.fetchone()[0])
    return ids


def _supersede(conn: psycopg.Connection, old_ids: list[int], new_id: int, valid_to=None) -> None:
    """Mark `old_ids` superseded by `new_id`. When `valid_to` is given (the
    new claim's own `valid_from`, per `portolan/merge/facts.py`'s
    `_supersede` -- world time the old claim stopped being true), it is
    written onto the old claims too, matching the real two-clocks
    behaviour instead of leaving `valid_to` null."""
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE claims SET superseded_at = now(), superseded_by = %s, "
            "valid_to = COALESCE(%s, valid_to) WHERE id = ANY(%s)",
            (new_id, valid_to, old_ids),
        )


def _seed_passing_scenario(conn: psycopg.Connection, truth: Truth) -> None:
    """Populate every table `score_graph` reads with a state that should
    satisfy every `expectations.merge` / `expectations.filter` entry in
    the truth table."""
    _insert_sources(conn, truth)

    f07_placement = next(p for p in truth.all_planted() if p.fact_id == "F07")
    f07_source = truth.manifest.source(f07_placement.source_id)
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO coverage_ledger (connector, day, seen, sampled) VALUES (%s,%s,%s,%s)",
            (f07_source.type, f07_source.date, 10, 1),
        )

        for slug in (
            "ashgrove-halvardcare-renewal",
            "northlake-competitor-mentions",
            "larkspur-memorial-late-deliveries",
            "ashgrove-medical-night-mode-request",
            "kestrel-churn",
            "wrenfield-epic-integration",
        ):
            cur.execute(
                "INSERT INTO notes (slug, type, title) VALUES (%s,'signal',%s)", (slug, slug)
            )

    f11_ids = _insert_claims_for_fact(conn, truth, "ashgrove-halvardcare-renewal", "F11")
    f11b_ids = _insert_claims_for_fact(conn, truth, "ashgrove-halvardcare-renewal", "F11b")
    _supersede(conn, f11_ids, f11b_ids[0], valid_to=truth.fact("F11b").valid_from)

    f15_ids = _insert_claims_for_fact(conn, truth, "northlake-competitor-mentions", "F15")
    f15b_ids = _insert_claims_for_fact(conn, truth, "northlake-competitor-mentions", "F15b")
    _supersede(conn, f15_ids, f15b_ids[0])

    _insert_claims_for_fact(conn, truth, "larkspur-memorial-late-deliveries", "F02")
    _insert_claims_for_fact(conn, truth, "ashgrove-medical-night-mode-request", "F19")
    _insert_claims_for_fact(conn, truth, "kestrel-churn", "F34")

    # v1.1: the second reversal pair (F40 -> F40b, "two clocks" test).
    # Both facts carry an explicit `valid_from` in the truth table (unlike
    # F15/F15b) because `portolan/merge/facts.py`'s
    # `_explicit_supersession_date` now REQUIRES both claims to have a
    # real world date, strictly ordered, before it will supersede at all
    # -- no more falling back to the source's date. Seed both dates so
    # this fixture matches what a correctly-behaving merge pipeline would
    # actually write.
    f40_ids = _insert_claims_for_fact(conn, truth, "wrenfield-epic-integration", "F40")
    f40b_ids = _insert_claims_for_fact(conn, truth, "wrenfield-epic-integration", "F40b")
    _supersede(conn, f40_ids, f40b_ids[0], valid_to=truth.fact("F40b").valid_from)

    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO review_queue (kind, payload, source_slug) VALUES "
            "('unknown_entity', %s, %s)",
            (
                psycopg.types.json.Json(
                    {"mention": "Maria Chen", "email": "maria.chen.ohio@gmail.com"}
                ),
                "S-email-058",
            ),
        )
        cur.execute(
            "INSERT INTO review_queue (kind, payload, source_slug) VALUES "
            "('unknown_entity', %s, %s)",
            (psycopg.types.json.Json({"mention": "Renata Solis", "title": "CFO"}), "S-rss-006"),
        )
        cur.execute(
            "INSERT INTO review_queue (kind, payload, source_slug) VALUES "
            "('ambiguous_entity', %s, %s)",
            (
                psycopg.types.json.Json(
                    {
                        "mention": "Larkspur",
                        "attach_to": "larkspur",
                        "candidates": ["larkspur", "larkspur-memorial"],
                    }
                ),
                "S-email-052",
            ),
        )

        cur.execute(
            "INSERT INTO registry_entities (kind, canonical_name, crm_id) "
            "VALUES ('account','Pinecrest Health Partners','A-1001') RETURNING id"
        )
        pinecrest_id = cur.fetchone()[0]
        cur.execute(
            "INSERT INTO registry_entities (kind, canonical_name, crm_id, parent_id) "
            "VALUES ('account','Larkspur Regional Health','A-1002', %s) RETURNING id",
            (pinecrest_id,),
        )
        larkspur_id = cur.fetchone()[0]
        cur.execute(
            "INSERT INTO registry_entities (kind, canonical_name, crm_id, parent_id) "
            "VALUES ('account','Larkspur Memorial Hospital','A-1003', %s)",
            (larkspur_id,),
        )

        # v1.1's second hierarchy (F69: "Wrenfield Medical Center is owned
        # by Wrenfield Health"), independent of the Pinecrest/Larkspur one
        # above -- Wrenfield Health is its own root account (`parent:
        # null` in truth-table.yaml's `accounts:`), not a descendant of
        # Larkspur Memorial. See `expectations.merge.parent_chains` in the
        # truth table and its comment for why this is two separate chains
        # rather than one.
        cur.execute(
            "INSERT INTO registry_entities (kind, canonical_name, crm_id) "
            "VALUES ('account','Wrenfield Health','A-1090') RETURNING id"
        )
        wrenfield_id = cur.fetchone()[0]
        cur.execute(
            "INSERT INTO registry_entities (kind, canonical_name, crm_id, parent_id) "
            "VALUES ('account','Wrenfield Medical Center','A-1091', %s)",
            (wrenfield_id,),
        )


def test_score_graph_all_expectations_pass_on_a_correctly_seeded_db(db, truth: Truth) -> None:
    _seed_passing_scenario(db, truth)

    score = score_graph(db, truth)

    failures = score.failures()
    assert failures == [], f"unexpected failures: {[(f.name, f.detail) for f in failures]}"
    assert score.passed_count == len(score.results)
    assert score.passed_count >= 15  # every merge + filter expectation in the truth table


def test_score_graph_reports_failure_with_evidence_when_supersession_missing(
    db, truth: Truth
) -> None:
    _seed_passing_scenario(db, truth)
    # Undo just the F11 -> F11b supersession link; everything else stays correct.
    with db.cursor() as cur:
        cur.execute("UPDATE claims SET superseded_at = NULL, superseded_by = NULL")

    score = score_graph(db, truth)

    supersession_results = {r.name: r for r in score.results if r.name.startswith("supersession:")}
    assert supersession_results["supersession:F11->F11b"].passed is False
    assert supersession_results["supersession:F15->F15b"].passed is False
    # Evidence names which claim ids were checked, so the failure is
    # diagnosable without re-querying the database by hand.
    assert supersession_results["supersession:F11->F11b"].evidence["old_claim_ids"]

    # Unrelated expectations (cross-source citation counts, filter checks)
    # are unaffected by this one broken link.
    cross_source = next(r for r in score.results if r.name == "cross_source_single_claim:F11")
    assert cross_source.passed is True


def test_score_graph_reports_failures_on_an_empty_database(db, truth: Truth) -> None:
    score = score_graph(db, truth)
    assert score.results  # every expectation was at least attempted
    assert score.failed_count > 0
    assert all(isinstance(r.detail, str) and r.detail for r in score.results)
