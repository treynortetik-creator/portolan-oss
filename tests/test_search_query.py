"""Tests for portolan/search/query.py.

DB-backed, FakeEmbedder throughout. Builds an 8-note graph with fictional
names from corpus/truth-table.yaml (Ashgrove, Larkspur, Northlake, Tidewater
-- Halvard Medical's synthetic corpus), dates spread from 2025-05-01 to
2026-08-30, and a handful of `links` rows, then exercises RRF fusion, title
boost, recency, `expand`, the zero-result control query, and `since`.

Design notes baked into the corpus (see `corpus()` fixture below):
  - `ashgrove-medical-center` deliberately does NOT mention deliveries, so
    it is reachable for the "late deliveries" query only via `expand`
    (a `body` link from `late-tidewater-deliveries-midwest`).
  - `late-tidewater-deliveries-midwest` (fresh) and
    `late-tidewater-deliveries-midwest-duplicate` (stale, ~15 months old)
    are near-duplicate content and matching titles, isolating recency as
    the deciding factor between them.
  - `portal-password-change-requests` never contains the word "reset": the
    keyword leg (OR semantics) finds it for "portal password reset" through
    the two words it does share, and for "credential reset" not at all, so
    with `_SynonymEmbedder` it is a vector-only hit.
    `portal-password-reset-failures` contains "reset" and is the "found by
    both engines" comparison.
  - `northlake-clinical-engineering-2025` / `...-2026` share the same
    vocabulary; the `since` tests filter between them by `updated`.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import date
from typing import ClassVar

import pytest

from portolan.search.embed import FakeEmbedder
from portolan.search.index import reindex_all
from portolan.search.query import TITLE_BOOST_ALL, _title_boost, explain, search

NOTES = [
    dict(
        slug="ashgrove-health",
        type="account",
        title="Ashgrove Health",
        description="Regional health system in the Midwest, parent of Ashgrove Medical Center.",
        body_md=(
            "Ashgrove Health is a nonprofit health system headquartered in the Midwest "
            "region. It oversees several facilities including Ashgrove Medical Center."
        ),
        crm_id="A-1010",
        parent_slug=None,
        updated="2025-10-01",
    ),
    dict(
        slug="ashgrove-medical-center",
        type="account",
        title="Ashgrove Medical Center",
        description="A hospital facility, part of Ashgrove Health.",
        body_md=(
            "Ashgrove Medical Center is a hospital facility. "
            "Maria Chen is the Director of Procurement. Rob Tallis manages the ICU "
            "nursing staff."
        ),
        crm_id="A-1011",
        parent_slug="ashgrove-health",
        updated="2026-08-28",
    ),
    dict(
        slug="late-tidewater-deliveries-midwest",
        type="signal",
        title="Late Tidewater Deliveries in the Midwest",
        description=(
            "A rising pattern of late Tidewater consumable deliveries across Midwest accounts."
        ),
        body_md=(
            "Multiple Midwest hospitals reported late Tidewater deliveries during "
            "August 2026. Consumable shipments arrived days late at several "
            "facilities, and Midwest customers have repeatedly complained about the "
            "delays."
        ),
        crm_id=None,
        parent_slug=None,
        updated="2026-08-30",
    ),
    dict(
        slug="late-tidewater-deliveries-midwest-duplicate",
        type="signal",
        title="Late Tidewater Deliveries in the Midwest",
        description=(
            "An earlier report of the same pattern of late Tidewater deliveries across the Midwest."
        ),
        body_md=(
            "Multiple Midwest hospitals reported late Tidewater deliveries during "
            "early 2025. Consumable shipments arrived days late at several "
            "facilities, and Midwest customers have repeatedly complained about the "
            "delays."
        ),
        crm_id=None,
        parent_slug=None,
        updated="2025-06-15",
    ),
    dict(
        slug="portal-password-change-requests",
        type="signal",
        title="Portal Password Change Requests",
        description="Repeated help-desk tickets asking for portal password changes.",
        body_md=(
            "Customers have submitted many portal password change requests this "
            "quarter. The portal password change process takes too long according "
            "to several tickets about portal password changes."
        ),
        crm_id=None,
        parent_slug=None,
        updated="2026-07-20",
    ),
    dict(
        slug="portal-password-reset-failures",
        type="signal",
        title="Portal Password Reset Failures",
        description="Customers reporting that the portal password reset flow fails.",
        body_md=(
            "Several customers reported portal password reset failures this week. "
            "The portal password reset link does not work and customers cannot "
            "complete a portal password reset."
        ),
        crm_id=None,
        parent_slug=None,
        updated="2026-07-22",
    ),
    dict(
        slug="northlake-clinical-engineering-2025",
        type="theme",
        title="Northlake Clinical Engineering Escalation (2025)",
        description="An early escalation about clinical engineering staffing at Northlake.",
        body_md=(
            "Derek Hollis, Director of Clinical Engineering at Northlake General "
            "Hospital, escalated a pump maintenance backlog affecting the clinical "
            "engineering team."
        ),
        crm_id=None,
        parent_slug=None,
        updated="2025-05-01",
    ),
    dict(
        slug="northlake-clinical-engineering-2026",
        type="theme",
        title="Northlake Clinical Engineering Escalation (2026)",
        description="A follow-up escalation about clinical engineering staffing at Northlake.",
        body_md=(
            "Derek Hollis, Director of Clinical Engineering at Northlake General "
            "Hospital, again escalated the clinical engineering staffing backlog "
            "this year."
        ),
        crm_id=None,
        parent_slug=None,
        updated="2026-08-05",
    ),
]

LINKS = [
    ("late-tidewater-deliveries-midwest", "ashgrove-medical-center", "body"),
]

SOURCE_SLUG = "S-call-007"


@pytest.fixture()
def corpus(db):
    """Insert the 8-note graph (raw SQL, per the build instructions -- the
    notes writer is wave B1's), index it with FakeEmbedder, and return the
    (conn, embedder) pair every test in this file uses."""
    embedder = FakeEmbedder()
    with db.cursor() as cur:
        cur.execute(
            """
            INSERT INTO sources (slug, source_type, s3_key, content_hash,
                                  source_date, extraction_status)
            VALUES (%s, 'call', 's3://bucket/s-call-007.mp4', 'fake-hash-1', '2026-07-30', 'done')
            """,
            (SOURCE_SLUG,),
        )
        for note in NOTES:
            cur.execute(
                """
                INSERT INTO notes (slug, type, title, description, crm_id,
                                    parent_slug, body_md, updated)
                VALUES (%(slug)s, %(type)s, %(title)s, %(description)s, %(crm_id)s, %(parent_slug)s,
                        %(body_md)s, %(updated)s)
                """,
                note,
            )
        cur.execute(
            "INSERT INTO links (from_slug, to_slug, kind) VALUES (%s, %s, 'provenance')",
            ("late-tidewater-deliveries-midwest", SOURCE_SLUG),
        )
        for from_slug, to_slug, kind in LINKS:
            cur.execute(
                "INSERT INTO links (from_slug, to_slug, kind) VALUES (%s, %s, %s)",
                (from_slug, to_slug, kind),
            )

    report = reindex_all(db, embedder)
    assert report.notes_reindexed == len(NOTES)
    return db, embedder


def test_title_query_ranks_the_matching_note_first(corpus):
    conn, embedder = corpus
    result = search(conn, embedder, "Ashgrove Medical Center", k=5)
    assert result.hits
    assert result.hits[0].note_slug == "ashgrove-medical-center"


class _SynonymEmbedder(FakeEmbedder):
    """`FakeEmbedder` that reads "credential" as "password" and "reset" as
    "change" before hashing: the one thing a semantic model does that no
    keyword index can, so a note sharing none of the query's words can
    still be found, by the vector leg alone."""

    _SYNONYMS: ClassVar[dict[str, str]] = {"credential": "password", "reset": "change"}

    def embed(self, texts: list[str]) -> list[list[float]]:
        mapped = [" ".join(self._SYNONYMS.get(w, w) for w in t.lower().split()) for t in texts]
        return super().embed(mapped)


def test_note_matching_some_query_words_is_found_by_the_keyword_leg(corpus):
    conn, embedder = corpus
    result = search(conn, embedder, "portal password reset", k=10)
    slugs = {h.note_slug: h for h in result.hits}
    # shares "portal" and "password" with the query, never "reset"
    assert "fts" in slugs["portal-password-change-requests"].via


def test_query_found_only_by_vector_reports_via_vector(corpus):
    conn, _ = corpus
    result = search(conn, _SynonymEmbedder(), "credential reset", k=10)
    slugs = {h.note_slug: h for h in result.hits}
    assert slugs["portal-password-change-requests"].via == ["vector"]


def test_rrf_prefers_note_found_by_both_engines_over_one_engine(corpus):
    conn, _ = corpus
    result = search(conn, _SynonymEmbedder(), "credential reset", k=10)
    slugs = [h.note_slug for h in result.hits]
    both_idx = slugs.index("portal-password-reset-failures")
    vector_only_idx = slugs.index("portal-password-change-requests")
    assert both_idx < vector_only_idx

    assert result.hits[both_idx].via == ["fts", "vector"]
    assert result.hits[vector_only_idx].via == ["vector"]


def test_recency_lifts_fresh_note_over_stale_near_duplicate(corpus):
    conn, embedder = corpus
    result = search(conn, embedder, "late tidewater deliveries midwest", k=10)
    slugs = [h.note_slug for h in result.hits]
    assert "late-tidewater-deliveries-midwest" in slugs
    assert "late-tidewater-deliveries-midwest-duplicate" in slugs
    fresh_idx = slugs.index("late-tidewater-deliveries-midwest")
    stale_idx = slugs.index("late-tidewater-deliveries-midwest-duplicate")
    assert fresh_idx < stale_idx


def test_expand_pulls_in_a_linked_note(corpus):
    conn, embedder = corpus
    without_expand = search(conn, embedder, "late tidewater deliveries midwest", k=5, expand=0)
    with_expand = search(conn, embedder, "late tidewater deliveries midwest", k=5, expand=1)

    assert "ashgrove-medical-center" not in {h.note_slug for h in without_expand.hits}
    expanded_slugs = {h.note_slug: h for h in with_expand.hits}
    assert "ashgrove-medical-center" in expanded_slugs
    neighbour = expanded_slugs["ashgrove-medical-center"]
    assert neighbour.expanded_from == "late-tidewater-deliveries-midwest"
    assert neighbour.via == []


def test_nonsense_query_returns_zero_hits_with_control_ok_true(corpus):
    conn, embedder = corpus
    result = search(conn, embedder, "zephyr crystalline saxophone glockenspiel", k=10)
    assert result.hits == []
    assert result.control_ok is True


def test_empty_index_returns_control_ok_false(db):
    embedder = FakeEmbedder()
    result = search(db, embedder, "anything at all", k=10)
    assert result.hits == []
    assert result.control_ok is False


def test_since_filters_by_updated(corpus):
    conn, embedder = corpus
    query = "Northlake clinical engineering escalation"

    unfiltered = search(conn, embedder, query, k=10)
    unfiltered_slugs = {h.note_slug for h in unfiltered.hits}
    assert "northlake-clinical-engineering-2025" in unfiltered_slugs
    assert "northlake-clinical-engineering-2026" in unfiltered_slugs

    filtered = search(conn, embedder, query, k=10, since=date(2026, 1, 1))
    filtered_slugs = {h.note_slug for h in filtered.hits}
    assert "northlake-clinical-engineering-2026" in filtered_slugs
    assert "northlake-clinical-engineering-2025" not in filtered_slugs


def test_types_filter_restricts_to_the_given_note_types(corpus):
    conn, embedder = corpus
    result = search(conn, embedder, "Ashgrove Midwest", k=10, types=["account"])
    assert result.hits
    assert all(_note_type(conn, h.note_slug) == "account" for h in result.hits)


def _note_type(conn, slug: str) -> str:
    with conn.cursor() as cur:
        cur.execute("SELECT type FROM notes WHERE slug = %s", (slug,))
        return cur.fetchone()[0]


def test_source_slugs_come_from_provenance_links(corpus):
    conn, embedder = corpus
    result = search(conn, embedder, "late tidewater deliveries midwest", k=5)
    hit = next(h for h in result.hits if h.note_slug == "late-tidewater-deliveries-midwest")
    assert hit.source_slugs == [SOURCE_SLUG]


def test_explain_renders_hits_and_the_zero_result_case(corpus):
    conn, embedder = corpus
    result = search(conn, embedder, "Ashgrove Medical Center", k=3)
    text = explain(result)
    assert "ashgrove-medical-center" in text
    assert "score=" in text

    zero = search(conn, embedder, "zephyr crystalline saxophone glockenspiel", k=3)
    zero_text = explain(zero)
    assert "No results" in zero_text
    assert "healthy" in zero_text


# ---------------------------------------------------------------------
# `note_slugs` / `entity_slugs` scoping (added for the merge candidate
# gate's cross-note half, review G16, `portolan.merge.gate`): a small,
# self-contained graph (an account note plus two signal notes structurally
# linked to it, and one wholly unrelated note sharing no vocabulary at
# all) so the scope actually restricts the search *space*, not just what
# gets reported -- see `search()`'s and `_notes_linked_to_entities`'s
# docstrings for why that has to happen in the SQL.
# ---------------------------------------------------------------------

SCOPE_NOTES = [
    dict(
        slug="scope-account",
        type="account",
        title="Scope Account",
        description="An account used only for search-scoping tests.",
        crm_id="A-9001",
        body_md="Scope Account is a hospital system used only for search-scoping tests.",
        updated="2026-08-01",
    ),
    dict(
        slug="scope-signal-a",
        type="signal",
        title="Late deliveries at Scope Account",
        description="A late-delivery signal for Scope Account.",
        crm_id=None,
        body_md=(
            "Scope Account reported a late delivery of consumables in late "
            "July 2026. [[scope-account]]"
        ),
        updated="2026-08-02",
    ),
    dict(
        slug="scope-signal-b",
        type="signal",
        title="Pricing pressure at Scope Account",
        description="A pricing-pressure signal for Scope Account.",
        crm_id=None,
        body_md="Scope Account pushed back on pricing during renewal talks. [[scope-account]]",
        updated="2026-08-03",
    ),
    dict(
        slug="scope-unrelated",
        type="signal",
        title="Unrelated portal issue",
        description="Shares no vocabulary with Scope Account on purpose.",
        crm_id=None,
        body_md="A totally different customer had a login problem with their web portal.",
        updated="2026-08-04",
    ),
]


@pytest.fixture()
def scoped_corpus(db):
    embedder = FakeEmbedder()
    with db.cursor() as cur:
        for note in SCOPE_NOTES:
            cur.execute(
                """
                INSERT INTO notes (slug, type, title, description, crm_id, body_md, updated)
                VALUES (%(slug)s, %(type)s, %(title)s, %(description)s, %(crm_id)s,
                        %(body_md)s, %(updated)s)
                """,
                note,
            )
        # scope-signal-a/-b each carry a `[[scope-account]]` body link (the
        # `## Related`/description-wikilink pattern `portolan.notes.persist`
        # writes for real signal notes) -- `scope-unrelated` carries none.
        for slug in ("scope-signal-a", "scope-signal-b"):
            cur.execute(
                "INSERT INTO links (from_slug, to_slug, kind) VALUES (%s, %s, 'body')",
                (slug, "scope-account"),
            )
    report = reindex_all(db, embedder)
    assert report.notes_reindexed == len(SCOPE_NOTES)
    return db, embedder


def test_note_slugs_filter_restricts_the_search_space(scoped_corpus):
    conn, embedder = scoped_corpus
    result = search(
        conn, embedder, "Scope Account", k=10, note_slugs=["scope-account", "scope-signal-a"]
    )
    slugs = {h.note_slug for h in result.hits}
    assert slugs <= {"scope-account", "scope-signal-a"}
    assert "scope-signal-b" not in slugs
    assert "scope-unrelated" not in slugs


def test_entity_slugs_filter_finds_the_linked_neighbourhood(scoped_corpus):
    conn, embedder = scoped_corpus
    result = search(conn, embedder, "Scope Account", k=10, entity_slugs=["scope-account"])
    slugs = {h.note_slug for h in result.hits}
    assert "scope-account" in slugs
    assert "scope-signal-a" in slugs
    assert "scope-signal-b" in slugs
    assert "scope-unrelated" not in slugs


def test_entity_slugs_scoped_to_an_isolated_note_finds_nothing_else(scoped_corpus):
    conn, embedder = scoped_corpus
    result = search(conn, embedder, "Scope Account", k=10, entity_slugs=["scope-unrelated"])
    slugs = {h.note_slug for h in result.hits}
    # scope-unrelated has no structural links to anything -- the other
    # three notes must never appear, however they'd otherwise rank.
    assert "scope-account" not in slugs
    assert "scope-signal-a" not in slugs
    assert "scope-signal-b" not in slugs


def test_note_slugs_and_entity_slugs_combine_as_an_intersection(scoped_corpus):
    conn, embedder = scoped_corpus
    result = search(
        conn,
        embedder,
        "Scope Account",
        k=10,
        note_slugs=["scope-signal-b", "scope-unrelated"],
        entity_slugs=["scope-account"],
    )
    slugs = {h.note_slug for h in result.hits}
    # entity_slugs=[scope-account] admits {scope-account, scope-signal-a,
    # scope-signal-b}; note_slugs admits {scope-signal-b, scope-unrelated}
    # -- only scope-signal-b satisfies both.
    assert slugs <= {"scope-signal-b"}


# --------------------------------------------------------------------
# Bug hunt 2026-09-18, P1-1: `types=`/`since=` must be applied inside
# candidate retrieval (both engines), not as a post-fusion filter on the
# already pool-truncated result -- otherwise a real match that simply
# didn't make the *global* top-of-pool can be dropped and reported as
# "genuine nothing recorded" (`control_ok=True`).
# --------------------------------------------------------------------

_DENSE_PRICING_BODY = "Pricing pricing pricing pricing pricing pricing came up repeatedly " * 3
_DENSE_PRICING_DESCRIPTION = "Pricing pricing pricing discussion."
# One real mention -- weak on both engines: verified directly against
# `FakeEmbedder` before writing this fixture that this note's matching
# chunk scores far below the 40 dense filler notes' chunks on cosine
# similarity (0.41 vs 0.87-0.96), so it is excluded from BOTH engines'
# pools, not just one.
_WEAK_PRICING_BODY = "The customer mentioned pricing once during the call."


@pytest.fixture()
def pool_starvation_corpus(db):
    """40 notes that densely repeat "pricing" (enough to fill and rank
    ahead of everything else in both engines' candidate pools --
    `FTS_CANDIDATE_POOL`/the vector `3k` fetch are both well under 40),
    plus one note that only mentions it once. The one real match is
    reachable only by restricting the *search itself* to its type/date,
    not by filtering after the pool has already been decided."""
    embedder = FakeEmbedder()
    with db.cursor() as cur:
        for i in range(40):
            cur.execute(
                """
                INSERT INTO notes (slug, type, title, description, body_md, updated)
                VALUES (%s, 'product', %s, %s, %s, '2020-01-01')
                """,
                (
                    f"pricing-filler-{i:02d}",
                    f"Pricing Filler Product {i}",
                    _DENSE_PRICING_DESCRIPTION,
                    _DENSE_PRICING_BODY,
                ),
            )
        cur.execute(
            """
            INSERT INTO notes (slug, type, title, description, body_md, updated)
            VALUES ('pricing-signal', 'signal', 'Renewal Terms Call',
                    'A single pricing mention.', %s, '2026-09-18')
            """,
            (_WEAK_PRICING_BODY,),
        )
    report = reindex_all(db, embedder)
    assert report.notes_reindexed == 41
    return db, embedder


def test_types_filter_finds_a_note_the_global_pool_would_have_starved_out(
    pool_starvation_corpus,
):
    conn, embedder = pool_starvation_corpus

    filtered = search(conn, embedder, "pricing", k=12, types=["signal"])

    assert [h.note_slug for h in filtered.hits] == ["pricing-signal"]
    assert filtered.control_ok is True


def test_since_filter_finds_a_note_the_global_pool_would_have_starved_out(
    pool_starvation_corpus,
):
    """Same pool-starvation shape as the `types=` case above, but for
    `since=`: the one note inside the date window is out-ranked on raw
    text relevance by 40 older notes, so it must be found by restricting
    the search to the window, not by filtering the fused pool afterward."""
    conn, embedder = pool_starvation_corpus

    filtered = search(conn, embedder, "pricing", k=12, since=date(2026, 1, 1))

    assert [h.note_slug for h in filtered.hits] == ["pricing-signal"]
    assert filtered.control_ok is True


def test_types_filter_rejects_a_near_miss_type_name(corpus):
    """A near-miss type name (`"signals"` for `"signal"`) used to
    silently match zero notes and report a healthy empty search rather
    than the caller's typo."""
    conn, embedder = corpus
    with pytest.raises(ValueError, match="signals"):
        search(conn, embedder, "Ashgrove", k=5, types=["signals"])


# --------------------------------------------------------------------
# Bug hunt 2026-09-18, P1-2: the zero-result control query must probe the
# vector index (chunks/embeddings) too, not just `notes.search` -- a
# GENERATED STORED tsvector populated independently of whether anything
# was ever embedded.
# --------------------------------------------------------------------


def test_control_query_reports_broken_when_chunks_are_empty_but_notes_remain(corpus):
    conn, embedder = corpus
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM chunks")
        (chunks_before,) = cur.fetchone()
        cur.execute("SELECT count(*) FROM notes")
        (notes_before,) = cur.fetchone()
    assert chunks_before > 0
    assert notes_before > 0

    with conn.cursor() as cur:
        cur.execute("DELETE FROM chunks")

    result = search(conn, embedder, "zephyr crystalline saxophone glockenspiel", k=10)

    assert result.hits == []
    assert result.control_ok is False, (
        "a dead chunk/embedding index must not report as healthy just because "
        "notes.search (populated independently of chunks) still has rows"
    )


# --------------------------------------------------------------------
# Bug hunt 2026-09-18, P1-3: search must be deterministic -- a stable
# secondary sort key in every ranked query, so two workers (or two
# processes with different PYTHONHASHSEEDs) answer from the same
# evidence given the same database and query.
# --------------------------------------------------------------------

_TIED_BODY = (
    "Renewal objection escalation risk noted repeatedly during the call "
    "about a renewal objection escalation."
)


@pytest.fixture()
def tied_rank_corpus(db):
    """20 notes with byte-identical bodies (and descriptions), so both
    `ts_rank_cd` and cosine distance against "renewal objection
    escalation" tie exactly across all of them -- the shape the bug hunt
    reproduced against (60 notes tied at `ts_rank_cd` 4.0)."""
    embedder = FakeEmbedder()
    with db.cursor() as cur:
        for i in range(20):
            cur.execute(
                """
                INSERT INTO notes (slug, type, title, description, body_md, updated)
                VALUES (%s, 'signal', %s, 'A tied signal note.', %s, %s)
                """,
                (
                    f"tie-{i:02d}",
                    f"Tie Note {i}",
                    _TIED_BODY,
                    f"2026-0{(i % 9) + 1}-01",
                ),
            )
    report = reindex_all(db, embedder)
    assert report.notes_reindexed == 20
    return db


_SEARCH_SUBPROCESS_SCRIPT = """
import json
import sys

from portolan import db as db_module
from portolan.search.embed import FakeEmbedder
from portolan.search.query import search

conn = db_module.connect(sys.argv[1])
result = search(conn, FakeEmbedder(), sys.argv[2], k=12)
print(json.dumps([hit.note_slug for hit in result.hits]))
"""


def _run_search_in_subprocess(script_path, db_url: str, query: str, *, hash_seed: str) -> list[str]:
    env = {**os.environ, "PYTHONHASHSEED": hash_seed}
    proc = subprocess.run(
        [sys.executable, str(script_path), db_url, query],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, f"subprocess (PYTHONHASHSEED={hash_seed}) failed: {proc.stderr}"
    return json.loads(proc.stdout)


def test_search_is_deterministic_across_process_hash_seeds(tied_rank_corpus, db_url: str, tmp_path):
    """Runs the exact same query against the exact same (already-indexed)
    database twice, in separate subprocesses with different
    `PYTHONHASHSEED`s, and asserts identical hit ordering -- the way the
    bug hunt itself reproduced P1-3 (three different seeds, three
    different top-5s on the same freshly-indexed DB)."""
    script = tmp_path / "run_search_subprocess.py"
    script.write_text(_SEARCH_SUBPROCESS_SCRIPT)

    first = _run_search_in_subprocess(script, db_url, "renewal objection escalation", hash_seed="1")
    second = _run_search_in_subprocess(
        script, db_url, "renewal objection escalation", hash_seed="98765"
    )

    assert len(first) >= 10, "expected the tied corpus to actually exercise a multi-way tie"
    assert first == second, (
        "the same query against the same database must return the same order "
        "regardless of the process's own (by default random) string hash seed"
    )


# --------------------------------------------------------------------
# Bug hunt 2026-09-18, P1-4: supersession is "not deleted," not "still
# current" -- `search()` must default to `living` notes only.
# --------------------------------------------------------------------


def test_search_excludes_a_superseded_note_by_default(corpus):
    conn, embedder = corpus
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE notes SET status = 'superseded' WHERE slug = %s",
            ("ashgrove-medical-center",),
        )

    result = search(conn, embedder, "Ashgrove Medical Center", k=5)
    assert "ashgrove-medical-center" not in {h.note_slug for h in result.hits}


def test_search_include_superseded_opts_back_in(corpus):
    conn, embedder = corpus
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE notes SET status = 'superseded' WHERE slug = %s",
            ("ashgrove-medical-center",),
        )

    result = search(conn, embedder, "Ashgrove Medical Center", k=5, include_superseded=True)
    assert "ashgrove-medical-center" in {h.note_slug for h in result.hits}


# --------------------------------------------------------------------
# Bug hunt 2026-09-18, P2-10: title boost (and unmatched-term reporting)
# must match on whole words, not a naive substring test.
# --------------------------------------------------------------------


def test_title_boost_does_not_match_a_short_fragment_inside_a_longer_word():
    """The bug hunt's own reproduction: a three/two-character query-term
    fragment earned the same "every term in title" boost as an exact
    title match under naive substring matching."""
    assert _title_boost(["ash"], "Ashgrove Regional", 1.0) == 0.0
    assert _title_boost(["t2"], "Quarterly 2026 T2026 review", 1.0) == 0.0


def test_title_boost_still_matches_a_real_whole_word():
    assert _title_boost(["ashgrove"], "Ashgrove Regional", 1.0) == TITLE_BOOST_ALL * 1.0
    assert _title_boost(["t2026"], "Quarterly 2026 T2026 review", 1.0) == TITLE_BOOST_ALL * 1.0
