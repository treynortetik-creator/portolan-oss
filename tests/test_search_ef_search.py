"""`hnsw.ef_search` for search()'s vector leg: set per query (`SET LOCAL`),
from the process's `search_hnsw_ef_search` setting unless the caller passes
a value.

pgvector's HNSW scan returns at most `ef_search` rows, so with the planner
kept on the index (`enable_seqscan` and `enable_sort` off: on a graph this
small it would otherwise sort every chunk exactly, the plan production
uses at 1.36M chunks is HNSW then an incremental sort) a value of 1 visibly
starves the vector leg. That is how these tests see the setting take
effect.
"""

from __future__ import annotations

import pytest

from portolan import config
from portolan.search import query as q
from portolan.search.embed import FakeEmbedder
from portolan.search.index import reindex_all
from portolan.search.query import search

WORDS = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel"]


@pytest.fixture()
def graph(db):
    with db.cursor() as cur:
        for word in WORDS:
            cur.execute(
                "INSERT INTO notes (slug, type, title, body_md, updated) "
                "VALUES (%s, 'signal', %s, %s, '2026-09-01')",
                (f"pump-{word}", f"Pump {word}", f"The {word} pump alarm was reported."),
            )
        cur.execute("SET enable_seqscan = off")
        cur.execute("SET enable_sort = off")
    reindex_all(db, FakeEmbedder())
    return db


def _vector_hits(result) -> int:
    return sum("vector" in h.via for h in result.hits)


def test_ef_search_bounds_the_vector_legs_hnsw_scan(graph):
    few, _ = q._vector_candidates(graph, FakeEmbedder(), "pump alarm", 10, ef_search=1)
    many, _ = q._vector_candidates(graph, FakeEmbedder(), "pump alarm", 10, ef_search=100)
    assert len(few) == 1
    assert len(many) == len(WORDS)


def test_ef_search_ends_with_the_query_on_an_autocommit_connection(graph):
    with graph.cursor() as cur:
        cur.execute("SHOW hnsw.ef_search")
        session_value = cur.fetchone()[0]
    q._vector_candidates(graph, FakeEmbedder(), "pump alarm", 10, ef_search=1)
    with graph.cursor() as cur:
        cur.execute("SHOW hnsw.ef_search")
        assert cur.fetchone()[0] == session_value != "1"


def test_search_uses_the_configured_ef_search(graph, monkeypatch):
    monkeypatch.setattr(config.settings, "search_hnsw_ef_search", 1)
    assert _vector_hits(search(graph, FakeEmbedder(), "pump alarm", k=10)) == 1
    monkeypatch.setattr(config.settings, "search_hnsw_ef_search", 100)
    assert _vector_hits(search(graph, FakeEmbedder(), "pump alarm", k=10)) == len(WORDS)


def test_search_ef_search_argument_overrides_the_setting(graph, monkeypatch):
    monkeypatch.setattr(config.settings, "search_hnsw_ef_search", 100)
    result = search(graph, FakeEmbedder(), "pump alarm", k=10, ef_search=1)
    assert _vector_hits(result) == 1
