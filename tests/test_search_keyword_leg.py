"""The keyword leg (`portolan.search.query._fts_candidates`): Okapi BM25 over
`notes.search` with OR semantics, checked against a BM25 computed here in
Python from the same tsvectors, plus the behaviours that make it usable on
natural-language questions (a partial match still ranks, a rare word
outweighs common ones, a very common word scores but does not nominate).
"""

from __future__ import annotations

import math

import pytest

from portolan.search import query as q
from portolan.search.embed import FakeEmbedder
from portolan.search.index import reindex_all

NOTES = {
    "renewal-risk-ashgrove": (
        "Ashgrove renewal risk",
        "Ashgrove raised pricing at the renewal call. Pricing came up twice and the renewal "
        "owner wants a discount before signing.",
    ),
    "late-deliveries-midwest": (
        "Late deliveries in the Midwest",
        "Deliveries arrived late at three Midwest hospitals. Late deliveries again in August.",
    ),
    "zanzibar-pilot": (
        "Pilot notes",
        "The Zanzibar clinic pilot started in May. Staff training went well and the pumps "
        "were installed on schedule, with deliveries on time and no alarms reported.",
    ),
    "pricing-committee": (
        "Pricing committee",
        "The pricing committee met to review discount bands for the next fiscal year.",
    ),
    "alarm-fatigue": (
        "Alarm fatigue",
        "Nurses reported alarm fatigue on the night shift; alarms were silenced repeatedly.",
    ),
}


@pytest.fixture()
def graph(db):
    with db.cursor() as cur:
        for slug, (title, body) in NOTES.items():
            cur.execute(
                "INSERT INTO notes (slug, type, title, body_md, updated) "
                "VALUES (%s, 'signal', %s, %s, '2026-09-01')",
                (slug, title, body),
            )
    reindex_all(db, FakeEmbedder())
    return db


def _reference_bm25(conn, query: str) -> list[str]:
    """BM25 as the module documents it, computed in Python from the rows'
    own tsvectors: Lucene idf over living notes, tf and length from
    positions, average length over the notes that contain a query word."""
    with conn.cursor() as cur:
        cur.execute("SELECT lexeme FROM unnest(to_tsvector('english', %s))", (query,))
        terms = {r[0] for r in cur.fetchall()}
        cur.execute(
            "SELECT n.slug, u.lexeme, cardinality(u.positions) "
            "FROM notes n, unnest(n.search) u WHERE n.status = 'living'"
        )
        tf: dict[str, dict[str, int]] = {}
        for slug, lexeme, count in cur.fetchall():
            tf.setdefault(slug, {})[lexeme] = count
    n = len(tf)
    df = {t: sum(1 for doc in tf.values() if t in doc) for t in terms}
    matching = [s for s, doc in tf.items() if terms & doc.keys()]
    avg = sum(sum(tf[s].values()) for s in matching) / len(matching)
    k1, b = q.BM25_K1, q.BM25_B
    scores = {}
    for s in matching:
        norm = k1 * (1 - b + b * sum(tf[s].values()) / avg)
        scores[s] = sum(
            math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5)) * f * (k1 + 1) / (f + norm)
            for t, f in tf[s].items()
            if t in terms
        )
    return sorted(scores, key=lambda s: (-scores[s], s))


@pytest.mark.parametrize(
    "query",
    [
        "Did Ashgrove ask for a discount at the renewal?",
        "late deliveries and alarms",
        "pricing discount",
    ],
)
def test_ranking_is_okapi_bm25(graph, query: str):
    ranked, _ = q._fts_candidates(graph, query, 10)
    assert ranked == _reference_bm25(graph, query)


def test_a_note_matching_one_query_word_still_ranks(graph):
    # only "alarm" is in the graph: three times in one note, once in another
    ranked, _ = q._fts_candidates(graph, "Which accounts complained about alarm volume?", 10)
    assert ranked == ["alarm-fatigue", "zanzibar-pilot"]


def test_a_rare_word_outweighs_common_ones(graph):
    ranked, _ = q._fts_candidates(graph, "Zanzibar deliveries pricing", 10)
    assert ranked[0] == "zanzibar-pilot"


def test_a_very_common_word_scores_but_does_not_nominate(graph, monkeypatch):
    monkeypatch.setattr(q, "KEYWORD_RESCORE_POOL", 1)
    # "pricing" is in 2 of 5 notes, over the cap of max(0.1 x 5, 1) = 1
    ranked, _ = q._fts_candidates(graph, "pricing zanzibar", 10)
    assert ranked == ["zanzibar-pilot"]
    # with no word under the cap, the rarest one nominates on its own
    ranked, _ = q._fts_candidates(graph, "pricing", 10)
    assert ranked == ["pricing-committee", "renewal-risk-ashgrove"]


def test_the_cited_chunk_is_the_one_holding_the_query_words(db):
    # paragraphs long enough that the chunker (1,200 chars) keeps each alone
    body = "\n\n".join(
        [f"Paragraph {i} is about staffing and rosters. " * 27 for i in range(3)]
        + ["The Zanzibar clinic asked for a new pump."]
    )
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO notes (slug, type, title, body_md, updated) "
            "VALUES ('long-note', 'signal', 'Long note', %s, '2026-09-01')",
            (body,),
        )
    reindex_all(db, FakeEmbedder())
    ranked, chunks = q._fts_candidates(db, "zanzibar pump", 10)
    assert ranked == ["long-note"]
    assert "Zanzibar" in chunks["long-note"][1]
    assert "staffing" not in chunks["long-note"][1]


@pytest.mark.parametrize(
    "query", ["what is the", "O'Reilly's back\\slash 'quoted' | & ! <-> :*", ""]
)
def test_stopwords_and_tsquery_syntax_never_break_it(graph, query: str):
    ranked, chunks = q._fts_candidates(graph, query, 10)
    assert ranked == [] and chunks == {}


# The candidate step at a scale where some words are common: 40 living
# notes and one superseded one, the nominating threshold monkeypatched down
# to max(0.1 x 40, 5) = 5 notes so "pump", "alarm", "delivery" and
# "hospital" are common and "zanzibar", "quartz" and "jade" nominate.
RARE = {"zanzibar": (3, 17, 22), "quartz": (5, 6), "jade": (9, 17, 30, 31, 33)}


@pytest.fixture()
def wider_graph(db, monkeypatch):
    monkeypatch.setattr(q, "KEYWORD_RESCORE_POOL", 5)
    with db.cursor() as cur:
        for i in range(41):
            words = [
                *(["pump"] if i % 10 else []),
                *(["alarm"] if i % 3 else []),
                *(["delivery"] if i % 2 == 0 else []),
                *(["hospital"] if i % 5 != 1 else []),
                *[w for w, notes in RARE.items() if i in notes],
            ]
            body = "Note about " + " and ".join(words or ["nothing"]) + "."
            if i == 40:
                body += " Zanzibar quartz jade."
            cur.execute(
                "INSERT INTO notes (slug, type, title, body_md, status, updated) "
                "VALUES (%s, 'signal', %s, %s, %s, '2026-09-01')",
                (f"note-{i:02d}", f"Note {i}", body, "superseded" if i == 40 else "living"),
            )
    return db


def _reference_pool(conn, query: str, pool: int) -> tuple[list[str], dict[str, int]]:
    """The documented candidate step, exhaustively in Python over living
    notes' tsvectors: df per word, nominating words (df at most
    max(0.1 x notes, KEYWORD_RESCORE_POOL), else the rarest), the nominated
    notes ranked by summed idf over every query word they hold, then slug."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT lexeme FROM unnest(to_tsvector('english', %s)) ORDER BY lexeme", (query,)
        )
        terms = [r[0] for r in cur.fetchall()]
        cur.execute(
            "SELECT n.slug, array_agg(u.lexeme) FROM notes n, unnest(n.search) u "
            "WHERE n.status = 'living' GROUP BY n.slug"
        )
        docs = {slug: set(lexemes) for slug, lexemes in cur.fetchall()}
    n = len(docs)
    df = {t: sum(1 for d in docs.values() if t in d) for t in terms}
    df = {t: c for t, c in df.items() if c}
    if not df:
        return [], {}
    threshold = max(q.KEYWORD_COMMON_FRACTION * n, q.KEYWORD_RESCORE_POOL)
    nominators = {t for t, c in df.items() if c <= threshold}
    if not nominators:
        nominators = {min(df, key=lambda t: (df[t], terms.index(t)))}
    idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
    nominated = [s for s, d in docs.items() if d & nominators]
    coord = {s: sum(idf[t] for t in df if t in docs[s]) for s in nominated}
    return sorted(nominated, key=lambda s: (-round(coord[s], 9), s))[:pool], df


@pytest.mark.parametrize(
    "query",
    [
        "pump alarm at zanzibar",  # one rare word, three common ones
        "jade quartz delivery hospital pump",  # two rare words; common ones reorder
        "hospital pump alarm delivery",  # every word common: the rarest nominates alone
        "zanzibar",
        "tungsten",  # in no note
    ],
)
def test_candidate_pool_matches_the_exhaustive_reference(wider_graph, query: str):
    where, params = q._note_filters(
        "n", scope_list=None, types=None, since=None, status_values=["living"]
    )
    with wider_graph.cursor() as cur:
        cur.execute(
            "SELECT lexeme FROM unnest(to_tsvector('english', %s)) ORDER BY lexeme", (query,)
        )
        lexemes = [r[0] for r in cur.fetchall()]
        literals = [q._tsquery_literal(lx) for lx in lexemes]
        candidates, doc_freq, n_notes = q._keyword_pool(cur, lexemes, literals, where, params, 5)
    expected, expected_df = _reference_pool(wider_graph, query, 5)
    assert n_notes == 40
    assert candidates == expected
    assert doc_freq == expected_df
