"""Fused full-text + vector search over notes and chunks.

Spec: 01-spec/portolan-spec-v2.md sec 9 (RRF fusion, title boost, recency
with absolute decay, `expand` link hops, a control
query on zero results). The RRF, title-bonus and zero-result ideas come from
02-reference-implementation/memory-system/recall.py.

The keyword leg is Okapi BM25 over `notes.search` with OR semantics (a note
matching some of a question's words still ranks), computed in SQL from the
tsvector's own positions: no extension beyond pgvector, because Railway's
Postgres image is the official `postgres` image plus pgvector only
(01-spec/bench-erb-search-2026-09-22.md, "Keyword leg, rebuilt"). Query
and index text both pass through `portolan_search_text` (migration 0035), so
"Link/Epic" is the words "link" and "epic" on both sides.

`search()` is `_legs` (the two retrieval legs) then `_rank` (fusion, title
boost, recency, and two steps behind settings: per-query fusion weights,
`_route_weights`, on by default, and a local cross-encoder over the fused
pool, `portolan/search/rerank.py`, off by default). Within the keyword leg a
note's rank counts by the share of the query it holds (`KEYWORD_SHARE_FULL`).
Bench write-up, "Reranker and query routing".

Dates: `since=` reads `notes.content_date`, the newest `recorded` date among
the note's live claims, falling back to the note's last write
(`content_date_source = 'note_updated'`) when it has none (migration 0035),
so a graph loaded in one sitting keeps its claims' dates. Recency reads
`notes.updated` unless `search_recency_content_date` is on
(`_note_recency`; 01-spec/content-dates-2026-09-25.md).
"""

from __future__ import annotations

import contextlib
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time
from typing import TYPE_CHECKING, Literal, NamedTuple
from zoneinfo import ZoneInfo

import psycopg
from pydantic import BaseModel

from portolan.notes.models import NoteType
from portolan.search.embed import Embedder
from portolan.util import clock

if TYPE_CHECKING:
    from portolan.config import Settings
    from portolan.search.rerank import Reranker

# One word tokenizer for `_terms()`, `_title_boost()` and the
# `query_terms_unmatched` scan, so "term present" always means a whole word.
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-]+")

RRF_K = 60
# Equal weights: the best worst case over ERB dev recall and recall on
# paraphrased demo questions (01-spec/bench-erb-search-2026-09-22.md,
# "Fusion weight"). Below about 0.87 a note found only by the vector leg cannot
# outrank the keyword leg's tenth hit, so a question worded unlike its note is
# lost. At 1.0 a keyword and a vector hit at the same rank tie; `(-score, slug)`
# orders them.
FTS_WEIGHT = 1.0
VECTOR_WEIGHT = 1.0
# Query routing (`_route_weights`, setting `search_fusion_routing`; the weights
# above for every query when it is off). Chosen on ERB dev, the paraphrases and
# the demo questions (bench write-up, "Reranker and query routing"): the
# keyword profile is ERB dev's best fixed pair, the semantic one the
# paraphrases' best; thresholds 0.3 to 0.5 tied, the rule's tie-break took 0.3.
ROUTE_COVERAGE_THRESHOLD = 0.3
ROUTE_KEYWORD_WEIGHTS = (1.0, 0.2)
ROUTE_SEMANTIC_WEIGHTS = (1.0, 1.0)
# A keyword-leg note's rank counts in full when the note holds at least this
# share of the query's idf, and in proportion below it (`_keyword_scale`).
# Rank alone cannot tell the note that matched the question's rare words from
# one that shares a single stray word with it; fused by rank alone, a note
# found weakly by both legs outranked the vector leg's first hit, which is how
# a question worded unlike its note was lost. Chosen on the demo and paraphrase
# questions (0.3 to 1.0 all beat rank-only fusion on both) and checked on the
# held-out set (`portolan.bench.demo_search.HELDOUT_QUESTIONS`;
# docs/reviews/REVIEW-read-path.md). 0 turns it off.
KEYWORD_SHARE_FULL = 0.5
KEYWORD_SHARE_FLOOR = 0.0
FTS_CANDIDATE_POOL = 30  # floor on FTS candidates, like recall.py's max(k*3, 30)
VECTOR_FETCH_MULTIPLIER = 3  # "fetching 3k" per the contract
# Okapi BM25 with Lucene's defaults, the parameters of the BM25 baseline
# the ERB exam compares against (portolan/bench/erb.py `Bm25Index`).
BM25_K1 = 1.2
BM25_B = 0.75
# BM25 is computed exactly for this many notes per query: the ones with the
# highest summed idf over the query words they contain.
KEYWORD_RESCORE_POOL = 200
# A word nominates candidates only if it is in at most
# max(this share of notes, KEYWORD_RESCORE_POOL notes); the rarest query word
# always nominates. A commoner word still adds to every candidate's score.
# Both choices match exhaustive BM25 on the ERB exam's dev half (bench
# write-up, "Keyword leg, rebuilt").
KEYWORD_COMMON_FRACTION = 0.1
# `ORDER BY <=> LIMIT n` always returns n neighbours however irrelevant, so
# without a floor a nonsense query could never return zero hits. With
# `FakeEmbedder`, texts sharing no tokens score near 0.
VECTOR_MIN_SIMILARITY = 0.05
# x RRF max when every query term is in the title; a title holding some of
# them earns that share of it (`_title_boost`).
TITLE_BOOST_ALL = 0.5
RECENCY_FLOOR = 0.85
RECENCY_SPAN = 0.15
RECENCY_HALF_LIFE_DAYS = 180
# With `search_recency_content_date` on, a note with no live claims
# (`content_date_source = "note_updated"`) gets the floor: its write time says
# when it was rendered, not when its content was true
# (01-spec/content-dates-2026-09-25.md). None ranks it by its write time.
RECENCY_UNDATED_MULTIPLIER: float | None = RECENCY_FLOOR

# The keyword leg's query tsvector: the same text normalizer the generated
# `notes.search`/`chunks.search` columns apply (migration 0035).
QUERY_TSVECTOR_SQL = "to_tsvector('english', portolan_search_text(%s))"

EXPAND_FRONTIER = 3  # only the top-N hits are followed for `expand`
EXPAND_DISCOUNT = 0.5

# A query, not a document. The MCP `search` tool and the `/search` redirect
# reject longer text (`tests/test_mcp_server.py`); Atlas has its own bound.
MAX_SEARCH_QUERY_CHARS = 500

# Superseded means "not deleted", not "still current": `search()` returns
# living notes unless `include_superseded=True`.
_STATUS_LIVING_ONLY = ["living"]
_STATUS_ALL = ["living", "superseded", "archived"]

STOP_WORDS = {
    "the",
    "a",
    "an",
    "of",
    "to",
    "in",
    "on",
    "for",
    "and",
    "or",
    "is",
    "are",
    "what",
    "did",
    "we",
    "about",
    "how",
    "why",
    "i",
    "do",
    "does",
    "with",
    "our",
    "my",
    "at",
    "by",
    "it",
    "this",
    "that",
    "was",
    "were",
}


class SearchHit(BaseModel):
    note_slug: str
    title: str
    # Fused score (RRF, title boost, recency); with a reranker, its relevance
    # (0 to 1) times recency.
    score: float
    via: list[Literal["fts", "vector"]]
    best_chunk: str
    heading: str | None
    # Last write of the note (deployment zone).
    updated: date
    source_slugs: list[str]
    # Set on a hit added by `expand` (its `via` is then `[]`): the slug of the
    # hit it hopped from.
    expanded_from: str | None = None
    # What `since=` and recency read: the newest recorded date among the
    # note's live claims, or its last write when `content_date_source` is
    # "note_updated". None only on a hit built outside `search()`.
    content_date: date | None = None
    content_date_source: Literal["claims_recorded", "note_updated"] | None = None


class _NoteMeta(NamedTuple):
    title: str
    type: str
    content_date: datetime
    updated: datetime
    content_date_source: str


class SearchResult(BaseModel):
    hits: list[SearchHit]
    control_ok: bool
    query_terms_unmatched: list[str]


def _terms(query: str) -> list[str]:
    """Query terms for the title boost and `query_terms_unmatched`:
    lowercase, no stopwords, longer than two characters. The keyword leg
    uses Postgres's `english` parser instead."""
    return [w for w in _TOKEN_RE.findall(query.lower()) if w not in STOP_WORDS and len(w) > 2]


def _notes_linked_to_entities(conn: psycopg.Connection, entity_slugs: list[str]) -> set[str]:
    """Every note slug in the structural neighbourhood of `entity_slugs`:
    the entities themselves, plus any note carrying a `body` or `parent`
    link to or from one of them (never `provenance`/`supersession`, which
    point at sources or superseded notes, not siblings) -- e.g. every
    signal, contact or product note hanging off a given account note.
    Backs `search(..., entity_slugs=...)`, which the merge gate uses to find
    a duplicate fact that landed on a different note of the same account.
    """
    if not entity_slugs:
        return set()
    slugs = list(entity_slugs)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT from_slug FROM links WHERE to_slug = ANY(%s) AND kind IN ('body', 'parent')
            UNION
            SELECT to_slug FROM links WHERE from_slug = ANY(%s) AND kind IN ('body', 'parent')
            """,
            (slugs, slugs),
        )
        linked = {row[0] for row in cur.fetchall()}
    return linked | set(slugs)


def _tsquery_literal(lexeme: str) -> str:
    """One already-normalized lexeme as `tsquery` input text: quoted, with
    quotes doubled and backslashes escaped, so `::tsquery` matches exactly
    that lexeme and never re-parses it as operators."""
    return "'" + lexeme.replace("\\", "\\\\").replace("'", "''") + "'"


def _note_filters(
    alias: str,
    *,
    scope_list: list[str] | None,
    types: list[str] | None,
    since: date | None,
    status_values: list[str],
) -> tuple[str, list[object]]:
    """The `WHERE` clauses every keyword-leg query shares, and their params:
    status always, scope/type/date when given (pushed into the SQL, never a
    post-filter on a truncated pool; tests P1-1 in tests/test_search_query.py)."""
    clauses = [f"{alias}.status = ANY(%s)"]
    params: list[object] = [status_values]
    if scope_list is not None:
        clauses.append(f"{alias}.slug = ANY(%s)")
        params.append(scope_list)
    if types:
        clauses.append(f"{alias}.type = ANY(%s)")
        params.append(types)
    if since:
        clauses.append(f"{alias}.content_date >= %s")
        params.append(_day_start(since))
    return " AND ".join(clauses), params


def _day_start(day: date) -> datetime:
    """Midnight of `day` in the deployment's zone: `content_date >= this` is
    "its world date is on or after `day`", whatever the session's TimeZone,
    and still uses `idx_notes_content_date`."""
    return datetime.combine(day, time.min, tzinfo=ZoneInfo(clock.timezone_name()))


def _bm25_idf(n_notes: int, doc_freq: int) -> float:
    """Lucene's BM25 idf, never negative."""
    return math.log(1.0 + (n_notes - doc_freq + 0.5) / (doc_freq + 0.5))


def _keyword_pool(
    cur: psycopg.Cursor,
    lexemes: list[str],
    literals: list[str],
    where: str,
    params: list[object],
    pool: int,
) -> tuple[list[str], dict[str, int], int]:
    """Steps 2 and 3 of `_fts_candidates`: (the `pool` candidates by summed
    idf, then slug; {lexeme: document frequency} for every lexeme found;
    the note count), all within the filters `where`.

    Three passes over postings, never joining one word's postings to
    another's (the costly part at 150,000 notes; bench write-up,
    "Production path"): each word counted only up to just past the
    nominating threshold, which tells nominating words (exact count) from
    common ones; the nominating words' postings summed per note; then one
    scan per common word for its exact count and the nominated notes it
    is in.
    """
    cur.execute(
        f"""
        WITH total AS MATERIALIZED (
            SELECT count(*) AS n_notes, greatest(%s * count(*), %s) AS threshold
            FROM notes n WHERE {where}
        )
        SELECT t.i::int, total.n_notes, total.threshold, (
            SELECT count(*) FROM (
                SELECT 1 FROM notes n WHERE n.search @@ t.q::tsquery AND {where}
                LIMIT floor(total.threshold)::bigint + 1
            ) s
        )
        FROM unnest(%s::text[]) WITH ORDINALITY AS t(q, i) CROSS JOIN total
        ORDER BY t.i
        """,
        [KEYWORD_COMMON_FRACTION, KEYWORD_RESCORE_POOL, *params, *params, literals],
    )
    rows = cur.fetchall()
    n_notes, threshold = int(rows[0][1]), float(rows[0][2])
    # Capped at floor(threshold) + 1: a count within the threshold is exact.
    found = {int(i): int(count) for i, _n, _t, count in rows if count}
    if not found:
        return [], {}, n_notes
    doc_freq = {i: count for i, count in found.items() if count <= threshold}
    if not doc_freq:
        # Every word is common: the rarest one nominates on its own.
        order = sorted(found)
        cur.execute(
            f"""
            SELECT t.i::int,
                   (SELECT count(*) FROM notes n WHERE n.search @@ t.q::tsquery AND {where})
            FROM unnest(%s::text[]) WITH ORDINALITY AS t(q, i)
            """,
            [*params, [literals[i - 1] for i in order]],
        )
        exact = {order[pos - 1]: int(count) for pos, count in cur.fetchall()}
        rarest = min(exact, key=lambda i: (exact[i], i))
        doc_freq = {rarest: exact[rarest]}
    nominating = sorted(doc_freq)
    common = [i for i in sorted(found) if i not in doc_freq]

    # The `IN` sits in an aggregate's FILTER so it stays a hashed subplan over
    # the word's postings: as a join, the planner probes each nominated note's
    # TOASTed tsvector per word instead (seconds per query at 150,000 notes).
    cur.execute(
        f"""
        WITH nominated AS MATERIALIZED (
            SELECT p.tid, sum(t.idf) AS partial
            FROM unnest(%s::text[], %s::float8[]) AS t(q, idf)
            CROSS JOIN LATERAL (
                SELECT n.ctid AS tid FROM notes n WHERE n.search @@ t.q::tsquery AND {where}
            ) p
            GROUP BY p.tid
        ),
        common AS MATERIALIZED (
            SELECT t.i, x.df, x.tids
            FROM unnest(%s::text[], %s::int[]) AS t(q, i)
            CROSS JOIN LATERAL (
                SELECT count(*) AS df,
                       array_agg(n.ctid) FILTER (WHERE n.ctid IN (SELECT tid FROM nominated))
                           AS tids
                FROM notes n WHERE n.search @@ t.q::tsquery AND {where}
            ) x
        ),
        bonus AS (
            SELECT u.tid, sum(ln(1 + (%s - c.df + 0.5) / (c.df + 0.5))) AS extra
            FROM common c CROSS JOIN LATERAL unnest(c.tids) AS u(tid)
            GROUP BY u.tid
        ),
        scored AS (
            SELECT m.tid, m.partial + coalesce(b.extra, 0) AS coord
            FROM nominated m LEFT JOIN bonus b USING (tid)
        ),
        cut AS (
            SELECT coalesce(
                (SELECT coord FROM scored ORDER BY coord DESC LIMIT 1 OFFSET %s - 1),
                '-infinity'::float8
            ) AS v
        )
        SELECT n.slug, (SELECT json_object_agg(i, df) FROM common)
        FROM scored s JOIN notes n ON n.ctid = s.tid CROSS JOIN cut
        WHERE s.coord >= cut.v
        ORDER BY s.coord DESC, n.slug ASC
        LIMIT %s
        """,
        [
            [literals[i - 1] for i in nominating],
            [_bm25_idf(n_notes, doc_freq[i]) for i in nominating],
            *params,
            [literals[i - 1] for i in common],
            common,
            *params,
            float(n_notes),
            pool,
            pool,
        ],
    )
    rows = cur.fetchall()
    if not rows:
        return [], {}, n_notes
    doc_freq.update({int(i): int(df) for i, df in (rows[0][1] or {}).items()})
    return [row[0] for row in rows], {lexemes[i - 1]: df for i, df in doc_freq.items()}, n_notes


def _fts_candidates(
    conn: psycopg.Connection,
    query: str,
    limit: int,
    *,
    scope: set[str] | None = None,
    types: list[str] | None = None,
    since: date | None = None,
    include_superseded: bool = False,
) -> tuple[list[str], dict[str, tuple[str | None, str]]]:
    """`_fts_leg` without its coverage signal."""
    ranked, chunks, _coverage = _fts_leg(
        conn,
        query,
        limit,
        scope=scope,
        types=types,
        since=since,
        include_superseded=include_superseded,
    )
    return ranked, chunks


def _fts_leg(
    conn: psycopg.Connection,
    query: str,
    limit: int,
    *,
    scope: set[str] | None = None,
    types: list[str] | None = None,
    since: date | None = None,
    include_superseded: bool = False,
) -> tuple[list[str], dict[str, tuple[str | None, str]], float]:
    """`_keyword_leg` without its per-note shares."""
    ranked, chunks, coverage, _share = _keyword_leg(
        conn,
        query,
        limit,
        scope=scope,
        types=types,
        since=since,
        include_superseded=include_superseded,
    )
    return ranked, chunks, coverage


def _keyword_leg(
    conn: psycopg.Connection,
    query: str,
    limit: int,
    *,
    scope: set[str] | None = None,
    types: list[str] | None = None,
    since: date | None = None,
    include_superseded: bool = False,
) -> tuple[list[str], dict[str, tuple[str | None, str]], float, dict[str, float]]:
    """The keyword leg: notes ranked by Okapi BM25 over `notes.search`
    (title, description and body), OR semantics, so a question matching
    some of a note's words still finds it.

    1. The query's words are Postgres's own `english` lexemes
       (`QUERY_TSVECTOR_SQL`: slash- and dot-joined words split, stopwords
       out, stemmed), deduplicated.
    2. Document frequency per lexeme and the note count, both within the
       filters below; idf is Lucene's.
    3. Candidates: among the notes holding at least one nominating lexeme
       (document frequency at most max(`KEYWORD_COMMON_FRACTION` x notes,
       `KEYWORD_RESCORE_POOL`), plus the rarest lexeme always), the
       `KEYWORD_RESCORE_POOL` (or `limit`, if larger) with the highest
       summed idf over every query lexeme they contain.
    4. Exact BM25 (k1 1.2, b 0.75) for those candidates, tf and document
       length read from the tsvector's positions; average length over the
       candidates. Order `(-score, slug)`, top `limit`.

    Steps 2 and 3 (`_keyword_pool`) read postings (the GIN index on a large
    graph), not note text; only the candidates' tsvectors (step 4) and their
    chunks are read.

    `scope` (`search()`'s `note_slugs`/`entity_slugs`, already one slug
    set), `types`, `since` and `include_superseded` restrict every step in
    the SQL itself, so a filtered search weighs a note against the filtered
    set, never against a pool truncated on the whole graph first.
    `include_superseded=False` is `living` notes only.

    Returns (ranked note_slugs, {note_slug: (heading, chunk_text)},
    coverage): for each ranked note, its chunk with the highest summed idf x
    saturated tf over the query's lexemes (ties to the lower chunk id), when
    any chunk contains one. `coverage` is the query routing signal
    (`_route_weights`): the largest share of the query's idf that one
    candidate note holds, a lexeme found in no note counting at the idf of
    document frequency 0; 0.0 when nothing matched. `share` is the same
    figure per ranked note: the part of the query's idf that note holds, which
    `_rank` uses to weigh the note's keyword rank (`KEYWORD_SHARE_FULL`).
    """
    scope_list = list(scope) if scope is not None else None
    status_values = _STATUS_ALL if include_superseded else _STATUS_LIVING_ONLY
    where, params = _note_filters(
        "n", scope_list=scope_list, types=types, since=since, status_values=status_values
    )

    with conn.cursor() as cur:
        cur.execute(f"SELECT lexeme FROM unnest({QUERY_TSVECTOR_SQL}) ORDER BY lexeme", (query,))
        lexemes = [row[0] for row in cur.fetchall()]
        if not lexemes or limit <= 0:
            return [], {}, 0.0, {}
        literals = [_tsquery_literal(lx) for lx in lexemes]

        candidates, doc_freq, n_notes = _keyword_pool(
            cur, lexemes, literals, where, params, max(KEYWORD_RESCORE_POOL, limit)
        )
        if not candidates:
            return [], {}, 0.0, {}
        idf = {lex: _bm25_idf(n_notes, int(df)) for lex, df in doc_freq.items()}
        query_lexemes = sorted(idf)

        cur.execute(
            """
            SELECT n.slug,
                   sum(coalesce(cardinality(u.positions), 1)) AS doc_len,
                   array_agg(u.lexeme) FILTER (WHERE u.lexeme = ANY(%s)) AS lexemes,
                   array_agg(coalesce(cardinality(u.positions), 1))
                       FILTER (WHERE u.lexeme = ANY(%s)) AS tfs
            FROM notes n CROSS JOIN LATERAL unnest(n.search) AS u
            WHERE n.slug = ANY(%s)
            GROUP BY n.slug
            """,
            (query_lexemes, query_lexemes, candidates),
        )
        stats = [(slug, int(dl), lx or [], tf or []) for slug, dl, lx, tf in cur.fetchall()]
        if not stats:
            return [], {}, 0.0, {}
        query_idf = sum(idf.get(lx, _bm25_idf(n_notes, 0)) for lx in lexemes)
        held = {slug: sum(idf[lx] for lx in lxs) / query_idf for slug, _dl, lxs, _t in stats}
        coverage = max(held.values())
        avg_len = sum(max(1, dl) for _s, dl, _l, _t in stats) / len(stats)
        scores: dict[str, float] = {}
        for slug, dl, lxs, tfs in stats:
            norm = BM25_K1 * (1.0 - BM25_B + BM25_B * max(1, dl) / avg_len)
            scores[slug] = sum(
                idf[lx] * tf * (BM25_K1 + 1.0) / (tf + norm)
                for lx, tf in zip(lxs, tfs, strict=True)
            )
        # `(-score, slug)`: equal scores are common in small graphs, and the
        # order must depend only on the data (tests P1-3).
        ranked = sorted((s for s in scores if scores[s] > 0), key=lambda s: (-scores[s], s))[:limit]
        if not ranked:
            return [], {}, 0.0, {}

        cur.execute(
            """
            SELECT DISTINCT ON (c.note_slug) c.note_slug, c.heading, c.text
            FROM chunks c
            CROSS JOIN LATERAL (
                SELECT sum(t.idf * cardinality(u.positions) / (cardinality(u.positions) + %s))
                    AS score
                FROM unnest(c.search) AS u
                JOIN unnest(%s::text[], %s::float8[]) AS t(lex, idf) ON t.lex = u.lexeme
            ) m
            WHERE c.note_slug = ANY(%s) AND m.score > 0
            ORDER BY c.note_slug, m.score DESC, c.id ASC
            """,
            (BM25_K1, query_lexemes, [idf[lx] for lx in query_lexemes], ranked),
        )
        chunk_lookup = {slug: (heading, text) for slug, heading, text in cur.fetchall()}

    return ranked, chunk_lookup, coverage, {slug: held[slug] for slug in ranked}


def _vector_candidates(
    conn: psycopg.Connection,
    embedder: Embedder,
    query: str,
    k: int,
    *,
    scope: set[str] | None = None,
    types: list[str] | None = None,
    since: date | None = None,
    include_superseded: bool = False,
    ef_search: int | None = None,
) -> tuple[list[str], dict[str, tuple[str | None, str]]]:
    """Vector candidates by cosine distance (`<=>`) over `chunks.embedding`:
    fetch `3k` nearest chunks, dedupe to notes keeping each note's single
    best (nearest) chunk, in rank order. `<=>` because the HNSW index
    (0001_init.sql) is built with `vector_cosine_ops`.

    `scope`, `types`, `since` and `include_superseded` restrict the
    nearest-neighbour query itself, as in `_fts_candidates`, so a filter
    never drops a note that was cut from an unfiltered pool.

    `ef_search`, when given, is `SET LOCAL hnsw.ef_search` for this query:
    an HNSW scan returns at most that many chunks before any filter, so it
    bounds recall on a large graph. It lasts until the transaction ends
    (the caller's own, when one is already open). None leaves the session's
    value.
    """
    [query_vec] = embedder.embed([query])
    fetch_n = VECTOR_FETCH_MULTIPLIER * k
    scope_list = list(scope) if scope is not None else None
    status_values = _STATUS_ALL if include_superseded else _STATUS_LIVING_ONLY

    in_transaction = conn.transaction() if ef_search is not None else contextlib.nullcontext()
    with in_transaction, conn.cursor() as cur:
        if ef_search is not None:
            cur.execute(f"SET LOCAL hnsw.ef_search = {int(ef_search)}")
        cur.execute(
            f"""
            SELECT c.note_slug, c.heading, c.text, c.embedding <=> %s::vector AS distance
            FROM chunks c
            JOIN notes n ON n.slug = c.note_slug
            WHERE n.status = ANY(%s)
            {"AND c.note_slug = ANY(%s)" if scope_list is not None else ""}
            {"AND n.type = ANY(%s)" if types else ""}
            {"AND n.content_date >= %s" if since else ""}
            ORDER BY distance ASC, n.updated DESC, c.note_slug ASC
            LIMIT %s
            """,
            (
                query_vec,
                status_values,
                *([scope_list] if scope_list is not None else []),
                *([types] if types else []),
                *([_day_start(since)] if since else []),
                fetch_n,
            ),
        )
        rows = cur.fetchall()

    ranked: list[str] = []
    chunk_lookup: dict[str, tuple[str | None, str]] = {}
    for note_slug, heading, text, distance in rows:
        if (1.0 - float(distance)) < VECTOR_MIN_SIMILARITY:
            break  # rows are distance-ascending, so nothing after this clears the floor either
        if note_slug in chunk_lookup:
            continue
        chunk_lookup[note_slug] = (heading, text)
        ranked.append(note_slug)
    return ranked, chunk_lookup


def _rrf(
    lists_with_weights: list[tuple[list[str], float, str]],
    *,
    scale: dict[str, dict[str, float]] | None = None,
) -> tuple[list[str], dict[str, set[str]], dict[str, float]]:
    """Reciprocal Rank Fusion: each ranked list contributes
    `weight / (RRF_K + rank)` (rank 0-based) to every slug in it. `via`
    records which named lists each slug appeared in (the source of
    `SearchHit.via`). Sorted by `(-score, slug)` so ties are deterministic
    (tests P1-3).

    `scale`, `{list name: {slug: factor}}`, multiplies one slug's
    contribution from one list (a slug not named counts in full).
    """
    score: dict[str, float] = defaultdict(float)
    via: dict[str, set[str]] = defaultdict(set)
    for lst, weight, name in lists_with_weights:
        factors = (scale or {}).get(name) or {}
        for rank, slug in enumerate(lst):
            score[slug] += weight * factors.get(slug, 1.0) * (1.0 / (RRF_K + rank))
            via[slug].add(name)
    ordered = sorted(score, key=lambda s: (-score[s], s))
    return ordered, via, dict(score)


def _title_boost(query_terms: list[str], title: str, rrf_max: float) -> float:
    """`TITLE_BOOST_ALL` x `rrf_max` x the share of query terms that are
    whole words of the title (tokenized like `_terms()`, so "ash" does not
    match "ashgrove"). Proportional so that one generic word shared with
    many titles (a product family name) cannot lift a weak match over the
    notes both legs rank first (`tests/test_chat_title_boost.py`).
    """
    if not query_terms or rrf_max <= 0:
        return 0.0
    title_words = set(_TOKEN_RE.findall(title.lower()))
    present = sum(1 for t in query_terms if t in title_words)
    return TITLE_BOOST_ALL * rrf_max * present / len(query_terms)


def _recency_multiplier(when: datetime, *, now: datetime | None = None) -> float:
    """`0.85 + 0.15 * 0.5^(age_days/180)` from `when` (`_note_recency` picks
    the date). A date in the future counts as age 0."""
    now = now or datetime.now(UTC)
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    age_days = max(0.0, (now - when).total_seconds() / 86400.0)
    return RECENCY_FLOOR + RECENCY_SPAN * (0.5 ** (age_days / RECENCY_HALF_LIFE_DAYS))


def _note_recency(
    meta: _NoteMeta, *, now: datetime | None = None, content_date: bool | None = None
) -> float:
    """The note's recency multiplier. `content_date` off (None: the
    `search_recency_content_date` setting, off by default): from `notes.updated`,
    whose trigger resets it on any row write. On: from the newest claim's
    recorded date, and `RECENCY_UNDATED_MULTIPLIER` for a note with no live
    claims. Off by default because on it cost one of 15 paraphrased demo
    questions (01-spec/content-dates-2026-09-25.md)."""
    if content_date is None:
        content_date = _settings().search_recency_content_date
    if not content_date:
        return _recency_multiplier(meta.updated, now=now)
    if meta.content_date_source == "note_updated" and RECENCY_UNDATED_MULTIPLIER is not None:
        return RECENCY_UNDATED_MULTIPLIER
    return _recency_multiplier(meta.content_date, now=now)


def _load_notes_meta(conn: psycopg.Connection, slugs: list[str]) -> dict[str, _NoteMeta]:
    if not slugs:
        return {}
    with conn.cursor() as cur:
        cur.execute(
            "SELECT slug, title, type, content_date, updated, content_date_source "
            "FROM notes WHERE slug = ANY(%s)",
            (list(slugs),),
        )
        return {row[0]: _NoteMeta(*row[1:]) for row in cur.fetchall()}


def _source_slugs_for(conn: psycopg.Connection, note_slug: str) -> list[str]:
    return _source_slugs(conn, [note_slug])[note_slug]


def _source_slugs(conn: psycopg.Connection, note_slugs: list[str]) -> dict[str, list[str]]:
    """`{note_slug: its provenance sources, sorted}` for every slug given, in
    one read."""
    out: dict[str, list[str]] = {slug: [] for slug in note_slugs}
    if not note_slugs:
        return out
    with conn.cursor() as cur:
        cur.execute(
            "SELECT from_slug, to_slug FROM links "
            "WHERE from_slug = ANY(%s) AND kind = 'provenance' ORDER BY from_slug, to_slug",
            (list(note_slugs),),
        )
        for from_slug, to_slug in cur.fetchall():
            out[from_slug].append(to_slug)
    return out


def _fallback_chunks(
    conn: psycopg.Connection, note_slugs: list[str]
) -> dict[str, tuple[str | None, str]]:
    """`_fallback_chunk` for several notes in one read."""
    if not note_slugs:
        return {}
    with conn.cursor() as cur:
        cur.execute(
            "SELECT DISTINCT ON (note_slug) note_slug, heading, text FROM chunks "
            "WHERE note_slug = ANY(%s) ORDER BY note_slug, id",
            (list(note_slugs),),
        )
        return {slug: (heading, text) for slug, heading, text in cur.fetchall()}


def _fallback_chunk(conn: psycopg.Connection, note_slug: str) -> tuple[str | None, str]:
    """A representative chunk for a note neither engine matched at chunk
    level (an `expand` neighbour, or a keyword hit with no matching chunk):
    its lowest-id chunk, the title chunk when the note has a title."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT heading, text FROM chunks WHERE note_slug = %s ORDER BY id LIMIT 1",
            (note_slug,),
        )
        row = cur.fetchone()
    return (row[0], row[1]) if row else (None, "")


def _most_common_title_word(conn: psycopg.Connection) -> str | None:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT word FROM (
                SELECT regexp_split_to_table(lower(title), '[^a-z0-9]+') AS word FROM notes
            ) words
            WHERE length(word) > 2
            GROUP BY word
            ORDER BY count(*) DESC, word ASC
            LIMIT 1
            """
        )
        row = cur.fetchone()
    return row[0] if row else None


def _control_ok(conn: psycopg.Connection) -> bool:
    """Run when the real query returned zero hits (spec sec 9). Takes the
    most common title word (longer than two characters) and requires a
    full-text hit for it in `notes`, a non-empty `chunks`, and a chunk hit
    for it. `True` means the zero is a genuine "nothing recorded"; `False`
    (no such word, or any probe empty) means the index looks broken or
    empty. Checking `chunks` too catches a dead chunk index, which the
    `notes.search` column alone would not show.
    """
    control_term = _most_common_title_word(conn)
    if control_term is None:
        return False
    with conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM notes WHERE search @@ websearch_to_tsquery('english', %s) LIMIT 1",
            (control_term,),
        )
        if cur.fetchone() is None:
            return False
        cur.execute("SELECT 1 FROM chunks LIMIT 1")
        if cur.fetchone() is None:
            return False
        cur.execute(
            "SELECT 1 FROM chunks WHERE search @@ websearch_to_tsquery('english', %s) LIMIT 1",
            (control_term,),
        )
        return cur.fetchone() is not None


def _build_hit(
    conn: psycopg.Connection,
    slug: str,
    score: float,
    via: set[str],
    notes_meta: dict[str, _NoteMeta],
    chunk_lookup: dict[str, tuple[str | None, str]],
    *,
    expanded_from: str | None = None,
    sources: dict[str, list[str]] | None = None,
) -> SearchHit:
    """`sources` is `_source_slugs` already read for a batch of hits; without
    it, or for a slug it does not name, the hit's sources are read here."""
    meta = notes_meta[slug]
    heading, text = chunk_lookup.get(slug) or _fallback_chunk(conn, slug)
    if not text:
        heading, text = _fallback_chunk(conn, slug)
    source_slugs = (sources or {}).get(slug)
    if source_slugs is None:
        source_slugs = _source_slugs_for(conn, slug)
    return SearchHit(
        note_slug=slug,
        title=meta.title,
        score=round(score, 6),
        via=sorted(via),  # type: ignore[arg-type]
        best_chunk=text,
        heading=heading,
        updated=_world_day(meta.updated),
        source_slugs=source_slugs,
        expanded_from=expanded_from,
        content_date=_world_day(meta.content_date),
        content_date_source=meta.content_date_source,  # type: ignore[arg-type]
    )


def _world_day(value: datetime | date) -> date:
    """The deployment zone's calendar day, not the session's: `.date()` on a
    UTC timestamp reads tomorrow for evening writes west of UTC."""
    if isinstance(value, datetime):
        return clock.world_date(value) if value.tzinfo is not None else value.date()
    return value


def _expand(
    conn: psycopg.Connection,
    hits: list[SearchHit],
    *,
    hops: int,
) -> list[SearchHit]:
    """Follow `body`/`parent` links out of the top `EXPAND_FRONTIER` hits,
    adding neighbour notes not already present, scored at `EXPAND_DISCOUNT`
    x the originating hit's score. Each further hop starts from the first
    `EXPAND_FRONTIER` newly added neighbours.
    """
    seen = {h.note_slug for h in hits}
    all_hits = list(hits)
    frontier = hits[:EXPAND_FRONTIER]

    for _ in range(hops):
        next_frontier: list[SearchHit] = []
        for hit in frontier:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT DISTINCT to_slug FROM links "
                    "WHERE from_slug = %s AND kind IN ('body', 'parent')",
                    (hit.note_slug,),
                )
                neighbour_slugs = [row[0] for row in cur.fetchall()]
            if not neighbour_slugs:
                continue
            meta = _load_notes_meta(conn, neighbour_slugs)
            for n_slug in neighbour_slugs:
                if n_slug in seen or n_slug not in meta:
                    continue  # already present, or to_slug points at a source, not a note
                seen.add(n_slug)
                new_hit = _build_hit(
                    conn,
                    n_slug,
                    hit.score * EXPAND_DISCOUNT,
                    set(),
                    meta,
                    {},
                    expanded_from=hit.note_slug,
                )
                all_hits.append(new_hit)
                next_frontier.append(new_hit)
        frontier = next_frontier[:EXPAND_FRONTIER]
        if not frontier:
            break

    return all_hits


@dataclass
class Legs:
    """What the two retrieval legs returned for one query: everything
    `_rank` needs, so a bench can fetch it once and rank it many ways."""

    query: str
    fts_rank: list[str]
    fts_chunks: dict[str, tuple[str | None, str]]
    vector_rank: list[str]
    vector_chunks: dict[str, tuple[str | None, str]]
    coverage: float  # `_fts_leg`'s routing signal
    # {note_slug: the share of the query's idf that note holds}, for the keyword
    # leg's notes (`_keyword_leg`). Empty (legs recorded before it existed):
    # every keyword rank counts in full.
    fts_share: dict[str, float] = field(default_factory=dict)


def search(
    conn: psycopg.Connection,
    embedder: Embedder,
    query: str,
    *,
    k: int = 12,
    expand: int = 0,
    types: list[str] | None = None,
    since: date | None = None,
    note_slugs: list[str] | None = None,
    entity_slugs: list[str] | None = None,
    include_superseded: bool = False,
    ef_search: int | None = None,
    fusion_routing: bool | None = None,
    reranker: Reranker | Literal[False] | None = None,
    rerank_pool: int | None = None,
    recency_content_date: bool | None = None,
) -> SearchResult:
    """Fused keyword + vector search (see the module docstring and
    01-spec/portolan-spec-v2.md sec 9).

    `ef_search` is the vector leg's `hnsw.ef_search`; None means the
    deployment's `search_hnsw_ef_search` setting (`portolan.config.settings`).
    Likewise `fusion_routing` (None: `search_fusion_routing`; see
    `_route_weights`), `reranker` (None: the `search_rerank` model, or none
    when that is "off"; False: never rerank) and `rerank_pool` (None:
    `search_rerank_pool`) and `recency_content_date` (None:
    `search_recency_content_date`; see `_note_recency`).

    `note_slugs` and `entity_slugs` narrow the search space itself, not just
    the reported results: `note_slugs` to that exact set, `entity_slugs` to
    the entities plus their `body`/`parent` neighbours
    (`_notes_linked_to_entities`). Given both, a note must satisfy both.

    `types`, `since` and `include_superseded` are pushed into each engine's
    SQL too, never applied after fusion: filtering a truncated global pool
    would drop real matches and report them as "nothing recorded". `types`
    is checked against `NoteType` first, so a typo ("signals") raises
    `ValueError` instead of returning a healthy-looking empty result.
    """
    if types:
        valid_types = {t.value for t in NoteType}
        invalid = sorted(set(types) - valid_types)
        if invalid:
            raise ValueError(
                f"Unknown note type(s): {', '.join(invalid)}. "
                f"Valid types: {', '.join(sorted(valid_types))}."
            )

    scope: set[str] | None = None
    if note_slugs is not None:
        scope = set(note_slugs)
    if entity_slugs:
        linked = _notes_linked_to_entities(conn, entity_slugs)
        scope = linked if scope is None else (scope & linked)

    if scope is not None and not scope:
        return SearchResult(
            hits=[], control_ok=_control_ok(conn), query_terms_unmatched=_terms(query)
        )

    legs = _legs(
        conn,
        embedder,
        query,
        k,
        scope=scope,
        types=types,
        since=since,
        include_superseded=include_superseded,
        ef_search=ef_search,
    )
    return _rank(
        conn,
        legs,
        k=k,
        expand=expand,
        fusion_routing=fusion_routing,
        reranker=reranker,
        rerank_pool=rerank_pool,
        recency_content_date=recency_content_date,
    )


def _settings() -> Settings:
    """The process's settings, read once at import: `get_settings()` rebuilds
    them on every call (about 3 ms, a tenth of a demo-graph search)."""
    from portolan import config  # local: importing this module loads no settings

    return config.settings


def _legs(
    conn: psycopg.Connection,
    embedder: Embedder,
    query: str,
    k: int,
    *,
    scope: set[str] | None = None,
    types: list[str] | None = None,
    since: date | None = None,
    include_superseded: bool = False,
    ef_search: int | None = None,
) -> Legs:
    """Both retrieval legs for `search(k=k)`, filters already resolved."""
    fts_rank, fts_chunks, coverage, fts_share = _keyword_leg(
        conn,
        query,
        max(k * 3, FTS_CANDIDATE_POOL),
        scope=scope,
        types=types,
        since=since,
        include_superseded=include_superseded,
    )
    if ef_search is None:
        ef_search = _settings().search_hnsw_ef_search
    vector_rank, vector_chunks = _vector_candidates(
        conn,
        embedder,
        query,
        k,
        scope=scope,
        types=types,
        since=since,
        include_superseded=include_superseded,
        ef_search=ef_search,
    )
    return Legs(query, fts_rank, fts_chunks, vector_rank, vector_chunks, coverage, fts_share)


def _keyword_scale(share: dict[str, float]) -> dict[str, float]:
    """Each keyword-leg note's factor on its own keyword rank: its share of
    the query's idf over `KEYWORD_SHARE_FULL`, at most 1 and at least
    `KEYWORD_SHARE_FLOOR`."""
    if not KEYWORD_SHARE_FULL:
        return {}
    return {
        slug: max(KEYWORD_SHARE_FLOOR, min(1.0, held / KEYWORD_SHARE_FULL))
        for slug, held in share.items()
    }


def _route_weights(coverage: float) -> tuple[float, float]:
    """(keyword, vector) RRF weights for a query whose best keyword match
    holds `coverage` of its idf (`_fts_leg`), when `search_fusion_routing`
    is on: the keyword-led profile when one note holds most of the query's
    rare words (names, ids), the semantic profile otherwise (a paraphrase)."""
    if coverage >= ROUTE_COVERAGE_THRESHOLD:
        return ROUTE_KEYWORD_WEIGHTS
    return ROUTE_SEMANTIC_WEIGHTS


def _rerank_passages(
    conn: psycopg.Connection,
    slugs: list[str],
    chunk_lookup: dict[str, tuple[str | None, str]],
) -> list[str]:
    """What a reranker reads for each note: its title chunk (chunk 0, title
    and description), then its cited chunk, or its first body chunk when the
    cited one is the title chunk. The tokenizer cuts the tail past the
    model's length limit."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT note_slug, heading, text FROM (
                SELECT note_slug, heading, text,
                       row_number() OVER (PARTITION BY note_slug ORDER BY id) AS n
                FROM chunks WHERE note_slug = ANY(%s)
            ) c
            WHERE n <= 2
            ORDER BY note_slug, n
            """,
            (slugs,),
        )
        first_two: dict[str, list[tuple[str | None, str]]] = defaultdict(list)
        for slug, heading, text in cur.fetchall():
            first_two[slug].append((heading, text))
    out = []
    for slug in slugs:
        chunks = first_two.get(slug) or [(None, "")]
        title_text = chunks[0][1]
        heading, text = chunk_lookup.get(slug, (None, ""))
        if not text or text == title_text:
            heading, text = chunks[1] if len(chunks) > 1 else (None, "")
        out.append("\n".join(part for part in (title_text, heading, text) if part))
    return out


def _rank(
    conn: psycopg.Connection,
    legs: Legs,
    *,
    k: int,
    expand: int = 0,
    fusion_routing: bool | None = None,
    reranker: Reranker | Literal[False] | None = None,
    rerank_pool: int | None = None,
    recency_content_date: bool | None = None,
) -> SearchResult:
    """Fusion, title boost, recency, the optional rerank, hits, `expand` and
    the unmatched-terms scan: everything `search()` does after the legs."""
    query_terms = _terms(legs.query)
    if fusion_routing is None:
        fusion_routing = _settings().search_fusion_routing
    fts_weight, vector_weight = (
        _route_weights(legs.coverage) if fusion_routing else (FTS_WEIGHT, VECTOR_WEIGHT)
    )
    fused_slugs, via_map, rrf_scores = _rrf(
        [(legs.fts_rank, fts_weight, "fts"), (legs.vector_rank, vector_weight, "vector")],
        scale={"fts": _keyword_scale(legs.fts_share)},
    )

    notes_meta = _load_notes_meta(conn, fused_slugs)
    # Filters were applied in each engine's SQL; this only drops a slug whose
    # note no longer resolves.
    candidates = [s for s in fused_slugs if s in notes_meta]

    if not candidates:
        return SearchResult(
            hits=[], control_ok=_control_ok(conn), query_terms_unmatched=query_terms
        )

    rrf_max = max(rrf_scores[s] for s in candidates)
    # One clock reading per query, so notes with equal `updated` tie exactly
    # and the slug decides (tests/test_rerank_eval.py).
    now = datetime.now(UTC)
    if recency_content_date is None:
        recency_content_date = _settings().search_recency_content_date
    scored: list[tuple[str, float]] = []
    for slug in candidates:
        meta = notes_meta[slug]
        boost = _title_boost(query_terms, meta.title, rrf_max)
        final = (rrf_scores[slug] + boost) * _note_recency(
            meta, now=now, content_date=recency_content_date
        )
        scored.append((slug, final))
    # `(-score, slug)` for deterministic ties, as in `_rrf`.
    scored.sort(key=lambda pair: (-pair[1], pair[0]))

    chunk_lookup = {
        **legs.vector_chunks,
        **legs.fts_chunks,
    }  # fts wins ties: it matched the literal query text

    if reranker is None:
        from portolan.search.rerank import get_reranker

        reranker = get_reranker(_settings()) or False
    if reranker:
        pool = max(k, _settings().search_rerank_pool if rerank_pool is None else rerank_pool)
        head = [slug for slug, _score in scored[:pool]]
        relevance = reranker.score(legs.query, _rerank_passages(conn, head, chunk_lookup))
        # The reranker replaces fusion and the title boost (it reads the title);
        # recency still multiplies. Past the pool nothing is reranked or returned.
        scored = sorted(
            (
                (
                    slug,
                    rel
                    * _note_recency(notes_meta[slug], now=now, content_date=recency_content_date),
                )
                for slug, rel in zip(head, relevance, strict=True)
            ),
            key=lambda pair: (-pair[1], pair[0]),
        )

    top = [slug for slug, _score in scored[:k]]
    # One read each for the hits' sources and for the chunk of any hit neither
    # leg matched at chunk level, not one per hit.
    sources = _source_slugs(conn, top)
    missing = [slug for slug in top if not (chunk_lookup.get(slug) or (None, ""))[1]]
    chunk_lookup.update(_fallback_chunks(conn, missing))
    hits = [
        _build_hit(conn, slug, score, via_map[slug], notes_meta, chunk_lookup, sources=sources)
        for slug, score in scored[:k]
    ]

    if expand:
        hits = _expand(conn, hits, hops=expand)

    # Whole-word matching, as in `_title_boost`.
    matched_terms: set[str] = set()
    for hit in hits:
        blob = f"{hit.title} {hit.best_chunk}".lower()
        blob_words = set(_TOKEN_RE.findall(blob))
        matched_terms |= {t for t in query_terms if t in blob_words}
    unmatched = _terms_in_no_note(
        conn, sorted(set(query_terms) - matched_terms), [h.note_slug for h in hits]
    )

    return SearchResult(hits=hits, control_ok=True, query_terms_unmatched=unmatched)


def _terms_in_no_note(
    conn: psycopg.Connection, terms: list[str], note_slugs: list[str]
) -> list[str]:
    """The `terms` that none of `note_slugs` holds anywhere (title,
    description or body), compared as the keyword leg compares them
    (`notes.search` lexemes), so "renewals" is found in a note that says
    "renewal" and a word in a chunk other than the one shown still counts. A
    caller reads an unmatched term as "nothing on file about this"."""
    if not terms or not note_slugs:
        return terms
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT t.term
            FROM unnest(%s::text[]) AS t(term)
            CROSS JOIN LATERAL plainto_tsquery('english', portolan_search_text(t.term)) AS tq
            WHERE numnode(tq) > 0
              AND NOT EXISTS (
                  SELECT 1 FROM notes n WHERE n.slug = ANY(%s) AND n.search @@ tq
              )
            ORDER BY t.term
            """,
            (terms, note_slugs),
        )
        return [row[0] for row in cur.fetchall()]


def explain(result: SearchResult) -> str:
    """Short text block per hit; the `python -m portolan.search` CLI prints it."""
    if not result.hits:
        if result.control_ok:
            lines = [
                "No results. The index looks healthy (a control query for a word known "
                'to exist in the graph came back) -- this is a genuine "nothing recorded", '
                "not a broken search."
            ]
        else:
            lines = [
                "No results, AND the control query also found nothing. The search index "
                "looks broken or empty, not merely lacking an answer to this query -- "
                'do not report this as "nothing recorded" without checking the index.'
            ]
        if result.query_terms_unmatched:
            lines.append("Query terms: " + ", ".join(result.query_terms_unmatched))
        return "\n".join(lines)

    lines = []
    for i, hit in enumerate(result.hits, 1):
        via = "+".join(hit.via) if hit.via else "expand"
        heading = f" ({hit.heading})" if hit.heading else ""
        expanded = f" [via link from {hit.expanded_from}]" if hit.expanded_from else ""
        sources = f" | sources: {', '.join(hit.source_slugs)}" if hit.source_slugs else ""
        snippet = " ".join(hit.best_chunk.split())[:220]
        lines.append(
            f"{i}. {hit.title} [{hit.note_slug}] via={via} score={hit.score:.4f} "
            f"updated={hit.updated.isoformat()}{heading}{expanded}\n"
            f"   › {snippet}{sources}"
        )
    if result.query_terms_unmatched:
        lines.append("Unmatched query terms: " + ", ".join(result.query_terms_unmatched))
    return "\n".join(lines)
