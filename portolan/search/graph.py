"""Graph-aware retrieval over Portolan's own note graph, three modes behind one flag.

`graph_search(..., mode=...)` wraps `portolan.search.query.search` (mode `"off"` is
exactly that call) and adds one of:

- `"expand"`: seed and expand. The fused search's top `SEED_K` notes are origins; every
  note within `hops` shared-entity steps of one of them gets a proximity bonus scaled by
  that seed's score.
- `"entity"`: entity first. Accounts, contacts, products and competitors named in the
  question (registry names and aliases, `detect_entities`) are the origins instead.
- `"ppr"`: personalized PageRank (HippoRAG's idea) over the local graph, restarting at
  the named entities and the search seeds.

Every candidate (the search's own top k plus every note the graph reached) is then
scored `text * (1 + MODE_WEIGHT[mode] * proximity)` (`COMBINE`). `text` is the search's
own score divided by its best; a note only the graph reached scores below the search's
last hit, ordered by the same fused relevance computed for it with the whole graph's
word statistics. `proximity` is in [0, 1] and never comes from a note's own seed. With
no proximity (no seed reaches anything, no entity named) the search's result stands.

One step (`_step`) moves between content notes (signals, actions, ...) through the entity
notes they link to: a signal steps to its account, the account's descendants (the CRM
hierarchy, spec sec 3: rollups walk the `parent` chain) and every other note linked to
them. An entity linked to more than `HUB_MAX_CONTENT` content notes (a product every
signal mentions) is not stepped through unless the question names it.

`window=(since, until)` keeps only candidates holding a claim recorded (world clock,
`claims.recorded`) in that range: "what changed for account X last week".

Measured in `01-spec/bench-graph-retrieval-2026-09-23.md`; the MCP `search` tool does
not call this module.
"""

from __future__ import annotations

import re
import time
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date
from typing import Literal

import psycopg

from portolan.search.embed import Embedder
from portolan.search.query import (
    _STATUS_ALL,
    _STATUS_LIVING_ONLY,
    BM25_B,
    BM25_K1,
    FTS_WEIGHT,
    QUERY_TSVECTOR_SQL,
    VECTOR_MIN_SIMILARITY,
    VECTOR_WEIGHT,
    SearchHit,
    SearchResult,
    _bm25_idf,
    _build_hit,
    _keyword_scale,
    _load_notes_meta,
    _note_recency,
    _rrf,
    _terms,
    _title_boost,
    _tsquery_literal,
    search,
)
from portolan.util import clock

GraphMode = Literal["off", "expand", "entity", "ppr"]
GRAPH_MODES: tuple[str, ...] = ("off", "expand", "entity", "ppr")

# Note types the walk steps *through*; every other type is content.
ENTITY_TYPES = frozenset({"account", "contact", "product", "competitor"})
# Registry kinds a question can name. `other` (the vendor itself, its branches) would
# anchor every question.
_DETECTABLE_KINDS = ("account", "contact", "product", "competitor", "internal_person")

SEED_K = 10
DEFAULT_HOPS = 1
# Search hits a `window` filters, as `search()` floors its keyword pool.
WINDOW_POOL = 30
POOL_CAP = 300
HUB_MAX_CONTENT = 30
# Up to this many content notes an entity passes its full weight on; above it the weight
# falls as 1/degree (HippoRAG's node specificity), named entities included.
HUB_SOFT_DEGREE = 10
STEP_DECAY = 0.5
# An entity note reached on the way (an account page, a product page) is context, not
# the answer: its proximity is scaled by this.
BRIDGE_WEIGHT = 0.5
MODE_WEIGHT = {"expand": 0.5, "entity": 1.0, "ppr": 1.0}
# "multiply": text x (1 + weight x proximity), so the graph lifts only notes the question
# also matches; "add": text + weight x proximity.
COMBINE: Literal["multiply", "add"] = "multiply"
# Text score, as a share of the search's last hit, of a graph-reached note the question
# does not match at all.
GRAPH_ONLY_FLOOR = 0.1
PPR_RESTART = 0.5  # HippoRAG's damping
PPR_ITERATIONS = 30
# Longest registry name, in words, tried when matching a question.
_MAX_ALIAS_WORDS = 6
_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9&'’.\-]*")


# ---------------------------------------------------------------------------
# Entity detection (registry lookup, no model)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EntityMatch:
    """A registry entity named in the question, and the notes that stand for it:
    its own note, or its account's note for a contact with none of its own."""

    surface: str
    kind: str
    name: str
    anchor_slugs: tuple[str, ...]


def _norm_word(word: str) -> str:
    word = word.lower().replace("’", "'")
    if word.endswith("'s"):
        word = word[:-2]
    return re.sub(r"[^a-z0-9&\-]", "", word)


def _case_sensitive(alias: str) -> bool:
    """Short one-word aliases ("T2", "AH", "Link") match only as written: lowercased
    they are ordinary words or initials."""
    return " " not in alias.strip() and (len(alias.strip()) <= 4 or alias.isupper())


def _registry_index(conn: psycopg.Connection) -> dict[str, list[tuple[str, int]]]:
    """{normalised name or alias: [(surface as stored, entity id)]}."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT e.canonical_name, e.id FROM registry_entities e
            WHERE e.active AND e.kind = ANY(%s)
            UNION
            SELECT a.alias, e.id FROM registry_aliases a
            JOIN registry_entities e ON e.id = a.entity_id
            WHERE e.active AND e.kind = ANY(%s)
            """,
            (list(_DETECTABLE_KINDS), list(_DETECTABLE_KINDS)),
        )
        rows = cur.fetchall()
    index: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for surface, entity_id in rows:
        if not surface or surface.strip().lower().startswith("the "):
            continue
        key = " ".join(_norm_word(w) for w in _WORD_RE.findall(surface))
        if key:
            index[key].append((surface.strip(), int(entity_id)))
    return index


def detect_entities(conn: psycopg.Connection, question: str) -> list[EntityMatch]:
    """Registry entities named in `question`, longest name first, left to right.
    A name shared by two entities ("Ashgrove") anchors both."""
    index = _registry_index(conn)
    words = _WORD_RE.findall(question)
    norm = [_norm_word(w) for w in words]
    matched: dict[int, str] = {}
    i = 0
    while i < len(words):
        step = 1
        for n in range(min(_MAX_ALIAS_WORDS, len(words) - i), 0, -1):
            candidates = index.get(" ".join(norm[i : i + n]))
            if not candidates:
                continue
            written = " ".join(w.rstrip(".,;:?!") for w in words[i : i + n])
            written = written.removesuffix("'s").removesuffix("’s")
            ok = [
                (surface, eid)
                for surface, eid in candidates
                if not _case_sensitive(surface) or written == surface
            ]
            if ok:
                for surface, eid in ok:
                    matched.setdefault(eid, surface)
                step = n
                break
        i += step
    if not matched:
        return []
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT e.id, e.kind, e.canonical_name, e.note_slug, p.note_slug
            FROM registry_entities e
            LEFT JOIN registry_entities p ON p.id = e.parent_id
            WHERE e.id = ANY(%s)
            ORDER BY e.id
            """,
            (list(matched),),
        )
        rows = cur.fetchall()
    out: list[EntityMatch] = []
    for eid, kind, name, note_slug, parent_note in rows:
        anchor = note_slug or (parent_note if kind == "contact" else None)
        if anchor:
            out.append(EntityMatch(matched[eid], kind, name, (anchor,)))
    return out


# ---------------------------------------------------------------------------
# The graph: links (body, parent) in both directions, plus the account hierarchy
# ---------------------------------------------------------------------------


@dataclass
class _Graph:
    conn: psycopg.Connection
    statuses: list[str]
    types: dict[str, str] = field(default_factory=dict)
    _adj: dict[str, set[str]] = field(default_factory=dict)
    _desc: dict[str, set[str]] = field(default_factory=dict)

    def load(self, slugs: Iterable[str]) -> None:
        """Neighbours (both directions, `body`/`parent` links, notes of an allowed
        status) of every slug not loaded yet: one query per call."""
        todo = [s for s in set(slugs) if s not in self._adj]
        if not todo:
            return
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT slug, type FROM notes WHERE slug = ANY(%s) AND status = ANY(%s)",
                (todo, self.statuses),
            )
            self.types.update(dict(cur.fetchall()))
            cur.execute(
                """
                SELECT l.from_slug, nf.type, l.to_slug, nt.type
                FROM links l
                JOIN notes nf ON nf.slug = l.from_slug AND nf.status = ANY(%s)
                JOIN notes nt ON nt.slug = l.to_slug AND nt.status = ANY(%s)
                WHERE l.kind IN ('body', 'parent')
                  AND (l.from_slug = ANY(%s) OR l.to_slug = ANY(%s))
                """,
                (self.statuses, self.statuses, todo, todo),
            )
            rows = cur.fetchall()
        wanted = set(todo)
        for s in todo:
            self._adj[s] = set()
        for a, a_type, b, b_type in rows:
            if a == b:
                continue
            self.types.setdefault(a, a_type)
            self.types.setdefault(b, b_type)
            if a in wanted:
                self._adj[a].add(b)
            if b in wanted:
                self._adj[b].add(a)

    def neighbours(self, slug: str) -> set[str]:
        self.load([slug])
        return self._adj[slug]

    def descendants(self, accounts: Iterable[str]) -> dict[str, set[str]]:
        """Every account below each of `accounts` in the `parent` hierarchy."""
        accounts = list(accounts)
        todo = [a for a in set(accounts) if a not in self._desc]
        if todo:
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    WITH RECURSIVE sub(root, slug) AS (
                        SELECT r, r FROM unnest(%s::text[]) AS r
                        UNION
                        SELECT sub.root, n.slug FROM notes n JOIN sub ON n.parent_slug = sub.slug
                        WHERE n.status = ANY(%s)
                    )
                    SELECT root, slug FROM sub WHERE root <> slug
                    """,
                    (todo, self.statuses),
                )
                rows = cur.fetchall()
            for a in todo:
                self._desc[a] = set()
            for root, slug in rows:
                self._desc[root].add(slug)
                self.types.setdefault(slug, "account")
        return {a: self._desc[a] for a in accounts}

    def is_entity(self, slug: str) -> bool:
        return self.types.get(slug) in ENTITY_TYPES

    def content_degree(self, slug: str) -> int:
        return sum(1 for n in self.neighbours(slug) if not self.is_entity(n))


def _step(
    graph: _Graph, frontier: dict[str, float], *, protected: set[str]
) -> tuple[dict[str, float], dict[str, float]]:
    """One shared-entity step from `frontier` ({slug: weight}): (content notes reached,
    entity notes reached), each {slug: weight}. A content note passes its weight to its
    entities, an account to its descendants, and each entity to its content notes,
    scaled down above `HUB_SOFT_DEGREE` of them."""
    graph.load(frontier)
    entities: dict[str, float] = {}
    content: dict[str, float] = {}

    def offer(target: dict[str, float], slug: str, w: float) -> None:
        if w > target.get(slug, 0.0):
            target[slug] = w

    for slug, w in frontier.items():
        if graph.is_entity(slug):
            offer(entities, slug, w)
            continue
        for n in graph.neighbours(slug):
            offer(entities if graph.is_entity(n) else content, n, w)
    accounts = [e for e in entities if graph.types.get(e) == "account"]
    for root, below in graph.descendants(accounts).items():
        for d in below:
            offer(entities, d, entities[root])
    graph.load(entities)
    contacts: dict[str, float] = {}
    for e, w in entities.items():
        degree = graph.content_degree(e)
        if degree > HUB_MAX_CONTENT and e not in protected:
            continue
        scale = min(1.0, HUB_SOFT_DEGREE / max(1, degree))
        for n in graph.neighbours(e):
            if graph.is_entity(n):
                if graph.types.get(n) == "contact":
                    offer(contacts, n, w * scale)
            else:
                offer(content, n, w * scale)
    for n, w in contacts.items():
        offer(entities, n, w)
    return content, entities


def _neighbourhood(
    graph: _Graph, origin: str, hops: int, *, protected: set[str]
) -> dict[str, float]:
    """{slug: weight in (0, 1]} within `hops` steps of `origin` (itself excluded): 1.0
    one step away, `STEP_DECAY` per further step, `BRIDGE_WEIGHT` on entity notes except
    the accounts below an account origin (a rollup's members are what it asks about)."""
    best: dict[str, float] = {}
    graph.load([origin])
    members = graph.descendants([origin])[origin] if graph.types.get(origin) == "account" else set()
    frontier = {origin: 1.0}
    for step in range(max(0, hops)):
        decay = STEP_DECAY if step else 1.0
        content, entities = _step(graph, frontier, protected=protected)
        frontier = {}
        for slug, w in content.items():
            w *= decay
            if slug != origin and w > best.get(slug, 0.0):
                best[slug] = w
                frontier[slug] = w
        for slug, w in entities.items():
            w *= decay * (1.0 if slug in members else BRIDGE_WEIGHT)
            if slug != origin and w > best.get(slug, 0.0):
                best[slug] = w
        if not frontier:
            break
    return best


def _proximity(
    graph: _Graph, origins: dict[str, float], hops: int, *, protected: set[str]
) -> dict[str, tuple[float, str]]:
    """{slug: (max over origins of origin weight x neighbourhood weight, that origin)},
    capped at `POOL_CAP` notes."""
    out: dict[str, tuple[float, str]] = {}
    for origin, ow in origins.items():
        for slug, w in _neighbourhood(graph, origin, hops, protected=protected).items():
            if ow * w > out.get(slug, (0.0, ""))[0]:
                out[slug] = (ow * w, origin)
    ranked = sorted(out.items(), key=lambda kv: (-kv[1][0], kv[0]))[:POOL_CAP]
    return dict(ranked)


def _ppr(graph: _Graph, nodes: Iterable[str], restart: dict[str, float]) -> dict[str, float]:
    """Personalized PageRank over the subgraph induced by `nodes` (undirected, the
    account hierarchy included), restarting to `restart` with probability
    `PPR_RESTART`. Dangling mass goes back to the restart vector."""
    node_set = set(nodes) | set(restart)
    graph.load(node_set)
    adj: dict[str, set[str]] = {n: set() for n in node_set}
    for n in node_set:
        for m in graph.neighbours(n):
            if m in node_set:
                adj[n].add(m)
                adj[m].add(n)
    accounts = [n for n in node_set if graph.types.get(n) == "account"]
    for root, below in graph.descendants(accounts).items():
        for d in below & node_set:
            adj[root].add(d)
            adj[d].add(root)
    total = sum(restart.values()) or 1.0
    r = {n: restart.get(n, 0.0) / total for n in node_set}
    p = dict(r)
    for _ in range(PPR_ITERATIONS):
        nxt = {n: PPR_RESTART * r[n] for n in node_set}
        dangling = 0.0
        for n, mass in p.items():
            if not mass:
                continue
            if not adj[n]:
                dangling += mass
                continue
            share = (1.0 - PPR_RESTART) * mass / len(adj[n])
            for m in adj[n]:
                nxt[m] += share
        if dangling:
            for n in node_set:
                nxt[n] += (1.0 - PPR_RESTART) * dangling * r[n]
        p = nxt
    return p


# ---------------------------------------------------------------------------
# Text relevance for a candidate set, with the whole graph's statistics
# ---------------------------------------------------------------------------


class _QueryVectorCache:
    """Embeds each distinct text once per `graph_search` call: the global search and
    the candidate scoring share the query vector."""

    def __init__(self, inner: Embedder) -> None:
        self.inner = inner
        self.dim = inner.dim
        self._cache: dict[str, list[float]] = {}

    def embed(self, texts: list[str]) -> list[list[float]]:
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            for t, v in zip(missing, self.inner.embed(missing), strict=True):
                self._cache[t] = v
        return [self._cache[t] for t in texts]


def _bm25_rank(
    conn: psycopg.Connection, query: str, slugs: list[str], statuses: list[str]
) -> tuple[list[str], dict[str, tuple[str | None, str]], dict[str, float]]:
    """Okapi BM25 (the keyword leg's parameters) for `slugs` only, with document
    frequencies over every note of an allowed status, so a candidate set never changes
    a word's weight. Returns (ranked slugs with a positive score, {slug: best chunk},
    {slug: the share of the query's idf the note holds}, as `_keyword_leg` reports it)."""
    with conn.cursor() as cur:
        cur.execute(f"SELECT lexeme FROM unnest({QUERY_TSVECTOR_SQL}) ORDER BY lexeme", (query,))
        lexemes = [r[0] for r in cur.fetchall()]
        if not lexemes or not slugs:
            return [], {}, {}
        cur.execute(
            """
            SELECT t.lx, (SELECT count(*) FROM notes n
                          WHERE n.search @@ t.lit::tsquery AND n.status = ANY(%s)),
                   (SELECT count(*) FROM notes n WHERE n.status = ANY(%s))
            FROM unnest(%s::text[], %s::text[]) AS t(lx, lit)
            """,
            (statuses, statuses, lexemes, [_tsquery_literal(lx) for lx in lexemes]),
        )
        rows = cur.fetchall()
        n_notes = int(rows[0][2]) if rows else 0
        idf = {lx: _bm25_idf(n_notes, int(df)) for lx, df, _n in rows if df}
        if not idf:
            return [], {}, {}
        query_lexemes = sorted(idf)
        cur.execute(
            """
            SELECT n.slug,
                   sum(coalesce(cardinality(u.positions), 1)) AS doc_len,
                   array_agg(u.lexeme) FILTER (WHERE u.lexeme = ANY(%s)),
                   array_agg(coalesce(cardinality(u.positions), 1))
                       FILTER (WHERE u.lexeme = ANY(%s))
            FROM notes n CROSS JOIN LATERAL unnest(n.search) AS u
            WHERE n.slug = ANY(%s)
            GROUP BY n.slug
            """,
            (query_lexemes, query_lexemes, slugs),
        )
        stats = [(s, int(dl), lx or [], tf or []) for s, dl, lx, tf in cur.fetchall()]
        if not stats:
            return [], {}, {}
        query_idf = sum(idf.get(lx, _bm25_idf(n_notes, 0)) for lx in lexemes)
        share = {s: sum(idf[lx] for lx in lxs) / query_idf for s, _dl, lxs, _t in stats}
        avg_len = sum(max(1, dl) for _s, dl, _l, _t in stats) / len(stats)
        scores: dict[str, float] = {}
        for slug, dl, lxs, tfs in stats:
            norm = BM25_K1 * (1.0 - BM25_B + BM25_B * max(1, dl) / avg_len)
            scores[slug] = sum(
                idf[lx] * tf * (BM25_K1 + 1.0) / (tf + norm)
                for lx, tf in zip(lxs, tfs, strict=True)
            )
        ranked = sorted((s for s in scores if scores[s] > 0), key=lambda s: (-scores[s], s))
        if not ranked:
            return [], {}, {}
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
        chunks = {slug: (heading, text) for slug, heading, text in cur.fetchall()}
    return ranked, chunks, {s: share[s] for s in ranked}


def _vector_rank(
    conn: psycopg.Connection, query_vec: list[float], slugs: list[str]
) -> tuple[list[str], dict[str, tuple[str | None, str]]]:
    """Every candidate's best chunk by exact cosine similarity (no HNSW, so a small set
    on a large graph is never cut by the index's candidate limit)."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT ON (c.note_slug) c.note_slug, c.heading, c.text,
                   1 - (c.embedding <=> %s::vector) AS sim
            FROM chunks c
            WHERE c.note_slug = ANY(%s) AND c.embedding IS NOT NULL
            ORDER BY c.note_slug, sim DESC, c.id
            """,
            (query_vec, slugs),
        )
        rows = [r for r in cur.fetchall() if float(r[3]) >= VECTOR_MIN_SIMILARITY]
    rows.sort(key=lambda r: (-float(r[3]), r[0]))
    return [r[0] for r in rows], {r[0]: (r[1], r[2]) for r in rows}


@dataclass
class _Scored:
    text: dict[str, float]
    via: dict[str, set[str]]
    chunks: dict[str, tuple[str | None, str]]


def _text_scores(
    conn: psycopg.Connection,
    embedder: Embedder,
    query: str,
    candidates: list[str],
    statuses: list[str],
) -> _Scored:
    """`search()`'s scoring (BM25 and vector legs, RRF with each keyword rank scaled by
    the note's share of the query, title boost, recency) over `candidates`, divided by
    the best candidate's score."""
    if not candidates:
        return _Scored({}, {}, {})
    fts_rank, fts_chunks, fts_share = _bm25_rank(conn, query, candidates, statuses)
    [qvec] = embedder.embed([query])
    vec_rank, vec_chunks = _vector_rank(conn, qvec, candidates)
    _order, via_sets, rrf = _rrf(
        [(fts_rank, FTS_WEIGHT, "fts"), (vec_rank, VECTOR_WEIGHT, "vector")],
        scale={"fts": _keyword_scale(fts_share)},
    )
    via: dict[str, set[str]] = dict(via_sets)
    chunks = {**vec_chunks, **fts_chunks}
    if not rrf:
        return _Scored({}, {}, chunks)
    meta = _load_notes_meta(conn, list(rrf))
    terms = _terms(query)
    rrf_max = max(rrf.values())
    scores = {
        s: (v + _title_boost(terms, meta[s][0], rrf_max)) * _note_recency(meta[s])
        for s, v in rrf.items()
        if s in meta
    }
    top = max(scores.values())
    return _Scored({s: v / top for s, v in scores.items()}, dict(via), chunks)


def _window_notes(
    conn: psycopg.Connection, since: date, until: date, candidates: list[str]
) -> set[str]:
    """Candidates holding at least one claim recorded (world clock, the deployment's
    calendar day) in [since, until]."""
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT DISTINCT c.note_slug FROM claims c
            WHERE {clock.world_date_sql("c.recorded")} BETWEEN %s AND %s
              AND c.note_slug = ANY(%s)
            """,
            (since, until, candidates),
        )
        return {r[0] for r in cur.fetchall()}


# ---------------------------------------------------------------------------
# The entry point
# ---------------------------------------------------------------------------


@dataclass
class GraphTrace:
    """What a graph mode did, for the bench and the CLI: seeds, named entities,
    candidate count, and seconds per stage."""

    mode: str
    seeds: list[str] = field(default_factory=list)
    entities: list[EntityMatch] = field(default_factory=list)
    pool_size: int = 0
    seconds: dict[str, float] = field(default_factory=dict)


def graph_search(
    conn: psycopg.Connection,
    embedder: Embedder,
    query: str,
    *,
    mode: GraphMode = "off",
    k: int = 12,
    hops: int = DEFAULT_HOPS,
    window: tuple[date, date] | None = None,
    include_superseded: bool = False,
    ef_search: int | None = None,
    trace: GraphTrace | None = None,
) -> SearchResult:
    """Fused search, then the graph mode's candidates and rescoring (module docstring).
    `mode="off"` with no `window` returns `search()`'s result unchanged."""
    if mode not in GRAPH_MODES:
        raise ValueError(f"Unknown graph mode {mode!r}. Valid modes: {', '.join(GRAPH_MODES)}.")
    trace = trace if trace is not None else GraphTrace(mode=mode)
    trace.mode = mode
    cached = _QueryVectorCache(embedder)
    t0 = time.perf_counter()
    plain = mode == "off" and window is None
    # `search()`'s top k depends on k (its candidate pools scale with it), so the seeds
    # are the very list mode "off" returns; a window needs a deeper list to filter.
    base = search(
        conn,
        cached,
        query,
        k=k if window is None else max(WINDOW_POOL, k),
        include_superseded=include_superseded,
        ef_search=ef_search,
    )
    trace.seconds["search"] = time.perf_counter() - t0
    if plain:
        return base

    statuses = _STATUS_ALL if include_superseded else _STATUS_LIVING_ONLY
    graph = _Graph(conn, statuses)
    top = max((h.score for h in base.hits), default=0.0) or 1.0
    seeds = {h.note_slug: h.score / top for h in base.hits[:SEED_K]}
    trace.seeds = list(seeds)

    t1 = time.perf_counter()
    entities = detect_entities(conn, query) if mode in ("entity", "ppr") else []
    trace.entities = entities
    anchors = {s: 1.0 for e in entities for s in e.anchor_slugs}
    proximity: dict[str, tuple[float, str]] = {}
    if mode == "expand":
        proximity = _proximity(graph, seeds, hops, protected=set())
    elif mode == "entity" and anchors:
        proximity = _proximity(graph, anchors, hops, protected=set(anchors))
    elif mode == "ppr":
        proximity = _ppr_proximity(graph, seeds, anchors, hops)
    trace.seconds["graph"] = time.perf_counter() - t1

    if not proximity and window is None:
        return SearchResult(
            hits=base.hits[:k],
            control_ok=base.control_ok,
            query_terms_unmatched=base.query_terms_unmatched,
        )

    t2 = time.perf_counter()
    base_hits = {h.note_slug: h for h in base.hits}
    candidates = list(dict.fromkeys([*base_hits, *proximity]))
    if window is not None:
        keep = _window_notes(conn, window[0], window[1], candidates)
        candidates = [c for c in candidates if c in keep]
    trace.pool_size = len(candidates)
    # A search hit keeps its own score; a note only the graph found scores below the
    # search's last hit, ordered among its kind by the same relevance, and one sharing
    # nothing with the question still takes a slot the search left empty.
    text = {s: base_hits[s].score / top for s in candidates if s in base_hits}
    floor = min(text.values(), default=1.0)
    extra = [s for s in candidates if s not in base_hits]
    scored = _text_scores(conn, cached, query, extra, statuses)
    text.update({s: floor * max(scored.text.get(s, 0.0), GRAPH_ONLY_FLOOR) for s in extra})
    weight = MODE_WEIGHT.get(mode, 0.0)
    if COMBINE == "multiply":
        final = {s: text[s] * (1.0 + weight * proximity.get(s, (0.0, ""))[0]) for s in candidates}
    else:
        final = {s: text[s] + weight * proximity.get(s, (0.0, ""))[0] for s in candidates}
    ranked = sorted((s for s in final if final[s] > 0), key=lambda s: (-final[s], s))[:k]
    meta = _load_notes_meta(conn, [s for s in ranked if s not in base_hits])
    hits: list[SearchHit] = []
    for slug in ranked:
        score = round(final[slug], 6)
        if slug in base_hits:
            hits.append(base_hits[slug].model_copy(update={"score": score}))
        elif slug in meta:
            via = scored.via.get(slug, set())
            origin = proximity.get(slug, (0.0, ""))[1]
            hits.append(
                _build_hit(
                    conn,
                    slug,
                    score,
                    via,
                    meta,
                    scored.chunks,
                    expanded_from=origin or None,
                )
            )
    trace.seconds["rescore"] = time.perf_counter() - t2
    return SearchResult(
        hits=hits, control_ok=base.control_ok, query_terms_unmatched=base.query_terms_unmatched
    )


def _ppr_proximity(
    graph: _Graph, seeds: dict[str, float], anchors: dict[str, float], hops: int
) -> dict[str, tuple[float, str]]:
    """PPR over the notes within `hops` steps of the restart nodes (the named entities
    at 1.0, the search seeds at their normalised score). A note's proximity is the mass
    it holds beyond its own restart share (what the walk brought it from elsewhere),
    divided by the largest such mass; entity notes scaled by `BRIDGE_WEIGHT`."""
    restart: dict[str, float] = dict(anchors)
    for s, w in seeds.items():
        restart[s] = max(restart.get(s, 0.0), w)
    reach = _proximity(graph, restart, hops, protected=set(anchors))
    mass = _ppr(graph, [*reach, *restart], restart)
    total = sum(restart.values()) or 1.0
    walked = {s: m - PPR_RESTART * restart.get(s, 0.0) / total for s, m in mass.items()}
    peak = max(walked.values(), default=0.0)
    if peak <= 0:
        return {}
    out: dict[str, tuple[float, str]] = {}
    for s, m in walked.items():
        if m <= 0:
            continue
        value = m / peak
        if graph.is_entity(s):
            value *= BRIDGE_WEIGHT
        out[s] = (value, reach.get(s, (0.0, s))[1])
    return out


def changed_since(
    conn: psycopg.Connection,
    embedder: Embedder,
    query: str,
    *,
    since: date,
    until: date | None = None,
    mode: GraphMode = "entity",
    k: int = 12,
    hops: int = DEFAULT_HOPS,
) -> SearchResult:
    """ "What changed for X since `since`": `graph_search` restricted to notes with a
    claim recorded in the window (world clock), `until` defaulting to today."""
    until = until or clock.today()
    return graph_search(conn, embedder, query, mode=mode, k=k, hops=hops, window=(since, until))
