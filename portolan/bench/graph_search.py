"""Graph retrieval bench on Portolan's demo graph: single-hop against multi-hop questions.

The graph is `portolan.bench.demo_search.seed_demo_graph` (the fake extractor over the
synthetic Halvard corpus, `corpus/out`). Single-hop questions are that module's
`DEMO_QUESTIONS` and `PARAPHRASE_QUESTIONS`. `MULTI_HOP_QUESTIONS` were written on
2026-09-23 by reading the demo graph's claims and links (evaluation only, never training
data): each needs two or more notes, reached through the account hierarchy, a shared
account, a competitor or product, a person, or an action.

Gold is a tuple of requirements; a requirement is met when any one of its notes is in the
top k (the same fact can sit on an action note and on its signal note). Scores:
requirement recall at k, answer support (every requirement met), first-gold rank for
MRR, and latency. `window_questions` asks "what changed for account X in the week of
D" from the claims' world clock. Runner: `scripts/bench/graph_retrieval.py`; write-up:
`01-spec/bench-graph-retrieval-2026-09-23.md`.
"""

from __future__ import annotations

import contextlib
import statistics
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import psycopg

from portolan.bench.demo_search import DEMO_QUESTIONS, PARAPHRASE_QUESTIONS
from portolan.bench.erb import TOP_K
from portolan.search.graph import GraphTrace, graph_search
from portolan.search.query import search
from portolan.util import clock

__all__ = [
    "MAX_REQUIREMENTS",
    "MULTI_HOP_QUESTIONS",
    "GraphQuestion",
    "aggregate_rows",
    "entity_note_slugs",
    "evaluate",
    "graph_params",
    "rescoring_overhead",
    "score_hits",
    "single_hop_questions",
    "window_questions",
]


# A question with more gold notes than this cannot be answered from a top 10 that also
# carries the notes linking them.
MAX_REQUIREMENTS = 6
WINDOW_PER_WEEK = 2


@dataclass(frozen=True)
class GraphQuestion:
    qid: str
    question: str
    category: str
    requirements: tuple[tuple[str, ...], ...]
    hop: str = "multi"
    window: tuple[date, date] | None = None


def _one(*slugs: str) -> tuple[tuple[str, ...], ...]:
    """Every slug required."""
    return tuple((s,) for s in slugs)


_A2 = "action-2-sam-okafor-promised-ashgrove-health-a-written-delivery-root-"
_A3 = "action-3-cobalt-ridge-medical-center-agreed-to-pilot-two-tidewater-t3"
_A4 = "action-4-priya-natarajan-promised-wrenfield-health-a-tidewater-link-e"
_A5 = "action-5-dana-whitlock-committed-to-sending-driftwood-community-hospi"

MULTI_HOP_QUESTIONS: tuple[GraphQuestion, ...] = (
    GraphQuestion(
        "mh_01",
        "Which hospitals owned by Pinecrest Health Partners have had late consumable deliveries?",
        "hierarchy",
        (
            ("signal-other-larkspur-regional-health", "larkspur-regional-health"),
            ("larkspur-memorial-hospital",),
            ("larkspur-north-hospital",),
            ("signal-complaint-larkspur-memorial-hospital",),
            ("signal-complaint-larkspur-north-hospital",),
        ),
    ),
    GraphQuestion(
        "mh_02",
        "Have any of Ashgrove Health's facilities complained about late deliveries?",
        "hierarchy",
        _one(
            "ashgrove-medical-center",
            "ashgrove-riverside-hospital",
            "signal-complaint-ashgrove-medical-center",
            "signal-complaint-ashgrove-riverside-hospital",
        ),
    ),
    GraphQuestion(
        "mh_03",
        "How many Tidewater pumps does Wrenfield Health have installed across its facilities?",
        "hierarchy",
        (
            ("signal-other-wrenfield-medical-center", "wrenfield-medical-center"),
            ("signal-usage-wrenfield-medical-center",),
        ),
    ),
    GraphQuestion(
        "mh_04",
        "Which Northlake Health System hospital had late T3 deliveries, and how many?",
        "hierarchy",
        _one("northlake-general-hospital", "signal-complaint-northlake-general-hospital"),
    ),
    GraphQuestion(
        "mh_05",
        "Is there any unanswered customer question at a Wrenfield Health facility?",
        "hierarchy",
        _one("wrenfield-east-hospital", "signal-question-wrenfield-east-hospital"),
    ),
    GraphQuestion(
        "mh_06",
        "What have Wrenfield Health and its hospitals asked or been told about the "
        "Tidewater Link integration timeline?",
        "hierarchy",
        (
            ("signal-request-wrenfield-health",),
            ("signal-commitment-wrenfield-health", _A4),
            ("wrenfield-east-hospital",),
            ("signal-question-wrenfield-east-hospital",),
        ),
    ),
    GraphQuestion(
        "mh_07",
        "The account that moved its HalvardCare renewal to June 2027: what did its CFO "
        "say about late deliveries?",
        "shared_account",
        _one("signal-decision-ashgrove-health", "signal-risk-ashgrove-health"),
    ),
    GraphQuestion(
        "mh_08",
        "Which account with annual recurring revenue above two million dollars has had "
        "delivery complaints at its hospitals, and when does its contract renew?",
        "attribute",
        (
            ("signal-other-ashgrove-health",),
            ("ashgrove-medical-center", "ashgrove-riverside-hospital"),
            (
                "signal-complaint-ashgrove-medical-center",
                "signal-complaint-ashgrove-riverside-hospital",
            ),
            ("signal-decision-ashgrove-health",),
        ),
    ),
    GraphQuestion(
        "mh_09",
        "Which account froze capital purchases, and which of our products does it praise?",
        "shared_account",
        _one("signal-risk-sable-valley-health", "signal-praise-sable-valley-health"),
    ),
    GraphQuestion(
        "mh_10",
        "The hospital that threatened to pilot a second consumables supplier: did it "
        "follow through?",
        "shared_account",
        _one("signal-risk-larkspur-memorial-hospital", "signal-praise-larkspur-memorial-hospital"),
    ),
    GraphQuestion(
        "mh_11",
        "Which accounts has Brightline Medical approached or won, and what pricing "
        "pressure are those accounts putting on us?",
        "competitor",
        _one(
            "signal-competitor_mention-vireo-surgical-group",
            "signal-competitor_mention-anchor-bay-surgical-partners",
            "signal-decision-kestrel-medical-group",
            "signal-pricing-anchor-bay-surgical-partners",
        ),
    ),
    GraphQuestion(
        "mh_12",
        "The competitor that recalled its pump line: which of our accounts have been "
        "evaluating or comparing its pumps?",
        "competitor",
        _one(
            "signal-competitor_mention-corvane-systems",
            "signal-competitor_mention-northlake-health-system",
            "signal-competitor_mention-driftwood-community-hospital",
        ),
    ),
    GraphQuestion(
        "mh_13",
        "Is the Northlake site that is evaluating Corvane pumps also having delivery "
        "problems with us?",
        "competitor",
        _one(
            "signal-competitor_mention-northlake-health-system",
            "signal-complaint-northlake-general-hospital",
        ),
    ),
    GraphQuestion(
        "mh_14",
        "What has Felix Arden told us about deliveries since August?",
        "person",
        _one(
            "signal-complaint-larkspur-memorial-hospital",
            "signal-risk-larkspur-memorial-hospital",
            "signal-praise-larkspur-memorial-hospital",
            "signal-complaint-larkspur-north-hospital",
        ),
    ),
    GraphQuestion(
        "mh_15",
        "What did the materials manager who replaced Tom Ridley say about our delivery "
        "performance in September?",
        "person",
        _one("tom-ridley", "signal-praise-cobalt-ridge-medical-center"),
    ),
    GraphQuestion(
        "mh_16",
        "Who took over from Nina Vale, and what had her practice been offered by a competitor?",
        "person",
        _one("nina-vale", "signal-competitor_mention-vireo-surgical-group"),
    ),
    GraphQuestion(
        "mh_17",
        "What did Derek Hollis's organization ask us about Cerner, and was it answered?",
        "person",
        _one("derek-hollis", "signal-question-northlake-health-system"),
    ),
    GraphQuestion(
        "mh_18",
        "What does the account Maria Chen works for pay us each year, and when is its "
        "contract up for renewal?",
        "person",
        _one("maria-chen", "signal-other-ashgrove-health", "signal-decision-ashgrove-health"),
    ),
    GraphQuestion(
        "mh_19",
        "Sam Okafor promised one account a written root-cause summary. What else is at "
        "stake with that account's renewal?",
        "action",
        (
            (_A2, "signal-commitment-ashgrove-health"),
            ("signal-risk-ashgrove-health",),
            ("signal-decision-ashgrove-health",),
        ),
    ),
    GraphQuestion(
        "mh_20",
        "Dana Whitlock owes one hospital a Tidewater Link project timeline. What has "
        "that hospital asked us for?",
        "action",
        (
            (_A5, "signal-commitment-driftwood-community-hospital"),
            ("signal-request-driftwood-community-hospital",),
        ),
    ),
    GraphQuestion(
        "mh_21",
        "Which account agreed to pilot T3 units, and what problems had it been having "
        "with T2 deliveries before that?",
        "action",
        (
            (_A3, "signal-commitment-cobalt-ridge-medical-center"),
            ("signal-complaint-cobalt-ridge-medical-center",),
        ),
    ),
    GraphQuestion(
        "mh_22",
        "Which hospital lent consumables to another account during the August delivery "
        "delays, and has that lending hospital itself had late deliveries?",
        "cross_account",
        _one(
            "signal-complaint-cobalt-ridge-medical-center",
            "signal-complaint-larkspur-memorial-hospital",
        ),
    ),
    GraphQuestion(
        "mh_23",
        "Which accounts run Tidewater T3 pumps today, and has any of them asked for a T3 "
        "feature change?",
        "product",
        _one(
            "signal-usage-ashgrove-medical-center",
            "signal-usage-larkspur-memorial-hospital",
            "signal-request-ashgrove-medical-center",
        ),
    ),
    GraphQuestion(
        "mh_24",
        "Of the accounts that asked for Tidewater Link to connect to Epic, which have we "
        "made a written commitment to?",
        "product",
        (
            ("signal-request-driftwood-community-hospital",),
            ("signal-request-larkspur-regional-health",),
            ("signal-request-st-aldric-health",),
            ("signal-request-wrenfield-health",),
            ("signal-commitment-driftwood-community-hospital", _A5),
            ("signal-commitment-wrenfield-health", _A4),
        ),
    ),
    GraphQuestion(
        "mh_25",
        "What did Wrenfield Health decide about Epic integration before it reversed "
        "course, and what did we promise after the reversal?",
        "shared_account",
        (
            ("signal-request-wrenfield-health",),
            ("signal-commitment-wrenfield-health", _A4),
        ),
    ),
    GraphQuestion(
        "mh_26",
        "What has Larkspur Regional Health's supply-chain VP been pressing for, and did "
        "the system's new CFO start before or after that?",
        "shared_account",
        _one(
            "signal-pricing-larkspur-regional-health",
            "signal-contact_change-larkspur-regional-health",
        ),
    ),
)


def single_hop_questions() -> tuple[GraphQuestion, ...]:
    """`DEMO_QUESTIONS` and `PARAPHRASE_QUESTIONS`, every expected note required."""
    out = []
    for category, questions in (("demo", DEMO_QUESTIONS), ("paraphrase", PARAPHRASE_QUESTIONS)):
        for q in questions:
            out.append(GraphQuestion(q.qid, q.question, category, _one(*q.expected), "single"))
    return tuple(out)


def _monday(d: date) -> date:
    return d - timedelta(days=d.weekday())


def window_questions(
    conn: psycopg.Connection, *, since: date = date(2026, 7, 1), limit: int = 12
) -> tuple[GraphQuestion, ...]:
    """ "What changed for <account> in the week of <Monday>?" for (account, week) pairs
    with at least one claim recorded that week (world clock) that names the account
    (`[[slug]]` in the claim text, as `portolan.report.diff.quiet_accounts` reads it),
    newest weeks first, `WINDOW_PER_WEEK` a week. CRM rows are left out: an export
    dates every field to the export day. Gold: every note holding such a claim, at most
    `MAX_REQUIREMENTS` of them."""
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT a.slug, a.title, {clock.world_date_sql("c.recorded")} AS d, c.note_slug
            FROM claims c
            JOIN sources s ON s.slug = c.source_slug AND s.source_type <> 'crm'
            JOIN notes a ON a.type = 'account'
                AND c.claim_text LIKE '%%[[' || a.slug || ']]%%'
            WHERE {clock.world_date_sql("c.recorded")} >= %s
            """,
            (since,),
        )
        rows = cur.fetchall()
    weeks: dict[tuple[str, str, date], set[str]] = {}
    for slug, title, d, note in rows:
        weeks.setdefault((slug, title, _monday(d)), set()).add(note)
    ordered = sorted(
        (key for key in weeks if len(weeks[key]) <= MAX_REQUIREMENTS),
        key=lambda key: (-key[2].toordinal(), key[0]),
    )
    per_week: dict[date, int] = {}
    out: list[GraphQuestion] = []
    for slug, title, monday in ordered:
        if len(out) >= limit or per_week.get(monday, 0) >= WINDOW_PER_WEEK:
            continue
        per_week[monday] = per_week.get(monday, 0) + 1
        out.append(
            GraphQuestion(
                f"tw_{len(out) + 1:02d}",
                f"What changed for {title} in the week of {monday.isoformat()}?",
                "window",
                _one(*sorted(weeks[(slug, title, monday)])),
                "window",
                (monday, monday + timedelta(days=6)),
            )
        )
    return tuple(out)


@contextlib.contextmanager
def graph_params(**overrides: Any) -> Iterator[None]:
    """`portolan.search.graph` module constants set for the block, then restored (the
    bench's way to compare settings, as `demo_search.fusion_weights` does)."""
    from portolan.search import graph as g

    saved = {name: getattr(g, name) for name in overrides}
    for name, value in overrides.items():
        setattr(g, name, value)
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(g, name, value)


def score_hits(
    retrieved: Sequence[str],
    q: GraphQuestion,
    k: int = TOP_K,
    *,
    entity_slugs: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Requirement recall and support at `k`. A requirement whose notes are all entity
    notes (`entity_slugs`: an account page that only says which system a hospital
    belongs to) is a bridge; `answer_*` repeats both numbers without the bridges."""
    top = list(dict.fromkeys(retrieved))[:k]
    met = [any(s in top for s in req) for req in q.requirements]
    answer = [
        ok
        for req, ok in zip(q.requirements, met, strict=True)
        if not all(s in entity_slugs for s in req)
    ] or met
    gold = {s for req in q.requirements for s in req}
    first = next((i for i, s in enumerate(top, 1) if s in gold), None)
    return {
        "requirement_recall_pct": 100.0 * sum(met) / len(met),
        "supported": all(met),
        "answer_recall_pct": 100.0 * sum(answer) / len(answer),
        "answer_supported": all(answer),
        "first_gold_rank": first,
        "unmet": [list(req) for req, ok in zip(q.requirements, met, strict=True) if not ok],
    }


def entity_note_slugs(conn: psycopg.Connection) -> frozenset[str]:
    """Entity notes with no claim of their own: pages that only link."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT n.slug FROM notes n
            WHERE n.type IN ('account', 'contact', 'product', 'competitor')
              AND NOT EXISTS (SELECT 1 FROM claims c WHERE c.note_slug = n.slug)
            """
        )
        return frozenset(r[0] for r in cur.fetchall())


def evaluate(
    conn: psycopg.Connection,
    embedder: Any,
    questions: Sequence[GraphQuestion],
    *,
    mode: str,
    hops: int = 1,
    use_window: bool = False,
    use_since: bool = False,
    k: int = TOP_K,
) -> dict[str, Any]:
    """Every question through `graph_search` (`mode`, `hops`; the question's window when
    `use_window`), or through `search(since=<window start>)` when `use_since` (the
    existing filter, on `notes.updated`): per-question rows and the aggregate."""
    rows = []
    entity_slugs = entity_note_slugs(conn)
    for q in questions:
        trace = GraphTrace(mode=mode)
        start = time.perf_counter()
        if use_since and q.window:
            result = search(conn, embedder, q.question, k=k, since=q.window[0])
        else:
            result = graph_search(
                conn,
                embedder,
                q.question,
                mode=mode,  # type: ignore[arg-type]
                k=k,
                hops=hops,
                window=q.window if use_window else None,
                trace=trace,
            )
        seconds = time.perf_counter() - start
        retrieved = [h.note_slug for h in result.hits]
        rows.append(
            {
                "question_id": q.qid,
                "category": q.category,
                "hop": q.hop,
                "retrieved": retrieved,
                **score_hits(retrieved, q, k, entity_slugs=entity_slugs),
                "seconds": round(seconds, 4),
                "pool_size": trace.pool_size,
                "entities": [e.name for e in trace.entities],
                "window_precision_pct": _window_precision(conn, retrieved[:k], q.window)
                if q.window
                else None,
            }
        )
    return {"rows": rows, "overall": aggregate_rows(rows)}


def rescoring_overhead(
    conn: psycopg.Connection,
    embedder: Any,
    questions: Sequence[str],
    *,
    pool_sizes: Sequence[int] = (50, 200),
    seed: str = "graph-bench-overhead",
) -> dict[str, Any]:
    """What a graph mode adds to a search on a large graph, whatever its links: one
    neighbour load for ten seeds, and scoring `pool_size` notes the graph reached (a
    fixed sample of living notes stands in for them). Seconds per question, p50/p95."""
    from portolan.search.graph import _Graph, _text_scores

    with conn.cursor() as cur:
        cur.execute(
            "SELECT slug FROM notes WHERE status = 'living' ORDER BY md5(slug || %s)", (seed,)
        )
        slugs = [r[0] for r in cur.fetchall()]
    out: dict[str, Any] = {}
    for size in pool_sizes:
        load, score = [], []
        for i, question in enumerate(questions):
            pool = slugs[i * size : (i + 1) * size]
            graph = _Graph(conn, ["living"])
            t0 = time.perf_counter()
            graph.load(pool[:SEED_LOAD])
            t1 = time.perf_counter()
            _text_scores(conn, embedder, question, pool, ["living"])
            t2 = time.perf_counter()
            load.append(t1 - t0)
            score.append(t2 - t1)
        out[str(size)] = {
            "questions": len(questions),
            "neighbour_load_p50_ms": round(1000 * statistics.median(load), 1),
            "rescore_p50_ms": round(1000 * statistics.median(score), 1),
            "rescore_p95_ms": round(1000 * sorted(score)[int(0.95 * (len(score) - 1))], 1),
        }
    return out


SEED_LOAD = 10


def _window_precision(
    conn: psycopg.Connection, retrieved: list[str], window: tuple[date, date]
) -> float | None:
    """Share of the returned notes holding a claim recorded inside the window."""
    if not retrieved:
        return None
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT count(DISTINCT c.note_slug) FROM claims c
            WHERE c.note_slug = ANY(%s)
              AND {clock.world_date_sql("c.recorded")} BETWEEN %s AND %s
            """,
            (retrieved, window[0], window[1]),
        )
        (n,) = cur.fetchone()
    return 100.0 * n / len(retrieved)


def aggregate_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"count": 0}
    ranks = [r["first_gold_rank"] for r in rows]
    secs = sorted(r["seconds"] for r in rows)
    precision = [
        r["window_precision_pct"] for r in rows if r.get("window_precision_pct") is not None
    ]
    return {
        "count": len(rows),
        "requirement_recall_pct": round(
            sum(r["requirement_recall_pct"] for r in rows) / len(rows), 2
        ),
        "supported_pct": round(100.0 * sum(1 for r in rows if r["supported"]) / len(rows), 2),
        "answer_recall_pct": round(sum(r["answer_recall_pct"] for r in rows) / len(rows), 2),
        "answer_supported_pct": round(
            100.0 * sum(1 for r in rows if r["answer_supported"]) / len(rows), 2
        ),
        "hit_at_1_pct": round(100.0 * sum(1 for x in ranks if x == 1) / len(rows), 2),
        "mrr_at_10": round(sum(1.0 / x for x in ranks if x) / len(rows), 4),
        "p50_ms": round(1000 * statistics.median(secs), 1),
        "p95_ms": round(1000 * secs[min(len(secs) - 1, int(0.95 * len(secs)))], 1),
        "mean_pool": round(sum(r["pool_size"] for r in rows) / len(rows), 1),
        "window_precision_pct": round(sum(precision) / len(precision), 2) if precision else None,
    }
