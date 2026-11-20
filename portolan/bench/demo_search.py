"""Search check on Portolan's own demo graph, so a change tuned on the ERB
exam cannot quietly hurt the graph a customer is shown.

The graph is the one the web tests and `tests/verify_seed.py` build: one
`pipeline.run` with the fake extractor over the synthetic Halvard corpus
(`corpus/out`), `seed_demo_graph` below. The 20 questions are the kind a
rep types into chat, written on 2026-09-22 by reading that graph's claims
(not ERB, not copied claim text), each with the note(s) that hold the
answer. `PARAPHRASE_QUESTIONS` asks about the same graph in other words.
Scored with ERB's own metric definitions (`portolan.bench.erb`): document
recall at 10, hit@k, MRR@10.

Run: `uv run python scripts/bench/demo_search.py` (see its docstring).
Regression floor: `tests/test_demo_search.py`.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psycopg

from portolan.bench.erb import TOP_K, aggregate, document_recall_pct, first_gold_rank

__all__ = [
    "DEMO_QUESTIONS",
    "HELDOUT_QUESTIONS",
    "PARAPHRASE_QUESTIONS",
    "DemoQuestion",
    "evaluate_demo_search",
    "fusion_weights",
    "lexeme_overlap",
    "seed_demo_graph",
]

CORPUS_OUT = Path(__file__).resolve().parents[2] / "corpus" / "out"


@dataclass(frozen=True)
class DemoQuestion:
    qid: str
    question: str
    expected: tuple[str, ...]


DEMO_QUESTIONS: tuple[DemoQuestion, ...] = (
    DemoQuestion(
        "demo_01",
        "Which account got a quote from Brightline that was cheaper than ours?",
        ("signal-competitor_mention-vireo-surgical-group",),
    ),
    DemoQuestion(
        "demo_02",
        "Who is the new CFO at Larkspur?",
        ("signal-contact_change-larkspur-regional-health",),
    ),
    DemoQuestion(
        "demo_03",
        "When does Ashgrove's service contract come up for renewal?",
        ("signal-decision-ashgrove-health",),
    ),
    DemoQuestion(
        "demo_04",
        "What discount is Larkspur Regional pushing for on consumables?",
        ("signal-pricing-larkspur-regional-health",),
    ),
    DemoQuestion(
        "demo_05",
        "Which competitor had to recall one of its pumps?",
        ("signal-competitor_mention-corvane-systems",),
    ),
    DemoQuestion(
        "demo_06",
        "Is Northlake looking at other pump vendors?",
        ("signal-competitor_mention-northlake-health-system",),
    ),
    DemoQuestion(
        "demo_07",
        "Which customer stopped buying new capital equipment this year?",
        ("signal-risk-sable-valley-health",),
    ),
    DemoQuestion(
        "demo_08",
        "What did St. Aldric dispute on its invoice?",
        ("signal-pricing-st-aldric-health",),
    ),
    DemoQuestion(
        "demo_09",
        "Which accounts want Tidewater Link hooked up to Epic?",
        (
            "signal-request-driftwood-community-hospital",
            "signal-request-larkspur-regional-health",
            "signal-request-st-aldric-health",
            "signal-request-wrenfield-health",
        ),
    ),
    DemoQuestion(
        "demo_10",
        "How many T3 pumps are running in Ashgrove Medical Center's ICUs?",
        ("signal-usage-ashgrove-medical-center",),
    ),
    DemoQuestion(
        "demo_11",
        "Which customer is happy with the Tidewater Link interface since go-live?",
        ("signal-praise-sable-valley-health",),
    ),
    DemoQuestion(
        "demo_12",
        "Why did Kestrel Medical Group leave us?",
        ("signal-decision-kestrel-medical-group",),
    ),
    DemoQuestion(
        "demo_13",
        "Does Cobalt Ridge expect money back on its bill because deliveries were late?",
        ("signal-pricing-cobalt-ridge-medical-center",),
    ),
    DemoQuestion(
        "demo_14",
        "Which hospital is thinking about a second consumables supplier?",
        ("signal-risk-larkspur-memorial-hospital",),
    ),
    DemoQuestion(
        "demo_15",
        "What did we promise Ashgrove about the delivery problems?",
        (
            "signal-commitment-ashgrove-health",
            "action-2-sam-okafor-promised-ashgrove-health-a-written-delivery-root-",
        ),
    ),
    DemoQuestion(
        "demo_16",
        "Were there false occlusion alarms on T2 pumps anywhere?",
        ("signal-complaint-larkspur-north-hospital",),
    ),
    DemoQuestion(
        "demo_17",
        "How much recurring revenue do we get from Wrenfield Health?",
        ("signal-other-wrenfield-health",),
    ),
    DemoQuestion(
        "demo_18",
        "Who owns Larkspur Regional Health?",
        ("signal-other-larkspur-regional-health",),
    ),
    DemoQuestion(
        "demo_19",
        "Which ICU wanted its pumps to be quieter at night?",
        ("signal-request-ashgrove-medical-center",),
    ),
    DemoQuestion(
        "demo_20",
        "Did anyone answer Northlake's question about Cerner?",
        ("signal-question-northlake-health-system",),
    ),
)


# ERB's questions share words with their documents; these 15 ask about the
# same graph in other words, written 2026-09-23 from its claims so that few
# of a question's content words appear in the note that answers it (measured
# by `lexeme_overlap`; `tests/test_demo_search.py` holds the mean down). They
# are half of the rule that set search()'s fusion weights (bench write-up,
# "Fusion weight").
PARAPHRASE_QUESTIONS: tuple[DemoQuestion, ...] = (
    DemoQuestion(
        "para_01",
        "Which rival manufacturer pulled a product off the market?",
        ("signal-competitor_mention-corvane-systems",),
    ),
    DemoQuestion(
        "para_02",
        "Which client brought in someone new to run its finances?",
        ("signal-contact_change-larkspur-regional-health",),
    ),
    DemoQuestion(
        "para_03",
        "Who put a hold on buying big-ticket hardware until next year?",
        ("signal-risk-sable-valley-health",),
    ),
    DemoQuestion(
        "para_04",
        "Which site says the interface has worked perfectly from the day it was switched on?",
        ("signal-praise-sable-valley-health",),
    ),
    DemoQuestion(
        "para_05",
        "Which former client defected to a rival supplier?",
        ("signal-decision-kestrel-medical-group",),
    ),
    DemoQuestion(
        "para_06",
        "Who pushed back on a bill because one charge appeared twice?",
        ("signal-pricing-st-aldric-health",),
    ),
    DemoQuestion(
        "para_07",
        "Which building's query about whether the rollout schedule covered it went unanswered?",
        ("signal-question-wrenfield-east-hospital",),
    ),
    DemoQuestion(
        "para_08",
        "Where are 18 infusion devices deployed over critical care and surgical wards?",
        ("signal-usage-wrenfield-medical-center",),
    ),
    DemoQuestion(
        "para_09",
        "Which customer wants money knocked off its next statement to compensate for tardy "
        "shipments?",
        ("signal-pricing-cobalt-ridge-medical-center",),
    ),
    DemoQuestion(
        "para_10",
        "Which intensive care team wanted the beeping turned down during overnight shifts?",
        ("signal-request-ashgrove-medical-center",),
    ),
    DemoQuestion(
        "para_11",
        "Which accounts belong to a parent organisation?",
        ("signal-other-larkspur-regional-health", "signal-other-wrenfield-medical-center"),
    ),
    DemoQuestion(
        "para_12",
        "Who complimented how precisely the device delivers medication?",
        ("signal-praise-driftwood-community-hospital",),
    ),
    DemoQuestion(
        "para_13",
        "Which system plans to move all its facilities onto one pump model?",
        ("signal-decision-larkspur-regional-health",),
    ),
    DemoQuestion(
        "para_14",
        "Where is an engineer threatening to try another vendor for disposables?",
        ("signal-risk-larkspur-memorial-hospital",),
    ),
    DemoQuestion(
        "para_15",
        "Which site will trial a couple of the newer devices in its ER this autumn?",
        (
            "signal-commitment-cobalt-ridge-medical-center",
            "action-3-cobalt-ridge-medical-center-agreed-to-pilot-two-tidewater-t3",
        ),
    ),
)


# Held out. Written 2026-11-20 from the demo graph's claims, before any of
# that day's search changes were measured, and never used to choose a
# parameter: `DEMO_QUESTIONS` and `PARAPHRASE_QUESTIONS` set the fusion
# weights, so a change tuned on them is checked here. Sixteen ask in other
# words, eight name the account or person. A change that only helps the two
# tuning sets and not this one moved the goalposts.
HELDOUT_QUESTIONS: tuple[DemoQuestion, ...] = (
    DemoQuestion(
        "held_01",
        "Who took over as practice administrator at Vireo after the previous one departed?",
        ("nina-vale",),
    ),
    DemoQuestion(
        "held_02",
        "Which customer was approached by a competitor's salesperson but declined to meet?",
        ("signal-competitor_mention-anchor-bay-surgical-partners",),
    ),
    DemoQuestion(
        "held_03",
        "Has any hospital had to borrow supplies from a neighbouring site?",
        ("signal-complaint-cobalt-ridge-medical-center",),
    ),
    DemoQuestion(
        "held_04",
        "What staffing change at the distribution depot was made to fix shipping?",
        ("signal-decision-ashgrove-medical-center",),
    ),
    DemoQuestion(
        "held_05",
        "Who asked for instructions translated into another language?",
        ("signal-request-st-aldric-health",),
    ),
    DemoQuestion(
        "held_06",
        "Which account wants its automatic restocking level looked at?",
        ("signal-request-driftwood-community-hospital",),
    ),
    DemoQuestion(
        "held_07",
        "How much does Ashgrove Health pay us per year?",
        ("signal-other-ashgrove-health",),
    ),
    DemoQuestion(
        "held_08",
        "Who was promoted to run facilities at Northlake?",
        ("derek-hollis",),
    ),
    DemoQuestion(
        "held_09",
        "Which surgical practice has no pumps in its recovery area?",
        ("signal-usage-vireo-surgical-group",),
    ),
    DemoQuestion(
        "held_10",
        "Did the threat to bring in another supplier at Larkspur Memorial go away?",
        ("signal-praise-larkspur-memorial-hospital",),
    ),
    DemoQuestion(
        "held_11",
        "Where did a new tower prompt a look at a rival's pumps?",
        ("signal-competitor_mention-northlake-health-system",),
    ),
    DemoQuestion(
        "held_12",
        "Who praised how quickly our technician reacted to an alarm fault?",
        ("signal-praise-larkspur-memorial-hospital",),
    ),
    DemoQuestion(
        "held_13",
        "What regular meeting did we agree to hold with Northlake?",
        (
            "signal-commitment-northlake-health-system",
            "action-1-dana-whitlock-committed-to-a-quarterly-business-review-with-",
        ),
    ),
    DemoQuestion(
        "held_14",
        "Which small practice asked for cheaper supplies because money is tight?",
        ("signal-pricing-anchor-bay-surgical-partners",),
    ),
    DemoQuestion(
        "held_15",
        "Will the finance chief at Ashgrove bring up shipping problems when the contract is "
        "renegotiated?",
        ("signal-risk-ashgrove-health",),
    ),
    DemoQuestion(
        "held_16",
        "Who changed jobs from purchasing to clinical operations?",
        ("maria-chen",),
    ),
    DemoQuestion(
        "held_17",
        "How many pumps does Larkspur Memorial have?",
        ("signal-usage-larkspur-memorial-hospital",),
    ),
    DemoQuestion(
        "held_18",
        "Which hospital got three overdue shipments of the newer pump in under a month?",
        ("signal-complaint-northlake-general-hospital",),
    ),
    DemoQuestion(
        "held_19",
        "Who replaced Tom Ridley?",
        ("tom-ridley",),
    ),
    DemoQuestion(
        "held_20",
        "What test plan did we owe Wrenfield?",
        (
            "signal-commitment-wrenfield-health",
            "action-4-priya-natarajan-promised-wrenfield-health-a-tidewater-link-e",
        ),
    ),
    DemoQuestion(
        "held_21",
        "Which pharmacy lead hinted they were shopping around for alternatives?",
        ("signal-competitor_mention-meadowbrook-community-hospital",),
    ),
    DemoQuestion(
        "held_22",
        "Which customer changed its mind and now wants the records-system hookup started "
        "right away?",
        ("signal-request-wrenfield-health",),
    ),
    DemoQuestion(
        "held_23",
        "Was a joint call promised to go over why deliveries slipped?",
        ("signal-commitment-larkspur-memorial-hospital",),
    ),
    DemoQuestion(
        "held_24",
        "Which community hospital once compared a competitor's devices but dropped the idea?",
        ("signal-competitor_mention-driftwood-community-hospital",),
    ),
)


def lexeme_overlap(conn: psycopg.Connection, question: DemoQuestion) -> float:
    """Share of the question's Postgres `english` lexemes found in any of its
    gold notes' `notes.search` (title, description and body): what the
    keyword leg can match on."""
    from portolan.search.query import QUERY_TSVECTOR_SQL

    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT count(*) FILTER (WHERE EXISTS (
                       SELECT 1 FROM notes n, unnest(n.search) u
                       WHERE n.slug = ANY(%s) AND u.lexeme = q.lexeme)),
                   count(*)
            FROM unnest({QUERY_TSVECTOR_SQL}) AS q
            """,
            (list(question.expected), question.question),
        )
        found, total = cur.fetchone()
    return found / total if total else 0.0


@contextlib.contextmanager
def fusion_weights(fts: float, vector: float) -> Iterator[None]:
    """`search()`'s RRF weights set to (fts, vector) for the block, then
    restored: the bench's way to compare weights through `search()` itself
    (recency and the title boost included), without a public parameter."""
    from portolan.search import query as q

    saved = (q.FTS_WEIGHT, q.VECTOR_WEIGHT)
    q.FTS_WEIGHT, q.VECTOR_WEIGHT = fts, vector
    try:
        yield
    finally:
        q.FTS_WEIGHT, q.VECTOR_WEIGHT = saved


def seed_demo_graph(conn: psycopg.Connection, store_root: Path, embedder: Any) -> None:
    """The demo graph on an empty, migrated database: `pipeline.run` with
    the fake extractor over `corpus/out`, exactly as `tests/test_web.py`'s
    `demo_conn` seeds it, indexed with `embedder`."""
    from portolan.config import Settings
    from portolan.connectors.file_connector import FileConnector
    from portolan.fake_extractor import FakePlacements, build_fake_llm_client
    from portolan.pipeline import run as pipeline_run
    from portolan.registry import Registry
    from portolan.store import LocalStore

    settings = Settings(
        database_url="postgresql://unused/portolan_test_demo_search",
        tenant="test",
        store_root=store_root,
        llm_provider="fake",
        embed_provider="fake",
    )
    pipeline_run(
        conn,
        connector=FileConnector(CORPUS_OUT),
        llm=build_fake_llm_client(FakePlacements(CORPUS_OUT), tenant=settings.tenant),
        embedder=embedder,
        registry=Registry.load(conn),
        store=LocalStore(store_root),
        settings=settings,
    )


def evaluate_demo_search(
    conn: psycopg.Connection,
    embedder: Any,
    *,
    search_fn: Callable[..., Any] | None = None,
    questions: Sequence[DemoQuestion] = DEMO_QUESTIONS,
) -> dict[str, Any]:
    """Every question through `search_fn` (default `portolan.search.search`,
    k=10, no expand): per-question rows and the aggregate."""
    if search_fn is None:
        from portolan.search.query import search as search_fn

    rows = []
    for q in questions:
        hits = [h.note_slug for h in search_fn(conn, embedder, q.question, k=TOP_K).hits]
        rows.append(
            {
                "question_id": q.qid,
                "expected": list(q.expected),
                "retrieved": hits,
                "document_recall_pct": document_recall_pct(hits, q.expected),
                "first_gold_rank": first_gold_rank(hits, q.expected),
                "invalid_extra_docs": None,
            }
        )
    return {"rows": rows, "overall": aggregate(rows)}
