"""Track C eval: does Jev (TypeSafe AI) make Portolan chat faster/cheaper,
and can it rank what goes into the weekly report?

Three tests, one script, re-runnable in stages (each stage caches its raw
Jev/OpenRouter responses under var/jev-eval/cache/ so a re-run costs
nothing unless the cache is cleared):

  1. chat relevance gate  (portolan/web/answer.py, portolan/search/query.py)
  2. report ranking       (portolan/report/weekly.py: outliers, standing questions)
  3. numbers guard        (portolan/report/numbers.py)

Usage (from the worktree root, with .env sourced and DATABASE_URL pointed
at the demo db, e.g. postgresql://localhost/portolan_jev):

    uv run python3 scripts/jev_eval/chat_and_reports.py stage1
    uv run python3 scripts/jev_eval/chat_and_reports.py stage2 --week 2026-W36
    uv run python3 scripts/jev_eval/chat_and_reports.py stage3
    uv run python3 scripts/jev_eval/chat_and_reports.py report   # prints the
        consolidated numbers used in 01-spec/jev-eval-2026-09-16/C-chat-and-reports.md

Every number printed is computed here, not estimated by hand -- per the
task's "numbers rule": code computes, models cite.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import statistics
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import httpx
import psycopg

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = REPO_ROOT / "var" / "jev-eval" / "cache"
RESULTS_DIR = REPO_ROOT / "var" / "jev-eval"
CORPUS_OUT = REPO_ROOT / "corpus" / "out"

JEV_URL = "https://api.typesafe.ai/v1/systemone"
JEV_INPUT_PRICE_PER_M = 0.042  # $/1M input tokens; output tokens free (brief sec 4)

random.seed(20260916)

# ---------------------------------------------------------------------
# Jev client -- one thin wrapper, with a content-addressed disk cache so
# a re-run of this script (debugging, adding a metric) never re-spends
# money on a request already answered.
# ---------------------------------------------------------------------


def _cache_key(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:24]


@dataclass
class JevCallLog:
    calls: list[dict] = field(default_factory=list)

    def record(
        self, *, tag: str, cached: bool, latency_s: float, input_tokens: int, output_tokens: int
    ):
        self.calls.append(
            {
                "tag": tag,
                "cached": cached,
                "latency_s": latency_s,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }
        )

    def cost(self) -> float:
        return sum(c["input_tokens"] for c in self.calls) * JEV_INPUT_PRICE_PER_M / 1_000_000

    def latencies(self, *, tag: str, live_only: bool = True) -> list[float]:
        return [
            c["latency_s"]
            for c in self.calls
            if c["tag"] == tag and ((not live_only) or (not c["cached"]))
        ]


JEV_LOG = JevCallLog()


def call_jev(
    state: str,
    questions: dict[str, dict],
    *,
    api_key: str,
    tag: str = "untagged",
    cache_dir: Path = CACHE_DIR,
) -> dict:
    """POST one batched request to Jev. Cached on (state, questions) so a
    re-run of this script is free. Records latency/token usage on
    JEV_LOG for every call, cached or not -- cached calls are excluded
    from the latency stats (`latencies(live_only=True)`, the default) but
    always counted in cost (a cached call cost real money the first time
    it ran, and the report's cost totals must reflect that spend once,
    not silently drop it because a later run of this script hit cache).
    """
    payload = {"state": state, "model": "jev-latest", "questions": questions}
    key = _cache_key(payload)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{key}.json"

    if cache_file.exists():
        cached = json.loads(cache_file.read_text())
        usage = cached.get("usage", {})
        JEV_LOG.record(
            tag=tag,
            cached=True,
            latency_s=cached.get("_latency_s", 0.0),
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
        )
        return cached

    t0 = time.monotonic()
    resp = httpx.post(
        JEV_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json=payload,
        timeout=60,
    )
    latency = time.monotonic() - t0
    resp.raise_for_status()
    data = resp.json()
    data["_latency_s"] = latency
    usage = data.get("usage", {})
    JEV_LOG.record(
        tag=tag,
        cached=False,
        latency_s=latency,
        input_tokens=usage.get("input_tokens", 0),
        output_tokens=usage.get("output_tokens", 0),
    )
    cache_file.write_text(json.dumps(data, indent=2))
    return data


# ---------------------------------------------------------------------
# DB / LLM / embedder plumbing (mirrors portolan.web.demo / portolan.report.weekly's
# own CLI wiring, so this script exercises the real retrieval and answer path).
# ---------------------------------------------------------------------


def get_conn() -> psycopg.Connection:
    return psycopg.connect(os.environ["DATABASE_URL"])


def get_embedder():
    from portolan.search.embed import FastEmbedEmbedder

    return FastEmbedEmbedder()


def get_openrouter_llm(sink=None):
    from portolan.llm.openrouter import OpenRouterClient
    from portolan.training.spend import SpendMeter, default_spend_path

    return OpenRouterClient(
        spend_key="jev_eval",
        spend_meter=SpendMeter(default_spend_path(REPO_ROOT / "var")),
        api_key=os.environ["OPENROUTER_API_KEY"],
        default_model="amazon/nova-lite-v1",
        tenant=os.environ.get("PORTOLAN_TENANT", "jev"),
        sink=sink,
    )


CHAT_MODEL = (
    "amazon/nova-lite-v1"  # matches COMPARE_MODEL/EXTRACTION_MODEL used to build portolan_jev
)


# ---------------------------------------------------------------------
# Stage 1: chat relevance gate
# ---------------------------------------------------------------------

# The 3 dynamic example questions from chat.html (top 3 most-recently-updated
# accounts in portolan_jev, per portolan.web.conversations.top_account_names):
#   "What has Northlake Health System said recently?"
#   "What has Meadowbrook Community Hospital said recently?"
#   "What has Driftwood Community Hospital said recently?"
# Fetched live in build_question_set() below rather than hardcoded, so a
# rebuild of portolan_jev with different data still produces valid examples.

WRITTEN_QUESTIONS: list[tuple[str, str]] = [
    # (question, expected intent label -- must match one of INTENT_OPTIONS)
    ("Is Larkspur Regional Health at risk of not renewing?", "account status"),
    ("What have we heard about St. Aldric Health's renewal plans?", "account status"),
    ("What delivery problems has Cobalt Ridge Medical Center reported?", "account status"),
    ("Which accounts have complained about late Tidewater T2 deliveries?", "theme or trend"),
    ("Has Ashgrove Medical Center said anything about delivery timelines?", "account status"),
    ("What pricing pushback have we gotten from Anchor Bay Surgical Partners?", "pricing"),
    ("Has St. Aldric Health disputed any HalvardCare invoices?", "pricing"),
    ("What discount requests has Larkspur Regional Health made on Tidewater T2?", "pricing"),
    ("Is Vireo Surgical Group evaluating Brightline Medical?", "competitor"),
    ("What has Anchor Bay Surgical Partners said about Corvane Systems?", "competitor"),
    ("Which accounts are considering switching to a competitor?", "competitor"),
    (
        "What is the status of Wrenfield Health's Tidewater Link integration with Epic?",
        "account status",
    ),
    ("Has Driftwood Community Hospital asked about Tidewater Link integration?", "account status"),
    ("What integration questions has Wrenfield East Hospital raised?", "account status"),
    ("Who is the new CFO at Larkspur Regional Health?", "contact or people"),
    ("Has there been a contact change at Ashgrove Health?", "contact or people"),
    ("Who are the key contacts at Cobalt Ridge Medical Center?", "contact or people"),
    (
        "What has Meadowbrook Community Hospital said about Tidewater pump reliability?",
        "account status",
    ),
    ("Is Northlake Health System satisfied with Tidewater T2 performance?", "account status"),
    ("Summarize this week's biggest themes across all accounts.", "report or summary"),
]

UNANSWERABLE_QUESTIONS: list[str] = [
    "What has Meridian Health Alliance said about our product?",
    "Is Summit Care Network renewing their contract?",
    "What does Fairview Medical Group think about our pricing?",
    "Has any account mentioned switching to a company called Halcyon Devices?",
    "What is our Net Promoter Score this quarter?",
]

INTENT_OPTIONS = {
    "account status": "Asks how one specific named account is doing overall",
    "theme or trend": "Asks about a pattern across multiple accounts, or a trend over time",
    "pricing": "Asks about pricing, discounts, invoices, or cost",
    "competitor": "Asks about a named competitor or switching vendors",
    "contact or people": "Asks about a specific person, role, or a contact change",
    "report or summary": "Asks for a broad summary or the weekly report content",
    "other": "None of the above",
}
EXPECTED_INTENT_UNANSWERABLE = (
    "other"  # they don't fit an intent about *our* data either way; scored separately
)


def build_question_set(conn: psycopg.Connection) -> list[dict]:
    from portolan.web.conversations import top_account_names

    examples = [f"What has {name} said recently?" for name in top_account_names(conn, 3)]
    out = []
    for q in examples:
        out.append({"question": q, "expected_intent": "account status", "answerable": True})
    for q, intent in WRITTEN_QUESTIONS:
        out.append({"question": q, "expected_intent": intent, "answerable": True})
    for q in UNANSWERABLE_QUESTIONS:
        out.append(
            {"question": q, "expected_intent": EXPECTED_INTENT_UNANSWERABLE, "answerable": False}
        )
    return out


def get_real_answer(conn, llm, embedder, question: str):
    from portolan.search.query import search
    from portolan.web.answer import answer_question

    result = search(conn, embedder, question, k=12, expand=0)
    answer = answer_question(conn, llm, embedder, question, k=12, expand=0, model=CHAT_MODEL)
    cited_slugs = set()
    for sentence in answer.sentences:
        for c in sentence.citations:
            cited_slugs.add(c.note_slug)
    return result, answer, cited_slugs


def jev_relevance_request(question: str, hits: list, *, mode: str, api_key: str) -> dict:
    """`mode`="batched": one Jev request, `state` carries the question plus
    every numbered candidate passage, 12 noul relevance sub-questions
    (one per candidate) + 1 noul "answerable at all" + 1 choice intent.
    `mode`="per_candidate": one Jev request PER candidate (measured
    separately per the task's "measure both").
    """
    if mode == "batched":
        passages = []
        for i, hit in enumerate(hits, 1):
            snippet = " ".join(hit.best_chunk.split())[:600]
            passages.append(f"--- Passage {i} (note: {hit.title}) ---\n{snippet}")
        state = f"User question: {question}\n\n" + "\n\n".join(passages)

        questions = {}
        for i in range(1, len(hits) + 1):
            questions[f"relevant_{i}"] = {
                "type": "noul",
                "instructions": (
                    f"Does Passage {i} above contain information that answers the user question?"
                ),
            }
        questions["answerable"] = {
            "type": "noul",
            "instructions": "Can the user question be answered from the passages above at all?",
        }
        questions["intent"] = {
            "type": "choice",
            "instructions": "What is the intent of the user's question?",
            "criteria": INTENT_OPTIONS,
        }
        return call_jev(state, questions, api_key=api_key, tag="stage1_batched")

    if mode == "per_candidate":
        # One request per candidate: state = that one passage, one noul question.
        responses = []
        for hit in hits:
            snippet = " ".join(hit.best_chunk.split())[:600]
            state = f"User question: {question}\n\nPassage (note: {hit.title}):\n{snippet}"
            questions = {
                "relevant": {
                    "type": "noul",
                    "instructions": "Does this passage contain information that answers the user question?",
                }
            }
            responses.append(
                call_jev(state, questions, api_key=api_key, tag="stage1_per_candidate")
            )
        return {"per_candidate_responses": responses}

    raise ValueError(mode)


def approx_tokens(text: str) -> float:
    """chars/4 heuristic -- documented, not hidden -- used only where a real
    token count from a model call isn't available (the after-gating estimate
    in stage1's cost section)."""
    return len(text) / 4.0


def run_stage1(
    conn, embedder, llm, *, api_key: str, per_candidate_subsample: int = 8, limit: int | None = None
) -> dict:
    from portolan.web.answer import CONTEXT_CHAR_BUDGET, _context_block

    questions = build_question_set(conn)
    if limit is not None:
        questions = questions[:limit]
    per_question = []

    subsample_idx = set(
        random.sample(range(len(questions)), k=min(per_candidate_subsample, len(questions)))
    )

    for idx, q in enumerate(questions):
        question = q["question"]
        result, answer, cited_slugs = get_real_answer(conn, llm, embedder, question)
        hits = result.hits[:12]

        jev_batched = jev_relevance_request(question, hits, mode="batched", api_key=api_key)
        answers = jev_batched.get("answers", {})

        candidate_rows = []
        for i, hit in enumerate(hits, 1):
            noul = answers.get(f"relevant_{i}", {}).get("noul")
            candidate_rows.append(
                {
                    "note_slug": hit.note_slug,
                    "title": hit.title,
                    "cited": hit.note_slug in cited_slugs,
                    "jev_relevant_noul": noul,
                }
            )
        answerable_noul = answers.get("answerable", {}).get("noul")
        intent_choice = answers.get("intent", {}).get("choice")
        intent_confidence = answers.get("intent", {}).get("confidence")

        # Context char budget actually sent to the writer model for this
        # question (before gating) -- rebuild via the real function so the
        # char count matches what answer_question() actually used.
        context_before, _, _ = _context_block(conn, hits, char_budget=CONTEXT_CHAR_BUDGET)

        row = {
            "question": question,
            "expected_intent": q["expected_intent"],
            "answerable_expected": q["answerable"],
            "n_candidates": len(hits),
            "candidates": candidate_rows,
            "cited_slugs": sorted(cited_slugs),
            "answer_had_content": answer.control_ok
            and not (len(answer.sentences) == 1 and not answer.sentences[0].citations),
            "jev_answerable_noul": answerable_noul,
            "jev_intent_choice": intent_choice,
            "jev_intent_confidence": intent_confidence,
            "real_answer_usage": answer.usage,
            "context_chars_before": len(context_before),
        }

        if idx in subsample_idx and hits:
            jev_relevance_request(question, hits, mode="per_candidate", api_key=api_key)
            row["per_candidate_sampled"] = True

        per_question.append(row)
        print(
            f"  [{idx + 1}/{len(questions)}] {question[:60]!r} -> "
            f"{len(cited_slugs)} cited, answerable_noul={answerable_noul}",
            flush=True,
        )

    return {"questions": per_question}


# ---------------------------------------------------------------------
# Stage 2: report ranking
# ---------------------------------------------------------------------


def dump_report_candidates(
    conn: psycopg.Connection, *, since: date, as_of: date, week: str
) -> dict:
    """Every signal claim in [since, as_of], with the report code's actual
    inclusion decision for the Outliers section (NOT EXISTS theme_members,
    mirroring portolan.report.outliers.list_outliers's own SQL exactly,
    just without discarding the excluded half)."""
    from portolan.report.accounts import home_account_slug

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT c.id, c.note_slug, c.fact_type, c.claim_text, c.source_slug,
                   c.locator, c.recorded,
                   EXISTS (SELECT 1 FROM theme_members tm WHERE tm.claim_id = c.id) AS in_theme
            FROM claims c
            JOIN notes n ON n.slug = c.note_slug
            WHERE n.type = 'signal' AND c.archived = false
              AND c.recorded::date BETWEEN %s AND %s
            ORDER BY c.recorded, c.id
            """,
            (since, as_of),
        )
        rows = cur.fetchall()

    candidates = []
    for (
        claim_id,
        note_slug,
        fact_type,
        claim_text,
        source_slug,
        locator,
        recorded,
        in_theme,
    ) in rows:
        account = home_account_slug(
            conn, note_slug=note_slug, fact_type=fact_type, claim_text=claim_text
        )
        candidates.append(
            {
                "claim_id": claim_id,
                "note_slug": note_slug,
                "account_slug": account,
                "fact_type": fact_type,
                "claim_text": claim_text,
                "source_slug": source_slug,
                "locator": locator,
                "recorded": str(recorded),
                "code_included_in_outliers": not in_theme,
            }
        )

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT sq.id, sq.reader, sq.question, sqa.answer_md, sqa.citations, sqa.week
            FROM standing_questions sq
            JOIN standing_question_answers sqa ON sqa.question_id = sq.id AND sqa.week = %s
            WHERE sq.active = true
            ORDER BY sq.id
            """,
            (week,),
        )
        sq_rows = cur.fetchall()
    standing = [
        {
            "question_id": qid,
            "reader": reader,
            "question": question,
            "answer_md": answer_md,
            "citations": citations,
            "week": week,
            "code_included": True,  # questions_feed._active_questions is unfiltered -- every active Q always shows
        }
        for qid, reader, question, answer_md, citations, week in sq_rows
    ]
    return {"outlier_candidates": candidates, "standing_questions": standing}


NEWSWORTHINESS_LEVELS = [
    "Not newsworthy -- routine, no action implied",
    "Minor -- worth knowing, no urgency",
    "Notable -- a customer-success leader should see this this week",
    "Major -- urgent, likely needs action this week",
]


def score_report_candidates(data: dict, *, api_key: str) -> dict:
    candidates = data["outlier_candidates"]
    BATCH = 15
    scored = []
    for start in range(0, len(candidates), BATCH):
        batch = candidates[start : start + BATCH]
        items = []
        for i, c in enumerate(batch, 1):
            items.append(
                f"--- Item {i} (account: {c['account_slug']}, type: {c['fact_type']}) ---\n{c['claim_text']}"
            )
        state = "\n\n".join(items)
        questions = {}
        for i in range(1, len(batch) + 1):
            questions[f"newsworthy_{i}"] = {
                "type": "score",
                "instructions": f"Rate the newsworthiness of Item {i} above for a weekly customer-success report.",
                "criteria": NEWSWORTHINESS_LEVELS,
            }
            questions[f"leader_{i}"] = {
                "type": "noul",
                "instructions": f"Would a customer-success leader need to see Item {i} above this week?",
            }
        resp = call_jev(state, questions, api_key=api_key, tag="stage2_outliers")
        answers = resp.get("answers", {})
        for i, c in enumerate(batch, 1):
            scored.append(
                {
                    **c,
                    "jev_newsworthy_score": answers.get(f"newsworthy_{i}", {}).get("score"),
                    "jev_leader_noul": answers.get(f"leader_{i}", {}).get("noul"),
                }
            )

    standing_scored = []
    if data["standing_questions"]:
        items = []
        for i, sq in enumerate(data["standing_questions"], 1):
            items.append(f"--- Item {i} ---\nQ: {sq['question']}\nA: {sq['answer_md']}")
        state = "\n\n".join(items)
        questions = {}
        for i in range(1, len(data["standing_questions"]) + 1):
            questions[f"newsworthy_{i}"] = {
                "type": "score",
                "instructions": f"Rate the newsworthiness of Item {i} above for a weekly customer-success report.",
                "criteria": NEWSWORTHINESS_LEVELS,
            }
            questions[f"leader_{i}"] = {
                "type": "noul",
                "instructions": f"Would a customer-success leader need to see Item {i} above this week?",
            }
        resp = call_jev(state, questions, api_key=api_key, tag="stage2_standing")
        answers = resp.get("answers", {})
        for i, sq in enumerate(data["standing_questions"], 1):
            standing_scored.append(
                {
                    **sq,
                    "jev_newsworthy_score": answers.get(f"newsworthy_{i}", {}).get("score"),
                    "jev_leader_noul": answers.get(f"leader_{i}", {}).get("noul"),
                }
            )

    return {"outlier_candidates_scored": scored, "standing_questions_scored": standing_scored}


# ---------------------------------------------------------------------
# Stage 3: numbers guard
# ---------------------------------------------------------------------

NUMERAL_RE = re.compile(r"(?<![\w./:-])(\d[\d,]*(?:\.\d+)?)(?![\w./:-])")
DATE_RE = re.compile(r"\b(19|20)\d{2}-\d{2}-\d{2}\b")
YEAR_ONLY_RE = re.compile(r"^(19|20)\d{2}$")
TIME_RE = re.compile(r"\b\d{1,2}:\d{2}(:\d{2})?\b")
HEADER_LINE_RE = re.compile(r"^(Message-ID|From|To|Cc|Date|Subject|Reply-To)\s*:", re.IGNORECASE)
TIMESTAMP_PREFIX_RE = re.compile(r"^\s*\[?\d{1,2}:\d{2}(:\d{2})?\]?\s*")
# A real quantity claim: needs a magnitude cue nearby, not a bare digit that
# happens to survive the date/time filters (a page number, a line-item
# index, etc). This is deliberately narrower than "any digit" -- the goal
# is 40 sentences that are genuinely about a number, matching the worked
# examples in the source corpus (percentages, dollar amounts, counts).
QUANTITY_CUE_RE = re.compile(
    r"(\$\s*\d|%|\bpercent\b|\bpoints?\b|\bunits?\b|\bdays?\b|\bweeks?\b|\bmonths?\b|"
    r"\bpumps?\b|\bbeds?\b|\bline item\b|\bcredit\b|\binvoice\b|\bdiscount\b|\brate\b)",
    re.IGNORECASE,
)


def _looks_like_bare_year_time_or_date(text: str, match_start: int, match_end: int) -> bool:
    # crude guard so the picked "numeral" isn't a bare calendar year, the
    # digits of an ISO date, or part of a clock timestamp -- keeps the
    # test about real quantities, not "09" lifted out of "09:00:00".
    window = text[max(0, match_start - 8) : match_end + 8]
    if DATE_RE.search(window) or TIME_RE.search(window):
        return True
    tok = text[match_start:match_end]
    return bool(YEAR_ONLY_RE.match(tok))


def _extract_source_files() -> list[Path]:
    files = []
    for sub in ("email", "tickets", "calls"):
        d = CORPUS_OUT / sub
        if d.exists():
            files.extend(sorted(d.glob("*")))
    return files


def _file_text(path: Path) -> str:
    if path.suffix == ".json":
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            return path.read_text()
        parts = [data.get("subject", ""), data.get("body", "")]
        for c in data.get("comments", []) or []:
            parts.append(c.get("body", ""))
        return "\n".join(p for p in parts if p)
    lines = [line for line in path.read_text().splitlines() if not HEADER_LINE_RE.match(line)]
    return "\n".join(lines)


def _sentences_with_numeral(text: str) -> list[tuple[str, str]]:
    """Returns (sentence, numeral) pairs for sentences containing a
    genuine quantity numeral (not a date, bare year, clock time, or a
    call-transcript speaker timestamp), with a nearby magnitude cue
    (percent, $, units, days, ...) so the "numeral" is actually the
    subject of the sentence, not an incidental digit."""
    out = []
    for sent in re.split(r"(?<=[.!?])\s+", text):
        sent = TIMESTAMP_PREFIX_RE.sub("", sent).strip()
        # strip a leading "[00:08:54] Speaker Name:" call-transcript prefix
        sent = re.sub(r"^[A-Z][\w .'-]{1,40}:\s*", "", sent)
        if len(sent) < 20 or len(sent) > 400:
            continue
        if not QUANTITY_CUE_RE.search(sent):
            continue
        for m in NUMERAL_RE.finditer(sent):
            if _looks_like_bare_year_time_or_date(sent, m.start(), m.end()):
                continue
            out.append((sent, m.group(1)))
            break
    return out


# Real corpus sentence-splitting/regex extraction over the raw call/email/
# ticket text (`_sentences_with_numeral`, `_extract_source_files`, kept
# above) turned out too noisy for this test on inspection: this corpus's
# claims and raw prose mostly spell quantities out in words ("three
# pumps", "twelve points off") rather than digits, and stripping
# timestamps/headers reliably out of free-form call transcripts is
# fragile. Auto-mining 20 clean, genuinely-verbatim digit-numeral
# sentences out of 180 source files turned out to need more hand
# fact-checking than 20 hand-built ones -- so these 20 are hand-built
# instead: real account/product names from portolan_jev's own corpus
# (`corpus/truth-table.yaml`), a one-paragraph synthetic "source" for
# each with a digit-form quantity actually in it, and a "sentence" that
# restates that exact number. `build_numeral_pairs` derives the 20
# negatives from these by substituting a different digit that does NOT
# appear anywhere in that source paragraph (checked programmatically
# below, not assumed).
HAND_BUILT_NUMERAL_SOURCES: list[dict] = [
    {
        "source_text": "Larkspur Regional Health's VP Supply Chain confirmed on 2026-08-25 that the 12 percent volume discount on Tidewater T2 consumables is expected at the 2027 renewal, given the delivery issues this year.",
        "sentence": "Larkspur Regional Health expects a 12 percent volume discount on Tidewater T2 consumables.",
        "numeral": "12",
    },
    {
        "source_text": "Vireo Surgical Group's practice administrator said the Brightline Medical proposal came in 15 percent cheaper than what they currently pay Halvard for consumables.",
        "sentence": "The Brightline Medical proposal was 15 percent cheaper than Halvard's pricing.",
        "numeral": "15",
    },
    {
        "source_text": "Halvard's operations manager reported that the Dayton distribution branch's on-time delivery rate fell to 61 percent for the two weeks ending 2026-08-20.",
        "sentence": "The Dayton distribution branch's on-time delivery rate fell to 61 percent.",
        "numeral": "61",
    },
    {
        "source_text": "St. Aldric Health's billing contact flagged a duplicated line item of $4,800 on the May 2026 HalvardCare invoice and requested a credit memo.",
        "sentence": "St. Aldric Health disputed a duplicated line item of $4,800 on its HalvardCare invoice.",
        "numeral": "4,800",
    },
    {
        "source_text": "Cobalt Ridge Medical Center's materials manager said its 2026 HalvardCare invoice included a pricing error worth $610 that has since been corrected.",
        "sentence": "Cobalt Ridge Medical Center flagged a $610 pricing error on its HalvardCare invoice.",
        "numeral": "610",
    },
    {
        "source_text": "Anchor Bay Surgical Partners' practice administrator said they run 6 Tidewater T2 pumps across two operating rooms as of September 2026.",
        "sentence": "Anchor Bay Surgical Partners runs 6 Tidewater T2 pumps across its operating rooms.",
        "numeral": "6",
    },
    {
        "source_text": "Northlake General Hospital's biomed lead said the Tidewater T3 order placed on 2026-08-15 arrived 3 days late, the third late delivery in three weeks.",
        "sentence": "Northlake General Hospital's most recent Tidewater delivery arrived 3 days late.",
        "numeral": "3",
    },
    {
        "source_text": "Meadowbrook Community Hospital's pharmacy director said their most recent Tidewater T2 consumable order was 5 days late, the second such delay this quarter.",
        "sentence": "Meadowbrook Community Hospital's Tidewater T2 order arrived 5 days late.",
        "numeral": "5",
    },
    {
        "source_text": "Driftwood Community Hospital's pharmacy director asked for a Tidewater Link connection to Epic, noting the hospital has now made 3 separate requests in two months.",
        "sentence": "Driftwood Community Hospital has made 3 separate requests for a Tidewater Link connection to Epic.",
        "numeral": "3",
    },
    {
        "source_text": "Ashgrove Medical Center's ICU nurse manager said the last 2 Tidewater T2 deliveries came in right on schedule.",
        "sentence": "Ashgrove Medical Center's last 2 Tidewater T2 deliveries were on schedule.",
        "numeral": "2",
    },
    {
        "source_text": "Wrenfield Health's supply chain lead said the facility has 14 Tidewater T2 pumps in active clinical use across its two campuses.",
        "sentence": "Wrenfield Health has 14 Tidewater T2 pumps in active clinical use.",
        "numeral": "14",
    },
    {
        "source_text": "Sable Valley Health's biomed lead reported that 4 occlusion alarms triggered falsely on Tidewater T2 units in July 2026.",
        "sentence": "4 occlusion alarms triggered falsely on Sable Valley Health's Tidewater T2 units.",
        "numeral": "4",
    },
    {
        "source_text": "Kestrel Medical Group's director said they moved 100 percent of their pump volume to Brightline Medical after ending the Halvard relationship in February 2026.",
        "sentence": "Kestrel Medical Group moved 100 percent of its pump volume to Brightline Medical.",
        "numeral": "100",
    },
    {
        "source_text": "St. Aldric Health's clinical engineering lead asked for 25 Spanish-language quick-start cards for the Tidewater T2.",
        "sentence": "St. Aldric Health requested 25 Spanish-language quick-start cards for the Tidewater T2.",
        "numeral": "25",
    },
    {
        "source_text": "Larkspur Memorial Hospital's biomed lead confirmed deliveries had been on time for 2 straight weeks and said he was dropping the second-supplier idea for now.",
        "sentence": "Larkspur Memorial Hospital reported 2 straight weeks of on-time deliveries.",
        "numeral": "2",
    },
    {
        "source_text": "Northlake Health System's VP Operations said the health system operates 9 Tidewater T3 units across its two hospitals.",
        "sentence": "Northlake Health System operates 9 Tidewater T3 units.",
        "numeral": "9",
    },
    {
        "source_text": "Cobalt Ridge Medical Center's materials manager said this was the 2nd late Tidewater T2 delivery in three weeks, arriving 4 days behind schedule.",
        "sentence": "Cobalt Ridge Medical Center's delivery arrived 4 days behind schedule.",
        "numeral": "4",
    },
    {
        "source_text": "Anchor Bay Surgical Partners' practice administrator asked for a discount citing budget pressure, referencing a $50,000 annual consumables spend.",
        "sentence": "Anchor Bay Surgical Partners referenced a $50,000 annual consumables spend.",
        "numeral": "50,000",
    },
    {
        "source_text": "Wrenfield East Hospital's facilities director said the Tidewater Link rollout has been delayed by 6 weeks relative to the original plan.",
        "sentence": "Wrenfield East Hospital's Tidewater Link rollout has been delayed by 6 weeks.",
        "numeral": "6",
    },
    {
        "source_text": "Ashgrove Riverside Hospital's supply chain contact said 8 of its 10 Tidewater T2 pumps have been upgraded to the latest firmware as of September 2026.",
        "sentence": "8 of Ashgrove Riverside Hospital's Tidewater T2 pumps have been upgraded to the latest firmware.",
        "numeral": "8",
    },
]


def build_numeral_pairs(n_positive: int = 20, n_negative: int = 20) -> list[dict]:
    positives = []
    for i, item in enumerate(HAND_BUILT_NUMERAL_SOURCES[:n_positive]):
        assert item["numeral"] in item["source_text"], f"numeral not actually in source, item {i}"
        positives.append({**item, "source_path": f"synthetic-{i:02d}"})

    negatives = []
    for i, item in enumerate(HAND_BUILT_NUMERAL_SOURCES[:n_negative]):
        real = item["numeral"].replace(",", "")
        real_n = float(real)
        for delta in (7, 11, 13, 17, 19, 23, -7, -11):
            fake_n = real_n + delta
            if fake_n <= 0:
                continue
            fake_str = str(int(fake_n)) if fake_n == int(fake_n) else str(fake_n)
            if fake_str not in item["source_text"]:
                break
        numeral_pattern = re.compile(r"(?<!\w)" + re.escape(item["numeral"]) + r"(?!\w)")
        fake_sentence, n_subs = numeral_pattern.subn(fake_str, item["sentence"], count=1)
        assert n_subs == 1, (
            f"numeral {item['numeral']!r} not found as a whole token in {item['sentence']!r}"
        )
        negatives.append(
            {
                "source_path": f"synthetic-{i:02d}",
                "source_text": item["source_text"],
                "sentence": fake_sentence,
                "numeral": fake_str,
                "original_numeral": item["numeral"],
            }
        )

    pairs = []
    for p in positives:
        pairs.append({**p, "label": "positive", "expected_noul_high": True})
    for n in negatives:
        pairs.append({**n, "label": "negative", "expected_noul_high": False})
    return pairs


def run_stage3(*, api_key: str) -> dict:
    pairs = build_numeral_pairs(20, 20)
    results = []
    for i, pair in enumerate(pairs):
        state = pair["source_text"][:6000]  # stay well inside the ~32k token / 150k char budget
        questions = {
            "numbers_ok": {
                "type": "noul",
                "instructions": (
                    "Does every number in this sentence appear verbatim in the source text above? "
                    f'Sentence: "{pair["sentence"]}"'
                ),
            }
        }
        resp = call_jev(state, questions, api_key=api_key, tag="stage3")
        noul = resp.get("answers", {}).get("numbers_ok", {}).get("noul")
        results.append(
            {
                "label": pair["label"],
                "sentence": pair["sentence"],
                "numeral": pair["numeral"],
                "source_path": pair["source_path"],
                "jev_noul": noul,
            }
        )
        print(
            f"  [{i + 1}/{len(pairs)}] {pair['label']:8s} numeral={pair['numeral']!r:>8} "
            f"noul={noul}",
            flush=True,
        )
    return {"pairs": results}


# ---------------------------------------------------------------------
# Analysis / report -- consolidates stage1/2/3's raw JSON into the numbers
# used in 01-spec/jev-eval-2026-09-16/C-chat-and-reports.md. Every number
# below is computed here, printed by this script -- never hand-estimated.
# ---------------------------------------------------------------------


def _pct(n: int, d: int) -> float:
    return round(100.0 * n / d, 1) if d else 0.0


def _percentile(values: list[float], p: float) -> float:
    """Linear-interpolation percentile (same method as numpy's default)."""
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (k - f) * (s[c] - s[f])


def analyze_stage1(threshold: float = 0.5) -> dict:
    data = json.loads((RESULTS_DIR / "stage1_raw.json").read_text())
    qs = data["questions"]

    answerable_qs = [q for q in qs if q["answerable_expected"]]
    unanswerable_qs = [q for q in qs if not q["answerable_expected"]]

    def _pr_at(thr: float) -> dict:
        tp = fp = fn = tn = 0
        for q in answerable_qs:
            for c in q["candidates"]:
                jev_yes = (c["jev_relevant_noul"] or 0) >= thr
                if c["cited"] and jev_yes:
                    tp += 1
                elif c["cited"] and not jev_yes:
                    fn += 1
                elif (not c["cited"]) and jev_yes:
                    fp += 1
                else:
                    tn += 1
        return {
            "threshold": thr,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": round(tp / (tp + fp), 3) if (tp + fp) else None,
            "recall": round(tp / (tp + fn), 3) if (tp + fn) else None,
        }

    pr_table = [_pr_at(t) for t in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)]
    at_threshold = _pr_at(threshold)
    tp, fp, fn, tn = at_threshold["tp"], at_threshold["fp"], at_threshold["fn"], at_threshold["tn"]
    precision, recall = at_threshold["precision"], at_threshold["recall"]

    # Unanswerable flagging: Jev's own "answerable" noul should read LOW
    # for the 5 genuinely-unanswerable questions, HIGH for the rest.
    correct_unanswerable_flags = sum(
        1 for q in unanswerable_qs if (q["jev_answerable_noul"] or 1) < threshold
    )
    correct_answerable_flags = sum(
        1 for q in answerable_qs if (q["jev_answerable_noul"] or 0) >= threshold
    )

    intent_correct = sum(1 for q in qs if q["jev_intent_choice"] == q["expected_intent"])
    # The 5 unanswerable questions were all hand-labeled expected_intent="other"
    # regardless of surface topic -- a labeling choice, not a Jev failure mode,
    # since a question can be topically about pricing/competitors and still be
    # unanswerable (wrong account). Reported separately so it isn't silently
    # blended into the headline number.
    intent_correct_answerable = sum(
        1 for q in answerable_qs if q["jev_intent_choice"] == q["expected_intent"]
    )

    # latency: calls are tagged at call time (call_jev(..., tag=...)), so
    # this filters on that tag rather than assuming any position/order in
    # the log -- stage1_batched (one per question) vs stage1_per_candidate
    # (the 8-question subsample, one request per candidate).
    all_calls = data["_jev_log"]
    batched_latencies = [
        c["latency_s"] for c in all_calls if c["tag"] == "stage1_batched" and not c["cached"]
    ]
    per_candidate_latencies = [
        c["latency_s"] for c in all_calls if c["tag"] == "stage1_per_candidate" and not c["cached"]
    ]

    n_sampled_questions = sum(1 for q in qs if q.get("per_candidate_sampled"))
    n_per_candidate_requests = sum(q["n_candidates"] for q in qs if q.get("per_candidate_sampled"))

    # token/cost before vs after gating: "after" keeps only candidates
    # Jev marked relevant (>= threshold) for each answerable question, and
    # estimates the resulting context size via the same
    # tokens-per-char ratio the REAL call observed for that question
    # (documented heuristic, not a second model call).
    total_before_tokens = 0
    total_after_tokens_est = 0
    total_before_chars = 0
    total_after_chars = 0
    for q in answerable_qs:
        real_tokens = q["real_answer_usage"].get("input_tokens", 0)
        before_chars = q["context_chars_before"]
        if before_chars <= 0 or real_tokens <= 0:
            continue
        ratio = real_tokens / before_chars
        kept_candidates = sum(
            1 for c in q["candidates"] if (c["jev_relevant_noul"] or 0) >= threshold
        )
        after_chars = (
            before_chars * (kept_candidates / q["n_candidates"]) if q["n_candidates"] else 0
        )
        total_before_tokens += real_tokens
        total_after_tokens_est += after_chars * ratio
        total_before_chars += before_chars
        total_after_chars += after_chars

    from portolan.pricing import estimate_cost

    cost_before = estimate_cost(CHAT_MODEL, total_before_tokens, 0)
    cost_after = estimate_cost(CHAT_MODEL, round(total_after_tokens_est), 0)

    return {
        "n_answerable": len(answerable_qs),
        "n_unanswerable": len(unanswerable_qs),
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "pr_by_threshold": pr_table,
        "unanswerable_flag_accuracy_pct": _pct(correct_unanswerable_flags, len(unanswerable_qs)),
        "answerable_flag_accuracy_pct": _pct(correct_answerable_flags, len(answerable_qs)),
        "intent_accuracy_pct": _pct(intent_correct, len(qs)),
        "intent_accuracy_answerable_only_pct": _pct(intent_correct_answerable, len(answerable_qs)),
        "n_questions": len(qs),
        "batched_latency_median_s": round(statistics.median(batched_latencies), 3)
        if batched_latencies
        else None,
        "batched_latency_p95_s": round(_percentile(batched_latencies, 0.95), 3)
        if batched_latencies
        else None,
        "batched_latency_n": len(batched_latencies),
        "per_candidate_latency_median_s": round(statistics.median(per_candidate_latencies), 3)
        if per_candidate_latencies
        else None,
        "per_candidate_latency_p95_s": round(_percentile(per_candidate_latencies, 0.95), 3)
        if per_candidate_latencies
        else None,
        "per_candidate_latency_n": len(per_candidate_latencies),
        "n_sampled_questions_for_per_candidate": n_sampled_questions,
        "n_per_candidate_requests": n_per_candidate_requests,
        "total_before_tokens": total_before_tokens,
        "total_after_tokens_est": round(total_after_tokens_est),
        "total_before_chars": total_before_chars,
        "total_after_chars": round(total_after_chars),
        "cost_before_usd": round(cost_before, 6),
        "cost_after_usd_est": round(cost_after, 6),
        "token_reduction_pct": round(100 * (1 - total_after_tokens_est / total_before_tokens), 1)
        if total_before_tokens
        else None,
    }


def analyze_stage2() -> dict:
    data = json.loads((RESULTS_DIR / "stage2_raw.json").read_text())
    outliers = data["outlier_candidates_scored"]
    standing = data["standing_questions_scored"]

    # Agreement: does Jev's noul>=0.5 (leader_noul) match the code's
    # own theme-membership-based inclusion decision?
    agree = sum(
        1 for c in outliers if (c["jev_leader_noul"] >= 0.5) == c["code_included_in_outliers"]
    )
    disagreements = [
        c for c in outliers if (c["jev_leader_noul"] >= 0.5) != c["code_included_in_outliers"]
    ]

    return {
        "n_outlier_candidates": len(outliers),
        "n_code_included": sum(1 for c in outliers if c["code_included_in_outliers"]),
        "agreement_pct": _pct(agree, len(outliers)),
        "n_disagreements": len(disagreements),
        "disagreements": disagreements,
        "n_standing_questions": len(standing),
        "standing_all_code_included": all(s["code_included"] for s in standing),
        "standing_mean_leader_noul": round(
            statistics.mean(s["jev_leader_noul"] for s in standing), 3
        )
        if standing
        else None,
    }


def analyze_stage3(threshold: float = 0.5) -> dict:
    data = json.loads((RESULTS_DIR / "stage3_raw.json").read_text())
    pairs = data["pairs"]
    positives = [p for p in pairs if p["label"] == "positive"]
    negatives = [p for p in pairs if p["label"] == "negative"]

    # Safety framing: a "miss" on a negative (fabricated number rated
    # numbers_ok) is the dangerous direction -- a fabricated number would
    # sail through the numbers guard. A "miss" on a positive (real number
    # rated NOT ok) is the safe-but-annoying direction -- a true sentence
    # gets rejected.
    false_negatives_dangerous = sum(1 for p in negatives if (p["jev_noul"] or 0) >= threshold)
    false_positives_annoying = sum(1 for p in positives if (p["jev_noul"] or 1) < threshold)

    correct = sum(1 for p in positives if (p["jev_noul"] or 0) >= threshold)
    correct += sum(1 for p in negatives if (p["jev_noul"] or 1) < threshold)

    return {
        "n_pairs": len(pairs),
        "n_positive": len(positives),
        "n_negative": len(negatives),
        "threshold": threshold,
        "accuracy_pct": _pct(correct, len(pairs)),
        "dangerous_misses": false_negatives_dangerous,  # fabricated number waved through
        "dangerous_miss_pct": _pct(false_negatives_dangerous, len(negatives)),
        "annoying_misses": false_positives_annoying,  # real number rejected
        "annoying_miss_pct": _pct(false_positives_annoying, len(positives)),
    }


def print_report() -> None:
    s1 = analyze_stage1()
    s2 = analyze_stage2()
    s3 = analyze_stage3()

    print("=== Stage 1: chat relevance gate ===")
    print(json.dumps(s1, indent=2))
    print()
    print("=== Stage 2: report ranking ===")
    print(json.dumps({k: v for k, v in s2.items() if k != "disagreements"}, indent=2))
    print()
    print("Disagreements (code decision vs Jev leader_noul>=0.5):")
    for d in s2["disagreements"][:10]:
        print(
            f"  code_included={d['code_included_in_outliers']!s:5} "
            f"jev_leader_noul={d['jev_leader_noul']:.2f} jev_score={d['jev_newsworthy_score']:.2f} "
            f"| {d['fact_type']:10} | {d['claim_text'][:90]}"
        )
    print()
    print("=== Stage 3: numbers guard ===")
    print(json.dumps(s3, indent=2))
    print()

    # Total spend across all three stages' raw logs (every call ever made,
    # cached or not -- a cache hit cost real money the first time it ran).
    # Read from the saved files, not the live JEV_LOG, since `report` can
    # be re-run standalone with an empty in-process log.
    total_jev_input = total_jev_output = total_jev_calls = 0
    for name in ("stage1_raw", "stage2_raw", "stage3_raw"):
        path = RESULTS_DIR / f"{name}.json"
        if not path.exists():
            continue
        log = json.loads(path.read_text()).get("_jev_log", [])
        total_jev_calls += len(log)
        total_jev_input += sum(c["input_tokens"] for c in log)
        total_jev_output += sum(c["output_tokens"] for c in log)
    jev_cost = total_jev_input * JEV_INPUT_PRICE_PER_M / 1_000_000

    from portolan.pricing import estimate_cost as _estimate_cost

    d1 = json.loads((RESULTS_DIR / "stage1_raw.json").read_text())
    or_input = sum(q["real_answer_usage"].get("input_tokens", 0) for q in d1["questions"])
    or_output = sum(q["real_answer_usage"].get("output_tokens", 0) for q in d1["questions"])
    or_cost = _estimate_cost(CHAT_MODEL, or_input, or_output)

    print("=== Total spend (this eval) ===")
    print(
        json.dumps(
            {
                "jev_calls": total_jev_calls,
                "jev_input_tokens": total_jev_input,
                "jev_output_tokens": total_jev_output,
                "jev_cost_usd": round(jev_cost, 6),
                "openrouter_chat_calls": len(d1["questions"]),
                "openrouter_input_tokens": or_input,
                "openrouter_output_tokens": or_output,
                "openrouter_cost_usd": round(or_cost, 6),
                "total_cost_usd": round(jev_cost + or_cost, 6),
            },
            indent=2,
        )
    )

    _save("analysis", {"stage1": s1, "stage2": s2, "stage3": s3})


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def _save(name: str, data: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}.json"
    path.write_text(json.dumps(data, indent=2, default=str))
    print(f"wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["stage1", "stage2", "stage3", "smoke", "report"])
    parser.add_argument("--week", default="2026-W34")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if args.stage == "report":
        print_report()
        return

    api_key = os.environ["TYPESAFE_API_KEY"]

    if args.stage == "smoke":
        resp = call_jev(
            "The sky is blue.",
            {"q": {"type": "noul", "instructions": "Does this mention a color?"}},
            api_key=api_key,
        )
        print(json.dumps(resp, indent=2))
        return

    if args.stage == "stage1":
        conn = get_conn()
        embedder = get_embedder()
        llm = get_openrouter_llm()
        try:
            data = run_stage1(conn, embedder, llm, api_key=api_key, limit=args.limit)
        finally:
            conn.close()
        data["_jev_log"] = JEV_LOG.calls
        _save("stage1_raw", data)
        return

    if args.stage == "stage2":
        from portolan.report.weekly import week_bounds

        since, as_of = week_bounds(args.week)
        conn = get_conn()
        try:
            data = dump_report_candidates(conn, since=since, as_of=as_of, week=args.week)
            scored = score_report_candidates(data, api_key=api_key)
        finally:
            conn.close()
        merged = {**data, **scored, "week": args.week, "since": str(since), "as_of": str(as_of)}
        merged["_jev_log"] = JEV_LOG.calls
        _save("stage2_raw", merged)
        return

    if args.stage == "stage3":
        data = run_stage3(api_key=api_key)
        data["_jev_log"] = JEV_LOG.calls
        _save("stage3_raw", data)
        return


if __name__ == "__main__":
    main()
