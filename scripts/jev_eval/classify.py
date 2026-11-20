#!/usr/bin/env python3
"""Track B: does Jev (TypeSafe AI) do Portolan's ingest-time classification,
and how does it compare with the current extractor's classification?

Reads:
  - corpus/truth-table.yaml   (73 planted facts, 4 distractors, 20 accounts)
  - corpus/out/{calls,tickets,email,rss,crm}  (the source documents)
  - portolan/extract/schema.py  (FACT_TYPES -- Portolan's closed fact-type list)

Runs four batched Jev tests per source item (never one question per call, per
the brief's 11.5x-cheaper batching rule):
  1. Per planted fact (73): choice fact_type, choice account, choice product,
     score severity, score sentiment, noul is_commitment.
  2. Per distractor (4) and per random non-fact passage (10): the same battery,
     to measure false-confidence on text with no real fact.
  3. Per sampled source document (30): choice main_account, three nouls
     (has_complaint / has_request / has_pricing).

Every number in the eventual report is computed and printed by this script,
never estimated by hand -- see the final "REPORT DATA" JSON block on stdout
and var/jev-eval/report_data.json.

Caches every raw Jev response under var/jev-eval/raw/<sha256>.json so re-runs
cost nothing for unchanged requests. The API key is read from
os.environ["TYPESAFE_API_KEY"]; never logged or printed.

Usage:
    set -a; source .env; set +a
    python scripts/jev_eval/classify.py [--concurrency 4] [--limit N]
                                         [--skip-latency-bench]
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import os
import random
import re
import statistics
import sys
import time
from pathlib import Path

import httpx
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from portolan.extract.schema import FACT_TYPES, FactType  # noqa: E402

TRUTH_TABLE_PATH = REPO_ROOT / "corpus" / "truth-table.yaml"
CORPUS_OUT = REPO_ROOT / "corpus" / "out"
VAR_DIR = REPO_ROOT / "var" / "jev-eval"
RAW_DIR = VAR_DIR / "raw"

API_URL = "https://api.typesafe.ai/v1/systemone"
PRICE_PER_M_INPUT_TOKENS = 0.042
SEED = 42

# Copied verbatim from 01-spec/model-bench-2026-09-12.md, the 14-model
# extraction comparison's own printed row for amazon/nova-lite-v1 (v0 corpus:
# 38 planted facts, 4 distractors, 78 sources). This is the reference this
# script's own printed Jev numbers get set beside in the report -- not
# re-measured here, just carried through as a labeled constant.
NOVA_LITE_BENCH_V0 = {
    "source": "01-spec/model-bench-2026-09-12.md, 14-model comparison table",
    "corpus_sources": 78,
    "corpus_facts": 38,
    "recall_pct": 56.2,
    "precision_pct": 14.3,
    "in_tokens": 698643,
    "out_tokens": 54408,
    "cost_usd": 0.055,
    "wall_s": 94,
}

SEVERITY_LEVELS = ["Informational", "Minor", "Significant", "Critical"]
SENTIMENT_LEVELS = ["Negative", "Neutral", "Positive"]
COMMITMENT_INSTRUCTIONS = (
    "Does this text contain an explicit commitment or promise made by a "
    "specific named person (customer-side or vendor-side)? Answer yes only "
    "if a named individual commits to a concrete future action."
)


# ---------------------------------------------------------------------------
# Loading truth table + registry-shaped data
# ---------------------------------------------------------------------------


def load_truth_table() -> dict:
    return yaml.safe_load(TRUTH_TABLE_PATH.read_text(encoding="utf-8"))


def build_accounts(tt: dict) -> list[dict]:
    """The 20 registry accounts, one line description each (kind, region,
    parent), mirroring what `portolan/registry` actually imports from
    `corpus/out/crm/accounts.csv` -- built here from the truth table because
    it carries the same slug ids the truth table's `facts[].entities` use."""
    accounts = tt["accounts"]
    by_id = {a["id"]: a for a in accounts}
    out = []
    for a in accounts:
        kind_label = a["kind"].replace("_", " ")
        desc = f"{kind_label} in the {a['region']} region"
        parent = by_id.get(a.get("parent")) if a.get("parent") else None
        if parent:
            desc += f", part of {parent['name']}"
        if a.get("status"):
            desc += f" ({a['status'].replace('_', ' ')})"
        out.append({"id": a["id"], "name": a["name"], "description": desc})
    return out


def build_products(tt: dict) -> list[dict]:
    """meta.products (5 items) from the truth table -- Portolan's product
    list -- plus a 'none' option, per the task."""
    products = tt["meta"]["products"]
    out = []
    for p in products:
        aliases = p.get("aliases") or []
        desc = f"Also called: {', '.join(aliases)}." if aliases else ""
        out.append({"id": p["id"], "name": p["name"], "description": desc})
    out.append(
        {
            "id": "none",
            "name": "None",
            "description": "No specific Halvard product or product family is named or implied.",
        }
    )
    return out


def build_fact_type_criteria() -> dict[str, str]:
    """Portolan's own closed fact-type list, `portolan.extract.schema.FACT_TYPES`
    -- the definition text is copied verbatim (same text the real extractor's
    prompt renders), 12 options, no 'other'-as-catch-all bias added."""
    return {ft.value: FACT_TYPES[ft].definition for ft in FactType}


def build_alias_index(tt: dict) -> dict[str, list[str]]:
    """entity id -> list of surface strings (name + aliases) to search for in
    call transcripts, used only to locate a fact's text span."""
    idx: dict[str, list[str]] = {}
    for a in tt["accounts"]:
        idx[a["id"]] = [a["name"], *a.get("aliases", [])]
    for p in tt["meta"]["products"]:
        idx[p["id"]] = [p["name"], *p.get("aliases", [])]
    for c in tt["meta"]["competitors"]:
        idx[c["id"]] = [c["name"], *c.get("aliases", [])]
    for person in tt["meta"]["internal_people"]:
        idx[person["id"]] = [person["name"]]
    for b in tt["meta"]["branches"]:
        idx[b["id"]] = [b["name"]]
    return idx


def build_contacts_to_account(tt: dict) -> dict[str, str]:
    return {c["id"]: c["account"] for c in tt.get("contacts", []) if c.get("account")}


# ---------------------------------------------------------------------------
# Source document loaders
# ---------------------------------------------------------------------------


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_call_text(source_id: str) -> str:
    return _read(CORPUS_OUT / "calls" / f"{source_id}.txt")


def load_email_text(source_id: str) -> str:
    return _read(CORPUS_OUT / "email" / f"{source_id}.txt")


def load_ticket_text(source_id: str) -> str:
    data = json.loads(_read(CORPUS_OUT / "tickets" / f"{source_id}.json"))
    lines = [
        f"Ticket {data['id']} — {data.get('subject', '')}",
        f"Requester: {data.get('requester', '')}, Account: {data.get('account', '')}, "
        f"Created: {data.get('created', '')}",
    ]
    for c in data.get("comments", []):
        lines.append(f"[{c['number']}] {c['author']}: {c['body']}")
    return "\n".join(lines)


_RSS_ITEMS_CACHE: dict[str, dict] | None = None


def _rss_items() -> dict[str, dict]:
    global _RSS_ITEMS_CACHE
    if _RSS_ITEMS_CACHE is None:
        items: dict[str, dict] = {}
        for fn in ("industry-news.json", "linkedin-larkspur.json"):
            for it in json.loads(_read(CORPUS_OUT / "rss" / fn)):
                items[it["id"]] = it
        _RSS_ITEMS_CACHE = items
    return _RSS_ITEMS_CACHE


def load_rss_text(source_id: str) -> str:
    it = _rss_items()[source_id]
    return f"{it['title']}\n\n{it['summary']}"


_ACCOUNTS_CSV_CACHE: dict[str, dict] | None = None


def _accounts_csv() -> dict[str, dict]:
    global _ACCOUNTS_CSV_CACHE
    if _ACCOUNTS_CSV_CACHE is None:
        rows = {}
        with (CORPUS_OUT / "crm" / "accounts.csv").open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                rows[row["crm_id"]] = row
        _ACCOUNTS_CSV_CACHE = rows
    return _ACCOUNTS_CSV_CACHE


def load_crm_account_text(source_id: str) -> str:
    """S-crm-A1010 -> a rendered CRM account record. Only used for the four
    facts (F35, F36, F68, F69) whose sole planted source is a CRM row, not a
    natural-language document."""
    raw = source_id.split("S-crm-", 1)[1]  # e.g. "A1010" -> accounts.csv crm_id "A-1010"
    m = re.match(r"^([A-Za-z]+)(\d+)$", raw)
    crm_id = f"{m.group(1)}-{m.group(2)}" if m else raw
    rows = _accounts_csv()
    row = rows[crm_id]
    parent = rows.get(row["parent_crm_id"]) if row.get("parent_crm_id") else None
    arr = int(row["arr"])
    txt = (
        f"CRM account record — {row['name']} ({row['crm_id']}), "
        f"kind: {row['kind']}, region: {row['region']}, domain: {row['domain']}, "
        f"annual recurring revenue: ${arr:,}"
    )
    if parent:
        txt += f", parent account: {parent['name']}"
    if row.get("status"):
        txt += f", status: {row['status']}"
    return txt


def load_source_text(source_id: str, source_type: str) -> str:
    if source_type == "call":
        return load_call_text(source_id)
    if source_type == "email":
        return load_email_text(source_id)
    if source_type == "ticket":
        return load_ticket_text(source_id)
    if source_type == "rss":
        return load_rss_text(source_id)
    if source_type == "crm":
        return load_crm_account_text(source_id)
    raise ValueError(f"unhandled source type: {source_type}")


# ---------------------------------------------------------------------------
# Call-transcript span extraction (facts whose primary source is a call)
# ---------------------------------------------------------------------------


def extract_call_span(
    text: str,
    phrasing: str | None,
    entities: list[str],
    alias_index: dict[str, list[str]],
    window: int = 1600,
) -> tuple[str, str]:
    """Returns (state_text, method) where method in
    {"quote_span", "entity_span", "full_document"}."""
    idx = -1
    if phrasing:
        m = re.search(r'"([^"]{6,200})"', phrasing)
        if m:
            quote = m.group(1)
            idx = text.find(quote)
            if idx == -1:
                words = quote.split()
                if len(words) > 4:
                    partial = " ".join(words[:5])
                    idx = text.lower().find(partial.lower())
    method = "quote_span"
    if idx == -1:
        method = "entity_span"
        for ent in entities:
            for alias in alias_index.get(ent, []):
                pos = text.find(alias)
                if pos != -1:
                    idx = pos
                    break
            if idx != -1:
                break
    if idx == -1:
        return text, "full_document"
    start = max(0, idx - window)
    end = min(len(text), idx + window)
    return text[start:end], method


# ---------------------------------------------------------------------------
# Jev client
# ---------------------------------------------------------------------------


def cache_key(state: str, questions: dict, item_id: str = "") -> str:
    # item_id disambiguates two different logical test items that happen to
    # share byte-identical state+questions (e.g. two facts planted in the
    # same short ticket/email) -- otherwise they'd collide on one cache file
    # and, under concurrent fresh calls, one item's on-disk cached answer
    # could silently become the OTHER item's real (and possibly different,
    # since Jev is not perfectly deterministic) response on a later re-run.
    payload = json.dumps(
        {"state": state, "questions": questions, "item_id": item_id}, sort_keys=True
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class JevClient:
    def __init__(self, api_key: str, concurrency: int) -> None:
        self.api_key = api_key
        self.sem = asyncio.Semaphore(concurrency)
        self.client = httpx.AsyncClient(timeout=90.0)
        self.calls_made = 0
        self.cache_hits = 0

    async def aclose(self) -> None:
        await self.client.aclose()

    async def call(
        self,
        state: str,
        questions: dict,
        *,
        bypass_cache: bool = False,
        tag: str = "",
        cache_namespace: str = "main",
    ) -> tuple[dict, float, bool]:
        # cache_namespace keeps the latency benchmark's fresh, repeated calls
        # (same state+questions as some main-experiment requests, deliberately
        # re-sent to measure real latency) from ever overwriting the
        # main-experiment's own cached, scored answer for that same content.
        key = cache_key(state, questions, item_id=tag)
        cache_path = RAW_DIR / f"{cache_namespace}-{key}.json"
        if not bypass_cache and cache_path.exists():
            self.cache_hits += 1
            return json.loads(cache_path.read_text(encoding="utf-8")), 0.0, True

        body = {"state": state, "model": "jev-latest", "questions": questions}
        async with self.sem:
            t0 = time.monotonic()
            last_exc: Exception | None = None
            for attempt in range(5):
                try:
                    resp = await self.client.post(
                        API_URL,
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        json=body,
                    )
                except httpx.RequestError as e:
                    last_exc = e
                    await asyncio.sleep(1.5 * (attempt + 1))
                    continue
                if resp.status_code == 200:
                    elapsed = time.monotonic() - t0
                    data = resp.json()
                    cache_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
                    self.calls_made += 1
                    return data, elapsed, False
                if resp.status_code in (429, 529) and attempt < 4:
                    retry_after = float(resp.headers.get("retry-after", 2 * (attempt + 1)))
                    await asyncio.sleep(retry_after)
                    continue
                raise RuntimeError(
                    f"Jev call failed [{tag}]: HTTP {resp.status_code} {resp.text[:300]}"
                )
            raise RuntimeError(f"Jev call failed after retries [{tag}]: {last_exc}")


# ---------------------------------------------------------------------------
# Question-set builders
# ---------------------------------------------------------------------------


def build_fact_questions(
    fact_type_criteria: dict, account_criteria: dict, product_criteria: dict
) -> dict:
    return {
        "fact_type": {
            "type": "choice",
            "instructions": (
                "Which category best describes the customer/vendor signal in this text? "
                "Pick the single best fit from this closed list."
            ),
            "criteria": fact_type_criteria,
        },
        "account": {
            "type": "choice",
            "instructions": (
                "Which of Halvard Medical's registered customer accounts is this text about? "
                "Pick the single best match."
            ),
            "criteria": account_criteria,
        },
        "product": {
            "type": "choice",
            "instructions": "Which Halvard product or product family does this text concern, if any?",
            "criteria": product_criteria,
        },
        "severity": {
            "type": "score",
            "instructions": (
                "How severe is this signal for the Halvard-customer relationship -- "
                "does it need urgent attention?"
            ),
            "criteria": SEVERITY_LEVELS,
        },
        "sentiment": {
            "type": "score",
            "instructions": "What is the customer's overall sentiment toward Halvard in this text?",
            "criteria": SENTIMENT_LEVELS,
        },
        "is_commitment": {"type": "noul", "instructions": COMMITMENT_INSTRUCTIONS},
    }


def build_doc_questions(account_criteria_with_none: dict) -> dict:
    return {
        "main_account": {
            "type": "choice",
            "instructions": (
                "Which customer account is this document mainly about? Pick 'none' if the "
                "document is internal or not about one specific customer account."
            ),
            "criteria": account_criteria_with_none,
        },
        "has_complaint": {
            "type": "noul",
            "instructions": "Does this document contain at least one customer complaint (something went wrong)?",
        },
        "has_request": {
            "type": "noul",
            "instructions": "Does this document contain at least one customer request for something new?",
        },
        "has_pricing": {
            "type": "noul",
            "instructions": (
                "Does this document contain at least one pricing, discount, budget, or invoice signal?"
            ),
        },
    }


# ---------------------------------------------------------------------------
# Request builders
# ---------------------------------------------------------------------------


def pick_primary_source(fact: dict) -> dict:
    srcs = fact["sources"]
    non_crm = [s for s in srcs if s["type"] != "crm"]
    return non_crm[0] if non_crm else srcs[0]


def build_fact_requests(
    tt: dict, account_ids: set[str], product_ids: set[str], alias_index: dict[str, list[str]]
) -> list[dict]:
    out = []
    for fact in tt["facts"]:
        if fact["type"] == "distractor":
            continue
        src = pick_primary_source(fact)
        raw_text = load_source_text(src["id"], src["type"])
        if src["type"] == "call":
            state, method = extract_call_span(
                raw_text, fact.get("phrasing"), fact.get("entities", []), alias_index
            )
        else:
            state, method = raw_text, "full_document"

        entities = fact.get("entities", [])
        expected_accounts = sorted({e for e in entities if e in account_ids})
        expected_products = sorted({e for e in entities if e in product_ids})
        out.append(
            {
                "kind": "fact",
                "id": fact["id"],
                "claim": fact["claim"],
                "expected_type": fact["type"],
                "expected_account": expected_accounts
                or None,  # lists (JSON-safe); None means no customer account
                "expected_product": expected_products or ["none"],
                "source_id": src["id"],
                "source_type": src["type"],
                "span_method": method,
                "state": state,
            }
        )
    return out


def build_distractor_requests(tt: dict) -> list[dict]:
    out = []
    for d in tt["facts"]:
        if d["type"] != "distractor":
            continue
        src = d["sources"][0]
        raw_text = load_source_text(src["id"], src["type"])
        out.append(
            {
                "kind": "distractor",
                "id": d["id"],
                "claim": d["claim"],
                "source_id": src["id"],
                "source_type": src["type"],
                "span_method": "full_document",
                "state": raw_text,
            }
        )
    return out


def _referenced_source_ids(tt: dict) -> dict[str, set[str]]:
    refs: dict[str, set[str]] = {"call": set(), "ticket": set(), "email": set(), "rss": set()}
    srcs = tt["sources"]
    for item in srcs["calls"]["items"]:
        refs["call"].add(item["id"])
    for item in srcs["tickets"]["items"]:
        refs["ticket"].add(item["id"])
    for item in srcs["emails"]["items"]:
        refs["email"].add(item["id"])
    for item in srcs["rss"]["items"]:
        refs["rss"].add(item["id"])
    return refs


def build_noise_requests(tt: dict, n: int, seed: int) -> list[dict]:
    """n random passages from source files NOT enumerated in the truth
    table's per-source-type `items` lists -- i.e. the generator's declared
    'noise' pool: no planted fact, no distractor."""
    refs = _referenced_source_ids(tt)
    candidates: list[tuple[str, str]] = []  # (source_type, source_id)

    for p in sorted((CORPUS_OUT / "calls").glob("S-call-*.txt")):
        sid = p.stem
        if sid not in refs["call"]:
            candidates.append(("call", sid))
    for p in sorted((CORPUS_OUT / "tickets").glob("S-ticket-*.json")):
        sid = p.stem
        if sid not in refs["ticket"]:
            candidates.append(("ticket", sid))
    for p in sorted((CORPUS_OUT / "email").glob("S-email-*.txt")):
        sid = p.stem
        if sid not in refs["email"]:
            candidates.append(("email", sid))
    for sid in sorted(_rss_items().keys()):
        if sid not in refs["rss"]:
            candidates.append(("rss", sid))

    rng = random.Random(seed)
    chosen = rng.sample(candidates, k=min(n, len(candidates)))

    out = []
    for source_type, sid in chosen:
        raw_text = load_source_text(sid, source_type)
        if source_type == "call" and len(raw_text) > 1600:
            max_start = len(raw_text) - 1400
            start = rng.randint(0, max_start)
            state = raw_text[start : start + 1400]
            method = "random_window"
        else:
            state = raw_text
            method = "full_document"
        out.append(
            {
                "kind": "noise",
                "id": f"noise-{source_type}-{sid}",
                "source_id": sid,
                "source_type": source_type,
                "span_method": method,
                "state": state,
            }
        )
    return out


def _facts_by_source(tt: dict) -> dict[str, list[dict]]:
    by_type: dict[str, dict] = {f["id"]: f["type"] for f in tt["facts"]}
    out: dict[str, list[str]] = {}
    for section in ("calls", "tickets", "emails"):
        for item in tt["sources"][section]["items"]:
            out[item["id"]] = [by_type.get(fid, "?") for fid in item.get("facts", [])]
    return out


def build_doc_requests(tt: dict, n: int, seed: int) -> list[dict]:
    """30 sampled source documents drawn from the truth table's own
    calls/tickets/emails `items` lists (the only sources with explicit,
    checkable ground truth: an account, or a derivable one, plus each
    document's own fact-type membership). CRM and RSS are excluded from this
    sample -- they are not narrative documents in the same sense."""
    contacts_to_account = build_contacts_to_account(tt)
    facts_by_id = {f["id"]: f for f in tt["facts"]}

    candidates: list[dict] = []
    for item in tt["sources"]["calls"]["items"]:
        candidates.append({"source_type": "call", **item})
    for item in tt["sources"]["tickets"]["items"]:
        candidates.append({"source_type": "ticket", **item})
    for item in tt["sources"]["emails"]["items"]:
        candidates.append({"source_type": "email", **item})

    rng = random.Random(seed + 1)
    chosen = rng.sample(candidates, k=min(n, len(candidates)))

    out = []
    for item in chosen:
        sid = item["id"]
        source_type = item["source_type"]
        raw_text = load_source_text(sid, source_type)

        if "account" in item:
            gt_account = item["account"]
        elif source_type == "email":
            from_id = item.get("from")
            gt_account = contacts_to_account.get(from_id)  # None if internal person
        else:
            gt_account = None

        fact_ids = item.get("facts", [])
        fact_types = {facts_by_id[fid]["type"] for fid in fact_ids if fid in facts_by_id}
        out.append(
            {
                "kind": "doc",
                "id": sid,
                "source_type": source_type,
                "gt_account": gt_account,
                "gt_has_complaint": "complaint" in fact_types,
                "gt_has_request": "request" in fact_types,
                "gt_has_pricing": "pricing" in fact_types,
                "is_distractor": bool(item.get("distractor")),
                "state": raw_text,
            }
        )
    return out


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


async def run_batch(
    jev: JevClient, requests: list[dict], questions: dict, *, bypass_cache: bool = False
) -> tuple[list[dict], list[float]]:
    async def _one(req: dict) -> tuple[dict, float]:
        data, elapsed, cached = await jev.call(
            req["state"], questions, bypass_cache=bypass_cache, tag=req["id"]
        )
        return {**req, "response": data, "cached": cached}, elapsed

    results_elapsed = await asyncio.gather(*[_one(r) for r in requests])
    results = [r for r, _ in results_elapsed]
    elapsed = [e for _, e in results_elapsed if e > 0]
    return results, elapsed


async def run_latency_benchmark(
    jev: JevClient, sample_requests: list[dict], questions: dict, n: int
) -> dict:
    """Fresh (cache-bypassed) timing on the same n requests, once sequentially
    (concurrency 1) and once through the shared client's configured
    concurrency (already 4 by default). Always makes real calls -- meant to
    measure real network + model latency, not cache hits."""
    subset = sample_requests[:n]
    bench_in_tok = 0
    bench_out_tok = 0

    t0 = time.monotonic()
    seq_elapsed = []
    for req in subset:
        data, elapsed, _ = await jev.call(
            req["state"],
            questions,
            bypass_cache=True,
            tag=f"seqbench-{req['id']}",
            cache_namespace="bench-seq",
        )
        seq_elapsed.append(elapsed)
        bench_in_tok += data["usage"]["input_tokens"]
        bench_out_tok += data["usage"]["output_tokens"]
    seq_wall = time.monotonic() - t0

    t0 = time.monotonic()

    async def _one(req):
        data, elapsed, _ = await jev.call(
            req["state"],
            questions,
            bypass_cache=True,
            tag=f"concbench-{req['id']}",
            cache_namespace="bench-conc",
        )
        return elapsed, data["usage"]["input_tokens"], data["usage"]["output_tokens"]

    conc_results = await asyncio.gather(*[_one(r) for r in subset])
    conc_wall = time.monotonic() - t0
    conc_elapsed = [e for e, _, _ in conc_results]
    for _, itok, otok in conc_results:
        bench_in_tok += itok
        bench_out_tok += otok

    def stats(xs):
        xs = sorted(xs)
        return {
            "n": len(xs),
            "total_s": round(sum(xs), 3),
            "mean_s": round(statistics.mean(xs), 3) if xs else 0,
            "median_s": round(statistics.median(xs), 3) if xs else 0,
            "p95_s": round(xs[int(0.95 * (len(xs) - 1))], 3) if xs else 0,
        }

    return {
        "n": len(subset),
        "sequential": {"wall_s": round(seq_wall, 3), **stats(seq_elapsed)},
        "concurrency_4": {"wall_s": round(conc_wall, 3), **stats(conc_elapsed)},
        "speedup_x": round(seq_wall / conc_wall, 2) if conc_wall > 0 else None,
        "total_calls": 2 * len(subset),
        "input_tokens": bench_in_tok,
        "output_tokens": bench_out_tok,
        "cost_usd": round(bench_in_tok / 1_000_000 * PRICE_PER_M_INPUT_TOKENS, 6),
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_choice_field(results: list[dict], qkey: str, expected_key: str) -> dict:
    total = 0
    correct = 0
    confusions: dict[str, int] = {}
    excluded = 0
    for r in results:
        expected = r.get(expected_key)
        if expected is None:
            excluded += 1
            continue
        predicted = r["response"]["answers"][qkey]["choice"]
        total += 1
        if predicted in expected:
            correct += 1
        else:
            exp_label = "|".join(sorted(expected))
            key = f"{exp_label} -> {predicted}"
            confusions[key] = confusions.get(key, 0) + 1
    acc = round(100 * correct / total, 1) if total else None
    top_confusions = sorted(confusions.items(), key=lambda kv: -kv[1])[:10]
    return {
        "n_scored": total,
        "n_excluded_no_ground_truth": excluded,
        "correct": correct,
        "accuracy_pct": acc,
        "confusion_pairs": top_confusions,
    }


def score_bool_field(
    results: list[dict], qkey: str, expected_key: str, threshold: float = 0.5
) -> dict:
    total = 0
    correct = 0
    tp = fp = tn = fn = 0
    for r in results:
        expected = r[expected_key]
        predicted_p = r["response"]["answers"][qkey]["noul"]
        predicted = predicted_p >= threshold
        total += 1
        if predicted == expected:
            correct += 1
        if predicted and expected:
            tp += 1
        elif predicted and not expected:
            fp += 1
        elif not predicted and not expected:
            tn += 1
        else:
            fn += 1
    return {
        "n_scored": total,
        "correct": correct,
        "accuracy_pct": round(100 * correct / total, 1) if total else None,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "threshold": threshold,
    }


def product_family(slug: str) -> str:
    """Mirrors portolan.report.themes._product_family / waves._product_family
    exactly: the slug segment before its first hyphen."""
    if slug == "none":
        return "none"
    return slug.split("-", 1)[0]


def score_product_family(results: list[dict]) -> dict:
    total = correct = 0
    confusions: dict[str, int] = {}
    for r in results:
        expected_families = {product_family(e) for e in r["expected_product"]}
        predicted = r["response"]["answers"]["product"]["choice"]
        predicted_family = product_family(predicted)
        total += 1
        if predicted_family in expected_families:
            correct += 1
        else:
            key = f"{'|'.join(sorted(expected_families))} -> {predicted_family}"
            confusions[key] = confusions.get(key, 0) + 1
    return {
        "n_scored": total,
        "correct": correct,
        "accuracy_pct": round(100 * correct / total, 1) if total else None,
        "confusion_pairs": sorted(confusions.items(), key=lambda kv: -kv[1])[:10],
    }


def false_confidence_stats(results: list[dict], threshold: float = 0.5) -> dict:
    fact_type_conf = [r["response"]["answers"]["fact_type"]["confidence"] for r in results]
    account_conf = [r["response"]["answers"]["account"]["confidence"] for r in results]
    product_not_none = [r["response"]["answers"]["product"]["choice"] != "none" for r in results]
    n = len(results)

    def rate_over(xs, thr):
        return round(100 * sum(1 for x in xs if x >= thr) / len(xs), 1) if xs else None

    return {
        "n": n,
        "fact_type_confidence_mean": round(statistics.mean(fact_type_conf), 3) if n else None,
        "fact_type_confidence_ge_0.5_pct": rate_over(fact_type_conf, 0.5),
        "fact_type_confidence_ge_0.7_pct": rate_over(fact_type_conf, 0.7),
        "account_confidence_mean": round(statistics.mean(account_conf), 3) if n else None,
        "account_confidence_ge_0.5_pct": rate_over(account_conf, 0.5),
        "account_confidence_ge_0.7_pct": rate_over(account_conf, 0.7),
        "product_assigned_not_none_pct": round(100 * sum(product_not_none) / n, 1) if n else None,
    }


def usage_totals(results: list[dict]) -> dict:
    in_tok = sum(r["response"]["usage"]["input_tokens"] for r in results)
    out_tok = sum(r["response"]["usage"]["output_tokens"] for r in results)
    return {
        "n_requests": len(results),
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "cost_usd": round(in_tok / 1_000_000 * PRICE_PER_M_INPUT_TOKENS, 6),
    }


def severity_sentiment_summary(results: list[dict]) -> dict:
    sev_scores = [r["response"]["answers"]["severity"]["score"] for r in results]
    sen_scores = [r["response"]["answers"]["sentiment"]["score"] for r in results]

    def hist(scores, legend):
        n_levels = len(legend)
        counts = {legend[str(i)]: 0 for i in range(n_levels)}
        for s in scores:
            level = min(n_levels - 1, max(0, round(s)))
            counts[legend[str(level)]] += 1
        return counts

    sev_legend = results[0]["response"]["answers"]["severity"]["legend"] if results else {}
    sen_legend = results[0]["response"]["answers"]["sentiment"]["legend"] if results else {}

    ordered = sorted(range(len(results)), key=lambda i: sev_scores[i])
    n = len(ordered)
    pick_idx = (
        sorted({ordered[0], ordered[n // 4], ordered[n // 2], ordered[(3 * n) // 4], ordered[-1]})
        if n
        else []
    )
    spot_checks = []
    for i in pick_idx:
        r = results[i]
        spot_checks.append(
            {
                "id": r["id"],
                "claim": r.get("claim"),
                "severity_score": r["response"]["answers"]["severity"]["score"],
                "severity_legend": sev_legend,
                "sentiment_score": r["response"]["answers"]["sentiment"]["score"],
                "sentiment_legend": sen_legend,
            }
        )

    return {
        "n": len(results),
        "severity_mean": round(statistics.mean(sev_scores), 3) if sev_scores else None,
        "severity_stdev": round(statistics.stdev(sev_scores), 3) if len(sev_scores) > 1 else None,
        "severity_histogram": hist(sev_scores, sev_legend) if sev_legend else {},
        "sentiment_mean": round(statistics.mean(sen_scores), 3) if sen_scores else None,
        "sentiment_stdev": round(statistics.stdev(sen_scores), 3) if len(sen_scores) > 1 else None,
        "sentiment_histogram": hist(sen_scores, sen_legend) if sen_legend else {},
        "spot_checks": spot_checks,
    }


def pick_example_misses(results: list[dict], n: int = 5) -> list[dict]:
    misses = []
    for r in results:
        pred_type = r["response"]["answers"]["fact_type"]["choice"]
        pred_account = r["response"]["answers"]["account"]["choice"]
        pred_product = r["response"]["answers"]["product"]["choice"]
        type_miss = pred_type != r["expected_type"]
        account_miss = (
            r["expected_account"] is not None and pred_account not in r["expected_account"]
        )
        product_miss = pred_product not in r["expected_product"]
        if type_miss or account_miss or product_miss:
            misses.append(
                {
                    "id": r["id"],
                    "claim": r["claim"],
                    "source_id": r["source_id"],
                    "span_method": r["span_method"],
                    "expected_type": r["expected_type"],
                    "predicted_type": pred_type,
                    "type_miss": type_miss,
                    "expected_account": sorted(r["expected_account"])
                    if r["expected_account"]
                    else None,
                    "predicted_account": pred_account,
                    "account_miss": account_miss,
                    "expected_product": sorted(r["expected_product"]),
                    "predicted_product": pred_product,
                    "product_miss": product_miss,
                }
            )
    return misses[:n]


def score_doc_level(results: list[dict]) -> dict:
    acc_total = acc_correct = 0
    confusions: dict[str, int] = {}
    for r in results:
        gt = r["gt_account"]
        predicted = r["response"]["answers"]["main_account"]["choice"]
        expected = gt if gt is not None else "none"
        acc_total += 1
        if predicted == expected:
            acc_correct += 1
        else:
            key = f"{expected} -> {predicted}"
            confusions[key] = confusions.get(key, 0) + 1

    complaint = score_bool_field(results, "has_complaint", "gt_has_complaint")
    request = score_bool_field(results, "has_request", "gt_has_request")
    pricing = score_bool_field(results, "has_pricing", "gt_has_pricing")

    return {
        "main_account": {
            "n_scored": acc_total,
            "correct": acc_correct,
            "accuracy_pct": round(100 * acc_correct / acc_total, 1) if acc_total else None,
            "confusion_pairs": sorted(confusions.items(), key=lambda kv: -kv[1])[:10],
        },
        "has_complaint": complaint,
        "has_request": request,
        "has_pricing": pricing,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def async_main(args: argparse.Namespace) -> dict:
    api_key = os.environ["TYPESAFE_API_KEY"]
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    tt = load_truth_table()
    accounts = build_accounts(tt)
    products = build_products(tt)
    account_ids = {a["id"] for a in accounts}
    product_ids = {p["id"] for p in products if p["id"] != "none"}
    alias_index = build_alias_index(tt)

    fact_type_criteria = build_fact_type_criteria()
    account_criteria = {a["id"]: f"{a['name']} — {a['description']}" for a in accounts}
    account_criteria_with_none = {
        **account_criteria,
        "none": "Internal document, or not mainly about one specific customer account.",
    }
    product_criteria = {p["id"]: f"{p['name']} — {p['description']}".rstrip(" —") for p in products}

    fact_questions = build_fact_questions(fact_type_criteria, account_criteria, product_criteria)
    doc_questions = build_doc_questions(account_criteria_with_none)

    fact_requests = build_fact_requests(tt, account_ids, product_ids, alias_index)
    distractor_requests = build_distractor_requests(tt)
    noise_requests = build_noise_requests(tt, n=10, seed=SEED)
    doc_requests = build_doc_requests(tt, n=30, seed=SEED)

    if args.limit:
        fact_requests = fact_requests[: args.limit]
        distractor_requests = distractor_requests[: args.limit]
        noise_requests = noise_requests[: min(args.limit, len(noise_requests))]
        doc_requests = doc_requests[: min(args.limit, len(doc_requests))]

    print(
        f"Loaded {len(fact_requests)} fact requests, {len(distractor_requests)} distractors, "
        f"{len(noise_requests)} noise passages, {len(doc_requests)} doc-level requests.",
        file=sys.stderr,
    )

    jev = JevClient(api_key, concurrency=args.concurrency)
    t_main_start = time.monotonic()

    fact_results, _fact_elapsed = await run_batch(jev, fact_requests, fact_questions)
    distractor_results, _distractor_elapsed = await run_batch(
        jev, distractor_requests, fact_questions
    )
    noise_results, _noise_elapsed = await run_batch(jev, noise_requests, fact_questions)
    doc_results, _doc_elapsed = await run_batch(jev, doc_requests, doc_questions)

    main_wall_s = time.monotonic() - t_main_start

    latency_bench = None
    if not args.skip_latency_bench:
        print(
            f"Running latency benchmark ({args.latency_bench_n} requests, fresh calls)...",
            file=sys.stderr,
        )
        latency_bench = await run_latency_benchmark(
            jev, fact_requests, fact_questions, n=args.latency_bench_n
        )

    await jev.aclose()

    non_fact_results = distractor_results + noise_results

    meta = {
        "n_facts": len(fact_results),
        "n_distractors": len(distractor_results),
        "n_noise": len(noise_results),
        "n_docs": len(doc_results),
        "concurrency": args.concurrency,
        "main_run_wall_s": round(main_wall_s, 2),
        "api_calls_made_this_run": jev.calls_made,
        "cache_hits_this_run": jev.cache_hits,
    }

    # fact_type expected needs to be a set for score_choice_field's `in expected` check
    for r in fact_results:
        r["expected_type_set"] = [r["expected_type"]]

    accuracy = {
        "fact_type": score_choice_field(fact_results, "fact_type", "expected_type_set"),
        "account": score_choice_field(fact_results, "account", "expected_account"),
        "product_exact": score_choice_field(fact_results, "product", "expected_product"),
        "product_family": score_product_family(fact_results),
    }

    usage = {
        "facts": usage_totals(fact_results),
        "distractors": usage_totals(distractor_results),
        "noise": usage_totals(noise_results),
        "docs": usage_totals(doc_results),
        "latency_benchmark_extra_calls": {
            "n_requests": (latency_bench["n"] * 2) if latency_bench else 0,
        },
    }
    usage["experiment_total"] = {
        "n_requests": sum(
            usage[k]["n_requests"] for k in ("facts", "distractors", "noise", "docs")
        ),
        "input_tokens": sum(
            usage[k]["input_tokens"] for k in ("facts", "distractors", "noise", "docs")
        ),
        "output_tokens": sum(
            usage[k]["output_tokens"] for k in ("facts", "distractors", "noise", "docs")
        ),
        "cost_usd": round(
            sum(usage[k]["cost_usd"] for k in ("facts", "distractors", "noise", "docs")), 6
        ),
    }
    bench_cost = latency_bench["cost_usd"] if latency_bench else 0.0
    bench_calls = latency_bench["total_calls"] if latency_bench else 0
    usage["grand_total_including_latency_benchmark"] = {
        "n_requests": usage["experiment_total"]["n_requests"] + bench_calls,
        "cost_usd": round(usage["experiment_total"]["cost_usd"] + bench_cost, 6),
        "note": (
            "latency benchmark re-sends the first N fact requests twice (sequential + "
            "concurrency-4 passes) purely to time them; its tokens/cost are real API spend "
            "but are not part of the classification dataset scored above."
        ),
    }

    false_confidence = false_confidence_stats(non_fact_results)
    sev_sen = severity_sentiment_summary(fact_results)
    example_misses = pick_example_misses(fact_results, n=5)
    doc_scores = score_doc_level(doc_results)

    span_methods = {}
    for r in fact_requests:
        span_methods[r["span_method"]] = span_methods.get(r["span_method"], 0) + 1

    report = {
        "meta": meta,
        "span_methods": span_methods,
        "accuracy_per_fact_field": accuracy,
        "false_confidence_on_non_fact_text": false_confidence,
        "severity_sentiment": sev_sen,
        "example_misses": example_misses,
        "doc_level": doc_scores,
        "usage_and_cost": usage,
        "nova_lite_bench_v0_reference": NOVA_LITE_BENCH_V0,
        "latency_benchmark": latency_bench,
    }

    (VAR_DIR / "report_data.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )

    full_results = {
        "fact_results": fact_results,
        "distractor_results": distractor_results,
        "noise_results": noise_results,
        "doc_results": doc_results,
    }
    (VAR_DIR / "results.json").write_text(
        json.dumps(full_results, indent=2, default=str), encoding="utf-8"
    )

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--limit", type=int, default=None, help="debug: cap each request group to N"
    )
    parser.add_argument("--skip-latency-bench", action="store_true")
    parser.add_argument("--latency-bench-n", type=int, default=24)
    args = parser.parse_args()

    if "TYPESAFE_API_KEY" not in os.environ:
        print("TYPESAFE_API_KEY not set -- source .env first", file=sys.stderr)
        sys.exit(1)

    report = asyncio.run(async_main(args))
    print("\n=== REPORT DATA (also written to var/jev-eval/report_data.json) ===")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
