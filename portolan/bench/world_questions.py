"""WP10: search-evaluation questions built from a v2 world's state
(`01-spec/world-generator-v2-2026-09-22.md` section 7, "Search-eval
questions", and WP10's row in section 9).

Code chooses what each question asks and which documents answer it; an
open-weight model only words it. Four kinds:

- `single_fact`: one scripted fact. Gold = every source carrying a label of
  that fact id (a WP5 restatement or recovery and a WP6 near-duplicate copy
  carry the same id, so they count).
- `multi_doc`: two or three facts from one arc (a chain of events at one
  account), each in a different planned document. Gold = the union.
- `recency`: the latest state. `supersession` asks for the value a WP6
  near-duplicate changed (gold = the new fact's sources; a "must not" answer
  fact names the old value); `arc_latest` asks for the latest fact of an arc
  whose earlier facts sit in other documents.
- `cross_source`: a CRM account record's field joined with a fact from a
  call the record links (a true link that WP6's link noise left in place).

Answer facts are the world's own claims (`world_claim`), or, for a CRM field
and a "must not" line, a sentence the wording model writes and code checks
for the exact value (`open_weight_model`). No sentence a Claude agent wrote
reaches a question or an answer fact (CLAUDE.md, training-data provenance).

Validation (`validate_question`): BM25 over the world's sources must rank a
gold source in its top 50, then one open-weight judge reads the gold
sources together with BM25's top non-gold sources, shuffled and unlabeled,
and says which sources state each answer fact. A question is kept only when
every answer fact is stated by a gold source and no other source gives a
different answer (for a recency question, an older source giving the older
state is expected and recorded as superseded). Non-gold sources that also
answer become `valid_doc_ids`, which the invalid-extra-documents metric
excludes, as ERB's gold correction does. Every drop is logged with its stage
and reason.

Every paid call is metered under `v2_search_eval`
(`portolan.training.spend`), checked before it starts against the run's cap
at a worst case (full `max_tokens`, twice the list price).
"""

from __future__ import annotations

import calendar
import hashlib
import json
import random
import re
import threading
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from portolan.bench.erb import Bm25Index, bm25_tokenize
from portolan.bench.world_v2 import LoadedWorld, load_world
from portolan.llm.json_util import extract_json_object

__all__ = [
    "DEFAULT_COUNTS",
    "DEFAULT_JUDGE_MODEL",
    "DEFAULT_WORDING_MODELS",
    "GENERATOR_VERSION",
    "KINDS",
    "PARAPHRASE_LEVELS",
    "SPEND_KEY",
    "AnswerItem",
    "BudgetExceeded",
    "BuildConfig",
    "BuildResult",
    "CallLog",
    "ModelReply",
    "QuestionIntent",
    "QuestionWorld",
    "SearchEvalBudget",
    "build_intents",
    "build_questions",
    "check_wording",
    "estimate_run_cost",
    "fixture_judge_call",
    "fixture_wording_call",
    "judge_decision",
    "lexical_overlap",
    "load_question_world",
    "model_allowed",
    "openrouter_model_call",
    "paraphrase_level",
    "world_bm25_index",
    "write_build_outputs",
]

SPEND_KEY = "v2_search_eval"
GENERATOR_VERSION = "wp10-2026-09-23.1"
KINDS: tuple[str, ...] = ("single_fact", "multi_doc", "recency", "cross_source")
KIND_PREFIX = {"single_fact": "sf", "multi_doc": "md", "recency": "rc", "cross_source": "cs"}
PARAPHRASE_LEVELS: tuple[str, ...] = ("close", "moderate", "heavy")
_PARAPHRASE_WEIGHTS = (0.3, 0.4, 0.3)

# Section 7's 120 per test world covers nine types; these four are WP10's.
DEFAULT_COUNTS: dict[str, int] = {
    "single_fact": 40,
    "multi_doc": 16,
    "recency": 16,
    "cross_source": 24,
}

# Wording: open-weight writers from three families (the v2 writer rotation);
# judge: the label judge's model. Both must be on the training-set builder's
# teacher allowlist, since question text may become training or eval text.
DEFAULT_WORDING_MODELS: tuple[str, ...] = (
    "qwen/qwen3-235b-a22b-2507",
    "deepseek/deepseek-v4-flash",
    "mistralai/mistral-small-3.2-24b-instruct",
)
DEFAULT_JUDGE_MODEL = "deepseek/deepseek-v4.1-flash"

BM25_DEPTH = 50
EXTRAS_DEPTH = 10
# CRM fields a cross-source question may ask for (parent_crm_id is an id).
CROSS_SOURCE_FIELDS: tuple[str, ...] = (
    "renewal_date",
    "arr",
    "renewal_stage",
    "competitors",
    "owner_previous",
    "status",
)
# Instruction words for the wording prompt, never copied into output.
TOPIC_WORDS: dict[str, str] = {
    "complaint": "a complaint",
    "risk": "a risk to the account",
    "praise": "praise",
    "pricing": "pricing",
    "commitment": "a commitment",
    "decision": "a decision",
    "request": "a request",
    "question": "a question the customer asked",
    "competitor_mention": "a competitor",
    "contact_change": "a change of contact",
    "usage": "usage",
    "other": "an account detail",
}
ARC_WORDS: dict[str, str] = {
    "rocky_go_live": "a rocky go-live",
    "renewal_price_pushback": "a renewal with price pushback",
    "competitive_eval_save_or_churn": "a competitive evaluation",
    "champion_transition": "a champion leaving",
    "expansion": "an expansion",
    "incident_escalation_exec": "an incident escalation",
    "quiet_steady": "the account's routine",
}
CRM_FIELD_WORDS: dict[str, str] = {
    "renewal_date": "the renewal date",
    "arr": "the annual recurring revenue (ARR)",
    "renewal_stage": "the renewal stage",
    "competitors": "the competitors on record",
    "owner_previous": "the account owner and the previous owner",
    "status": "the account status",
}


class BudgetExceeded(RuntimeError):
    """The run's `v2_search_eval` cap would be crossed by the next call."""


# ---------------------------------------------------------------------
# The world, as the question builder reads it
# ---------------------------------------------------------------------


@dataclass
class SourceInfo:
    source_id: str
    plan_id: str
    kind: str
    date: str
    text: str
    true_account_id: str | None
    role: str | None = None


@dataclass
class QuestionWorld:
    root: Path
    world_id: str
    vendor_name: str
    vertical: str
    accounts: dict[str, dict[str, Any]]
    anchor_names: dict[str, list[str]]
    people: dict[str, dict[str, Any]]
    entity_names: dict[str, str]
    facts: dict[str, dict[str, Any]]
    events: dict[str, dict[str, Any]]
    arcs: list[dict[str, Any]]
    sources: dict[str, SourceInfo]
    fact_docs: dict[str, set[str]]
    fact_spans: dict[tuple[str, str], list[tuple[int, int]]]
    crm_records: list[dict[str, Any]]
    supersessions: list[dict[str, Any]]
    loaded: LoadedWorld
    # WP11's `--dry-run` worlds carry fixture text: structure only, never worded.
    dry_run_world: bool = False

    def plan_of(self, source_id: str) -> str:
        return self.sources[source_id].plan_id

    def names(self) -> list[str]:
        """The vendor's and every account, alias, product, competitor and
        person name."""
        out: set[str] = {self.vendor_name, *self.entity_names.values()}
        for a in self.accounts.values():
            out.add(a["name"])
            out.update(a.get("aliases") or [])
        return sorted(n for n in out if n)


def _iso(value: Any) -> str:
    if isinstance(value, date):
        return value.isoformat()
    return str(value or "")[:10]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _unique_anchor_names(accounts: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    """An account's full name, plus each alias no other account uses as a
    name or alias (a system and its facilities share their short name)."""
    owners: dict[str, set[str]] = defaultdict(set)
    for aid, a in accounts.items():
        for n in [a["name"], *(a.get("aliases") or [])]:
            owners[_norm(n)].add(aid)
    out: dict[str, list[str]] = {}
    for aid, a in accounts.items():
        names = [a["name"]]
        names += [n for n in a.get("aliases") or [] if owners[_norm(n)] == {aid} and n not in names]
        out[aid] = names
    return out


def load_question_world(root: Path | str) -> QuestionWorld:
    """Read an exported v2 world: `world.yaml`, `sources.jsonl`,
    `labels.jsonl` (through WP9's `load_world`, so a label whose span does
    not match its source is excluded), `world_plan.json` (CRM records and
    their links) and, when WP6 ran, `noise/noise_manifest.json`
    (supersessions, their new facts, and the CRM link lists after link
    noise)."""
    root = Path(root)
    loaded = load_world(root)
    world = yaml.safe_load((root / "world.yaml").read_text(encoding="utf-8")) or {}
    plan_path = root / "world_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8")) if plan_path.exists() else {}
    noise_path = root / "noise" / "noise_manifest.json"
    noise = json.loads(noise_path.read_text(encoding="utf-8")) if noise_path.exists() else {}
    manifest_path = root / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    )

    vendor = world.get("vendor") or {}
    accounts = {
        a["id"]: {
            "name": a["name"],
            "aliases": list(a.get("aliases") or []),
            "crm_id": a.get("crm_id"),
            "parent": a.get("parent"),
        }
        for a in world.get("accounts") or []
    }
    people = {
        p["id"]: {
            "name": p["name"],
            "title": p.get("title") or p.get("role") or "",
            "side": p.get("side"),
            "account_id": p.get("account_id"),
        }
        for p in world.get("people") or []
    }
    entity_names: dict[str, str] = {}
    for prod in vendor.get("products") or []:
        entity_names[prod["id"]] = prod["name"]
    for comp in vendor.get("competitors") or []:
        entity_names[comp["id"]] = comp["name"]
    for pid, p in people.items():
        entity_names[pid] = p["name"]

    facts: dict[str, dict[str, Any]] = {}
    for f in world.get("facts") or []:
        facts[f["id"]] = {**f, "date": _iso(f.get("date"))}
    for f in noise.get("new_facts") or []:
        facts[f["id"]] = {**f, "date": _iso(f.get("date"))}
    events = {e["id"]: {**e, "date": _iso(e.get("date"))} for e in world.get("events") or []}

    raw_sources = {
        r.get("source_id"): r for r in _read_jsonl(root / "sources.jsonl") if r.get("source_id")
    }
    sources: dict[str, SourceInfo] = {}
    for sid, src in loaded.sources.items():
        raw = raw_sources.get(sid, {})
        sources[sid] = SourceInfo(
            source_id=sid,
            plan_id=str(raw.get("plan_id") or src.plan_id or sid),
            kind=src.kind,
            date=src.date.isoformat(),
            text=src.text,
            true_account_id=raw.get("true_account_id") or src.account_id,
            role=raw.get("role"),
        )

    fact_docs: dict[str, set[str]] = defaultdict(set)
    fact_spans: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    for lab in loaded.labels:
        extra = lab.model_extra or {}
        ids = {lab.fact_id}
        for key in ("restates_fact_id", "recovers_fact_id"):
            if isinstance(extra.get(key), str) and extra[key]:
                ids.add(extra[key])
        for fid in ids:
            fact_docs[fid].add(lab.doc_id)
            fact_spans[(fid, lab.doc_id)].append((lab.char_start, lab.char_end))

    overrides = noise.get("crm_linked_overrides") or {}
    crm_records = []
    for rec in plan.get("crm_accounts") or []:
        rec = dict(rec)
        rendered = overrides.get(rec["id"]) or {}
        # A link counts only if it is true (planned from shared events) and
        # still present after WP6's link noise.
        rec["effective_linked_calls"] = [
            c
            for c in rec.get("linked_calls") or []
            if not rendered or c in (rendered.get("linked_calls") or [])
        ]
        crm_records.append(rec)

    return QuestionWorld(
        root=root,
        world_id=loaded.world_id,
        vendor_name=str(vendor.get("name") or loaded.supplier_name or ""),
        vertical=str(world.get("vertical") or vendor.get("vertical") or ""),
        accounts=accounts,
        anchor_names=_unique_anchor_names(accounts),
        people=people,
        entity_names=entity_names,
        facts=facts,
        events=events,
        arcs=list(world.get("arcs") or []),
        sources=sources,
        fact_docs=dict(fact_docs),
        fact_spans=dict(fact_spans),
        crm_records=crm_records,
        supersessions=list(noise.get("supersessions") or []),
        loaded=loaded,
        dry_run_world=bool(isinstance(manifest, dict) and manifest.get("dry_run")),
    )


# ---------------------------------------------------------------------
# Text checks (code-side, no model)
# ---------------------------------------------------------------------

_NORM_RE = re.compile(r"[^0-9a-z]+")
_NUM_RE = re.compile(r"(?<![A-Za-z0-9])\$?\d[\d,.:/-]*%?(?![A-Za-z0-9])")
_ORDINAL_RE = re.compile(r"(?<![A-Za-z0-9])(\d{1,2})(?:st|nd|rd|th)\b", re.IGNORECASE)
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_YEAR_RE = re.compile(r"^(19|20)\d\d$")
_WORD_RE = re.compile(r"[A-Za-z0-9']+")
_NUMBER_WORDS = {
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
    11: "eleven",
    12: "twelve",
    13: "thirteen",
    14: "fourteen",
    15: "fifteen",
    16: "sixteen",
    17: "seventeen",
    18: "eighteen",
    19: "nineteen",
    20: "twenty",
    30: "thirty",
    40: "forty",
    50: "fifty",
    60: "sixty",
    70: "seventy",
    80: "eighty",
    90: "ninety",
}
_STOPWORDS = frozenset(
    "a an and are as at be been but by can could did do does for from had has have how i if "
    "in into is it its me my of on or our so than that the their them then there these they "
    "this those to up us was we were what when where which who whom why will with would you "
    "your about any all also what's who's".split()
)


def _norm(text: str) -> str:
    return " ".join(_NORM_RE.sub(" ", text.lower()).split())


def _contains_phrase(haystack_norm: str, phrase: str) -> bool:
    p = _norm(phrase)
    return bool(p) and f" {p} " in f" {haystack_norm} "


def _normalize_number(token: str) -> str:
    t = token.strip().lstrip("$").rstrip("%").rstrip(".,:/-")
    if _ISO_DATE_RE.match(t):
        return t
    return t.replace(",", "")


def numeric_tokens(text: str) -> set[str]:
    """Normalized numbers a text states: `775,773` and `$775773` are one
    value, an ISO date stays whole, `9th` reads 9. Digits inside a name
    (`Meralight T3`) are not numbers."""
    out = {_normalize_number(m.group(0)) for m in _NUM_RE.finditer(text)}
    out.update(m.group(1) for m in _ORDINAL_RE.finditer(text))
    return {t for t in out if t}


def _date_phrases(iso: str) -> list[str]:
    _, m, d = (int(x) for x in iso.split("-"))
    if not 1 <= m <= 12:
        return []
    months = {calendar.month_name[m], calendar.month_abbr[m]}
    out = [iso, f"{m}/{d}", f"{m:02d}/{d:02d}"]
    for mon in months:
        out += [f"{mon} {d}", f"{d} {mon}"]
    return out


def _value_phrases(value: str) -> tuple[set[str], list[str]]:
    """(numbers, phrases) a question must not contain to keep `value` hidden."""
    nums = numeric_tokens(value)
    phrases: list[str] = []
    for n in nums:
        if _ISO_DATE_RE.match(n):
            phrases += _date_phrases(n)
    if not nums and value.strip():
        phrases.append(value.strip())
    return nums, phrases


def forbidden_for(answer_texts: Iterable[str], allowed_texts: Iterable[str]) -> tuple[list, list]:
    """Numbers and phrases in the answer that the asker does not already
    know (names, the time window, background claims). A bare year stays
    allowed: "in 2026" places a question without answering it."""
    allowed = set()
    for t in allowed_texts:
        allowed |= numeric_tokens(t)
    nums: set[str] = set()
    phrases: list[str] = []
    for text in answer_texts:
        for n in numeric_tokens(text):
            if n in allowed or _YEAR_RE.match(n):
                continue
            nums.add(n)
            if _ISO_DATE_RE.match(n):
                phrases += _date_phrases(n)
            elif n.isdigit() and int(n) in _NUMBER_WORDS:
                phrases.append(_NUMBER_WORDS[int(n)])
    return sorted(nums), sorted(set(phrases))


def _longest_common_run(a: Sequence[str], b: Sequence[str]) -> int:
    best = 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            if x == y:
                cur[j] = prev[j - 1] + 1
                best = max(best, cur[j])
        prev = cur
    return best


def _masked_words(text: str, names: Sequence[str]) -> list[str]:
    norm = f" {_norm(text)} "
    for n in sorted({_norm(x) for x in names if x}, key=len, reverse=True):
        if n:
            norm = norm.replace(f" {n} ", " <name> ")
    return norm.split()


COPY_RUN_LIMIT = 6


def content_words(text: str) -> list[str]:
    return [t for t in bm25_tokenize(text) if t not in _STOPWORDS and len(t) > 1]


def lexical_overlap(question: str, gold_texts: Iterable[str]) -> float:
    """Share of the question's content words that occur in its gold
    sources: what a keyword search can match on (1.0 = every word)."""
    words = set(content_words(question))
    if not words:
        return 0.0
    gold: set[str] = set()
    for t in gold_texts:
        gold.update(bm25_tokenize(t))
    return round(len(words & gold) / len(words), 4)


def paraphrase_level(overlap: float) -> str:
    """The measured level, next to the requested one."""
    if overlap >= 0.75:
        return "close"
    if overlap >= 0.5:
        return "moderate"
    return "heavy"


# ---------------------------------------------------------------------
# Intents: what each question asks, decided by code
# ---------------------------------------------------------------------


@dataclass
class AnswerItem:
    item_id: str
    kind: str  # claim | crm_field | must_not
    text: str | None = None
    fact_id: str | None = None
    doc_ids: list[str] = field(default_factory=list)
    values: dict[str, str] = field(default_factory=dict)
    negative: bool = False
    text_source: str = "world_claim"


@dataclass
class QuestionIntent:
    intent_id: str
    world_id: str
    kind: str
    subkind: str
    account_id: str
    account_name: str
    anchor_names: list[str]
    products: list[str]
    people: list[dict[str, str]]
    topic: str
    when: str
    answer: list[AnswerItem]
    gold_doc_ids: list[str]
    fact_ids: list[str]
    event_ids: list[str] = field(default_factory=list)
    arc_id: str | None = None
    background: list[str] = field(default_factory=list)
    latest_date: str | None = None
    superseded_doc_ids: list[str] = field(default_factory=list)
    forbidden_numbers: list[str] = field(default_factory=list)
    forbidden_phrases: list[str] = field(default_factory=list)
    paraphrase_requested: str = "moderate"

    @property
    def positive_items(self) -> list[AnswerItem]:
        return [a for a in self.answer if not a.negative]


def _stable_rng(*parts: Any) -> random.Random:
    return random.Random(hashlib.sha256(":".join(map(str, parts)).encode()).hexdigest())


def _months(dates: Iterable[str]) -> str:
    months = sorted({d[:7] for d in dates if d})
    if not months:
        return ""
    return months[0] if len(months) == 1 else f"{months[0]} to {months[-1]}"


def _fact_context(w: QuestionWorld, facts: Sequence[dict[str, Any]]) -> dict[str, Any]:
    products: list[str] = []
    people: list[dict[str, str]] = []
    for f in facts:
        for e in f.get("entities") or []:
            name = w.entity_names.get(e)
            if e.startswith("product-") and name and name not in products:
                products.append(name)
        sp = w.people.get(f.get("speaker") or "")
        if sp and all(p["name"] != sp["name"] for p in people):
            people.append({"name": sp["name"], "title": sp["title"]})
    return {"products": products, "people": people}


def _usable_fact(w: QuestionWorld, fid: str) -> dict[str, Any] | None:
    f = w.facts.get(fid)
    if not f or not f.get("claim") or f.get("account_id") not in w.accounts:
        return None
    if not w.fact_docs.get(fid):
        return None
    return f


def _make_intent(
    w: QuestionWorld,
    kind: str,
    subkind: str,
    account_id: str,
    answer: list[AnswerItem],
    facts: Sequence[dict[str, Any]],
    *,
    topic: str,
    background: Sequence[str] = (),
    arc_id: str | None = None,
    latest_date: str | None = None,
    superseded: Iterable[str] = (),
) -> QuestionIntent:
    ctx = _fact_context(w, facts)
    gold = sorted({d for a in answer if not a.negative for d in a.doc_ids})
    when = _months(f["date"] for f in facts)
    allowed = [
        *w.names(),
        *background,
        *(p["title"] for p in ctx["people"]),
        when,
    ]
    answer_texts = [a.text for a in answer if a.kind == "claim" and not a.negative and a.text]
    nums, phrases = forbidden_for(answer_texts, allowed)
    for a in answer:
        # A CRM value is the answer; an older value must not steer the asker either.
        values = list(a.values.values()) if a.kind == "crm_field" else []
        if a.kind == "must_not":
            values = [v.strip() for v in a.values.get("old", "").split(",") if v.strip()]
        for v in values:
            n2, p2 = _value_phrases(v)
            nums = sorted(set(nums) | n2)
            phrases = sorted(set(phrases) | set(p2))
    return QuestionIntent(
        intent_id="",
        world_id=w.world_id,
        kind=kind,
        subkind=subkind,
        account_id=account_id,
        account_name=w.accounts[account_id]["name"],
        anchor_names=list(w.anchor_names[account_id]),
        products=ctx["products"],
        people=ctx["people"],
        topic=topic,
        when=when,
        answer=answer,
        gold_doc_ids=gold,
        fact_ids=sorted({a.fact_id for a in answer if a.fact_id and not a.negative}),
        event_ids=sorted({f["event_id"] for f in facts if f.get("event_id")}),
        arc_id=arc_id,
        background=list(background),
        latest_date=latest_date,
        superseded_doc_ids=sorted(set(superseded) - set(gold)),
        forbidden_numbers=nums,
        forbidden_phrases=phrases,
    )


def _claim_item(item_id: str, w: QuestionWorld, fid: str) -> AnswerItem:
    return AnswerItem(
        item_id=item_id,
        kind="claim",
        text=w.facts[fid]["claim"],
        fact_id=fid,
        doc_ids=sorted(w.fact_docs[fid]),
    )


def _superseded_ids(w: QuestionWorld) -> set[str]:
    return {s["old_fact_id"] for s in w.supersessions if s.get("old_fact_id")}


def single_fact_candidates(w: QuestionWorld) -> list[QuestionIntent]:
    old = _superseded_ids(w)
    out = []
    for fid in sorted(w.fact_docs):
        f = _usable_fact(w, fid)
        if f is None or fid in old or f.get("supersedes"):
            continue
        out.append(
            _make_intent(
                w,
                "single_fact",
                "lookup",
                f["account_id"],
                [_claim_item("A1", w, fid)],
                [f],
                topic=TOPIC_WORDS.get(str(f.get("fact_type")), "an account detail"),
            )
        )
    return out


def _arc_labeled_facts(w: QuestionWorld, arc: dict[str, Any]) -> list[dict[str, Any]]:
    """The arc's facts that some exported source states, in event order."""
    old = _superseded_ids(w)
    by_event: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for fid, f in w.facts.items():
        if fid in old or f.get("supersedes"):
            continue
        if f.get("event_id") in arc.get("event_ids", []) and _usable_fact(w, fid):
            by_event[f["event_id"]].append(f)
    out = []
    for eid in arc.get("event_ids") or []:
        out += sorted(by_event.get(eid, []), key=lambda f: f["id"])
    return out


def _plans(w: QuestionWorld, fid: str) -> set[str]:
    return {w.plan_of(d) for d in w.fact_docs[fid] if d in w.sources}


def _spread_pick(w: QuestionWorld, facts: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    """Up to k facts spanning the arc (first, last, then the middle), each
    adding a planned document the earlier picks do not cover."""
    if not facts:
        return []
    middle = sorted(range(1, len(facts) - 1), key=lambda i: (abs(i - len(facts) // 2), i))
    order = list(dict.fromkeys([0, len(facts) - 1, *middle]))
    picked: list[dict[str, Any]] = []
    covered: set[str] = set()
    for i in order:
        f = facts[i]
        plans = _plans(w, f["id"])
        if not plans or plans <= covered:
            continue
        picked.append(f)
        covered |= plans
        if len(picked) == k:
            break
    return sorted(picked, key=lambda f: (f["date"], f["id"]))


def multi_doc_candidates(w: QuestionWorld) -> list[QuestionIntent]:
    out = []
    for arc in sorted(w.arcs, key=lambda a: a["id"]):
        if arc.get("account_id") not in w.accounts:
            continue
        facts = _arc_labeled_facts(w, arc)
        # A long arc gives a second question over the facts the first left out.
        for _ in range(2):
            picked = _spread_pick(w, facts, 3)
            if len(picked) < 2:
                break
            answer = [_claim_item(f"A{i}", w, f["id"]) for i, f in enumerate(picked, 1)]
            out.append(
                _make_intent(
                    w,
                    "multi_doc",
                    "arc",
                    arc["account_id"],
                    answer,
                    picked,
                    topic=ARC_WORDS.get(str(arc.get("arc_type")), "one story at the account"),
                    arc_id=arc["id"],
                )
            )
            used = {f["id"] for f in picked}
            facts = [f for f in facts if f["id"] not in used]
    return out


def _differing_values(old_claim: str, new_claim: str) -> list[str]:
    old_nums = numeric_tokens(old_claim)
    return sorted(old_nums - numeric_tokens(new_claim))


def recency_candidates(w: QuestionWorld) -> list[QuestionIntent]:
    out = []
    for s in sorted(w.supersessions, key=lambda s: s.get("new_fact_id") or ""):
        new_id, old_id = s.get("new_fact_id"), s.get("old_fact_id")
        new, old = _usable_fact(w, new_id or ""), _usable_fact(w, old_id or "")
        if new is None or old is None:
            continue
        answer = [_claim_item("A1", w, new_id)]
        old_values = _differing_values(old["claim"], new["claim"])
        if old_values:
            answer.append(
                AnswerItem(
                    item_id="N1",
                    kind="must_not",
                    fact_id=old_id,
                    values={"old": ", ".join(old_values), "old_claim": old["claim"]},
                    negative=True,
                    text_source="open_weight_model",
                )
            )
        out.append(
            _make_intent(
                w,
                "recency",
                "supersession",
                new["account_id"],
                answer,
                [new],
                topic=TOPIC_WORDS.get(str(new.get("fact_type")), "an account detail"),
                latest_date=new["date"],
                superseded=w.fact_docs[old_id],
            )
        )
    for arc in sorted(w.arcs, key=lambda a: a["id"]):
        if arc.get("account_id") not in w.accounts:
            continue
        facts = _arc_labeled_facts(w, arc)
        if len(facts) < 2:
            continue
        latest = facts[-1]
        latest_plans = _plans(w, latest["id"])
        earlier = [
            f
            for f in facts
            if f["date"] < latest["date"] and not (_plans(w, f["id"]) <= latest_plans)
        ]
        if not earlier:
            continue
        first = earlier[0]
        out.append(
            _make_intent(
                w,
                "recency",
                "arc_latest",
                arc["account_id"],
                [_claim_item("A1", w, latest["id"])],
                [first, latest],
                topic=ARC_WORDS.get(str(arc.get("arc_type")), "one story at the account"),
                background=[first["claim"]],
                arc_id=arc["id"],
                latest_date=latest["date"],
                superseded={d for f in earlier for d in w.fact_docs[f["id"]]},
            )
        )
    return out


def cross_source_candidates(w: QuestionWorld) -> list[QuestionIntent]:
    old = _superseded_ids(w)
    calls_by_plan: dict[str, list[str]] = defaultdict(list)
    for sid, s in w.sources.items():
        if s.kind == "call":
            calls_by_plan[s.plan_id].append(sid)
    out = []
    for rec in sorted(w.crm_records, key=lambda r: r["id"]):
        account_id = rec.get("account_id")
        if account_id not in w.accounts or rec["id"] not in w.sources:
            continue
        fields = [
            ff
            for ff in rec.get("field_facts") or []
            if ff["fields"][-1] in CROSS_SOURCE_FIELDS
            and rec["id"] in w.fact_docs.get(ff["id"], ())
        ]
        if not fields:
            continue
        pairs = []
        for plan_id in rec.get("effective_linked_calls") or []:
            for sid in sorted(calls_by_plan.get(plan_id, [])):
                if w.sources[sid].true_account_id != account_id:
                    continue
                for fid in sorted(w.fact_docs):
                    f = _usable_fact(w, fid)
                    if f is None or fid in old or f.get("supersedes"):
                        continue
                    if sid in w.fact_docs[fid] and f["account_id"] == account_id:
                        pairs.append((sid, f))
        for i, (_sid, f) in enumerate(pairs):
            ff = fields[i % len(fields)]
            values = {name: rec["row"].get(name, "") for name in ff["fields"]}
            if not all(values.values()):
                continue
            crm_item = AnswerItem(
                item_id="A1",
                kind="crm_field",
                fact_id=ff["id"],
                doc_ids=[rec["id"]],
                values=values,
                text_source="open_weight_model",
            )
            out.append(
                _make_intent(
                    w,
                    "cross_source",
                    ff["fields"][-1],
                    account_id,
                    [crm_item, _claim_item("A2", w, f["id"])],
                    [f],
                    topic=CRM_FIELD_WORDS[ff["fields"][-1]],
                )
            )
    return out


def _round_robin_pick(
    candidates: list[QuestionIntent], n: int, key: Callable[[QuestionIntent], str], rng
) -> list[QuestionIntent]:
    groups: dict[str, list[QuestionIntent]] = defaultdict(list)
    for c in candidates:
        groups[key(c)].append(c)
    for g in groups.values():
        rng.shuffle(g)
    picked: list[QuestionIntent] = []
    names = sorted(groups)
    while len(picked) < n and any(groups.values()):
        for name in names:
            if groups[name] and len(picked) < n:
                picked.append(groups[name].pop())
    return picked


def _spread_key(w: QuestionWorld, kind: str) -> Callable[[QuestionIntent], str]:
    """What a kind's picks are spread over: fact type for single facts, the
    sub-kind for recency and cross-source (the CRM field), accounts for arcs."""

    def key(c: QuestionIntent) -> str:
        if kind == "single_fact":
            return str(w.facts[c.fact_ids[0]].get("fact_type"))
        if kind in ("recency", "cross_source"):
            return c.subkind
        return c.account_id

    return key


def build_intents(
    w: QuestionWorld, counts: dict[str, int], *, seed: int = 0
) -> tuple[list[QuestionIntent], dict[str, int]]:
    """Choose each kind's questions: deterministic in the world and `seed`,
    spread over fact types (single facts) and accounts (the rest). Returns
    the intents and the number of candidates each kind had."""
    rng = _stable_rng("wp10", w.world_id, seed)
    pools = {
        "single_fact": single_fact_candidates(w),
        "multi_doc": multi_doc_candidates(w),
        "recency": recency_candidates(w),
        "cross_source": cross_source_candidates(w),
    }
    supply = {k: len(v) for k, v in pools.items()}
    intents: list[QuestionIntent] = []
    for kind in KINDS:
        picked = _round_robin_pick(pools[kind], counts.get(kind, 0), _spread_key(w, kind), rng)
        for i, intent in enumerate(picked, 1):
            intent.intent_id = f"{w.world_id}:{KIND_PREFIX[kind]}-{i:03d}"
            intent.paraphrase_requested = rng.choices(PARAPHRASE_LEVELS, _PARAPHRASE_WEIGHTS)[0]
            intents.append(intent)
    return intents, supply


# ---------------------------------------------------------------------
# Model calls, budget, call log
# ---------------------------------------------------------------------


@dataclass
class ModelReply:
    data: dict[str, Any] | None
    text: str
    cost_usd: float


# (model, system, user, max_tokens, json_schema, temperature) -> reply
ModelCall = Callable[..., ModelReply]

_FALLBACK_PRICE = (1.0, 3.0)
PRICE_SAFETY = 2.0  # OpenRouter's peak-hour overrides roughly double some list prices


def model_price(model: str) -> tuple[float, float]:
    """(prompt, completion) USD per million tokens, for the pre-call check
    only; the recorded cost is OpenRouter's own `usage.cost`."""
    from corpus.generator.v2.writer import ALLOWED_WRITER_MODELS
    from portolan.pricing import PRICING

    if model in ALLOWED_WRITER_MODELS:
        info = ALLOWED_WRITER_MODELS[model]
        return info.prompt_usd_per_m, info.completion_usd_per_m
    if model in PRICING:
        p = PRICING[model]
        return p.input_per_million, p.output_per_million
    return _FALLBACK_PRICE


def worst_case_usd(model: str, prompt_chars: int, max_tokens: int) -> float:
    pin, pout = model_price(model)
    prompt_tokens = prompt_chars / 3.0
    return PRICE_SAFETY * (prompt_tokens * pin + max_tokens * pout) / 1e6


def model_allowed(model: str) -> bool:
    """Open-weight and on the training-set builder's teacher allowlist."""
    from portolan.training.label_judge import judge_model_allowed
    from portolan.training.teachers import is_open_weight

    return is_open_weight(model) and judge_model_allowed(model)


class SearchEvalBudget:
    """The run's cap on `v2_search_eval`: spent (the meter's growth since
    the run began, or the recorded costs without a meter) plus every call in
    flight at its worst case must stay under `cap_usd`."""

    def __init__(self, cap_usd: float, meter: Any = None, spend_key: str = SPEND_KEY) -> None:
        self.cap_usd = cap_usd
        self.meter = meter
        self.spend_key = spend_key
        self.baseline = meter.corpus_usd(spend_key) if meter is not None else 0.0
        self._recorded = 0.0
        self._reserved = 0.0
        self._lock = threading.Lock()

    def spent(self) -> float:
        if self.meter is not None:
            return max(0.0, self.meter.corpus_usd(self.spend_key) - self.baseline)
        return self._recorded

    def reserve(self, worst: float) -> None:
        with self._lock:
            if self.spent() + self._reserved + worst > self.cap_usd:
                raise BudgetExceeded(
                    f"{self.spend_key}: spent ${self.spent():.4f} + in flight "
                    f"${self._reserved:.4f} + next call ${worst:.4f} > cap ${self.cap_usd:.2f}"
                )
            self._reserved += worst

    def release(self, worst: float, cost: float) -> None:
        with self._lock:
            self._reserved = max(0.0, self._reserved - worst)
            self._recorded += cost


class CallLog:
    """Every model call of a run, one JSON line each (append-only)."""

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self.rows: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def add(self, row: dict[str, Any]) -> None:
        with self._lock:
            self.rows.append(row)
            if self.path is not None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _metered_call(
    call: ModelCall,
    budget: SearchEvalBudget,
    log: CallLog,
    *,
    purpose: str,
    intent_id: str,
    model: str,
    system: str,
    user: str,
    max_tokens: int,
    schema: dict[str, Any] | None,
    temperature: float | None,
) -> ModelReply | None:
    worst = worst_case_usd(model, len(system) + len(user), max_tokens)
    budget.reserve(worst)
    cost = 0.0
    try:
        reply = call(
            model=model,
            system=system,
            user=user,
            max_tokens=max_tokens,
            json_schema=schema,
            temperature=temperature,
        )
        cost = reply.cost_usd
    except BudgetExceeded:
        raise
    except Exception as exc:  # a failed call is logged and counted as a failed attempt
        log.add(
            {
                "purpose": purpose,
                "intent_id": intent_id,
                "model": model,
                "ok": False,
                "error": f"{type(exc).__name__}: {str(exc)[:300]}",
                "cost_usd": 0.0,
            }
        )
        return None
    finally:
        budget.release(worst, cost)
    log.add(
        {
            "purpose": purpose,
            "intent_id": intent_id,
            "model": model,
            "ok": True,
            "cost_usd": round(cost, 8),
            "prompt_sha256": hashlib.sha256((system + "\n" + user).encode()).hexdigest(),
            "reply": reply.data if reply.data is not None else reply.text[:4000],
        }
    )
    return reply


def openrouter_model_call(
    api_key: str, meter: Any, *, max_retries: int = 3, transport: Any = None
) -> ModelCall:
    """One `OpenRouterClient` per thread, metered in `meter` under
    `v2_search_eval`; returns each call's cost from the client's sink.
    `transport` is for tests (an `httpx.MockTransport`)."""
    from portolan.llm.openrouter import OpenRouterClient
    from portolan.metering import InMemorySink

    local = threading.local()

    def call(
        *,
        model: str,
        system: str,
        user: str,
        max_tokens: int,
        json_schema: dict[str, Any] | None,
        temperature: float | None,
    ) -> ModelReply:
        client = getattr(local, "client", None)
        if client is None:
            client = OpenRouterClient(
                spend_key=SPEND_KEY,
                spend_meter=meter,
                api_key=api_key,
                default_model=model,
                sink=InMemorySink(),
                tenant="bench-v2-search-eval",
                max_retries=max_retries,
                transport=transport,
            )
            local.client = client
        before = len(client.sink.events)
        result = client.complete(
            system=system,
            user=user,
            json_schema=json_schema,
            model=model,
            surface="audit",
            max_tokens=max_tokens,
            validate_json=False,
            temperature=temperature,
        )
        cost = sum((e.cost_estimate or 0.0) for e in client.sink.events[before:])
        return ModelReply(data=result.data, text=result.text or "", cost_usd=cost)

    return call


# ---------------------------------------------------------------------
# Wording
# ---------------------------------------------------------------------

WORDING_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "question": {"type": "string"},
        "statements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "string"}, "text": {"type": "string"}},
                "required": ["id", "text"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["question", "statements"],
    "additionalProperties": False,
}

_PARAPHRASE_INSTRUCTIONS = {
    "close": "Use the plain words a person would type into a search box. You may reuse "
    "words from the facts.",
    "moderate": "Reword it: use different words than the facts use for the main action and "
    "the topic. Keep the names.",
    "heavy": "Ask it the way a colleague would in conversation, in your own words: avoid the "
    "facts' own nouns and verbs and describe things by what they are or do. Keep the "
    "account's name.",
}
_KIND_INSTRUCTIONS = {
    "single_fact": "Ask for the one fact in ANSWER.",
    "multi_doc": "The ANSWER facts are steps of one story at this account, spread over "
    "several documents. Ask one question about the story as a whole (what happened and how "
    "it developed) whose full answer needs every ANSWER fact. Do not list the facts.",
    "recency": "The asker knows the BACKGROUND and wants the latest state of this matter. "
    "Ask for the current or most recent status; the answer is the ANSWER fact.",
    "supersession": "The ANSWER is the latest version of something that changed; OLD was the "
    "earlier version. Ask for the current or latest state. Do not mention the older value.",
    "cross_source": "The asker wants two things joined: what the CRM record for the account "
    "shows (the CRM field in ANSWER) and what was said on a call with that account (the "
    "other ANSWER fact). Ask one question that needs both.",
}
WORDING_SYSTEM = """You write one question for a search test over a company's records.
The company is {vendor}, a {vertical} vendor to healthcare customers. A colleague will \
type your question into the company's search box.

Rules:
- The answer to your question is the ANSWER. The question must not contain the answer: \
no numbers, dates, amounts, names of the outcome or results that appear in the ANSWER.
- Name the customer account with one of these names: {anchors}.
- One question of one or two sentences, ending with a question mark. No preamble.
- {kind_instruction}
- Paraphrase: {paraphrase}
{statement_rules}
Reply with JSON only: {{"question": "...", "statements": [{{"id": "...", "text": "..."}}]}}"""


def _statement_rules(intent: QuestionIntent) -> str:
    lines = []
    for a in intent.answer:
        if a.kind == "crm_field":
            vals = "; ".join(f"{k} = {v}" for k, v in a.values.items())
            lines.append(
                f'- Also write statement "{a.item_id}": one plain sentence saying what the CRM '
                f"record of {intent.account_name} shows for {vals}. Name the account and copy "
                "each value exactly as given."
            )
        elif a.kind == "must_not":
            lines.append(
                f'- Also write statement "{a.item_id}": one sentence that begins "The answer '
                f'must not" and says the answer must not give the older value from OLD, which '
                f"is: {a.values.get('old', '')}. Copy that value exactly."
            )
    return "\n".join(lines) + ("\n" if lines else "")


def wording_prompt(intent: QuestionIntent, vendor: str, vertical: str) -> tuple[str, str]:
    system = WORDING_SYSTEM.format(
        vendor=vendor,
        vertical=vertical.replace("_", " ") or "healthcare technology",
        anchors=", ".join(f'"{n}"' for n in intent.anchor_names),
        kind_instruction=_KIND_INSTRUCTIONS.get(intent.subkind, _KIND_INSTRUCTIONS[intent.kind]),
        paraphrase=_PARAPHRASE_INSTRUCTIONS[intent.paraphrase_requested],
        statement_rules=_statement_rules(intent),
    )
    brief: dict[str, Any] = {
        "account": intent.account_name,
        "products": intent.products,
        "people": intent.people,
        "topic": intent.topic,
        "when": intent.when,
    }
    if intent.background:
        brief["BACKGROUND"] = intent.background
    answer = []
    for a in intent.answer:
        if a.kind == "claim":
            answer.append({"id": a.item_id, "fact": a.text})
        elif a.kind == "crm_field":
            answer.append({"id": a.item_id, "crm_record": a.values})
        elif a.kind == "must_not":
            brief["OLD"] = a.values.get("old_claim", "")
    brief["ANSWER"] = answer
    return system, json.dumps(brief, ensure_ascii=False, indent=1)


def _statement_has_value(text: str, value: str) -> bool:
    nums = numeric_tokens(value)
    if nums:
        return nums <= numeric_tokens(text)
    return _contains_phrase(_norm(text), value)


def check_wording(
    intent: QuestionIntent, question: str, statements: dict[str, str], names: Sequence[str]
) -> list[str]:
    """Code checks on a worded question; an empty list means it passes."""
    problems: list[str] = []
    q = question.strip()
    words = _WORD_RE.findall(q)
    if not q.endswith("?"):
        problems.append("no_question_mark")
    if not 5 <= len(words) <= 60:
        problems.append("length")
    qn = _norm(q)
    if not any(_contains_phrase(qn, n) for n in intent.anchor_names):
        problems.append("account_not_named")
    leaked = sorted(set(intent.forbidden_numbers) & numeric_tokens(q))
    leaked += [p for p in intent.forbidden_phrases if _contains_phrase(qn, p)]
    if leaked:
        problems.append("answer_value_in_question:" + ",".join(leaked[:5]))
    qwords = _masked_words(q, names)
    for a in intent.positive_items:
        if a.kind == "claim" and a.text:
            if _longest_common_run(qwords, _masked_words(a.text, names)) >= COPY_RUN_LIMIT:
                problems.append(f"copies_answer:{a.item_id}")
    for a in intent.answer:
        if a.kind not in ("crm_field", "must_not"):
            continue
        text = (statements.get(a.item_id) or "").strip()
        if not text:
            problems.append(f"statement_missing:{a.item_id}")
            continue
        if len(_WORD_RE.findall(text)) > 45:
            problems.append(f"statement_length:{a.item_id}")
        if a.kind == "crm_field":
            if not all(_statement_has_value(text, v) for v in a.values.values()):
                problems.append(f"statement_value:{a.item_id}")
            if not any(_contains_phrase(_norm(text), n) for n in intent.anchor_names):
                problems.append(f"statement_account:{a.item_id}")
        else:
            old = [v.strip() for v in a.values.get("old", "").split(",") if v.strip()]
            if not all(_statement_has_value(text, v) for v in old):
                problems.append(f"statement_value:{a.item_id}")
            if " not " not in f" {_norm(text)} ":
                problems.append(f"statement_not_negative:{a.item_id}")
    return problems


def _parse_wording(reply: ModelReply) -> tuple[str, dict[str, str]]:
    data = reply.data if isinstance(reply.data, dict) else extract_json_object(reply.text or "")
    if not isinstance(data, dict):
        return "", {}
    question = data.get("question")
    statements: dict[str, str] = {}
    for s in data.get("statements") or []:
        if isinstance(s, dict) and isinstance(s.get("id"), str) and isinstance(s.get("text"), str):
            statements[s["id"].strip()] = s["text"].strip()
    return (question.strip() if isinstance(question, str) else ""), statements


# ---------------------------------------------------------------------
# Validation: BM25 top 50, then one blind judge
# ---------------------------------------------------------------------

JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "fact": {"type": "integer"},
                    "doc_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["fact", "doc_ids"],
                "additionalProperties": False,
            },
        },
        "documents": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "doc_id": {"type": "string"},
                    "answers": {"type": "string", "enum": ["full", "partial", "none"]},
                    "conflict": {"type": "string", "enum": ["yes", "no"]},
                },
                "required": ["doc_id", "answers", "conflict"],
                "additionalProperties": False,
            },
        },
        "question_clear": {"type": "string", "enum": ["yes", "no"]},
        "reason": {"type": "string"},
    },
    "required": ["facts", "documents", "question_clear", "reason"],
    "additionalProperties": False,
}

JUDGE_SYSTEM = """You check one question from a search test against documents from a \
company's records (call transcripts, emails, support tickets, chat, CRM rows, notes). The \
company is {vendor}. Answer in JSON only.

1. For each numbered fact, list the ids of the documents that state it: the same meaning, \
even in other words; a misspelled or misheard name still counts. A document that only \
mentions the topic does not state the fact.
2. For each document, say whether it answers the question: "full" (it alone answers it), \
"partial" (it gives part of the answer) or "none". Say "conflict": "yes" when the document \
gives an answer to the question that disagrees with the facts.
3. Say "question_clear": "yes" when a colleague reading the question would know which \
account and which matter it asks about, else "no".
{recency}
Reply: {{"facts": [{{"fact": 1, "doc_ids": ["..."]}}], "documents": [{{"doc_id": "...", \
"answers": "full|partial|none", "conflict": "yes|no"}}], "question_clear": "yes|no", \
"reason": "one sentence"}}"""

_RECENCY_NOTE = (
    "The question asks for the latest state. A document dated earlier may give an earlier "
    "state; mark it conflict yes if it disagrees with the facts.\n"
)


def world_bm25_index(w: QuestionWorld) -> Bm25Index:
    index = Bm25Index()
    for sid in sorted(w.sources):
        index.add(sid, w.sources[sid].text)
    index.finalize()
    return index


def _excerpt(
    text: str, limit: int, keep: Sequence[tuple[int, int]] = (), terms: frozenset[str] = frozenset()
) -> str:
    """The whole text when it fits; else the windows holding `keep` spans,
    then the windows richest in `terms`, in document order."""
    if len(text) <= limit:
        return text
    size = 1500
    windows = [(i, min(len(text), i + size)) for i in range(0, len(text), size)]

    def score(win: tuple[int, int]) -> tuple[int, int]:
        must = any(s < win[1] and e > win[0] for s, e in keep)
        hits = sum(1 for t in bm25_tokenize(text[win[0] : win[1]]) if t in terms)
        return (1 if must else 0, hits)

    ranked = sorted(windows, key=lambda win: (score(win), -win[0]), reverse=True)
    chosen: list[tuple[int, int]] = []
    total = 0
    for win in ranked:
        if total + (win[1] - win[0]) > limit and chosen:
            continue
        chosen.append(win)
        total += win[1] - win[0]
    chosen.sort()
    return "\n[...]\n".join(text[s:e] for s, e in chosen)


def judge_prompt(
    w: QuestionWorld,
    intent: QuestionIntent,
    question: str,
    answer_facts: list[str],
    shown: list[str],
    *,
    gold_chars: int,
    extra_chars: int,
) -> tuple[str, str]:
    terms = frozenset(content_words(question))
    docs = []
    for sid in shown:
        src = w.sources[sid]
        if sid in intent.gold_doc_ids:
            keep = [sp for fid in intent.fact_ids for sp in w.fact_spans.get((fid, sid), [])]
            text = _excerpt(src.text, gold_chars, keep, terms)
        else:
            text = _excerpt(src.text, extra_chars, (), terms)
        docs.append({"doc_id": sid, "date": src.date, "kind": src.kind, "text": text})
    system = JUDGE_SYSTEM.format(
        vendor=w.vendor_name, recency=_RECENCY_NOTE if intent.kind == "recency" else ""
    )
    user = json.dumps(
        {
            "question": question,
            "facts": [{"fact": i, "text": t} for i, t in enumerate(answer_facts, 1)],
            "documents": docs,
        },
        ensure_ascii=False,
        indent=1,
    )
    return system, user


@dataclass
class JudgeOutcome:
    keep: bool
    reason: str | None
    valid_doc_ids: list[str] = field(default_factory=list)
    superseded_doc_ids: list[str] = field(default_factory=list)
    conflicting_doc_ids: list[str] = field(default_factory=list)
    gold_unconfirmed_doc_ids: list[str] = field(default_factory=list)
    fact_support: dict[str, list[str]] = field(default_factory=dict)
    judge_reason: str = ""


def judge_decision(
    intent: QuestionIntent,
    data: dict[str, Any] | None,
    shown: Sequence[str],
    source_dates: dict[str, str],
    n_facts: int,
) -> JudgeOutcome:
    """Code's verdict from the judge's answers: keep only when every answer
    fact is stated by a gold source, the question is clear, and no other
    source gives a different answer (an older source in a recency question
    is superseded, not a conflict)."""
    if not isinstance(data, dict) or not isinstance(data.get("documents"), list):
        return JudgeOutcome(False, "judge_no_verdict")
    shown_set = set(shown)
    gold = set(intent.gold_doc_ids)
    support: dict[int, set[str]] = defaultdict(set)
    for row in data.get("facts") or []:
        if isinstance(row, dict) and isinstance(row.get("fact"), int):
            ids = row.get("doc_ids") or []
            support[row["fact"]] |= {d for d in ids if isinstance(d, str) and d in shown_set}
    verdicts: dict[str, tuple[str, str]] = {}
    for row in data["documents"]:
        if isinstance(row, dict) and row.get("doc_id") in shown_set:
            verdicts[row["doc_id"]] = (str(row.get("answers")), str(row.get("conflict")))
    fact_support = {str(i): sorted(support.get(i, set())) for i in range(1, n_facts + 1)}
    unsupported = [i for i in range(1, n_facts + 1) if not (support.get(i, set()) & gold)]
    supporting = set().union(*support.values()) if support else set()
    valid, superseded, conflicts = [], [], []
    for sid in shown:
        if sid in gold:
            continue
        answers, conflict = verdicts.get(sid, ("none", "no"))
        if conflict == "yes":
            older = intent.latest_date and source_dates.get(sid, "") < intent.latest_date
            if intent.kind == "recency" and (older or sid in intent.superseded_doc_ids):
                superseded.append(sid)
            else:
                conflicts.append(sid)
        elif answers in ("full", "partial") or sid in supporting:
            valid.append(sid)
    unconfirmed = sorted(
        g
        for g in gold & shown_set
        if g not in supporting and verdicts.get(g, ("none", "no"))[0] == "none"
    )
    reason = None
    if str(data.get("question_clear")) != "yes":
        reason = "judge_question_unclear"
    elif unsupported:
        reason = "judge_fact_not_in_gold:" + ",".join(map(str, unsupported))
    elif conflicts:
        reason = "judge_conflicting_document"
    return JudgeOutcome(
        keep=reason is None,
        reason=reason,
        valid_doc_ids=sorted(valid),
        superseded_doc_ids=sorted(superseded),
        conflicting_doc_ids=sorted(conflicts),
        gold_unconfirmed_doc_ids=unconfirmed,
        fact_support=fact_support,
        judge_reason=str(data.get("reason") or "")[:500],
    )


# ---------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------


@dataclass
class BuildConfig:
    counts: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_COUNTS))
    seed: int = 0
    wording_models: tuple[str, ...] = DEFAULT_WORDING_MODELS
    judge_model: str = DEFAULT_JUDGE_MODEL
    bm25_depth: int = BM25_DEPTH
    extras_depth: int = EXTRAS_DEPTH
    wording_attempts: int = 2
    judge_attempts: int = 2
    wording_max_tokens: int = 500
    judge_max_tokens: int = 1500
    wording_temperature: float = 0.7
    gold_doc_chars: int = 12_000
    extra_doc_chars: int = 5_000
    concurrency: int = 4
    dry_run: bool = False


@dataclass
class BuildResult:
    world_id: str
    intents: list[QuestionIntent]
    supply: dict[str, int]
    questions: list[dict[str, Any]]
    dropped: list[dict[str, Any]]
    spent_usd: float
    budget_stopped: bool
    config: BuildConfig

    def report(self) -> dict[str, Any]:
        by_kind: dict[str, dict[str, int]] = {}
        for kind in KINDS:
            planned = sum(1 for i in self.intents if i.kind == kind)
            kept = sum(1 for q in self.questions if q["kind"] == kind)
            by_kind[kind] = {
                "candidates": self.supply.get(kind, 0),
                "planned": planned,
                "kept": kept,
                "dropped": planned - kept,
            }
        reasons: dict[str, int] = defaultdict(int)
        stages: dict[str, int] = defaultdict(int)
        for d in self.dropped:
            reasons[d["reason"].split(":")[0]] += 1
            stages[d["stage"]] += 1
        planned = len(self.intents)
        levels: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for q in self.questions:
            levels[q["paraphrase_requested"]][q["paraphrase_level"]] += 1
        return {
            "generator_version": GENERATOR_VERSION,
            "world_id": self.world_id,
            "dry_run": self.config.dry_run,
            "planned": planned,
            "kept": len(self.questions),
            "drop_rate": round((planned - len(self.questions)) / planned, 4) if planned else 0.0,
            "by_kind": by_kind,
            "drops_by_stage": dict(sorted(stages.items())),
            "drops_by_reason": dict(sorted(reasons.items())),
            "paraphrase_requested_vs_measured": {k: dict(v) for k, v in sorted(levels.items())},
            "spent_usd": round(self.spent_usd, 6),
            "budget_stopped": self.budget_stopped,
            "wording_models": list(self.config.wording_models),
            "judge_model": self.config.judge_model,
            "bm25_depth": self.config.bm25_depth,
            "extras_depth": self.config.extras_depth,
        }


def _drop(intent: QuestionIntent, stage: str, reason: str, **extra: Any) -> dict[str, Any]:
    return {
        "intent_id": intent.intent_id,
        "kind": intent.kind,
        "subkind": intent.subkind,
        "stage": stage,
        "reason": reason,
        **extra,
    }


def _word_one(
    w: QuestionWorld,
    intent: QuestionIntent,
    cfg: BuildConfig,
    call: ModelCall,
    budget: SearchEvalBudget,
    log: CallLog,
) -> tuple[str, dict[str, str], str, list[dict[str, Any]]] | dict[str, Any]:
    names = w.names()
    start = int(hashlib.sha256(intent.intent_id.encode()).hexdigest(), 16)
    attempts: list[dict[str, Any]] = []
    system, user = wording_prompt(intent, w.vendor_name, w.vertical)
    for n in range(cfg.wording_attempts):
        model = cfg.wording_models[(start + n) % len(cfg.wording_models)]
        reply = _metered_call(
            call,
            budget,
            log,
            purpose="wording",
            intent_id=intent.intent_id,
            model=model,
            system=system,
            user=user,
            max_tokens=cfg.wording_max_tokens,
            schema=WORDING_SCHEMA,
            temperature=cfg.wording_temperature,
        )
        if reply is None:
            attempts.append({"model": model, "problems": ["call_failed"]})
            continue
        question, statements = _parse_wording(reply)
        problems = (
            ["no_question"] if not question else check_wording(intent, question, statements, names)
        )
        attempts.append({"model": model, "question": question, "problems": problems})
        if not problems:
            return question, statements, model, attempts
    last = attempts[-1]["problems"] if attempts else ["no_attempt"]
    return _drop(intent, "wording", "wording_failed:" + ";".join(last), attempts=attempts)


def _answer_facts(
    intent: QuestionIntent, statements: dict[str, str]
) -> list[tuple[AnswerItem, str]]:
    out = []
    for a in intent.answer:
        text = a.text if a.kind == "claim" else statements.get(a.item_id, "")
        out.append((a, text or ""))
    return out


def validate_question(
    w: QuestionWorld,
    index: Bm25Index,
    intent: QuestionIntent,
    question: str,
    statements: dict[str, str],
    wording_model: str,
    wording_attempts: list[dict[str, Any]],
    cfg: BuildConfig,
    call: ModelCall,
    budget: SearchEvalBudget,
    log: CallLog,
) -> dict[str, Any]:
    """The kept question row, or a drop row (`"stage"` present)."""
    ranked = index.search(question, cfg.bm25_depth)
    gold = set(intent.gold_doc_ids)
    rank = next((i for i, d in enumerate(ranked, 1) if d in gold), None)
    if rank is None:
        return _drop(
            intent,
            "bm25",
            f"no_gold_in_bm25_top_{cfg.bm25_depth}",
            question=question,
            wording_model=wording_model,
        )
    items = _answer_facts(intent, statements)
    positive = [(a, t) for a, t in items if not a.negative]
    extras = [d for d in ranked if d not in gold][: cfg.extras_depth]
    shown = sorted(gold | set(extras))
    _stable_rng("judge-order", intent.intent_id).shuffle(shown)
    system, user = judge_prompt(
        w,
        intent,
        question,
        [t for _a, t in positive],
        shown,
        gold_chars=cfg.gold_doc_chars,
        extra_chars=cfg.extra_doc_chars,
    )
    outcome = JudgeOutcome(False, "judge_no_verdict")
    for _ in range(cfg.judge_attempts):
        reply = _metered_call(
            call,
            budget,
            log,
            purpose="judge",
            intent_id=intent.intent_id,
            model=cfg.judge_model,
            system=system,
            user=user,
            max_tokens=cfg.judge_max_tokens,
            schema=JUDGE_SCHEMA,
            temperature=0.0,
        )
        if reply is None:
            continue
        data = reply.data if isinstance(reply.data, dict) else extract_json_object(reply.text)
        dates = {sid: w.sources[sid].date for sid in shown}
        outcome = judge_decision(intent, data, shown, dates, len(positive))
        if outcome.reason != "judge_no_verdict":
            break
    if not outcome.keep:
        return _drop(
            intent,
            "judge",
            outcome.reason or "judge_rejected",
            question=question,
            wording_model=wording_model,
            judge=asdict(outcome),
        )
    overlap = lexical_overlap(question, [w.sources[d].text for d in intent.gold_doc_ids])
    answer_facts = [t for _a, t in items]
    return {
        "question_id": intent.intent_id,
        "world_id": intent.world_id,
        "kind": intent.kind,
        "subkind": intent.subkind,
        "question": question,
        "expected_doc_ids": list(intent.gold_doc_ids),
        "valid_doc_ids": outcome.valid_doc_ids,
        "superseded_doc_ids": sorted(
            set(intent.superseded_doc_ids) | set(outcome.superseded_doc_ids)
        ),
        "answer_facts": answer_facts,
        "answer_fact_kinds": [a.kind for a, _t in items],
        "answer_fact_sources": [a.text_source for a, _t in items],
        "answer_fact_doc_ids": [list(a.doc_ids) for a, _t in items],
        "gold_answer": " ".join(t for a, t in items if not a.negative),
        "fact_ids": intent.fact_ids,
        "event_ids": intent.event_ids,
        "arc_id": intent.arc_id,
        "account_id": intent.account_id,
        "latest_date": intent.latest_date,
        "paraphrase_requested": intent.paraphrase_requested,
        "lexical_overlap": overlap,
        "paraphrase_level": paraphrase_level(overlap),
        "bm25_first_gold_rank": rank,
        "judge_fact_support": outcome.fact_support,
        "gold_unconfirmed_doc_ids": outcome.gold_unconfirmed_doc_ids,
        "judge_reason": outcome.judge_reason,
        "wording_model": wording_model,
        "wording_attempts": len(wording_attempts),
        "judge_model": cfg.judge_model,
        "generator_version": GENERATOR_VERSION,
        "dry_run": cfg.dry_run,
    }


def build_questions(
    w: QuestionWorld,
    cfg: BuildConfig,
    *,
    call: ModelCall,
    budget: SearchEvalBudget,
    log: CallLog,
    intents: list[QuestionIntent] | None = None,
    supply: dict[str, int] | None = None,
) -> BuildResult:
    """Word, then validate, every intent. A call the budget refuses stops
    the run: the intents not yet done are dropped as `budget_exhausted`."""
    if not cfg.dry_run and w.dry_run_world:
        raise ValueError(f"{w.root} is a dry-run world (fixture text): no paid wording")
    for m in (*cfg.wording_models, cfg.judge_model):
        if not cfg.dry_run and not model_allowed(m):
            raise ValueError(f"{m} is not an open-weight model on the teacher allowlist")
    if intents is None:
        intents, supply = build_intents(w, cfg.counts, seed=cfg.seed)
    index = world_bm25_index(w)
    stop = threading.Event()

    def one(intent: QuestionIntent) -> dict[str, Any]:
        if stop.is_set():
            return _drop(intent, "budget", "budget_exhausted")
        try:
            worded = _word_one(w, intent, cfg, call, budget, log)
            if isinstance(worded, dict):
                return worded
            question, statements, model, attempts = worded
            return validate_question(
                w, index, intent, question, statements, model, attempts, cfg, call, budget, log
            )
        except BudgetExceeded as exc:
            stop.set()
            return _drop(intent, "budget", "budget_exhausted", detail=str(exc))

    with ThreadPoolExecutor(max_workers=max(1, cfg.concurrency)) as pool:
        rows = list(pool.map(one, intents))
    questions = [r for r in rows if "stage" not in r]
    dropped = [r for r in rows if "stage" in r]
    return BuildResult(
        world_id=w.world_id,
        intents=intents,
        supply=supply or {},
        questions=questions,
        dropped=dropped,
        spent_usd=budget.spent(),
        budget_stopped=stop.is_set(),
        config=cfg,
    )


def estimate_run_cost(w: QuestionWorld, intents: list[QuestionIntent], cfg: BuildConfig) -> dict:
    """Expected cost (one wording call, a 150-token reply; one judge call, a
    600-token reply) and ceiling (every attempt at full `max_tokens`, twice
    the list price), from the prompts this run would send."""
    expected = ceiling = 0.0
    wording_models = cfg.wording_models or DEFAULT_WORDING_MODELS
    for intent in intents:
        system, user = wording_prompt(intent, w.vendor_name, w.vertical)
        chars = len(system) + len(user)
        pin, pout = model_price(wording_models[0])
        expected += (chars / 4 * pin + 150 * pout) / 1e6
        ceiling += cfg.wording_attempts * max(
            worst_case_usd(m, chars, cfg.wording_max_tokens) for m in wording_models
        )
        gold_chars = sum(
            min(len(w.sources[d].text), cfg.gold_doc_chars)
            for d in intent.gold_doc_ids
            if d in w.sources
        )
        judge_chars = 2500 + gold_chars + cfg.extras_depth * cfg.extra_doc_chars
        pin, pout = model_price(cfg.judge_model)
        expected += (judge_chars / 4 * pin + 600 * pout) / 1e6
        ceiling += cfg.judge_attempts * worst_case_usd(
            cfg.judge_model, judge_chars, cfg.judge_max_tokens
        )
    return {
        "questions": len(intents),
        "expected_usd": round(expected, 4),
        "ceiling_usd": round(ceiling, 4),
        "calls_expected": 2 * len(intents),
    }


def write_build_outputs(out: Path, result: BuildResult, estimate: dict | None = None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    if result.config.dry_run:
        (out / "DRY_RUN").write_text(
            "Fixture wording and judge: nothing here is a real question.\n", encoding="utf-8"
        )
    with (out / "intents.jsonl").open("w", encoding="utf-8") as fh:
        for i in result.intents:
            fh.write(json.dumps(asdict(i), ensure_ascii=False) + "\n")
    with (out / "questions.jsonl").open("w", encoding="utf-8") as fh:
        for q in sorted(result.questions, key=lambda q: q["question_id"]):
            fh.write(json.dumps(q, ensure_ascii=False) + "\n")
    with (out / "dropped.jsonl").open("w", encoding="utf-8") as fh:
        for d in sorted(result.dropped, key=lambda d: d["intent_id"]):
            fh.write(json.dumps(d, ensure_ascii=False, default=str) + "\n")
    report = result.report()
    if estimate is not None:
        report["estimate"] = estimate
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------
# Dry-run fixtures (no network; never real questions)
# ---------------------------------------------------------------------


def fixture_wording_call(w: QuestionWorld) -> ModelCall:
    """A stand-in wording model for `--dry-run` and tests: the account name
    plus the digit-free words of the first gold span, and each statement
    built from the values it must carry."""

    def call(*, model, system, user, max_tokens, json_schema, temperature) -> ModelReply:
        del model, max_tokens, json_schema, temperature
        brief = json.loads(user)
        account = brief["account"]
        span_words: list[str] = []
        for a in brief["ANSWER"]:
            fid = next(
                (
                    f
                    for f, meta in w.facts.items()
                    if a.get("fact") and meta.get("claim") == a.get("fact")
                ),
                None,
            )
            if fid:
                doc = sorted(w.fact_docs.get(fid, ()))[:1]
                for d in doc:
                    s, e = w.fact_spans[(fid, d)][0]
                    span_words = [
                        t for t in w.sources[d].text[s:e].split() if not any(c.isdigit() for c in t)
                    ][:8]
                break
        question = f"{account} {' '.join(span_words)}?".replace("  ", " ")
        statements = []
        for sid in re.findall(r'statement "([A-Z]\d)"', system):
            item = next((a for a in brief["ANSWER"] if a["id"] == sid), None)
            if item and "crm_record" in item:
                vals = ", ".join(f"{k} {v}" for k, v in item["crm_record"].items())
                statements.append({"id": sid, "text": f"{account} CRM {vals}."})
            else:
                old = re.search(r"which is: (.*?)\. Copy", system)
                statements.append({"id": sid, "text": f"The answer must not give {old.group(1)}."})
        data = {"question": question, "statements": statements}
        return ModelReply(data=data, text=json.dumps(data), cost_usd=0.0)

    return call


def fixture_judge_call(w: QuestionWorld) -> ModelCall:
    """A stand-in judge: each fact is stated where the labels put it."""

    def call(*, model, system, user, max_tokens, json_schema, temperature) -> ModelReply:
        del model, system, max_tokens, json_schema, temperature
        payload = json.loads(user)
        shown = [d["doc_id"] for d in payload["documents"]]
        facts = []
        for f in payload["facts"]:
            fid = next((k for k, v in w.facts.items() if v.get("claim") == f["text"]), None)
            docs = sorted(w.fact_docs.get(fid, set()) & set(shown)) if fid else []
            if not docs:
                crm = [d for d in shown if d.startswith("S-crm-")]
                docs = crm[:1]
            facts.append({"fact": f["fact"], "doc_ids": docs})
        supporting = {d for f in facts for d in f["doc_ids"]}
        documents = [
            {"doc_id": d, "answers": "partial" if d in supporting else "none", "conflict": "no"}
            for d in shown
        ]
        data = {"facts": facts, "documents": documents, "question_clear": "yes", "reason": "ok"}
        return ModelReply(data=data, text=json.dumps(data), cost_usd=0.0)

    return call


def dispatch_call(wording: ModelCall, judge: ModelCall) -> ModelCall:
    """Route by schema: the wording schema to `wording`, the rest to `judge`."""

    def call(**kwargs: Any) -> ModelReply:
        target = wording if kwargs.get("json_schema") is WORDING_SCHEMA else judge
        return target(**kwargs)

    return call
