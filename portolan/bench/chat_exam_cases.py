"""The chat exam's cases: GTM questions with checkable gold claims, a fixed train/test split,
and the lock on the test half (evaluation only, never training data;
01-spec/chat-exam-2026-09-29.md).

Sources, in the order preferred:

1. **World** (`chat_exam_graphs.WORLD_SPEC`, a held-out v2 world): questions from code
   templates over the world's own state; every gold claim is a planted fact's claim text or
   a CRM row rendered by `graph_world`, found on the seeded graph by exact text.
2. **Demo** (the Halvard demo graph): code templates over `corpus/truth-table.yaml` (gold is
   the truth table's claim text), plus the 20 multi-hop questions of
   `portolan.bench.agentic_questions` (hand-written from the graph, 2026-09-23).
3. **Hand-anchored**: CRM-versus-call cases on the demo graph, each gold claim a truth fact,
   and abstain questions whose absence is checked against the graph in code.

The ten demo questions (`scripts/demo_questions.txt`) were tuned against by reading their
failures, so they are `smoke` only and never scored as exam cases.

A case whose question or gold is already readable elsewhere in the repo (`public`: the
multi-hop set, and demo cases whose gold sits on one of its gold notes) is pinned to train.
No two cases share a gold fact (`_Picker`). The rest is split at random, stratified by type,
with `SPLIT_SEED`. Test cases live only in
`LOCKED_TEST_PATH`, gzip-compressed, a path `.claude/hooks/no-secrets.py` refuses to agents
working in a worktree; `load_split("test")` is for the runner, in-process.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import gzip
import json
import random
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import psycopg

__all__ = [
    "CASE_TYPES",
    "EXAM_DIR",
    "LOCKED_TEST_PATH",
    "SPLIT_SEED",
    "ClaimIndex",
    "ExamCase",
    "GoldClaim",
    "build_all_cases",
    "demo_cases",
    "load_split",
    "smoke_cases",
    "split_cases",
    "world_cases",
    "write_exam",
]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAM_DIR = REPO_ROOT / "evals" / "chat-exam"
# Guarded by `.claude/hooks/no-secrets.py` (`_LOCKED`): no Read, Grep or shell read from a
# worktree. `tests/test_no_secrets_hook.py` pins the refusal.
LOCKED_TEST_PATH = REPO_ROOT / "evals" / "locked" / "chat-exam-test.json.gz"
DEMO_QUESTIONS_PATH = REPO_ROOT / "scripts" / "demo_questions.txt"
TRUTH_PATH = REPO_ROOT / "corpus" / "truth-table.yaml"

CASE_TYPES = (
    "single_fact",
    "multi_doc",
    "recency",
    "cross_account",
    "abstain",
    "crm_conflict",
)
SPLIT_SEED = 20260929
# Share of the non-public cases that goes to test; the public ones are all train.
TEST_FRACTION = 0.62

# How many cases of each type each generator keeps (the pools are larger).
WORLD_TARGETS = {
    "single_fact": 9,
    "multi_doc": 5,
    "recency": 6,
    "cross_account": 7,
    "abstain": 7,
    "crm_conflict": 7,
}
DEMO_TARGETS = {"single_fact": 4, "recency": 2, "cross_account": 2}


@dataclass(frozen=True)
class GoldClaim:
    """One fact the answer must state, and where the graph holds it."""

    text: str
    notes: tuple[str, ...]
    claim_ids: tuple[int, ...] = ()
    ref: str = ""  # the world or truth-table fact id, when there is one

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "notes": list(self.notes),
            "claim_ids": list(self.claim_ids),
            "ref": self.ref,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GoldClaim:
        return cls(d["text"], tuple(d["notes"]), tuple(d.get("claim_ids", ())), d.get("ref", ""))


@dataclass(frozen=True)
class ExamCase:
    """`gold` is empty for an abstain case: the right outcome is no cited sentence."""

    id: str
    graph: str
    type: str
    question: str
    gold: tuple[GoldClaim, ...]
    origin: str
    public: bool = False
    # Why the case is left out of the exam (its gold is incomplete); kept through id
    # assignment and the split so every other case keeps its id and its half.
    dropped: str = ""

    @property
    def expects_abstain(self) -> bool:
        return self.type == "abstain"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "graph": self.graph,
            "type": self.type,
            "question": self.question,
            "gold": [g.to_dict() for g in self.gold],
            "origin": self.origin,
            "public": self.public,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExamCase:
        return cls(
            id=d["id"],
            graph=d["graph"],
            type=d["type"],
            question=d["question"],
            gold=tuple(GoldClaim.from_dict(g) for g in d["gold"]),
            origin=d["origin"],
            public=bool(d.get("public", False)),
        )


# ---------------------------------------------------------------------------
# The graph's claims, looked up by the fact text they were seeded from
# ---------------------------------------------------------------------------


def _base_text(claim_text: str) -> str:
    """A seeded claim's text without the ` ([[entity]] ...)` suffix the merge adds."""
    return claim_text.split(" ([[")[0].strip()


@dataclass
class ClaimIndex:
    by_text: dict[str, list[tuple[int, str, dt.date]]] = field(default_factory=dict)
    all_text: str = ""  # every note title and claim, lower-cased, for absence checks

    @classmethod
    def load(cls, conn: psycopg.Connection) -> ClaimIndex:
        by_text: dict[str, list[tuple[int, str, dt.date]]] = defaultdict(list)
        with conn.cursor() as cur:
            cur.execute("SELECT id, note_slug, claim_text, recorded::date FROM claims ORDER BY id")
            rows = cur.fetchall()
            cur.execute("SELECT title, coalesce(body_md, '') FROM notes")
            notes = cur.fetchall()
        for claim_id, note, text, recorded in rows:
            by_text[_base_text(text)].append((claim_id, note, recorded))
        blob = "\n".join([r[2] for r in rows] + [f"{t}\n{b}" for t, b in notes]).lower()
        return cls(dict(by_text), blob)

    def gold(self, text: str, ref: str = "") -> GoldClaim | None:
        hits = self.by_text.get(text.strip())
        if not hits:
            return None
        return GoldClaim(
            text=text.strip(),
            notes=tuple(sorted({note for _, note, _ in hits})),
            claim_ids=tuple(sorted(cid for cid, _, _ in hits)),
            ref=ref,
        )

    def recorded(self, text: str) -> set[dt.date]:
        """The dates the graph shows for this claim text: when it was said, which is what a
        question about a month or "since" means."""
        return {when for _, _, when in self.by_text.get(text.strip(), [])}

    def absent(self, *phrases: str) -> bool:
        return all(p.lower() not in self.all_text for p in phrases)


def _poss(name: str) -> str:
    return f"{name}'" if name.endswith("s") else f"{name}'s"


def _month(d: dt.date) -> str:
    return d.strftime("%B %Y")


class _Picker:
    """Draws candidate cases at random, never two sharing a gold fact: a fact one case's
    gold reveals is never another case's answer, so train gold cannot leak test gold."""

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.used: set[str] = set()

    def _fits(self, case: ExamCase) -> bool:
        refs = {g.ref or g.text for g in case.gold}
        if refs & self.used:
            return False
        self.used |= refs
        return True

    def take(self, pool: list[ExamCase], n: int) -> list[ExamCase]:
        out: list[ExamCase] = []
        for case in self.rng.sample(pool, len(pool)):
            if len(out) < n and self._fits(case):
                out.append(case)
        return out

    def round_robin(self, groups: dict[str, list[ExamCase]], n: int) -> list[ExamCase]:
        """Up to `n` cases, one from each group in turn (groups and cases shuffled)."""
        queues = {k: self.rng.sample(v, len(v)) for k, v in sorted(groups.items()) if v}
        order = sorted(queues)
        self.rng.shuffle(order)
        out: list[ExamCase] = []
        while len(out) < n and any(queues.values()):
            for key in order:
                while queues[key] and len(out) < n:
                    case = queues[key].pop()
                    if self._fits(case):
                        out.append(case)
                        break
        return out


# ---------------------------------------------------------------------------
# Question templates shared by both graphs (code, never a model's wording)
# ---------------------------------------------------------------------------

# fact type -> (template naming the product, template without it)
_SINGLE = {
    "complaint": (
        "What did {account} complain about regarding {product} in {month}?",
        "What did {account} complain about in {month}?",
    ),
    "request": (
        "What did {account} ask us for regarding {product} in {month}?",
        "What did {account} ask us for in {month}?",
    ),
    "risk": (
        "What risk came up at {account} around {product} in {month}?",
        "What risk came up at {account} in {month}?",
    ),
    "praise": (
        "What did {account} praise about {product} in {month}?",
        "What did {account} praise in {month}?",
    ),
    "commitment": (
        "What commitment was made to {account} about {product} in {month}?",
        "What commitment was made to or by {account} in {month}?",
    ),
    "pricing": (
        "What pricing issue did {account} raise about {product} in {month}?",
        "What pricing issue did {account} raise in {month}?",
    ),
    "usage": (
        "What usage update did {account} give on {product} in {month}?",
        "What usage update did {account} give in {month}?",
    ),
    "decision": (
        "What did {account} decide about {product} in {month}?",
        "What did {account} decide in {month}?",
    ),
    "question": (
        "What question did {account} ask about {product} in {month}?",
        "What question did {account} ask in {month}?",
    ),
    "competitor_mention": (
        "Which competitor came up at {account} in {month}, and what was said?",
        "Which competitor came up at {account} in {month}, and what was said?",
    ),
    "contact_change": (
        "What contact change happened at {account} in {month}?",
        "What contact change happened at {account} in {month}?",
    ),
}
_NAMES_PRODUCT = {"competitor_mention": False, "contact_change": False}

# fact type -> (verb phrase naming the product, verb phrase without it)
_VERB = {
    "complaint": ("complained about {product}", "complained"),
    "request": ("made requests about {product}", "made requests"),
    "risk": ("raised risks around {product}", "raised risks"),
    "praise": ("praised {product}", "gave praise"),
    "competitor_mention": ("brought up a competitor to {product}", "brought up a competitor"),
    "commitment": ("had commitments made about {product}", "had commitments made to them"),
    "pricing": ("raised pricing issues about {product}", "raised pricing issues"),
    "usage": ("gave usage updates on {product}", "gave usage updates"),
    "contact_change": ("had contact changes affecting {product}", "had contact changes"),
    "decision": ("made decisions about {product}", "made decisions"),
    "question": ("asked questions about {product}", "asked questions"),
}
_EVENT_PHRASE = {"qbr": "QBR", "go_live": "go-live"}
# A claim that dates the renewal itself: "renewal date of D", "renewal process ... decision
# expected by D", "During the renewal on D".
_RENEWAL_DATE_RE = re.compile(
    r"(?:renewal date of|renewal process\b.*\bexpected by|During the renewal on) (20\d\d-\d\d-\d\d)"
)


def _family(products: dict[str, str]) -> str | None:
    """The product line whose name starts every other product's name (`Lodepath` for
    `Lodepath T2`, ...): a question naming it could mean any of them."""
    for pid, name in products.items():
        if all(n == name or n.startswith(name + " ") for n in products.values()):
            return pid
    return None


# ---------------------------------------------------------------------------
# World cases
# ---------------------------------------------------------------------------


def world_cases(conn: psycopg.Connection, world: Any, plan: Any) -> list[ExamCase]:
    """Every world case, `WORLD_TARGETS` of each type, from the seeded graph on `conn`."""
    from portolan.bench.graph_world import _crm_sentence

    idx = ClaimIndex.load(conn)
    rng = random.Random(f"chat-exam:{world.id}:{SPLIT_SEED}")
    picker = _Picker(rng)
    accounts = {a.id: a for a in world.accounts}
    products = {p.id: p.name for p in world.vendor.products}
    competitors = {c.id: c.name for c in world.vendor.competitors}
    vendor = world.vendor.name

    # Each fact dated as the graph shows it: the planner files some facts in a later document
    # than their event, and a question's month or "since" means when it was said. A fact the
    # graph shows under two dates is left out.
    gold_of: dict[str, GoldClaim] = {}
    facts = []
    for f in sorted(world.facts, key=lambda f: f.id):
        g = idx.gold(f.claim, f.id)
        dates = idx.recorded(f.claim)
        if g is not None and len(dates) == 1:
            gold_of[f.id] = g
            facts.append(f.model_copy(update={"date": next(iter(dates))}))

    family = _family(products)

    def product_of(f: Any) -> str | None:
        return next((e for e in f.entities if e in products), None)

    def specific(f: Any) -> str | None:
        """The fact's product, unless it is the product line a question could not tell
        apart from its members."""
        prod = product_of(f)
        return None if prod == family else prod

    crm_gold: dict[tuple[str, str], GoldClaim] = {}
    crm_rows: dict[str, dict[str, str]] = {}
    for record in plan.crm_accounts:
        name = accounts[record.account_id].name
        crm_rows[record.account_id] = record.row
        for ff in record.field_facts:
            value = "; ".join(str(record.row.get(f, "")) for f in ff.fields)
            g = idx.gold(_crm_sentence(name, ff.fields[0], value, ff.date), ff.id)
            if g is not None:
                crm_gold[(record.account_id, ff.fields[0])] = g

    cases: dict[str, list[ExamCase]] = {t: [] for t in CASE_TYPES}

    def add(kind: str, question: str, gold: Sequence[GoldClaim], origin: str) -> ExamCase:
        return ExamCase("", "world", kind, question, tuple(gold), origin)

    # -- single fact: exactly one fact matches the question's account, product and month
    by_apm = Counter((f.account_id, product_of(f), _month(f.date)) for f in facts)
    by_atm = Counter((f.account_id, f.fact_type.value, _month(f.date)) for f in facts)
    by_am = Counter((f.account_id, _month(f.date)) for f in facts)
    single_pool: dict[str, list[ExamCase]] = defaultdict(list)
    for f in facts:
        t = f.fact_type.value
        if t not in _SINGLE:
            continue
        product = product_of(f)
        names_product = _NAMES_PRODUCT.get(t, True) and product is not None
        unique = (
            by_apm[(f.account_id, product, _month(f.date))] == 1
            if names_product
            else by_atm[(f.account_id, t, _month(f.date))] == 1
            and by_apm[(f.account_id, product, _month(f.date))] == 1
        )
        if product == family:
            unique = unique and by_am[(f.account_id, _month(f.date))] == 1
        if not unique:
            continue
        template = _SINGLE[t][0 if names_product else 1]
        q = template.format(
            account=accounts[f.account_id].name,
            product=products.get(product or "", ""),
            month=_month(f.date),
        )
        single_pool[t].append(add("single_fact", q, [gold_of[f.id]], "world:single"))

    # -- multi document: an arc's later events; the account a CRM renewal month picks out
    events = {e.id: e for e in world.events}
    facts_by_event: dict[str, list[Any]] = defaultdict(list)
    for f in facts:
        facts_by_event[f.event_id].append(f)
    arc_pool: list[ExamCase] = []
    for arc in sorted(world.arcs, key=lambda a: a.id):
        chain = [events[e] for e in arc.event_ids if e in events and facts_by_event.get(e)]
        said = {e.id: facts_by_event[e.id][0].date for e in chain}
        chain.sort(key=lambda e: (said[e.id], e.id))
        if len(chain) < 3:
            continue
        first, later = chain[0], chain[1:4]
        if said[later[0].id] <= said[first.id]:
            continue  # "next" must come after the first event was said
        product = next((products[p] for p in first.product_ids if p in products), None)
        with_product = f" with {product}" if product else ""
        q = (
            f"After the {_EVENT_PHRASE.get(first.type.value, first.type.value.replace('_', ' '))}"
            f"{with_product} at "
            f"{accounts[arc.account_id].name} in {_month(said[first.id])}, what happened next?"
        )
        gold = [gold_of[facts_by_event[e.id][0].id] for e in later]
        arc_pool.append(add("multi_doc", q, gold, "world:arc"))
    bridge_pool: list[ExamCase] = []
    renew_month = Counter(
        _month(accounts[a].renewal_date)
        for a, field_name in crm_gold
        if field_name == "renewal_date"
    )
    for (acc, field_name), crm in sorted(crm_gold.items()):
        if field_name != "renewal_date" or renew_month[_month(accounts[acc].renewal_date)] != 1:
            continue
        risks = sorted(
            (f for f in facts if f.account_id == acc and f.fact_type.value == "risk"),
            key=lambda f: (f.date, f.id),
        )
        if len(risks) < 2 or risks[-1].date == risks[-2].date:
            continue
        q = (
            f"The account whose contract renews in {_month(accounts[acc].renewal_date)}: "
            "what is the most recent risk it has raised?"
        )
        bridge_pool.append(add("multi_doc", q, [crm, gold_of[risks[-1].id]], "world:crm_bridge"))

    # -- recency: what we have heard since a date; the latest word on one product
    since_pool: list[ExamCase] = []
    latest_pool: list[ExamCase] = []
    by_account: dict[str, list[Any]] = defaultdict(list)
    for f in facts:
        by_account[f.account_id].append(f)
    for acc, fs in sorted(by_account.items()):
        dates = sorted({f.date for f in fs}, reverse=True)
        for n in (2, 3):
            if len(dates) <= n:
                break
            since = dates[n - 1]
            gold_facts = sorted((f for f in fs if f.date >= since), key=lambda f: (f.date, f.id))
            if 2 <= len(gold_facts) <= 4:
                q = (
                    f"What have we heard from {accounts[acc].name} since {since.isoformat()}?"
                    if n == 2
                    else f"What has changed at {accounts[acc].name} since {since.isoformat()}?"
                )
                since_pool.append(
                    add("recency", q, [gold_of[f.id] for f in gold_facts], "world:since")
                )
                break
        by_product: dict[str, list[Any]] = defaultdict(list)
        for f in fs:
            if specific(f):
                by_product[specific(f)].append(f)
        for prod, pfs in sorted(by_product.items()):
            pfs.sort(key=lambda f: (f.date, f.id))
            if len(pfs) >= 3 and pfs[-1].date != pfs[-2].date:
                q = (
                    f"What is the latest we have heard from {accounts[acc].name} about "
                    f"{products[prod]}?"
                )
                latest_pool.append(add("recency", q, [gold_of[pfs[-1].id]], "world:latest"))

    # -- cross account: several accounts, one filter
    cross_pool: dict[str, list[ExamCase]] = defaultdict(list)
    groups: dict[tuple[str, str, str], list[Any]] = defaultdict(list)
    for f in facts:
        if f.fact_type.value in _VERB and specific(f):
            groups[(f.fact_type.value, specific(f), _month(f.date))].append(f)
    for (t, prod, month), fs in sorted(groups.items()):
        accs = {f.account_id for f in fs}
        if 2 <= len(accs) <= 4 and len(fs) <= 4:
            verb = _VERB[t][0].format(product=products[prod])
            q = f"Which accounts {verb} in {month}, and what did they say?"
            cross_pool["type_product_month"].append(
                add("cross_account", q, [gold_of[f.id] for f in fs], "world:type_product_month")
            )
    children: dict[str, list[str]] = defaultdict(list)
    for a in world.accounts:
        if a.parent:
            children[a.parent].append(a.id)
    for parent, kids in sorted(children.items()):
        per: dict[tuple[str, str], list[Any]] = defaultdict(list)
        for f in facts:
            if f.account_id in kids and f.fact_type.value in _VERB:
                per[(f.fact_type.value, _month(f.date))].append(f)
        for (t, month), fs in sorted(per.items()):
            if len({f.account_id for f in fs}) >= 2 and len(fs) <= 4:
                q = (
                    f"Which of {_poss(accounts[parent].name)} facilities {_VERB[t][1]} in {month}, "
                    "and what did they say?"
                )
                cross_pool["hierarchy"].append(
                    add("cross_account", q, [gold_of[f.id] for f in fs], "world:hierarchy")
                )
    mentions: dict[tuple[str, str], list[Any]] = defaultdict(list)
    for f in facts:
        if f.fact_type.value == "competitor_mention":
            for e in f.entities:
                if e in competitors:
                    mentions[(e, _month(f.date))].append(f)
    for (cid, month), fs in sorted(mentions.items()):
        if 2 <= len({f.account_id for f in fs}) <= 3 and len(fs) <= 4:
            q = f"Which accounts brought up {competitors[cid]} in {month}, and in what context?"
            cross_pool["competitor"].append(
                add("cross_account", q, [gold_of[f.id] for f in fs], "world:competitor")
            )

    # -- abstain: questions the graph holds nothing for, checked against its text
    fact_accounts = sorted(by_account)
    rng_accounts = rng.sample(fact_accounts, len(fact_accounts))
    line = products.get(family or "", world.vendor.products[0].name)
    abstain_pool: list[ExamCase] = []

    def abstain(q: str, origin: str, *absent: str) -> None:
        if idx.absent(*absent):
            abstain_pool.append(add("abstain", q, [], origin))

    a1, a2, a3, a4 = (accounts[a].name for a in rng_accounts[:4])
    abstain(f"What NPS score has {a1} given {vendor}?", "world:abstain_nps", "nps", "net promoter")
    fake_account = next(
        f"{stem} Community Hospital"
        for stem in ("Harrowmere", "Brindlecote", "Corrowick", "Dunmarsh", "Eskerfield")
        if idx.absent(stem)
    )
    abstain(
        f"What complaints has {fake_account} raised about {line}?",
        "world:abstain_account",
        fake_account.split()[0],
    )
    abstain(f"Which accounts have deployed the {line} X9?", "world:abstain_product", " x9")
    before = world.window_start.replace(day=1) - dt.timedelta(days=1)
    if not any(
        f.account_id == rng_accounts[1]
        and min(idx.recorded(f.claim), default=dt.date.max) <= before
        for f in world.facts
    ):
        abstain(
            f"What did {a2} tell us in {_month(before)}?",
            "world:abstain_before_window",
            _month(before).split()[0] + " " + str(before.year),
        )
    abstain(f"How many licensed beds does {a3} have?", "world:abstain_attribute", " beds")
    fake_person = next(
        n for n in ("Marisol Quenby", "Tobias Wrenhall", "Ilse Varnholt") if idx.absent(n)
    )
    abstain(f"What did {fake_person} at {a4} ask us for?", "world:abstain_person", fake_person)
    abstain(f"Which EHR does {a1} run, Epic or Cerner?", "world:abstain_ehr", "epic", "cerner")

    # -- CRM versus calls: the CRM row says one thing; the calls bear on it
    crm_pool: dict[str, list[ExamCase]] = defaultdict(list)
    for acc in sorted(crm_rows):
        row = crm_rows[acc]
        name = accounts[acc].name
        risks = sorted(
            (f for f in by_account.get(acc, []) if f.fact_type.value == "risk"),
            key=lambda f: (f.date, f.id),
        )
        stage_gold = crm_gold.get((acc, "renewal_stage"))
        stage = row.get("renewal_stage")
        # "The two most recent" must pick out exactly two.
        if stage_gold and len(risks) >= 2 and (len(risks) == 2 or risks[-3].date < risks[-2].date):
            recent = [gold_of[f.id] for f in risks[-2:]]
            if stage == "renewed":
                q = (
                    f"Our CRM shows {_poss(name)} renewal as renewed. What are the two most recent "
                    "risks our conversations with them raised, and do they cut against that?"
                )
            else:
                q = (
                    f"Our CRM flags {_poss(name)} renewal as {stage.replace('_', ' ')}. What "
                    "are the two most recent risks our conversations with them raised?"
                )
            crm_pool["stage"].append(add("crm_conflict", q, recent, "world:crm_stage"))
        # Calls that put the renewal on a date the CRM row does not carry: all of them.
        renewal_gold = crm_gold.get((acc, "renewal_date"))
        differing = [
            f
            for f in sorted(by_account.get(acc, []), key=lambda f: (f.date, f.id))
            if (said := _RENEWAL_DATE_RE.search(f.claim)) and said.group(1) != row["renewal_date"]
        ]
        if renewal_gold and 1 <= len(differing) <= 3:
            q = (
                f"Our CRM says {_poss(name)} contract renews on {row['renewal_date']}. Have our "
                "conversations with them said anything different about the renewal timing?"
            )
            gold = [gold_of[f.id] for f in differing]
            crm_pool["renewal_date"].append(add("crm_conflict", q, gold, "world:crm_renewal_date"))

    # Rarest first, so the picker's one-case-per-fact rule costs the abundant types.
    cases["crm_conflict"] = picker.round_robin(crm_pool, WORLD_TARGETS["crm_conflict"])
    bridges = picker.take(bridge_pool, 2)
    cases["multi_doc"] = bridges + picker.take(arc_pool, WORLD_TARGETS["multi_doc"] - len(bridges))
    cases["cross_account"] = picker.round_robin(cross_pool, WORLD_TARGETS["cross_account"])
    since_cases = picker.take(since_pool, 3)
    cases["recency"] = since_cases + picker.take(
        latest_pool, WORLD_TARGETS["recency"] - len(since_cases)
    )
    cases["single_fact"] = picker.round_robin(single_pool, WORLD_TARGETS["single_fact"])
    cases["abstain"] = abstain_pool[: WORLD_TARGETS["abstain"]]

    out: list[ExamCase] = []
    for kind in CASE_TYPES:
        for i, case in enumerate(cases[kind], 1):
            out.append(_with_id(case, f"w-{_ABBR[kind]}-{i:02d}"))
    return out


_ABBR = {
    "single_fact": "sf",
    "multi_doc": "md",
    "recency": "rc",
    "cross_account": "xa",
    "abstain": "ab",
    "crm_conflict": "cc",
}


def _with_id(case: ExamCase, case_id: str) -> ExamCase:
    return dataclasses.replace(case, id=case_id)


# ---------------------------------------------------------------------------
# Demo cases
# ---------------------------------------------------------------------------

# The 20 multi-hop questions, by the exam's types (read from each question, 2026-09-29).
MULTIHOP_TYPES = {
    "mh_01": "multi_doc",
    "mh_02": "multi_doc",
    "mh_03": "multi_doc",
    "mh_04": "recency",
    "mh_05": "recency",
    "mh_06": "multi_doc",
    "mh_07": "recency",
    "mh_08": "multi_doc",
    "mh_09": "cross_account",
    "mh_10": "cross_account",
    "mh_11": "multi_doc",
    "mh_12": "recency",
    "mh_13": "multi_doc",
    "mh_14": "multi_doc",
    "mh_15": "multi_doc",
    "mh_16": "multi_doc",
    "mh_17": "multi_doc",
    "mh_18": "cross_account",
    "mh_19": "recency",
    "mh_20": "multi_doc",
}

# Topics the ten demo questions were tuned on; the demo templates leave them alone.
SMOKE_TOPIC_ACCOUNTS = {"larkspur", "larkspur-memorial", "larkspur-north"}
SMOKE_TOPIC_TYPES = {"complaint", "pricing", "request", "competitor_mention"}

# CRM versus what customers said, on the demo graph: (question, truth fact ids).
DEMO_CRM_CASES = (
    (
        "Our CRM lists Derek Hollis as Northlake Health System's Director of Clinical "
        "Engineering. Is that still his role?",
        ("F59",),
    ),
    (
        "Our CRM shows Sable Valley Health as an active $720,000 account with nothing flagged. "
        "Is there anything a rep should know before pitching new equipment there?",
        ("F14",),
    ),
    (
        "Our CRM has Northlake's Tidewater T3 tower expansion in qualification. Do our "
        "conversations with Northlake suggest any risk to it?",
        ("F15b",),
    ),
    (
        "Our CRM marks Kestrel Medical Group as churned in 2026-02. What do our records say "
        "about why?",
        ("F34",),
    ),
)

DEMO_ABSTAIN = (
    ("What CSAT score did Vireo Surgical Group give us last quarter?", ("csat",)),
    ("Which accounts have ordered the Tidewater T4?", ("t4",)),
    ("What did Meadowbrook Community Hospital tell us in January 2026?", ("january 2026",)),
    ("How many beds does Driftwood Community Hospital have?", (" beds",)),
    ("What did Harriet Colvane at St. Aldric Health ask us for?", ("colvane",)),
)


def _truth_names(table: Any) -> tuple[dict[str, str], dict[str, str], set[str]]:
    accounts = {a.id: a.name for a in table.accounts}
    products = {p["id"]: p["name"] for p in table.meta.get("products", [])}
    competitors = {c["id"] for c in table.meta.get("competitors", [])}
    return accounts, products, competitors


def demo_cases(conn: psycopg.Connection) -> list[ExamCase]:
    """The demo graph's cases: truth-table templates, the CRM and abstain cases above, and
    the multi-hop set (public, so train only)."""
    from portolan.bench.agentic_questions import DEMO_MULTIHOP_QUESTIONS
    from portolan.bench.truth import TruthTable

    idx = ClaimIndex.load(conn)
    table = TruthTable.load(TRUTH_PATH)
    accounts, products, _competitors = _truth_names(table)
    rng = random.Random(f"chat-exam:demo:{SPLIT_SEED}")
    public_notes = {n for q in DEMO_MULTIHOP_QUESTIONS for n in q.expected}

    facts = []
    gold_of: dict[str, GoldClaim] = {}
    account_of: dict[str, str] = {}
    for f in table.facts:
        if f.is_distractor or not f.claim or f.date is None:
            continue
        g = idx.gold(f.claim, f.id)
        acc = next((e for e in f.entities if e in accounts), None)
        if g is None or acc is None:
            continue
        gold_of[f.id] = g
        account_of[f.id] = acc
        facts.append(f)

    picker = _Picker(rng)

    def cand(kind: str, q: str, gold: Sequence[GoldClaim], origin: str) -> ExamCase:
        pub = any(set(g.notes) & public_notes for g in gold)
        return ExamCase("", "demo", kind, q, tuple(gold), origin, pub)

    # CRM versus conversations: every one kept; their facts are then off limits.
    crm = []
    for q, fact_ids in DEMO_CRM_CASES:
        gold = [gold_of[f] for f in fact_ids if f in gold_of]
        if len(gold) == len(fact_ids):
            crm.append(cand("crm_conflict", q, gold, "demo:crm_hand"))
    crm = picker.take(crm, len(crm))

    # The templates stay off the ten demo questions' topics.
    usable = [
        f
        for f in facts
        if account_of[f.id] not in SMOKE_TOPIC_ACCOUNTS and f.type not in SMOKE_TOPIC_TYPES
    ]
    usable_ids = {f.id for f in usable}

    # single fact: one fact of that type for that account in that month, on no multi-hop note
    key = Counter((account_of[f.id], f.type, _month(f.date)) for f in facts)
    single_pool: dict[str, list[ExamCase]] = defaultdict(list)
    for f in usable:
        if (
            f.type in _SINGLE
            and key[(account_of[f.id], f.type, _month(f.date))] == 1
            and not set(gold_of[f.id].notes) & public_notes
        ):
            q = _SINGLE[f.type][1].format(account=accounts[account_of[f.id]], month=_month(f.date))
            single_pool[f.type].append(cand("single_fact", q, [gold_of[f.id]], "demo:truth_single"))

    # recency: what we have heard from one account since a date
    by_account: dict[str, list[Any]] = defaultdict(list)
    for f in facts:
        by_account[account_of[f.id]].append(f)
    since_pool = []
    for acc, fs in sorted(by_account.items()):
        dates = sorted({f.date for f in fs}, reverse=True)
        if len(dates) < 3:
            continue
        since = dates[1]
        recent = sorted((f for f in fs if f.date >= since), key=lambda f: (f.date, f.id))
        if 2 <= len(recent) <= 4 and all(f.id in usable_ids for f in recent):
            q = f"What have we heard from {accounts[acc]} since {since.isoformat()}?"
            since_pool.append(
                cand("recency", q, [gold_of[f.id] for f in recent], "demo:truth_since")
            )

    # cross account: which accounts, one fact type, one month
    groups: dict[tuple[str, str], list[Any]] = defaultdict(list)
    for f in usable:
        if f.type in _VERB:
            groups[(f.type, _month(f.date))].append(f)
    everything: Counter[tuple[str, str]] = Counter((f.type, _month(f.date)) for f in facts)
    cross_pool = []
    for (t, month), fs in sorted(groups.items()):
        if 2 <= len({account_of[f.id] for f in fs}) <= 4 and len(fs) <= 4:
            q = f"Which accounts {_VERB[t][1]} in {month}, and what did they say?"
            case = cand("cross_account", q, [gold_of[f.id] for f in fs], "demo:truth_cross")
            if everything[(t, month)] != len(fs):
                # The month holds facts on the demo questions' topics too: the gold would
                # miss them. Drawn as before, then left out.
                case = dataclasses.replace(case, dropped="gold misses facts on smoke topics")
            cross_pool.append(case)

    picked = [
        *crm,
        *picker.take(cross_pool, DEMO_TARGETS["cross_account"]),
        *picker.take(since_pool, DEMO_TARGETS["recency"]),
        *picker.round_robin(single_pool, DEMO_TARGETS["single_fact"]),
        # abstain: checked absent from the graph's text
        *(
            cand("abstain", q, [], "demo:abstain")
            for q, absent in DEMO_ABSTAIN
            if idx.absent(*absent)
        ),
    ]
    out: list[ExamCase] = []
    for kind in CASE_TYPES:
        for i, case in enumerate((c for c in picked if c.type == kind), 1):
            out.append(_with_id(case, f"d-{_ABBR[kind]}-{i:02d}"))

    # the multi-hop set: its questions and gold are in portolan/bench/agentic_questions.py
    for q in DEMO_MULTIHOP_QUESTIONS:
        gold = tuple(GoldClaim(text, tuple(q.expected)) for text in q.answer_facts)
        out.append(
            ExamCase(
                f"d-{q.qid}", "demo", MULTIHOP_TYPES[q.qid], q.question, gold, "demo:multihop", True
            )
        )
    return out


def smoke_cases() -> list[ExamCase]:
    """The ten demo questions: run for regressions, never scored as exam cases. The last,
    an NPS question, is an abstain."""
    lines = [
        line.strip()
        for line in DEMO_QUESTIONS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return [
        ExamCase(
            f"smoke-{i:02d}",
            "demo",
            "abstain" if "NPS" in q else "smoke",
            q,
            (),
            "demo:smoke",
            True,
        )
        for i, q in enumerate(lines, 1)
    ]


# ---------------------------------------------------------------------------
# Split and storage
# ---------------------------------------------------------------------------


def split_cases(
    cases: Sequence[ExamCase], *, seed: int = SPLIT_SEED, test_fraction: float = TEST_FRACTION
) -> dict[str, list[str]]:
    """{"train": ids, "test": ids}: public cases to train; the rest, per type in id order,
    shuffled with `seed` and cut at `test_fraction`."""
    ids = [c.id for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate case ids")
    train = [c.id for c in cases if c.public]
    test: list[str] = []
    for kind in CASE_TYPES:
        pool = sorted(c.id for c in cases if not c.public and c.type == kind)
        random.Random(f"{seed}:{kind}").shuffle(pool)
        n_test = round(len(pool) * test_fraction)
        test += pool[:n_test]
        train += pool[n_test:]
    return {"train": sorted(train), "test": sorted(test)}


def build_all_cases(
    demo_conn: psycopg.Connection, world_conn: psycopg.Connection
) -> list[ExamCase]:
    from portolan.bench import graph_world
    from portolan.bench.chat_exam_graphs import WORLD_SPEC

    world, plan = graph_world.build_world(WORLD_SPEC)
    return [*world_cases(world_conn, world, plan), *demo_cases(demo_conn)]


def _dump(cases: Iterable[ExamCase]) -> str:
    return json.dumps([c.to_dict() for c in cases], indent=1, ensure_ascii=False) + "\n"


def write_exam(
    cases: Sequence[ExamCase],
    split: dict[str, list[str]],
    *,
    exam_dir: Path = EXAM_DIR,
    locked_path: Path = LOCKED_TEST_PATH,
) -> dict[str, Any]:
    """Train and smoke in the clear, test gzip-compressed under the locked path, and the
    split's ids; a dropped case is left out of both halves and listed with its reason.
    Returns counts by split and type (ids and numbers only)."""
    by_id = {c.id: c for c in cases}
    kept = {name: [i for i in ids if not by_id[i].dropped] for name, ids in split.items()}
    exam_dir.mkdir(parents=True, exist_ok=True)
    locked_path.parent.mkdir(parents=True, exist_ok=True)
    train = [by_id[i] for i in kept["train"]]
    test = [by_id[i] for i in kept["test"]]
    (exam_dir / "train.json").write_text(_dump(train), encoding="utf-8")
    (exam_dir / "smoke.json").write_text(_dump(smoke_cases()), encoding="utf-8")
    locked_path.write_bytes(gzip.compress(_dump(test).encode("utf-8"), mtime=0))
    counts = {name: dict(Counter(by_id[i].type for i in ids)) for name, ids in kept.items()}
    meta = {
        "seed": SPLIT_SEED,
        "test_fraction": TEST_FRACTION,
        "rule": "public cases (readable elsewhere in the repo) are train; the rest is split "
        "per type at random with the seed; dropped cases are drawn and split, then left out",
        "train": kept["train"],
        "test": kept["test"],
        "dropped": {c.id: c.dropped for c in cases if c.dropped},
        "counts": counts,
        "graphs": dict(Counter(c.graph for c in cases if not c.dropped)),
    }
    (exam_dir / "split.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    return counts


def load_split(
    name: str, *, exam_dir: Path = EXAM_DIR, locked_path: Path = LOCKED_TEST_PATH
) -> list[ExamCase]:
    """`train`, `test`, `smoke` or `both` (train then test). Test is read in-process from
    the locked file; nothing here prints it."""
    if name == "both":
        return [
            *load_split("train", exam_dir=exam_dir),
            *load_split("test", locked_path=locked_path),
        ]
    if name == "test":
        raw = gzip.decompress(locked_path.read_bytes()).decode("utf-8")
    elif name in ("train", "smoke"):
        raw = (exam_dir / f"{name}.json").read_text(encoding="utf-8")
    else:
        raise ValueError(f"unknown split {name!r}")
    return [ExamCase.from_dict(d) for d in json.loads(raw)]
