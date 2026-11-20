"""Truth table -> one brief per source.

A `Brief` is everything a prose writer (human, subagent, or the Anthropic
API) needs to produce one source: who is talking, what date it is, which
facts must be planted where, the exact phrasing constraint for each fact,
what must never appear, and — for noise/distractor sources — a plausible
topic with zero facts.

This module does no writing. It only reads `corpus/truth-table.yaml` and
turns it into structured `Brief` objects that `prose.py` (and, in manual
mode, a human or a delegated writer) consumes, and that `manifest.py`
cross-checks the finished corpus against.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

# Mirrors portolan/extract/schema.py QUOTE_REQUIRED_TYPES exactly — kept as
# a local constant rather than importing from portolan.extract, since this
# package must not depend on the extractor another agent owns.
QUOTE_REQUIRED_TYPES: frozenset[str] = frozenset(
    {"complaint", "praise", "competitor_mention", "commitment", "pricing"}
)

SourceType = Literal["call", "ticket", "email", "rss", "crm", "slack", "teams"]

# --------------------------------------------------------------------------
# Per-fact overrides the truth table expresses in prose, not structured data.
# --------------------------------------------------------------------------

# Some facts land in more than one source with a DIFFERENT phrasing per
# source (the truth table folds both into one `phrasing` string, e.g. F02:
# 'ticket says "4 days late"; the call says "that late order last week"').
# Keyed by (fact_id, source_id).
PHRASING_BY_SOURCE: dict[tuple[str, str], str] = {
    ("F02", "S-ticket-031"): 'the ticket must say "4 days late"',
    ("F02", "S-call-009"): 'the call must call it "that late order last week"',
}

# Literal strings that must never appear in the source that plants this
# fact, beyond what a plain reading of `phrasing` already implies. Checked
# case-insensitively by the writer/reviewer.
FORBIDDEN_OVERRIDES: dict[str, list[str]] = {
    "F01": ["Dayton", "Dayton branch", "Dayton distribution"],
    "F03": ["Memorial", "Larkspur Memorial"],
    "F06": ["risk", "churn", "at risk", "churning"],
    "F17": ["competitor", "Corvane", "Brightline"],
    # F07 additionally forbids every account and product name — computed
    # dynamically in build_briefs() from the registry, since that list must
    # track the truth table rather than being retyped here.
}

# Distractor sources carry a specific instruction instead of facts.
DISTRACTOR_INSTRUCTIONS: dict[str, str] = {
    "D01": (
        "Write an industry-news item about a 'Riverside Clinic' in Bend, "
        "Oregon adopting a rival vendor's infusion pump. This must read as "
        "an unrelated facility — different state, different ownership, "
        "no mention of Ashgrove, Halvard, or the Tidewater line — so an "
        "alias matcher that fires on the bare word 'Riverside' produces a "
        "false positive against Ashgrove Riverside Hospital."
    ),
    "D02": (
        "Write a real out-of-office auto-reply from Owen Castellane (VP "
        "Operations, St. Aldric Health), subject line 'Automatic reply: "
        "<original subject>', reacting to an inbound message from Priya "
        "Natarajan. Standard corporate auto-reply register: away dates, "
        "who to contact instead, no new facts."
    ),
    "D03": (
        "Write a ten-reply email thread following S-email-021 (Owen "
        "Castellane's request for a Spanish-language Tidewater T2 "
        "quick-start card). Every reply quotes the ENTIRE prior message "
        "chain above it with '>' prefixes, in the standard bottom-posting "
        "style, so message 10 contains the full history nine times over. "
        "The new content in each reply is short logistics chatter "
        "(shipping address, print proof, quantity) — no new planted facts, "
        "nothing that should extract as a fact distinct from F21."
    ),
    "D04": (
        "Write a routine, low-stakes check-in call between Nina Vale "
        "(Vireo Surgical Group) and Priya Natarajan about scheduling a "
        "training date for new OR staff. Friendly, a little small talk, "
        "no complaints, no pricing, no competitors, no decisions — nothing "
        "that should extract as a fact of any type."
    ),
}

NOISE_CALL_TOPICS = [
    "a routine quarterly check-in with no open issues",
    "scheduling a biomed refresher training session",
    "confirming receipt of a firmware update notice",
    "a courtesy call after a clean quarter",
]

NOISE_TICKET_SUBJECTS = [
    "Password reset for the HalvardCare portal",
    "How do I run a self-test on a Tidewater T2 pump?",
    "Requesting a replacement charging cradle",
    "Alarm tripped during routine maintenance — resolved",
    "Question about consumable reorder minimums",
    "Portal login shows the wrong facility",
    "How to export a usage report from the portal",
    "Replacement battery for a Tidewater T3",
    "Confirming a scheduled preventive-maintenance visit",
    "Where do I find the in-service video library?",
]

NOISE_EMAIL_SUBJECTS = [
    "Scheduling: quarterly business review",
    "Halvard Medical newsletter — Q3 product update",
    "Calendar invite: biomed refresher training",
    "Internal: territory coverage this week",
    "Reminder: preventive maintenance window next month",
    "Following up on our call",
    "Halvard Medical newsletter — regulatory update",
    "Internal: expense report reminder",
]

NOISE_RSS_TITLES_INDUSTRY = [
    "Infusion pump interoperability standards see slow adoption",
    "Hospital group purchasing organizations report flat capital spend",
    "Nurse staffing ratios remain a top operations concern",
    "Survey: biomed teams want faster vendor field response",
]

NOISE_RSS_TITLES_LARKSPUR = [
    "Larkspur Regional Health posts about a nurse appreciation event",
    "Larkspur Regional Health shares a community health fair recap",
    "Larkspur Regional Health congratulates a retiring biomed engineer",
]


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_truth_table(path: str | Path) -> dict[str, Any]:
    """Load and return the truth table as a plain dict."""
    return _load_yaml(Path(path))


# --------------------------------------------------------------------------
# Registry: name/role/email lookups shared by every module that writes text.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PersonRef:
    id: str
    name: str
    role: str | None
    email: str | None
    account: str | None  # None for internal Halvard staff


@dataclass(frozen=True)
class AccountRef:
    id: str
    crm_id: str
    name: str
    kind: str
    parent: str | None
    domain: str
    arr: int
    region: str
    aliases: list[str]
    status: str | None


class Registry:
    """Read-only lookups over the truth table's entities."""

    def __init__(self, truth: dict[str, Any]) -> None:
        self._truth = truth
        self._people: dict[str, PersonRef] = {}
        for c in truth["contacts"]:
            self._people[c["id"]] = PersonRef(
                id=c["id"],
                name=c["name"],
                role=c.get("title"),
                email=c.get("email"),
                account=c.get("account"),
            )
        for p in truth["meta"]["internal_people"]:
            self._people[p["id"]] = PersonRef(
                id=p["id"], name=p["name"], role=p.get("role"), email=p.get("email"), account=None
            )
        self._accounts: dict[str, AccountRef] = {}
        for a in truth["accounts"]:
            self._accounts[a["id"]] = AccountRef(
                id=a["id"],
                crm_id=a["crm_id"],
                name=a["name"],
                kind=a["kind"],
                parent=a.get("parent"),
                domain=a["domain"],
                arr=a["arr"],
                region=a["region"],
                aliases=a.get("aliases", []),
                status=a.get("status"),
            )
        self._products = {p["id"]: p for p in truth["meta"]["products"]}
        self._competitors = {c["id"]: c for c in truth["meta"]["competitors"]}
        self._branches = {b["id"]: b for b in truth["meta"]["branches"]}

    def person(self, person_id: str) -> PersonRef:
        return self._people[person_id]

    def account(self, account_id: str) -> AccountRef:
        return self._accounts[account_id]

    def all_accounts(self) -> list[AccountRef]:
        return list(self._accounts.values())

    def all_people(self) -> list[PersonRef]:
        return list(self._people.values())

    def product_name(self, product_id: str) -> str:
        return self._products[product_id]["name"]

    def all_product_names(self) -> list[str]:
        names: list[str] = []
        for p in self._products.values():
            names.append(p["name"])
            names.extend(p.get("aliases", []))
        return names

    def all_account_names(self) -> list[str]:
        names: list[str] = []
        for a in self._accounts.values():
            names.append(a.name)
            names.extend(a.aliases)
        return names

    def competitor_name(self, competitor_id: str) -> str:
        return self._competitors[competitor_id]["name"]

    def branch_name(self, branch_id: str) -> str:
        return self._branches[branch_id]["name"]

    def branch_for_account(self, account_id: str) -> str | None:
        for b in self._branches.values():
            if account_id in b.get("serves", []):
                return b["id"]
        return None


# --------------------------------------------------------------------------
# Brief data model
# --------------------------------------------------------------------------


@dataclass
class FactPlant:
    """One fact this source must plant."""

    fact_id: str
    fact_type: str
    claim: str
    entities: list[str]
    date: str
    speaker: str | None
    position: str | None  # "early" | "middle" | "late" | None
    phrasing: str | None
    forbidden: list[str]
    quote_required: bool


@dataclass
class BriefParticipant:
    id: str
    name: str
    role: str | None
    email: str | None


@dataclass
class Brief:
    """Everything needed to write one source."""

    source_id: str
    source_type: SourceType
    date: str
    participants: list[BriefParticipant]
    accounts: list[str]
    target_tokens: int | None
    facts: list[FactPlant]
    is_noise: bool = False
    is_distractor: bool = False
    distractor_id: str | None = None
    distractor_instruction: str | None = None
    topic: str | None = None
    internal_only: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def must_not_appear(self) -> list[str]:
        out: list[str] = []
        for f in self.facts:
            out.extend(f.forbidden)
        return out


def _tokens_for_words(words: int) -> int:
    return round(words / 0.75)


def words_for_tokens(tokens: int) -> int:
    return round(tokens * 0.75)


def _participant_refs(registry: Registry, ids: list[str]) -> list[BriefParticipant]:
    out = []
    for pid in ids:
        p = registry.person(pid)
        out.append(BriefParticipant(id=p.id, name=p.name, role=p.role, email=p.email))
    return out


def _facts_by_id(truth: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {f["id"]: f for f in truth["facts"]}


def _forbidden_for(fact_id: str, registry: Registry) -> list[str]:
    # F46 (v1.1, added 2026-09-13) is the same shape as F07: an internal ops
    # email about the Dayton branch with no customer or product named.
    if fact_id in ("F07", "F46"):
        return (
            list(FORBIDDEN_OVERRIDES.get(fact_id, []))
            + registry.all_account_names()
            + registry.all_product_names()
        )
    return list(FORBIDDEN_OVERRIDES.get(fact_id, []))


def _phrasing_for(fact_id: str, source_id: str, default: str | None) -> str | None:
    return PHRASING_BY_SOURCE.get((fact_id, source_id), default)


def _round_robin(items: list[str], index: int) -> str:
    """Pick item `index` from a shuffled copy of `items`, wrapping (and
    re-shuffling with a new, still-deterministic seed) after each full pass
    — spreads noise topics/subjects evenly instead of `rng.choice`'s
    occasional immediate repeat. Deterministic in `index` alone, so it does
    not depend on — or disturb — any other rng's call order."""
    cycle_len = len(items)
    pass_number, position = divmod(index, cycle_len)
    shuffled = list(items)
    random.Random(f"{tuple(items)}:{pass_number}").shuffle(shuffled)
    return shuffled[position]


def _fact_plant(
    fact_row: dict[str, Any],
    source_id: str,
    speaker: str | None,
    position: str | None,
    registry: Registry,
) -> FactPlant:
    return FactPlant(
        fact_id=fact_row["id"],
        fact_type=fact_row["type"],
        claim=fact_row["claim"],
        entities=fact_row.get("entities", []),
        date=str(fact_row["date"]),
        speaker=speaker,
        position=position,
        phrasing=_phrasing_for(fact_row["id"], source_id, fact_row.get("phrasing")),
        forbidden=_forbidden_for(fact_row["id"], registry),
        quote_required=fact_row["type"] in QUOTE_REQUIRED_TYPES,
    )


def _facts_for_source(truth: dict[str, Any], source_id: str, registry: Registry) -> list[FactPlant]:
    """Every fact (F.. and distractor D..) whose `sources` list names this source id."""
    out: list[FactPlant] = []
    for fact_row in truth["facts"]:
        if fact_row["type"] == "distractor":
            continue
        for src in fact_row.get("sources", []):
            if src["id"] == source_id:
                out.append(
                    _fact_plant(
                        fact_row, source_id, src.get("speaker"), src.get("position"), registry
                    )
                )
    return out


def build_calls(
    truth: dict[str, Any], registry: Registry, rng: random.Random, reduced: bool
) -> list[Brief]:
    briefs: list[Brief] = []
    distractor_by_source = {
        src["id"]: fr["id"]
        for fr in truth["facts"]
        if fr["type"] == "distractor"
        for src in fr.get("sources", [])
    }
    for item in truth["sources"]["calls"]["items"]:
        sid = item["id"]
        facts = _facts_for_source(truth, sid, registry)
        distractor_id = distractor_by_source.get(sid)
        target = words_for_tokens(3000) if (distractor_id and reduced) else None
        if not distractor_id:
            # Planted, fact-bearing call. Reduced manual-mode band: 6,000-9,000
            # tokens, scaled up slightly with fact/participant density so
            # "middle" facts still land past a 4k-token segment boundary.
            base = 6000 if reduced else item["length_tokens"]
            if reduced:
                base += 500 * max(0, len(facts) - 1) + 500 * max(0, len(item["participants"]) - 2)
                base = min(base, 9000)
                base = MANUAL_LENGTH_OVERRIDE_TOKENS.get(sid, base)
            target = words_for_tokens(base)
        briefs.append(
            Brief(
                source_id=sid,
                source_type="call",
                date=str(item["date"]),
                participants=_participant_refs(registry, item["participants"]),
                accounts=[item["account"]],
                target_tokens=None,
                facts=facts,
                is_distractor=bool(distractor_id),
                distractor_id=distractor_id,
                distractor_instruction=DISTRACTOR_INSTRUCTIONS.get(distractor_id)
                if distractor_id
                else None,
                extra={"target_words": target},
            )
        )
    # Noise calls: routine check-ins with existing accounts, no facts.
    noise_count = 2 if reduced else truth["sources"]["calls"]["noise"]["count"]
    noise_words = words_for_tokens(3000) if reduced else words_for_tokens(12000)
    active_accounts = [
        a for a in registry.all_accounts() if a.status is None and a.kind != "capital_partner"
    ]
    account_reps = [
        p["id"] for p in truth["meta"]["internal_people"] if p.get("role") == "account rep"
    ]
    for i in range(noise_count):
        account = rng.choice(active_accounts)
        contacts = [
            c for c in truth["contacts"] if c.get("account") == account.id and c.get("email")
        ]
        # Fallback (build instructions item 1, generalized 2026-09-19 for
        # `worlds.py`'s freshly-generated truth tables): the FIRST contact
        # in the truth table, not a hardcoded id from the hand-built
        # corpus -- no new `rng` draw here, so the existing corpus's noise
        # draw sequence is unchanged whenever `contacts` is non-empty.
        contact_id = rng.choice(contacts)["id"] if contacts else truth["contacts"][0]["id"]
        rep_id = rng.choice(account_reps)
        briefs.append(
            Brief(
                source_id=f"S-call-n{i + 1:02d}",
                source_type="call",
                date=str(truth["meta"]["window"]["start"]),
                participants=_participant_refs(registry, [contact_id, rep_id]),
                accounts=[account.id],
                target_tokens=None,
                facts=[],
                is_noise=True,
                topic=_round_robin(NOISE_CALL_TOPICS, i),
                extra={"target_words": noise_words},
            )
        )
    return briefs


def build_tickets(
    truth: dict[str, Any], registry: Registry, rng: random.Random, reduced: bool
) -> list[Brief]:
    briefs: list[Brief] = []
    for item in truth["sources"]["tickets"]["items"]:
        sid = item["id"]
        facts = _facts_for_source(truth, sid, registry)
        briefs.append(
            Brief(
                source_id=sid,
                source_type="ticket",
                date=str(item["date"]),
                participants=_participant_refs(registry, [item["requester"]]),
                accounts=[item["account"]],
                target_tokens=None,
                facts=facts,
                extra={"requester": item["requester"]},
            )
        )
    noise_count = 15 if reduced else truth["sources"]["tickets"]["noise"]["count"]
    active_accounts = [
        a for a in registry.all_accounts() if a.status is None and a.kind != "capital_partner"
    ]
    for i in range(noise_count):
        account = rng.choice(active_accounts)
        contacts = [c for c in truth["contacts"] if c.get("account") == account.id]
        requester = rng.choice(contacts)["id"] if contacts else truth["contacts"][0]["id"]
        d = truth["meta"]["window"]["start"]
        briefs.append(
            Brief(
                source_id=f"S-ticket-n{i + 1:03d}",
                source_type="ticket",
                date=str(d),
                participants=_participant_refs(registry, [requester]),
                accounts=[account.id],
                target_tokens=None,
                facts=[],
                is_noise=True,
                topic=_round_robin(NOISE_TICKET_SUBJECTS, i),
                extra={"requester": requester},
            )
        )
    return briefs


def build_emails(
    truth: dict[str, Any], registry: Registry, rng: random.Random, reduced: bool
) -> list[Brief]:
    briefs: list[Brief] = []
    distractor_by_source = {
        src["id"]: fr["id"]
        for fr in truth["facts"]
        if fr["type"] == "distractor"
        for src in fr.get("sources", [])
    }
    for item in truth["sources"]["emails"]["items"]:
        sid = item["id"]
        facts = _facts_for_source(truth, sid, registry)
        distractor_id = distractor_by_source.get(sid)
        participants_ids = [item["from"]] + list(item.get("to", []))
        briefs.append(
            Brief(
                source_id=sid,
                source_type="email",
                date=str(item["date"]),
                participants=_participant_refs(registry, participants_ids),
                accounts=[],
                target_tokens=None,
                facts=facts,
                is_distractor=bool(distractor_id),
                distractor_id=distractor_id,
                distractor_instruction=DISTRACTOR_INSTRUCTIONS.get(distractor_id)
                if distractor_id
                else None,
                internal_only=bool(item.get("internal_only", False)),
                extra={
                    "from": item["from"],
                    "to": list(item.get("to", [])),
                    "replies": item.get("replies"),
                },
            )
        )
    noise_count = 20 if reduced else truth["sources"]["emails"]["noise"]["count"]
    active_accounts = [
        a for a in registry.all_accounts() if a.status is None and a.kind != "capital_partner"
    ]
    # Every internal person, not a hardcoded id list — generalized
    # 2026-09-19 for `worlds.py`'s freshly-generated truth tables. Same
    # content and order as the original hardcoded list for
    # `corpus/truth-table.yaml` (its `meta.internal_people` lists exactly
    # these five people in this order), so the original corpus's noise
    # draw sequence is unchanged.
    reps = [p["id"] for p in truth["meta"]["internal_people"]]
    for i in range(noise_count):
        internal = rng.random() < 0.35
        if internal:
            from_id, to_id = rng.sample(reps, 2)
        else:
            account = rng.choice(active_accounts)
            contacts = [c for c in truth["contacts"] if c.get("account") == account.id]
            from_id = rng.choice(contacts)["id"] if contacts else rng.choice(reps)
            to_id = rng.choice(reps)
        d = truth["meta"]["window"]["start"]
        briefs.append(
            Brief(
                source_id=f"S-email-n{i + 1:03d}",
                source_type="email",
                date=str(d),
                participants=_participant_refs(registry, [from_id, to_id]),
                accounts=[],
                target_tokens=None,
                facts=[],
                is_noise=True,
                topic=_round_robin(NOISE_EMAIL_SUBJECTS, i),
                internal_only=internal,
                extra={"from": from_id, "to": [to_id]},
            )
        )
    return briefs


def build_rss(
    truth: dict[str, Any], registry: Registry, rng: random.Random, reduced: bool
) -> list[Brief]:
    briefs: list[Brief] = []
    distractor_by_source = {
        src["id"]: fr["id"]
        for fr in truth["facts"]
        if fr["type"] == "distractor"
        for src in fr.get("sources", [])
    }
    for item in truth["sources"]["rss"]["items"]:
        sid = item["id"]
        facts = _facts_for_source(truth, sid, registry)
        distractor_id = distractor_by_source.get(sid)
        briefs.append(
            Brief(
                source_id=sid,
                source_type="rss",
                date=str(item["date"]),
                participants=[],
                accounts=[],
                target_tokens=None,
                facts=facts,
                is_distractor=bool(distractor_id),
                distractor_id=distractor_id,
                distractor_instruction=DISTRACTOR_INSTRUCTIONS.get(distractor_id)
                if distractor_id
                else None,
                extra={"feed": item["feed"]},
            )
        )
    noise_count = 8 if reduced else truth["sources"]["rss"]["noise"]["count"]
    for i in range(noise_count):
        feed = "industry-news" if i % 2 == 0 else "linkedin-larkspur"
        d = truth["meta"]["window"]["start"]
        briefs.append(
            Brief(
                source_id=f"S-rss-n{i + 1:03d}",
                source_type="rss",
                date=str(d),
                participants=[],
                accounts=[],
                target_tokens=None,
                facts=[],
                is_noise=True,
                topic=_round_robin(
                    NOISE_RSS_TITLES_INDUSTRY
                    if feed == "industry-news"
                    else NOISE_RSS_TITLES_LARKSPUR,
                    i // 2,
                ),
                extra={"feed": feed},
            )
        )
    return briefs


def build_crm(truth: dict[str, Any], registry: Registry) -> list[Brief]:
    """CRM briefs are informational only — structured.py writes the CSVs
    directly from the truth table's accounts/contacts/opportunities. One
    brief per logical CRM source id keeps the manifest's source list
    complete and lets manifest.py point every CRM fact at the right file.
    """
    briefs: list[Brief] = []
    for item in truth["sources"]["crm"]["items"]:
        sid = item["id"]
        facts = _facts_for_source(truth, sid, registry)
        briefs.append(
            Brief(
                source_id=sid,
                source_type="crm",
                date=str(item.get("export_date", truth["meta"]["window"]["end"])),
                participants=[],
                accounts=[],
                target_tokens=None,
                facts=facts,
                extra={"object": item["object"], "export_date": item.get("export_date")},
            )
        )
    return briefs


# S-call-011 plants F04 as two mentions "twenty minutes apart in the same
# call" (both tagged position=middle). The reduced manual-mode band (6,000-
# 9,000 tokens) caps a call at roughly 45 minutes of dialogue, and the
# middle 35-65% window of a 45-minute call is itself only ~13.5 minutes
# wide — too narrow to fit a full 20-minute gap without spilling out of
# "middle". This source is pinned to the top of the reduced band (9,000
# tokens) and the phrasing constraint is relaxed to "as much of a gap as
# the middle window allows" rather than a literal 20 minutes; see
# GENERATOR.md for the full note.
MANUAL_LENGTH_OVERRIDE_TOKENS: dict[str, int] = {"S-call-011": 9000}

# F04's phrasing note asks for "two separate mentions twenty minutes apart
# in the same call" — deliberately TWO ⟦F04⟧...⟦/F04⟧ marker pairs in
# S-call-011, not a duplicate-marker mistake. markers.verify_exactly_one and
# manifest.validate() (set-based on source id, not placement count) both
# already treat this correctly; this constant is what tells the call-marker
# verification step to allow it.
MULTI_MENTION_FACTS: frozenset[str] = frozenset({"F04"})

NOISE_SEED = 20260912


def build_briefs(
    truth: dict[str, Any], mode: Literal["manual", "api"], reduced: bool
) -> list[Brief]:
    """Build every source brief: planted, distractor, and noise."""
    registry = Registry(truth)
    rng = random.Random(NOISE_SEED)
    briefs: list[Brief] = []
    briefs.extend(build_calls(truth, registry, rng, reduced))
    briefs.extend(build_tickets(truth, registry, rng, reduced))
    briefs.extend(build_emails(truth, registry, rng, reduced))
    briefs.extend(build_rss(truth, registry, rng, reduced))
    briefs.extend(build_crm(truth, registry))
    return briefs


# --------------------------------------------------------------------------
# Chat (Slack / Teams) and CRM-activity-note briefs — added 2026-09-19 for
# the synthetic-data pipeline (`corpus/generator/worlds.py`, `cli.py
# synth`). These two source kinds don't exist in the hand-built
# `corpus/truth-table.yaml`, so nothing above this point changes: every
# existing `build_calls`/`build_tickets`/`build_emails`/`build_rss`/
# `build_crm` call and test keeps consuming that file exactly as before.
# --------------------------------------------------------------------------


def build_chat_briefs(
    kind: Literal["slack", "teams"], truth: dict[str, Any], registry: Registry
) -> list[Brief]:
    """One brief per `truth["sources"][kind]["items"]` entry -- a single
    short chat message. Unlike `build_calls`/`build_tickets`/`build_emails`,
    there is no separate `noise:` count to auto-generate here: `worlds.py`
    already writes every item (fact-bearing AND noise) directly into
    `items`, so `is_noise` is simply "this item's source id carries no
    facts" (`_facts_for_source` returns `[]` for those, same test every
    other `is_noise` here already relies on).

    Each item: `{id, date, channel, account, speaker, is_customer, thread}`.
    `extra["channel"]`/`extra["is_customer"]`/`extra["thread"]` carry the
    fields `cli.py`'s chat-writing step needs that don't fit the `Brief`
    dataclass's existing typed fields.
    """
    briefs: list[Brief] = []
    section = truth.get("sources", {}).get(kind)
    if not section:
        return briefs
    for item in section.get("items", []):
        sid = item["id"]
        facts = _facts_for_source(truth, sid, registry)
        briefs.append(
            Brief(
                source_id=sid,
                source_type=kind,
                date=str(item["date"]),
                participants=_participant_refs(registry, [item["speaker"]]),
                accounts=[item["account"]] if item.get("account") else [],
                target_tokens=None,
                facts=facts,
                is_noise=not facts,
                extra={
                    "channel": item["channel"],
                    "is_customer": bool(item.get("is_customer", True)),
                    "thread": bool(item.get("thread", False)),
                },
            )
        )
    return briefs


def build_crm_notes(truth: dict[str, Any], registry: Registry) -> list[Brief]:
    """One brief per `truth["sources"]["crm_notes"]["items"]` entry -- a
    short free-text CRM activity note (build instructions item 3: "CRM:
    ... free-text activity notes"). Same "no separate noise count" shape
    as `build_chat_briefs` -- `worlds.py` writes every note, fact-bearing
    or not, directly into `items`.

    Each item: `{id, date, account, contact, author, note_id}`.
    """
    briefs: list[Brief] = []
    section = truth.get("sources", {}).get("crm_notes")
    if not section:
        return briefs
    for item in section.get("items", []):
        sid = item["id"]
        facts = _facts_for_source(truth, sid, registry)
        briefs.append(
            Brief(
                source_id=sid,
                source_type="crm",
                date=str(item["date"]),
                participants=_participant_refs(registry, [item["author"]]),
                accounts=[item["account"]] if item.get("account") else [],
                target_tokens=None,
                facts=facts,
                is_noise=not facts,
                extra={
                    "note_id": item["note_id"],
                    "account_id": item.get("account"),
                    "contact_id": item.get("contact"),
                    "author_id": item["author"],
                },
            )
        )
    return briefs
