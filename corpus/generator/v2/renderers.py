"""WP3: code renderers for every document part that needs no model (spec
section 2: "Code renders the structure: headers, ids, timestamps, CRM
fields, link lists, quoted-reply blocks and signatures. Models write only
the text fields.").

`render(plan, world, doc_plan, pack)` returns a `RenderedDoc`: the
document's files with every structural value final, plus one `Slot` per
place WP4's writer must fill -- a scripted fact, a decoy, a stretch of
allowed colour, or a subject line. The skeleton has the shape the spec's
section 2 table asks for (a call of 1,000 to 4,000 words in 50 to 150
turns, a ticket of 3 to 9 comments, an email thread of 2 to 7 messages, a
chat day of several messages), so WP4 fills slots in place and never has
to invent structure; each colour slot carries a `target_words` length.

**Provenance** (spec Build notes, WP2 review): no code- or Claude-written
prose reaches a file. What code writes is limited to header keys, JSON
keys, closed-vocabulary values (`STRUCTURAL_VOCABULARY`), and world values
(names, titles, ids, dates, emails). `tests/test_world_generator_v2_wp3.py`
tokenizes every rendered file of a world and fails on any other word. The
first build rendered one code template as every subject line
("<account> -- email thread") and fallback names ("Account Rep",
"unknown"); subjects are now `SlotKind.SUBJECT` slots and a participant is
always a real person.

Placeholders are unique per slot (`[SLOT <slot_id> ...]`), so WP4 replaces
by exact string; WP4 must fail a document in which any `[SLOT` or
`[QUOTE` survives. A `[QUOTE <message id>]` placeholder is *code's* job
after WP4 has written the quoted message (`RenderedDoc.quote_ops`): copy
that message's body with `> ` before every line and every span marker
removed. Facts in quoted text are then stripped by `prepare_text` and are
never labeled (spec section 5).

Timestamps and dates are final: a call turn's time comes from the
cumulative `target_words` at about 2.6 words per second, and every unit
that states a fact is dated on or after that fact. If WP4 re-times a call
from the words it actually wrote, it must recompute the call locators.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
from datetime import date, datetime, timedelta
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from corpus.generator.structured import (
    call_locator,
    seconds_to_hhmmss,
    slack_channel_path,
    slack_locator,
    slack_ts,
    teams_channel_path,
    teams_created_datetime,
    teams_locator,
    ticket_locator,
)
from corpus.generator.v2.models import Account, Fact, Person, PersonSide, World
from corpus.generator.v2.packs.schema import DecoyClass, VerticalPack
from corpus.generator.v2.planner import (
    CRM_ACCOUNT_FIELDS,
    CRM_ACCOUNTS_PATH,
    CrmAccountRecordPlan,
    DocKind,
    DocPlan,
    DocRole,
    PlannedDecoy,
    WorldDocPlan,
)

__all__ = [
    "STRUCTURAL_VOCABULARY",
    "QuoteOp",
    "RenderedDoc",
    "Slot",
    "SlotKind",
    "materialize_crm_accounts",
    "materialize_rendered_doc",
    "render",
    "render_chat_directories",
    "slack_user_id",
]


class SlotKind(StrEnum):
    FACT = "fact"
    DECOY = "decoy"
    COLOR = "color"  # allowed colour only (spec 3.1): no customer fact
    SUBJECT = "subject"  # a subject line or document title: no customer fact


class Slot(BaseModel):
    """One placeholder WP4's writer must replace. `locator` is the unit's
    production locator (`None` for a subject line, which sits in header
    metadata). `files` lists every file the placeholder appears in (an
    email thread repeats its subject in every message)."""

    model_config = ConfigDict(extra="forbid")

    slot_id: str
    doc_id: str
    kind: SlotKind
    fact_id: str | None = None
    decoy_id: str | None = None
    speaker_person_id: str | None = None
    locator: str | None = None
    files: list[str] = Field(default_factory=list)
    placeholder_text: str
    target_words: int | None = None
    unit_date: date | None = None


class QuoteOp(BaseModel):
    """Code expands `placeholder_text` in `file` into the quoted body of
    `quoted_message_id` (from `quoted_file`) once WP4 has written it."""

    model_config = ConfigDict(extra="forbid")

    file: str
    placeholder_text: str
    quoted_file: str
    quoted_message_id: str


class RenderedDoc(BaseModel):
    """One planned document's skeleton: `files` maps a `corpus/out`-
    relative path to its exact text (JSON pre-serialized)."""

    model_config = ConfigDict(extra="forbid")

    plan_id: str
    kind: DocKind
    role: DocRole
    files: dict[str, str] = Field(default_factory=dict)
    slots: list[Slot] = Field(default_factory=list)
    quote_ops: list[QuoteOp] = Field(default_factory=list)


# Every literal word the renderers write outside a slot, a quote
# placeholder or a world value. Lowercase; the provenance test splits on
# non-letters, so `createdDateTime` is one token and `account_crm_id`
# three.
STRUCTURAL_VOCABULARY: frozenset[str] = frozenset(
    {
        # call / meeting / doc headers
        "date", "call", "type", "account", "attendees", "meeting", "document",
        "author", "title",
        # closed-vocabulary subtypes (planner.CALL_TYPE_BY_EVENT etc.)
        "implementation", "expansion", "qbr", "escalation", "renewal", "check",
        "in", "training", "plan", "release", "notes", "summary", "proposal",
        "prep", "review", "how", "to", "guide",
        # email headers and the client attribution line
        "message", "id", "from", "subject", "reply", "re", "related", "links",
        "on", "wrote",
        # ticket JSON
        "requester", "crm", "created", "severity", "sev", "tier", "sla", "due",
        "status", "history", "assignee", "linked", "issues", "calls",
        "comments", "number", "body", "open", "pending", "solved",
        # chat JSON and directories
        "user", "text", "ts", "thread", "displayname", "content",
        "createddatetime", "replytoid", "name", "real", "is", "bot", "private",
        "false", "true", "null", "general", "ext",
        # CRM note JSON
        "contact",
        # tiers (models.AccountTier)
        "enterprise", "mid", "market", "smb",
    }
)  # fmt: skip

_TAG_FACT = "[SLOT {slot_id} fact={fact_id} type={fact_type} directive={directive}]"
_TAG_DECOY = "[SLOT {slot_id} decoy={decoy_id} class={decoy_class} near={near_type}]"
_TAG_COLOR = "[SLOT {slot_id} color words={words}]"
_TAG_SUBJECT = "[SLOT {slot_id} subject]"
_TAG_QUOTE = "[QUOTE {message_id}]"


# --------------------------------------------------------------------------
# Content units (facts and decoys, in the order they are stated)
# --------------------------------------------------------------------------


class _Unit:
    __slots__ = ("decoy", "fact", "on", "speaker_id")

    def __init__(
        self,
        fact: Fact | None,
        decoy: PlannedDecoy | None,
        speaker_id: str | None,
        on: date,
    ):
        self.fact = fact
        self.decoy = decoy
        self.speaker_id = speaker_id
        self.on = on


def _content_units(
    plan: DocPlan, world: World, decoys_by_id: dict[str, PlannedDecoy], rng: random.Random
) -> list[_Unit]:
    fact_by_id = {f.id: f for f in world.facts}
    people = world.person_by_id()
    units = [
        _Unit(fact_by_id[fid], None, fact_by_id[fid].speaker, fact_by_id[fid].date)
        for fid in plan.fact_ids
    ]
    for did in plan.decoy_ids:
        decoy = decoys_by_id[did]
        speaker = decoy.speaker_person_id if decoy.speaker_person_id in people else None
        pos = rng.randrange(len(units) + 1)
        on = units[pos - 1].on if pos > 0 else plan.date
        units.insert(pos, _Unit(None, decoy, speaker, max(on, plan.date)))
    return units


def _unit_tag(unit: _Unit, slot_id: str, pack: VerticalPack) -> str:
    if unit.fact is not None:
        return _TAG_FACT.format(
            slot_id=slot_id,
            fact_id=unit.fact.id,
            fact_type=unit.fact.fact_type.value,
            directive=unit.fact.directive,
        )
    assert unit.decoy is not None
    # Fails loudly on a decoy class the pack does not define; never copies
    # the class's `description`/`example` text into the skeleton.
    decoy_class: DecoyClass = next(
        c for c in pack.decoy_classes if c.id == unit.decoy.decoy_class_id
    )
    return _TAG_DECOY.format(
        slot_id=slot_id,
        decoy_id=unit.decoy.id,
        decoy_class=decoy_class.id,
        near_type=unit.decoy.near_type.value,
    )


def _unit_slot(
    unit: _Unit,
    plan: DocPlan,
    slot_id: str,
    text: str,
    locator: str,
    file: str,
    target_words: int,
    on: date,
) -> Slot:
    return Slot(
        slot_id=slot_id,
        doc_id=plan.id,
        kind=SlotKind.FACT if unit.fact is not None else SlotKind.DECOY,
        fact_id=unit.fact.id if unit.fact is not None else None,
        decoy_id=unit.decoy.id if unit.decoy is not None else None,
        speaker_person_id=unit.speaker_id,
        locator=locator,
        files=[file],
        placeholder_text=text,
        target_words=target_words,
        unit_date=on,
    )


def _color_slot(
    plan: DocPlan,
    slot_id: str,
    words: int,
    speaker_id: str | None,
    locator: str | None,
    file: str,
    on: date,
) -> tuple[str, Slot]:
    text = _TAG_COLOR.format(slot_id=slot_id, words=words)
    return text, Slot(
        slot_id=slot_id,
        doc_id=plan.id,
        kind=SlotKind.COLOR,
        speaker_person_id=speaker_id,
        locator=locator,
        files=[file],
        placeholder_text=text,
        target_words=words,
        unit_date=on,
    )


def _subject_slot(plan: DocPlan, files: list[str]) -> tuple[str, Slot]:
    slot_id = f"{plan.id}-subject"
    text = _TAG_SUBJECT.format(slot_id=slot_id)
    return text, Slot(
        slot_id=slot_id,
        doc_id=plan.id,
        kind=SlotKind.SUBJECT,
        files=files,
        placeholder_text=text,
        unit_date=plan.date,
    )


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------


def _rng(plan: DocPlan, world: World) -> random.Random:
    return random.Random(f"wp3-render:{world.id}:{plan.id}")


def _spread_positions(rng: random.Random, n_slots: int, n_units: int, lo: int) -> list[int]:
    """`n_units` sorted distinct positions in `[lo, n_slots)`, spread
    across the document rather than bunched."""
    pool = list(range(min(lo, n_slots - n_units), n_slots))
    return sorted(rng.sample(pool, n_units))


class _Cast:
    """A document's participants, split by side, with speaker picking that
    alternates sides the way a two-party conversation does."""

    def __init__(self, plan: DocPlan, world: World):
        people = world.person_by_id()
        self.people = [people[pid] for pid in plan.participants if pid in people]
        if not self.people:
            raise ValueError(f"{plan.id}: a rendered document needs at least one participant")
        self.by_id = {p.id: p for p in self.people}
        self.customer = [p for p in self.people if p.side == PersonSide.CUSTOMER]
        self.vendor = [p for p in self.people if p.side == PersonSide.VENDOR]

    def other_side(self, person_id: str | None, rng: random.Random, on: date) -> Person:
        current = self.by_id.get(person_id) if person_id else None
        if current is None:
            pool = self.customer or self.vendor
        elif current.side == PersonSide.CUSTOMER:
            pool = self.vendor or [p for p in self.customer if p.id != current.id]
        else:
            pool = self.customer or [p for p in self.vendor if p.id != current.id]
        active = [p for p in pool if _employed(p, on)] or pool or self.people
        return active[rng.randrange(len(active))]


def _employed(person: Person, on: date) -> bool:
    return person.tenure.start <= on and (person.tenure.end is None or person.tenure.end >= on)


def _org_name(person: Person, world: World) -> str:
    if person.side == PersonSide.VENDOR or not person.account_id:
        return world.vendor.name
    account = world.account_by_id().get(person.account_id)
    return account.name if account else world.vendor.name


def _attendee_line(people: list[Person], world: World) -> str:
    return "; ".join(f"{p.name} ({p.title}, {_org_name(p, world)})" for p in people)


def _account(plan: DocPlan, world: World) -> Account | None:
    return world.account_by_id().get(plan.account_id) if plan.account_id else None


def _linked_crm_id(plan: DocPlan, world: World) -> str | None:
    """The account's CRM id, only when the plan's CRM link is wired (the
    ~10% missing-link noise drops it from the document too, so "merge
    without links" is really tested)."""
    account = _account(plan, world)
    if account is None or not plan.links.get("crm_account"):
        return None
    return account.crm_id


def _color_words(rng: random.Random, short: bool = False) -> int:
    if short:
        return rng.choice([4, 6, 8, 10, 12, 15, 18])
    r = rng.random()
    if r < 0.35:
        return rng.randint(4, 12)
    if r < 0.8:
        return rng.randint(13, 30)
    return rng.randint(31, 60)


# --------------------------------------------------------------------------
# Call transcript
# --------------------------------------------------------------------------


def render_call(
    plan: DocPlan, world: World, decoys_by_id: dict[str, PlannedDecoy], pack: VerticalPack
) -> RenderedDoc:
    rng = _rng(plan, world)
    cast = _Cast(plan, world)
    file = f"calls/{plan.id}.txt"
    units = _content_units(plan, world, decoys_by_id, rng)

    # spec section 2: 1,000 to 4,000 words, median about 1,500.
    total_words = int(min(4000, max(1000, rng.lognormvariate(7.31, 0.3))))
    unit_words = [rng.randint(25, 45) for _ in units]
    color_budget = max(0, total_words - sum(unit_words))
    color_lengths: list[int] = []
    while sum(color_lengths) < color_budget:
        color_lengths.append(_color_words(rng))
    n_turns = len(color_lengths) + len(units)
    positions = _spread_positions(rng, n_turns, len(units), lo=2)
    unit_at = dict(zip(positions, range(len(units)), strict=True))

    header = [f"Date: {plan.date.isoformat()}", f"Call type: {plan.subtype or 'check_in'}"]
    account = _account(plan, world)
    crm_id = _linked_crm_id(plan, world)
    if account is not None:
        header.append(f"Account: {account.name}" + (f" ({crm_id})" if crm_id else ""))
    header.append(f"Attendees: {_attendee_line(cast.people, world)}")

    clock = rng.randint(8, 16) * 3600 + rng.choice([0, 15, 30, 45]) * 60
    lines: list[str] = []
    slots: list[Slot] = []
    previous: str | None = None
    color_iter = iter(color_lengths)
    for turn in range(n_turns):
        ts = seconds_to_hhmmss(clock)
        slot_id = f"{plan.id}-s{turn + 1:03d}"
        if turn in unit_at:
            idx = unit_at[turn]
            unit = units[idx]
            speaker_id = unit.speaker_id if unit.speaker_id in cast.by_id else None
            if speaker_id is None:
                speaker_id = cast.other_side(previous, rng, plan.date).id
            words = unit_words[idx]
            text = _unit_tag(unit, slot_id, pack)
            slots.append(
                _unit_slot(unit, plan, slot_id, text, call_locator(ts), file, words, plan.date)
            )
        else:
            next_unit = units[unit_at[turn + 1]] if (turn + 1) in unit_at else None
            speaker = cast.other_side(previous, rng, plan.date)
            if (
                next_unit is not None
                and speaker.id == next_unit.speaker_id
                and len(cast.people) > 2
            ):
                speaker = cast.other_side(speaker.id, rng, plan.date)
            speaker_id = speaker.id
            words = next(color_iter)
            text, slot = _color_slot(
                plan, slot_id, words, speaker_id, call_locator(ts), file, plan.date
            )
            slots.append(slot)
        lines.append(f"[{ts}] {cast.by_id[speaker_id].name}: {text}")
        previous = speaker_id
        clock += max(2, round(words / 2.6)) + rng.randint(1, 3)

    content = "\n".join(header) + "\n\n" + "\n\n".join(lines) + "\n"
    return RenderedDoc(
        plan_id=plan.id, kind=plan.kind, role=plan.role, files={file: content}, slots=slots
    )


# --------------------------------------------------------------------------
# Support ticket
# --------------------------------------------------------------------------

_SLA_DAYS = {"sev1": 1, "sev2": 2, "sev3": 5, "sev4": 10}


def _severity(plan: DocPlan, world: World) -> str:
    types = {e.type.value for e in world.events if e.id in plan.event_ids}
    if "escalation" in types:
        return "sev1"
    if "incident" in types:
        return "sev2"
    if plan.role == DocRole.ROUTINE:
        return "sev4"
    return "sev3"


def render_ticket(
    plan: DocPlan, world: World, decoys_by_id: dict[str, PlannedDecoy], pack: VerticalPack
) -> RenderedDoc:
    rng = _rng(plan, world)
    cast = _Cast(plan, world)
    file = f"tickets/{plan.id}.json"
    units = _content_units(plan, world, decoys_by_id, rng)
    account = _account(plan, world)

    # spec section 2: 3 to 9 comments, ERB Jira median 6.
    n_comments = max(len(units) + 2, rng.choice([3, 4, 5, 6, 6, 6, 7, 8, 9]))
    first_speaker = cast.by_id.get(units[0].speaker_id or "") if units else None
    first_is_vendor = first_speaker is not None and first_speaker.side == PersonSide.VENDOR
    positions = _spread_positions(rng, n_comments, len(units), lo=1 if first_is_vendor else 0)
    unit_at = dict(zip(positions, range(len(units)), strict=True))
    requester = cast.customer[0] if cast.customer else cast.people[0]
    assignee = cast.vendor[0] if cast.vendor else cast.people[-1]

    comments: list[dict] = []
    slots: list[Slot] = []
    on = plan.date
    previous: str | None = None
    for i in range(n_comments):
        number = i + 1
        slot_id = f"{plan.id}-s{number:03d}"
        locator = ticket_locator(plan.id, number)
        if i in unit_at:
            unit = units[unit_at[i]]
            on = max(on, unit.on)
            author_id = unit.speaker_id if unit.speaker_id in cast.by_id else requester.id
            words = rng.randint(30, 70)
            body = _unit_tag(unit, slot_id, pack)
            slots.append(_unit_slot(unit, plan, slot_id, body, locator, file, words, on))
        else:
            if i > 0:
                upcoming = [units[unit_at[j]].on for j in range(i + 1, n_comments) if j in unit_at]
                cap = min([*upcoming, world.window_end])
                on = min(on + timedelta(days=rng.randint(0, 2)), max(on, cap))
            author_id = requester.id if i == 0 else cast.other_side(previous, rng, on).id
            words = rng.randint(15, 60)
            body, slot = _color_slot(plan, slot_id, words, author_id, locator, file, on)
            slots.append(slot)
        comments.append(
            {
                "number": number,
                "author": cast.by_id[author_id].name,
                "created": on.isoformat(),
                "body": body,
            }
        )
        previous = author_id

    severity = _severity(plan, world)
    last = date.fromisoformat(comments[-1]["created"])
    history = [{"status": "open", "date": plan.date.isoformat()}]
    if n_comments > 2:
        history.append({"status": "pending", "date": comments[1]["created"]})
    if (world.window_end - last).days >= 7:
        history.append({"status": "solved", "date": last.isoformat()})

    subject_text, subject_slot = _subject_slot(plan, [file])
    same_event = plan.links.get("same_event", [])
    payload = {
        "id": plan.id,
        "requester": requester.name,
        "account": account.name if account else "",
        "account_crm_id": _linked_crm_id(plan, world) or "",
        "created": plan.date.isoformat(),
        "subject": subject_text,
        "severity": severity,
        "tier": account.tier.value if account else "",
        "sla_due": (plan.date + timedelta(days=_SLA_DAYS[severity])).isoformat(),
        "status": history[-1]["status"],
        "status_history": history,
        "assignee": assignee.name,
        "linked_issues": [d for d in same_event if d.startswith("S-ticket-")],
        "related_calls": [d for d in same_event if d.startswith("S-call-")],
        "comments": comments,
    }
    return RenderedDoc(
        plan_id=plan.id,
        kind=plan.kind,
        role=plan.role,
        files={file: json.dumps(payload, indent=2) + "\n"},
        slots=[subject_slot, *slots],
    )


# --------------------------------------------------------------------------
# Email thread
# --------------------------------------------------------------------------


def _address(person: Person) -> str:
    return f"{person.name} <{person.email}>"


def _signature(person: Person, world: World) -> str:
    """Code-rendered signature block from the persona's signature style
    label (spec section 2 lists signatures as structure). Block styles use
    the RFC 3676 `-- ` delimiter, which `prepare_text` strips."""
    style = person.persona.signature
    first = person.name.split()[0]
    initials = "".join(part[0] for part in person.name.replace("-", " ").split() if part)
    if style == "full_block":
        return f"-- \n{person.name}\n{person.title}\n{_org_name(person, world)}\n{person.email}"
    if style == "full_name_title":
        return f"-- \n{person.name}\n{person.title}"
    if style == "first_name_only":
        return first
    if style == "initials_only":
        return initials
    return ""


class _Message:
    __slots__ = ("sender", "units")

    def __init__(self, sender: str, units: list[_Unit]):
        self.sender = sender
        self.units = units


def _plan_messages(units: list[_Unit], cast: _Cast, rng: random.Random, on: date) -> list[_Message]:
    messages: list[_Message] = []
    for unit in units:
        sender = unit.speaker_id if unit.speaker_id in cast.by_id else None
        if sender is None:
            sender = cast.other_side(messages[-1].sender if messages else None, rng, unit.on).id
        if messages and messages[-1].sender == sender and len(messages[-1].units) < 2:
            messages[-1].units.append(unit)
            continue
        if messages and messages[-1].sender == sender and len(messages) < 6:
            messages.append(_Message(cast.other_side(sender, rng, unit.on).id, []))
        messages.append(_Message(sender, [unit]))
    if not messages:
        messages.append(_Message(cast.other_side(None, rng, on).id, []))
    # spec section 2: 2 to 7 messages.
    extra = rng.choice([1, 1, 1, 2, 2, 3])
    while len(messages) < 7 and (len(messages) < 2 or extra > 0):
        messages.append(_Message(cast.other_side(messages[-1].sender, rng, on).id, []))
        extra -= 1
    return messages


def render_email(
    plan: DocPlan, world: World, decoys_by_id: dict[str, PlannedDecoy], pack: VerticalPack
) -> RenderedDoc:
    rng = _rng(plan, world)
    cast = _Cast(plan, world)
    domain = world.vendor.message_id_domain
    units = _content_units(plan, world, decoys_by_id, rng)
    messages = _plan_messages(units, cast, rng, plan.date)
    msg_ids = [f"{plan.id}-m{i + 1}" for i in range(len(messages))]
    files_by_msg = {mid: f"email/{mid}.txt" for mid in msg_ids}
    subject_text, subject_slot = _subject_slot(plan, list(files_by_msg.values()))
    quoting = rng.random() < 0.4  # spec section 2: quoted replies in about 40% of threads
    crm_id = _linked_crm_id(plan, world)
    related = plan.links.get("same_event", [])

    moment = datetime.combine(plan.date, datetime.min.time()) + timedelta(
        hours=rng.randint(8, 16), minutes=rng.randint(0, 59)
    )
    last_moment = datetime.combine(world.window_end, datetime.min.time()) + timedelta(hours=23)
    files: dict[str, str] = {}
    slots: list[Slot] = [subject_slot]
    quote_ops: list[QuoteOp] = []
    counter = 0
    for i, message in enumerate(messages):
        msg_id = msg_ids[i]
        file = files_by_msg[msg_id]
        if i > 0:
            moment += timedelta(minutes=rng.randint(20, 600))
            if moment.hour >= 19 or moment.hour < 7:
                # Replies land in working hours: roll to the next morning.
                next_day = moment.date() + timedelta(days=1 if moment.hour >= 19 else 0)
                moment = datetime.combine(next_day, datetime.min.time()) + timedelta(
                    hours=rng.randint(7, 10), minutes=rng.randint(0, 59)
                )
        for unit in message.units:
            if moment.date() < unit.on:
                moment = datetime.combine(unit.on, datetime.min.time()) + timedelta(
                    hours=rng.randint(8, 16), minutes=rng.randint(0, 59)
                )
        moment = min(moment, last_moment + timedelta(minutes=i))
        sender = cast.by_id[message.sender]
        recipients = (
            [p for p in cast.people if p.side != sender.side and _employed(p, moment.date())]
            or [p for p in cast.people if p.id != sender.id]
            or [sender]
        )

        paragraphs: list[str] = []
        # Paragraph plan: an optional colour opener, the unit paragraphs,
        # an optional colour closer. `None` marks a colour paragraph.
        layout: list[_Unit | None] = []
        if not message.units or rng.random() < 0.4:
            layout.append(None)
        layout.extend(message.units)
        if message.units and rng.random() < 0.25:
            layout.append(None)
        for entry in layout:
            counter += 1
            slot_id = f"{plan.id}-s{counter:03d}"
            locator = f"{msg_id}@{domain}#p{len(paragraphs) + 1}"
            if entry is None:
                words = _color_words(rng, short=True) + 6
                text, slot = _color_slot(
                    plan, slot_id, words, sender.id, locator, file, moment.date()
                )
            else:
                text = _unit_tag(entry, slot_id, pack)
                slot = _unit_slot(
                    entry, plan, slot_id, text, locator, file, rng.randint(25, 60), moment.date()
                )
            paragraphs.append(text)
            slots.append(slot)

        headers = [
            f"Message-ID: <{msg_id}@{domain}>",
            f"From: {_address(sender)}",
            f"To: {', '.join(_address(p) for p in recipients)}",
            f"Date: {moment.strftime('%Y-%m-%d %H:%M:%S')} -0700",
            f"Subject: {subject_text}" if i == 0 else f"Subject: RE: {subject_text}",
        ]
        if i > 0:
            headers.append(f"In-Reply-To: <{msg_ids[i - 1]}@{domain}>")
        if crm_id:
            headers.append(f"X-Related-Account: {crm_id}")
        if related:
            headers.append(f"X-Related-Links: {', '.join(related)}")

        body = "\n\n".join(paragraphs)
        signature = _signature(sender, world)
        if signature:
            body += f"\n\n{signature}"
        if quoting and i > 0:
            previous = cast.by_id[messages[i - 1].sender]
            placeholder = _TAG_QUOTE.format(message_id=msg_ids[i - 1])
            body += (
                f"\n\nOn {moment.date().isoformat()}, {_address(previous)} wrote:\n{placeholder}"
            )
            quote_ops.append(
                QuoteOp(
                    file=file,
                    placeholder_text=placeholder,
                    quoted_file=files_by_msg[msg_ids[i - 1]],
                    quoted_message_id=msg_ids[i - 1],
                )
            )
        files[file] = "\n".join(headers) + "\n\n" + body + "\n"

    return RenderedDoc(
        plan_id=plan.id,
        kind=plan.kind,
        role=plan.role,
        files=files,
        slots=slots,
        quote_ops=quote_ops,
    )


# --------------------------------------------------------------------------
# Slack / Teams channel-day
# --------------------------------------------------------------------------


def slack_user_id(person: Person) -> str:
    """Slack exports carry a user id, not a name, in `user`; the name is in
    `slack/users.json` (`render_chat_directories`)."""
    return "U" + hashlib.sha256(person.id.encode()).hexdigest()[:8].upper()


def render_chat(
    plan: DocPlan, world: World, decoys_by_id: dict[str, PlannedDecoy], pack: VerticalPack
) -> RenderedDoc:
    rng = _rng(plan, world)
    cast = _Cast(plan, world)
    is_teams = plan.kind == DocKind.TEAMS
    channel = plan.channel or "general"
    day = plan.date.isoformat()
    path = (teams_channel_path if is_teams else slack_channel_path)(channel, day)
    locate = teams_locator if is_teams else slack_locator
    units = _content_units(plan, world, decoys_by_id, rng)

    n_messages = len(units) + rng.randint(2, 7)
    positions = _spread_positions(rng, n_messages, len(units), lo=0)
    unit_at = dict(zip(positions, range(len(units)), strict=True))
    seconds = rng.randint(8, 11) * 3600 + rng.randint(0, 3599)

    messages: list[dict] = []
    slots: list[Slot] = []
    previous: str | None = None
    last_unit_message: int | None = None
    for i in range(n_messages):
        index = i + 1
        slot_id = f"{plan.id}-s{index:03d}"
        locator = locate(channel, day, index)
        if i in unit_at:
            unit = units[unit_at[i]]
            user_id = unit.speaker_id if unit.speaker_id in cast.by_id else None
            if user_id is None:
                user_id = cast.other_side(previous, rng, plan.date).id
            text = _unit_tag(unit, slot_id, pack)
            slots.append(
                _unit_slot(unit, plan, slot_id, text, locator, path, rng.randint(12, 40), plan.date)
            )
        else:
            user_id = cast.other_side(previous, rng, plan.date).id
            text, slot = _color_slot(
                plan, slot_id, _color_words(rng, short=True), user_id, locator, path, plan.date
            )
            slots.append(slot)
        # Threads: a reply right after a fact message sometimes goes in its
        # thread (spec section 2: "threads, handles, reactions").
        parent = (
            last_unit_message
            if last_unit_message is not None and i == last_unit_message + 1 and rng.random() < 0.4
            else None
        )
        person = cast.by_id[user_id]
        if is_teams:
            messages.append(
                {
                    "id": f"{plan.id}-m{index}",
                    "from": {"user": {"displayName": person.name}},
                    "body": {"content": text},
                    "createdDateTime": teams_created_datetime(day, seconds),
                    "replyToId": messages[parent]["id"] if parent is not None else None,
                }
            )
        else:
            messages.append(
                {
                    "type": "message",
                    "user": slack_user_id(person),
                    "text": text,
                    "ts": slack_ts(day, seconds),
                    "thread_ts": messages[parent]["ts"] if parent is not None else None,
                }
            )
        if i in unit_at:
            last_unit_message = i
        previous = user_id
        seconds = min(seconds + rng.randint(40, 3000), 23 * 3600 + i)

    return RenderedDoc(
        plan_id=plan.id,
        kind=plan.kind,
        role=plan.role,
        files={path: json.dumps(messages, indent=2) + "\n"},
        slots=slots,
    )


def render_chat_directories(world: World, doc_plan: WorldDocPlan) -> dict[str, str]:
    """`slack/users.json` and `slack/channels.json`, the shape v1's
    `structured.write_slack_users`/`write_slack_channels` write."""
    users = [
        {
            "id": slack_user_id(p),
            "name": p.chat_handle.lstrip("@"),
            "real_name": p.name,
            "is_bot": False,
        }
        for p in world.people
    ]
    accounts = world.account_by_id()
    channels: dict[str, dict] = {}
    for plan in doc_plan.doc_plans:
        if plan.kind != DocKind.SLACK or not plan.channel or plan.channel in channels:
            continue
        account = accounts.get(plan.account_id) if plan.account_id else None
        channels[plan.channel] = {
            "id": "C" + hashlib.sha256(plan.channel.encode()).hexdigest()[:8].upper(),
            "name": plan.channel,
            "is_private": False,
            "account": account.crm_id if account else None,
        }
    return {
        "slack/users.json": json.dumps(users, indent=2) + "\n",
        "slack/channels.json": json.dumps(
            sorted(channels.values(), key=lambda c: c["name"]), indent=2
        )
        + "\n",
    }


# --------------------------------------------------------------------------
# CRM activity note, meeting notes, doc
# --------------------------------------------------------------------------


def _paragraph_units(
    plan: DocPlan,
    units: list[_Unit],
    rng: random.Random,
    n_color: int,
    locator_for,
    file: str,
    pack: VerticalPack,
    author_id: str | None,
    unit_words: tuple[int, int],
    color_words: tuple[int, int],
) -> tuple[list[str], list[Slot]]:
    n = len(units) + n_color
    positions = _spread_positions(rng, n, len(units), lo=0)
    unit_at = dict(zip(positions, range(len(units)), strict=True))
    paragraphs: list[str] = []
    slots: list[Slot] = []
    for i in range(n):
        slot_id = f"{plan.id}-s{i + 1:03d}"
        locator = locator_for(i + 1)
        if i in unit_at:
            unit = units[unit_at[i]]
            text = _unit_tag(unit, slot_id, pack)
            slots.append(
                _unit_slot(
                    unit, plan, slot_id, text, locator, file, rng.randint(*unit_words), plan.date
                )
            )
        else:
            text, slot = _color_slot(
                plan, slot_id, rng.randint(*color_words), author_id, locator, file, plan.date
            )
            slots.append(slot)
        paragraphs.append(text)
    return paragraphs, slots


def render_crm_note(
    plan: DocPlan,
    world: World,
    decoys_by_id: dict[str, PlannedDecoy],
    pack: VerticalPack,
    contact_crm_ids: dict[str, str],
) -> RenderedDoc:
    """v1's activity-note JSON (`structured.write_activity_note`), body in
    paragraphs with the `crm` locator `crm/activity_notes/<id>.json#p<n>`
    v1 already uses. Spec section 2: "2 to 6 lines in a rep's register"."""
    rng = _rng(plan, world)
    cast = _Cast(plan, world)
    relpath = f"crm/activity_notes/{plan.id}.json"
    units = _content_units(plan, world, decoys_by_id, rng)
    author = cast.vendor[0] if cast.vendor else cast.people[0]
    n_color = rng.randint(1, 2) if units else rng.randint(2, 4)
    paragraphs, slots = _paragraph_units(
        plan,
        units,
        rng,
        n_color,
        lambda p: f"{relpath}#p{p}",
        relpath,
        pack,
        author.id,
        unit_words=(12, 35),
        color_words=(5, 20),
    )
    subject_text, subject_slot = _subject_slot(plan, [relpath])
    account = _account(plan, world)
    contact = cast.customer[0] if cast.customer else None
    payload = {
        "id": plan.id,
        # A CRM note lives inside the CRM, so it always carries its account.
        "account_crm_id": account.crm_id if account else "",
        "contact_crm_id": contact_crm_ids.get(contact.id) if contact else None,
        "author": author.name,
        "created_date": plan.date.isoformat(),
        "subject": subject_text,
        "body": "\n\n".join(paragraphs),
    }
    return RenderedDoc(
        plan_id=plan.id,
        kind=plan.kind,
        role=plan.role,
        files={relpath: json.dumps(payload, indent=2) + "\n"},
        slots=[subject_slot, *slots],
    )


def render_meeting_notes_or_doc(
    plan: DocPlan, world: World, decoys_by_id: dict[str, PlannedDecoy], pack: VerticalPack
) -> RenderedDoc:
    """Plain-text container with a structural header and one paragraph per
    unit. Locator `<dir>/<id>.md#p<n>` (spec section 9: `manifest.
    LOCATOR_PATTERNS` gains meeting_notes and doc kinds in WP4)."""
    rng = _rng(plan, world)
    cast = _Cast(plan, world)
    is_doc = plan.kind == DocKind.DOC
    relpath = f"{'docs' if is_doc else 'meeting_notes'}/{plan.id}.md"
    units = _content_units(plan, world, decoys_by_id, rng)
    author = cast.vendor[0] if cast.vendor else cast.people[0]
    n_color = rng.randint(2, 4) if is_doc else rng.randint(1, 3)
    paragraphs, slots = _paragraph_units(
        plan,
        units,
        rng,
        n_color,
        lambda p: f"{relpath}#p{p}",
        relpath,
        pack,
        author.id,
        unit_words=(25, 60),
        color_words=(20, 80),
    )
    account = _account(plan, world)
    crm_id = _linked_crm_id(plan, world)
    header = [f"Date: {plan.date.isoformat()}"]
    header.append(f"Document type: {plan.subtype}" if is_doc else f"Meeting: {plan.subtype}")
    if account is not None:
        header.append(f"Account: {account.name}" + (f" ({crm_id})" if crm_id else ""))
    header.append(
        f"Author: {author.name}" if is_doc else f"Attendees: {_attendee_line(cast.people, world)}"
    )
    extra: list[Slot] = []
    if is_doc:
        title_text, title_slot = _subject_slot(plan, [relpath])
        header.append(f"Title: {title_text}")
        extra.append(title_slot)
    content = "\n".join(header) + "\n\n" + "\n\n".join(paragraphs) + "\n"
    return RenderedDoc(
        plan_id=plan.id,
        kind=plan.kind,
        role=plan.role,
        files={relpath: content},
        slots=[*extra, *slots],
    )


# --------------------------------------------------------------------------
# Dispatcher
# --------------------------------------------------------------------------


def render(
    plan: DocPlan,
    world: World,
    doc_plan: WorldDocPlan,
    pack: VerticalPack,
) -> RenderedDoc:
    decoys_by_id = {d.id: d for d in doc_plan.decoys}
    if plan.kind == DocKind.CALL:
        return render_call(plan, world, decoys_by_id, pack)
    if plan.kind == DocKind.TICKET:
        return render_ticket(plan, world, decoys_by_id, pack)
    if plan.kind == DocKind.EMAIL:
        return render_email(plan, world, decoys_by_id, pack)
    if plan.kind in (DocKind.SLACK, DocKind.TEAMS):
        return render_chat(plan, world, decoys_by_id, pack)
    if plan.kind == DocKind.CRM_NOTE:
        return render_crm_note(plan, world, decoys_by_id, pack, doc_plan.contact_crm_ids)
    return render_meeting_notes_or_doc(plan, world, decoys_by_id, pack)


# --------------------------------------------------------------------------
# Materialization
# --------------------------------------------------------------------------


def materialize_rendered_doc(doc: RenderedDoc, out_dir: Path) -> list[Path]:
    written: list[Path] = []
    for relative_path, content in doc.files.items():
        path = out_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        written.append(path)
    return written


def materialize_crm_accounts(records: list[CrmAccountRecordPlan], out_dir: Path) -> Path:
    """Writes the code-rendered CRM account records to `crm/accounts.csv`
    in `CRM_ACCOUNT_FIELDS` order -- the shape `FileConnector` reads and
    `planner.render_crm_row_text` computes label spans against."""
    path = out_dir / CRM_ACCOUNTS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(CRM_ACCOUNT_FIELDS))
        writer.writeheader()
        for record in records:
            writer.writerow(record.row)
    return path
