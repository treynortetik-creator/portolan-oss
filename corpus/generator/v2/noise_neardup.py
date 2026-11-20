"""WP6: near-duplicate documents (`01-spec/world-generator-v2-2026-09-22.md`
section 5: "Near-duplicate, one fact changed ... Writer call: new version of
a document (revised quote email, updated ticket summary) with one value
changed. Both labeled; the new fact `supersedes` the old").

WP3 plans each near-duplicate as a later document of the same kind as its
original (`DocRole.NEAR_DUPLICATE`, `links["shadows"]`). Code picks the
original (the planned shadow when WP4 wrote it with a changeable fact, else
another written document of the same kind), picks one labeled fact whose
claim carries a date, a percentage or a count that its span states, and
computes the new value. The new fact's claim is the old claim with that one
value replaced; its `supersedes` names the old fact.

An open-weight writer from WP4's allowlist then writes the new version from
the original's own words, every labeled statement wrapped in its marker, in
the kind's own output protocol. The reply goes through WP4's skeleton
(`renderers.render` for the near-duplicate's plan), parse, assembly,
preparation and checks, plus three of our own: the changed statement's span
carries the new value (n1) and not the old one (n2), and every sentence of
the new version is a light edit of a sentence of the original (n3), since
an added sentence can state an unscripted fact. The words of the new
version all come from the writer; code only picks what changes.

Spend goes on WP4's `RunBudget` (corpus key `synthetic_v2`, the run's hard
cap checked before every call).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import random
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Literal

from corpus.generator.markers import strip_all_markers
from corpus.generator.v2.assembly import (
    BLOCK_KINDS,
    LINE_KINDS,
    assemble,
    parse_draft,
    prepare_sources,
    skeleton_requests_subject,
)
from corpus.generator.v2.models import Fact, World
from corpus.generator.v2.noise import (
    OP_NEAR_DUPLICATE,
    NoisedDecoySpan,
    NoisedDoc,
    NoisedLabel,
    NoiseOp,
)
from corpus.generator.v2.noise_text import (
    CHAT_KINDS,
    ParseError,
    assign_original_keys,
    key_index,
    locate,
    parse_written,
)
from corpus.generator.v2.packs.schema import VerticalPack
from corpus.generator.v2.planner import DocKind, DocPlan, DocRole, PlannedDecoy, WorldDocPlan
from corpus.generator.v2.renderers import render
from corpus.generator.v2.verify import (
    CHECKS_VERSION,
    DocVerification,
    SlotSpec,
    copied_elsewhere,
    coverage_extras,
    make_labels,
    verify_document,
)
from corpus.generator.v2.voice import natural_date_options
from corpus.generator.v2.writer import (
    CallRecord,
    PreparedSourceRecord,
    SlotOutcome,
    WriterBudgetExceeded,
    WriterCallError,
    WriterCapExhausted,
    WriterSession,
    WrittenDoc,
    _call,
    build_cast,
    slot_specs_for,
)
from corpus.generator.v2.writer_prompt import format_spec
from portolan.extract.schema import FactType
from portolan.training.overlap_gate import numerals_and_names_covered

__all__ = [
    "NEARDUP_PROMPT_VERSION",
    "NearDupConfig",
    "NearDupResult",
    "ValueChange",
    "added_sentences",
    "build_neardup_prompt",
    "choose_change",
    "value_present",
    "write_near_duplicates",
]

NEARDUP_PROMPT_VERSION = "wp6-neardup-2026-09-23.4"
# A sentence of the new version (six words or more) must be a light edit of
# some sentence of the original: difflib ratio at least this. A lower best
# match is an added sentence, which can state an unscripted fact
# (`test_added_sentence_is_caught` holds the live example).
ADDED_SENTENCE_RATIO = 0.6
LABEL_SOURCE_NEAR_DUPLICATE = "near_duplicate_writer"

# Fact types where a changed value is a real revision (a moved due date, a
# revised price), preferred when choosing which fact changes.
_PREFERRED_TYPES = (FactType.COMMITMENT, FactType.PRICING, FactType.DECISION, FactType.RISK)

_WHAT: dict[DocKind, str] = {
    DocKind.TICKET: "a later copy of this support ticket, with its comments revised",
    DocKind.EMAIL: "a revised version of this email thread",
    DocKind.CRM_NOTE: "a revised version of this CRM activity note",
    DocKind.DOC: "a revised version of this document",
    DocKind.MEETING_NOTES: "a revised version of these meeting notes",
    DocKind.SLACK: "a later chat day that goes over the same points again",
    DocKind.TEAMS: "a later chat day that goes over the same points again",
    DocKind.CALL: "a later call that goes over the same points again",
}

_ISO_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
_PERCENT_RE = re.compile(r"(?<![\d.])(\d{1,3})%")
_COUNT_RE = re.compile(
    r"(?<![\d.\-])(\d{1,3})(?=\s+(?:business\s+|working\s+|calendar\s+)?"
    r"(days?|weeks?|months?|hours?|units?|beds?|sites?|seats?|licenses?|users?)\b)"
)


@dataclass
class NearDupConfig:
    seed: int
    limit: int | None = None  # at most this many near-duplicates attempted per run
    temperature: float = 0.5
    max_model_switches: int = 3


@dataclass(frozen=True)
class ValueChange:
    fact_id: str
    new_fact_id: str
    value_kind: Literal["date", "percent", "count"]
    old_value: str  # as the claim writes it: "2026-01-28", "15%", "7"
    new_value: str
    old_human: str
    new_human: str
    new_claim: str


@dataclass
class NearDupResult:
    docs: dict[str, NoisedDoc]
    new_facts: list[Fact]
    supersessions: list[dict[str, Any]]
    report: dict[str, Any]


# --------------------------------------------------------------------------
# Choosing what changes
# --------------------------------------------------------------------------


def _human_date(d: date) -> str:
    return f"{d.strftime('%B')} {d.day}, {d.year}"


def value_present(kind: str, value: str, text: str, source_date: date) -> bool:
    """Is `value` stated in `text`, in digits or a spelled or relative form
    the numeral rule accepts (`verify.coverage_extras`)? The source date
    itself is not counted as stated."""
    extras = coverage_extras(text, source_date).split()[1:]
    if kind == "date":
        return value in text or value in extras
    ok, _names = numerals_and_names_covered(value.rstrip("%"), text + " " + " ".join(extras))
    return ok


def _find_value(claim: str, anchors: list[str]) -> tuple[str, re.Match[str]] | None:
    blocked = [(m.start(), m.end()) for a in anchors for m in re.finditer(re.escape(a), claim)]
    for kind, pattern in (("date", _ISO_RE), ("percent", _PERCENT_RE), ("count", _COUNT_RE)):
        for m in pattern.finditer(claim):
            if not any(s <= m.start() < e for s, e in blocked):
                return kind, m
    return None


def choose_change(
    fact: Fact,
    span_text: str,
    source_date: date,
    rng: random.Random,
    *,
    latest: date | None = None,
) -> ValueChange | None:
    """The one value of `fact` that changes, when its claim carries a date,
    a percentage or a count that the labeled span actually states. A changed
    date other than a commitment's due date stays on or before `latest` (the
    new version's own date), so it never states an event in its future."""
    found = _find_value(fact.claim, fact.anchors)
    if found is None:
        return None
    kind, m = found
    old = m.group(1)
    if not value_present(kind, old if kind != "percent" else f"{old}%", span_text, source_date):
        return None
    if kind == "date":
        old_date = date.fromisoformat(old)
        delta = timedelta(days=rng.randint(7, 28))
        new_date = old_date + delta
        if latest is not None and fact.fact_type != FactType.COMMITMENT and new_date > latest:
            new_date = old_date - delta
        new_value, old_value = new_date.isoformat(), old
        old_human, new_human = _human_date(old_date), _human_date(new_date)
    elif kind == "percent":
        n = int(old)
        new_n = min(95, max(5, n + rng.choice((-15, -10, -5, 5, 10, 15))))
        if new_n == n:
            new_n = n + 5 if n <= 90 else n - 5
        old_value, new_value = f"{n}%", f"{new_n}%"
        old_human, new_human = old_value, new_value
    else:
        n = int(old)
        new_n = max(1, n + rng.choice((-3, -2, 2, 3, 5)))
        if new_n == n:
            new_n = n + 2
        unit = m.group(2)
        old_value, new_value = str(n), str(new_n)
        old_human, new_human = f"{n} {unit}", f"{new_n} {unit}"
    start, end = m.span(1) if kind != "percent" else m.span(0)
    new_claim = fact.claim[:start] + new_value + fact.claim[end:]
    return ValueChange(
        fact_id=fact.id,
        new_fact_id=f"{fact.id}s",
        value_kind=kind,
        old_value=old_value,
        new_value=new_value,
        old_human=old_human,
        new_human=new_human,
        new_claim=new_claim,
    )


def superseding_fact(old: Fact, change: ValueChange, when: date) -> Fact:
    """The new fact. It is dated on its changed date when that is a past
    event's date, else on the new version's date `when`."""
    on = when
    if change.value_kind == "date" and old.date.isoformat() == change.old_value:
        on = min(date.fromisoformat(change.new_value), when)
    return old.model_copy(
        update={
            "id": change.new_fact_id,
            "claim": change.new_claim,
            "date": on,
            "valid_from": when,
            "valid_to": None,
            "supersedes": old.id,
        }
    )


def _rank(seed: int, *parts: object) -> str:
    return hashlib.sha256(":".join(map(str, (seed, *parts))).encode()).hexdigest()


@dataclass(frozen=True)
class Pick:
    shadow: WrittenDoc
    change: ValueChange
    replaced: bool
    date: date


def pick_shadow(
    plan: DocPlan,
    docs: dict[str, WrittenDoc],
    world: World,
    world_plan: WorldDocPlan,
    used: set[str],
    seed: int,
) -> Pick | None:
    """The planned shadow when WP4 wrote it with a changeable fact, else the
    best written document of the same kind (the near-duplicate's kind is
    fixed by WP3's plan). A replacement is dated 1 to 14 days after it, as
    WP3 dates a planned one; a chat day is never moved onto a channel day
    another document already holds."""
    plans = world_plan.doc_by_id()
    facts = {f.id: f for f in world.facts}
    rng = random.Random(f"wp6-neardup:{seed}:{plan.id}")
    taken = {(p.channel, p.date) for p in world_plan.doc_plans if p.channel is not None}

    def change_for(doc: WrittenDoc, on: date) -> ValueChange | None:
        doc_plan = plans[doc.plan_id]
        labels = sorted(
            (x for x in doc.labels if x.fact_id in facts),
            key=lambda x: (facts[x.fact_id].fact_type not in _PREFERRED_TYPES, x.fact_id),
        )
        for label in labels:
            change_rng = random.Random(f"wp6-neardup:{seed}:{plan.id}:{label.fact_id}")
            for source_date in dict.fromkeys([doc_plan.date, label.date]):
                change = choose_change(
                    facts[label.fact_id], label.text, source_date, change_rng, latest=on
                )
                if change is not None:
                    return change
        return None

    planned = (plan.links.get("shadows") or [None])[0]
    doc = docs.get(planned) if planned else None
    if doc is not None and doc.status == "written" and planned not in used:
        change = change_for(doc, plan.date)
        if change is not None:
            return Pick(doc, change, False, plan.date)
    candidates = sorted(
        (
            d
            for d in docs.values()
            if d.status == "written"
            and d.kind == plan.kind
            and d.labels
            and d.plan_id not in used
            and d.plan_id in plans
            and plans[d.plan_id].role == DocRole.EVENT
        ),
        key=lambda d: (
            plans[d.plan_id].account_id != plan.account_id,
            _rank(seed, plan.id, d.plan_id),
        ),
    )
    for d in candidates:
        shadow_plan = plans[d.plan_id]
        on = min(shadow_plan.date + timedelta(days=1 + rng.randrange(14)), world.window_end)
        if d.kind in CHAT_KINDS and (shadow_plan.channel, on) in taken:
            continue
        change = change_for(d, on)
        if change is not None:
            return Pick(d, change, True, on)
    return None


# --------------------------------------------------------------------------
# The original, as the writer sees it
# --------------------------------------------------------------------------

_ADDRESS_RE = re.compile(r"^\s*(.*?)\s*<[^>]+>\s*$")


def _display(world: World, author: str | None) -> str:
    """A unit author as the writer protocol names people: a full name
    ("Name <address>" and bare addresses resolve to the person)."""
    if author is None:
        return ""
    m = _ADDRESS_RE.match(author)
    if m and m.group(1):
        return m.group(1)
    for p in world.people:
        if author in (p.email, p.name):
            return p.name
    return author


def marked_original(
    doc: WrittenDoc,
    doc_date: date,
    world: World,
    marker_for: dict[tuple[str, int, int], str],
) -> tuple[str, str]:
    """`(marked, plain)`: the original's own words in the kind's writer
    protocol, each labeled span wrapped in `marker_for[(doc_id, start,
    end)]` in `marked`. Code headers are left out; the skeleton renders
    them again."""
    text = parse_written(
        plan_id=doc.plan_id,
        kind=doc.kind,
        doc_date=doc_date,
        files=dict(doc.files),
        prepared=[(p.source_id, p.source_type, p.path) for p in doc.prepared_sources],
        prepared_text={p.source_id: p.text for p in doc.prepared_sources},
    )
    located = locate(text)
    assign_original_keys(text, located)
    index = key_index(text)
    inserts: dict[int, list[tuple[int, str]]] = {}
    for (source_id, start, end), marker in marker_for.items():
        hits = [index[(source_id, k)][0] for k in range(start, end) if (source_id, k) in index]
        if len(hits) != end - start:
            raise ParseError(f"span {marker} is not in the original's units")
        unit = hits[0][0]
        i0, i1 = min(i for _u, i in hits), max(i for _u, i in hits) + 1
        inserts.setdefault(unit.uid, []).extend([(i1, f"⟦/{marker}⟧"), (i0, f"⟦{marker}⟧")])
    blocks: list[str] = []
    plain: list[str] = []
    for unit in text.live_units(role="writer"):
        body = unit.text
        for pos, token in sorted(inserts.get(unit.uid, []), key=lambda p: -p[0]):
            body = body[:pos] + token + body[pos:]
        who = _display(world, unit.author)
        plain.append(unit.text)
        if doc.kind in LINE_KINDS:
            blocks.append(f"{who}: {body}")
        elif doc.kind in BLOCK_KINDS:
            blocks.append(f"### {who}\n{body}")
        else:
            blocks.append(body)
    sep = "\n" if doc.kind in LINE_KINDS else "\n\n"
    return sep.join(blocks), "\n\n".join(plain)


# --------------------------------------------------------------------------
# Prompt
# --------------------------------------------------------------------------

NEARDUP_SYSTEM = """You revise one business document for a synthetic training corpus. \
Everything is fictional: the companies, people and products are invented.

You get the ORIGINAL document and write its NEW VERSION: {what}. Exactly one thing has \
changed, and the CHANGE line says what.

RULES
- Stay close to the original: the same people, the same points in the same order, most \
sentences kept word for word or lightly edited. Add no new sections.
- Each statement wrapped in markers like ⟦F00001⟧...⟦/F00001⟧ appears in the new version exactly \
once, still wrapped in the same two markers, said by the same person, and saying the same thing. \
A marked statement stays inside one {unit}.
- Apply the CHANGE inside the statement it names; that statement no longer mentions the old \
value. Everything outside that statement stays as it was, except a date: the new version has a \
later date, so say the dates the WHEN lines give.
- Add no sentence the original does not have. Apart from the CHANGE, add no fact: no new \
numbers, dates, names, complaints, requests, risks, prices, plans or commitments.
- Name only people the original names. Never write a marker id outside its markers.

FORMAT
{format_spec}

Output only the new version, in that format. No commentary, no headings of your own, no code \
fences."""

_UNIT_WORD: dict[DocKind, str] = {
    DocKind.CALL: "line (one turn)",
    DocKind.SLACK: "message line",
    DocKind.TEAMS: "message line",
    DocKind.EMAIL: "paragraph of one message",
    DocKind.TICKET: "paragraph of one comment",
}


def _change_line(change: ValueChange) -> str:
    what = {"date": "the date", "percent": "the figure", "count": "the number"}[change.value_kind]
    return (
        f"CHANGE: in the statement marked ⟦{change.new_fact_id}⟧, {what} is now "
        f"{change.new_human} instead of {change.old_human}."
    )


def build_neardup_prompt(
    *,
    plan: DocPlan,
    original_kind: DocKind,
    original_text: str,
    original_subject: str | None,
    want_subject: bool,
    original_words: int,
    account_name: str,
    change: ValueChange,
    specs: list[SlotSpec],
    speakers: dict[str, str],
) -> tuple[str, str]:
    what = _WHAT.get(original_kind, "a revised version of this document")
    words = max(120, min(1800, original_words))
    user = [
        f"DOCUMENT: {what}, about {account_name}, dated "
        + plan.date.strftime("%A %B %d, %Y").replace(" 0", " ")
        + ".",
        f"LENGTH: about {words} words.",
    ]
    if want_subject:
        user.append("SUBJECT: requested. Start the output with a `Subject:` line.")
    user += ["", "ORIGINAL (a line that starts with a name is what that person said or wrote):"]
    if want_subject and original_subject:
        user.append(f"Subject: {original_subject}")
    user += [original_text, "", _change_line(change), "", "MARKED STATEMENTS (each exactly once)"]
    for n, spec in enumerate(specs, start=1):
        who = speakers.get(spec.marker_id)
        line = f"{n}. ⟦{spec.marker_id}⟧…⟦/{spec.marker_id}⟧" + (f", said by {who}" if who else "")
        if spec.anchors:
            line += ". Keep these names in it: " + ", ".join(f'"{a}"' for a in spec.anchors)
        line += "."
        if spec.marker_id == change.new_fact_id:
            line += " This is the statement with the CHANGE."
        user.append(line)
        when = _when(spec, change, plan)
        if when:
            user.append(f"   WHEN: {when}")
    pairs = ", ".join(f"⟦{s.marker_id}⟧…⟦/{s.marker_id}⟧" for s in specs)
    user += [
        "",
        "BEFORE YOU ANSWER, CHECK",
        f"- These marker pairs each appear exactly once: {pairs}.",
        f"- ⟦{change.new_fact_id}⟧ says {change.new_human}, not {change.old_human}.",
        "- Everything else follows the original closely; no sentence is new.",
    ]
    system = NEARDUP_SYSTEM.format(
        what=what,
        unit=_UNIT_WORD.get(original_kind, "paragraph"),
        format_spec=format_spec(original_kind),
    )
    return system, "\n".join(user)


def _when(spec: SlotSpec, change: ValueChange, plan: DocPlan) -> str | None:
    """How the new version, dated later, says the statement's date."""
    if not spec.is_fact:
        return None
    if spec.marker_id == change.new_fact_id and change.value_kind == "date":
        options = natural_date_options(date.fromisoformat(change.new_value), plan.date)
    elif (
        spec.date_options
        and spec.fact_date is not None
        and spec.claim
        and _ISO_RE.search(spec.claim)
    ):
        options = list(spec.date_options)
    else:
        return None
    return "say the date as " + " or ".join(f'"{o}"' for o in options) + "."


# --------------------------------------------------------------------------
# One near-duplicate
# --------------------------------------------------------------------------


@dataclass
class _Attempt:
    call_kind: str
    model: str
    verification: DocVerification
    assembled: Any
    prepared: list
    changed_ok: bool

    def score(self) -> tuple[int, ...]:
        v = self.verification.score()
        return (v[0], int(self.changed_ok), *v[1:])


def _shingles(text: str) -> set[tuple[str, ...]]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return {tuple(tokens[i : i + 5]) for i in range(max(0, len(tokens) - 4))}


def jaccard_5gram(a: str, b: str) -> float:
    sa, sb = _shingles(a), _shingles(b)
    return round(len(sa & sb) / len(sa | sb), 4) if sa | sb else 0.0


_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def _sentences(text: str) -> list[str]:
    out = []
    for s in _SENTENCE_RE.split(strip_all_markers(text)):
        s = re.sub(r"^(?:### |[A-Z][\w'.\-]*(?: [A-Z][\w'.\-]*){0,3}: )", "", s.strip())
        if s:
            out.append(" ".join(re.findall(r"[a-z0-9]+", s.lower())))
    return out


def added_sentences(original: str, new: str) -> list[str]:
    """Sentences of `new` (six words or more) with no close match in
    `original`."""
    pool = [s for s in _sentences(original) if s]
    added = []
    for s in _sentences(new):
        if len(s.split()) < 6:
            continue
        best = max((SequenceMatcher(None, s, o).ratio() for o in pool), default=0.0)
        if best < ADDED_SENTENCE_RATIO:
            added.append(s)
    return added


def _fingerprint(plan: DocPlan, shadow: WrittenDoc, change: ValueChange) -> str:
    payload = {
        "version": NEARDUP_PROMPT_VERSION,
        "checks": CHECKS_VERSION,
        "plan": plan.model_dump(mode="json"),
        "shadow": shadow.input_fingerprint,
        "shadow_labels": [x.model_dump(mode="json") for x in shadow.labels],
        "change": change.__dict__,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def nd_plan_for(plan: DocPlan, pick: Pick, world_plan: WorldDocPlan) -> DocPlan:
    """The near-duplicate's plan as written: the original's account, people,
    events and channel, and its facts (the changed one under its new id)."""
    shadow_plan = world_plan.doc_by_id()[pick.shadow.plan_id]
    fact_ids: list[str] = []
    for x in pick.shadow.labels:
        marker = pick.change.new_fact_id if x.fact_id == pick.change.fact_id else x.fact_id
        if marker not in fact_ids:
            fact_ids.append(marker)
    decoy_ids = list(
        dict.fromkeys(
            d.decoy_id
            for d in pick.shadow.decoy_spans
            if d.decoy_id in {p.id for p in world_plan.decoys}
        )
    )
    return plan.model_copy(
        update={
            "date": pick.date,
            "account_id": shadow_plan.account_id,
            "participants": list(shadow_plan.participants),
            "event_ids": list(shadow_plan.event_ids),
            "subtype": shadow_plan.subtype,
            "channel": shadow_plan.channel,
            "fact_ids": fact_ids,
            "decoy_ids": decoy_ids,
            "is_negative": False,
            "links": {**plan.links, "shadows": [pick.shadow.plan_id]},
        }
    )


def write_near_duplicate(
    plan: DocPlan,
    pick: Pick,
    world: World,
    world_plan: WorldDocPlan,
    pack: VerticalPack,
    *,
    session: WriterSession,
    config: NearDupConfig,
) -> tuple[NoisedDoc, Fact]:
    """Write, verify and label one near-duplicate. Raises
    `WriterBudgetExceeded` before a call that would pass the run's cap."""
    shadow, change = pick.shadow, pick.change
    facts = {f.id: f for f in world.facts}
    nd_plan = nd_plan_for(plan, pick, world_plan)
    new_fact = superseding_fact(facts[change.fact_id], change, nd_plan.date)
    world2 = world.model_copy(update={"facts": [*world.facts, new_fact]})
    decoys_by_id = {d.id: d for d in world_plan.decoys}
    fingerprint = _fingerprint(nd_plan, shadow, change)

    rendered = render(nd_plan, world2, world_plan, pack)
    # Each statement keeps the speaker who said it in the original (WP4's
    # check 4 verified that person there).
    said_by = {
        (change.new_fact_id if x.fact_id == change.fact_id else x.fact_id): x.speaker_person_id
        for x in shadow.labels
    }
    specs = [
        dataclasses.replace(s, speaker_person_id=said_by[s.marker_id])
        if said_by.get(s.marker_id)
        else s
        for s in slot_specs_for(rendered, nd_plan, world2, decoys_by_id)
    ]
    _cast, _author, people = build_cast(nd_plan, world2, specs)
    want_subject = skeleton_requests_subject(rendered)

    marker_for: dict[tuple[str, int, int], str] = {}
    for x in shadow.labels:
        marker_for[(x.doc_id, x.char_start, x.char_end)] = (
            change.new_fact_id if x.fact_id == change.fact_id else x.fact_id
        )
    for d in shadow.decoy_spans:
        if d.decoy_id in nd_plan.decoy_ids:
            marker_for[(d.doc_id, d.char_start, d.char_end)] = d.decoy_id
    shadow_plan = world_plan.doc_by_id()[shadow.plan_id]
    original, original_plain = marked_original(shadow, shadow_plan.date, world, marker_for)
    if want_subject and shadow.subject:
        # The subject line is the writer's too, and the new version keeps it.
        original_plain = f"{shadow.subject}\n\n{original_plain}"
    person = world.person_by_id()
    speakers = {
        s.marker_id: person[s.speaker_person_id].name
        for s in specs
        if nd_plan.kind in LINE_KINDS | BLOCK_KINDS and s.speaker_person_id in person
    }
    account = world.account_by_id().get(nd_plan.account_id or "")
    system, user = build_neardup_prompt(
        plan=nd_plan,
        original_kind=shadow.kind,
        original_text=original,
        original_subject=shadow.subject,
        want_subject=want_subject,
        original_words=shadow.word_count,
        account_name=account.name if account else world.vendor.name,
        change=change,
        specs=specs,
        speakers=speakers,
    )
    changed_slot = next(s.slot_id for s in specs if s.marker_id == change.new_fact_id)
    max_tokens = max(800, min(6000, int(max(120, shadow.word_count) * 1.4 * 1.6) + 400))
    calls: list[CallRecord] = []

    def evaluate(call_kind: str, model: str, raw: str) -> _Attempt:
        draft = parse_draft(nd_plan.kind, raw, people, want_subject=want_subject)
        try:
            assembled = assemble(
                plan=nd_plan, rendered=rendered, world=world2, draft=draft, people=people
            )
            prepared = prepare_sources(assembled)
        except Exception as exc:  # noqa: BLE001 -- a broken draft is a failed attempt
            v = DocVerification([], [*draft.fatal, f"could not assemble: {exc}"], [])
            return _Attempt(call_kind, model, v, None, [], False)
        v = verify_document(assembled, prepared, specs, world2, extra_fatal=list(draft.fatal))
        v.fatal.extend(
            f"n3:added_sentence ({s[:80]!r})"
            for s in added_sentences(original_plain, assembled.writer_text())
        )
        result = v.result_for(changed_slot)
        if result.found is not None:
            span = result.found.text
            on = result.found.prepared.date if result.found.prepared else nd_plan.date
            if not value_present(change.value_kind, change.new_value, span, on):
                result.failures.append("n1:new_value_not_in_span")
            if value_present(change.value_kind, change.old_value, span, on):
                result.failures.append("n2:old_value_still_in_span")
        return _Attempt(call_kind, model, v, assembled, prepared, result.passed)

    def attempt(
        call_kind: Literal["initial", "regenerate"], exclude: set[str], feedback: list[str]
    ) -> _Attempt | None:
        prompt_user = user
        if feedback:
            prompt_user += "\n\nAN EARLIER DRAFT FAILED THESE CHECKS; AVOID THEM\n" + "\n".join(
                f"- {f}" for f in feedback
            )
        tried = set(exclude)
        for _ in range(config.max_model_switches + 1):
            try:
                model = session.pool.pick(tried)
            except WriterCapExhausted:
                return None
            tried.add(model)
            try:
                raw = _call(
                    session,
                    plan=nd_plan,
                    call_kind=call_kind,
                    model=model,
                    system=system,
                    user=prompt_user,
                    max_tokens=max_tokens,
                    temperature=config.temperature,
                    calls=calls,
                )
            except WriterCallError:
                session.pool.cool(model)
                continue
            session.pool.ok(model)
            return evaluate(call_kind, model, raw)
        return None

    first = attempt("initial", set(), [])
    best = first
    if first is not None and (first.verification.fatal or not _all_facts_pass(first)):
        feedback = [
            f"{r.spec.marker_id}: " + "; ".join(f.split(":", 1)[1] for f in r.failures)
            for r in first.verification.results
            if not r.passed and r.spec.is_fact
        ] + [f"document: {f}" for f in first.verification.fatal]
        second = attempt("regenerate", {first.model}, feedback)
        if second is not None and second.score() >= first.score():
            best = second
    doc = _finish(nd_plan, pick, world2, decoys_by_id, best, first, calls, original_plain)
    doc.input_fingerprint = fingerprint
    return doc, new_fact


def _all_facts_pass(a: _Attempt) -> bool:
    return a.changed_ok and all(r.passed for r in a.verification.results if r.spec.is_fact)


def _finish(
    nd_plan: DocPlan,
    pick: Pick,
    world2: World,
    decoys_by_id: dict[str, PlannedDecoy],
    best: _Attempt | None,
    first: _Attempt | None,
    calls: list[CallRecord],
    original_plain: str,
) -> NoisedDoc:
    change = pick.change
    op = NoiseOp(
        type=OP_NEAR_DUPLICATE,
        params={
            "role": "near_duplicate",
            "shadow_id": pick.shadow.plan_id,
            "shadow_replaced": pick.replaced,
            "changed_fact_id": change.fact_id,
            "new_fact_id": change.new_fact_id,
            "value_kind": change.value_kind,
            "old_value": change.old_value,
            "new_value": change.new_value,
            "doc_date": nd_plan.date.isoformat(),
        },
    )
    base: dict[str, Any] = {
        "plan_id": nd_plan.id,
        "kind": nd_plan.kind,
        "input_fingerprint": "",
        "calls": calls,
        "noise": [op],
        "true_account_id": nd_plan.account_id,
        "filed_account_id": nd_plan.account_id,
    }
    if best is None:
        return NoisedDoc(status="error", reason="no writer produced a reply", **base)
    outcomes = []
    for r in best.verification.results:
        r0 = (
            first.verification.result_for(r.spec.slot_id)
            if first and first.verification.results
            else r
        )
        status = (
            "dropped"
            if not r.passed
            else ("regenerated" if best.call_kind == "regenerate" else "verified")
        )
        outcomes.append(
            SlotOutcome(
                slot_id=r.spec.slot_id,
                marker_id=r.spec.marker_id,
                slot_kind="fact" if r.spec.is_fact else "decoy",
                status=status,
                first_failures=list(r0.failures),
                final_failures=list(r.failures),
                dropped_text=r.found.text if (not r.passed and r.found is not None) else None,
            )
        )
    notes = list(best.verification.notes)
    notes += [f"unscripted {u}" for u in best.verification.unscripted]
    base.update(
        writer_model=best.model,
        final_call_kind=best.call_kind,
        slot_outcomes=outcomes,
        notes=notes,
        fatal=list(best.verification.fatal),
    )
    if best.verification.fatal or best.assembled is None:
        reason = "final attempt failed a document check: " + "; ".join(best.verification.fatal)
        return NoisedDoc(status="dropped", reason=reason, **base)
    if not best.changed_ok:
        return NoisedDoc(status="dropped", reason="the changed statement never verified", **base)
    copied = copied_elsewhere(best.verification, best.assembled)
    if copied:
        base["notes"] = [*base["notes"], f"unmarked text repeats the script wording of {copied}"]
    labels, decoy_spans = make_labels(
        best.verification,
        plan=nd_plan,
        world=world2,
        decoys_by_id=decoys_by_id,
        writer_model=best.model,
    )
    noised_labels = []
    for x in labels:
        data = x.model_dump()
        data.update(
            label_source=LABEL_SOURCE_NEAR_DUPLICATE,
            noise=[OP_NEAR_DUPLICATE],
            supersedes=change.fact_id if x.fact_id == change.new_fact_id else None,
        )
        noised_labels.append(NoisedLabel.model_validate(data))
    noised_decoys = [
        NoisedDecoySpan.model_validate({**d.model_dump(), "noise": [OP_NEAR_DUPLICATE]})
        for d in decoy_spans
    ]
    new_text = best.assembled.writer_text()
    op.params["jaccard_5gram"] = jaccard_5gram(original_plain, new_text)
    specs = [r.spec for r in best.verification.results]
    return NoisedDoc(
        status="written",
        subject=best.assembled.subject,
        word_count=len(new_text.split()),
        files=best.assembled.files,
        prepared_sources=[
            PreparedSourceRecord(
                source_id=p.source_id, source_type=p.source_type, path=p.path, text=p.text
            )
            for p in best.prepared
        ],
        labels=noised_labels,
        decoy_spans=noised_decoys,
        noise_counts={
            "labels_in": sum(1 for s in specs if s.is_fact),
            "labels_kept": len(noised_labels),
            "labels_dropped": sum(1 for s in specs if s.is_fact) - len(noised_labels),
            "decoys_in": sum(1 for s in specs if not s.is_fact),
            "decoys_kept": len(noised_decoys),
        },
        **base,
    )


# --------------------------------------------------------------------------
# The world's near-duplicates
# --------------------------------------------------------------------------


def _load_prior(path: Path, fingerprint: str) -> NoisedDoc | None:
    if not path.exists():
        return None
    try:
        doc = NoisedDoc.model_validate_json(path.read_text(encoding="utf-8"))
    except ValueError:
        return None
    if doc.input_fingerprint != fingerprint or doc.status == "error":
        return None
    return doc


def write_near_duplicates(
    world: World,
    world_plan: WorldDocPlan,
    pack: VerticalPack,
    docs: dict[str, WrittenDoc],
    *,
    session: WriterSession,
    config: NearDupConfig,
    out_dir: Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> NearDupResult:
    """Write every planned near-duplicate (up to `config.limit`), date
    order, reusing a finished one from `out_dir/docs/` when its inputs are
    unchanged."""
    planned = sorted(
        (p for p in world_plan.doc_plans if p.role == DocRole.NEAR_DUPLICATE),
        key=lambda p: (p.date, p.id),
    )
    out: dict[str, NoisedDoc] = {}
    new_facts: list[Fact] = []
    supersessions: list[dict[str, Any]] = []
    used: set[str] = set()
    stop_reason: str | None = None
    skipped: list[str] = []
    for plan in planned:
        if config.limit is not None and len(out) >= config.limit:
            break
        pick = pick_shadow(plan, docs, world, world_plan, used, config.seed)
        if pick is None:
            skipped.append(plan.id)
            continue
        used.add(pick.shadow.plan_id)
        nd_plan = nd_plan_for(plan, pick, world_plan)
        fingerprint = _fingerprint(nd_plan, pick.shadow, pick.change)
        prior = (
            _load_prior(out_dir / "docs" / f"{plan.id}.json", fingerprint)
            if out_dir is not None
            else None
        )
        old_fact = next(f for f in world.facts if f.id == pick.change.fact_id)
        if prior is not None:
            doc, new_fact = prior, superseding_fact(old_fact, pick.change, nd_plan.date)
        else:
            try:
                doc, new_fact = write_near_duplicate(
                    plan, pick, world, world_plan, pack, session=session, config=config
                )
            except WriterBudgetExceeded as exc:
                stop_reason = str(exc)
                break
        out[plan.id] = doc
        if doc.status == "written":
            if doc.writer_model and prior is None:
                session.pool.assign(doc.writer_model)
            new_facts.append(new_fact)
            supersessions.append(
                {
                    "old_fact_id": pick.change.fact_id,
                    "new_fact_id": pick.change.new_fact_id,
                    "old_valid_to": (nd_plan.date - timedelta(days=1)).isoformat(),
                    "near_duplicate_id": plan.id,
                    "shadow_id": pick.shadow.plan_id,
                }
            )
        if progress is not None:
            progress(
                f"{plan.id} {doc.status} shadow={pick.shadow.plan_id} change={pick.change.fact_id} "
                f"{pick.change.old_human} -> {pick.change.new_human} writer={doc.writer_model} "
                f"labels={len(doc.labels)} cost=${doc.cost_usd:.4f}"
                + (f" reason={doc.reason}" if doc.reason else "")
            )
    n_planned = sum(1 for p in world_plan.doc_plans if p.role == DocRole.NEAR_DUPLICATE)
    report = {
        "planned": n_planned,
        "planned_share_of_documents": round(n_planned / max(1, len(world_plan.doc_plans)), 4),
        "attempted": len(out),
        "by_status": {
            s: sum(1 for d in out.values() if d.status == s)
            for s in ("written", "dropped", "error")
        },
        "skipped_no_changeable_original": skipped,
        "calls": sum(len(d.calls) for d in out.values()),
        "cost_usd": round(sum(d.cost_usd for d in out.values()), 6),
        "stop_reason": stop_reason,
    }
    return NearDupResult(out, new_facts, supersessions, report)
