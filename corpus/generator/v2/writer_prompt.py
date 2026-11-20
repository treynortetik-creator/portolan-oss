"""WP4 helper: writer prompt assembly.

`01-spec/world-generator-v2-2026-09-22.md` section 3.1: "The writer prompt
... gives the format spec, the world slice at that date, participant
personas, the script, and a list of *allowed color* (scheduling, greetings,
how-to answers, weather, product walkthroughs). The rule is: state each
scripted fact once inside ⟦F…⟧…⟦/F…⟧ and each decoy inside ⟦D…⟧…⟦/D…⟧, and
state no other customer fact."

Everything in this module is an INSTRUCTION to the writer, never text that
is meant to reach a document. A fact is given as meaning, not as a sentence
to paste (`voice.fact_gist`): the two parties appear as "[the customer]" and
"[the vendor]" with a voice line saying how THIS speaker refers to each
(we/our for their own side, you for the other), the date as people say it
relative to the document (`voice.natural_date_options`), the nicknames the
speaker may use, the tone and the directive. Checks 5 and 12 reject a span
that repeats the note's wording. Decoy classes are described by the pack's
`description` only -- the pack's `example` sentence is never shown.
"""

from __future__ import annotations

import calendar
import hashlib
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from corpus.generator.v2.assembly import BLOCK_KINDS, LINE_KINDS
from corpus.generator.v2.models import EventType, Person, PersonSide, World
from corpus.generator.v2.packs.schema import DecoyClass, VerticalPack
from corpus.generator.v2.planner import DocKind, DocPlan, DocRole
from corpus.generator.v2.verify import MAX_SPAN_WORDS, SlotSpec
from corpus.generator.v2.voice import VENDOR_NOTE_KINDS, short_forms

__all__ = [
    "PROMPT_VERSION",
    "PromptContext",
    "RepairTarget",
    "build_initial_prompt",
    "build_repair_prompt",
    "format_spec",
    "length_target",
]

# Bump when prompt wording changes: part of every document's resume
# fingerprint, so a changed prompt never silently reuses an old draft.
PROMPT_VERSION = "wp4-2026-09-23.15-decoy-patterns"

FORMATS_DIR = Path(__file__).resolve().parent / "formats"

_FORMAT_FILE: dict[DocKind, str] = {
    DocKind.CALL: "call.md",
    DocKind.SLACK: "chat.md",
    DocKind.TEAMS: "chat.md",
    DocKind.EMAIL: "email.md",
    DocKind.TICKET: "ticket.md",
    DocKind.CRM_NOTE: "crm_note.md",
    DocKind.MEETING_NOTES: "meeting_notes.md",
    DocKind.DOC: "doc.md",
}


def format_spec(kind: DocKind) -> str:
    return (FORMATS_DIR / _FORMAT_FILE[kind]).read_text(encoding="utf-8").strip()


def _stable_int(*parts: object) -> int:
    return int(hashlib.sha256(":".join(str(p) for p in parts).encode()).hexdigest()[:12], 16)


# (min words, max words, min units, max units, unit noun) per kind; spec
# section 2's realism targets (a call is 1,000 to 4,000 words, median about
# 1,500; an email thread 2 to 7 messages; a ticket 3 to 9 comments; a CRM
# note 2 to 6 lines).
_LENGTHS: dict[DocKind, tuple[int, int, int, int, str]] = {
    DocKind.CALL: (1000, 2000, 40, 110, "turns"),
    DocKind.EMAIL: (140, 480, 2, 7, "messages"),
    DocKind.TICKET: (160, 520, 3, 9, "comments"),
    DocKind.SLACK: (120, 380, 8, 24, "messages"),
    DocKind.TEAMS: (120, 380, 8, 24, "messages"),
    DocKind.CRM_NOTE: (30, 110, 2, 6, "lines"),
    DocKind.MEETING_NOTES: (180, 450, 5, 14, "paragraphs"),
    DocKind.DOC: (260, 700, 5, 14, "paragraphs"),
}


def length_target(
    plan: DocPlan, n_slots: int, skeleton: tuple[int, int] | None = None
) -> tuple[int, tuple[int, int], str]:
    """Deterministic per-document `(target words, (min units, max units),
    unit noun)`. `skeleton` is WP3's plan for this document (total
    `target_words`, unit count) when it has one; the writer is held to it
    within a band."""
    lo, hi, ulo, uhi, noun = _LENGTHS[plan.kind]
    words = lo + _stable_int("len", plan.id) % (hi - lo + 1)
    if skeleton is not None and skeleton[0] > 0:
        words, units = skeleton
        ulo, uhi = max(2, round(units * 0.8)), max(2, round(units * 1.2))
    ulo = max(ulo, n_slots + (1 if plan.kind in LINE_KINDS | BLOCK_KINDS else 0))
    uhi = max(uhi, ulo)
    return words, (ulo, uhi), noun


# --------------------------------------------------------------------------
# Plain-language glosses (instructions only)
# --------------------------------------------------------------------------

DIRECTIVE_GLOSS: dict[str, str] = {
    "hedged": "hedged: the speaker softens it (I think, probably, looks like) but the fact stays clear",
    "numbers_spelled_out": "numbers spelled out: write numbers as words (fifteen percent, three days)",
    "colloquial": "colloquial: casual, conversational wording",
    "embedded_mid_sentence": "embedded mid-sentence: the fact sits in the middle of a longer sentence "
    "about something else",
    "passive": "passive: use a passive construction",
    "through_an_alias": "through a nickname: call the product or person by the nickname given, not "
    "the full name",
    "corrected_mid_sentence": "corrected mid-sentence: the speaker starts to say something else, "
    "corrects themselves, then states the fact",
    "list_item": "list item: the fact is one item in a short list (a spoken list in a call or chat, "
    "a line starting with '- ' in a written document)",
    "terse": "terse: very short, clipped wording",
    "rambling": "rambling: a long, meandering sentence (the marked part still stays under the word limit)",
    "formal": "formal: formal, precise wording",
    "via_quoted_reply": "via quoted reply: the speaker picks up something said earlier and restates it "
    "in their own new words (never inside '>' quoted lines)",
    "second_language_phrasing": "second-language phrasing: a fluent non-native speaker's wording, "
    "slightly unusual word order or article use, still clear",
    "implicit": "implicit: the fact is implied rather than spelled out, but a colleague reading it "
    "would still know it",
    "indirect": "indirect: the speaker gets at it sideways (a complaint as a question, a decision as "
    "a plan), still unmistakable",
}

_FACT_TYPE_WORDS: dict[str, str] = {
    "complaint": "a complaint",
    "request": "a request",
    "praise": "praise",
    "risk": "a risk to the relationship or deal",
    "competitor_mention": "a competitor mention",
    "commitment": "a commitment (a promise with an owner and a deadline)",
    "decision": "a decision",
    "contact_change": "a change of contact or role",
    "pricing": "a pricing point",
    "usage": "a usage fact",
    "question": "an open question",
    "other": "an account fact",
}

_FACT_TONE: dict[str, str] = {
    "complaint": "unhappy",
    "risk": "worried",
    "praise": "pleased",
    "competitor_mention": "matter-of-fact",
    "pricing": "businesslike",
}

_OFF_TOPIC_THEMES = (
    "holiday party logistics",
    "an office move and desk assignments",
    "a laptop and tooling upgrade",
    "team lunch and a birthday",
    "parking and building access changes",
    "an internal wiki clean-up",
)

_EVENT_WORDS: dict[EventType, str] = {
    EventType.GO_LIVE: "a go-live",
    EventType.RELEASE: "a product release",
    EventType.INCIDENT: "a product incident",
    EventType.ESCALATION: "an escalation",
    EventType.QBR: "a quarterly business review",
    EventType.RENEWAL_STEP: "a renewal step",
    EventType.PRICE_NEGOTIATION: "a price negotiation",
    EventType.COMPETITIVE_EVAL: "a competitive evaluation",
    EventType.CHAMPION_DEPARTURE: "a champion leaving",
    EventType.EXPANSION: "an expansion",
    EventType.COMMITMENT_MADE: "a commitment being made",
    EventType.COMMITMENT_DUE: "a commitment coming due",
    EventType.DECISION: "a decision",
    EventType.POLICY_CHANGE: "a policy change",
}

_DOC_KIND_BY_EVENT: dict[EventType, str] = {
    EventType.RENEWAL_STEP: "renewal proposal",
    EventType.PRICE_NEGOTIATION: "renewal proposal",
    EventType.RELEASE: "release notes",
    EventType.GO_LIVE: "implementation plan",
    EventType.QBR: "QBR summary",
    EventType.POLICY_CHANGE: "policy summary",
    EventType.EXPANSION: "implementation plan",
}


# --------------------------------------------------------------------------
# Prompt context
# --------------------------------------------------------------------------


@dataclass
class PromptContext:
    plan: DocPlan
    world: World
    pack: VerticalPack
    specs: list[SlotSpec]
    cast: list[Person]
    author: Person | None
    want_subject: bool
    event_type: EventType | None = None
    story_so_far: list[str] = field(default_factory=list)
    sibling_titles: list[str] = field(default_factory=list)
    feedback: list[str] = field(default_factory=list)
    # WP3's skeleton: (total target words, unit count), and its code header
    # text that reaches the extractor (a call's or doc's Account/Attendees).
    skeleton: tuple[int, int] | None = None
    structural_text: str = ""
    departed: frozenset[str] = frozenset()
    # WP5's regeneration (spec section 3.4): sentences an earlier draft
    # stated without a script line. Empty for every WP4 call.
    do_not_state: list[str] = field(default_factory=list)


def _persona_notes(p: Person, kind: DocKind) -> str:
    persona = p.persona
    bits = [f"{persona.speech_register.replace('_', ' ')} register", persona.formality]
    if persona.chat_style and kind in (DocKind.SLACK, DocKind.TEAMS):
        bits.append(f"chat style {persona.chat_style.replace('_', ' ')}")
    # Sign-off style only where a sign-off exists: a CRM note turned the
    # label into a sentence ("All parties signed off with full blocks") in
    # the 2026-09-22 smoke.
    if persona.signature and persona.signature != "none" and kind in BLOCK_KINDS:
        bits.append(f"signs off with {persona.signature.replace('_', ' ')}")
    if persona.hedging:
        bits.append("hedges often")
    if persona.typo_rate >= 0.02:
        bits.append("makes occasional typos")
    if persona.second_language_features:
        feats = ", ".join(f.replace("_", " ") for f in persona.second_language_features)
        bits.append(f"second-language English ({feats})")
    return "; ".join(bits)


def _person_line(
    p: Person, world: World, plan: DocPlan, speakers: set[str], departed: set[str]
) -> str:
    accounts = world.account_by_id()
    where = world.vendor.name if p.side == PersonSide.VENDOR else accounts[p.account_id].name
    side = "vendor" if p.side == PersonSide.VENDOR else "customer"
    extra = ""
    if plan.kind in (DocKind.SLACK, DocKind.TEAMS):
        extra = f", handle {p.chat_handle}"
    left = (p.tenure.end is not None and p.tenure.end <= plan.date) or p.id in departed
    if left and p.id not in speakers:
        return (
            f"- {p.name} ({side}): former {p.title} at {where}, has left. Mentioned only; "
            "does not speak or write."
        )
    if plan.kind not in LINE_KINDS | BLOCK_KINDS:
        return f"- {p.name} ({side}): {p.title} at {where}."
    return (
        f"- {p.name} ({side}): {p.title} at {where}{extra}. Voice: {_persona_notes(p, plan.kind)}."
    )


def _document_line(ctx: PromptContext) -> str:
    plan, world = ctx.plan, ctx.world
    account = world.account_by_id().get(plan.account_id) if plan.account_id else None
    acct = account.name if account else "the vendor's own team"
    date_str = plan.date.strftime("%A %B %d, %Y").replace(" 0", " ")
    ev = _EVENT_WORDS.get(ctx.event_type) if ctx.event_type else None
    topic = f" about {plan.topic_leaf.lower()}" if plan.topic_leaf else ""
    kind = plan.kind
    if kind == DocKind.CALL:
        what = f"a recorded call between {acct} and {world.vendor.name}"
        what += f" about {ev}" if ev else (f" (a training or how-to call{topic})" if topic else "")
    elif kind in (DocKind.SLACK, DocKind.TEAMS):
        platform = "Teams" if kind == DocKind.TEAMS else "Slack"
        what = f"one day of the shared {platform} channel #{plan.channel or 'general'} between {acct} and {world.vendor.name}"
    elif kind == DocKind.EMAIL:
        what = f"an email thread between {acct} and {world.vendor.name}"
        what += f" (routine{topic})" if topic and not ev else ""
    elif kind == DocKind.TICKET:
        what = f"a support ticket from {acct} in {world.vendor.name}'s help desk"
        what += f" (a routine how-to request{topic})" if topic and not ev else ""
    elif kind == DocKind.CRM_NOTE:
        who = ctx.author.name if ctx.author else "the account rep"
        what = f"a CRM activity note {who} wrote about {acct}"
    elif kind == DocKind.MEETING_NOTES:
        what = f"{world.vendor.name}'s internal meeting notes about {acct}"
    elif kind == DocKind.DOC:
        doc_kind = (
            _DOC_KIND_BY_EVENT.get(ctx.event_type, "working document")
            if ctx.event_type
            else "how-to guide"
        )
        what = f"a {doc_kind} from {world.vendor.name} for {acct}{topic}"
    else:
        what = f"a document about {acct}"
    if plan.role == DocRole.OFF_TOPIC:
        theme = _OFF_TOPIC_THEMES[_stable_int("theme", plan.id) % len(_OFF_TOPIC_THEMES)]
        what = f"{what.split(' between ')[0]} among {world.vendor.name}'s own staff about {theme}"
        return (
            f"DOCUMENT: {what}, dated {date_str}. It names no customer, product deal or competitor."
        )
    line = f"DOCUMENT: {what}, dated {date_str}."
    if plan.role == DocRole.HARD_NEGATIVE:
        line += " It states no fact about the customer: only the SCRIPT's near-miss statements."
    return line


def _world_lines(ctx: PromptContext) -> list[str]:
    world, plan = ctx.world, ctx.plan
    v = world.vendor

    def _product(p_name: str) -> str:
        forms = [f.text for f in short_forms(p_name, world)]
        return p_name + (f" (people say {', '.join(forms)})" if forms else "")

    products = ", ".join(_product(p.name) for p in v.products)
    lines = [f"- Vendor: {v.name}. It sells {ctx.pack.vendor_what}. Products: {products}."]
    account = world.account_by_id().get(plan.account_id) if plan.account_id else None
    if account is not None and plan.role != DocRole.OFF_TOPIC:
        parent = world.account_by_id().get(account.parent) if account.parent else None
        part_of = f", part of {parent.name}" if parent else ""
        aliases = f" (short name {', '.join(account.aliases)})" if account.aliases else ""
        lines.append(
            f"- Customer account: {account.name}{aliases}, a {account.kind.value} account in the "
            f"{account.region} region{part_of}."
        )
    needs_competitors = any(
        (s.fact_type is not None and s.fact_type.value == "competitor_mention")
        or (s.near_type is not None and s.near_type.value == "competitor_mention")
        or any(c.name in s.anchors for c in world.vendor.competitors)
        for s in ctx.specs
    )
    if needs_competitors:
        lines.append(
            "- Competitors (mention one only where the script says so): "
            + ", ".join(c.name for c in world.vendor.competitors)
            + "."
        )
    else:
        lines.append("- Do not mention any competitor.")
    return lines


def _reference_instruction(spec: SlotSpec, same: str) -> str | None:
    if spec.reference_anchor is None or spec.reference_form is None:
        return None
    return (
        f'Inside the span, refer to {spec.reference_anchor} as "{spec.reference_form}" instead of '
        f'the full name; the full name "{spec.reference_anchor}" must appear EARLIER in the same '
        f"{same} (before the span)."
    )


def _natural_day(d: date) -> str:
    return f"{calendar.day_name[d.weekday()]} {calendar.month_name[d.month]} {d.day}, {d.year}"


def _speaker_person(spec: SlotSpec, world: World) -> Person | None:
    return world.person_by_id().get(spec.speaker_person_id or "")


def _speaker_name(spec: SlotSpec, world: World) -> str | None:
    person = _speaker_person(spec, world)
    return person.name if person else None


def _voice_line(spec: SlotSpec, ctx: PromptContext) -> str | None:
    """How THIS speaker refers to the two parties: their own side is "we",
    the other side "you" (check 11 rejects a span naming the speaker's own
    organization, check 10 one naming both)."""
    world, kind = ctx.world, ctx.plan.kind
    vendor = world.vendor.name
    account = world.account_by_id().get(spec.account_id)
    acct = account.name if account else "the customer"
    short = account.aliases[0] if account and account.aliases else acct
    speaker = _speaker_person(spec, world)
    if kind in LINE_KINDS | BLOCK_KINDS:
        side = speaker.side.value if speaker else spec.expected_side
        who = speaker.name.split()[0] if speaker else "The speaker"
        if side == PersonSide.CUSTOMER.value:
            return (
                f"Point of view: {who} works for the customer ({acct}), so [the customer] is "
                f'"we", "our", "us", and [the vendor] ({vendor}) is "you" when talking to them. '
                f'Inside the marked sentence never name {acct} (or "{short}"), and never name both '
                "companies."
            )
        if side == PersonSide.VENDOR.value:
            return (
                f"Point of view: {who} works for the vendor ({vendor}), so [the vendor] is "
                f'"we", "our", "I", and [the customer] ({acct}) is "you", "your team" when '
                f"talking to them. Inside the marked sentence never say {vendor}."
            )
        return None
    if kind in VENDOR_NOTE_KINDS:
        return (
            f'Point of view: written by {vendor} staff for colleagues: [the vendor] is "we" or left out, '
            f'[the customer] is "{short}" or "they". Never say {vendor} inside the marked sentence.'
        )
    return None


def _names_line(spec: SlotSpec) -> str | None:
    groups: list[str] = []
    for anchor in spec.anchors:
        if anchor == spec.reference_anchor:
            continue
        forms = [f.text for f in spec.forms_for(anchor)][:2]
        alt = " (or " + ", ".join(f'"{f}"' for f in forms) + ")" if forms else ""
        groups.append(f'"{anchor}"{alt}')
    if not groups:
        return None
    return (
        "Inside the marked sentence, name: " + "; ".join(groups) + ". A nickname is fine once the "
        "full name has come up; or the turn just before names it and the marked sentence refers back."
    )


def _when_line(spec: SlotSpec, plan: DocPlan) -> str | None:
    if spec.fact_date is None or not spec.date_options:
        return None
    options = " or ".join(f'"{o}"' for o in spec.date_options)
    reference = spec.date_reference or plan.date
    where = "message" if plan.kind == DocKind.EMAIL else "document"
    if spec.fact_date == reference:
        return (
            f"When: the day of this {where} ({_natural_day(reference)}). Say {options}, or leave "
            "the date out when the meaning is clear. Never write a date as digits."
        )
    return (
        f"When: {_natural_day(spec.fact_date)}; this {where} is dated {_natural_day(reference)}. "
        f"Say it as {options}. Never write a date as digits."
    )


def _fact_brief(
    spec: SlotSpec, ctx: PromptContext, same: str, *, name_parties: bool = False
) -> list[str]:
    """The lines that describe one fact as meaning (shared by the initial
    prompt and the repair prompt). Party names go in the document-level
    NAMES TO SAY ONCE block, except per email message (each message is its
    own source) and when a repair must keep or add them."""
    assert spec.gist is not None
    lines = [f"Meaning (notes, never words to reuse): {spec.gist}"]
    voice = _voice_line(spec, ctx)
    if voice:
        lines.append(voice)
    names = _names_line(spec)
    if names:
        lines.append(names)
    ref = _reference_instruction(spec, same)
    if ref:
        lines.append(ref)
    when = _when_line(spec, ctx.plan)
    if when:
        lines.append(when)
    if spec.party_terms and (same != "document" or name_parties):
        lines.append(
            f"Name once in the same {same}, outside the marked sentence (a greeting, or the "
            "sender's sign-off with their organization): "
            + ", ".join(f'"{t}"' for t in spec.party_terms)
            + "."
        )
    if spec.required_terms:
        lines.append(
            f"Keep these words in the same {same}: "
            + ", ".join(f'"{t}"' for t in spec.required_terms)
            + "."
        )
    if spec.fact_type is not None and spec.fact_type.value in _FACT_TONE:
        lines.append(f"Tone: {_FACT_TONE[spec.fact_type.value]}.")
    if spec.directive:
        gloss = DIRECTIVE_GLOSS.get(spec.directive, spec.directive.replace("_", " "))
        lines.append(f"Style: {gloss}.")
    return lines


def _names_once(ctx: PromptContext) -> list[str]:
    """Party names check 6 needs in a single-source document (an email
    names them per message instead, in each item)."""
    if ctx.plan.kind == DocKind.EMAIL:
        return []
    header = ctx.structural_text.lower()
    return list(
        dict.fromkeys(
            t for s in ctx.specs if s.is_fact for t in s.party_terms if t.lower() not in header
        )
    )


def _said_by(spec: SlotSpec, ctx: PromptContext) -> str:
    if ctx.plan.kind not in LINE_KINDS | BLOCK_KINDS:
        return ""
    speaker = _speaker_name(spec, ctx.world)
    if speaker:
        return f" said by {speaker}"
    if spec.is_fact and spec.expected_side:
        return f" said by one of the {spec.expected_side} people"
    return ""


# One instruction per `DecoyClass.forbidden_patterns` id; check 5 enforces
# the same rule (`verify.DECOY_FORBIDDEN_PATTERNS`).
_FORBIDDEN_PATTERN_LINES: dict[str, str] = {
    "competitor_evaluation": (
        "It must not name a competitor or any other vendor, and must not say which product "
        "anyone uses, is looking at, is switching to or has ruled out: that is a real fact "
        "about competitors, not a near-miss."
    ),
}


def _script_lines(ctx: PromptContext) -> list[str]:
    classes: dict[str, DecoyClass] = {c.id: c for c in ctx.pack.decoy_classes}
    out: list[str] = []
    same = "message" if ctx.plan.kind == DocKind.EMAIL else "document"
    # A point-of-view line is the same for every item by one speaker.
    seen: dict[str, int] = {}

    def _once(line: str, n: int) -> str:
        if not line.startswith("Point of view:"):
            return line
        if line in seen:
            return f"{line.split(':', 1)[0]}: as in item {seen[line]}."
        seen[line] = n
        return line

    for n, spec in enumerate(ctx.specs, start=1):
        by = _said_by(spec, ctx)
        if spec.is_fact:
            assert spec.fact_type is not None
            kind_words = _FACT_TYPE_WORDS.get(spec.fact_type.value, "a fact")
            out.append(f"{n}. {spec.marker_id}{by}: {kind_words}.")
            out.extend(f"   {_once(line, n)}" for line in _fact_brief(spec, ctx, same))
        else:
            cls = classes.get(spec.decoy_class_id or "")
            desc = cls.description if cls else "a statement that sounds like a fact but is not one"
            near = (
                _FACT_TYPE_WORDS.get(spec.near_type.value, "a fact") if spec.near_type else "a fact"
            )
            out.append(f"{n}. {spec.marker_id}{by}: a decoy.")
            out.append(
                f"   A near-miss that sounds like {near} but is NOT a real fact about the "
                f"account. Pattern: {desc} It must not state anything that actually happened "
                "at the customer account. Say it the way people do; do not add a disclaimer "
                "explaining that it is unverified or irrelevant."
            )
            forbidden = cls.forbidden_patterns if cls else list(spec.forbidden_patterns)
            out.extend(f"   {_FORBIDDEN_PATTERN_LINES[p]}" for p in forbidden)
            voice = _voice_line(spec, ctx)
            if voice:
                out.append(f"   {_once(voice, n)}")
    return out


SYSTEM_TEMPLATE = """You write one realistic business document for a synthetic training corpus. \
Everything is fictional: the companies, people and products are invented.

You get a SCRIPT of facts to state and near-miss statements (decoys) to include, the PEOPLE \
involved, and the WORLD they live in. Follow these rules exactly.

MARKERS
- Every SCRIPT item has an id: F00001 is a fact, D00001 is a decoy. State each item exactly once.
- Wrap the WHOLE sentence that states an item in its two markers: the opening marker ⟦F00001⟧ \
goes right before the sentence and the closing marker ⟦/F00001⟧ right after it. The closing \
marker is the opening one with a slash after the first bracket.
- A correct marked sentence looks like this (angle brackets stand for your own words):
  <words> ⟦F00001⟧<one sentence in the speaker's own words that gets item F00001 across>⟦/F00001⟧ <words>
- A marked sentence is under {max_words} words and stays inside ONE {unit}; it never runs \
across a line break. Never mark just a list of names.
- Put each marked sentence where the fact comes up in the conversation, said by the person \
the SCRIPT names. Never state a fact first and then repeat it with markers, and never collect \
the marked sentences at the end.
- Mark nothing else. Never write an item id outside its markers.

VOICE
- People talk about their own organization as "we", "our", "us" and to the other side as \
"you". A customer never names their own hospital or company inside a marked sentence, the \
vendor's staff never name the vendor, and no marked sentence names both companies.
- Each Meaning line is a note about what happened, with [the customer] and [the vendor] standing \
for the two companies. Get the meaning across the way that person would say it: partly, \
indirectly, hedged, with nicknames for products, the way the Point of view and Style lines \
describe. Never reuse three or more words in a row from a Meaning note (names aside).
- The first time a product comes up, someone says its full name; after that its nickname is \
fine. A fact may take two turns: someone asks about the product or the date, and the marked \
answer refers back to it ("yeah, it's still doing that").
- Dates are said the way people say them ("today", "yesterday", "last Tuesday", "the 28th", \
"January 28"). Never write a date as digits.

NOTHING ELSE FACTUAL (a reader will extract every fact in the text; only the SCRIPT's may be there)
- Each SCRIPT fact is said once. Others may react to it ("got it", "that's rough", "thanks for \
flagging"), but nobody repeats it, summarizes it or recaps the conversation.
- Outside the marked sentences, add NO: promise, follow-up or action item ("I'll send...", \
"we'll get back to you by...", "I'll loop in engineering", "let's schedule..."), number of any \
kind (no counts, percentages, prices, versions, hours, times or dates), product, system, company \
or place name that is not in WORLD, person not in PEOPLE, plan, decision, result, metric, or \
report of how anything is going. Invent no tools, versions, integrations or features.
- Every other sentence is one of these: a greeting or goodbye; small talk (weather, weekend, \
travel, a holiday); logistics of THIS conversation (audio, video, screen sharing, who is \
joining, running late); thanks; a question about where something is in the product and an answer \
that points to a menu or page without saying what it shows; agreeing, acknowledging, asking the \
other person to go on.
- Close with thanks and goodbye, not with next steps.

FORMAT
{format_spec}

Output only the document, in that format. No commentary, no headings of your own, no code fences."""


_UNIT_WORD: dict[DocKind, str] = {
    DocKind.CALL: "line (one turn)",
    DocKind.SLACK: "message line",
    DocKind.TEAMS: "message line",
    DocKind.EMAIL: "paragraph of one message",
    DocKind.TICKET: "paragraph of one comment",
}


def system_prompt(kind: DocKind) -> str:
    return SYSTEM_TEMPLATE.format(
        max_words=min(50, MAX_SPAN_WORDS),
        unit=_UNIT_WORD.get(kind, "line of one paragraph"),
        format_spec=format_spec(kind),
    )


def build_initial_prompt(ctx: PromptContext) -> tuple[str, str]:
    plan, world = ctx.plan, ctx.world
    words, (ulo, uhi), noun = length_target(plan, len(ctx.specs))
    user: list[str] = [_document_line(ctx)]
    user.append(f"LENGTH: about {words} words, {ulo} to {uhi} {noun}.")
    if ctx.want_subject:
        user.append("SUBJECT: requested. Start the output with a `Subject:` line.")
    user.append("")
    user.append(f"WORLD (as of {_natural_day(plan.date)})")
    user.extend(_world_lines(ctx))
    user.append("")
    if ctx.cast:
        if plan.kind in LINE_KINDS | BLOCK_KINDS:
            user.append("PEOPLE (use these exact full names; nobody else speaks or writes)")
        else:
            user.append("PEOPLE (the only people you may name)")
        speakers = {s.speaker_person_id for s in ctx.specs if s.speaker_person_id}
        # The person a contact-change fact is about has left (the world's
        # tenure may not say so yet): mentioned, never a speaker.
        facts = {f.id: f for f in world.facts}
        people = world.person_by_id()
        departed = set(ctx.departed) | {
            e
            for s in ctx.specs
            if s.is_fact and s.fact_type is not None and s.fact_type.value == "contact_change"
            for e in facts[s.marker_id].entities
            if e in people
        }
        user.extend(_person_line(p, world, plan, speakers, departed) for p in ctx.cast)
        if ctx.author is not None and plan.kind not in LINE_KINDS | BLOCK_KINDS:
            user.append(f"The document is written by {ctx.author.name}.")
        user.append("")
    names_once = _names_once(ctx)
    if names_once:
        user.append(
            "NAMES TO SAY ONCE (spelled exactly, outside the marked sentences: in a greeting, an "
            "introduction, or a sign-off with the writer's organization)"
        )
        user.extend(f'- "{name}"' for name in names_once)
        user.append("")
    if ctx.specs:
        user.append("SCRIPT (state these in this order)")
        user.extend(_script_lines(ctx))
    else:
        user.append(
            "SCRIPT: this document has no scripted facts. It must state no fact about any customer; "
            "use only allowed color. Do not use any ⟦ ⟧ markers."
        )
    if ctx.story_so_far:
        user.append("")
        user.append(
            "STORY SO FAR (already said in earlier documents; do not restate these as new facts)"
        )
        user.extend(f"- {s}" for s in ctx.story_so_far)
    if ctx.sibling_titles:
        user.append("")
        user.append("OTHER DOCUMENTS ON THIS TOPIC (make this one clearly different)")
        user.extend(f"- {t}" for t in ctx.sibling_titles)
    if ctx.feedback:
        user.append("")
        user.append("AN EARLIER DRAFT FAILED THESE CHECKS; AVOID THEM")
        user.extend(f"- {f}" for f in ctx.feedback)
    if ctx.do_not_state:
        user.append("")
        user.append(
            "DO NOT STATE (an earlier draft said these, and they are not in the SCRIPT; "
            "leave them and anything like them out)"
        )
        user.extend(f'- "{s}"' for s in ctx.do_not_state)
    user.append("")
    user.append("BEFORE YOU ANSWER, CHECK")
    user.append(f"- The document has at least {ulo} {noun} and about {words} words.")
    if ctx.specs:
        pairs = ", ".join(f"⟦{s.marker_id}⟧…⟦/{s.marker_id}⟧" for s in ctx.specs)
        user.append(f"- These marker pairs each appear exactly once: {pairs}.")
        user.append(
            "- Each marked sentence names what its 'Inside the marked sentence' line lists, or "
            "refers back to the turn just before that names it."
        )
        if plan.kind in LINE_KINDS | BLOCK_KINDS:
            user.append("- Each marked sentence is said by the person the SCRIPT names.")
        user.append(
            "- No marked sentence names the speaker's own organization, and none names both "
            "companies."
        )
        if names_once:
            user.append(
                "- Every name under NAMES TO SAY ONCE appears, outside the marked sentences."
            )
    else:
        user.append("- No ⟦ ⟧ markers anywhere, and no fact about any customer.")
    user.append("- No sentence reuses a Meaning line's wording, and no date is written as digits.")
    user.append(
        "- Outside the marked sentences: no promise or next step, no number or date, no name "
        "outside WORLD and PEOPLE, and no fact said a second time."
    )
    if ctx.do_not_state:
        user.append("- Nothing from the DO NOT STATE list is said, in any wording.")
    return system_prompt(plan.kind), "\n".join(user)


# --------------------------------------------------------------------------
# Targeted repair (spec section 3.2: "one targeted repair call re-inserts
# only the missing fact into a named paragraph")
# --------------------------------------------------------------------------


@dataclass
class RepairTarget:
    """One unit the repair call must rewrite (`insert_after` set: a NEW unit
    to add after that unit instead). `instructions` are fact-specific."""

    unit_index: int
    author_name: str | None
    instructions: list[str]
    insert_after: bool = False

    @property
    def label(self) -> str:
        return f"NEW AFTER U{self.unit_index}" if self.insert_after else f"U{self.unit_index}"


REPAIR_SYSTEM = """You repair one unit (a line, message, comment or paragraph) of a business \
document for a synthetic training corpus. Everything is fictional.

Rules:
- Rewrite only the units you are asked to rewrite. Keep each rewrite in the voice and flow of the \
document around it.
- Wrap each scripted statement in its markers exactly as instructed, like ⟦F00001⟧...⟦/F00001⟧. \
Keep every OTHER ⟦...⟧ marker already in that unit exactly as it is.
- A marked span is one or two sentences, under {max_words} words, on a single line.
- The Meaning notes are for you, not text: never reuse their wording and never write a date \
as digits. Say it the way that person would say it: their own organization is "we", the other \
side is "you"; never name the speaker's own organization or both companies inside the markers.
- Never mark just a list of names.
- State no other fact about the customer.

Reply with each rewritten unit introduced by its label line, exactly like this:
=== U3 ===
<the new text of unit 3>
Do not write the speaker's name at the start of the text; the speaker is fixed. No commentary."""


def build_repair_prompt(
    ctx: PromptContext, numbered_units: list[str], targets: list[RepairTarget]
) -> tuple[str, str]:
    user = [_document_line(ctx), "", "DOCUMENT (units numbered)"]
    user.extend(numbered_units)
    user.append("")
    user.append("REWRITE THESE UNITS")
    for t in targets:
        who = f" (by {t.author_name})" if t.author_name else ""
        verb = (
            f"Write a NEW unit{who} to insert after U{t.unit_index}"
            if t.insert_after
            else f"Rewrite U{t.unit_index}{who}"
        )
        user.append(f"=== {t.label} === {verb}:")
        user.extend(f"  - {i}" for i in t.instructions)
    return REPAIR_SYSTEM.format(max_words=min(50, MAX_SPAN_WORDS)), "\n".join(user)


def fact_instruction(
    spec: SlotSpec, ctx: PromptContext, *, problems: list[str], name_parties: bool = False
) -> str:
    """One repair instruction for one failing fact slot: the same meaning
    brief the initial prompt gives, scoped to "this unit"."""
    brief = _fact_brief(spec, ctx, "unit", name_parties=name_parties)
    parts = [f"state {spec.marker_id} inside ⟦{spec.marker_id}⟧...⟦/{spec.marker_id}⟧.", *brief]
    if spec.reference_anchor:
        parts.append(
            f'Either put "{spec.reference_anchor}" in the span, or refer to it as '
            f'"{spec.reference_form}" and name "{spec.reference_anchor}" earlier in this unit.'
        )
    if problems:
        parts.append("Problems found: " + "; ".join(problems) + ".")
    return " ".join(parts)
