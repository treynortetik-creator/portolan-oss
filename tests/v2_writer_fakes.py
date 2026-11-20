"""Stub writer backends for the WP4 writer tests. No network.

`render_document` reads the writer prompt the way an obedient model
would -- the PEOPLE list, each SCRIPT item's marker id, speaker (or speaker
side), the names to use inside the marked sentence, the reference
instruction, the date phrase, the party names to say once outside the
marked sentence and the words to keep -- and returns a document in the
kind's output protocol with every marker placed correctly.
Its filler words are fixed test tokens, never meant as corpus text.
A per-marker `mutate` callback lets a test
break one slot on purpose to drive repair, regeneration and drop paths.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from corpus.generator.v2.writer import WriterCallError, WriterReply

_ITEM_RE = re.compile(r"^(\d+)\. ([FD]\d+)(?: said by ([^:]+))?:")
_SIDE_SPEAKER_RE = re.compile(r"^one of the (vendor|customer) people$")
_NAMES_RE = re.compile(r"Inside the marked sentence, name: (.+?)\. A nickname")
_ONCE_RE = re.compile(
    r"Name once in the same \w+, outside the marked sentence \([^)]*\): (.+?)\.(?: |$)"
)
_KEEP_RE = re.compile(r"Keep these words in the same \w+: (.+?)\.(?: |$)")
_WHEN_RE = re.compile(r"When: (.+?)Never write a date")
_REF_RE = re.compile(r'refer to (.+?) as "(.+?)" instead of')
_QUOTED_RE = re.compile(r'"([^"]+)"')
_PERSON_RE = re.compile(r"^- (.+?) \((vendor|customer)\):")


@dataclass
class ScriptItem:
    marker_id: str
    speaker: str | None
    must: list[str] = field(default_factory=list)  # names inside the marked sentence
    also: list[str] = field(default_factory=list)  # words to keep (inside the span here)
    once: list[str] = field(default_factory=list)  # party names, outside the span
    date_phrase: str | None = None
    ref_anchor: str | None = None
    ref_form: str | None = None


def _first_quoted_per_group(text: str) -> list[str]:
    """`"Wayfinder T3" (or "T3"); "Palisade Devices"` -> the full names."""
    out = []
    for group in text.split("; "):
        found = _QUOTED_RE.findall(group)
        if found:
            out.append(found[0])
    return out


def _date_phrase(when: str) -> str | None:
    if "leave the date out" in when:
        return None
    found = _QUOTED_RE.findall(when)
    return found[0] if found else None


def read_item_line(item: ScriptItem, line: str) -> None:
    """Fill `item` from one SCRIPT line (or one joined repair instruction)."""
    m = _NAMES_RE.search(line)
    if m:
        item.must = _first_quoted_per_group(m.group(1))
    m = _ONCE_RE.search(line)
    if m:
        item.once = _QUOTED_RE.findall(m.group(1))
    m = _KEEP_RE.search(line)
    if m:
        item.also = _QUOTED_RE.findall(m.group(1))
    m = _WHEN_RE.search(line)
    if m:
        item.date_phrase = _date_phrase(m.group(1))
    m = _REF_RE.search(line)
    if m:
        item.ref_anchor, item.ref_form = m.group(1), m.group(2)


def kind_of(system: str) -> str:
    if "# Call transcript" in system:
        return "lines"
    if "# Chat channel day" in system:
        return "lines"
    if "# Email thread" in system or "# Support ticket" in system:
        return "blocks"
    return "prose"


def parse_names_once(user: str) -> list[str]:
    """The document-level NAMES TO SAY ONCE block (not in an email)."""
    names: list[str] = []
    in_block = False
    for line in user.split("\n"):
        if line.startswith("NAMES TO SAY ONCE"):
            in_block = True
            continue
        if in_block:
            if not line.startswith("- "):
                break
            names += _QUOTED_RE.findall(line)
    return names


def parse_prompt(user: str) -> tuple[list[str], list[ScriptItem], bool]:
    people: list[str] = []
    sides: dict[str, str] = {}
    items: list[ScriptItem] = []
    want_subject = "SUBJECT: requested" in user
    section = None
    for line in user.split("\n"):
        if line.startswith("PEOPLE"):
            section = "people"
            continue
        if line.startswith("SCRIPT"):
            section = "script"
            continue
        if not line.strip():
            if section == "people":
                section = None
            continue
        if section == "people":
            m = _PERSON_RE.match(line)
            if m:
                people.append(m.group(1))
                sides[m.group(1)] = m.group(2)
        elif section == "script":
            m = _ITEM_RE.match(line)
            if m:
                speaker = m.group(3)
                side = _SIDE_SPEAKER_RE.match(speaker or "")
                if side:
                    speaker = next((p for p in people if sides[p] == side.group(1)), None)
                items.append(ScriptItem(m.group(2), speaker))
                continue
            if items:
                read_item_line(items[-1], line.strip())
    return people, items, want_subject


def _span_text(item: ScriptItem) -> str:
    # The number keeps two scripted spans with the same names distinct.
    parts = ["fixture", "span", "number", item.marker_id[1:].lstrip("0")]
    if item.ref_form:
        parts.append(item.ref_form)
    parts.extend(item.must)
    parts.extend(item.also)
    if item.date_phrase:
        parts.append(item.date_phrase)
    parts.append("noted")
    return " ".join(parts)


def _context(item: ScriptItem) -> str:
    """Words said before the marked sentence in the same unit: the party
    names to say once outside the span, and the full name a reference
    stands in for (for email, the same message is the same source)."""
    context = ""
    if item.once:
        context += f"fixture greeting for {' and '.join(item.once)} first. "
    if item.ref_anchor:
        context += f"fixture context about {item.ref_anchor} first. "
    return context


Mutator = Callable[[ScriptItem, str], tuple[str, str | None]]


def render_document(
    system: str,
    user: str,
    *,
    mutate: dict[str, Mutator] | None = None,
) -> str:
    """The obedient reply. `mutate[marker_id](item, span)` may return a
    replacement `(span, speaker_override)`; a span of `""` omits the
    statement entirely."""
    mutate = mutate or {}
    people, items, want_subject = parse_prompt(user)
    protocol = kind_of(system)
    first = people[0] if people else None
    second = people[1] if len(people) > 1 else first
    units: list[tuple[str | None, str]] = []
    once = parse_names_once(user)
    opening = f"fixture opening words for {' and '.join(once)}" if once else "fixture opening words"
    units.append((first, opening))
    if second is not None and second != first:
        units.append((second, "fixture reply words"))
    for item in items:
        span = _span_text(item)
        speaker = item.speaker or first
        if item.marker_id in mutate:
            span, override = mutate[item.marker_id](item, span)
            speaker = override or speaker
        context = _context(item)
        if span:
            units.append(
                (
                    speaker,
                    f"{context}fixture lead ⟦{item.marker_id}⟧{span}⟦/{item.marker_id}⟧ tail",
                )
            )
    closing = "fixture closing words"
    # Meet the prompt's LENGTH line, as an obedient writer does (a reply far
    # shorter than planned is a truncated one).
    m = re.search(r"^LENGTH: about (\d+) words", user, re.MULTILINE)
    written = sum(len(text.split()) for _who, text in units) + 3
    if m and int(m.group(1)) > written:
        closing += " fixture" * (int(m.group(1)) - written)
    units.append((second, closing))
    out: list[str] = []
    if want_subject:
        out.append("Subject: fixture subject line")
    for who, text in units:
        if protocol == "lines":
            out.append(f"{who}: {text}")
        elif protocol == "blocks":
            out.append(f"### {who}")
            out.append("fixture greeting")
            out.append("")
            out.append(text)
            out.append("")
        else:
            out.append(text)
            out.append("")
    return "\n".join(out).strip() + "\n"


def repair_reply(user: str, fix: Callable[[str, str], str] | None = None) -> str:
    """A repair reply: for every `=== Ux ===` / `=== NEW AFTER Ux ===` target,
    re-state each named fact correctly by re-reading its instruction."""
    out: list[str] = []
    for block in re.split(r"(?m)^(?==== )", user.split("REWRITE THESE UNITS", 1)[1]):
        m = re.match(r"=== (NEW AFTER U\d+|U\d+) ===", block)
        if not m:
            continue
        label = m.group(1)
        pieces = ["fixture repaired words"]
        for inst in re.findall(r"  - (.+)", block):
            fm = re.match(r"state (\w+) inside", inst)
            if not fm:
                continue
            item = ScriptItem(fm.group(1), None)
            read_item_line(item, inst)
            ref = re.search(r'Either put "(.+?)" in the span', inst)
            span = " ".join(
                [
                    "fixed",
                    *([ref.group(1)] if ref else []),
                    *item.must,
                    *item.also,
                    *([item.date_phrase] if item.date_phrase else []),
                    "done",
                ]
            )
            if fix is not None:
                span = fix(item.marker_id, span)
            if item.once:
                pieces.append(f"fixture greeting for {' and '.join(item.once)} first.")
            pieces.append(f"⟦{item.marker_id}⟧{span}⟦/{item.marker_id}⟧")
        out.append(f"=== {label} ===")
        out.append(" ".join(pieces))
    return "\n".join(out) + "\n"


@dataclass
class FakeCall:
    model: str
    system: str
    user: str
    max_tokens: int


@dataclass
class ScriptedBackend:
    """`reply(model, system, user, n)` decides each reply (n = call number,
    0-based). Records every call. `cost` per call is fixed."""

    reply: Callable[[str, str, str, int], str]
    cost: float = 0.001
    calls: list[FakeCall] = field(default_factory=list)
    fail_models: set[str] = field(default_factory=set)

    def complete(self, *, model, system, user, max_tokens, temperature):  # noqa: ANN001
        self.calls.append(FakeCall(model, system, user, max_tokens))
        if model in self.fail_models:
            raise WriterCallError(f"{model}: status 503 (fake)")
        text = self.reply(model, system, user, len(self.calls) - 1)
        return WriterReply(text, self.cost, len(system + user) // 4, len(text) // 4)


def obedient(model: str, system: str, user: str, n: int) -> str:
    if "REWRITE THESE UNITS" in user:
        return repair_reply(user)
    return render_document(system, user)
