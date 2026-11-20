"""WP4 helper: turn a writer's raw output into a finished document.

`01-spec/world-generator-v2-2026-09-22.md` section 2: "Code renders the
structure: headers, ids, timestamps, CRM fields, link lists, quoted-reply
blocks and signatures. Models write only the text fields."

The writer (an open-weight model, `corpus.generator.v2.writer`) returns
plain text in a small per-kind protocol (`formats/<kind>.md`):

- **lines** (call, slack, teams): one `Full Name: text` line per turn or
  message.
- **blocks** (ticket, email): a `### Full Name` line opens each comment or
  message; the lines under it are its body. An optional first line
  `Subject: ...` carries a writer-written subject.
- **prose** (crm_note, meeting_notes, doc, hard_negative, off_topic,
  near_duplicate): paragraphs separated by blank lines.

`parse_draft` reads that protocol into `DraftUnit`s. `assemble` then renders
the final files with code-owned structure only (timestamps, JSON envelopes,
email headers, message ids, comment numbers) around the writer's words,
reusing the `RenderedDoc` skeleton from WP3 for file paths and container
fields. `prepare_sources` runs the finished files through the exact path
production uses (`FileConnector` where it reads the kind, then
`portolan.ingest.prepare.prepare_text`) so every label span is measured
against the text the extractor actually receives.

Provenance rule (spec Build notes, 2026-09-22 WP2 review): every word in a
unit's text comes from the writer. This module only splits, joins,
re-wraps, strips marker glyphs and renders structure. It never writes a
sentence. The two structural strings it does render besides headers are
the email client's quote attribution line (`On <date>, <name> <email>
wrote:`) above a code-quoted earlier message, and nothing else.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import re
import tempfile
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

from corpus.generator.markers import (
    MARKER_TOKEN_RE,
    STRAY_MARKER_GLYPH_RE,
    normalize_marker_glyphs,
)
from corpus.generator.structured import (
    call_locator,
    seconds_to_hhmmss,
    slack_locator,
    slack_ts,
    teams_created_datetime,
    teams_locator,
    ticket_locator,
)
from corpus.generator.v2.models import Person, PersonSide, World
from corpus.generator.v2.planner import DocKind, DocPlan
from corpus.generator.v2.renderers import RenderedDoc, SlotKind, slack_user_id
from portolan.connectors.file_connector import FileConnector
from portolan.extract.schema import SourceType
from portolan.ingest.models import LocatorSpan, RawSource
from portolan.ingest.prepare import prepare_text

__all__ = [
    "LINE_KINDS",
    "BLOCK_KINDS",
    "PROSE_KINDS",
    "SPEAKER_KINDS",
    "AssembledDoc",
    "AssembledUnit",
    "Cell",
    "Draft",
    "DraftUnit",
    "MarkerSpan",
    "PeopleIndex",
    "PreparedSource",
    "assemble",
    "locate_markers",
    "parse_draft",
    "prepare_sources",
    "skeleton_requests_subject",
]

LINE_KINDS = frozenset({DocKind.CALL, DocKind.SLACK, DocKind.TEAMS})
BLOCK_KINDS = frozenset({DocKind.TICKET, DocKind.EMAIL})
PROSE_KINDS = frozenset({DocKind.CRM_NOTE, DocKind.MEETING_NOTES, DocKind.DOC})
# Kinds whose locator unit has an author the speaker check can compare
# (spec section 3.2 check 4: "line speaker, From, comment author, chat
# user"). Prose containers have none.
SPEAKER_KINDS = LINE_KINDS | BLOCK_KINDS

_PROSE_SOURCE_TYPE: dict[DocKind, SourceType] = {
    DocKind.CRM_NOTE: SourceType.CRM,
    DocKind.MEETING_NOTES: SourceType.DOC,
    DocKind.DOC: SourceType.DOC,
}


def _stable_int(*parts: object) -> int:
    return int(hashlib.sha256(":".join(str(p) for p in parts).encode()).hexdigest()[:12], 16)


# --------------------------------------------------------------------------
# People resolution
# --------------------------------------------------------------------------

_HONORIFIC_RE = re.compile(r"^(dr|mr|mrs|ms|mx|prof)\.?\s+", re.IGNORECASE)


def _norm_name(name: str) -> str:
    name = re.sub(r"[*_`]", "", name).strip()
    name = re.sub(r"\s*\(.*?\)\s*$", "", name)  # "Name (Vendor)"
    name = re.sub(r"\s+[-–—|,]\s+.*$", "", name)  # "Name - Account Rep"
    name = _HONORIFIC_RE.sub("", name)
    return " ".join(name.split()).casefold()


@dataclass
class PeopleIndex:
    """Resolves a writer-given display name to one `Person` of the world:
    exact full name first, then a unique first name, then a unique last
    name, preferring the document's own cast (`preferred`) and falling back
    to every person on the document's account plus the vendor's staff."""

    preferred: list[Person]
    others: list[Person] = field(default_factory=list)

    def resolve(self, raw_name: str) -> Person | None:
        key = _norm_name(raw_name)
        if not key:
            return None
        for pool in (self.preferred, self.others):
            exact = [p for p in pool if p.name.casefold() == key]
            if len(exact) == 1:
                return exact[0]
        for pool in (self.preferred, self.others):
            first = [p for p in pool if p.name.split()[0].casefold() == key]
            if len(first) == 1:
                return first[0]
            last = [p for p in pool if p.name.split()[-1].casefold() == key]
            if len(last) == 1:
                return last[0]
        # A misspelled cast name (seen live 2026-09-22: "Adechukukwu" for
        # "Adechukwu") still names that person; the rendered author always
        # comes from the Person record, so this fixes structure only.
        scored = sorted(
            (
                (difflib.SequenceMatcher(None, key, p.name.casefold()).ratio(), p)
                for p in self.preferred
            ),
            key=lambda sp: -sp[0],
        )
        if scored and scored[0][0] >= 0.85 and (len(scored) == 1 or scored[1][0] < 0.85):
            return scored[0][1]
        return None

    @property
    def everyone(self) -> list[Person]:
        return [*self.preferred, *self.others]


# --------------------------------------------------------------------------
# Draft parsing
# --------------------------------------------------------------------------


@dataclass
class DraftUnit:
    """One writer unit: a call turn, a chat message, a ticket comment, an
    email message, or a prose paragraph. `text` still carries markers."""

    text: str
    person_id: str | None = None
    author_name: str | None = None


@dataclass
class Draft:
    subject: str | None
    units: list[DraftUnit]
    notes: list[str] = field(default_factory=list)
    # Doc-level structural problems: fatal for the attempt (it cannot be
    # rendered as-is), never repaired by code.
    fatal: list[str] = field(default_factory=list)


_FENCE_RE = re.compile(r"^\s*```[\w-]*\s*$")
_TIMESTAMP_PREFIX_RE = re.compile(r"^\s*\[?\(?\d{1,2}:\d{2}(?::\d{2})?\)?\]?\s*[-–—]?\s*")
_LINE_SPEAKER_RE = re.compile(
    r"^\s*(?:[-*•]\s+)?(\*{0,2})([^:\n]{1,80}?)\1\s*:\s*(\*{0,2})\s*(.*)$"
)
_LEADING_MARKERS_RE = re.compile(r"^((?:⟦/?[FD]\d+[a-z]?⟧\s*)+)([^:⟦⟧\n]{1,80}?:\s*)(.*)$")
_BLOCK_HEADER_RE = re.compile(r"^\s*#{2,4}\s*(?:from:\s*)?(.+?)\s*#*\s*$", re.IGNORECASE)
_SUBJECT_RE = re.compile(r"^\s*\**\s*(?:subject|title)\s*:\s*\**\s*(.+?)\s*$", re.IGNORECASE)
_PREAMBLE_RE = re.compile(r"^(here is|here's|here are|sure[,!.]|below is|certainly)", re.IGNORECASE)
_NAMEISH_RE = re.compile(r"^[A-Z][\w'.\-]*(?:\s+[A-Z][\w'.\-]*){0,3}$")


# Single capitalized words that open a `Label: text` line without naming a
# speaker (a preamble or a stray label, not an invented person).
_NON_SPEAKER_LABELS = frozenset(
    {
        "note",
        "notes",
        "agenda",
        "summary",
        "transcript",
        "subject",
        "title",
        "action",
        "update",
        "re",
        "fwd",
        "ps",
        "step",
        "call",
        "date",
        "time",
        "attendees",
        "participants",
    }
)


def _looks_like_speaker(raw_name: str) -> bool:
    name = re.sub(r"[*_`]", "", raw_name).strip()
    if not _NAMEISH_RE.match(name):
        return False
    return not any(word.casefold().strip(".") in _NON_SPEAKER_LABELS for word in name.split())


def _strip_fences(raw: str) -> list[str]:
    lines = normalize_marker_glyphs(raw.replace("\r\n", "\n").replace("\r", "\n")).split("\n")
    return [line for line in lines if not _FENCE_RE.match(line)]


def parse_draft(kind: DocKind, raw: str, people: PeopleIndex, *, want_subject: bool) -> Draft:
    """Read a writer's raw output into units, per `kind`'s protocol."""
    lines = _strip_fences(raw)
    if kind in LINE_KINDS:
        return _parse_lines(lines, people)
    if kind in BLOCK_KINDS:
        return _parse_blocks(lines, people, want_subject=want_subject)
    return _parse_prose(lines, want_subject=want_subject)


def _parse_lines(lines: list[str], people: PeopleIndex) -> Draft:
    units: list[DraftUnit] = []
    notes: list[str] = []
    fatal: list[str] = []
    dropped_preamble = 0
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        line = _TIMESTAMP_PREFIX_RE.sub("", line)
        # A marker the writer put before the speaker name ("⟦D00004⟧Kawan
        # Gorbova: ...", seen live 2026-09-22) moves after it: structure.
        lead = _LEADING_MARKERS_RE.match(line)
        if lead is not None:
            line = f"{lead.group(2)}{lead.group(1)}{lead.group(3)}"
        m = _LINE_SPEAKER_RE.match(line)
        person = people.resolve(m.group(2)) if m else None
        if person is not None and m is not None:
            units.append(
                DraftUnit(text=m.group(4).strip(), person_id=person.id, author_name=person.name)
            )
            continue
        if m is not None and _looks_like_speaker(m.group(2)):
            # A named speaker who is not in the cast: the writer invented a
            # person. Fatal for this attempt (spec section 3.1: "state no
            # other customer fact" -- a new named person is one).
            fatal.append(f"unknown speaker {m.group(2).strip()!r}")
            units.append(DraftUnit(text=m.group(4).strip(), author_name=m.group(2).strip()))
            continue
        if not units:
            dropped_preamble += 1
            continue
        # A continuation line: structure fix, joined onto the turn above.
        units[-1].text = f"{units[-1].text} {line}".strip()
        notes.append("joined a continuation line onto the previous turn")
    if dropped_preamble:
        notes.append(f"dropped {dropped_preamble} line(s) before the first speaker line")
    units = [u for u in units if u.text]
    if not units:
        fatal.append("no speaker lines found")
    return Draft(subject=None, units=units, notes=notes, fatal=fatal)


def _parse_blocks(lines: list[str], people: PeopleIndex, *, want_subject: bool) -> Draft:
    units: list[DraftUnit] = []
    notes: list[str] = []
    fatal: list[str] = []
    subject: str | None = None
    body: list[str] = []
    current: DraftUnit | None = None
    dropped_preamble = 0

    def _close() -> None:
        nonlocal subject
        if current is None:
            return
        if not units:
            # A `Subject:` line the writer put under the first header is
            # the thread's subject, not body text.
            first = next((i for i, line in enumerate(body) if line.strip()), None)
            if first is not None:
                sm = _SUBJECT_RE.match(body[first])
                if sm is not None:
                    if subject is None:
                        subject = sm.group(1).strip()
                    del body[first]
        text = "\n".join(body).strip("\n")
        text = re.sub(r"\n\s*\n+", "\n\n", text)
        text = "\n".join(line.rstrip() for line in text.split("\n"))
        current.text = text.strip()
        units.append(current)

    for raw_line in lines:
        header = _BLOCK_HEADER_RE.match(raw_line)
        if header is not None:
            _close()
            body = []
            name = header.group(1)
            person = people.resolve(name)
            if person is None:
                fatal.append(f"unknown author {name.strip()!r}")
                current = DraftUnit(text="", author_name=name.strip())
            else:
                current = DraftUnit(text="", person_id=person.id, author_name=person.name)
            continue
        if current is None:
            sm = _SUBJECT_RE.match(raw_line)
            if sm is not None and subject is None:
                subject = sm.group(1).strip()
                continue
            if raw_line.strip():
                dropped_preamble += 1
            continue
        body.append(raw_line)
    _close()

    if subject is not None and not want_subject:
        notes.append("ignored an unrequested Subject line")
        subject = None
    if want_subject and not subject:
        fatal.append("requested Subject line missing")
    if dropped_preamble:
        notes.append(f"dropped {dropped_preamble} line(s) before the first ### header")
    empty = [u for u in units if not u.text]
    if empty:
        notes.append(f"dropped {len(empty)} empty block(s)")
    units = [u for u in units if u.text]
    if not units:
        fatal.append("no ### blocks found")
    return Draft(subject=subject, units=units, notes=notes, fatal=fatal)


_MD_HEADING_RE = re.compile(r"^\s*#{1,6}\s+\S")


def _parse_prose(lines: list[str], *, want_subject: bool = False) -> Draft:
    # A markdown heading is never part of a prose container (seen live
    # 2026-09-22: a writer echoed its format spec's own title as line 1).
    headings = [line for line in lines if _MD_HEADING_RE.match(line)]
    lines = [line for line in lines if not _MD_HEADING_RE.match(line)]
    subject: str | None = None
    first = next((i for i, line in enumerate(lines) if line.strip()), None)
    if first is not None:
        sm = _SUBJECT_RE.match(lines[first])
        if sm is not None:
            subject = sm.group(1).strip()
            del lines[first]
    text = "\n".join(line.rstrip() for line in lines).strip("\n")
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    notes: list[str] = []
    if len(paragraphs) > 1 and _PREAMBLE_RE.match(paragraphs[0]) and paragraphs[0].endswith(":"):
        paragraphs = paragraphs[1:]
        notes.append("dropped a preamble paragraph")
    if headings:
        notes.append(f"dropped {len(headings)} markdown heading line(s)")
    units = [DraftUnit(text=p) for p in paragraphs]
    fatal = [] if units else ["empty document"]
    if subject is not None and not want_subject:
        notes.append("ignored an unrequested Subject line")
        subject = None
    if want_subject and not subject:
        fatal.append("requested Subject line missing")
    return Draft(subject=subject, units=units, notes=notes, fatal=fatal)


# --------------------------------------------------------------------------
# Marker location
# --------------------------------------------------------------------------


# A marker id the writer wrapped in some other glyph (seen live 2026-09-22:
# `✦D00004`), or a bare id left behind: never a word, always removed, and
# reported as unbalanced so its slot fails check 1.
_MANGLED_ID_RE = r"(?<![A-Za-z0-9])[^\w\s⟦⟧〚〛]?/?([FD]\d{5}[a-z]?)(?![0-9])[^\w\s⟦⟧〚〛]?"
_TOKEN_OR_STRAY_RE = re.compile(
    rf"{MARKER_TOKEN_RE.pattern}|{STRAY_MARKER_GLYPH_RE.pattern}|{_MANGLED_ID_RE}"
)


@dataclass(frozen=True)
class MarkerSpan:
    """One balanced marker pair, in CLEAN (marker-stripped) offsets of the
    unit's text."""

    marker_id: str
    start: int
    end: int


def locate_markers(marked: str) -> tuple[str, list[MarkerSpan], list[str], list[str]]:
    """Strip every marker token from `marked`, returning `(clean_text,
    balanced_spans, unbalanced_ids, overlap_ids)`. Offsets in
    `balanced_spans` index `clean_text`. An opener with no closer (or the
    reverse) inside this one unit is reported in `unbalanced_ids` -- a span
    that crosses a unit boundary shows up this way on both sides."""
    clean_parts: list[str] = []
    pos = 0
    clean_len = 0
    open_at: dict[str, int] = {}
    spans: list[MarkerSpan] = []
    unbalanced: list[str] = []
    for m in _TOKEN_OR_STRAY_RE.finditer(marked):
        chunk = marked[pos : m.start()]
        clean_parts.append(chunk)
        clean_len += len(chunk)
        pos = m.end()
        # A marker between two spaces leaves a double space: drop one.
        clean_so_far_ends_in_space = next(
            (part[-1] in " \t" for part in reversed(clean_parts) if part), False
        )
        if clean_so_far_ends_in_space and marked[pos : pos + 1] in (" ", "\t"):
            pos += 1
        if m.group(3) is not None:
            unbalanced.append(m.group(3))  # a mangled marker: removed, reported
            continue
        if m.group(2) is None:
            continue  # a stray bracket glyph: removed, it is not a word
        is_close, marker_id = m.group(1) == "/", m.group(2)
        if not is_close and marker_id in open_at:
            # The same opener twice: the writer dropped the closer's slash.
            is_close = True
        if not is_close:
            open_at[marker_id] = clean_len
        elif marker_id in open_at:
            spans.append(MarkerSpan(marker_id, open_at.pop(marker_id), clean_len))
        else:
            unbalanced.append(marker_id)
    clean_parts.append(marked[pos:])
    unbalanced.extend(open_at)
    clean = "".join(clean_parts)
    # Trim whitespace the writer left just inside a marker: a label never
    # starts or ends on a space.
    trimmed: list[MarkerSpan] = []
    for sp in spans:
        start, end = sp.start, sp.end
        while start < end and clean[start].isspace():
            start += 1
        while end > start and clean[end - 1].isspace():
            end -= 1
        trimmed.append(MarkerSpan(sp.marker_id, start, end))
    # And the unit itself never starts or ends on whitespace a removed
    # marker left behind.
    lead = len(clean) - len(clean.lstrip())
    clean = clean.strip()
    spans = [
        MarkerSpan(
            sp.marker_id,
            min(max(0, sp.start - lead), len(clean)),
            min(max(0, sp.end - lead), len(clean)),
        )
        for sp in trimmed
    ]
    overlap: list[str] = []
    ordered = sorted(spans, key=lambda s: s.start)
    for a, b in zip(ordered, ordered[1:], strict=False):
        if b.start < a.end:
            overlap.extend([a.marker_id, b.marker_id])
    return clean, spans, unbalanced, overlap


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------


@dataclass
class Cell:
    """The smallest locator unit inside one `AssembledUnit`: `[start, end)`
    into the unit's clean text, the manifest-style v2 locator, and the
    locator id `prepare_text` gives that same text (`prepared_locator`)."""

    start: int
    end: int
    locator: str
    prepared_locator: str
    source_index: int


@dataclass
class AssembledUnit:
    index: int  # 1-based over the whole document (repair addresses it)
    author_person_id: str | None
    author_name: str | None
    marked_text: str
    clean_text: str
    spans: list[MarkerSpan]
    unbalanced: list[str]
    overlap: list[str]
    cells: list[Cell]
    source_index: int = 0


@dataclass
class SourceInfo:
    source_id: str
    path: str
    source_type: SourceType
    date: date


@dataclass
class AssembledDoc:
    kind: DocKind
    files: dict[str, str]
    units: list[AssembledUnit]
    sources: list[SourceInfo]
    subject: str | None
    notes: list[str] = field(default_factory=list)

    def writer_text(self) -> str:
        """Every writer-written string in the document (for doc-level scans)."""
        parts = [u.clean_text for u in self.units]
        if self.subject:
            parts.insert(0, self.subject)
        return "\n".join(parts)


def skeleton_requests_subject(rendered: RenderedDoc) -> bool:
    """True when the WP3 skeleton leaves a subject line or document title to
    the writer (a `SlotKind.SUBJECT` slot: email and ticket subjects, CRM
    note subjects, doc titles)."""
    return any(slot.kind == SlotKind.SUBJECT for slot in rendered.slots)


def _skeleton_subject(rendered: RenderedDoc) -> str | None:
    for path, content in rendered.files.items():
        if path.startswith("email/"):
            for line in content.split("\n"):
                if line.startswith("Subject:"):
                    return line.split(":", 1)[1].strip()
        if path.startswith("tickets/"):
            try:
                return str(json.loads(content).get("subject") or "") or None
            except ValueError:
                return None
    return None


def _email_header_value(rendered: RenderedDoc, key: str) -> str | None:
    """A structural header of the skeleton's first message (`X-Related-
    Account`, `X-Related-Links`): code-owned, carried onto every message."""
    first = sorted(p for p in rendered.files if p.startswith("email/"))
    if not first:
        return None
    for line in rendered.files[first[0]].split("\n"):
        if not line.strip():
            break
        if line.startswith(f"{key}:"):
            return line.split(":", 1)[1].strip()
    return None


def _prose_header(rendered: RenderedDoc, subject: str | None) -> str:
    """The code header of a meeting-notes or doc skeleton (Date, Meeting or
    Document type, Account, Attendees or Author, Title): every line above
    the first body slot, with the title slot filled by the writer's
    subject."""
    path = next(iter(rendered.files))
    header: list[str] = []
    for line in rendered.files[path].split("\n"):
        if "[SLOT" in line and "subject]" not in line:
            break
        header.append(line)
    text = "\n".join(header).strip("\n")
    if subject is not None:
        text = re.sub(r"\[SLOT [^\]]*subject\]", lambda _m: subject, text)
    return f"{text}\n\n" if text else ""


def unit_dates(draft_units: list[AssembledUnit], world: World, start: date) -> list[date]:
    """A date per unit: never before the previous unit or `start`, and never
    before a fact the unit states (WP3's rule for threads that run forward:
    each message or comment is dated on or after its own fact)."""
    facts = {f.id: f.date for f in world.facts}
    out: list[date] = []
    current = start
    for unit in draft_units:
        dates = [facts[s.marker_id] for s in unit.spans if s.marker_id in facts]
        current = max([current, *dates])
        out.append(current)
    return out


def _skeleton_wants_quotes(rendered: RenderedDoc) -> bool:
    """WP3 plans quoted replies as `QuoteOp`s (`[QUOTE ...]` placeholders)."""
    return bool(rendered.quote_ops) or any(
        "[QUOTE" in content for content in rendered.files.values()
    )


def _call_header(rendered: RenderedDoc) -> str:
    """Any preamble a future skeleton puts above the first timestamped line
    (spec section 2: a call header with call_type, attendees,
    `crm_account_id`) is code structure and is kept verbatim."""
    for path, content in rendered.files.items():
        if path.startswith("calls/"):
            header: list[str] = []
            for line in content.split("\n"):
                if re.match(r"^\[\d{2}:\d{2}:\d{2}\]", line):
                    break
                header.append(line)
            text = "\n".join(header).strip("\n")
            return f"{text}\n" if text else ""
    return ""


def _unit(index: int, du: DraftUnit, source_index: int = 0) -> AssembledUnit:
    clean, spans, unbalanced, overlap = locate_markers(du.text)
    return AssembledUnit(
        index=index,
        author_person_id=du.person_id,
        author_name=du.author_name,
        marked_text=du.text,
        clean_text=clean,
        spans=spans,
        unbalanced=unbalanced,
        overlap=overlap,
        cells=[],
        source_index=source_index,
    )


def assemble(
    *,
    plan: DocPlan,
    rendered: RenderedDoc,
    world: World,
    draft: Draft,
    people: PeopleIndex,
) -> AssembledDoc:
    """Render the final files for `plan` from `draft` (writer words) plus
    code-owned structure. Paths and container fields come from the WP3
    skeleton in `rendered`."""
    kind = plan.kind
    if kind == DocKind.CALL:
        return _assemble_call(plan, rendered, draft)
    if kind in (DocKind.SLACK, DocKind.TEAMS):
        return _assemble_chat(plan, rendered, world, draft)
    if kind == DocKind.TICKET:
        return _assemble_ticket(plan, rendered, world, draft)
    if kind == DocKind.EMAIL:
        return _assemble_email(plan, rendered, world, draft, people)
    return _assemble_prose(plan, rendered, draft)


def _single_path(rendered: RenderedDoc, prefix: str | None = None) -> str:
    paths = [p for p in rendered.files if prefix is None or p.startswith(prefix)]
    if len(paths) != 1:
        raise ValueError(f"{rendered.plan_id}: expected one skeleton file, got {sorted(paths)}")
    return paths[0]


def _call_start_seconds(rendered: RenderedDoc, plan: DocPlan) -> int:
    for content in rendered.files.values():
        m = re.search(r"^\[(\d{2}):(\d{2}):(\d{2})\]", content, re.MULTILINE)
        if m:
            return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))
    return 3 + _stable_int("call-t0", plan.id) % 20


def _assemble_call(plan: DocPlan, rendered: RenderedDoc, draft: Draft) -> AssembledDoc:
    path = _single_path(rendered, "calls/")
    header = _call_header(rendered)
    t = _call_start_seconds(rendered, plan)
    lines: list[str] = []
    units: list[AssembledUnit] = []
    for i, du in enumerate(draft.units, start=1):
        unit = _unit(i, du)
        ts = seconds_to_hhmmss(t)
        loc = call_locator(ts)
        unit.cells = [Cell(0, len(unit.clean_text), loc, loc, 0)]
        units.append(unit)
        lines.append(f"[{ts}] {du.author_name}: {unit.clean_text}")
        words = len(unit.clean_text.split())
        t += max(2, round(words / 2.6)) + 1 + _stable_int("call-gap", plan.id, i) % 3
    text = header + "\n".join(lines) + "\n"
    source = SourceInfo(plan.id, path, SourceType.CALL, plan.date)
    return AssembledDoc(DocKind.CALL, {path: text}, units, [source], None, list(draft.notes))


def _assemble_chat(
    plan: DocPlan, rendered: RenderedDoc, world: World, draft: Draft
) -> AssembledDoc:
    is_teams = plan.kind == DocKind.TEAMS
    by_id = world.person_by_id()
    path = _single_path(rendered)
    channel = plan.channel or "general"
    date_str = plan.date.isoformat()
    seconds = 8 * 3600 + 900 + (_stable_int("chat-t0", plan.id) % 3600)
    messages: list[dict] = []
    units: list[AssembledUnit] = []
    for i, du in enumerate(draft.units, start=1):
        flat = DraftUnit(" ".join(du.text.split()), du.person_id, du.author_name)
        unit = _unit(i, flat)
        loc = (teams_locator if is_teams else slack_locator)(channel, date_str, i)
        unit.cells = [Cell(0, len(unit.clean_text), loc, f"L{i}", 0)]
        units.append(unit)
        if is_teams:
            messages.append(
                {
                    "id": f"{plan.id}-m{i}",
                    "from": {"user": {"displayName": du.author_name}},
                    "body": {"content": unit.clean_text},
                    "createdDateTime": teams_created_datetime(date_str, seconds),
                    "replyToId": None,
                }
            )
        else:
            person = by_id.get(du.person_id or "")
            messages.append(
                {
                    "type": "message",
                    "user": slack_user_id(person) if person else du.author_name,
                    "text": unit.clean_text,
                    "ts": slack_ts(date_str, seconds),
                    "thread_ts": None,
                }
            )
        seconds += 40 + _stable_int("chat-gap", plan.id, i) % 1500
        seconds = min(seconds, 86399)
    text = json.dumps(messages, indent=2, ensure_ascii=False) + "\n"
    source = SourceInfo(plan.id, path, SourceType.CHAT, plan.date)
    return AssembledDoc(plan.kind, {path: text}, units, [source], None, list(draft.notes))


def chat_transcript_text(assembled: AssembledDoc) -> str:
    """The plain text a chat day is handed to extraction as (one `Name:
    message` line per message, the same rendering v1's
    `training_export._chat_group_text_and_quotes` uses). No connector reads
    chat files yet; this is the stand-in."""
    return "".join(f"{u.author_name}: {u.clean_text}\n" for u in assembled.units)


def _assemble_ticket(
    plan: DocPlan, rendered: RenderedDoc, world: World, draft: Draft
) -> AssembledDoc:
    path = _single_path(rendered, "tickets/")
    payload = json.loads(rendered.files[path])
    ticket_id = str(payload.get("id") or plan.id)
    if draft.subject:
        payload["subject"] = draft.subject
    comments: list[dict] = []
    units: list[AssembledUnit] = []
    for i, du in enumerate(draft.units, start=1):
        unit = _unit(i, du)
        loc = ticket_locator(ticket_id, i)
        unit.cells = [Cell(0, len(unit.clean_text), loc, loc, 0)]
        units.append(unit)
    for i, (unit, on) in enumerate(zip(units, unit_dates(units, world, plan.date), strict=True), 1):
        comments.append(
            {
                "number": i,
                "author": unit.author_name,
                "created": on.isoformat(),
                "body": unit.clean_text,
            }
        )
    payload["comments"] = comments
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    source = SourceInfo(ticket_id, path, SourceType.TICKET, plan.date)
    subject = str(payload.get("subject") or "") or None
    return AssembledDoc(DocKind.TICKET, {path: text}, units, [source], subject, list(draft.notes))


def _paragraph_cells(clean: str) -> list[tuple[int, int]]:
    """`[start, end)` of each blank-line-separated paragraph of `clean`,
    matching `prepare._prepare_email`'s own `re.split(r"\\n\\s*\\n")` plus
    `.strip()`."""
    cells: list[tuple[int, int]] = []
    pos = 0
    for m in re.finditer(r"\n\s*\n", clean + "\n\n"):
        chunk = clean[pos : m.start()]
        lead = len(chunk) - len(chunk.lstrip())
        stripped = chunk.strip()
        if stripped:
            start = pos + lead
            cells.append((start, start + len(stripped)))
        pos = m.end()
    return cells


def _other_side(person: Person | None, cast: list[Person]) -> list[Person]:
    if person is None:
        return cast[:1]
    return [p for p in cast if p.side != person.side][:1] or [p for p in cast if p.id != person.id][
        :1
    ]


def _assemble_email(
    plan: DocPlan, rendered: RenderedDoc, world: World, draft: Draft, people: PeopleIndex
) -> AssembledDoc:
    domain = world.vendor.message_id_domain
    by_id = world.person_by_id()
    subject = draft.subject or _skeleton_subject(rendered) or ""
    quote = _skeleton_wants_quotes(rendered)
    cast = [by_id[u.person_id] for u in draft.units if u.person_id in by_id]
    cast_unique = list({p.id: p for p in [*cast, *people.preferred]}.values())
    moment = datetime.combine(plan.date, datetime.min.time()) + timedelta(
        hours=8, minutes=_stable_int("email-t0", plan.id) % 150
    )
    files: dict[str, str] = {}
    units: list[AssembledUnit] = []
    sources: list[SourceInfo] = []
    prev: tuple[str, str, AssembledUnit, Person | None] | None = None
    related = [
        (key, value)
        for key in ("X-Related-Account", "X-Related-Links")
        if (value := _email_header_value(rendered, key))
    ]
    drafted = [_unit(i, du, source_index=i - 1) for i, du in enumerate(draft.units, start=1)]
    dates = unit_dates(drafted, world, plan.date)
    for i, (du, unit) in enumerate(zip(draft.units, drafted, strict=True), start=1):
        msg_id = f"{plan.id}-m{i}"
        if dates[i - 1] > moment.date():
            moment = datetime.combine(dates[i - 1], datetime.min.time()) + timedelta(
                hours=8, minutes=_stable_int("email-day", plan.id, i) % 150
            )
        sender = by_id.get(du.person_id) if du.person_id else None
        from_addr = f"{sender.name} <{sender.email}>" if sender else f"noreply@{domain}"
        recipients = _other_side(sender, cast_unique)
        to_addrs = [f"{p.name} <{p.email}>" for p in recipients] or [f"support@{domain}"]
        date_line = moment.strftime("%Y-%m-%d %H:%M:%S") + " -0700"
        header = [
            f"Message-ID: <{msg_id}@{domain}>",
            f"From: {from_addr}",
            f"To: {', '.join(to_addrs)}",
            f"Date: {date_line}",
            f"Subject: {subject if i == 1 else 'RE: ' + subject}",
        ]
        if prev is not None:
            header.append(f"In-Reply-To: <{prev[0]}@{domain}>")
        header.extend(f"{key}: {value}" for key, value in related)
        body = unit.clean_text
        if quote and prev is not None and i % 2 == 0:
            quoted = "\n".join(
                f"> {line}" if line else ">" for line in prev[2].clean_text.split("\n")
            )
            who = prev[3]
            who_str = f"{who.name} <{who.email}>" if who else prev[1]
            body = f"{body}\n\nOn {prev[1]}, {who_str} wrote:\n{quoted}"
        files[f"email/{msg_id}.txt"] = "\n".join(header) + "\n\n" + body + "\n"
        prepared_id = f"{msg_id}@{domain}"
        unit.cells = [
            Cell(s, e, f"{prepared_id}#p{k}", f"{prepared_id}#p{k}", i - 1)
            for k, (s, e) in enumerate(_paragraph_cells(unit.clean_text), start=1)
        ]
        units.append(unit)
        sources.append(SourceInfo(msg_id, f"email/{msg_id}.txt", SourceType.EMAIL, moment.date()))
        prev = (msg_id, date_line, unit, sender)
        moment += timedelta(minutes=17 + _stable_int("email-gap", plan.id, i) % 170)
    return AssembledDoc(DocKind.EMAIL, files, units, sources, subject or None, list(draft.notes))


def _assemble_prose(plan: DocPlan, rendered: RenderedDoc, draft: Draft) -> AssembledDoc:
    """A CRM note fills the skeleton's activity-note JSON (`subject`,
    `body`); meeting notes and docs keep the skeleton's code header above
    the writer's paragraphs. Cells index the text `prepare_sources` hands
    the extractor: the note body for a CRM note, the whole file otherwise."""
    path = _single_path(rendered)
    source_type = _PROSE_SOURCE_TYPE.get(plan.kind, SourceType.DOC)
    is_crm = source_type == SourceType.CRM
    header = "" if is_crm else _prose_header(rendered, draft.subject)
    units: list[AssembledUnit] = []
    texts: list[str] = []
    line_no = header.count("\n") + 1
    for i, du in enumerate(draft.units, start=1):
        unit = _unit(i, du)
        cells: list[Cell] = []
        pos = 0
        for line in unit.clean_text.split("\n"):
            if line.strip():
                prepared = "crm" if is_crm else f"L{line_no}"
                cells.append(Cell(pos, pos + len(line), f"{plan.id}#p{i}", prepared, 0))
            pos += len(line) + 1
            line_no += 1
        line_no += 1  # the blank separator line
        unit.cells = cells
        units.append(unit)
        texts.append(unit.clean_text)
    body = "\n\n".join(texts) + "\n"
    if is_crm:
        try:
            payload = json.loads(rendered.files[path])
        except ValueError:
            payload = {}
        if draft.subject is not None:
            payload["subject"] = draft.subject
        payload["body"] = body.rstrip("\n")
        text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    else:
        text = header + body
    source = SourceInfo(plan.id, path, source_type, plan.date)
    return AssembledDoc(plan.kind, {path: text}, units, [source], draft.subject, list(draft.notes))


# --------------------------------------------------------------------------
# Prepared text (spec section 3.2 check 8)
# --------------------------------------------------------------------------


@dataclass
class PreparedSource:
    source_id: str
    source_type: str
    path: str
    date: date
    text: str
    locators: list[LocatorSpan]

    def region(self, locator: str) -> tuple[int, int] | None:
        if self.source_type == SourceType.CRM.value and self.locators:
            return self.locators[0].start, self.locators[0].end
        for span in self.locators:
            if span.locator == locator:
                return span.start, span.end
        return None


def prepare_sources(assembled: AssembledDoc) -> list[PreparedSource]:
    """The exact text production hands the extractor for each source of
    this document: `FileConnector` (call, ticket, email) reading the
    finished files from a scratch directory, then `prepare_text`. Chat and
    prose kinds have no connector yet; their `RawSource` is built directly
    (chat as `Name: message` lines, a CRM note as a `crm` row with its
    path, everything else as a `doc`)."""
    out: list[PreparedSource] = []
    if assembled.kind in (DocKind.CALL, DocKind.TICKET, DocKind.EMAIL):
        with tempfile.TemporaryDirectory(prefix="wp4-prepare-") as tmp:
            root = Path(tmp)
            for rel, content in assembled.files.items():
                target = root / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
            connector = FileConnector(root)
            refs = {ref.ref.get("path"): ref for ref in connector.list_sources()}
            for info in assembled.sources:
                ref = refs.get(info.path)
                if ref is None:
                    raise ValueError(f"FileConnector did not list {info.path}")
                raw = connector.fetch(ref)
                prepared = prepare_text(raw)
                out.append(
                    PreparedSource(
                        raw.source_id,
                        raw.source_type.value,
                        info.path,
                        info.date,
                        prepared.text,
                        prepared.locators,
                    )
                )
        return out

    info = assembled.sources[0]
    if assembled.kind in (DocKind.SLACK, DocKind.TEAMS):
        text = chat_transcript_text(assembled)
        metadata: dict[str, str] = {}
    elif info.source_type == SourceType.CRM:
        # No connector reads activity-note JSON (WP3 review, not-fixed 10):
        # the stand-in is the note body, the text a CRM export surfaces.
        text = "\n\n".join(u.clean_text for u in assembled.units) + "\n"
        metadata = {"path": info.path, "crm_id": info.source_id}
    else:
        text = assembled.files[info.path]
        metadata = {}
    raw = RawSource(
        source_id=info.source_id,
        source_type=info.source_type,
        source_date=info.date,
        text=text,
        metadata=metadata,
    )
    prepared = prepare_text(raw)
    out.append(
        PreparedSource(
            info.source_id,
            info.source_type.value,
            info.path,
            info.date,
            prepared.text,
            prepared.locators,
        )
    )
    return out


def vendor_and_customer(cast: list[Person]) -> tuple[list[Person], list[Person]]:
    return (
        [p for p in cast if p.side == PersonSide.VENDOR],
        [p for p in cast if p.side == PersonSide.CUSTOMER],
    )
