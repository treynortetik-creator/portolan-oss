"""WP6 helper: a written document as editable text that remembers where
every character came from.

The noise stage (`corpus.generator.v2.noise`) edits a WP4 document's words
(ASR errors, truncation, mojibake, HTML leftovers, forwarded copies) and must
still say, for every label, exactly which characters of the NEW prepared
text carry it. This module makes that exact:

1. `parse_written` splits a document's files back into units (a call turn, a
   ticket comment, an email message body, a chat message, a prose
   paragraph). `noise.noise_document` noises a document only when rendering
   those units again reproduces its files and prepared text byte for byte.
2. Every unit character carries the set of ORIGINAL prepared-text positions
   (`Key` = source id, offset) it stands for. An `Edit` that replaces text
   hands the replaced keys to the new characters; a deletion records which
   operation removed each key (`DocText.deleted_by`).
3. `locate` renders the edited document, runs WP4's own preparation path
   (`assembly.prepare_sources`: `FileConnector` plus `prepare_text`), and
   finds every unit character's position in the new prepared text, checking
   each one character by character.
4. `remap_span` turns an old `[start, end)` into the new span (and any
   forwarded copies of it) or a drop reason.

Nothing here writes words. Structure strings (a forwarded-message header
block, an email quote attribution line) are rendered from values the
document's own headers already carry.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from corpus.generator.v2.assembly import (
    PROSE_KINDS,
    AssembledDoc,
    AssembledUnit,
    PreparedSource,
    SourceInfo,
    prepare_sources,
)
from corpus.generator.v2.planner import DocKind
from portolan.extract.schema import SourceType

__all__ = [
    "Edit",
    "Key",
    "Located",
    "ParseError",
    "Source",
    "SpanCopy",
    "SpanRemap",
    "DocText",
    "Unit",
    "apply_edits",
    "assign_original_keys",
    "locate",
    "parse_written",
    "remap_span",
    "remove_source",
    "remove_unit",
    "render_files",
]

Key = tuple[str, int]
NO_KEYS: frozenset[Key] = frozenset()

CHAT_KINDS = frozenset({DocKind.SLACK, DocKind.TEAMS})


class ParseError(ValueError):
    """The document's files are not in the shape WP4 renders; it is passed
    through un-noised rather than guessed at."""


@dataclass
class Unit:
    uid: int
    source_id: str
    author: str | None
    text: str
    keys: list[frozenset[Key]] = field(default_factory=list)
    # The noise operation that produced each character (None: an original
    # character, or one copied verbatim by a forward).
    ops: list[str | None] = field(default_factory=list)
    role: str = "writer"  # "writer" | "forward_copy" | "header" (code structure)
    meta: dict[str, Any] = field(default_factory=dict)
    removed: bool = False

    def __post_init__(self) -> None:
        if not self.keys:
            self.keys = [NO_KEYS] * len(self.text)
        if not self.ops:
            self.ops = [None] * len(self.text)


@dataclass
class Source:
    source_id: str
    path: str
    source_type: SourceType
    date: date
    meta: dict[str, Any] = field(default_factory=dict)
    removed: bool = False


@dataclass
class DocText:
    plan_id: str
    kind: DocKind
    date: date
    sources: list[Source]
    units: list[Unit]
    deleted_by: dict[Key, str] = field(default_factory=dict)

    def live_sources(self) -> list[Source]:
        return [s for s in self.sources if not s.removed]

    def live_units(self, source_id: str | None = None, role: str | None = None) -> list[Unit]:
        return [
            u
            for u in self.units
            if not u.removed
            and (source_id is None or u.source_id == source_id)
            and (role is None or u.role == role)
        ]

    def source(self, source_id: str) -> Source:
        return next(s for s in self.sources if s.source_id == source_id)

    def next_uid(self) -> int:
        return max((u.uid for u in self.units), default=-1) + 1


# --------------------------------------------------------------------------
# Edits
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Edit:
    """Replace `unit.text[start:end]` with `new`. `start == end` inserts."""

    start: int
    end: int
    new: str
    op: str


def apply_edits(doc: DocText, unit: Unit, edits: list[Edit]) -> None:
    """Apply non-overlapping edits to one unit. New characters inherit the
    keys of the characters they replace (so a misheard name still carries
    the span it sits in); a pure deletion records its operation in
    `doc.deleted_by`. Two insertions at one position are refused."""
    ordered = sorted(edits, key=lambda e: (e.start, e.end))
    for a, b in zip(ordered, ordered[1:], strict=False):
        if a.end > b.start or (a.start == a.end == b.start == b.end):
            raise ValueError(f"overlapping edits {a} and {b}")
    for e in ordered:
        if not (0 <= e.start <= e.end <= len(unit.text)):
            raise ValueError(f"edit {e} outside unit of length {len(unit.text)}")
    for e in reversed(ordered):
        replaced = unit.keys[e.start : e.end]
        union = frozenset().union(*replaced) if replaced else NO_KEYS
        if not e.new:
            for key in union:
                doc.deleted_by.setdefault(key, e.op)
        unit.text = unit.text[: e.start] + e.new + unit.text[e.end :]
        unit.keys[e.start : e.end] = [union] * len(e.new)
        unit.ops[e.start : e.end] = [e.op] * len(e.new)


def remove_unit(doc: DocText, unit: Unit, op: str) -> None:
    unit.removed = True
    for keys in unit.keys:
        for key in keys:
            doc.deleted_by.setdefault(key, op)


def remove_source(doc: DocText, source: Source, op: str) -> None:
    source.removed = True
    for unit in doc.units:
        if unit.source_id == source.source_id and not unit.removed:
            remove_unit(doc, unit, op)


# --------------------------------------------------------------------------
# Parsing (the inverse of `assembly.assemble`)
# --------------------------------------------------------------------------

_CALL_TS_RE = re.compile(r"^\[\d{2}:\d{2}:\d{2}\]")
_CALL_LINE_RE = re.compile(r"^\[(\d{2}:\d{2}:\d{2})\] (.+?): (.*)$")
# WP4's email quote block: `On <date>, <who> wrote:` then `> ` lines, last
# thing in the body (`assembly._assemble_email`).
_EMAIL_TAIL_RE = re.compile(r"\n\nOn [^\n]+ wrote:\n>[^\n]*(?:\n>[^\n]*)*\Z")


def parse_written(
    *,
    plan_id: str,
    kind: DocKind,
    doc_date: date,
    files: dict[str, str],
    prepared: list[tuple[str, str, str]],
    prepared_text: dict[str, str] | None = None,
) -> DocText:
    """`prepared` is `[(source_id, source_type, path), ...]` in the order
    WP4 recorded its prepared sources; `prepared_text` maps a source id to
    its prepared text (a Slack day needs it for display names)."""
    if kind == DocKind.CALL:
        return _parse_call(plan_id, doc_date, files, prepared)
    if kind == DocKind.TICKET:
        return _parse_ticket(plan_id, doc_date, files, prepared)
    if kind == DocKind.EMAIL:
        return _parse_email(plan_id, doc_date, files, prepared)
    if kind in CHAT_KINDS:
        return _parse_chat(plan_id, kind, doc_date, files, prepared, prepared_text)
    if kind in PROSE_KINDS:
        return _parse_prose(plan_id, kind, doc_date, files, prepared)
    raise ParseError(f"kind {kind} is not a written document kind")


def _one(files: dict[str, str], prepared: list[tuple[str, str, str]]) -> tuple[str, str, str]:
    if len(files) != 1 or len(prepared) != 1:
        raise ParseError(f"expected one file and one prepared source, got {sorted(files)}")
    source_id, source_type, path = prepared[0]
    if path not in files:
        raise ParseError(f"prepared source path {path!r} is not among the files")
    return source_id, source_type, path


def _parse_call(plan_id, doc_date, files, prepared) -> DocText:
    source_id, _st, path = _one(files, prepared)
    text = files[path]
    if not text.endswith("\n"):
        raise ParseError("call file does not end with a newline")
    lines = text[:-1].split("\n")
    first = next((i for i, line in enumerate(lines) if _CALL_TS_RE.match(line)), None)
    if first is None:
        raise ParseError("call has no timestamped line")
    header = "\n".join(lines[:first]) + ("\n" if first else "")
    units: list[Unit] = []
    for i, line in enumerate(lines[first:]):
        m = _CALL_LINE_RE.match(line)
        if m is None:
            raise ParseError(f"call line {first + i + 1} is not `[ts] Name: text`")
        units.append(Unit(i, source_id, m.group(2), m.group(3), meta={"ts": m.group(1)}))
    source = Source(source_id, path, SourceType.CALL, doc_date, meta={"header": header})
    return DocText(plan_id, DocKind.CALL, doc_date, [source], units)


def _parse_ticket(plan_id, doc_date, files, prepared) -> DocText:
    source_id, _st, path = _one(files, prepared)
    try:
        payload = json.loads(files[path])
    except ValueError as exc:
        raise ParseError(f"ticket is not JSON: {exc}") from exc
    comments = payload.get("comments")
    if not isinstance(comments, list) or not comments:
        raise ParseError("ticket has no comments")
    if str(payload.get("id") or plan_id) != source_id:
        raise ParseError("ticket id does not match its prepared source id")
    units = []
    for i, c in enumerate(comments):
        if not isinstance(c.get("body"), str) or "number" not in c:
            raise ParseError("ticket comment without a number or body")
        units.append(Unit(i, source_id, c.get("author"), c["body"], meta={"comment": dict(c)}))
    source = Source(source_id, path, SourceType.TICKET, doc_date, meta={"payload": payload})
    return DocText(plan_id, DocKind.TICKET, doc_date, [source], units)


def _header_value(header_lines: list[str], name: str) -> str:
    for line in header_lines:
        if line.lower().startswith(name.lower() + ":"):
            return line.split(":", 1)[1].strip()
    return ""


def _parse_email(plan_id, doc_date, files, prepared) -> DocText:
    by_path = {path: (sid, st) for sid, st, path in prepared}
    if set(by_path) != set(files):
        raise ParseError("email files and prepared sources differ")
    sources: list[Source] = []
    units: list[Unit] = []
    for i, (path, text) in enumerate(files.items()):
        if not text.endswith("\n") or "\n\n" not in text:
            raise ParseError(f"{path}: not a header block, a blank line and a body")
        cut = text.index("\n\n")
        header_lines = text[:cut].split("\n")
        body = text[cut + 2 : -1]
        tail_match = _EMAIL_TAIL_RE.search(body)
        writer_text = body[: tail_match.start()] if tail_match else body
        tail = body[tail_match.start() :] if tail_match else ""
        if not writer_text.strip():
            raise ParseError(f"{path}: empty message body")
        source_id = by_path[path][0]
        meta = {
            "header_lines": header_lines,
            "tail": tail,
            "from": _header_value(header_lines, "From"),
            "to": _header_value(header_lines, "To"),
            "date_line": _header_value(header_lines, "Date"),
            "subject": _header_value(header_lines, "Subject"),
            "message_id": _header_value(header_lines, "Message-ID").strip("<>"),
        }
        sources.append(Source(source_id, path, SourceType.EMAIL, doc_date, meta=meta))
        units.append(Unit(i, source_id, meta["from"], writer_text))
    return DocText(plan_id, DocKind.EMAIL, doc_date, sources, units)


_PREPARED_LINE_RE = re.compile(r"^\[L\d+\] (.*)$")


def _parse_chat(plan_id, kind, doc_date, files, prepared, prepared_text) -> DocText:
    source_id, _st, path = _one(files, prepared)
    try:
        messages = json.loads(files[path])
    except ValueError as exc:
        raise ParseError(f"chat day is not JSON: {exc}") from exc
    if not isinstance(messages, list) or not messages:
        raise ParseError("chat day has no messages")
    # A Slack message carries a user id; the extractor sees the display name
    # (`assembly.chat_transcript_text`), which only the prepared text holds.
    lines = [
        m.group(1)
        for m in map(_PREPARED_LINE_RE.match, (prepared_text or {}).get(source_id, "").split("\n"))
        if m
    ]
    units = []
    for i, m in enumerate(messages):
        if kind == DocKind.TEAMS:
            author, text = m["from"]["user"]["displayName"], m["body"]["content"]
        else:
            text = m["text"]
            line = lines[i] if i < len(lines) else ""
            if not line.endswith(f": {text}"):
                raise ParseError("a Slack message's name is not in the prepared text")
            author = line[: len(line) - len(text) - 2]
        if "\n" in text:
            raise ParseError("a chat message spans lines")
        units.append(Unit(i, source_id, author, text, meta={"message": copy.deepcopy(m)}))
    source = Source(source_id, path, SourceType.CHAT, doc_date)
    return DocText(plan_id, kind, doc_date, [source], units)


# A code-rendered header block (`Date:`, `Account:`, `Title:` ...) above a
# doc's or meeting note's paragraphs: structure, never noised as words.
_HEADER_LINE_RE = re.compile(r"^[A-Z][A-Za-z ]{0,30}: \S")


def _parse_prose(plan_id, kind, doc_date, files, prepared) -> DocText:
    source_id, source_type, path = _one(files, prepared)
    if SourceType(source_type) == SourceType.CRM:
        return _parse_crm_note(plan_id, kind, doc_date, files[path], source_id, path)
    text = files[path]
    if not text.endswith("\n"):
        raise ParseError("prose file does not end with a newline")
    paragraphs = text[:-1].split("\n\n")
    if any(not p for p in paragraphs):
        raise ParseError("prose file has an empty paragraph")
    units = [Unit(i, source_id, None, p) for i, p in enumerate(paragraphs)]
    if len(units) > 1 and all(_HEADER_LINE_RE.match(line) for line in paragraphs[0].split("\n")):
        units[0].role = "header"
    source = Source(source_id, path, SourceType(source_type), doc_date)
    return DocText(plan_id, kind, doc_date, [source], units)


def _parse_crm_note(plan_id, kind, doc_date, content: str, source_id: str, path: str) -> DocText:
    try:
        payload = json.loads(content)
    except ValueError as exc:
        raise ParseError(f"CRM note is not JSON: {exc}") from exc
    body = payload.get("body")
    if not isinstance(body, str) or not body.strip():
        raise ParseError("CRM note has no body")
    paragraphs = body.split("\n\n")
    if any(not p for p in paragraphs):
        raise ParseError("CRM note body has an empty paragraph")
    units = [Unit(i, source_id, None, p) for i, p in enumerate(paragraphs)]
    source = Source(source_id, path, SourceType.CRM, doc_date, meta={"payload": payload})
    return DocText(plan_id, kind, doc_date, [source], units)


# --------------------------------------------------------------------------
# Rendering (the same shapes `assembly` writes)
# --------------------------------------------------------------------------


def render_files(doc: DocText) -> dict[str, str]:
    kind = doc.kind
    files: dict[str, str] = {}
    for source in doc.live_sources():
        units = doc.live_units(source.source_id)
        if kind == DocKind.CALL:
            body = "\n".join(f"[{u.meta['ts']}] {u.author}: {u.text}" for u in units)
            files[source.path] = source.meta["header"] + body + "\n"
        elif kind == DocKind.TICKET:
            payload = dict(source.meta["payload"])
            payload["comments"] = [{**u.meta["comment"], "body": u.text} for u in units]
            files[source.path] = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
        elif kind == DocKind.EMAIL:
            writer = [u for u in units if u.role == "writer"]
            forwards = [u for u in units if u.role == "forward_copy"]
            body = "".join(u.text for u in writer)
            body += "".join(u.meta["prefix"] + u.text for u in forwards)
            body += source.meta["tail"]
            files[source.path] = "\n".join(source.meta["header_lines"]) + "\n\n" + body + "\n"
        elif kind in CHAT_KINDS:
            messages = []
            for u in units:
                m = copy.deepcopy(u.meta["message"])
                if kind == DocKind.TEAMS:
                    m["body"]["content"] = u.text
                else:
                    m["text"] = u.text
                messages.append(m)
            files[source.path] = json.dumps(messages, indent=2, ensure_ascii=False) + "\n"
        elif source.source_type == SourceType.CRM:
            payload = dict(source.meta["payload"])
            payload["body"] = "\n\n".join(u.text for u in units)
            files[source.path] = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
        else:
            files[source.path] = "\n\n".join(u.text for u in units) + "\n"
    return files


# --------------------------------------------------------------------------
# Locating unit characters in the prepared text
# --------------------------------------------------------------------------


@dataclass
class Located:
    files: dict[str, str]
    prepared: list[PreparedSource]
    # uid -> position of each character in its source's prepared text
    positions: dict[int, list[int | None]]
    mismatches: int = 0

    def prepared_for(self, source_id: str) -> PreparedSource:
        return next(p for p in self.prepared if p.source_id == source_id)


def _stub(doc: DocText, files: dict[str, str]) -> AssembledDoc:
    units = [
        AssembledUnit(
            index=i,
            author_person_id=None,
            author_name=u.author,
            marked_text=u.text,
            clean_text=u.text,
            spans=[],
            unbalanced=[],
            overlap=[],
            cells=[],
            source_index=0,
        )
        for i, u in enumerate(doc.live_units(), start=1)
    ]
    sources = [SourceInfo(s.source_id, s.path, s.source_type, s.date) for s in doc.live_sources()]
    return AssembledDoc(doc.kind, files, units, sources, None)


def _region(prepared: PreparedSource, locator: str) -> tuple[int, int] | None:
    for span in prepared.locators:
        if span.locator == locator:
            return span.start, span.end
    return None


def locate(doc: DocText) -> Located:
    """Render, prepare with WP4's own path, and position every character."""
    files = render_files(doc)
    prepared = prepare_sources(_stub(doc, files))
    by_source = {p.source_id: p for p in prepared}
    positions: dict[int, list[int | None]] = {}
    for source in doc.live_sources():
        p = by_source.get(source.source_id)
        units = doc.live_units(source.source_id)
        if p is None:
            for u in units:
                positions[u.uid] = [None] * len(u.text)
            continue
        if doc.kind == DocKind.CALL:
            _positions_call(source, units, p, positions)
        elif doc.kind == DocKind.TICKET:
            _positions_ticket(source, units, p, positions)
        elif doc.kind == DocKind.EMAIL:
            _positions_email(units, p, positions)
        elif doc.kind in CHAT_KINDS:
            _positions_chat(units, p, positions)
        elif source.source_type == SourceType.CRM:
            _positions_crm(units, p, positions)
        else:
            _positions_doc(units, p, positions)
    mismatches = 0
    for source in doc.live_sources():
        p = by_source.get(source.source_id)
        for u in doc.live_units(source.source_id):
            pos = positions[u.uid]
            for i, at in enumerate(pos):
                if at is None:
                    continue
                if p is None or at >= len(p.text) or p.text[at] != u.text[i]:
                    pos[i] = None
                    mismatches += 1
    return Located(files, prepared, positions, mismatches)


def _positions_call(source: Source, units: list[Unit], p: PreparedSource, out) -> None:
    offset = len(source.meta["header"])
    for u in units:
        prefix = f"[{u.meta['ts']}] {u.author}: "
        base = offset + len(prefix)
        out[u.uid] = [base + i for i in range(len(u.text))]
        offset += len(prefix) + len(u.text) + 1


def _positions_ticket(source: Source, units: list[Unit], p: PreparedSource, out) -> None:
    for u in units:
        number = u.meta["comment"].get("number")
        loc = f"{source.source_id}#{number}"
        region = _region(p, loc)
        if region is None:
            out[u.uid] = [None] * len(u.text)
            continue
        header = f"[{loc}] {u.author}: " if u.author else f"[{loc}] "
        body = u.text
        lead = len(body) - len(body.lstrip())
        keep = len(body.strip())
        base = region[0] + len(header)
        out[u.uid] = [
            base + (i - lead) if lead <= i < lead + keep else None for i in range(len(body))
        ]


def _paragraph_at(text: str, para: str, start: int) -> int:
    """Index of `para` in `text` at or after `start`, standing as a whole
    blank-line-separated paragraph, or -1."""
    idx = text.find(para, start)
    while idx != -1:
        before, after = text[:idx], text[idx + len(para) :]
        ok_before = not before.strip() or re.search(r"\n\s*\n\s*$", before) is not None
        ok_after = not after.strip() or re.match(r"\s*\n\s*\n", after) is not None
        if ok_before and ok_after:
            return idx
        idx = text.find(para, idx + 1)
    return -1


def _positions_email(units: list[Unit], p: PreparedSource, out) -> None:
    for u in units:
        out[u.uid] = [None] * len(u.text)
    cursor_unit, cursor_char = 0, 0
    for span in p.locators:
        prefix = f"[{span.locator}] "
        para_start = span.start + len(prefix)
        para = p.text[para_start : min(span.end, len(p.text))].rstrip("\n")
        if not para:
            continue
        for ui in range(cursor_unit, len(units)):
            u = units[ui]
            idx = _paragraph_at(u.text, para, cursor_char if ui == cursor_unit else 0)
            if idx == -1:
                continue
            for k in range(len(para)):
                out[u.uid][idx + k] = para_start + k
            cursor_unit, cursor_char = ui, idx + len(para)
            break


def _positions_chat(units: list[Unit], p: PreparedSource, out) -> None:
    for line_no, u in enumerate(units, start=1):
        region = _region(p, f"L{line_no}")
        if region is None:
            out[u.uid] = [None] * len(u.text)
            continue
        base = region[0] + len(f"[L{line_no}] {u.author}: ")
        out[u.uid] = [base + i for i in range(len(u.text))]


def _positions_crm(units: list[Unit], p: PreparedSource, out) -> None:
    # `prepare_sources` hands a CRM note over as its body paragraphs.
    body = "\n\n".join(u.text for u in units) + "\n"
    prefix_len = len(p.text) - len(body)
    offset = 0
    for u in units:
        out[u.uid] = [prefix_len + offset + i for i in range(len(u.text))]
        offset += len(u.text) + 2


def _positions_doc(units: list[Unit], p: PreparedSource, out) -> None:
    line_no = 1
    for u in units:
        pos: list[int | None] = []
        region = _region(p, f"L{line_no}")
        col = 0
        for ch in u.text:
            if ch == "\n":
                pos.append(None)
                line_no += 1
                region = _region(p, f"L{line_no}")
                col = 0
                continue
            if region is None:
                pos.append(None)
            else:
                pos.append(region[0] + len(f"[L{line_no}] ") + col)
            col += 1
        out[u.uid] = pos
        line_no += 2  # this paragraph's last line, then the blank separator


def assign_original_keys(doc: DocText, located: Located) -> None:
    """Tag every visible character of the unedited document with its own
    prepared-text position, the key labels are remapped by."""
    for u in doc.live_units():
        u.keys = [
            frozenset({(u.source_id, at)}) if at is not None else NO_KEYS
            for at in located.positions[u.uid]
        ]
        u.ops = [None] * len(u.text)


# --------------------------------------------------------------------------
# Remapping a span
# --------------------------------------------------------------------------


@dataclass
class SpanCopy:
    """One place a span now lives: a unit, `[start, end)` in its source's
    new prepared text, and the operations that touched it."""

    unit: Unit
    source_id: str
    start: int
    end: int
    text: str
    ops: set[str]
    prepared_locator: str | None


@dataclass
class SpanRemap:
    primary: SpanCopy | None
    copies: list[SpanCopy]
    drop_reason: str | None = None
    drop_op: str | None = None
    copy_failures: list[str] = field(default_factory=list)


def key_index(doc: DocText) -> dict[Key, list[tuple[Unit, int]]]:
    index: dict[Key, list[tuple[Unit, int]]] = {}
    for u in doc.live_units():
        for i, keys in enumerate(u.keys):
            for key in keys:
                index.setdefault(key, []).append((u, i))
    return index


def _locator_at(prepared: PreparedSource, start: int, end: int) -> str | None:
    for span in prepared.locators:
        if span.start <= start and end <= span.end:
            return span.locator
    return None


def remap_span(
    doc: DocText,
    index: dict[Key, list[tuple[Unit, int]]],
    located: Located,
    *,
    source_id: str,
    start: int,
    end: int,
    cut_ops: frozenset[str],
) -> SpanRemap:
    """Where the characters `[start, end)` of `source_id`'s ORIGINAL
    prepared text are now. The copy in the unit that held them originally is
    the primary; a verbatim copy in a forwarded block is a copy. A span
    loses its primary when any of its characters was removed by a
    `cut_ops` operation, split by other text, or is no longer visible."""
    span_keys = {(source_id, k) for k in range(start, end)}
    hits: dict[int, tuple[Unit, list[int]]] = {}
    for key in span_keys:
        for unit, i in index.get(key, []):
            hits.setdefault(unit.uid, (unit, []))[1].append(i)

    primary: SpanCopy | None = None
    copies: list[SpanCopy] = []
    reason: str | None = None
    reason_op: str | None = None
    copy_failures: list[str] = []
    for unit, idxs in hits.values():
        is_primary = unit.role == "writer" and unit.source_id == source_id
        result, why, op = _one_copy(doc, unit, sorted(set(idxs)), span_keys, located, cut_ops)
        if result is None:
            if is_primary:
                reason, reason_op = why, op
            else:
                copy_failures.append(f"{unit.source_id}: {why}")
            continue
        if is_primary:
            primary = result
        else:
            copies.append(result)
    if primary is None and reason is None:
        cut = sorted({doc.deleted_by[k] for k in span_keys if k in doc.deleted_by})
        reason = "span_removed" if cut else "span_not_found"
        reason_op = cut[0] if cut else None
    copies.sort(key=lambda c: (c.source_id, c.start))
    return SpanRemap(primary, copies, reason, reason_op, copy_failures)


def _one_copy(
    doc: DocText,
    unit: Unit,
    idxs: list[int],
    span_keys: set[Key],
    located: Located,
    cut_ops: frozenset[str],
) -> tuple[SpanCopy | None, str | None, str | None]:
    i0, i1 = idxs[0], idxs[-1] + 1
    present: set[Key] = set()
    ops: set[str] = set()
    for i in range(i0, i1):
        keys = unit.keys[i]
        inside = keys & span_keys
        if inside:
            present |= inside
            if keys - span_keys:
                return None, "span_merged_with_outside_text", unit.ops[i]
        elif keys or unit.ops[i] is None:
            return None, "span_split", None
        if unit.ops[i] is not None:
            ops.add(unit.ops[i])
    missing = span_keys - present
    for key in missing:
        op = doc.deleted_by.get(key)
        if op is None:
            return None, "span_split", None
        if op in cut_ops:
            return None, "span_cut", op
        ops.add(op)
    positions = located.positions.get(unit.uid)
    if positions is None:
        return None, "span_not_visible", None
    first = positions[i0]
    if first is None or any(positions[i] != first + (i - i0) for i in range(i0, i1)):
        return None, "span_not_visible", None
    prepared = located.prepared_for(unit.source_id)
    text = prepared.text[first : first + (i1 - i0)]
    if text != unit.text[i0:i1]:
        return None, "span_not_visible", None
    locator = _locator_at(prepared, first, first + len(text))
    return (
        SpanCopy(unit, unit.source_id, first, first + len(text), text, ops, locator),
        None,
        None,
    )
