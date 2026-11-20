"""Builds locator-annotated "prepared text" for each corpus source type.

This is a stand-in for `portolan.ingest.prepare_text` (build-contracts.md,
wave A2), which does not exist yet and which this module must not import.
Everything downstream (`extraction.py`) only depends on `PreparedSource`
(source id/type/date, participants, locator-annotated text) — a later wave
can drop this file in favor of the real connector's `prepare_text` without
`extraction.py` changing.

Locator conventions, per the wave B4 build instructions:
- **calls**: the corpus generator already injects `[HH:MM:SS]` at the start
  of every line; read as-is.
- **tickets**: JSON with numbered comments; each comment line is prefixed
  `[<ticket-id>#<comment-number>]`.
- **emails**: a `[msg:<Message-ID>]` header line, then each paragraph
  (split on blank lines) prefixed `[¶<n>]`, 1-indexed.
- **RSS items**: one line, `[url:<link> <date>]`.
- **CRM rows**: one line per row, `[<object>:<row-id>]` (`account:A-1010`,
  `contact:C-2001`, `opportunity:O-3001`) followed by its fields.
"""

from __future__ import annotations

import csv
import datetime as dt
import email as email_lib
import json
from dataclasses import dataclass, field
from pathlib import Path

from portolan.bench.truth import ManifestSource
from portolan.extract.schema import Participant, SourceType

DEFAULT_CORPUS_ROOT = Path("corpus/out")


@dataclass(frozen=True)
class PreparedSource:
    """Locator-annotated text plus the envelope metadata `extraction.py`
    needs to build a `SourceEnvelope` — everything `SourceEnvelope` wants
    except `registry_context` and `vocabulary`, which the bench has no
    registry to fill in yet (see `run_extraction`)."""

    source_id: str
    source_type: SourceType
    source_date: dt.date
    participants: list[Participant] = field(default_factory=list)
    text: str = ""


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _display_name(person_id: str) -> str:
    """Best-effort reversal of a truth-table person id back to a display
    name: ids are slugified names (`sam-okafor` -> `Sam Okafor`), so this
    is exact for every person id in the corpus without needing to load the
    truth table just to render a prompt's participant list."""
    return " ".join(part.capitalize() for part in person_id.split("-"))


def _participants_for(manifest_source: ManifestSource) -> list[Participant]:
    return [Participant(name=_display_name(pid)) for pid in manifest_source.participants]


# --------------------------------------------------------------------------
# calls
# --------------------------------------------------------------------------


def _prepare_call(path: Path) -> str:
    # Calls already carry `[HH:MM:SS]` locators at the start of every line.
    return _read_text(path).rstrip("\n")


# --------------------------------------------------------------------------
# tickets
# --------------------------------------------------------------------------


def _prepare_ticket(path: Path) -> str:
    data = json.loads(_read_text(path))
    ticket_id = data["id"]
    lines = [f"Ticket {ticket_id}: {data.get('subject', '')}".rstrip()]
    for comment in data.get("comments", []):
        number = comment["number"]
        author = comment.get("author", "unknown")
        body = " ".join(comment.get("body", "").split())
        lines.append(f"[{ticket_id}#{number}] {author}: {body}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# emails
# --------------------------------------------------------------------------


def _split_paragraphs(body: str) -> list[str]:
    paragraphs = [p.strip() for p in body.replace("\r\n", "\n").split("\n\n")]
    return [p for p in paragraphs if p]


def _prepare_email(path: Path) -> str:
    raw = _read_text(path)
    msg = email_lib.message_from_string(raw)
    message_id = (msg.get("Message-ID") or "").strip().strip("<>")
    body = msg.get_payload()
    if not isinstance(body, str):
        # Defensive: a payload the stdlib parser didn't treat as a flat
        # string (e.g. no blank-line-terminated header block). Fall back to
        # everything after the first blank line of the raw text.
        body = raw.split("\n\n", 1)[-1]

    lines = [f"[msg:{message_id}]"]
    for header in ("From", "To", "Subject", "Date"):
        value = msg.get(header)
        if value:
            lines.append(f"{header}: {value}")

    for i, paragraph in enumerate(_split_paragraphs(body), start=1):
        flat = " ".join(paragraph.split())
        lines.append(f"[¶{i}] {flat}")

    return "\n".join(lines)


# --------------------------------------------------------------------------
# RSS
# --------------------------------------------------------------------------


def _prepare_rss(path: Path, item_id: str) -> tuple[str, dt.date]:
    items = json.loads(_read_text(path))
    for item in items:
        if item.get("id") == item_id:
            link = item.get("link", "")
            item_date = item.get("date", "")
            title = item.get("title", "")
            summary = item.get("summary", "")
            text = f"[url:{link} {item_date}] {title} — {summary}"
            return text, dt.date.fromisoformat(item_date)
    raise KeyError(f"RSS item {item_id!r} not found in {path}")


# --------------------------------------------------------------------------
# CRM
# --------------------------------------------------------------------------

_CRM_KIND_BY_STEM = {"accounts": "account", "contacts": "contact"}


def _crm_kind_for(path: Path) -> str:
    stem = path.stem
    if stem.startswith("opportunities"):
        return "opportunity"
    return _CRM_KIND_BY_STEM.get(stem, stem)


def _crm_row_id(kind: str, row: dict[str, str]) -> str:
    if kind == "opportunity":
        return row.get("opportunity_id", "")
    return row.get("crm_id", "")


def _prepare_crm(path: Path, row_ids: set[str] | None) -> str:
    """One `[<kind>:<row-id>]` line per row. `row_ids`, when given,
    restricts output to those rows (the manifest source's own placements
    tell us which row(s) it represents — several manifest CRM sources can
    share one physical CSV, e.g. `S-crm-A1002` and `S-crm-A1010` both point
    at `crm/accounts.csv`). `None`/empty means every row in the file."""
    kind = _crm_kind_for(path)
    lines: list[str] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row_id = _crm_row_id(kind, row)
            if row_ids and row_id not in row_ids:
                continue
            fields = ", ".join(f"{k}={v}" for k, v in row.items())
            lines.append(f"[{kind}:{row_id}] {fields}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# dispatch
# --------------------------------------------------------------------------

_PREPARERS = {
    SourceType.CALL: lambda path, src: _prepare_call(path),
    SourceType.TICKET: lambda path, src: _prepare_ticket(path),
    SourceType.EMAIL: lambda path, src: _prepare_email(path),
}


def load_prepared_source(
    manifest_source: ManifestSource,
    corpus_root: Path | str = DEFAULT_CORPUS_ROOT,
) -> PreparedSource:
    """Read `manifest_source`'s underlying corpus file and build its
    locator-annotated prepared text."""
    root = Path(corpus_root)
    if not manifest_source.path:
        raise ValueError(f"manifest source {manifest_source.id!r} has no path")
    path = root / manifest_source.path
    source_type = SourceType(manifest_source.type)
    source_date = manifest_source.date

    if source_type in _PREPARERS:
        text = _PREPARERS[source_type](path, manifest_source)
    elif source_type == SourceType.RSS:
        text, parsed_date = _prepare_rss(path, manifest_source.id)
        source_date = source_date or parsed_date
    elif source_type == SourceType.CRM:
        row_ids = {p.line_or_comment for p in manifest_source.facts} or None
        text = _prepare_crm(path, row_ids)
    else:
        raise ValueError(
            f"corpus_loader has no handler for source_type={source_type!r} "
            f"(source {manifest_source.id!r})"
        )

    if source_date is None:
        raise ValueError(f"source {manifest_source.id!r} has no source_date")

    return PreparedSource(
        source_id=manifest_source.id,
        source_type=source_type,
        source_date=source_date,
        participants=_participants_for(manifest_source),
        text=text,
    )
