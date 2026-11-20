"""`FileConnector`: reads the `corpus/out` layout (build-contracts.md
"Connectors"; portolan-spec-v2.md sec 5.3 "Upload and export as the
fallback for everything else"): a directory of files standing in for what
a customer's call recorder, ticket export, CRM export and RSS feeds would
supply over the network.

Layout read (`root` = e.g. `corpus/out`):
- `calls/*.txt` — one call transcript per file, `[HH:MM:SS] Speaker: ...`
  lines (`portolan/ingest/prepare.py`'s `_CALL_LINE_RE` shape). One
  `SourceRef` per file.
- `tickets/*.json` — one ticket per file, `{id, requester, account,
  created, subject, comments: [{number, author, body}]}`. One `SourceRef`
  per file, keyed by the JSON's own `id` field.
- `email/*.eml` and `email/*.txt` — one message per file, an RFC-2822-ish
  header block (`Message-ID`, `From`, `To`, `Date`, `Subject`) followed by
  a blank line and the body. Both extensions are read (the real corpus
  under `corpus/out/email/` ships `.txt`; `.eml` is read too since that is
  the extension a real mail export usually carries). One `SourceRef` per
  file.
- `rss/*.json` — one feed per file, a JSON array of `{id, date, title,
  link, summary}` items. One `SourceRef` **per item**, not per file — a
  feed file accumulates many dated entries the way a real RSS.app export
  would (spec sec 5.3).
- `crm/*.csv` — one `SourceRef` **per row**, `source_type` `crm`. Three
  row shapes are recognised by their header columns, sniffed rather than
  by filename (a real CRM export's exact filenames vary):
    - accounts (`crm_id`, `kind`, `parent_crm_id`, `domain`, ...)
    - contacts (`crm_id`, `title`, `account_crm_id`, `email`, ...)
    - opportunities (`opportunity_id`, `account_crm_id`, `export_date`, ...)
  `crm_paths()` also exposes the accounts/contacts CSV pair, so the
  registry can be bulk-imported (`Registry.import_crm`) before
  `list_sources`; the per-row sources are still extracted like any other.

Source-id scheme: for call/ticket/email/rss, `source_id` is the file's own
id (`S-call-007`, the ticket JSON's `id` field, `S-email-021-thread`, the
feed item's own `id`). CRM rows have no per-row id in a raw CSV, so one is
minted: `S-crm-<crm_id-without-dashes>` for an account or contact row
(`A-1010` -> `S-crm-A1010`), and `S-crm-opp-<account_crm_id-without-
dashes>` for an opportunity row, with `-v2`, `-v3`, ... on later snapshots
of the same account's opportunity (ordered by `export_date`): each
re-export is a distinct, dated source. The scheme matches
`corpus/truth-table.yaml`'s source ids (`S-crm-A1002`, `S-crm-opp-A1010`,
`S-crm-opp-A1010-v2`, ...).

Participants: call speaker names, parsed from the transcript's own
`Speaker Name:` prefixes, plus any participant listed in `manifest.json`
(when present) not already captured that way — the corpus's own
`corpus/out/manifest.json` (present only in synthetic/bench corpora, never
required) occasionally lists a silent participant a transcript's spoken
lines don't name. Email participants come from the `From`/`To` headers.
Ticket participants are just the ticket's `requester`. RSS and CRM sources
carry no participants (spec: neither has a natural "who is present" model
in a raw feed item or CRM row).

`account_hints` come from `manifest.json`'s per-source `accounts` list
when the manifest is present, else `[]`. The manifest is never required:
without it, sources go through the filter on participant/registry signals
alone (spec sec 5.2).

`original_bytes` is the raw file for a source that owns a whole file (call,
ticket, email). RSS items and CRM rows share a file, so each is serialized
to its own canonical JSON (`json.dumps(..., sort_keys=True)`): two items
from one file must not share a stored original or a content hash.
"""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

from portolan.connectors.base import Connector, SourceRef, source_day
from portolan.extract.schema import SourceType
from portolan.ingest.models import Participant, RawSource
from portolan.util import clock

__all__ = ["CrmPaths", "FileConnector"]


@dataclass(frozen=True)
class CrmPaths:
    """What `FileConnector.crm_paths()` returns: the accounts/contacts CSV
    pair `Registry.import_crm(conn, accounts_csv, contacts_csv, ...)`
    wants, found by filename (`crm/accounts.csv`, `crm/contacts.csv`)."""

    accounts: Path
    contacts: Path


def _updated_at(day: date) -> datetime:
    return datetime.combine(day, datetime.min.time(), tzinfo=UTC)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


_CALL_SPEAKER_RE = re.compile(r"^\[(\d{2}:\d{2}:\d{2})\]\s*([^:]+):")
_HEADER_LINE_RE = re.compile(r"^([A-Za-z][A-Za-z-]*):\s?(.*)$")
_ADDR_RE = re.compile(r"^(?P<name>.*?)\s*<(?P<email>[^>]+)>$")


def _parse_email_headers(text: str) -> tuple[dict[str, str], str]:
    """Split a `Message-ID:`/`From:`/`To:`/`Date:`/`Subject:` header block
    from the body that follows the first blank line. Not a full RFC 2822
    parser (no folded headers, no MIME): this corpus's `.txt`/`.eml`
    files are flat single-part messages."""
    lines = text.split("\n")
    headers: dict[str, str] = {}
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            break
        match = _HEADER_LINE_RE.match(line)
        if match:
            headers[match.group(1)] = match.group(2)
        i += 1
    body = "\n".join(lines[i:])
    return headers, body


def _email_date(headers: dict[str, str]) -> date:
    """The `Date` header's day: an ISO date first token (this corpus's
    `2026-03-04 09:00:00 -0700`), else RFC 2822 (`Tue, 08 Sep 2026
    18:30:00 -0700`, what a real `.eml` export carries) in the org's zone,
    else today."""
    value = headers.get("Date", "").strip()
    try:
        return date.fromisoformat(value.split(" ")[0])
    except ValueError:
        pass
    try:
        return source_day(parsedate_to_datetime(value))
    except (TypeError, ValueError, IndexError):
        return clock.today()


def _parse_address_list(value: str) -> list[Participant]:
    participants: list[Participant] = []
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        match = _ADDR_RE.match(chunk)
        if match:
            participants.append(
                Participant(name=match.group("name").strip(), email=match.group("email").strip())
            )
        else:
            participants.append(Participant(name=chunk))
    return participants


def _render_crm_row_text(object_type: str, row: dict[str, str]) -> str:
    """A plain `key=value, ...` rendering of one CRM row — good enough for
    the registry scan and the extractor to read; nothing here needs to be
    pretty, only legible and complete."""
    fields = ", ".join(f"{key}={value}" for key, value in row.items() if value not in (None, ""))
    return f"{object_type}: {fields}"


class FileConnector:
    """Reads `corpus/out`-shaped directories. See module docstring."""

    name = "file"

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self._manifest_by_id = self._load_manifest()

    # ------------------------------------------------------------------
    # manifest.json (optional; never required)
    # ------------------------------------------------------------------

    def _load_manifest(self) -> dict[str, dict[str, Any]]:
        path = self.root / "manifest.json"
        if not path.is_file():
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        return {entry["id"]: entry for entry in data.get("sources", [])}

    def _manifest_entry(self, source_id: str) -> dict[str, Any] | None:
        return self._manifest_by_id.get(source_id)

    def _account_hints(self, source_id: str) -> list[str]:
        entry = self._manifest_entry(source_id)
        if not entry:
            return []
        return list(entry.get("accounts") or [])

    # ------------------------------------------------------------------
    # CRM bulk-import paths
    # ------------------------------------------------------------------

    def crm_paths(self) -> CrmPaths | None:
        accounts = self.root / "crm" / "accounts.csv"
        contacts = self.root / "crm" / "contacts.csv"
        if accounts.is_file() and contacts.is_file():
            return CrmPaths(accounts=accounts, contacts=contacts)
        return None

    # ------------------------------------------------------------------
    # list_sources
    # ------------------------------------------------------------------

    def list_sources(self, since: date | None = None) -> Iterable[SourceRef]:
        yield from self._list_calls(since)
        yield from self._list_tickets(since)
        yield from self._list_emails(since)
        yield from self._list_rss(since)
        yield from self._list_crm(since)

    def _list_calls(self, since: date | None) -> Iterable[SourceRef]:
        for path in sorted(self.root.glob("calls/*.txt")):
            source_id = path.stem
            row_date = self._call_date(source_id)
            if since is not None and row_date < since:
                continue
            yield SourceRef(
                source_id=source_id,
                source_type=SourceType.CALL,
                updated_at=_updated_at(row_date),
                ref={"path": str(path.relative_to(self.root))},
            )

    def _call_date(self, source_id: str) -> date:
        entry = self._manifest_entry(source_id)
        if entry and entry.get("date"):
            return date.fromisoformat(entry["date"])
        # No manifest date: today, never file mtime (git does not preserve mtime).
        return clock.today()

    def _list_tickets(self, since: date | None) -> Iterable[SourceRef]:
        for path in sorted(self.root.glob("tickets/*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            source_id = payload.get("id", path.stem)
            created = payload.get("created")
            row_date = date.fromisoformat(created) if created else clock.today()
            if since is not None and row_date < since:
                continue
            yield SourceRef(
                source_id=source_id,
                source_type=SourceType.TICKET,
                updated_at=_updated_at(row_date),
                ref={"path": str(path.relative_to(self.root))},
            )

    def _list_emails(self, since: date | None) -> Iterable[SourceRef]:
        paths = sorted(
            {*self.root.glob("email/*.eml"), *self.root.glob("email/*.txt")},
            key=lambda p: p.name,
        )
        for path in paths:
            source_id = path.stem
            headers, _ = _parse_email_headers(path.read_text(encoding="utf-8"))
            row_date = _email_date(headers)
            if since is not None and row_date < since:
                continue
            yield SourceRef(
                source_id=source_id,
                source_type=SourceType.EMAIL,
                updated_at=_updated_at(row_date),
                ref={"path": str(path.relative_to(self.root))},
            )

    def _list_rss(self, since: date | None) -> Iterable[SourceRef]:
        for path in sorted(self.root.glob("rss/*.json")):
            items = json.loads(path.read_text(encoding="utf-8"))
            for item in items:
                row_date = date.fromisoformat(item["date"])
                if since is not None and row_date < since:
                    continue
                yield SourceRef(
                    source_id=item["id"],
                    source_type=SourceType.RSS,
                    updated_at=_updated_at(row_date),
                    ref={"path": str(path.relative_to(self.root))},
                )

    def _list_crm(self, since: date | None) -> Iterable[SourceRef]:
        opp_occurrences: dict[str, list[tuple[dict[str, str], Path]]] = {}

        for path in sorted(self.root.glob("crm/*.csv")):
            rows = _read_csv(path)
            if not rows:
                continue
            fieldnames = set(rows[0].keys())

            if "opportunity_id" in fieldnames:
                for row in rows:
                    account_crm_id = row.get("account_crm_id", "")
                    opp_occurrences.setdefault(account_crm_id, []).append((row, path))
                continue

            if "title" in fieldnames and "account_crm_id" in fieldnames:
                object_type = "contact"
            elif "crm_id" in fieldnames:
                object_type = "account"
            else:
                # Unrecognised CSV shape under crm/ -- not one of the
                # three row shapes this connector knows how to read.
                continue

            for row in rows:
                ref = self._crm_ref_for_row(row, path, object_type)
                if ref is None:
                    continue
                if since is not None and ref.updated_at.date() < since:
                    continue
                yield ref

        for account_crm_id, occurrences in opp_occurrences.items():
            occurrences.sort(key=lambda item: (item[0].get("export_date") or "", str(item[1])))
            for index, (row, path) in enumerate(occurrences, start=1):
                ref = self._crm_ref_for_opportunity(row, path, account_crm_id, index)
                if since is not None and ref.updated_at.date() < since:
                    continue
                yield ref

    def _crm_ref_for_row(
        self, row: dict[str, str], path: Path, object_type: str
    ) -> SourceRef | None:
        crm_id = row.get("crm_id")
        if not crm_id:
            return None
        source_id = f"S-crm-{crm_id.replace('-', '')}"
        row_date = self._crm_row_date(source_id, row, object_type)
        return SourceRef(
            source_id=source_id,
            source_type=SourceType.CRM,
            updated_at=_updated_at(row_date),
            ref={
                "path": str(path.relative_to(self.root)),
                "object": object_type,
                "crm_id": crm_id,
                "row": row,
            },
        )

    def _crm_ref_for_opportunity(
        self, row: dict[str, str], path: Path, account_crm_id: str, occurrence: int
    ) -> SourceRef:
        opportunity_id = row.get("opportunity_id", "")
        base = f"S-crm-opp-{account_crm_id.replace('-', '')}"
        source_id = base if occurrence <= 1 else f"{base}-v{occurrence}"
        row_date = self._crm_row_date(source_id, row, "opportunity")
        return SourceRef(
            source_id=source_id,
            source_type=SourceType.CRM,
            updated_at=_updated_at(row_date),
            ref={
                "path": str(path.relative_to(self.root)),
                "object": "opportunity",
                "crm_id": opportunity_id,
                "row": row,
            },
        )

    def _crm_row_date(self, source_id: str, row: dict[str, str], object_type: str) -> date:
        if object_type == "opportunity":
            export_date = row.get("export_date")
            if export_date:
                try:
                    return date.fromisoformat(export_date)
                except ValueError:
                    pass
        entry = self._manifest_entry(source_id)
        if entry and entry.get("date"):
            return date.fromisoformat(entry["date"])
        # accounts.csv/contacts.csv rows carry no date column in this layout.
        return clock.today()

    # ------------------------------------------------------------------
    # fetch
    # ------------------------------------------------------------------

    def fetch(self, ref: SourceRef) -> RawSource:
        handler = {
            SourceType.CALL: self._fetch_call,
            SourceType.TICKET: self._fetch_ticket,
            SourceType.EMAIL: self._fetch_email,
            SourceType.RSS: self._fetch_rss,
            SourceType.CRM: self._fetch_crm,
        }.get(ref.source_type)
        if handler is None:
            raise ValueError(f"FileConnector cannot fetch source_type={ref.source_type!r}")
        return handler(ref)

    def _fetch_call(self, ref: SourceRef) -> RawSource:
        path = self.root / ref.ref["path"]
        original_bytes = path.read_bytes()
        text = original_bytes.decode("utf-8")

        speakers: dict[str, None] = {}
        for line in text.splitlines():
            match = _CALL_SPEAKER_RE.match(line)
            if match:
                name = match.group(2).strip()
                if name:
                    speakers.setdefault(name, None)
        participants = [Participant(name=name) for name in speakers]

        entry = self._manifest_entry(ref.source_id)
        if entry:
            existing = {p.name.strip().lower() for p in participants}
            for slug in entry.get("participants") or []:
                display = slug.replace("-", " ").title()
                if display.lower() not in existing:
                    participants.append(Participant(name=display))
                    existing.add(display.lower())

        return RawSource(
            source_id=ref.source_id,
            source_type=SourceType.CALL,
            source_date=self._call_date(ref.source_id),
            participants=participants,
            account_hints=self._account_hints(ref.source_id),
            text=text,
            original_bytes=original_bytes,
            original_content_type="text/plain; charset=utf-8",
            metadata={},
        )

    def _fetch_ticket(self, ref: SourceRef) -> RawSource:
        path = self.root / ref.ref["path"]
        original_bytes = path.read_bytes()
        payload = json.loads(original_bytes.decode("utf-8"))
        comments = payload.get("comments") or []
        flat_text = "\n\n".join(f"{c.get('author', '')}: {c['body']}" for c in comments)
        requester = payload.get("requester")
        participants = [Participant(name=requester)] if requester else []
        created = payload.get("created")
        source_date = date.fromisoformat(created) if created else clock.today()

        return RawSource(
            source_id=ref.source_id,
            source_type=SourceType.TICKET,
            source_date=source_date,
            participants=participants,
            account_hints=self._account_hints(ref.source_id),
            text=flat_text,
            original_bytes=original_bytes,
            original_content_type="application/json",
            metadata={"comments": comments, "subject": payload.get("subject", "")},
        )

    def _fetch_email(self, ref: SourceRef) -> RawSource:
        path = self.root / ref.ref["path"]
        original_bytes = path.read_bytes()
        text = original_bytes.decode("utf-8")
        headers, body = _parse_email_headers(text)

        from_participants = _parse_address_list(headers.get("From", ""))
        to_participants = _parse_address_list(headers.get("To", ""))

        source_date = _email_date(headers)

        message_id = headers.get("Message-ID", "").strip("<>") or ref.source_id
        from_email = from_participants[0].email if from_participants else None
        body = body.strip("\n")

        return RawSource(
            source_id=ref.source_id,
            source_type=SourceType.EMAIL,
            source_date=source_date,
            participants=[*from_participants, *to_participants],
            account_hints=self._account_hints(ref.source_id),
            text=(body + "\n") if body else "",
            original_bytes=original_bytes,
            original_content_type="message/rfc822",
            metadata={
                "message_id": message_id,
                "subject": headers.get("Subject", ""),
                "from_email": from_email,
            },
        )

    def _fetch_rss(self, ref: SourceRef) -> RawSource:
        path = self.root / ref.ref["path"]
        items = json.loads(path.read_text(encoding="utf-8"))
        item = next(i for i in items if i["id"] == ref.source_id)
        text = f"{item['title']}\n\n{item.get('summary', '')}".strip() + "\n"
        # Per-item bytes, not the whole feed file -- see module docstring
        # ("original_bytes") for why.
        original_bytes = json.dumps(item, sort_keys=True).encode("utf-8")

        return RawSource(
            source_id=ref.source_id,
            source_type=SourceType.RSS,
            source_date=date.fromisoformat(item["date"]),
            participants=[],
            account_hints=self._account_hints(ref.source_id),
            text=text,
            original_bytes=original_bytes,
            original_content_type="application/json",
            metadata={"link": item.get("link", "")},
        )

    def _fetch_crm(self, ref: SourceRef) -> RawSource:
        row: dict[str, str] = ref.ref["row"]
        object_type: str = ref.ref["object"]
        crm_id: str = ref.ref["crm_id"]
        csv_path: str = ref.ref["path"]
        text = _render_crm_row_text(object_type, row)
        # Per-row bytes, not the whole CSV file -- see module docstring
        # ("original_bytes") for why.
        original_bytes = json.dumps(row, sort_keys=True).encode("utf-8")
        source_date = self._crm_row_date(ref.source_id, row, object_type)

        return RawSource(
            source_id=ref.source_id,
            source_type=SourceType.CRM,
            source_date=source_date,
            participants=[],
            account_hints=self._account_hints(ref.source_id),
            text=text,
            original_bytes=original_bytes,
            original_content_type="application/json",
            metadata={"object": object_type, "crm_id": crm_id, "path": csv_path},
        )


# For a type checker: FileConnector satisfies the Connector protocol.
_: type[Connector] = FileConnector
