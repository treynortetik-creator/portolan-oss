"""Ticket connectors (build-contracts.md "Connectors": Tickets;
portolan-spec-v2.md sec 5.3: "Support tickets (export or API)").

Two connectors:

- `ZendeskConnector`: Zendesk's Ticketing API. Auth is HTTP Basic with
  `<email>/token` as the username and an API token as the password
  (Zendesk's documented token-auth convention:
  https://developer.zendesk.com/api-reference/ticketing/tickets/ticket-comments/,
  fetched 2026-09-12). `GET /api/v2/tickets.json` lists tickets,
  cursor-paginated via `page[size]` / `page[after]` and
  `meta.has_more` / `meta.after_cursor`
  (https://developer.zendesk.com/api-reference/ticketing/tickets/tickets/,
  which could not be fetched directly; the ticket-comments reference page
  documents the same cursor convention).
  `GET /api/v2/tickets/{id}/comments.json` lists a ticket's comments,
  same cursor style.
- `ExportTicketConnector`: a generic CSV/JSON reader for any other
  ticketing system's export, for "upload and export as the fallback"
  (portolan-spec-v2.md sec 5.3). Not vendor-specific, so no API doc
  applies; reads a local file, no network.

Not independently verified against a live Zendesk instance: the exact
field name for a comment's plain-text body varies by API version
(`body` vs `plain_body`); this connector prefers `plain_body` and falls
back to `body`.
"""

from __future__ import annotations

import base64
import csv
import json
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

import httpx

from portolan.connectors.base import (
    SourceRef,
    Transport,
    httpx_transport,
    parse_datetime,
    request_with_retry,
    require_allowed_origin,
    source_day,
)
from portolan.extract.schema import SourceType
from portolan.ingest.models import RawSource
from portolan.util import clock

__all__ = [
    "ExportTicketConfig",
    "ExportTicketConnector",
    "ZendeskConfig",
    "ZendeskConnector",
]


# --------------------------------------------------------------------
# Zendesk
# --------------------------------------------------------------------


@dataclass(frozen=True)
class ZendeskConfig:
    """A Zendesk subdomain, requester email and API token, never from
    code."""

    subdomain: str
    email: str
    api_token: str
    page_size: int = 100


def _comment_body(comment: dict) -> str:
    return comment.get("plain_body") or comment.get("body") or ""


class ZendeskConnector:
    """`build-contracts.md`'s `Connector` protocol over the Zendesk
    Ticketing API. `fetch` preserves comment numbering (1-based, in
    the order Zendesk returns them) in `RawSource.metadata["comments"]`
    so intake can inject `<source_id>#<n>` locators (`prepare.py`'s
    `_prepare_ticket`, verified against `corpus/out/manifest.json`).

    `since=None` (a full backfill) pages the plain ticket list, cursor-
    paginated. `since=<date>` (an incremental re-crawl) uses Zendesk's
    Incremental Export API, built for "everything changed since this
    timestamp": `GET /api/v2/incremental/tickets.json?start_time=<unix ts>`,
    followed via `after_url` until `end_of_stream`
    (https://developer.zendesk.com/api-reference/ticketing/ticket-management/incremental_exports/).
    """

    name = "zendesk"

    def __init__(
        self,
        config: ZendeskConfig,
        *,
        transport: Transport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._config = config
        self._transport = transport or httpx_transport(httpx.Client(timeout=30.0))
        self._sleep = sleep
        self._base_url = f"https://{config.subdomain}.zendesk.com"

    def _headers(self) -> dict[str, str]:
        token = base64.b64encode(
            f"{self._config.email}/token:{self._config.api_token}".encode()
        ).decode()
        return {"Authorization": f"Basic {token}"}

    def list_sources(self, since: date | None) -> Iterable[SourceRef]:
        if since is None:
            yield from self._list_all_tickets()
        else:
            yield from self._incremental_export(since)

    def _list_all_tickets(self) -> Iterable[SourceRef]:
        params: dict[str, object] = {"page[size]": self._config.page_size}
        while True:
            response = request_with_retry(
                self._transport,
                "GET",
                f"{self._base_url}/api/v2/tickets.json",
                params=params,
                headers=self._headers(),
                sleep=self._sleep,
                vendor="zendesk",
            )
            payload = response.json()
            for ticket in payload.get("tickets", []):
                yield SourceRef(
                    source_id=f"zendesk-{ticket['id']}",
                    source_type=SourceType.TICKET,
                    updated_at=parse_datetime(ticket["updated_at"]),
                    ref={"ticket": ticket},
                )
            meta = payload.get("meta") or {}
            if not meta.get("has_more"):
                break
            params = dict(params, **{"page[after]": meta["after_cursor"]})

    def _incremental_export(self, since: date) -> Iterable[SourceRef]:
        # UTC midnight, as HubSpot's `since`: a naive datetime would use the host's zone.
        start_time = int(datetime.combine(since, datetime.min.time(), tzinfo=UTC).timestamp())
        url = f"{self._base_url}/api/v2/incremental/tickets.json"
        params: dict[str, object] = {"start_time": start_time}
        while True:
            response = request_with_retry(
                self._transport,
                "GET",
                url,
                params=params,
                headers=self._headers(),
                sleep=self._sleep,
                vendor="zendesk",
            )
            payload = response.json()
            for ticket in payload.get("tickets", []):
                # The incremental export can return a ticket already
                # deleted since the export started (Zendesk's own
                # documented behaviour): it carries no useful comment
                # history to fetch, so it's skipped rather than yielded
                # as a source with nothing behind it.
                if ticket.get("status") == "deleted":
                    continue
                yield SourceRef(
                    source_id=f"zendesk-{ticket['id']}",
                    source_type=SourceType.TICKET,
                    updated_at=parse_datetime(ticket["updated_at"]),
                    ref={"ticket": ticket},
                )
            if payload.get("end_of_stream"):
                break
            after_url = payload.get("after_url")
            if not after_url:
                break
            # Our Basic-auth token goes with the next request: this subdomain only.
            require_allowed_origin(
                after_url, allowed_hosts=[urlsplit(self._base_url).hostname or ""], vendor="zendesk"
            )
            url = after_url
            params = None

    def _list_comments(self, ticket_id: object) -> list[dict]:
        comments: list[dict] = []
        params: dict[str, object] = {"page[size]": self._config.page_size}
        while True:
            response = request_with_retry(
                self._transport,
                "GET",
                f"{self._base_url}/api/v2/tickets/{ticket_id}/comments.json",
                params=params,
                headers=self._headers(),
                sleep=self._sleep,
                vendor="zendesk",
            )
            payload = response.json()
            for comment in payload.get("comments", []):
                comments.append(
                    {
                        "number": len(comments) + 1,
                        "author": comment.get("author_id"),
                        "created": comment.get("created_at"),
                        "body": _comment_body(comment),
                    }
                )
            meta = payload.get("meta") or {}
            if not meta.get("has_more"):
                break
            params = dict(params, **{"page[after]": meta["after_cursor"]})
        return comments

    def fetch(self, ref: SourceRef) -> RawSource:
        ticket = ref.ref["ticket"]
        ticket_id = ticket["id"]
        comments = self._list_comments(ticket_id)
        text = "\n\n".join(f"[{c['number']}] {c['body']}" for c in comments)
        return RawSource(
            source_id=f"zendesk-{ticket_id}",
            source_type=SourceType.TICKET,
            source_date=source_day(parse_datetime(ticket["created_at"])),
            participants=[],
            account_hints=[],
            text=text,
            original_bytes=json.dumps(
                {"ticket": ticket, "comments": comments}, sort_keys=True
            ).encode("utf-8"),
            original_content_type="application/json",
            metadata={"comments": comments, "subject": ticket.get("subject")},
        )


# --------------------------------------------------------------------
# Generic CSV/JSON export
# --------------------------------------------------------------------


@dataclass(frozen=True)
class ExportTicketConfig:
    """A local ticket export file, for any system without an API
    connector — "upload and export as the fallback" (spec sec 5.3).
    JSON shape: a list of `{id, subject, created, comments: [{number,
    author, created, body}]}` objects. CSV shape: one row per comment,
    columns `ticket_id, subject, created, comment_number, author,
    comment_created, body`, grouped here by `ticket_id`."""

    path: Path
    format: Literal["csv", "json"] = "json"


def _load_json_tickets(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv_tickets(path: Path) -> list[dict]:
    grouped: dict[str, dict] = {}
    order: list[str] = []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            ticket_id = row["ticket_id"]
            if ticket_id not in grouped:
                grouped[ticket_id] = {
                    "id": ticket_id,
                    "subject": row.get("subject", ""),
                    "created": row.get("created", ""),
                    "comments": [],
                }
                order.append(ticket_id)
            grouped[ticket_id]["comments"].append(
                {
                    "number": int(row["comment_number"]),
                    "author": row.get("author"),
                    "created": row.get("comment_created"),
                    "body": row.get("body", ""),
                }
            )
    return [grouped[tid] for tid in order]


class ExportTicketConnector:
    """`build-contracts.md`'s `Connector` protocol over a local CSV/JSON
    ticket export. No network; `list_sources` reads the whole file once
    and caches it. A re-sent identical export is a no-op at `intake()`
    (same source id, same bytes)."""

    name = "ticket_export"

    def __init__(self, config: ExportTicketConfig) -> None:
        self._config = config
        self._tickets: list[dict] | None = None

    def _load(self) -> list[dict]:
        if self._tickets is None:
            loader = _load_json_tickets if self._config.format == "json" else _load_csv_tickets
            self._tickets = loader(self._config.path)
        return self._tickets

    def list_sources(self, since: date | None) -> Iterable[SourceRef]:
        for ticket in self._load():
            updated_at = _export_ticket_updated_at(ticket)
            if since is not None and source_day(updated_at) < since:
                continue
            yield SourceRef(
                source_id=f"export-{ticket['id']}",
                source_type=SourceType.TICKET,
                updated_at=updated_at,
                ref={"ticket": ticket},
            )

    def fetch(self, ref: SourceRef) -> RawSource:
        ticket = ref.ref["ticket"]
        comments = sorted(ticket.get("comments", []), key=lambda c: c["number"])
        text = "\n\n".join(f"[{c['number']}] {c['body']}" for c in comments)
        source_date = source_day(_parse_export_date(ticket.get("created") or ""))
        return RawSource(
            source_id=f"export-{ticket['id']}",
            source_type=SourceType.TICKET,
            source_date=source_date,
            participants=[],
            account_hints=[],
            text=text,
            original_bytes=json.dumps(ticket, sort_keys=True).encode("utf-8"),
            original_content_type="application/json",
            metadata={"comments": comments, "subject": ticket.get("subject", "")},
        )


def _export_ticket_updated_at(ticket: dict) -> datetime:
    """The ticket's last change: its newest comment, else its creation. A
    re-crawl with `since` must still list an old ticket that gained a
    comment (`tests/test_bughunt_wave2.py`)."""
    stamps = [ticket.get("created") or ""]
    stamps += [c.get("created") or "" for c in ticket.get("comments", [])]
    parsed = [_parse_export_date(s) for s in stamps if s]
    if not parsed:
        return _parse_export_date("")
    return max(parsed, key=lambda d: d if d.tzinfo else d.replace(tzinfo=UTC))


def _parse_export_date(value: str) -> datetime:
    if not value:
        return clock.now()
    try:
        return parse_datetime(value)
    except ValueError:
        return datetime.fromisoformat(f"{value}T00:00:00+00:00")
