"""RSS/Atom connector (build-contracts.md "Connectors": RSS;
portolan-spec-v2.md sec 5.3: "RSS via an RSS.app account").

A plain feed reader over the stdlib `xml.etree.ElementTree` — no vendor
SDK is needed because both formats are open standards:
- RSS 2.0: https://www.rssboard.org/rss-specification
- Atom 1.0 (RFC 4287): https://www.rfc-editor.org/rfc/rfc4287
RSS.app (named in `01-spec/decisions-log.md` as the way a customer's
LinkedIn page becomes a feed) produces standard RSS 2.0, so no
RSS.app-specific parsing is needed; this connector reads whichever of the
two formats a configured feed URL serves.

Configured with a list of `{name, url, kind}` feeds (`kind` is
`"linkedin"` or `"news"`, passed through as `RawSource.metadata
["feed_kind"]`, uninterpreted). `since` filters on the item's published
date; items are deduped by GUID (RSS `<guid>`, Atom `<id>`, else the link)
and by link across every feed in one `list_sources` call, and the source id
is derived from the GUID, so a re-crawl of the same item is a no-op at
intake. `portolan.signals.feeds` reads per-account feeds through the same
`parse_feed`.
"""

from __future__ import annotations

import hashlib
import re
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from email.utils import parsedate_to_datetime
from typing import Literal

import httpx
from defusedxml.ElementTree import fromstring as defused_fromstring

from portolan.connectors.base import (
    SourceRef,
    Transport,
    httpx_transport,
    parse_datetime,
    request_with_retry,
    source_day,
)
from portolan.extract.schema import SourceType
from portolan.ingest.models import RawSource

__all__ = ["FeedConfig", "RSSConfig", "RSSConnector"]

_ATOM_NS = "{http://www.w3.org/2005/Atom}"


@dataclass(frozen=True)
class FeedConfig:
    name: str
    url: str
    kind: Literal["linkedin", "news"] = "news"


@dataclass(frozen=True)
class RSSConfig:
    feeds: list[FeedConfig]


@dataclass(frozen=True)
class _FeedItem:
    title: str
    link: str
    published: datetime
    summary: str
    raw_xml: str
    # RSS `<guid>` / Atom `<id>`; the link when the feed gives neither.
    guid: str = ""
    # RSS `<source>` text: the publisher a Google News search item names.
    source_name: str = ""
    # False when the feed gave no date and `published` is the parse time.
    dated: bool = True


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _hash_link(link: str) -> str:
    return hashlib.sha256(link.encode("utf-8")).hexdigest()[:12]


def _parse_rss2(root: ET.Element) -> list[_FeedItem]:
    items: list[_FeedItem] = []
    channel = root.find("channel")
    if channel is None:
        return items
    for item_el in channel.findall("item"):
        title = (item_el.findtext("title") or "").strip()
        link = (item_el.findtext("link") or "").strip()
        pub_date_raw = item_el.findtext("pubDate")
        published = parsedate_to_datetime(pub_date_raw) if pub_date_raw else datetime.now(UTC)
        summary = (item_el.findtext("description") or "").strip()
        guid = (item_el.findtext("guid") or "").strip() or link
        items.append(
            _FeedItem(
                title=title,
                link=link,
                published=published,
                summary=summary,
                raw_xml=ET.tostring(item_el, encoding="unicode"),
                guid=guid,
                source_name=(item_el.findtext("source") or "").strip(),
                dated=bool(pub_date_raw),
            )
        )
    return items


def _parse_atom(root: ET.Element) -> list[_FeedItem]:
    items: list[_FeedItem] = []
    for entry in root.findall(f"{_ATOM_NS}entry"):
        title = (entry.findtext(f"{_ATOM_NS}title") or "").strip()
        link_el = entry.find(f"{_ATOM_NS}link")
        link = (link_el.get("href") or "").strip() if link_el is not None else ""
        updated_raw = entry.findtext(f"{_ATOM_NS}updated") or entry.findtext(f"{_ATOM_NS}published")
        published = parse_datetime(updated_raw) if updated_raw else datetime.now(UTC)
        summary = (
            entry.findtext(f"{_ATOM_NS}summary") or entry.findtext(f"{_ATOM_NS}content") or ""
        ).strip()
        guid = (entry.findtext(f"{_ATOM_NS}id") or "").strip() or link
        items.append(
            _FeedItem(
                title=title,
                link=link,
                published=published,
                summary=summary,
                raw_xml=ET.tostring(entry, encoding="unicode"),
                guid=guid,
                dated=bool(updated_raw),
            )
        )
    return items


def parse_feed(xml_text: str) -> list[_FeedItem]:
    """Parse either an RSS 2.0 or an Atom feed document into `_FeedItem`s.

    Parsed with `defusedxml`, not the stdlib parser: the stdlib's docs flag
    it as open to entity-expansion (billion laughs) and external-entity
    attacks, and this text comes from a remote feed.
    """
    root = defused_fromstring(xml_text)
    if root.tag == "rss" or root.find("channel") is not None:
        return _parse_rss2(root)
    return _parse_atom(root)


class RSSConnector:
    """`build-contracts.md`'s `Connector` protocol over a list of RSS/Atom
    feed URLs. Dedup is by GUID and by link, across every configured feed."""

    name = "rss"

    def __init__(
        self,
        config: RSSConfig,
        *,
        transport: Transport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._config = config
        self._transport = transport or httpx_transport(httpx.Client(timeout=30.0))
        self._sleep = sleep

    def list_sources(self, since: date | None) -> Iterable[SourceRef]:
        seen: set[str] = set()
        for feed in self._config.feeds:
            response = request_with_retry(
                self._transport, "GET", feed.url, sleep=self._sleep, vendor="rss"
            )
            for item in parse_feed(response.text):
                if not item.link or item.link in seen or item.guid in seen:
                    continue
                seen.update({item.link, item.guid})
                if since is not None and source_day(item.published) < since:
                    continue
                source_id = f"rss-{_slugify(feed.name)}-{_hash_link(item.guid or item.link)}"
                yield SourceRef(
                    source_id=source_id,
                    source_type=SourceType.RSS,
                    updated_at=item.published,
                    ref={"feed": feed, "item": item},
                )

    def fetch(self, ref: SourceRef) -> RawSource:
        feed: FeedConfig = ref.ref["feed"]
        item: _FeedItem = ref.ref["item"]
        text = f"{item.title}\n\n{item.summary}".strip() if item.summary else item.title
        return RawSource(
            source_id=ref.source_id,
            source_type=SourceType.RSS,
            source_date=source_day(item.published),
            participants=[],
            account_hints=[],
            text=text,
            original_bytes=item.raw_xml.encode("utf-8"),
            original_content_type="application/xml",
            metadata={
                "link": item.link,
                "guid": item.guid,
                "feed_name": feed.name,
                "feed_kind": feed.kind,
                "title": item.title,
            },
        )
