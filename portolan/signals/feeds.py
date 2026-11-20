"""Feed targets: read one RSS/Atom feed and queue its new items for the judge.

Any feed URL works (`portolan.connectors.rss.parse_feed` reads RSS 2.0 and
Atom). In practice: an RSS.app feed of the account's LinkedIn company page or
of a news keyword search (RSS.app fetches the third-party site, Portolan only
fetches rss.app), or the account's own newsroom feed. LinkedIn is company pages
only: `check_feed_source` refuses, on every scan, a feed whose channel link is a
personal profile (`targets.py`).

Idempotent by GUID: `signal_items` is unique on `(account, 'feed_item', guid)`,
so a re-scan inserts nothing it has seen, and the same item in two of one
account's feeds is queued once. Items published before the window
(`feed_days`) are skipped rather than stored: the first scan of a new feed does
not backfill a year of posts into the weekly diff.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from urllib.parse import urlsplit

import psycopg
from defusedxml.ElementTree import fromstring as defused_fromstring
from psycopg.types.json import Jsonb

from portolan.connectors.rss import parse_feed
from portolan.signals.fetch import PoliteFetcher
from portolan.signals.html_text import html_to_text
from portolan.signals.targets import LINKEDIN_COMPANY_RE, PERSONAL_PROFILE_RE, Target

__all__ = [
    "DEFAULT_FEED_DAYS",
    "MAX_ITEMS_PER_FEED",
    "FeedOutcome",
    "PersonalProfileRefused",
    "check_feed_source",
    "collect_feed_items",
    "feed_channel_link",
]

DEFAULT_FEED_DAYS = 14
MAX_ITEMS_PER_FEED = 50
# What the judge reads of one item; a LinkedIn post or press release rarely needs more.
MAX_ITEM_BODY_CHARS = 4000


@dataclass
class FeedOutcome:
    target_id: int
    seen: int = 0
    new: int = 0
    too_old: int = 0
    error: str | None = None


class PersonalProfileRefused(Exception):
    """The feed follows a LinkedIn member profile, which is never monitored."""


_ATOM_NS = "{http://www.w3.org/2005/Atom}"


def feed_channel_link(xml_text: str) -> str:
    """The page a feed says it mirrors: RSS `channel/link`, else Atom's
    alternate `link`. RSS.app sets it to the followed LinkedIn page."""
    root = defused_fromstring(xml_text)
    channel = root.find("channel")
    if channel is not None:
        return (channel.findtext("link") or "").strip()
    for link in root.findall(f"{_ATOM_NS}link"):
        if link.get("rel", "alternate") == "alternate":
            return (link.get("href") or "").strip()
    return ""


def check_feed_source(target: Target, xml_text: str) -> None:
    """Refuse a feed that mirrors a personal LinkedIn profile, and a `linkedin`
    feed whose channel link is on LinkedIn but not a company page
    (`targets.py`, "LinkedIn is company pages only")."""
    link = feed_channel_link(xml_text)
    if PERSONAL_PROFILE_RE.match(link):
        raise PersonalProfileRefused(f"feed mirrors a personal LinkedIn profile ({link})")
    host = (urlsplit(link).hostname or "").lower()
    on_linkedin = host == "linkedin.com" or host.endswith(".linkedin.com")
    if target.feed_kind == "linkedin" and on_linkedin and not LINKEDIN_COMPANY_RE.match(link):
        raise PersonalProfileRefused(f"linkedin feed does not mirror a company page ({link})")


def _publisher(target: Target, source_name: str, link: str) -> str:
    if source_name:
        return source_name
    if target.label:
        return target.label
    if target.feed_kind == "linkedin":
        return "LinkedIn company page"
    return urlsplit(link).hostname or urlsplit(target.url).hostname or "feed"


def item_body(title: str, summary_html: str) -> str:
    """Title, then the description as plain text, capped: what the judge reads
    and the stored original shows."""
    summary = html_to_text(summary_html)
    if summary.strip() == title.strip():
        summary = ""
    body = f"{title}\n\n{summary}".strip() if summary else title.strip()
    return body[:MAX_ITEM_BODY_CHARS]


def collect_feed_items(
    conn: psycopg.Connection,
    target: Target,
    fetcher: PoliteFetcher,
    *,
    now: datetime,
    feed_days: int = DEFAULT_FEED_DAYS,
    max_items: int = MAX_ITEMS_PER_FEED,
) -> FeedOutcome:
    """Fetch `target`'s feed and insert each new, in-window item as a pending
    `signal_items` row. Fetch errors (robots, HTTP, bad XML) land in
    `outcome.error` and on the target's `last_error`; nothing raises."""
    outcome = FeedOutcome(target_id=target.id)
    try:
        response = fetcher.get(target.url)
        check_feed_source(target, response.text)
        items = parse_feed(response.text)
    except Exception as exc:  # recorded on the target, never fatal to the run
        outcome.error = f"{type(exc).__name__}: {exc}"[:500]
        return outcome

    floor = now - timedelta(days=feed_days)
    with conn.cursor() as cur:
        for item in items[:max_items]:
            outcome.seen += 1
            published = item.published
            if published.tzinfo is None:
                published = published.replace(tzinfo=UTC)
            if item.dated and published < floor:
                outcome.too_old += 1
                continue
            guid = item.guid or item.link
            if not guid or not item.link:
                continue
            body = item_body(item.title, item.summary)
            cur.execute(
                """
                INSERT INTO signal_items
                    (target_id, account_crm_id, item_kind, item_key, url, title, publisher,
                     published_at, observed_at, body, details)
                VALUES (%s, %s, 'feed_item', %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (account_crm_id, item_kind, item_key) DO NOTHING
                RETURNING id
                """,
                (
                    target.id,
                    target.account_crm_id,
                    guid,
                    item.link,
                    item.title[:500],
                    _publisher(target, item.source_name, item.link)[:200],
                    published if item.dated else None,
                    now,
                    body,
                    Jsonb({"feed_url": target.url, "feed_kind": target.feed_kind, "guid": guid}),
                ),
            )
            if cur.fetchone() is not None:
                outcome.new += 1
    conn.commit()
    return outcome
