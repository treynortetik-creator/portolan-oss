"""What to watch per account: rows in `signal_targets`.

The operator keeps a YAML file and imports it (`python -m portolan.signals
targets import FILE`); the table is the truth afterwards. Shape:

    accounts:
      - crm_id: A-1001
        feeds:
          - url: https://rss.app/feeds/XXXXXXXX.xml     # RSS.app feed of the company page
            kind: linkedin
            linkedin_company_url: https://www.linkedin.com/company/example-health/
            label: Example Health on LinkedIn
          - url: https://news.example.org/feed/          # the account's own newsroom feed
            kind: news
        pages:
          - url: https://www.example.org/about/leadership
            label: Leadership
            fetch: http                                  # or `browser` for a JavaScript-only page

**LinkedIn is company pages only** (the maintainer, 2026-09-23). A `linkedin` feed
must name the company page it follows (`linkedin_company_url`, a
`linkedin.com/company/<name>` URL); import refuses anything else, and so does
the database (`signal_targets_linkedin_company_only`). Personal profiles are
never monitored, by any target kind. `feeds.py` also checks each feed's own
channel link on every scan and refuses a feed that points at a personal
profile (`PERSONAL_PROFILE_RE`).

Import is an upsert on `(account, kind, url)`: re-importing the same file
changes nothing, an edited label, company page or fetch mode is updated, and a
target left out of the file is deactivated (`active = false`, never deleted,
because its snapshots and items still cite it).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import psycopg
import yaml

__all__ = [
    "LINKEDIN_COMPANY_RE",
    "PERSONAL_PROFILE_RE",
    "Target",
    "TargetImport",
    "import_targets",
    "load_targets",
    "parse_targets_yaml",
]

# The same pattern as the database CHECK in migration 0034.
LINKEDIN_COMPANY_RE = re.compile(r"^https://(www\.)?linkedin\.com/company/[^/?#]+/?")
# A LinkedIn member profile (`/in/<name>`, `/pub/...`): never a target, never a feed source.
PERSONAL_PROFILE_RE = re.compile(r"^https?://([a-z]{2,3}\.|www\.)?linkedin\.com/(in|pub)/", re.I)


@dataclass(frozen=True)
class Target:
    id: int
    account_crm_id: str
    kind: str  # "feed" | "page"
    url: str
    label: str | None
    feed_kind: str | None
    fetch_mode: str
    active: bool
    source_page_url: str | None = None
    last_checked_at: datetime | None = None


@dataclass
class TargetImport:
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    deactivated: int = 0


_COLUMNS = (
    "id, account_crm_id, kind, url, label, feed_kind, fetch_mode, active, source_page_url, "
    "last_checked_at"
)


def load_targets(conn: psycopg.Connection, *, active_only: bool = True) -> list[Target]:
    where = "WHERE active" if active_only else ""
    with conn.cursor() as cur:
        cur.execute(f"SELECT {_COLUMNS} FROM signal_targets {where} ORDER BY account_crm_id, id")
        return [Target(*r) for r in cur.fetchall()]


def _check_url(url: str) -> str:
    url = (url or "").strip()
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValueError(f"target url must be an http(s) URL with a host, got {url!r}")
    if PERSONAL_PROFILE_RE.match(url):
        raise ValueError(f"personal LinkedIn profiles are never monitored: {url!r}")
    return url


def _linkedin_company_url(feed: dict[str, Any]) -> str:
    url = str(feed.get("linkedin_company_url") or "").strip()
    if not LINKEDIN_COMPANY_RE.match(url):
        raise ValueError(
            "a linkedin feed must follow a LinkedIn company page: set linkedin_company_url "
            f"to https://www.linkedin.com/company/<name>/ (got {url!r}); personal "
            "profiles are never monitored"
        )
    return url


def parse_targets_yaml(data: dict[str, Any]) -> list[dict[str, Any]]:
    """The YAML's accounts as flat target dicts (`account_crm_id`, `kind`,
    `url`, `label`, `feed_kind`, `source_page_url`, `fetch_mode`), validated."""
    rows: list[dict[str, Any]] = []
    for account in data.get("accounts") or []:
        crm_id = str(account.get("crm_id") or "").strip()
        if not crm_id:
            raise ValueError(f"every account needs a crm_id: {account!r}")
        for feed in account.get("feeds") or []:
            feed_kind = feed.get("kind", "other")
            if feed_kind not in ("linkedin", "news", "other"):
                raise ValueError(f"feed kind must be linkedin, news or other: {feed!r}")
            source_page_url = _linkedin_company_url(feed) if feed_kind == "linkedin" else None
            rows.append(
                {
                    "account_crm_id": crm_id,
                    "kind": "feed",
                    "url": _check_url(feed.get("url")),
                    "label": feed.get("label"),
                    "feed_kind": feed_kind,
                    "source_page_url": source_page_url,
                    "fetch_mode": "http",
                }
            )
        for page in account.get("pages") or []:
            if isinstance(page, str):
                page = {"url": page}
            fetch_mode = page.get("fetch", "http")
            if fetch_mode not in ("http", "browser"):
                raise ValueError(f"page fetch must be http or browser: {page!r}")
            url = _check_url(page.get("url"))
            if (urlsplit(url).hostname or "").endswith("linkedin.com"):
                raise ValueError(
                    f"LinkedIn is read only through an RSS.app company-page feed, never "
                    f"fetched as a page: {url!r}"
                )
            rows.append(
                {
                    "account_crm_id": crm_id,
                    "kind": "page",
                    "url": url,
                    "label": page.get("label"),
                    "feed_kind": None,
                    "source_page_url": None,
                    "fetch_mode": fetch_mode,
                }
            )
    return rows


def import_targets(conn: psycopg.Connection, source: Path | dict[str, Any]) -> TargetImport:
    """Upsert the targets in `source` (a YAML path or its parsed dict) and
    deactivate every active target not in it (module docstring)."""
    data = yaml.safe_load(Path(source).read_text()) if isinstance(source, Path) else source
    rows = parse_targets_yaml(data or {})
    result = TargetImport()
    keep: set[tuple[str, str, str]] = set()
    with conn.cursor() as cur:
        for row in rows:
            key = (row["account_crm_id"], row["kind"], row["url"])
            keep.add(key)
            cur.execute(
                "SELECT label, feed_kind, source_page_url, fetch_mode, active "
                "FROM signal_targets WHERE account_crm_id = %s AND kind = %s AND url = %s",
                key,
            )
            existing = cur.fetchone()
            fields = (row["label"], row["feed_kind"], row["source_page_url"], row["fetch_mode"])
            if existing is None:
                cur.execute(
                    "INSERT INTO signal_targets (account_crm_id, kind, url, label, feed_kind, "
                    "source_page_url, fetch_mode) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    (*key, *fields),
                )
                result.added += 1
            elif tuple(existing) == (*fields, True):
                result.unchanged += 1
            else:
                cur.execute(
                    "UPDATE signal_targets SET label = %s, feed_kind = %s, source_page_url = %s, "
                    "fetch_mode = %s, active = true "
                    "WHERE account_crm_id = %s AND kind = %s AND url = %s",
                    (*fields, *key),
                )
                result.updated += 1
        cur.execute("SELECT account_crm_id, kind, url FROM signal_targets WHERE active")
        for key in cur.fetchall():
            if tuple(key) not in keep:
                cur.execute(
                    "UPDATE signal_targets SET active = false "
                    "WHERE account_crm_id = %s AND kind = %s AND url = %s",
                    tuple(key),
                )
                result.deactivated += 1
    conn.commit()
    return result
