"""Page targets: snapshot one public page and queue what changed since the last
snapshot for the judge.

Each scan fetches the page (plain HTTP through `PoliteFetcher`; a browser only
for a target marked `fetch: browser`, and only if Playwright is installed),
extracts sectioned text (`html_text.extract_page`), and stores a
`page_snapshots` row plus the raw HTML in the originals store under
`signals/snapshots/<target id>/<UTC time>.html`. The HTML is stored only when
the normalised content changed, so a quiet page costs one small row a week.

The first snapshot of a page is its baseline: nothing to compare, no item.
After that, `diff.diff_snapshots` compares against the latest earlier snapshot
and a meaningful change becomes one pending `page_change` item, keyed
`<previous snapshot id>-><snapshot id>` so it is queued once.

A page that looks JavaScript-built (`needs_browser`) is snapshotted but never
diffed on plain HTTP, since its empty shell would read as "everything removed".
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

import psycopg
from psycopg.types.json import Jsonb

from portolan.signals.diff import content_hash, diff_snapshots
from portolan.signals.fetch import PoliteFetcher, RobotsDisallowed
from portolan.signals.html_text import Section, extract_page
from portolan.signals.targets import Target
from portolan.store import Store

__all__ = [
    "BrowserFetch",
    "BrowserUnavailable",
    "PageOutcome",
    "latest_snapshot",
    "playwright_fetch",
    "snapshot_page",
]

BrowserFetch = Callable[[str, str], str]


class BrowserUnavailable(Exception):
    """Playwright is not installed in this environment."""


def playwright_fetch(url: str, user_agent: str) -> str:
    """The rendered HTML of `url` in headless Chromium. Optional extra:
    `uv pip install playwright && playwright install --only-shell chromium`
    (01-spec/external-signals-2026-09-23.md, "Browser pages")."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise BrowserUnavailable(
            "fetch: browser needs Playwright (uv pip install playwright; "
            "playwright install --only-shell chromium)"
        ) from exc
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            page = browser.new_page(user_agent=user_agent)
            page.goto(url, wait_until="networkidle", timeout=45_000)
            return page.content()
        finally:
            browser.close()


@dataclass
class PageOutcome:
    target_id: int
    snapshot_id: int | None = None
    baseline: bool = False
    changed: bool = False
    item_id: int | None = None
    needs_browser: bool = False
    error: str | None = None
    robots_disallowed: bool = False


@dataclass
class _Snapshot:
    id: int
    fetched_at: datetime
    content_hash: str
    html_key: str | None
    title: str
    sections: list[Section]
    needs_browser: bool


def latest_snapshot(
    conn: psycopg.Connection, target_id: int, *, before_id: int | None = None
) -> _Snapshot | None:
    """The newest snapshot of `target_id` (older than `before_id` when given)."""
    extra = "AND id < %s" if before_id is not None else ""
    params: tuple = (target_id, before_id) if before_id is not None else (target_id,)
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT id, fetched_at, content_hash, html_key, title, sections, needs_browser
            FROM page_snapshots WHERE target_id = %s {extra}
            ORDER BY fetched_at DESC, id DESC LIMIT 1
            """,
            params,
        )
        row = cur.fetchone()
    if row is None:
        return None
    return _Snapshot(
        id=row[0],
        fetched_at=row[1],
        content_hash=row[2],
        html_key=row[3],
        title=row[4] or "",
        sections=[Section.from_json(s) for s in (row[5] or [])],
        needs_browser=row[6],
    )


def utc_stamp(moment: datetime) -> str:
    """`20260923T123500Z`: `moment` in UTC (callers pass deployment-zone times)."""
    return moment.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")


def _human(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime("%Y-%m-%d %H:%M UTC")


def snapshot_page(
    conn: psycopg.Connection,
    store: Store,
    target: Target,
    fetcher: PoliteFetcher,
    *,
    now: datetime,
    browser_fetch: BrowserFetch | None = None,
) -> PageOutcome:
    """Snapshot `target` and queue a `page_change` item when it changed
    (module docstring). Errors land in the outcome; nothing raises."""
    outcome = PageOutcome(target_id=target.id)
    method = "http"
    try:
        if target.fetch_mode == "browser":
            if not fetcher.allowed(target.url):
                raise RobotsDisallowed(f"robots.txt disallows {target.url}")
            fetch = browser_fetch or playwright_fetch
            html = fetch(target.url, fetcher.user_agent)
            status = 200
            method = "browser"
        else:
            result = fetcher.get(target.url)
            html, status = result.text, result.status
    except RobotsDisallowed as exc:
        outcome.robots_disallowed = True
        outcome.error = str(exc)[:500]
        return outcome
    except Exception as exc:
        outcome.error = f"{type(exc).__name__}: {exc}"[:500]
        return outcome

    page = extract_page(html)
    needs_browser = page.needs_browser and method == "http"
    outcome.needs_browser = needs_browser
    digest = content_hash(page.sections)
    previous = latest_snapshot(conn, target.id)

    html_key = previous.html_key if previous and previous.content_hash == digest else None
    if html_key is None:
        html_key = f"signals/snapshots/{target.id}/{utc_stamp(now)}.html"
        store.put(html_key, html.encode("utf-8"), "text/html; charset=utf-8")

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO page_snapshots
                (target_id, url, fetched_at, http_status, fetch_method, content_hash,
                 html_key, title, sections, needs_browser)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                target.id,
                target.url,
                now,
                status,
                method,
                digest,
                html_key,
                page.title,
                Jsonb([s.to_json() for s in page.sections]),
                needs_browser,
            ),
        )
        (snapshot_id,) = cur.fetchone()
    conn.commit()
    outcome.snapshot_id = snapshot_id

    if previous is None:
        outcome.baseline = True
        return outcome
    if needs_browser or previous.needs_browser:
        return outcome

    change = diff_snapshots(previous.sections, page.sections, url=target.url, title=page.title)
    if change is None:
        return outcome
    outcome.changed = True
    body = change.render(observed=_human(now), previous=_human(previous.fetched_at))
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO signal_items
                (target_id, account_crm_id, item_kind, item_key, url, title, publisher,
                 observed_at, body, details, snapshot_id, previous_snapshot_id)
            VALUES (%s, %s, 'page_change', %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (account_crm_id, item_kind, item_key) DO NOTHING
            RETURNING id
            """,
            (
                target.id,
                target.account_crm_id,
                f"{previous.id}->{snapshot_id}",
                target.url,
                (page.title or target.label or target.url)[:500],
                (target.label or "website")[:200],
                now,
                body,
                Jsonb(
                    {
                        "changes": [c.to_json() for c in change.changes],
                        "snapshot": {
                            "id": snapshot_id,
                            "fetched_at": now.isoformat(),
                            "html_key": html_key,
                        },
                        "previous_snapshot": {
                            "id": previous.id,
                            "fetched_at": previous.fetched_at.isoformat(),
                            "html_key": previous.html_key,
                        },
                    }
                ),
                snapshot_id,
                previous.id,
            ),
        )
        row = cur.fetchone()
    conn.commit()
    outcome.item_id = row[0] if row else None
    return outcome
