"""Turn the app's HTML pages into plain dicts for the CLI. Each parser reads
the markers the template carries (`portolan/web/templates/`), the same ones
`scripts/verify.py` counts, so a page that renders but is empty parses to an
empty list rather than an error."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import parse_qsl, unquote, urlsplit

from portolan.client.chat import claim_quote
from portolan.client.htmltree import Node, Page

_OPEN_SUMMARY_RE = re.compile(r"(\d+) open, (\d+) overdue(?:, as of (\S+?)\.?$)?")
_REVIEW_EYEBROW_RE = re.compile(r"Review\s*·\s*(\d+) open")
_RESULT_COUNT_RE = re.compile(r"^(\d+) results?$")
_NOTE_COUNT_RE = re.compile(r"^(\d+) notes?$")
_PAGE_OF_RE = re.compile(r"page (\d+) of (\d+)")
_OWNER_COUNT_SUFFIX_RE = re.compile(r"\s*\(\d+\)\s*$")
_ACTION_ID_RE = re.compile(r"^/actions/(\d+)/")
_REVIEW_ID_RE = re.compile(r"^/review/(\d+)/")


def _text(node: Node | None) -> str:
    return node.text() if node is not None else ""


def _slug_from(href: str, prefix: str) -> str:
    path = urlsplit(href).path
    return unquote(path.removeprefix(prefix)) if path.startswith(prefix) else ""


def _card(dom: Node, eyebrow: str) -> Node | None:
    for card in dom.find_all("div", "card"):
        if _text(card.find("p", "eyebrow")) == eyebrow:
            return card
    return None


def snake(header: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", header.lower()).strip("_")


def _cell_value(text: str) -> Any:
    if text in ("", "—"):
        return None
    if re.fullmatch(r"-?\d+", text):
        return int(text)
    if re.fullmatch(r"-?\d+\.\d+", text):
        return float(text)
    return text


def table_records(table: Node | None) -> list[dict[str, Any]]:
    """Rows of a table keyed by its snake_cased header cells."""
    if table is None:
        return []
    rows = table.find_all("tr")
    if not rows:
        return []
    headers = [snake(c.text()) or f"col{i}" for i, c in enumerate(rows[0].find_all("th"))]
    records = []
    for row in rows[1:]:
        cells = row.find_all("td")
        if cells and headers:
            records.append({h: _cell_value(c.text()) for h, c in zip(headers, cells, strict=False)})
    return records


# ---- notes -------------------------------------------------------------


def parse_note(page: Page) -> dict[str, Any]:
    dom = page.dom
    meta = dom.find("div", "meta-row")
    badges = meta.find_all("span", "badge") if meta is not None else []
    note_type = ""
    for b in badges:
        for cls in b.attrs.get("class", "").split():
            if cls.startswith("type-"):
                note_type = cls.removeprefix("type-")
    status = next((b.text() for b in badges if not b.attrs.get("class", "").count("type-")), "")
    updated = created = ""
    if meta is not None:
        for span in meta.find_all("span", "updated"):
            text = span.text()
            if text.startswith("updated:"):
                updated = text.removeprefix("updated:").strip()
            elif text.startswith("created:"):
                created = text.removeprefix("created:").strip()
    owner = ""
    for p in dom.find_all("p", "muted"):
        if p.text().startswith("Owner:"):
            owner = p.text().removeprefix("Owner:").strip()
    body = dom.find("div", "note-body")
    links = []
    seen: set[str] = set()
    related = _card(dom, "Related")
    for scope in (body, related):
        for a in scope.find_all("a") if scope is not None else []:
            slug = _slug_from(a.attrs.get("href", ""), "/notes/")
            if slug and slug not in seen:
                seen.add(slug)
                links.append({"slug": slug, "title": a.text()})
    claims = []
    for div in dom.find_all("div", "claim"):
        cite = div.find("p", "claim-cite")
        source = cite.find("a") if cite is not None else None
        dig = div.find("a", "dig")
        original = dig.attrs.get("href") if dig is not None else None
        locator = dict(parse_qsl(urlsplit(original).query)).get("at", "") if original else ""
        claim_id = div.attrs.get("id", "").removeprefix("claim-")
        claims.append(
            {
                "id": int(claim_id) if claim_id.isdigit() else None,
                "text": _text(div.find("p", "claim-text")),
                "cite": _text(cite),
                "source": source.text() if source is not None else "",
                "locator": locator,
                "quote": claim_quote(div),
                "original": original,
            }
        )
    backlinks = []
    back_card = _card(dom, "Backlinks")
    for li in back_card.find_all("li") if back_card is not None else []:
        a = li.find("a")
        badge = li.find("span", "badge")
        if a is not None:
            backlinks.append(
                {
                    "slug": _slug_from(a.attrs.get("href", ""), "/notes/"),
                    "title": a.text(),
                    "kind": _text(badge),
                }
            )
    return {
        "title": _text(dom.find("h1")),
        "type": note_type,
        "status": status,
        "updated": updated,
        "created": created,
        "owner": owner or None,
        "body": _text(body),
        "claims": claims,
        "links": links,
        "backlinks": backlinks,
    }


# ---- search ------------------------------------------------------------


def parse_search(page: Page) -> dict[str, Any]:
    dom = page.dom
    count = 0
    for p in dom.find_all("p", "mono"):
        match = _RESULT_COUNT_RE.match(p.text())
        if match:
            count = int(match.group(1))
    hits = []
    for card in dom.find_all("div", "note-card"):
        title = card.find("p", "note-card-title")
        a = title.find("a") if title is not None else None
        updated = card.find("span", "updated")
        hits.append(
            {
                "slug": _slug_from(a.attrs.get("href", ""), "/atlas/") if a is not None else "",
                "title": _text(a),
                "updated": _text(updated).removeprefix("updated:").strip(),
                "snippet": _text(card.find("p", "note-card-body")),
            }
        )
    return {
        "count": count,
        "hits": hits,
        "index_broken": "Index looks broken" in page.text,
    }


# ---- the notes index (accounts) -------------------------------------------


def parse_notes_index(page: Page) -> dict[str, Any]:
    dom = page.dom
    total = 0
    for p in dom.find_all("p", "mono"):
        match = _NOTE_COUNT_RE.match(p.text())
        if match:
            total = int(match.group(1))
    pages = 1
    for span in dom.find_all("span", "mono"):
        match = _PAGE_OF_RE.search(span.text())
        if match:
            pages = int(match.group(2))
    rows = []
    for tr in dom.find_all("tr"):
        title = tr.find("td", attrs={"data-label": "Title"})
        a = title.find("a") if title is not None else None
        if a is None:
            continue
        rows.append(
            {
                "slug": _slug_from(a.attrs.get("href", ""), "/notes/"),
                "title": a.text(),
                "type": _text(tr.find("td", attrs={"data-label": "Type"})),
                "status": _text(tr.find("td", attrs={"data-label": "Status"})),
                "updated": _text(tr.find("td", attrs={"data-label": "Updated"})),
            }
        )
    return {"total": total, "pages": pages, "rows": rows}


# ---- actions -------------------------------------------------------------


def _action_row(tr: Node, owner: str | None) -> dict[str, Any] | None:
    what = tr.find("td", attrs={"data-label": "What"})
    if what is None:
        return None
    a = what.find("a")
    account = tr.find("td", attrs={"data-label": "Account"})
    account_a = account.find("a") if account is not None else None
    source = tr.find("td", attrs={"data-label": "Source"})
    source_a = source.find("a") if source is not None else None
    action_id = None
    for form in tr.find_all("form"):
        match = _ACTION_ID_RE.match(form.attrs.get("action", ""))
        if match:
            action_id = int(match.group(1))
    due = _text(tr.find("td", attrs={"data-label": "Due"}))
    return {
        "id": action_id,
        "owner": owner,
        "what": _text(a),
        "note": _slug_from(a.attrs.get("href", ""), "/notes/") if a is not None else "",
        "account": _text(account_a) or None,
        "account_slug": _slug_from(account_a.attrs.get("href", ""), "/notes/")
        if account_a is not None
        else None,
        "due": due.replace("overdue", "").strip() if due not in ("", "—") else None,
        "overdue": tr.has_class("flag"),
        "source": _slug_from(source_a.attrs.get("href", ""), "/originals/")
        if source_a is not None
        else None,
    }


def parse_actions(page: Page) -> dict[str, Any]:
    dom = page.dom
    open_count = overdue = 0
    as_of = None
    for p in dom.find_all("p", "muted"):
        match = _OPEN_SUMMARY_RE.search(p.text())
        if match:
            open_count, overdue, as_of = int(match.group(1)), int(match.group(2)), match.group(3)
    owners = []
    closed = []
    for card in dom.find_all("div", "card"):
        eyebrow = card.find("p", "eyebrow")
        table = card.find("table")
        if eyebrow is None or table is None:
            continue
        label = eyebrow.text()
        if label == "Closed this week":
            for tr in table.find_all("tr"):
                row = _action_row(tr, None)
                if row is not None:
                    row["owner"] = _text(tr.find("td", attrs={"data-label": "Owner"})) or None
                    row["closed"] = _text(tr.find("td", attrs={"data-label": "Closed"}))
                    row["status"] = _text(tr.find("td", attrs={"data-label": "Status"}))
                    row.pop("due", None)
                    row.pop("overdue", None)
                    closed.append(row)
            continue
        owner = _OWNER_COUNT_SUFFIX_RE.sub("", label).strip()
        rows = [r for tr in table.find_all("tr") if (r := _action_row(tr, owner)) is not None]
        owners.append({"owner": owner, "actions": rows})
    return {
        "open": open_count,
        "overdue": overdue,
        "as_of": as_of,
        "owners": owners,
        "actions": [a for o in owners for a in o["actions"]],
        "closed_this_week": closed,
    }


# ---- review ----------------------------------------------------------------


def parse_review(page: Page) -> dict[str, Any]:
    dom = page.dom
    open_count = 0
    for p in dom.find_all("p", "eyebrow"):
        match = _REVIEW_EYEBROW_RE.search(p.text())
        if match:
            open_count = int(match.group(1))
    pages = 1
    for span in dom.find_all("span", "mono"):
        match = _PAGE_OF_RE.search(span.text())
        if match:
            pages = int(match.group(2))
    cards = []
    for card in dom.find_all("div", "review-card"):
        row_id = None
        for form in card.find_all("form"):
            match = _REVIEW_ID_RE.match(form.attrs.get("action", ""))
            if match:
                row_id = int(match.group(1))
                break
        cards.append(
            {
                "id": row_id,
                "category": card.attrs.get("data-category", ""),
                "headline": _text(card.find("p", "review-headline")),
                "claim": _text(card.find("p", "claim-text")) or None,
                "opened": _text(card.find("span", "updated")).removeprefix("opened").strip(),
            }
        )
    return {"open": open_count, "pages": pages, "cards": cards}


# ---- usage -----------------------------------------------------------------


def parse_usage(page: Page) -> dict[str, Any]:
    dom = page.dom

    def records(table_id: str) -> list[dict[str, Any]]:
        return table_records(dom.find("table", attrs={"id": table_id}))

    totals = records("usage-totals")
    return {
        "totals": totals[0] if totals else {},
        "weeks": records("usage-weeks"),
        "people": records("usage-people"),
        "unanswered": records("usage-unanswered"),
    }


# ---- reports -----------------------------------------------------------------


def parse_reports(page: Page) -> list[dict[str, Any]]:
    reports = []
    for card in page.dom.find_all("div", "card"):
        a = card.find("a")
        if a is None or not a.attrs.get("href", "").startswith("/reports/"):
            continue
        if a.attrs["href"].endswith(".md"):
            continue
        reports.append(
            {
                "slug": _slug_from(a.attrs["href"], "/reports/"),
                "title": a.text(),
                "updated": _text(card.find("p", "updated")).removeprefix("updated:").strip(),
            }
        )
    return reports
