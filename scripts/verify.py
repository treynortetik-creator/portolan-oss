"""`scripts/verify.py`: HTTP-level proof that a running Portolan works.

    uv run python scripts/verify.py <target> [--base-url URL]

Guardrails plan 2026-09-22, rank 8 (`01-spec/guardrails-plan-2026-09-22.md`):
the same proof every run, as a file on disk that would look different if the
feature were broken. Every run writes `var/proof/<UTC timestamp>-<target>.json`
(gitignored: `var/` is in `.gitignore`) holding every check -- URL, status,
elapsed ms, byte count, and the content markers found -- and prints one line:

    PASS 24/24 -> var/proof/20260922T200000Z-all.json

Exit 0 when every check passes, 1 when any fails.

A status code alone proves little (an empty Actions page is still a 200), so
each page check also counts content markers read from the templates in
`portolan/web/templates/`, never from demo text: Actions needs a row with an
owner and a due date, Review an open row with a kind, the newest weekly
report an action-card line, and so on. `FEATURES.md` names the target for
each feature.

**Read-only.** GET requests, the login POST, and the MCP JSON-RPC handshake
(`initialize`, `notifications/initialized`, `tools/list`, one unauthenticated
POST that must be refused, then a DELETE that closes the session). No tool
call, no chat question, no form submission, no paid model call. Two GET pages
write an access-log row by design (`/originals/...` and `/sources/...`, spec
sec 12: "access log on every original opened").

**The one exception is the opt-in `chat` target**, never part of `all`:

    uv run python scripts/verify.py chat --question "..." [--question ...]
        [--questions-file scripts/demo_questions.txt] [--expect-nothing "..."]

Each question is a paid model call on a live site. It asks every question
the way the browser does (`POST /chat/new/stream`, then the same
conversation's `/chat/<id>/stream`, a multipart `question` field, the SSE
reply read to its `done` event), all in one conversation owned by
`--editor`. Per question it checks that an answer arrived, that it carries a
citation (or, for an `--expect-nothing` question, that it is the "Nothing
recorded" message), and that every citation resolves: the chip's note page
returns 200 and holds the cited claim, the "Open the original" page returns
200 and shows the block at the cited locator, and the claim's quote is on
that page (the production quote gate's rule, exact or near-verbatim; for a
CRM row, field by field, `row_quote_match`). It
records sentences withheld by the number gate (not a failure) and latency.
The result carries a `chat` list with each question, the answer text, and
every citation's outcome.

**Keys.** The org key comes from `--org-key-file` (default
`var/railway/org-key.txt`), else `PORTOLAN_ORG_KEY`; the MCP key from
`--mcp-key-file` (default `var/railway/mcp-key.txt`), else
`PORTOLAN_MCP_ORG_KEY` (the setting `docs/mcp-hosted.md` names). Neither key
is printed, written to the result file, or put in a URL: the org key travels
only in the login form body, the MCP key only in an `Authorization` header,
and the result records only where each key came from. The result is scanned
for both keys and the session cookie before it is written.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, quote, urlencode, urlsplit

import httpx

# The HTTP client pieces live in `portolan/client/` so the `portolan` CLI
# shares them; the names below are re-exported because
# `tests/test_verify_script.py` reaches them as `verify.<name>`.
from portolan.client.chat import (  # noqa: F401
    GAPS_PREFIX,
    INDEX_BROKEN_PREFIX,
    NOTHING_RECORDED_PREFIX,
    check_original,
    claim_quote,
    classify_outcome,
    parse_sentence_html,
    resolve_citation,
    sse_events,
    stream_question,
)
from portolan.client.htmltree import Node, Page, parse_html  # noqa: F401
from portolan.client.http import SESSION_COOKIE, make_http_client
from portolan.client.keys import Key, resolve_key  # noqa: F401
from portolan.client.mcp import (  # noqa: F401
    AI_ACCESS_CLOSED_CODE,
    MCP_PROTOCOL_VERSION,
    jsonrpc_error_code,
    jsonrpc_messages,
    jsonrpc_result,
)
from portolan.client.quotes import FUZZY_QUOTE_THRESHOLD, quote_match, row_quote_match  # noqa: F401

# The private names `tests/test_chat_passages.py` and `tests/test_originals_crm.py` call.
_check_original = check_original
_outcome = classify_outcome

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_ORG_KEY_FILE = REPO_ROOT / "var" / "railway" / "org-key.txt"
DEFAULT_MCP_KEY_FILE = REPO_ROOT / "var" / "railway" / "mcp-key.txt"
DEFAULT_OUT_DIR = REPO_ROOT / "var" / "proof"
ORG_KEY_ENV = "PORTOLAN_ORG_KEY"
MCP_KEY_ENV = "PORTOLAN_MCP_ORG_KEY"
DEFAULT_EDITOR = "verify.py"
DEFAULT_TIMEOUT_SECONDS = 30.0

# `portolan/web/routes/actions.py` UNASSIGNED_LABEL.
UNASSIGNED_OWNER = "Unassigned"
# The tools `docs/mcp-hosted.md` names as the door's core; the full list
# the server returns is recorded, but only these must be present, so a new
# tool does not fail the check. `tests/test_verify_script.py` pins these
# to `portolan.mcp_server.server.TOOL_NAMES`.
REQUIRED_MCP_TOOLS = ("search", "open_note", "open_source")

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}\b")
_REPORT_HREF_RE = re.compile(r"^/reports/(report-(\d{4})-W(\d{2}))$")
# `portolan.report.researcher.action_card_report_lines`: one line per
# action_card review row, `[verdict] what (state)`, the state one of
# proposed/confirmed/dismissed.
_ACTION_CARD_LINE_RE = re.compile(
    r"^\[[^\]]+\] .+\((?:proposed, confidence: [^)]*|confirmed[^)]*|dismissed)\)$"
)
_ACTION_CARD_PARTS_RE = re.compile(r"^\[[^\]]+\] (.+) \((proposed|confirmed|dismissed)[^)]*\)$")
# `portolan/web/review_presenter.py` _CATEGORY_ACTION_CARD; the page lists 25 cards a page.
ACTION_CARD_CATEGORY = "Proposed action"
# `portolan/web/review_presenter.py` _CATEGORY_ACTION_DONE.
ACTION_DONE_CATEGORY = "Proposed done"
REVIEW_PAGE_SIZE = 25
REVIEW_MAX_PAGES = 10
_OPEN_SUMMARY_RE = re.compile(r"(\d+) open, (\d+) overdue")
_REVIEW_EYEBROW_RE = re.compile(r"Review\s*·\s*(\d+) open")
_RESULT_COUNT_RE = re.compile(r"^(\d+) results?$")
_OWNER_COUNT_SUFFIX_RE = re.compile(r"\s*\(\d+\)\s*$")

# The chat stream sends nothing until the model has answered, so its read
# timeout covers a whole model call.
DEFAULT_CHAT_TIMEOUT_SECONDS = 180.0
EXPECT_ANSWER = "answer"
EXPECT_NOTHING = "nothing-recorded"


# ---------------------------------------------------------------------
# Pages and markers
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class Marker:
    """One content marker: `count(page)` must land in [minimum, maximum].
    Only the count is recorded, never the matched text."""

    name: str
    count: Callable[[Page], int]
    minimum: int = 1
    maximum: int | None = None


def evaluate_marker(marker: Marker, page: Page) -> dict[str, Any]:
    try:
        found = int(marker.count(page))
    except Exception as exc:  # a marker that cannot be read is a failed marker
        return {
            "name": marker.name,
            "found": None,
            "min": marker.minimum,
            "max": marker.maximum,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    ok = found >= marker.minimum and (marker.maximum is None or found <= marker.maximum)
    return {
        "name": marker.name,
        "found": found,
        "min": marker.minimum,
        "max": marker.maximum,
        "ok": ok,
    }


def count_elements(tag: str | None = None, cls: str | None = None, **attrs: str) -> Callable:
    normalized = {k.replace("_", "-"): v for k, v in attrs.items()}
    return lambda page: len(page.dom.find_all(tag, cls, normalized))


def count_substring(needle: str) -> Callable:
    return lambda page: page.text.count(needle)


def header_equals(name: str, expected: str) -> Callable:
    return lambda page: int(page.response.headers.get(name) == expected)


def header_startswith(name: str, prefix: str) -> Callable:
    return lambda page: int((page.response.headers.get(name) or "").startswith(prefix))


def sets_cookie(name: str) -> Callable:
    return lambda page: sum(
        1 for v in page.response.headers.get_list("set-cookie") if v.startswith(f"{name}=")
    )


# ---------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------


class Run:
    def __init__(
        self,
        *,
        base_url: str,
        org_key: Key,
        mcp_key: Key,
        editor: str,
        timeout: float,
        query: str | None,
        note_slug: str | None,
        report_slug: str | None,
        questions: list[tuple[str, str]] | None = None,
        chat_timeout: float = DEFAULT_CHAT_TIMEOUT_SECONDS,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.org_key = org_key
        self.mcp_key = mcp_key
        self.editor = editor
        self.timeout = timeout
        # (question, EXPECT_ANSWER | EXPECT_NOTHING), for the chat target.
        self.questions = questions or []
        self.chat_timeout = chat_timeout
        self.checks: list[dict[str, Any]] = []
        self.done: set[str] = set()
        self.derived: dict[str, Any] = {}
        # One record per question the chat target asked.
        self.chat: list[dict[str, Any]] = []
        if query:
            self.derived["query"] = query
            self.derived["query_source"] = "--query"
        if note_slug:
            self.derived["note_slug"] = note_slug
        if report_slug:
            self.derived["report_slug"] = report_slug
        # `session` carries the login cookie; `anon` never has one, for the
        # auth-gate checks and the public explainer.
        self.session = make_http_client(timeout, "portolan-verify/1")
        self.anon = make_http_client(timeout, "portolan-verify/1")

    def close(self) -> None:
        self.session.close()
        self.anon.close()

    def secrets(self) -> list[str]:
        values = [self.org_key.value, self.mcp_key.value, self.session.cookies.get(SESSION_COOKIE)]
        return [v for v in values if v]

    def fail(self, target: str, name: str, error: str, **fields: Any) -> None:
        self.checks.append(
            {"target": target, "name": name, "ok": False, "error": error, "status": None, **fields}
        )

    def request(
        self,
        target: str,
        name: str,
        method: str,
        path: str,
        *,
        expect: tuple[int, ...] = (200,),
        markers: tuple[Marker, ...] = (),
        anonymous: bool = False,
        **kwargs: Any,
    ) -> Page | None:
        url = f"{self.base_url}{path}"
        client = self.anon if anonymous else self.session
        check: dict[str, Any] = {
            "target": target,
            "name": name,
            "method": method,
            "path": path,
            "url": url,
            "expected_status": list(expect),
        }
        started = time.perf_counter()
        try:
            response = client.request(method, url, follow_redirects=False, **kwargs)
        except httpx.HTTPError as exc:
            check.update(
                status=None,
                elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
            )
            self.checks.append(check)
            return None
        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
        page = Page(response)
        marker_results = [evaluate_marker(m, page) for m in markers]
        errors = []
        if response.status_code not in expect:
            errors.append(f"status {response.status_code}, expected {list(expect)}")
        for m in marker_results:
            if not m["ok"]:
                wanted = f"{m['min']} to {m['max']}" if m["max"] is not None else f">= {m['min']}"
                errors.append(f"marker '{m['name']}' found {m['found']}, need {wanted}")
        check.update(
            status=response.status_code,
            elapsed_ms=elapsed_ms,
            bytes=len(response.content),
            content_type=response.headers.get("content-type"),
            markers=marker_results,
            ok=not errors,
            error="; ".join(errors) or None,
        )
        if response.headers.get("location"):
            check["location"] = response.headers["location"]
        self.checks.append(check)
        return page

    def note(self, **values: Any) -> None:
        """Record values this run derived (a report slug, a query) on the
        last check and for later targets."""
        self.derived.update(values)
        if self.checks:
            self.checks[-1].setdefault("derived", {}).update(values)

    def ensure(self, target: str) -> None:
        if target not in self.done:
            (TARGETS.get(target) or OPT_IN_TARGETS[target])(self)
            self.done.add(target)


# ---------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------


def target_health(run: Run) -> None:
    run.request(
        "health",
        "health probe runs SELECT 1",
        "GET",
        "/healthz",
        anonymous=True,
        markers=(
            Marker("status ok in the JSON body", lambda p: int(p.json() == {"status": "ok"})),
        ),
    )


def target_login(run: Run) -> None:
    run.request(
        "login",
        "home without a session redirects to /login",
        "GET",
        "/",
        expect=(302,),
        anonymous=True,
        markers=(Marker("Location is /login", header_equals("location", "/login")),),
    )
    run.request(
        "login",
        "login form",
        "GET",
        "/login",
        anonymous=True,
        markers=(
            Marker("org_key field", count_elements("input", name="org_key")),
            Marker("editor field", count_elements("input", name="editor")),
            Marker("form posts to /login", count_elements("form", method="post", action="/login")),
        ),
    )
    if not run.org_key.value:
        run.fail("login", "login POST", f"org key {run.org_key.source}")
        return
    page = run.request(
        "login",
        "login POST sets a session and redirects home",
        "POST",
        "/login",
        expect=(303,),
        data={"org_key": run.org_key.value, "editor": run.editor, "next": "/"},
        markers=(
            Marker("Location is /", header_equals("location", "/")),
            Marker("sets the session cookie", sets_cookie(SESSION_COOKIE)),
        ),
    )
    if page is None:
        return
    secure = any(
        h.startswith(f"{SESSION_COOKIE}=")
        and "secure" in {a.strip().lower() for a in h.split(";")[1:]}
        for h in page.response.headers.get_list("set-cookie")
    )
    if secure and urlsplit(run.base_url).scheme == "http":
        # httpx, like a browser, will not send a Secure cookie over plain
        # http, so every page after this would redirect to /login.
        run.checks[-1]["hint"] = (
            "the session cookie is Secure but the base URL is http; start a local app "
            "with PORTOLAN_COOKIE_SECURE=false"
        )


def _open_count(page: Page) -> int:
    summary = next(
        (
            p.text()
            for p in page.dom.find_all("p", cls="muted")
            if _OPEN_SUMMARY_RE.search(p.text())
        ),
        "",
    )
    match = _OPEN_SUMMARY_RE.search(summary)
    return int(match.group(1)) if match else 0


def _actions_with_owner_and_due(page: Page) -> int:
    """Rows in an owner's group (not Unassigned) whose Due cell holds a
    date and whose What cell links to the action's note."""
    count = 0
    for card in page.dom.find_all("div", cls="card"):
        eyebrow = card.find("p", cls="eyebrow")
        table = card.find("table")
        if eyebrow is None or table is None:
            continue
        owner = _OWNER_COUNT_SUFFIX_RE.sub("", eyebrow.text()).strip()
        if not owner or owner == UNASSIGNED_OWNER:
            continue
        for row in table.find_all("tr"):
            what = row.find("td", attrs={"data-label": "What"})
            due = row.find("td", attrs={"data-label": "Due"})
            if what is None or due is None or not _DATE_RE.match(due.text()):
                continue
            if any(a.attrs.get("href", "").startswith("/notes/") for a in what.find_all("a")):
                count += 1
    return count


def target_home(run: Run) -> None:
    run.ensure("login")
    page = run.request(
        "home",
        "chat home",
        "GET",
        "/",
        markers=(
            Marker("chat composer", count_elements("form", id="composer")),
            Marker(
                "signed-in name in the nav",
                lambda p: sum(
                    1 for s in p.dom.find_all("span", cls="who") if s.text() == run.editor
                ),
            ),
            Marker("example questions built from the graph", count_elements("button", "example-q")),
        ),
    )
    if page is not None and page.response.status_code == 200:
        footer = page.dom.find("footer", cls="site-footer")
        version = footer.find("span", cls="mono") if footer else None
        if version is not None:
            run.note(app_version=version.text())


def target_actions(run: Run) -> None:
    run.ensure("login")
    run.request(
        "actions",
        "open actions by owner",
        "GET",
        "/actions",
        markers=(
            Marker("open count in the summary line", _open_count),
            Marker("action rows with an owner and a due date", _actions_with_owner_and_due),
        ),
    )


def _review_open_count(page: Page) -> int:
    eyebrow = next(
        (p.text() for p in page.dom.find_all("p", cls="eyebrow") if "Review" in p.text()), ""
    )
    match = _REVIEW_EYEBROW_RE.search(eyebrow)
    return int(match.group(1)) if match else 0


def _review_cards_with_kind(page: Page) -> int:
    count = 0
    for card in page.dom.find_all("div", cls="review-card"):
        badge = card.find("span", cls="badge")
        if card.attrs.get("data-category", "").strip() and badge is not None and badge.text():
            count += 1
    return count


def target_review(run: Run) -> None:
    run.ensure("login")
    run.request(
        "review",
        "open review queue",
        "GET",
        "/review",
        markers=(
            Marker("open count in the eyebrow", _review_open_count),
            Marker("open rows with a kind", _review_cards_with_kind),
        ),
    )


def _done_cards(page: Page) -> list[Node]:
    return page.dom.find_all(
        "div", cls="review-card", attrs={"data-category": ACTION_DONE_CATEGORY}
    )


def _done_cards_with_evidence(page: Page) -> int:
    """Proposed-done cards showing the answering claim and a link to its
    original."""
    return sum(
        1
        for card in _done_cards(page)
        if card.find("p", cls="claim-text") is not None
        and any(a.attrs.get("href", "").startswith("/originals/") for a in card.find_all("a"))
    )


def _done_cards_naming_the_action(page: Page) -> int:
    count = 0
    for card in _done_cards(page):
        block = card.find("div", cls="review-action-done")
        links = block.find_all("a") if block is not None else []
        if any(a.attrs.get("href", "").startswith("/notes/") for a in links):
            count += 1
    return count


def target_action_done(run: Run) -> None:
    """Opt-in (not in `all`): proposals exist only after someone runs
    `python -m portolan.report propose-done` or `weekly --propose-done`."""
    run.ensure("login")
    run.request(
        "action-done",
        "proposed-done cards on /review",
        "GET",
        f"/review?kind={quote(ACTION_DONE_CATEGORY)}",
        markers=(
            Marker("proposed-done cards", lambda p: len(_done_cards(p))),
            Marker("cards with the answering claim and its original", _done_cards_with_evidence),
            Marker("cards linking the action they would close", _done_cards_naming_the_action),
        ),
    )


def _report_links(page: Page) -> list[tuple[int, int, str]]:
    found = set()
    for a in page.dom.find_all("a"):
        match = _REPORT_HREF_RE.match(a.attrs.get("href", ""))
        if match:
            found.add((int(match.group(2)), int(match.group(3)), match.group(1)))
    return sorted(found)


def _action_card_lines(page: Page) -> int:
    body = page.dom.find("div", cls="report-body")
    if body is None:
        return 0
    return sum(1 for li in body.find_all("li") if _ACTION_CARD_LINE_RE.match(li.text()))


def target_reports(run: Run) -> None:
    run.ensure("login")
    page = run.request(
        "reports",
        "weekly reports list",
        "GET",
        "/reports",
        markers=(Marker("weekly report links", lambda p: len(_report_links(p))),),
    )
    slug = run.derived.get("report_slug")
    if slug is None:
        links = _report_links(page) if page is not None and page.response.status_code == 200 else []
        if not links:
            run.fail(
                "reports", "newest weekly report", "no /reports/report-<week> link on /reports"
            )
            return
        slug = links[-1][2]
        run.note(report_slug=slug)
    run.request(
        "reports",
        "newest weekly report",
        "GET",
        f"/reports/{quote(slug)}",
        markers=(
            Marker(
                "report title",
                lambda p: sum(
                    1 for h in p.dom.find_all("h1") if h.text().startswith("Weekly report")
                ),
            ),
            Marker("report body sections", lambda p: _count_in(p, "div", "report-body", "h3")),
            Marker("action cards under Suggested actions", _action_card_lines),
            Marker(
                "markdown download link", count_elements("a", href=f"/reports/{quote(slug)}.md")
            ),
        ),
    )


def _open_action_card_headlines(run: Run, target: str) -> list[str] | None:
    """Every open action card's headline on `/review` (it ends with the
    card's `what`), or `None` when a page failed."""
    headlines: list[str] = []
    for page_no in range(1, REVIEW_MAX_PAGES + 1):
        page = run.request(
            target,
            f"open action cards on /review, page {page_no}",
            "GET",
            f"/review?kind={quote(ACTION_CARD_CATEGORY)}&page={page_no}",
        )
        if page is None or page.response.status_code != 200:
            return None
        cards = page.dom.find_all(
            "div", cls="review-card", attrs={"data-category": ACTION_CARD_CATEGORY}
        )
        for card in cards:
            headline = card.find("p", cls="review-headline")
            if headline is not None:
                headlines.append(headline.text())
        if len(cards) < REVIEW_PAGE_SIZE:
            break
    return headlines


def _cards_out_of_step(page: Page, open_headlines: list[str]) -> int:
    """Report card lines whose state disagrees with the review queue now:
    "proposed" with no open card for that `what`, or confirmed/dismissed
    while one is still open."""
    body = page.dom.find("div", cls="report-body")
    if body is None:
        return 0
    wrong = 0
    for li in body.find_all("li"):
        match = _ACTION_CARD_PARTS_RE.match(li.text())
        if match is None:
            continue
        what, state = match.group(1), match.group(2)
        still_open = any(h.endswith(what) for h in open_headlines)
        if (state == "proposed") != still_open:
            wrong += 1
    return wrong


def target_report_cards(run: Run) -> None:
    """The newest report's action-card lines read the review queue at view
    time: a card confirmed or dismissed on `/review` after the report was
    written must not still read "proposed" (and the reverse)."""
    run.ensure("reports")
    slug = run.derived.get("report_slug")
    if slug is None:
        run.fail("report-cards", "report card states", "no weekly report to check")
        return
    headlines = _open_action_card_headlines(run, "report-cards")
    if headlines is None:
        return
    run.request(
        "report-cards",
        "report card states match the review queue",
        "GET",
        f"/reports/{quote(slug)}",
        markers=(
            Marker("action cards under Suggested actions", _action_card_lines),
            Marker(
                "card lines out of step with the review queue",
                lambda p: _cards_out_of_step(p, headlines),
                minimum=0,
                maximum=0,
            ),
        ),
    )
    run.note(open_action_cards=len(headlines))


def _action_with_why(page: Page) -> tuple[str, str | None] | None:
    """The first `/actions` row carrying a researched one-line why: its
    note slug and its account's slug."""
    for row in page.dom.find_all("tr"):
        what = row.find("td", attrs={"data-label": "What"})
        if what is None or what.find("div", cls="muted") is None:
            continue
        link = next(
            (a for a in what.find_all("a") if a.attrs.get("href", "").startswith("/notes/")),
            None,
        )
        if link is None:
            continue
        account_cell = row.find("td", attrs={"data-label": "Account"})
        account = account_cell.find("a") if account_cell is not None else None
        account_slug = account.attrs["href"].removeprefix("/notes/") if account else None
        return link.attrs["href"].removeprefix("/notes/"), account_slug
    return None


def _md_evidence_lines(page: Page) -> int:
    section = page.text.split("\n## Evidence\n", 1)
    if len(section) < 2:
        return 0
    body = section[1].split("\n## ", 1)[0]
    return sum(1 for line in body.splitlines() if line.startswith("- ") and "[source: " in line)


def _atlas_open_actions(page: Page) -> int:
    card = page.dom.find("div", cls="note-actions")
    if card is None:
        return 0
    return sum(1 for a in card.find_all("a") if a.attrs.get("href", "").startswith("/atlas/"))


def target_action_note(run: Run) -> None:
    """A card-confirmed action: its `.md` download carries the why, its
    Atlas pane shows it, and its account's Atlas pane lists the action."""
    run.ensure("login")
    page = run.request("action-note", "actions with a researched why", "GET", "/actions")
    if page is None or page.response.status_code != 200:
        return
    found = _action_with_why(page)
    if found is None:
        run.fail(
            "action-note",
            "a card-confirmed action",
            "no row on /actions carries a researched why (confirm a card on /review first)",
        )
        return
    slug, account_slug = found
    run.note(action_note_slug=slug, action_account_slug=account_slug)
    run.request(
        "action-note",
        "action .md download",
        "GET",
        f"/notes/{quote(slug)}.md",
        markers=(
            Marker("attachment header", header_startswith("content-disposition", "attachment")),
            Marker("Why section", count_substring("\n## Why\n")),
            Marker("evidence lines with a source link", _md_evidence_lines),
            Marker(
                "What the system checked section", count_substring("\n## What the system checked\n")
            ),
            Marker("confirmed-by line", count_substring(", confirmed by ")),
            Marker("boilerplate synthesis", count_substring("is an open action."), 0, 0),
        ),
    )
    run.request(
        "action-note",
        "action in Atlas",
        "GET",
        f"/atlas/{quote(slug)}",
        markers=(
            Marker(
                "Why card",
                lambda p: sum(1 for e in p.dom.find_all("p", cls="eyebrow") if e.text() == "Why"),
            ),
        ),
    )
    if account_slug is None:
        run.fail("action-note", "account in Atlas", "the action row names no account")
        return
    run.request(
        "action-note",
        "account in Atlas lists its open actions",
        "GET",
        f"/atlas/{quote(account_slug)}",
        markers=(Marker("open actions linking into Atlas", _atlas_open_actions),),
    )


def _count_in(page: Page, tag: str, cls: str, inner_tag: str) -> int:
    container = page.dom.find(tag, cls)
    return len(container.find_all(inner_tag)) if container is not None else 0


def target_how_it_works(run: Run) -> None:
    run.request(
        "how-it-works",
        "public explainer, no session",
        "GET",
        "/how-it-works",
        anonymous=True,
        markers=(
            Marker("narration audio", count_substring("<audio")),
            Marker("link back to the app", count_substring('class="home-link" href="/"')),
            Marker("full explainer (bytes)", lambda p: len(p.response.content), minimum=100_000),
        ),
    )


def _folder_links(page: Page, label: str) -> list[Node]:
    for folder in page.dom.find_all("details", cls="atlas-folder"):
        summary = folder.find("summary")
        first_span = summary.find("span") if summary is not None else None
        if first_span is not None and first_span.text() == label:
            return folder.find_all("a", cls="atlas-note-link")
    return []


def target_atlas(run: Run) -> None:
    run.ensure("login")
    page = run.request(
        "atlas",
        "note tree",
        "GET",
        "/atlas",
        markers=(
            Marker("folders in the tree", count_elements("details", "atlas-folder")),
            Marker("note links in the tree", count_elements("a", "atlas-note-link")),
        ),
    )
    if page is None or page.response.status_code != 200:
        return
    all_links = page.dom.find_all("a", cls="atlas-note-link")
    signal_links = _folder_links(page, "Signals") or all_links
    account_links = _folder_links(page, "Accounts") or all_links
    derived: dict[str, Any] = {}
    if "note_slug" not in run.derived and signal_links:
        derived["note_slug"] = signal_links[0].attrs["href"].removeprefix("/atlas/").split("?")[0]
    if "query" not in run.derived and account_links:
        derived["query"] = account_links[0].text()
        derived["query_source"] = "first note title in the Atlas Accounts folder"
    if derived:
        run.note(**derived)


def _result_count(page: Page) -> int:
    for p in page.dom.find_all("p", cls="mono"):
        match = _RESULT_COUNT_RE.match(p.text())
        if match:
            return int(match.group(1))
    return 0


def target_search(run: Run) -> None:
    if "query" not in run.derived:
        run.ensure("atlas")
    else:
        run.ensure("login")
    query = run.derived.get("query")
    if not query:
        run.fail("search", "search redirect", "no query: pass --query or fix the atlas target")
        return
    page = run.request(
        "search",
        "search redirects to Atlas",
        "GET",
        f"/search?{urlencode({'q': query})}",
        expect=(302,),
        markers=(Marker("Location is /atlas?q=", header_startswith("location", "/atlas?q=")),),
    )
    location = page.response.headers.get("location") if page is not None else None
    if not location or not location.startswith("/atlas?q="):
        run.fail("search", "search results", "no /atlas?q= redirect to follow")
        return
    run.request(
        "search",
        "search results (fused keyword + vector)",
        "GET",
        location,
        markers=(
            Marker("result count line", _result_count),
            Marker("result cards", count_elements("div", "note-card")),
            Marker(
                "no 'index looks broken' warning",
                count_substring("Index looks broken"),
                minimum=0,
                maximum=0,
            ),
        ),
    )


def _original_links(page: Page) -> list[str]:
    return [
        a.attrs["href"]
        for a in page.dom.find_all("a", cls="dig")
        if a.attrs.get("href", "").startswith("/originals/") and "?at=" in a.attrs["href"]
    ]


def target_note(run: Run) -> None:
    if "note_slug" not in run.derived:
        run.ensure("atlas")
    else:
        run.ensure("login")
    slug = run.derived.get("note_slug")
    if not slug:
        run.fail("note", "note page", "no note slug: pass --note-slug or fix the atlas target")
        return
    page = run.request(
        "note",
        "note page with claims",
        "GET",
        f"/notes/{quote(slug)}",
        markers=(
            Marker("note title", lambda p: sum(1 for h in p.dom.find_all("h1") if h.text())),
            Marker("claims", count_elements("div", "claim")),
            Marker("links to the original at a locator", lambda p: len(_original_links(p))),
        ),
    )
    if page is not None and page.response.status_code == 200:
        links = _original_links(page)
        if links:
            run.note(original_href=links[0])


def target_source(run: Run) -> None:
    run.ensure("note")
    href = run.derived.get("original_href")
    if not href:
        run.fail("source", "source page", "no /originals/<slug>?at= link on the note page")
        return
    parts = urlsplit(href)
    source_slug = parts.path.removeprefix("/originals/")
    locator = dict(parse_qsl(parts.query)).get("at", "")
    run.note(source_slug=source_slug, locator=locator)
    run.request(
        "source",
        "source page",
        "GET",
        f"/sources/{source_slug}",
        markers=(
            Marker("open original button", count_elements("a", href=f"/originals/{source_slug}")),
            Marker(
                "claims citing this source at a locator",
                lambda p: sum(
                    1
                    for a in p.dom.find_all("a", cls="dig")
                    if a.attrs.get("href", "").startswith(f"/originals/{source_slug}?at=")
                ),
            ),
        ),
    )

    def _cited_block(page: Page) -> int:
        body = page.dom.find(attrs={"id": "original-body"})
        if body is None:
            return 0
        return len(body.find_all(attrs={"data-locator": locator}))

    run.request(
        "source",
        "original document at the cited locator",
        "GET",
        href,
        markers=(
            Marker("document body", count_elements(id="original-body")),
            Marker("block at the cited locator", _cited_block),
        ),
    )


def target_coverage(run: Run) -> None:
    run.ensure("login")
    run.request(
        "coverage",
        "ingest coverage ledger (all history)",
        "GET",
        "/coverage?days=3650",
        markers=(
            Marker("ledger rows", count_elements("td", data_label="Day")),
            Marker(
                "not-ingested and dead-letter sections",
                lambda p: sum(
                    1
                    for e in p.dom.find_all("p", cls="eyebrow")
                    if e.text().startswith(("Not ingested", "Dead letter"))
                ),
                minimum=2,
            ),
        ),
    )


def target_graph(run: Run) -> None:
    run.ensure("login")
    run.request(
        "graph",
        "graph page",
        "GET",
        "/graph",
        markers=(Marker("graph canvas", count_elements("div", id="graph-2d-canvas")),),
    )
    run.request(
        "graph",
        "graph data",
        "GET",
        "/graph.json",
        markers=(
            Marker("nodes", lambda p: len(p.json()["nodes"])),
            Marker("edges", lambda p: len(p.json()["edges"])),
        ),
    )


def _jsonrpc_error_code(page: Page | None, request_id: int) -> int | None:
    return None if page is None else jsonrpc_error_code(page.response, request_id)


def _hint_if_door_closed(run: Run, page: Page | None, request_id: int) -> None:
    """The check stays failed; this only says why and what to run."""
    if (
        run.checks
        and not run.checks[-1]["ok"]
        and _jsonrpc_error_code(page, request_id) == (AI_ACCESS_CLOSED_CODE)
    ):
        run.checks[-1]["hint"] = (
            f"AI access is off for this tenant (JSON-RPC error {AI_ACCESS_CLOSED_CODE}, "
            "migration 0031 leaves new databases closed): run "
            "`python -m portolan.mcp_server access on` against the app's database"
        )


def _jsonrpc_result(page: Page, request_id: int) -> dict[str, Any]:
    return jsonrpc_result(page.response, request_id)


def target_mcp(run: Run) -> None:
    accept = {"Accept": "application/json, text/event-stream"}
    run.request(
        "mcp",
        "MCP refuses a request with no key",
        "POST",
        "/mcp",
        expect=(401,),
        anonymous=True,
        headers=accept,
        json={"jsonrpc": "2.0", "id": 1, "method": "ping"},
    )
    if not run.mcp_key.value:
        run.fail("mcp", "MCP initialize", f"MCP key {run.mcp_key.source}")
        return
    auth = {**accept, "Authorization": f"Bearer {run.mcp_key.value}"}
    init = run.request(
        "mcp",
        "MCP initialize",
        "POST",
        "/mcp",
        anonymous=True,
        headers=auth,
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "portolan-verify", "version": "1"},
            },
        },
        markers=(
            Marker(
                "session id header", lambda p: int(bool(p.response.headers.get("mcp-session-id")))
            ),
            Marker(
                "server is portolan",
                lambda p: int(_jsonrpc_result(p, 1)["serverInfo"]["name"] == "portolan"),
            ),
        ),
    )
    _hint_if_door_closed(run, init, 1)
    session_id = init.response.headers.get("mcp-session-id") if init is not None else None
    if not session_id:
        run.fail("mcp", "MCP tools/list", "initialize returned no mcp-session-id")
        return
    try:
        protocol = _jsonrpc_result(init, 1).get("protocolVersion") or MCP_PROTOCOL_VERSION
    except (ValueError, KeyError, json.JSONDecodeError):
        protocol = MCP_PROTOCOL_VERSION
    session_headers = {**auth, "mcp-session-id": session_id, "mcp-protocol-version": protocol}
    run.request(
        "mcp",
        "MCP initialized notification",
        "POST",
        "/mcp",
        expect=(202,),
        anonymous=True,
        headers=session_headers,
        json={"jsonrpc": "2.0", "method": "notifications/initialized"},
    )
    tools = run.request(
        "mcp",
        "MCP tools/list",
        "POST",
        "/mcp",
        anonymous=True,
        headers=session_headers,
        json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        markers=(
            Marker("tools listed", lambda p: len(_jsonrpc_result(p, 2)["tools"])),
            Marker(
                "core tools present: " + ", ".join(REQUIRED_MCP_TOOLS),
                lambda p: len(
                    set(REQUIRED_MCP_TOOLS) & {t["name"] for t in _jsonrpc_result(p, 2)["tools"]}
                ),
                minimum=len(REQUIRED_MCP_TOOLS),
            ),
        ),
    )
    _hint_if_door_closed(run, tools, 2)
    if tools is not None and tools.response.status_code == 200:
        try:
            names = sorted(t["name"] for t in _jsonrpc_result(tools, 2)["tools"])
        except (ValueError, KeyError, json.JSONDecodeError):
            names = []
        run.note(mcp_tools=names)
    run.request(
        "mcp",
        "MCP session closed",
        "DELETE",
        "/mcp",
        expect=(200, 202, 204),
        anonymous=True,
        headers=session_headers,
    )


# ---------------------------------------------------------------------
# Chat (opt-in: every question is a paid model call)
# ---------------------------------------------------------------------


def _cited_page(run: Run, cache: dict[str, Page | None], name: str, path: str) -> Page | None:
    """GET `path` once per run (answers cite the same notes over and over);
    `None` unless it returned 200."""
    if path not in cache:
        page = run.request("chat", name, "GET", path)
        cache[path] = page if page is not None and page.response.status_code == 200 else None
    return cache[path]


def _check(run: Run, name: str, ok: bool, error: str | None = None, **fields: Any) -> bool:
    run.checks.append(
        {"target": "chat", "name": name, "ok": ok, "error": None if ok else error, **fields}
    )
    return ok


def _ask(
    run: Run,
    label: str,
    question: str,
    expect: str,
    conversation: str,
    notes: dict[str, Page | None],
    originals: dict[str, Page | None],
) -> dict[str, Any]:
    streamed = stream_question(
        run.session,
        run.base_url,
        question,
        conversation,
        timeout=run.timeout,
        chat_timeout=run.chat_timeout,
    )
    done = streamed["done"] or {}
    sentences = streamed["sentences"]
    citations = [c for s in sentences for c in s["citations"]]
    outcome = classify_outcome(streamed)

    arrived = _check(
        run,
        f"{label} answer arrived",
        outcome not in ("error", "no_answer"),
        streamed["error"] or "the stream ended with no sentence",
        method="POST",
        path=streamed["path"],
        url=f"{run.base_url}{streamed['path']}",
        status=streamed["status"],
        content_type=streamed["content_type"],
        elapsed_ms=streamed["elapsed_ms"],
        first_sentence_ms=streamed["first_sentence_ms"],
        answer_ms=streamed["answer_ms"],
        bytes=streamed["bytes"],
        sentences=len(sentences),
    )
    if expect == EXPECT_NOTHING:
        expected = _check(
            run,
            f"{label} answer is 'Nothing recorded'",
            outcome == "nothing_recorded",
            f"expected the 'Nothing recorded' message, got outcome {outcome!r} "
            f"with {len(citations)} citation(s)",
        )
    else:
        expected = _check(
            run,
            f"{label} answer carries a citation",
            bool(citations),
            f"no citation (outcome {outcome!r})",
            citations=len(citations),
        )
    resolved = True
    if citations:
        for citation in citations:
            resolve_citation(
                citation,
                lambda path: _cited_page(run, notes, "cited note page", path),
                lambda path: _cited_page(run, originals, "cited original", path),
            )
        problems = [p for c in citations for p in c["problems"]]
        more = f" (+{len(problems) - 5} more)" if len(problems) > 5 else ""
        resolved = _check(
            run,
            f"{label} every citation resolves",
            not problems,
            "; ".join(problems[:5]) + more,
            citations=len(citations),
            resolved=sum(1 for c in citations if c["ok"]),
            quotes_checked=sum(1 for c in citations if "quote_match" in c),
            quotes_found=sum(1 for c in citations if c.get("quote_match", "none") != "none"),
        )
    withheld = int(done.get("numerals_dropped") or 0)
    if streamed["done"] is not None:
        # Recorded, never failed: the gate withholding a sentence is the
        # gate working, and the answer's footnote says so.
        _check(run, f"{label} sentences withheld by the number gate", True, withheld=withheld)
    return {
        "label": label,
        "question": question,
        "expect": expect,
        "ok": arrived and expected and resolved,
        "outcome": outcome,
        "answer_text": " ".join(s["text"] for s in sentences),
        "sentences": sentences,
        "citations": len(citations),
        "withheld_sentences": withheld,
        "model": done.get("model"),
        "conversation_id": done.get("conversation_id"),
        "message_id": done.get("message_id"),
        "first_sentence_ms": streamed["first_sentence_ms"],
        "answer_ms": streamed["answer_ms"],
        "error": streamed["error"],
    }


def target_chat(run: Run) -> None:
    """Opt-in (not in `all`): every question is a paid model call on a live
    site. All questions go to one conversation, the first through
    `/chat/new/stream` as the composer does."""
    if not run.questions:
        run.fail("chat", "chat questions", "no question: pass --question or --questions-file")
        return
    run.ensure("login")
    conversation = "new"
    notes: dict[str, Page | None] = {}
    originals: dict[str, Page | None] = {}
    for n, (question, expect) in enumerate(run.questions, start=1):
        record = _ask(run, f"q{n}", question, expect, conversation, notes, originals)
        run.chat.append(record)
        if record["conversation_id"]:
            conversation = str(record["conversation_id"])
    if conversation != "new":
        run.note(chat_conversation_id=int(conversation))


TARGETS: dict[str, Callable[[Run], None]] = {
    "health": target_health,
    "login": target_login,
    "home": target_home,
    "actions": target_actions,
    "review": target_review,
    "reports": target_reports,
    "report-cards": target_report_cards,
    "action-note": target_action_note,
    "how-it-works": target_how_it_works,
    "atlas": target_atlas,
    "search": target_search,
    "note": target_note,
    "source": target_source,
    "coverage": target_coverage,
    "graph": target_graph,
    "mcp": target_mcp,
}
# Named on the command line only; `all` skips them because a healthy
# deployment can have none of what they check.
OPT_IN_TARGETS: dict[str, Callable[[Run], None]] = {
    "action-done": target_action_done,
    # Also a paid model call per question.
    "chat": target_chat,
}


# ---------------------------------------------------------------------
# Result file
# ---------------------------------------------------------------------


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def _scrub(serialized: str, secrets: list[str]) -> tuple[str, bool]:
    scrubbed = False
    for secret in secrets:
        if secret in serialized:
            serialized = serialized.replace(secret, "[redacted]")
            scrubbed = True
    return serialized, scrubbed


def _write_result(out_dir: Path, stamp: str, target: str, serialized: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stamp}-{target}.json"
    n = 2
    while path.exists():
        path = out_dir / f"{stamp}-{target}-{n}.json"
        n += 1
    path.write_text(serialized + "\n", encoding="utf-8")
    return path


def run_verify(
    target: str,
    *,
    base_url: str,
    org_key: Key,
    mcp_key: Key,
    out_dir: Path,
    editor: str = DEFAULT_EDITOR,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    query: str | None = None,
    note_slug: str | None = None,
    report_slug: str | None = None,
    questions: list[tuple[str, str]] | None = None,
    chat_timeout: float = DEFAULT_CHAT_TIMEOUT_SECONDS,
) -> tuple[dict[str, Any], Path]:
    started = datetime.now(UTC)
    run = Run(
        base_url=base_url,
        org_key=org_key,
        mcp_key=mcp_key,
        editor=editor,
        timeout=timeout,
        query=query,
        note_slug=note_slug,
        report_slug=report_slug,
        questions=questions,
        chat_timeout=chat_timeout,
    )
    try:
        names = list(TARGETS) if target == "all" else [target]
        for name in names:
            run.ensure(name)
        secrets = run.secrets()
    finally:
        run.close()

    passed = sum(1 for c in run.checks if c["ok"])
    result: dict[str, Any] = {
        "tool": "scripts/verify.py",
        "target": target,
        "base_url": run.base_url,
        "started_at": started.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
        "verifier_git_sha": _git_sha(),
        "editor": editor,
        "auth": {"org_key": org_key.source, "mcp_key": mcp_key.source},
        "ok": passed == len(run.checks) and bool(run.checks),
        "passed": passed,
        "total": len(run.checks),
        "derived": run.derived,
        "checks": run.checks,
    }
    if run.chat:
        result["chat"] = run.chat
    serialized, scrubbed = _scrub(json.dumps(result, indent=2, default=str), secrets)
    if scrubbed:
        # A secret reached the result: a bug in this script, never a pass.
        # The value is already replaced; fail the run so it is seen.
        result = json.loads(serialized)
        result["checks"].append(
            {
                "target": target,
                "name": "result file holds no secret",
                "ok": False,
                "status": None,
                "error": "a key or the session cookie appeared in the result and was redacted",
            }
        )
        result.update(ok=False, scrubbed=True, total=len(result["checks"]))
        serialized = json.dumps(result, indent=2, default=str)
    path = _write_result(out_dir, started.strftime("%Y%m%dT%H%M%SZ"), target, serialized)
    return result, path


def summary_line(result: dict[str, Any], path: Path) -> str:
    try:
        shown = path.resolve().relative_to(Path.cwd().resolve())
    except ValueError:
        shown = path
    verdict = "PASS" if result["ok"] else "FAIL"
    line = f"{verdict} {result['passed']}/{result['total']} -> {shown}"
    failed = [f"{c['target']}: {c['name']}" for c in result["checks"] if not c["ok"]]
    if failed:
        more = f" (+{len(failed) - 5} more)" if len(failed) > 5 else ""
        line += " (failed: " + "; ".join(failed[:5]) + more + ")"
    return line


def chat_line(record: dict[str, Any]) -> str:
    """One line per chat question: verdict, latency, counts. Never the answer."""
    verdict = "PASS" if record["ok"] else "FAIL"
    seconds = (
        f"{record['answer_ms'] / 1000:.1f} s" if record["answer_ms"] is not None else "no answer"
    )
    return (
        f"{record['label']} {verdict} {seconds}, {record['outcome']}, "
        f"{len(record['sentences'])} sentences, {record['citations']} citations, "
        f"{record['withheld_sentences']} withheld: {record['question']}"
    )


def collect_questions(
    questions: list[str], questions_file: Path | None, expect_nothing: list[str]
) -> list[tuple[str, str]]:
    """`--question`s, then the file's non-blank lines, then any
    `--expect-nothing` question not already listed; each once, in that
    order. An `--expect-nothing` question keeps its place in the list."""
    texts = [*questions]
    if questions_file is not None:
        texts += questions_file.read_text(encoding="utf-8").splitlines()
    nothing = {q.strip() for q in expect_nothing}
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for text in [*texts, *expect_nothing]:
        text = text.strip()
        if text and text not in seen:
            seen.add(text)
            out.append((text, EXPECT_NOTHING if text in nothing else EXPECT_ANSWER))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="verify.py",
        description="HTTP-level proof that a running Portolan works. Writes var/proof/*.json.",
    )
    parser.add_argument("target", choices=[*TARGETS, *OPT_IN_TARGETS, "all"])
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--org-key-file", type=Path, default=None)
    parser.add_argument("--mcp-key-file", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--editor", default=DEFAULT_EDITOR, help="display name for the login")
    parser.add_argument("--query", default=None, help="search query (default: from Atlas)")
    parser.add_argument("--note-slug", default=None, help="note to open (default: from Atlas)")
    parser.add_argument("--report-slug", default=None, help="report to open (default: newest)")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    chat = parser.add_argument_group("chat target (each question is a paid model call)")
    chat.add_argument("--question", action="append", default=[], help="a question to ask")
    chat.add_argument("--questions-file", type=Path, default=None, help="one question per line")
    chat.add_argument(
        "--expect-nothing",
        action="append",
        default=[],
        help="a question whose right answer is the 'Nothing recorded' message",
    )
    chat.add_argument(
        "--chat-timeout",
        type=float,
        default=DEFAULT_CHAT_TIMEOUT_SECONDS,
        help="seconds to wait for one answer",
    )
    args = parser.parse_args(argv)

    questions = collect_questions(args.question, args.questions_file, args.expect_nothing)
    if args.target == "chat" and not questions:
        parser.error("chat needs --question, --questions-file or --expect-nothing")
    if args.target != "chat" and questions:
        parser.error("--question, --questions-file and --expect-nothing are for the chat target")

    result, path = run_verify(
        args.target,
        base_url=args.base_url,
        org_key=resolve_key(args.org_key_file, DEFAULT_ORG_KEY_FILE, ORG_KEY_ENV),
        mcp_key=resolve_key(args.mcp_key_file, DEFAULT_MCP_KEY_FILE, MCP_KEY_ENV),
        out_dir=args.out_dir,
        editor=args.editor,
        timeout=args.timeout,
        query=args.query,
        note_slug=args.note_slug,
        report_slug=args.report_slug,
        questions=questions,
        chat_timeout=args.chat_timeout,
    )
    print(summary_line(result, path))
    for record in result.get("chat", []):
        print(chat_line(record))
    for check in result["checks"]:
        if not check["ok"] and check.get("hint"):
            print(f"hint: {check['target']}: {check['name']}: {check['hint']}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
