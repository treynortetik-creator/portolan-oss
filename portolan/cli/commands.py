"""Every `portolan <command>`: an argparse registration, a handler that
returns a `Result`, and a plain-text renderer. A handler talks to the app
only through `ctx.client` (`portolan/client/`), so a URL target and a local
app behave the same."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from typing import Any
from urllib.parse import parse_qsl, quote, urlsplit

from portolan.cli.core import (
    EXIT_ERROR,
    EXIT_FAILED,
    EXIT_REFUSED,
    CliError,
    Ctx,
    Session,
    UsageError,
)
from portolan.cli.pages import (
    parse_actions,
    parse_note,
    parse_notes_index,
    parse_reports,
    parse_review,
    parse_search,
    parse_usage,
)
from portolan.client.chat import check_original
from portolan.client.http import AppFailed, AppRefused

MAX_INDEX_PAGES = 10


@dataclass
class Result:
    data: dict[str, Any]
    exit: int = 0
    message: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def _clip(text: str, limit: int = 240) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


# ---- new ---------------------------------------------------------------------


def cmd_new(ctx: Ctx, args: argparse.Namespace) -> Result:
    ctx.session = None
    target = ctx.target
    base_url = ctx.base_url
    session = Session.create(
        ctx.session_path,
        target=target,
        base_url=base_url if target == "local" else None,
        editor=ctx.editor,
    )
    ctx.session = session
    return Result(
        {
            "session": str(ctx.session_path),
            "target": target,
            "base_url": base_url,
            "editor": ctx.editor,
        }
    )


def render_new(d: dict[str, Any]) -> str:
    return f"session {d['session']}: target {d['target']} ({d['base_url']}), editor {d['editor']}"


# ---- ask ---------------------------------------------------------------------


def _note_slug(citation: dict[str, Any]) -> str:
    return citation["note_path"].removeprefix("/notes/")


def cmd_ask(ctx: Ctx, args: argparse.Namespace) -> Result:
    session = ctx.session
    stored = session.get("conversation_id") if session else None
    conversation: str | int = "new" if args.new_conversation or stored is None else stored
    raw = ctx.client.ask(args.question, conversation)
    citations = [
        {
            "note": _note_slug(c),
            "claim_id": c["claim_id"],
            "quote": c.get("quote") or c.get("passage_quote"),
            "original": c["original_href"],
            "quote_match": c.get("quote_match") or c.get("passage_quote_match"),
            "ok": c["ok"],
            "problems": c["problems"],
        }
        for c in raw["citations"]
    ]
    cited_notes = list(dict.fromkeys(c["note"] for c in citations))
    data = {
        "question": raw["question"],
        "outcome": raw["outcome"],
        "answer": raw["answer"],
        "citations": citations,
        "cited_notes": cited_notes,
        "broken_citations": sum(1 for c in citations if not c["ok"]),
        "withheld": raw["withheld"],
        "latency_ms": raw["latency_ms"],
        "cost_usd": None,
        "model": raw["model"],
        "conversation_id": raw["conversation_id"],
        "new_conversation": conversation == "new",
        "error": raw["error"],
    }
    if session is not None and raw["conversation_id"]:
        session.set_conversation(int(raw["conversation_id"]))
    if raw["outcome"] == "error":
        return Result(data, EXIT_ERROR, f"the app could not answer: {raw['error']}")
    return Result(data)


def render_ask(d: dict[str, Any]) -> str:
    lines = [f"Q: {d['question']}", f"A: {d['answer'] or '(no answer)'}", ""]
    cost = "not exposed" if d["cost_usd"] is None else f"${d['cost_usd']}"
    lines.append(
        f"outcome {d['outcome']} | citations {len(d['citations'])} "
        f"({d['broken_citations']} broken) | withheld {d['withheld']} | "
        f"{d['latency_ms']} ms | cost {cost} | conversation {d['conversation_id']}"
    )
    for i, c in enumerate(d["citations"], start=1):
        claim = f" claim {c['claim_id']}" if c["claim_id"] is not None else ""
        lines.append(f"  {i}. {c['note']}{claim}{'' if c['ok'] else '  BROKEN'}")
        if c["original"]:
            lines.append(f"     original: {c['original']}")
        if c["quote"]:
            lines.append(f'     quote: "{_clip(c["quote"])}" ({c["quote_match"] or "unchecked"})')
        lines.extend(f"     problem: {p}" for p in c["problems"])
    return "\n".join(lines)


# ---- search ------------------------------------------------------------------


def _mcp_payload(result: dict[str, Any]) -> Any:
    if result.get("structuredContent") is not None:
        payload = result["structuredContent"]
        if isinstance(payload, dict) and set(payload) == {"result"}:
            return payload["result"]
        return payload
    for block in result.get("content") or []:
        if block.get("type") == "text":
            try:
                return json.loads(block["text"])
            except ValueError:
                return block["text"]
    return None


def cmd_search(ctx: Ctx, args: argparse.Namespace) -> Result:
    if args.since:
        try:
            date.fromisoformat(args.since)
        except ValueError as exc:
            raise UsageError(f"--since wants YYYY-MM-DD, not {args.since!r}") from exc
    if args.k is not None and args.k < 1:
        raise UsageError("--k must be at least 1")
    if args.k is None and not args.since:
        page = ctx.client.get("/atlas", {"q": args.query})
        parsed = parse_search(page)
        hits = parsed["hits"]
        data: dict[str, Any] = {
            "query": args.query,
            "source": "atlas page (as the browser searches)",
            "k": len(hits),
            "count": parsed["count"] or len(hits),
            "hits": hits,
            "control_ok": not parsed["index_broken"],
        }
    else:
        arguments: dict[str, Any] = {"query": args.query, "k": args.k or 12}
        if args.since:
            arguments["since"] = args.since
        with ctx.client.mcp() as mcp:
            result = mcp.call_tool("search", arguments)
        payload = _mcp_payload(result) or {}
        hits = [
            {
                "slug": h["note_slug"],
                "title": h["title"],
                "updated": h.get("updated"),
                "snippet": _clip(h.get("best_chunk") or "", 200),
                "via": h.get("via"),
            }
            for h in payload.get("hits", [])
        ]
        data = {
            "query": args.query,
            "source": "MCP search tool",
            "k": arguments["k"],
            "since": args.since,
            "count": len(hits),
            "hits": hits,
            "control_ok": payload.get("control_ok"),
        }
    data["slugs"] = [h["slug"] for h in hits]
    data["top"] = hits[0]["slug"] if hits else None
    return Result(data)


def render_search(d: dict[str, Any]) -> str:
    lines = [f'{d["count"]} result(s) for "{d["query"]}" via {d["source"]}']
    if d["control_ok"] is False:
        lines.append("WARNING: the search index looks broken")
    for i, h in enumerate(d["hits"], start=1):
        lines.append(f"  {i}. {h['slug']}  {h['title']}  (updated {h['updated']})")
        if h["snippet"]:
            lines.append(f"     {_clip(h['snippet'], 160)}")
    return "\n".join(lines)


# ---- note --------------------------------------------------------------------


def cmd_note(ctx: Ctx, args: argparse.Namespace) -> Result:
    page = ctx.client.get(f"/notes/{quote(args.slug, safe='')}")
    return Result({"slug": args.slug, **parse_note(page)})


def render_note(d: dict[str, Any]) -> str:
    lines = [
        f"{d['title']}  [{d['type']}, {d['status']}]  {d['slug']}",
        f"updated {d['updated']}, created {d['created']}"
        + (f", owner {d['owner']}" if d.get("owner") else ""),
    ]
    if d["body"]:
        lines.append(_clip(d["body"], 400))
    lines.append(f"Claims ({len(d['claims'])}):")
    for c in d["claims"]:
        lines.append(f"  - {_clip(c['text'], 200)}")
        lines.append(f"    {c['source']} @ {c['locator']}  {c['original'] or ''}")
        if c["quote"]:
            lines.append(f'    "{_clip(c["quote"], 200)}"')
    lines.append(f"Links ({len(d['links'])}): " + ", ".join(link["slug"] for link in d["links"]))
    lines.append(
        f"Backlinks ({len(d['backlinks'])}): "
        + ", ".join(f"{b['slug']} ({b['kind']})" for b in d["backlinks"])
    )
    return "\n".join(lines)


# ---- open --------------------------------------------------------------------


def cmd_open(ctx: Ctx, args: argparse.Namespace) -> Result:
    link = args.link
    parts = urlsplit(link)
    if parts.scheme:
        if urlsplit(ctx.base_url).netloc != parts.netloc:
            raise UsageError(f"{link!r} is not on the target {ctx.base_url}")
        link = parts.path + (f"?{parts.query}" if parts.query else "")
    if not link.startswith("/originals/"):
        raise UsageError("open wants an original link like /originals/<slug>?at=<locator>")
    page = ctx.client.get(link)
    body = page.dom.find(attrs={"id": "original-body"})
    citation: dict[str, Any] = {"original_href": link, "quote": args.quote}
    problems: list[str] = []
    check_original(citation, page, problems)
    blocks = [n for n in body.descendants() if "data-locator" in n.attrs] if body else []
    quote_checked = args.quote is not None
    if quote_checked and citation.get("quote_match", "none") == "none" and not problems:
        problems.append("the quote is not on the page")
    data = {
        "href": link,
        "status": page.response.status_code,
        "document": body is not None,
        "pdf": body is None and page.dom.find("iframe") is not None,
        "blocks": len(blocks),
        "locator": dict(parse_qsl(parts.query or urlsplit(link).query)).get("at", ""),
        "block_found": citation.get("cited_block_found"),
        "quote": args.quote,
        "quote_match": citation.get("quote_match") if quote_checked else None,
        "quote_in_cited_block": citation.get("quote_in_cited_block") if quote_checked else None,
        "quote_check": citation.get("quote_check") if quote_checked else None,
        "problems": problems,
        "ok": not problems,
    }
    if problems:
        return Result(data, EXIT_FAILED, "; ".join(problems))
    return Result(data)


def render_open(d: dict[str, Any]) -> str:
    kind = "PDF original" if d["pdf"] else f"document with {d['blocks']} block(s)"
    lines = [f"{d['href']}: {d['status']}, {kind}"]
    if d["locator"]:
        lines.append(f"locator {d['locator']}: {'found' if d['block_found'] else 'NOT found'}")
    if d["quote"] is not None:
        lines.append(
            f'quote "{_clip(d["quote"], 120)}": {d["quote_match"] or d["quote_check"] or "none"}'
            + (" (in the cited block)" if d["quote_in_cited_block"] else "")
        )
    lines.extend(f"problem: {p}" for p in d["problems"])
    return "\n".join(lines)


# ---- accounts ------------------------------------------------------------------


def _account_rows(ctx: Ctx, q: str | None) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    total = 0
    pages = 1
    number = 1
    while number <= min(pages, MAX_INDEX_PAGES):
        params: dict[str, Any] = {"type": "account", "page": number}
        if q:
            params["q"] = q
        parsed = parse_notes_index(ctx.client.get("/notes", params))
        rows.extend(parsed["rows"])
        total, pages = parsed["total"], parsed["pages"]
        number += 1
    return rows, total


def cmd_accounts(ctx: Ctx, args: argparse.Namespace) -> Result:
    rows, total = _account_rows(ctx, args.filter)
    return Result({"filter": args.filter, "total": total, "accounts": rows})


def render_accounts(d: dict[str, Any]) -> str:
    lines = [f"{d['total']} account(s), {len(d['accounts'])} listed"]
    lines.extend(
        f"  {a['slug']}  {a['title']}  ({a['status']}, updated {a['updated']})"
        for a in d["accounts"]
    )
    return "\n".join(lines)


def cmd_account(ctx: Ctx, args: argparse.Namespace) -> Result:
    wanted = args.name.strip()
    rows, _ = _account_rows(ctx, wanted)
    fold = wanted.casefold()
    exact = [r for r in rows if r["slug"].casefold() == fold] or [
        r for r in rows if r["title"].casefold() == fold
    ]
    matches = exact or rows
    if not matches:
        raise CliError(f"no account matches {wanted!r}", EXIT_FAILED)
    if len(matches) > 1:
        names = ", ".join(f"{r['slug']} ({r['title']})" for r in matches[:10])
        raise CliError(f"{len(matches)} accounts match {wanted!r}: {names}", EXIT_FAILED)
    slug = matches[0]["slug"]
    note = parse_note(ctx.client.get(f"/notes/{quote(slug, safe='')}"))
    actions = parse_actions(ctx.client.get("/actions"))
    mine = [a for a in actions["actions"] if a["account_slug"] == slug]
    return Result({"slug": slug, **note, "open_actions": mine})


def render_account(d: dict[str, Any]) -> str:
    lines = [render_note(d), f"Open actions ({len(d['open_actions'])}):"]
    lines.extend(
        f"  - {a['what']}  (owner {a['owner']}, due {a['due'] or 'none'}"
        f"{', OVERDUE' if a['overdue'] else ''})"
        for a in d["open_actions"]
    )
    return "\n".join(lines)


# ---- actions ---------------------------------------------------------------------


def cmd_actions(ctx: Ctx, args: argparse.Namespace) -> Result:
    parsed = parse_actions(ctx.client.get("/actions"))
    if args.owner:
        needle = args.owner.casefold()
        parsed["owners"] = [o for o in parsed["owners"] if needle in o["owner"].casefold()]
        parsed["actions"] = [a for o in parsed["owners"] for a in o["actions"]]
    parsed["owner_filter"] = args.owner
    return Result(parsed)


def render_actions(d: dict[str, Any]) -> str:
    lines = [f"{d['open']} open, {d['overdue']} overdue as of {d['as_of']}"]
    for group in d["owners"]:
        lines.append(f"{group['owner']} ({len(group['actions'])})")
        for a in group["actions"]:
            flag = "  OVERDUE" if a["overdue"] else ""
            lines.append(
                f"  #{a['id']} {_clip(a['what'], 100)}  [{a['account'] or '-'}] "
                f"due {a['due'] or '-'}{flag}"
            )
    if d["closed_this_week"]:
        lines.append(f"Closed this week: {len(d['closed_this_week'])}")
    return "\n".join(lines)


# ---- review ----------------------------------------------------------------------


def cmd_review_list(ctx: Ctx, args: argparse.Namespace) -> Result:
    params: dict[str, Any] = {"page": args.page}
    if args.kind:
        params["kind"] = args.kind
    parsed = parse_review(ctx.client.get("/review", params))
    parsed.update(kind=args.kind, page=args.page)
    return Result(parsed)


def render_review_list(d: dict[str, Any]) -> str:
    lines = [f"{d['open']} open; page {d['page']} of {d['pages']}, {len(d['cards'])} shown"]
    for c in d["cards"]:
        lines.append(
            f"  #{c['id']} [{c['category']}] {_clip(c['headline'], 120)}  (opened {c['opened']})"
        )
    return "\n".join(lines)


_DECISION_PATHS = {"accept": "resolve", "reject": "dismiss"}


def cmd_review_decide(ctx: Ctx, args: argparse.Namespace) -> Result:
    if not args.yes:
        raise UsageError(
            f"review decide writes to the app ({ctx.base_url}); "
            f"add --yes to {args.decision} row {args.id}"
        )
    endpoint = f"/review/{args.id}/{_DECISION_PATHS[args.decision]}"
    data = {"reason": args.reason} if args.reason and args.decision == "reject" else {}
    response = ctx.client.post(endpoint, data)
    location = response.headers.get("location", "")
    notice = dict(parse_qsl(urlsplit(location).query)).get("notice")
    return Result(
        {
            "id": args.id,
            "decision": args.decision,
            "endpoint": endpoint,
            "status": response.status_code,
            "location": location,
            "notice": notice,
            "already_handled": notice == "already_handled",
        }
    )


def render_review_decide(d: dict[str, Any]) -> str:
    note = f" (notice: {d['notice']})" if d["notice"] else ""
    return f"row {d['id']}: {d['decision']} -> {d['status']} {d['location']}{note}"


# ---- report -----------------------------------------------------------------------


def cmd_report_list(ctx: Ctx, args: argparse.Namespace) -> Result:
    reports = parse_reports(ctx.client.get("/reports"))
    return Result({"count": len(reports), "reports": reports})


def render_report_list(d: dict[str, Any]) -> str:
    lines = [f"{d['count']} report(s)"]
    lines.extend(f"  {r['slug']}  {r['title']}  (updated {r['updated']})" for r in d["reports"])
    return "\n".join(lines)


def cmd_report_show(ctx: Ctx, args: argparse.Namespace) -> Result:
    page = ctx.client.get(f"/reports/{quote(args.slug, safe='')}.md")
    return Result({"slug": args.slug, "chars": len(page.text), "markdown": page.text})


def render_report_show(d: dict[str, Any]) -> str:
    return d["markdown"]


def cmd_report_run(ctx: Ctx, args: argparse.Namespace) -> Result:
    """`python -m portolan.report weekly|monday ...`, run in this process's
    environment, so it reaches whatever database and model that environment
    names. Local only."""
    if ctx.target != "local":
        raise UsageError(f"report {args.report_command} runs a local module; use --target local")
    rest = [a for a in args.rest if a != "--"]
    command = [sys.executable, "-m", "portolan.report", args.report_command, *rest]
    completed = subprocess.run(command, check=False)  # noqa: S603 - our own module, argv list
    data = {
        "command": f"python -m portolan.report {args.report_command} {' '.join(rest)}".strip(),
        "exit_code": completed.returncode,
    }
    if completed.returncode != 0:
        return Result(data, EXIT_FAILED, f"the report module exited {completed.returncode}")
    return Result(data)


def render_report_run(d: dict[str, Any]) -> str:
    return f"{d['command']} exited {d['exit_code']}"


# ---- usage, health -------------------------------------------------------------------


def cmd_usage(ctx: Ctx, args: argparse.Namespace) -> Result:
    return Result(parse_usage(ctx.client.get("/admin/usage")))


def render_usage(d: dict[str, Any]) -> str:
    lines = ["Totals: " + ", ".join(f"{k} {v}" for k, v in d["totals"].items())]
    for w in d["weeks"]:
        lines.append(
            f"  week of {w.get('week_of')}: {w.get('questions')} question(s), "
            f"{w.get('active_users')} user(s)"
        )
    for u in d["unanswered"]:
        lines.append(f"  unanswered x{u.get('times_asked')}: {_clip(str(u.get('question')), 100)}")
    return "\n".join(lines)


def cmd_health(ctx: Ctx, args: argparse.Namespace) -> Result:
    started = time.perf_counter()
    page = ctx.client.get("/healthz", anonymous=True)
    latency = round((time.perf_counter() - started) * 1000, 1)
    try:
        body = page.json()
    except ValueError:
        body = None
    ok = body == {"status": "ok"}
    auth = "ok"
    refused = None
    org, _mcp = ctx.keys()
    if not org.value:
        auth = f"no key ({org.source})"
    else:
        try:
            ctx.client.login()
        except AppFailed as exc:
            auth = f"error: {exc}"
        except AppRefused as exc:
            auth = f"refused: {exc}"
            refused = str(exc)
    data = {
        "target": ctx.target,
        "base_url": ctx.base_url,
        "status": "ok" if ok else "unhealthy",
        "ok": ok,
        "latency_ms": latency,
        "auth": auth,
    }
    if not ok:
        return Result(data, EXIT_ERROR, f"/healthz answered {page.text[:120]!r}")
    if refused:
        return Result(data, EXIT_REFUSED, refused)
    return Result(data)


def render_health(d: dict[str, Any]) -> str:
    return f"{d['base_url']}: {d['status']} in {d['latency_ms']} ms; sign-in {d['auth']}"


# ---- mcp -----------------------------------------------------------------------------


def cmd_mcp_tools(ctx: Ctx, args: argparse.Namespace) -> Result:
    with ctx.client.mcp() as mcp:
        tools = mcp.list_tools()
    listed = [
        {
            "name": t["name"],
            "description": _clip((t.get("description") or "").split("\n\n")[0], 160),
            "arguments": sorted((t.get("inputSchema") or {}).get("properties", {})),
        }
        for t in tools
    ]
    return Result({"count": len(listed), "names": [t["name"] for t in listed], "tools": listed})


def render_mcp_tools(d: dict[str, Any]) -> str:
    lines = [f"{d['count']} tool(s)"]
    lines.extend(
        f"  {t['name']}({', '.join(t['arguments'])})  {t['description']}" for t in d["tools"]
    )
    return "\n".join(lines)


def cmd_mcp_call(ctx: Ctx, args: argparse.Namespace) -> Result:
    try:
        arguments = json.loads(args.arguments)
    except ValueError as exc:
        raise UsageError(f"mcp call wants a JSON object of arguments: {exc}") from exc
    if not isinstance(arguments, dict):
        raise UsageError('mcp call wants a JSON object of arguments, like \'{"query": "x"}\'')
    with ctx.client.mcp() as mcp:
        result = mcp.call_tool(args.tool, arguments)
    payload = _mcp_payload(result)
    data = {"tool": args.tool, "is_error": bool(result.get("isError")), "result": payload}
    if data["is_error"]:
        return Result(data, EXIT_FAILED, f"the tool reported an error: {_clip(str(payload))}")
    return Result(data)


def render_mcp_call(d: dict[str, Any]) -> str:
    body = d["result"] if isinstance(d["result"], str) else json.dumps(d["result"], indent=2)
    return f"{d['tool']}{' (error)' if d['is_error'] else ''}:\n{body}"


# ---- registration -------------------------------------------------------------------


@dataclass(frozen=True)
class Command:
    handler: Callable[[Ctx, argparse.Namespace], Result]
    render: Callable[[dict[str, Any]], str]


def register(sub: argparse._SubParsersAction, common: argparse.ArgumentParser) -> None:
    def add(name: str, handler, render, help_: str) -> argparse.ArgumentParser:
        parser = sub.add_parser(name, parents=[common], help=help_, description=help_)
        parser.set_defaults(command=Command(handler, render))
        return parser

    add("new", cmd_new, render_new, "start (or replace) the session file for this target")

    p = add("ask", cmd_ask, render_ask, "ask a question through the real chat path")
    p.add_argument("question")
    p.add_argument(
        "--new-conversation", action="store_true", help="do not continue the session's conversation"
    )

    p = add("search", cmd_search, render_search, "search the graph")
    p.add_argument("query")
    p.add_argument("--k", type=int, help="how many hits (uses the MCP search tool)")
    p.add_argument(
        "--since", help="YYYY-MM-DD, only notes with newer content (uses the MCP search tool)"
    )

    add("note", cmd_note, render_note, "a note's claims, links and backlinks").add_argument("slug")

    p = add("open", cmd_open, render_open, "open an /originals page and check a quote is on it")
    p.add_argument("link", help="/originals/<slug>?at=<locator>, or the same as a full URL")
    p.add_argument(
        "--quote", help="text that must be on the page (the production quote gate's rule)"
    )

    add("accounts", cmd_accounts, render_accounts, "list accounts").add_argument(
        "filter", nargs="?", help="title or slug contains"
    )
    add(
        "account", cmd_account, render_account, "one account: its note and open actions"
    ).add_argument("name", help="an account's slug or title")

    add("actions", cmd_actions, render_actions, "open actions by owner").add_argument(
        "--owner", help="only owners whose name contains this"
    )

    review = sub.add_parser("review", parents=[common], help="the review queue")
    review_sub = review.add_subparsers(dest="review_command", required=True)
    p = review_sub.add_parser("list", parents=[common], help="open review rows")
    p.set_defaults(command=Command(cmd_review_list, render_review_list))
    p.add_argument("--kind", help="a category label, as the page's filter")
    p.add_argument("--page", type=int, default=1)
    p = review_sub.add_parser(
        "decide", parents=[common], help="accept (confirm) or reject (dismiss) a row"
    )
    p.set_defaults(command=Command(cmd_review_decide, render_review_decide))
    p.add_argument("id", type=int)
    p.add_argument("decision", choices=sorted(_DECISION_PATHS))
    p.add_argument("--yes", action="store_true", help="required: this writes to the app")
    p.add_argument("--reason", help="stored with a reject")

    report = sub.add_parser("report", parents=[common], help="reports")
    report_sub = report.add_subparsers(dest="report_command", required=True)
    p = report_sub.add_parser("list", parents=[common], help="the weekly reports the app holds")
    p.set_defaults(command=Command(cmd_report_list, render_report_list))
    p = report_sub.add_parser("show", parents=[common], help="one report as markdown")
    p.set_defaults(command=Command(cmd_report_show, render_report_show))
    p.add_argument("slug")
    for name in ("weekly", "monday"):
        p = report_sub.add_parser(
            name, parents=[common], help=f"run `python -m portolan.report {name}` locally"
        )
        p.set_defaults(command=Command(cmd_report_run, render_report_run))
        p.add_argument("rest", nargs=argparse.REMAINDER, help=f"arguments for the {name} module")

    add("usage", cmd_usage, render_usage, "the /admin/usage numbers")
    add("health", cmd_health, render_health, "is the app up, and does the key sign in")

    mcp = sub.add_parser("mcp", parents=[common], help="the MCP endpoint")
    mcp_sub = mcp.add_subparsers(dest="mcp_command", required=True)
    p = mcp_sub.add_parser("tools", parents=[common], help="list the tools")
    p.set_defaults(command=Command(cmd_mcp_tools, render_mcp_tools))
    p = mcp_sub.add_parser("call", parents=[common], help="call a tool")
    p.set_defaults(command=Command(cmd_mcp_call, render_mcp_call))
    p.add_argument("tool")
    p.add_argument("arguments", nargs="?", default="{}", help="a JSON object")
