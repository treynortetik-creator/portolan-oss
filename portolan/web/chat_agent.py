"""Chat's agent mode (`PORTOLAN_CHAT_MODE=agent`): a seeded tool loop that
wiki-hops, then today's citation gate. Measured in
01-spec/chat-agent-2026-09-25.md; the design follows the seeded loop of
01-spec/bench-agentic-retrieval-2026-09-23.md.

1. **Seed.** `search()` on the question's own words (k=8, expand=1, as
   single mode) and `_build_context` over the hits: the first message holds
   exactly what the single-shot answer would have seen.
2. **Loop.** The model may call `search` (with `near=` and `since=`),
   `open_note` (claims, passages, links and backlinks) and
   `graph_neighbors`, through the MCP door's own tool functions
   (`portolan.mcp_server.tools`, which also write their usual `mcp:` usage
   rows), and ends by calling `answer` with the single-shot answer's schema.
3. **Gate.** `portolan.web.answer.gate_answer`, the single-shot gate, over a
   `_ChatContext` holding only what a tool returned in this loop: the seed's
   notes, each search hit's passage, and each opened note's synthesis,
   claims and passages. A neighbour listing carries titles only and makes
   nothing citable.

Limits (`portolan.config.Settings`): `chat_agent_max_tool_calls` model tool
calls (the seed is not one), `chat_agent_max_usd` per question (no research
turn starts that could pass it; see `_run_loop`) and
`chat_agent_timeout_seconds`, checked between steps and passed to each model
request as its timeout. A running search is not interrupted.

Any error, the timeout, the dollar cap with no answer yet, or a client with
no `complete_with_tools` falls back to the single-shot answer over the seed
search (`answer_from_search`), with the reason logged and kept in
`Answer.usage["agent"]["fallback"]`.

`run_chat_agent` is a generator: it yields short progress lines
("Reading <note title>…") and returns the `Answer`
(`StopIteration.value`); `answer_question_agent` drains it.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Generator
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import psycopg

from portolan.context import load_company_context, render_company_context_block
from portolan.extract.quote_gate import quote_is_supported
from portolan.llm.base import LLMClient, ToolCall, ToolTurn
from portolan.llm.json_util import extract_json_object
from portolan.mcp_server.tools import tool_graph_neighbors, tool_open_note, tool_search
from portolan.notes.models import Claim, NoteStatus
from portolan.notes.render import parse_synthesis
from portolan.pricing import PRICING, UnknownModelError, estimate_cost
from portolan.report.numbers import unsupported_cited_numerals
from portolan.search.embed import Embedder
from portolan.search.query import SearchHit, search
from portolan.web.answer import (
    ANSWER_JSON_SCHEMA,
    Answer,
    _adds_nothing,
    _build_context,
    _ChatContext,
    _make_passage,
    _passage_lines,
    answer_from_search,
    gate_answer,
    link_titles,
    no_hits_answer,
    strip_citation_markers,
)

__all__ = [
    "ANSWER_TOOL",
    "AgentLimits",
    "answer_question_agent",
    "run_chat_agent",
]

_logger = logging.getLogger(__name__)

ANSWER_TOOL = "answer"
SEED_K = 8
SEED_EXPAND = 1
SEARCH_K = 8
MAX_NEIGHBORS_SHOWN = 40
MAX_NEIGHBOR_DEPTH = 2
# Per opened note: its claims, then its passages (the note's chunks, in order)
# up to this many characters. A longer note says how many passages were left out.
OPEN_NOTE_MAX_CLAIMS = 60
OPEN_NOTE_PASSAGE_CHARS = 10_000
# The next turn's cost is guessed as the last turn's times this (the context grows).
NEXT_TURN_GROWTH = 1.25
# Priced with this row when neither the provider nor `PRICING` knows the model.
FALLBACK_PRICE_KEY = "claude-sonnet-5"

AGENT_SYSTEM_PROMPT = """You are Portolan's chat surface -- 'a living chart of what your \
customers actually said.' You answer one question from the organization's knowledge graph, \
reached only through your tools. Never invent a fact, name, date or number the tools did not \
return.

How to work:
- The first message holds a search on the question's own words, with each note's synthesis, \
claims and best passage. If that answers the question, call `answer` right away.
- Otherwise gather what is missing. `search` again with other words; `near` searches around \
notes you already found and the notes linked to them; `since` keeps material dated on or \
after a day. `open_note` reads a note whole: its synthesis, claims, passages, the notes it \
links to and the notes that link to it. `graph_neighbors` lists the notes linked to or from \
one. A signal note links to its account and products; an account links to its parent, \
contacts and signals.
- You have at most {max_calls} tool calls besides `answer`. Stop as soon as you can answer.

Answer by calling `answer`. Every sentence lists the note_slugs it is drawn from and cites \
its basis: claim_ids (the [claim_id=N] markers) and/or passages, each a passage_id (the \
p-... value of a [passage_id=...] marker) with a short quote copied word for word from that \
passage. Only notes, claims and passages a tool returned in this conversation can be cited; \
a note seen only as a title in a neighbour list cannot. Prefer a claim when one states the fact. A \
sentence that states a number must cite a claim or a quote that states that number. A \
sentence with no real basis should not be written, and no sentence says what the notes lack: \
that goes in gaps. If the notes don't answer the question, return an empty sentences list. \
In gaps, name in a few words each (at most three) what the \
question needs that the notes do not say; leave gaps empty when the answer is complete."""

TOOL_SPECS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": (
                "Fused keyword and semantic search over the notes. Returns up to "
                f"{SEARCH_K} notes, each with its best-matching passage."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to look for."},
                    "near": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Note slugs: search only these notes and the notes one link away."
                        ),
                    },
                    "since": {
                        "type": "string",
                        "description": "YYYY-MM-DD: only material dated on or after this day.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_note",
            "description": (
                "Read one note: synthesis, claims, passages, the notes it links to and the "
                "notes that link to it."
            ),
            "parameters": {
                "type": "object",
                "properties": {"slug": {"type": "string"}},
                "required": ["slug"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "graph_neighbors",
            "description": (
                "List the notes linked to or from a note (titles only; open_note one to read it)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "slug": {"type": "string"},
                    "depth": {"type": "integer", "description": "1 or 2 hops (default 1)."},
                },
                "required": ["slug"],
            },
        },
    },
]
ANSWER_SPEC: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": ANSWER_TOOL,
        "description": "Submit the final, cited answer. Ends the conversation.",
        "parameters": ANSWER_JSON_SCHEMA,
    },
}
FORCE_ANSWER = {"type": "function", "function": {"name": ANSWER_TOOL}}
EMPTY_ANSWER_RETRY = (
    "Not accepted yet: you have not searched beyond the first results. Search once or twice "
    "with other words (names, products, dates from the question), then call answer again."
)
NOTHING_KEPT_RETRY = (
    "Not accepted yet: the citation check kept none of your sentences, so the reader would see "
    '"nothing recorded". Why each was refused:'
)
NOTHING_KEPT_HOW = (
    "Cite only [claim_id=N] and [passage_id=p-...] markers that a tool returned in this "
    "conversation, list the note slug each belongs to, and copy quotes word for word. If a note "
    "you drew on showed no claim or passage for a fact, open_note it or search for the notes "
    "behind it (a report or theme links to its signal notes). Then call answer again."
)
# Sentences listed in the refusal; the rest are summed up in one line.
NOTHING_KEPT_SHOWN = 6


@dataclass(frozen=True)
class AgentLimits:
    max_tool_calls: int = 6
    max_usd: float = 0.15
    timeout_seconds: float = 45.0

    @classmethod
    def from_settings(cls, settings: Any) -> AgentLimits:
        return cls(
            max_tool_calls=settings.chat_agent_max_tool_calls,
            max_usd=settings.chat_agent_max_usd,
            timeout_seconds=settings.chat_agent_timeout_seconds,
        )


class _Stop(Exception):
    """Ends the loop without an answer; the reason goes to the fallback."""


@dataclass
class _Loop:
    """One question's loop state: what is citable (`ctx`), what was spent,
    and the ledger that ends up in `Answer.usage["agent"]`."""

    ctx: _ChatContext
    tenant: str
    titles: dict[str, str] = field(default_factory=dict)
    searched: list[SearchHit] = field(default_factory=list)
    tool_calls: int = 0
    by_tool: dict[str, int] = field(default_factory=dict)
    tool_errors: int = 0
    model_calls: int = 0
    spent_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    # Answers sent back for another look (`_run_loop`), in order.
    retries: list[str] = field(default_factory=list)

    def add_turn(self, turn: ToolTurn) -> float:
        cost = turn.cost_usd
        if cost is None:
            cost = _estimate_turn_cost(turn)
        self.model_calls += 1
        self.spent_usd += cost
        self.input_tokens += turn.input_tokens
        self.output_tokens += turn.output_tokens
        self.cached_tokens += turn.cached_tokens
        return cost

    def ledger(self, *, stop: str, wall_s: float, fallback: str | None) -> dict[str, Any]:
        return {
            "tool_calls": self.tool_calls,
            "by_tool": dict(self.by_tool),
            "tool_errors": self.tool_errors,
            "model_calls": self.model_calls,
            "cost_usd": round(self.spent_usd, 6),
            "stop": stop,
            "wall_s": round(wall_s, 2),
            "fallback": fallback,
            "retries": list(self.retries),
            "context_slugs": sorted(self.ctx.note_claim_ids),
        }


def _estimate_turn_cost(turn: ToolTurn) -> float:
    """List-price cost of a turn the provider did not price, so the dollar cap
    still counts it; an unknown model is priced as `FALLBACK_PRICE_KEY`."""
    key = turn.model if turn.model in PRICING else FALLBACK_PRICE_KEY
    try:
        return estimate_cost(key, turn.input_tokens, turn.output_tokens, turn.cached_tokens)
    except UnknownModelError:
        return 0.0


# ---------------------------------------------------------------------
# Tools: the door's functions, rendered for the model; each registers what
# it showed in the loop's `_ChatContext`, the only thing the gate accepts.
# ---------------------------------------------------------------------


def _hit_from_door(raw: dict[str, Any]) -> SearchHit:
    return SearchHit(
        note_slug=raw["note_slug"],
        title=raw["title"],
        score=raw["score"],
        via=raw["via"],
        best_chunk=raw["best_chunk"],
        heading=raw["heading"],
        updated=date.fromisoformat(raw["updated"][:10]),
        source_slugs=raw["source_slugs"],
        expanded_from=raw["expanded_from"],
        content_date=date.fromisoformat(raw["content_date"][:10]) if raw["content_date"] else None,
        content_date_source=raw["content_date_source"],
    )


def _search_tool(
    conn: psycopg.Connection, embedder: Embedder, loop: _Loop, args: dict[str, Any]
) -> str:
    query = str(args.get("query") or "").strip()
    if not query:
        raise ValueError("search needs a query.")
    near = args.get("near") or None
    if near is not None and not (isinstance(near, list) and all(isinstance(s, str) for s in near)):
        raise ValueError("near must be a list of note slugs.")
    since_raw = args.get("since")
    since = None
    if since_raw:
        try:
            since = date.fromisoformat(str(since_raw)[:10])
        except ValueError as exc:
            raise ValueError("since must be a date, YYYY-MM-DD.") from exc
    result = tool_search(
        conn, embedder, tenant=loop.tenant, query=query, k=SEARCH_K, since=since, near=near
    )
    hits = [_hit_from_door(h) for h in result["hits"]]
    loop.searched.extend(hits)
    scope = "".join(
        [f", near {', '.join(near)}" if near else "", f", since {since}" if since else ""]
    )
    if not hits:
        return f'No notes matched "{query}"{scope}.'
    lines = [f'{len(hits)} notes matched "{query}"{scope}:']
    titles = link_titles(conn, [hit.best_chunk for hit in hits])
    for hit in hits:
        loop.titles[hit.note_slug] = hit.title
        loop.ctx.note_claim_ids.setdefault(hit.note_slug, set())
        when = hit.content_date or hit.updated
        lines += ["", f"### Note {hit.note_slug}", f"Title: {hit.title}", f"Date: {when:%Y-%m-%d}"]
        passage = _make_passage(
            hit.note_slug, hit.heading, hit.best_chunk, hit.source_slugs, titles
        )
        if passage is not None:
            loop.ctx.passages[passage.passage_id] = passage
            lines += _passage_lines(passage)
    return "\n".join(lines)


def _claim_from_door(slug: str, raw: dict[str, Any]) -> Claim:
    return Claim(
        id=raw["id"],
        note_slug=slug,
        fact_type=raw["fact_type"],
        claim_text=raw["claim_text"],
        source_slug=raw["source_slug"],
        locator=raw["locator"],
        quote=raw["quote"],
        speaker=raw["speaker"],
        valid_from=raw["valid_from"],
        valid_to=raw["valid_to"],
        recorded=raw["recorded"],
        superseded_at=raw["superseded_at"],
        severity=raw["severity"],
        sentiment=raw["sentiment"],
        confidence=raw["confidence"],
    )


def _open_note_tool(conn: psycopg.Connection, loop: _Loop, args: dict[str, Any]) -> str:
    slug = str(args.get("slug") or "").strip()
    note = tool_open_note(conn, tenant=loop.tenant, slug=slug, link_details=True) if slug else None
    if note is None:
        raise ValueError(f"No note {slug!r}. Use a slug exactly as a search or link shows it.")
    title = note["title"] or slug
    loop.titles[slug] = title
    synthesis = strip_citation_markers(
        parse_synthesis(note["body_md"] or "") or note["description"] or title
    )
    status = f"Status: {note['status']}"
    if note["status"] != NoteStatus.LIVING.value:
        if note["superseded_by"]:
            status += f" (superseded by {note['superseded_by']})"
        else:
            status += f" (as of {note['updated'][:10]})"
    lines = [
        f"### Note {slug}",
        f"Title: {title}",
        f"Type: {note['type']}",
        f"Updated: {note['updated'][:10]}",
        status,
        "",
        synthesis,
    ]

    live = [c for c in note["claims"] if not c.get("archived")]
    shown = [_claim_from_door(slug, c) for c in live[:OPEN_NOTE_MAX_CLAIMS]]
    lines += ["", f"Claims ({len(shown)} of {len(live)}):" if live else "Claims: none recorded"]
    for claim in shown:
        lines.append(
            f"- [claim_id={claim.id}] {claim.claim_text} "
            f"(source: {claim.source_slug} @ {claim.locator})"
        )

    sources = [link["to_slug"] for link in note["links"] if link["kind"] == "provenance"]
    with conn.cursor() as cur:
        cur.execute("SELECT heading, text FROM chunks WHERE note_slug = %s ORDER BY id", (slug,))
        chunks = cur.fetchall()
    titles = link_titles(conn, [synthesis, *(text for _, text in chunks)])
    passage_lines: list[str] = []
    used = 0
    shown_passages = 0
    new_passages = []
    for heading, text in chunks:
        passage = _make_passage(slug, heading, text, sources, titles)
        if passage is None or _adds_nothing(passage, title, synthesis, note["description"], titles):
            continue
        block = _passage_lines(passage)
        cost = len("\n".join(block))
        if used + cost > OPEN_NOTE_PASSAGE_CHARS:
            break
        used += cost
        shown_passages += 1
        new_passages.append(passage)
        passage_lines += block
    if passage_lines:
        lines += ["", f"Passages ({shown_passages} of {len(chunks)} chunks):", *passage_lines]
        if shown_passages < len(chunks):
            lines.append(f'(More text not shown: search with near=["{slug}"] to find a part.)')

    note_links = [link for link in note["links"] if link["kind"] != "provenance"]
    if note_links:
        lines += ["", "Links from this note:"]
        lines += [f"- {x['to_slug']} ({x['kind']}) {x['title'] or ''}".rstrip() for x in note_links]
    if note.get("linked_from"):
        lines += ["", "Linked from:"]
        lines += [
            f"- {x['from_slug']} ({x['kind']}) {x['title'] or ''}".rstrip()
            for x in note["linked_from"]
        ]
    for link in note_links:
        if link.get("title"):
            loop.titles.setdefault(link["to_slug"], link["title"])

    # Registered only now that the whole block is built: what was shown is citable.
    ids = loop.ctx.note_claim_ids.setdefault(slug, set())
    for claim in shown:
        loop.ctx.claim_index[claim.id] = claim
        ids.add(claim.id)
    loop.ctx.note_synthesis[slug] = synthesis
    for passage in new_passages:
        loop.ctx.passages[passage.passage_id] = passage
    return "\n".join(lines)


def _neighbors_tool(conn: psycopg.Connection, loop: _Loop, args: dict[str, Any]) -> str:
    slug = str(args.get("slug") or "").strip()
    try:
        depth = int(args.get("depth") or 1)
    except (TypeError, ValueError):
        depth = 1
    depth = max(1, min(depth, MAX_NEIGHBOR_DEPTH))
    rows = tool_graph_neighbors(conn, tenant=loop.tenant, slug=slug, depth=depth)
    if not rows:
        return f"No notes are linked to or from {slug!r}."
    lines = [f"Notes linked to or from {slug} (titles only; open_note to read one):"]
    for row in rows[:MAX_NEIGHBORS_SHOWN]:
        loop.titles.setdefault(row["slug"], row["title"])
        arrow = "->" if row["direction"] == "out" else "<-"
        lines.append(
            f"- {row['slug']} [{row['type']}] {row['title']} "
            f"({arrow} {row['kind']}, via {row['via']}, updated {row['updated'][:10]})"
        )
    if len(rows) > MAX_NEIGHBORS_SHOWN:
        lines.append(f"({len(rows) - MAX_NEIGHBORS_SHOWN} more not shown.)")
    return "\n".join(lines)


def _progress_line(conn: psycopg.Connection, loop: _Loop, call: ToolCall) -> str:
    args = call.arguments or {}
    if call.name == "search":
        query = " ".join(str(args.get("query") or "").split())
        query = query if len(query) <= 60 else query[:57].rstrip() + "…"
        return f"Searching for “{query}”…"
    slug = str(args.get("slug") or "")
    if slug not in loop.titles:
        with conn.cursor() as cur:
            cur.execute("SELECT title FROM notes WHERE slug = %s", (slug,))
            row = cur.fetchone()
        if row is not None:
            loop.titles[slug] = row[0]
    title = loop.titles.get(slug) or slug
    if call.name == "open_note":
        return f"Reading {title}…"
    return f"Following links from {title}…"


# ---------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------


HISTORY_TURNS = 3
HISTORY_ANSWER_CHARS = 1500
HISTORY_QUESTION_CHARS = 500


@dataclass(frozen=True)
class HistoryTurn:
    """One earlier exchange of the conversation: the question, the answer's
    text, and the slugs that answer cited."""

    question: str
    answer: str
    slugs: tuple[str, ...]


def history_from_messages(messages: list[Any]) -> list[HistoryTurn]:
    """Pairs each user message with the assistant message after it (a
    `conversations.Message` list, oldest first). A question with no answer
    is dropped."""
    turns: list[HistoryTurn] = []
    pending: str | None = None
    for m in messages:
        if m.role == "user":
            pending = m.content_md
        elif pending is not None:
            slugs: list[str] = []
            for sentence in m.citations:
                for c in sentence.get("citations", []) or []:
                    slug = c.get("note_slug")
                    if slug and slug not in slugs:
                        slugs.append(slug)
            turns.append(HistoryTurn(pending, m.content_md, tuple(slugs)))
            pending = None
    return turns


def _standalone_query(question: str, history: list[HistoryTurn]) -> str:
    """The seed search text: a follow-up ("which of those...") only means
    something next to the question before it, so that question comes first."""
    if not history:
        return question
    return f"{history[-1].question[:HISTORY_QUESTION_CHARS]} {question}"


def _history_block(history: list[HistoryTurn]) -> str:
    if not history:
        return ""
    lines = ["Earlier in this conversation (oldest first):"]
    for turn in history[-HISTORY_TURNS:]:
        answer = turn.answer
        if len(answer) > HISTORY_ANSWER_CHARS:
            answer = answer[:HISTORY_ANSWER_CHARS] + "..."
        lines.append(f"Q: {turn.question[:HISTORY_QUESTION_CHARS]}\nA: {answer}")
        if turn.slugs:
            lines.append("Notes that answer cited: " + ", ".join(turn.slugs))
    lines.append(
        "The question below may refer to these ('those', 'it', 'the first one'). "
        "Reopen the cited notes with open_note when the answer needs their detail."
    )
    return "\n".join(lines) + "\n\n"


def _system_prompt(conn: psycopg.Connection, max_calls: int) -> str:
    prompt = AGENT_SYSTEM_PROMPT.format(max_calls=max_calls)
    company_context = load_company_context(conn)
    if company_context:
        prompt += "\n\n" + render_company_context_block(company_context.text)
    return prompt


def _text_message(role: str, text: str, *, cache: bool) -> dict[str, Any]:
    """A message; `cache` marks it as a prompt-cache breakpoint, which
    OpenRouter passes to Anthropic (the system prompt and the seed are
    re-sent on every turn)."""
    if not cache:
        return {"role": role, "content": text}
    return {
        "role": role,
        "content": [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}],
    }


def _assistant_message(turn: ToolTurn) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": turn.text or "",
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.raw_arguments or "{}"},
            }
            for call in turn.tool_calls
        ],
    }


def _answer_arguments(raw: dict[str, Any]) -> dict[str, Any]:
    """The `answer` call's arguments with `sentences` and `gaps` as lists.
    Sonnet 5 sometimes sends `sentences` as a JSON string, at times with the
    rest of the object (`, "gaps": [...]`) inside it (3 of 20 ERB answers,
    01-spec/chat-agent-2026-09-25.md); the leading array is read. Anything
    else raises, so the question falls back instead of reading as "nothing
    recorded"."""
    out = dict(raw)
    for key in ("sentences", "gaps"):
        value = out.get(key, [])
        if isinstance(value, str):
            _logger.warning("chat agent: the answer's %s arrived as a JSON string", key)
            text = value.strip()
            try:
                value = json.JSONDecoder().raw_decode(text)[0] if text else []
            except json.JSONDecodeError as exc:
                raise ValueError(f"the answer's {key} is text, not a list") from exc
        if not isinstance(value, list):
            raise ValueError(f"the answer's {key} is not a list")
        out[key] = value
    return out


def _refusal(raw: dict[str, Any], ctx: _ChatContext) -> str:
    """Why `gate_answer` refuses one written sentence, by the gate's own
    checks (`portolan.web.answer._validate_sentences`, `_passage_citation`
    and the numbers guard), in words the model can act on."""
    listed = [s for s in raw.get("note_slugs") or [] if isinstance(s, str)]
    slugs = [s for s in listed if s in ctx.note_claim_ids]
    problems = [f"note {s} was not returned by a tool" for s in listed if s not in slugs]
    for claim_id in raw.get("claim_ids") or []:
        claim = ctx.claim_index.get(claim_id) if isinstance(claim_id, int) else None
        if claim is None:
            problems.append(f"claim_id {claim_id} was not shown to you")
        elif claim.note_slug not in slugs:
            problems.append(f"claim_id {claim_id} is on {claim.note_slug}, not a listed note")
    for item in raw.get("passages") or []:
        item = item if isinstance(item, dict) else {}
        passage_id = str(item.get("passage_id", ""))
        quote = str(item.get("quote", "") or "").strip()
        passage = ctx.passages.get(passage_id)
        if passage is None:
            problems.append(f"passage_id {passage_id!r} is not a passage id you were shown")
        elif passage.note_slug not in slugs:
            problems.append(f"{passage_id} is on {passage.note_slug}, not a listed note")
        elif not quote_is_supported(quote, passage.text):
            problems.append(f"the quote from {passage_id} is not in that passage word for word")
        elif unsupported_cited_numerals(quote, [passage.text]):
            problems.append(f"the quote from {passage_id} states a number that passage does not")
    if problems:
        return "; ".join(problems)
    if not (slugs or raw.get("claim_ids") or raw.get("passages")):
        return "it cites nothing"
    return "it states a number that nothing it cites states"


def _nothing_kept(data: dict[str, Any], ctx: _ChatContext, question: str) -> str | None:
    """The refusal to send back when `gate_answer` would keep none of the
    answer's written sentences; `None` when it keeps at least one."""
    gated = gate_answer(data, ctx, question=question, control_ok=True, retrieved=[], usage={})
    if any(sentence.citations for sentence in gated.sentences):
        return None
    written = [
        raw
        for raw in data.get("sentences") or []
        if isinstance(raw, dict) and str(raw.get("text", "")).strip()
    ]
    if not written:
        return None
    lines = [NOTHING_KEPT_RETRY]
    for n, raw in enumerate(written[:NOTHING_KEPT_SHOWN], 1):
        lines.append(f"- Sentence {n}: {_refusal(raw, ctx)}.")
    if len(written) > NOTHING_KEPT_SHOWN:
        lines.append(f"- The other {len(written) - NOTHING_KEPT_SHOWN} sentences: likewise.")
    lines.append(NOTHING_KEPT_HOW)
    return "\n".join(lines)


def _run_loop(
    conn: psycopg.Connection,
    llm: Any,
    embedder: Embedder,
    loop: _Loop,
    *,
    question: str,
    seed_text: str,
    history_text: str = "",
    model: str,
    limits: AgentLimits,
    deadline: float,
    clock: Callable[[], float],
) -> Generator[str, None, dict[str, Any]]:
    """Turns until the model calls `answer`; returns its arguments. Raises
    `_Stop` at the timeout, or at the dollar cap before an answer.

    Dollar cap: a turn that may call research tools starts only if the spend
    so far plus two turns at the growth estimate (one research, one answer)
    stays under `limits.max_usd`; otherwise the turn offers only `answer`.
    No turn starts once the spend has reached the cap."""
    cache = model.startswith("anthropic/")
    messages: list[dict[str, Any]] = [
        _text_message("system", _system_prompt(conn, limits.max_tool_calls), cache=cache),
        _text_message(
            "user",
            f"{history_text}Question: {question}\n\n{seed_text}",
            cache=cache,
        ),
    ]
    force_answer = limits.max_tool_calls <= 0
    retried_empty = False
    retried_nothing_kept = False
    last_cost = 0.0
    handlers: dict[str, Callable[[dict[str, Any]], str]] = {
        "search": lambda a: _search_tool(conn, embedder, loop, a),
        "open_note": lambda a: _open_note_tool(conn, loop, a),
        "graph_neighbors": lambda a: _neighbors_tool(conn, loop, a),
    }
    for _ in range(limits.max_tool_calls + 3):
        remaining = deadline - clock()
        if remaining <= 0:
            raise _Stop("timeout")
        if loop.spent_usd >= limits.max_usd:
            raise _Stop("dollar_cap")
        guess = last_cost * NEXT_TURN_GROWTH
        if loop.spent_usd + 2 * guess > limits.max_usd:
            force_answer = True
        answer_only = force_answer
        turn = llm.complete_with_tools(
            messages=messages,
            tools=[ANSWER_SPEC] if answer_only else [*TOOL_SPECS, ANSWER_SPEC],
            model=model,
            surface="chat",
            tool_choice=FORCE_ANSWER if answer_only else "auto",
            timeout=remaining,
        )
        last_cost = loop.add_turn(turn)

        answer_call = next((c for c in turn.tool_calls if c.name == ANSWER_TOOL), None)
        if answer_call is not None:
            if answer_call.arguments is None:
                raise ValueError("the answer tool's arguments did not parse as JSON")
            data = _answer_arguments(answer_call.arguments)
            if answer_only:
                return data
            if not data["sentences"]:
                if loop.tool_calls or retried_empty:
                    return data
                # An empty answer from the seed alone gets one more look before "nothing recorded".
                retried_empty = True
                feedback = EMPTY_ANSWER_RETRY
            else:
                refusals = None if retried_nothing_kept else _nothing_kept(data, loop.ctx, question)
                if refusals is None:
                    return data
                # Written sentences the gate would all refuse get one look with the reasons.
                retried_nothing_kept = True
                loop.retries.append("nothing_kept")
                feedback = refusals
            messages.append(_assistant_message(turn))
            for call in turn.tool_calls:
                result = feedback if call is answer_call else "Not run."
                messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
            continue
        if not turn.tool_calls:
            data = extract_json_object(turn.text or "")
            if isinstance(data, dict) and "sentences" in data:
                return _answer_arguments(data)
            if answer_only:
                raise _Stop("no_answer")
            messages.append({"role": "assistant", "content": turn.text or "(no reply)"})
            messages.append({"role": "user", "content": "Call the answer tool now."})
            force_answer = True
            continue

        messages.append(_assistant_message(turn))
        for call in turn.tool_calls:
            handler = handlers.get(call.name)
            if answer_only or loop.tool_calls >= limits.max_tool_calls:
                result = "No more research: nothing was run. Call answer now."
                force_answer = True
            elif handler is None or call.arguments is None:
                result = f"Error: no tool {call.name!r} with those arguments."
                loop.tool_errors += 1
            else:
                if clock() >= deadline:
                    raise _Stop("timeout")
                yield _progress_line(conn, loop, call)
                loop.tool_calls += 1
                loop.by_tool[call.name] = loop.by_tool.get(call.name, 0) + 1
                try:
                    result = handler(call.arguments)
                except ValueError as exc:
                    result = f"Error: {exc}"
                    loop.tool_errors += 1
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
        if loop.tool_calls >= limits.max_tool_calls:
            force_answer = True
    raise _Stop("max_turns")


def run_chat_agent(
    conn: psycopg.Connection,
    llm: LLMClient,
    embedder: Embedder,
    question: str,
    *,
    model: str,
    fallback_model: str | None,
    limits: AgentLimits,
    tenant: str,
    history: list[HistoryTurn] | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> Generator[str, None, Answer]:
    """Yields progress lines; returns the gated `Answer` (see the module
    docstring). A database error in the seed search propagates, as in single
    mode; every later failure falls back."""
    started = clock()
    deadline = started + limits.timeout_seconds
    yield "Searching the graph…"
    history = history or []
    result = search(
        conn, embedder, _standalone_query(question, history), k=SEED_K, expand=SEED_EXPAND
    )
    if not result.hits and not result.control_ok:
        return no_hits_answer(result)
    ctx = _build_context(conn, result.hits) if result.hits else _ChatContext(text="")
    loop = _Loop(ctx=ctx, tenant=tenant)
    loop.titles.update({hit.note_slug: hit.title for hit in result.hits})
    searched = "the previous question and this one" if history else "the question's own words"
    seed_text = (
        f"A search on {searched} returned these notes:\n\n" + ctx.text
        if ctx.text
        else f"A search on {searched} returned nothing."
    )

    fallback: str | None = None
    stop = "answered"
    data: dict[str, Any] = {}
    if not callable(getattr(llm, "complete_with_tools", None)):
        fallback = "client_has_no_tool_calls"
    else:
        try:
            data = yield from _run_loop(
                conn,
                llm,
                embedder,
                loop,
                question=question,
                seed_text=seed_text,
                history_text=_history_block(history),
                model=model,
                limits=limits,
                deadline=deadline,
                clock=clock,
            )
        except _Stop as stopped:
            fallback = stop = str(stopped)
        except Exception as exc:  # noqa: BLE001 -- any failure falls back, logged below
            fallback = stop = f"error: {type(exc).__name__}: {str(exc)[:200]}"

    if fallback is not None:
        _logger.warning(
            "chat agent fell back to single-shot (%s) after %d tool calls, $%.4f",
            fallback,
            loop.tool_calls,
            loop.spent_usd,
        )
        yield "Writing the answer from the first search…"
        answer = answer_from_search(conn, llm, question, result, model=fallback_model)
        answer.usage["chat_mode"] = "single"
        answer.usage["agent"] = loop.ledger(stop=stop, wall_s=clock() - started, fallback=fallback)
        return answer

    yield "Checking citations…"
    usage: dict[str, Any] = {
        "input_tokens": loop.input_tokens,
        "output_tokens": loop.output_tokens,
        "cached_tokens": loop.cached_tokens,
        "model": model,
        "chat_mode": "agent",
    }
    answer = gate_answer(
        data,
        loop.ctx,
        question=question,
        control_ok=result.control_ok,
        retrieved=_unique_hits([*result.hits, *loop.searched]),
        usage=usage,
    )
    answer.usage["agent"] = loop.ledger(stop=stop, wall_s=clock() - started, fallback=None)
    # What the model wrote before the gate, so an empty reply is told apart from a gated one.
    answer.usage["agent"]["answer_sentences"] = len(data.get("sentences") or [])
    answer.usage["agent"]["answer_gaps"] = len(data.get("gaps") or [])
    _logger.info(
        "chat agent answered: %d tool calls %s, %d model calls, $%.4f, %.1fs",
        loop.tool_calls,
        json.dumps(loop.by_tool, sort_keys=True),
        loop.model_calls,
        loop.spent_usd,
        clock() - started,
    )
    return answer


def _unique_hits(hits: list[SearchHit]) -> list[SearchHit]:
    seen: set[str] = set()
    out = []
    for hit in hits:
        if hit.note_slug not in seen:
            seen.add(hit.note_slug)
            out.append(hit)
    return out


def answer_question_agent(
    conn: psycopg.Connection,
    llm: LLMClient,
    embedder: Embedder,
    question: str,
    **kwargs: Any,
) -> Answer:
    """`run_chat_agent`, with its progress lines dropped."""
    steps = run_chat_agent(conn, llm, embedder, question, **kwargs)
    while True:
        try:
            next(steps)
        except StopIteration as done:
            return done.value
