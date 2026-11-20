"""Agentic retrieval bench: a model answers a question through Portolan's MCP
door, iterating (decompose, search, read, refine), and its cited answer goes
through Portolan's citation and number gates. Compared, on the same
questions and model, with single-shot retrieval (`run_chat_path`, today's
chat path; `run_single_top10`, one search whose top 10 notes go to the
model whole). Write-up: `01-spec/bench-agentic-retrieval-2026-09-23.md`.

The agent sees the door exactly as an MCP client does: `create_server`'s own
tool descriptions and input schemas, and each result as the door returns it
(sanitized, with paging cursors). The tools run in-process through
`claude_agent_sdk.create_sdk_mcp_server`, the pattern
`portolan.report.researcher` uses. `DoorSession` adds what the bench needs
around the door: a per-question tool-call budget, a ledger of every call and
every note slug a result named, and `submit_answer`, whose input is gated
before the agent is told what was kept.

The gate (`gate_answer`) is the chat path's rules plus the ERB answer
sample's quote rule: a citation counts only if its note was returned to the
model in this session, its claim id (if any) belongs to that note, and its
quote is in that note's stored text (`quote_is_supported`); a sentence with no
valid citation is dropped, and so is one stating a number that its quotes,
cited claims and cited notes' syntheses do not state
(`unsupported_cited_numerals`, the chat path's numbers guard).
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import psycopg

from portolan.extract.quote_gate import quote_is_supported
from portolan.notes.render import parse_synthesis
from portolan.report.numbers import unsupported_cited_numerals

__all__ = [
    "AGENT_TOOL_NAMES",
    "AgentRun",
    "BenchQuestion",
    "DoorResult",
    "DoorSession",
    "GatedAnswer",
    "ModeRun",
    "NoteEvidence",
    "ToolCallRecord",
    "WEB_MODES",
    "agent_system_prompt",
    "bench_cost",
    "door_caller",
    "door_tool_specs",
    "gate_answer",
    "load_note_evidence",
    "recall_pct",
    "run_agent",
    "run_chat_path",
    "run_single_top10",
    "run_web_chat",
    "slugs_in_result",
]

# The six read tools the agent may call; `add_standing_question` writes and
# `coverage` has no ledger in either bench graph, so neither is offered.
AGENT_TOOL_NAMES = (
    "search",
    "open_note",
    "list_claims",
    "open_source",
    "graph_neighbors",
    "precedents",
)
SUBMIT_TOOL = "submit_answer"
MCP_SERVER_NAME = "portolan"

DEFAULT_MAX_TOOL_CALLS = 8
DEFAULT_MAX_SUBMITS = 2
DEFAULT_WALL_SECONDS = 240.0
# A backstop on model turns; the tool budget is the real cap.
DEFAULT_MAX_TURNS = 24
# Claude Code keeps a tool result inline up to this size instead of writing it to a
# file the agent (which has no Read tool here) could not open.
MAX_RESULT_SIZE_CHARS = 200_000

SINGLE_TOP_K = 10
CHAT_K = 8
CHAT_EXPAND = 1

CITATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "note_slug": {"type": "string"},
        "quote": {"type": "string"},
        "claim_id": {"type": ["integer", "null"]},
    },
    "required": ["note_slug", "quote"],
}
ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "sentences": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "citations": {"type": "array", "items": CITATION_SCHEMA},
                },
                "required": ["text", "citations"],
            },
        }
    },
    "required": ["sentences"],
}

ANSWER_CONTRACT = (
    "Each sentence carries at least one citation: the note_slug it comes from and a "
    "quote copied character for character from that note (its body, one of its claims' "
    "quote or claim_text, or a search hit's best_chunk); add the claim_id when the quote "
    "comes from a claim. A sentence without a verbatim supporting quote is deleted, and "
    "so is a sentence stating a number that its quotes and cited claims do not state. "
    "Be concise and include only what answers the question. If the records do not "
    "answer it, write one sentence saying so, with no citation."
)

SUBMIT_DESCRIPTION = (
    "Submit your final answer. `sentences` is a list of {text, citations}; each citation "
    "is {note_slug, quote, claim_id (optional)}. "
    + ANSWER_CONTRACT
    + " Returns which sentences were kept and, for any dropped, why; you may call it once "
    "more with the whole corrected answer (a new submission replaces the last), otherwise "
    "the kept sentences are your answer."
)

AGENT_PROMPT_TEMPLATE = """You answer one question from someone at an organization, using \
Portolan: the organization's knowledge graph of its customer and internal records, reached \
only through the tools you have.

What the Portolan server says about itself:
{door_instructions}

How to work:
- Search finds candidate notes; a hit shows one passage of a note. If the question has \
several parts, or needs facts from more than one record, search for each part, and reword a \
search when its results miss.
- Land on the best note: open it before you rely on it.
- Then hop from it to gather the evidence the question needs: the notes it links to and \
the notes that link to it (graph_neighbors, then open_note on the ones that matter), its \
further claims (list_claims), the document a claim came from (open_source), and what followed \
a theme before (precedents). A signal note links to its account and products; an account \
links to its parent, contacts and signals.
- You have at most {max_calls} research tool calls (submit_answer does not count). Stop as \
soon as you can answer.

When you are done, call submit_answer. {contract} After submit_answer accepts your answer, \
reply with the single word DONE."""

SINGLE_SYSTEM_PROMPT = (
    "You answer one question from someone at an organization using ONLY the notes "
    "supplied below, which a search of the organization's records returned; many may be "
    "irrelevant. Never invent a fact, name, date or number that is not in them. " + ANSWER_CONTRACT
)


@dataclass(frozen=True)
class BenchQuestion:
    """One question with ERB-shaped gold: `answer_facts` feed the completeness
    judge, `expected` the gold-document recall."""

    qid: str
    question: str
    gold_answer: str
    answer_facts: tuple[str, ...]
    expected: tuple[str, ...]
    kind: str = ""

    def as_judge_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "gold_answer": self.gold_answer,
            "answer_facts": list(self.answer_facts),
        }


def recall_pct(slugs: Iterable[str], expected: Sequence[str]) -> float:
    """Share of `expected` present in `slugs`, as a percentage."""
    gold = set(expected)
    if not gold:
        return 100.0
    return round(len(gold & set(slugs)) / len(gold) * 100, 2)


# ---------------------------------------------------------------------
# Evidence and the gate
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class NoteEvidence:
    """A note as stored: what a quote is checked against (`text`: title and
    body; each claim's `claim_text` and `quote`) and what the numbers guard
    allows at note level (`synthesis`, as the chat path picks it)."""

    slug: str
    text: str
    synthesis: str
    claims: Mapping[int, tuple[str, str | None]]


def load_note_evidence(conn: psycopg.Connection, slugs: Iterable[str]) -> dict[str, NoteEvidence]:
    """Stored text for each slug that exists (raw, never the door's sanitized
    copy), with every claim under it, archived ones included: the door shows
    those too."""
    wanted = sorted(set(slugs))
    if not wanted:
        return {}
    with conn.cursor() as cur:
        cur.execute(
            "SELECT slug, title, description, body_md FROM notes WHERE slug = ANY(%s)",
            (wanted,),
        )
        notes = cur.fetchall()
        cur.execute(
            "SELECT id, note_slug, claim_text, quote FROM claims WHERE note_slug = ANY(%s)",
            (wanted,),
        )
        claim_rows = cur.fetchall()
    claims: dict[str, dict[int, tuple[str, str | None]]] = {}
    for claim_id, note_slug, claim_text, quote in claim_rows:
        claims.setdefault(note_slug, {})[int(claim_id)] = (claim_text or "", quote)
    out: dict[str, NoteEvidence] = {}
    for slug, title, description, body_md in notes:
        body = body_md or ""
        synthesis = parse_synthesis(body) or description or title or ""
        out[slug] = NoteEvidence(
            slug=slug,
            text=f"{title or ''}\n\n{body}",
            synthesis=synthesis,
            claims=claims.get(slug, {}),
        )
    return out


def _as_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


@dataclass
class GatedAnswer:
    kept: list[dict[str, Any]]
    dropped: list[dict[str, Any]]
    quotes_total: int = 0
    quotes_supported: int = 0
    numerals_dropped: int = 0

    @property
    def text(self) -> str:
        """The kept sentences, citations removed, each ending in a stop: the
        text the judges read."""
        parts = []
        for s in self.kept:
            t = s["text"].strip()
            parts.append(t if t.endswith((".", "!", "?")) else t + ".")
        return " ".join(parts).strip()

    @property
    def cited_slugs(self) -> list[str]:
        seen: dict[str, None] = {}
        for s in self.kept:
            for c in s["citations"]:
                seen.setdefault(c["note_slug"], None)
        return list(seen)

    def summary(self) -> dict[str, int]:
        reasons: dict[str, int] = {}
        for d in self.dropped:
            reasons[d["reason"]] = reasons.get(d["reason"], 0) + 1
        return {
            "sentences": len(self.kept) + len(self.dropped),
            "kept": len(self.kept),
            "dropped": len(self.dropped),
            "quotes": self.quotes_total,
            "quotes_supported": self.quotes_supported,
            "numerals_dropped": self.numerals_dropped,
            **{f"dropped_{k}": v for k, v in sorted(reasons.items())},
        }


def gate_answer(
    sentences: Sequence[Mapping[str, Any]],
    evidence: Mapping[str, NoteEvidence],
    citable: Iterable[str],
) -> GatedAnswer:
    """Keep a sentence only if one of its citations is valid (see the module
    docstring) and every number it states is in what it validly cites.

    A dropped sentence's `reason` is `no_citation`, `note_not_returned` (the
    slug never came back from a tool, or does not exist), `claim_not_in_note`,
    `quote_not_supported`, or `numerals`; with several failing citations the
    first one's reason is reported."""
    allowed = set(citable)
    out = GatedAnswer(kept=[], dropped=[])
    for index, raw in enumerate(sentences):
        if not isinstance(raw, Mapping):
            out.dropped.append({"index": index, "text": str(raw), "reason": "no_citation"})
            continue
        text = str(raw.get("text") or "").strip()
        if not text:
            continue
        valid: list[dict[str, Any]] = []
        failure: str | None = None
        citations = raw.get("citations") or []
        for c in citations:
            if not isinstance(c, Mapping):
                failure = failure or "no_citation"
                continue
            slug = str(c.get("note_slug") or "")
            quote = str(c.get("quote") or "")
            claim_id = _as_int(c.get("claim_id"))
            if quote.strip():
                out.quotes_total += 1
            note = evidence.get(slug)
            if note is None or slug not in allowed:
                failure = failure or "note_not_returned"
                continue
            if claim_id is not None and claim_id not in note.claims:
                failure = failure or "claim_not_in_note"
                continue
            texts = [note.text]
            if claim_id is not None:
                claim_text, claim_quote = note.claims[claim_id]
                texts = [claim_text, claim_quote or "", note.text]
            if not any(quote_is_supported(quote, t) for t in texts if t):
                failure = failure or "quote_not_supported"
                continue
            out.quotes_supported += 1
            valid.append({"note_slug": slug, "claim_id": claim_id, "quote": quote})
        if not valid:
            out.dropped.append({"index": index, "text": text, "reason": failure or "no_citation"})
            continue
        cited_texts = [c["quote"] for c in valid]
        for c in valid:
            note = evidence[c["note_slug"]]
            if c["claim_id"] is not None:
                cited_texts.extend(t for t in note.claims[c["claim_id"]] if t)
            cited_texts.append(note.synthesis)
        if unsupported_cited_numerals(text, cited_texts):
            out.numerals_dropped += 1
            out.dropped.append({"index": index, "text": text, "reason": "numerals"})
            continue
        out.kept.append({"index": index, "text": text, "citations": valid})
    return out


# ---------------------------------------------------------------------
# The ledger: every tool call and every slug a result named
# ---------------------------------------------------------------------


@dataclass
class ToolCallRecord:
    tool: str
    arguments: dict[str, Any]
    seconds: float
    result_chars: int
    is_error: bool = False
    over_budget: bool = False
    returned: list[str] = field(default_factory=list)
    detail: str = ""
    # "search"; "land" (open_note on a note a search returned); "hop" (any other read:
    # a note reached through links, neighbours, a source or precedents); "submit".
    kind: str = ""


def slugs_in_result(tool: str, payload: Any) -> tuple[list[str], list[str], list[str]]:
    """`(seen, citable, opened)` note slugs in one door result.

    `seen`: every note slug the result names (search hits; a note opened and
    the notes its links point to; neighbours; notes extracted from a
    source). `citable`: notes whose text the result carried (search hits'
    passages, an opened note, a claims page). `opened`: notes opened in full."""
    seen: list[str] = []
    citable: list[str] = []
    opened: list[str] = []
    if tool == "search" and isinstance(payload, Mapping):
        for hit in payload.get("hits") or []:
            slug = hit.get("note_slug") if isinstance(hit, Mapping) else None
            if slug:
                seen.append(slug)
                citable.append(slug)
    elif tool in {"open_note", "list_claims"} and isinstance(payload, Mapping):
        slug = payload.get("slug")
        if slug:
            seen.append(slug)
            citable.append(slug)
            if tool == "open_note":
                opened.append(slug)
                # Provenance links point at sources, not notes.
                seen.extend(
                    link["to_slug"]
                    for link in payload.get("links") or []
                    if isinstance(link, Mapping)
                    and link.get("to_slug")
                    and link.get("kind") != "provenance"
                )
    elif tool == "graph_neighbors" and isinstance(payload, list):
        seen.extend(n["slug"] for n in payload if isinstance(n, Mapping) and n.get("slug"))
    elif tool == "open_source" and isinstance(payload, Mapping):
        seen.extend(s for s in payload.get("notes_extracted") or [] if isinstance(s, str))
    return seen, citable, opened


# ---------------------------------------------------------------------
# The door, in process, with a budget
# ---------------------------------------------------------------------


def _content(text: str | Sequence[str], *, is_error: bool = False) -> dict[str, Any]:
    blocks = [text] if isinstance(text, str) else list(text)
    out: dict[str, Any] = {"content": [{"type": "text", "text": t} for t in blocks]}
    if is_error:
        out["is_error"] = True
    return out


@dataclass(frozen=True)
class DoorResult:
    """One door result as an MCP client receives it: its text `blocks` (a list
    result is one block per item; `None` is no block at all), whether it is an
    error, and the parsed `payload` (the door's structured content) for the
    ledger."""

    blocks: list[str]
    is_error: bool
    payload: Any = None


class DoorSession:
    """One question's view of the door: the six read tools, each result the
    door's own, a budget of `max_tool_calls` research calls (a call past it
    is refused with a message, not run), and `submit_answer`.

    `call_door(name, arguments)` returns a `DoorResult`; `evidence_for(slugs)`
    loads stored text for the gate. Both are injected so tests need neither
    a database nor the MCP server.
    """

    def __init__(
        self,
        *,
        tool_specs: Sequence[Mapping[str, Any]],
        call_door: Callable[[str, dict[str, Any]], Any],
        evidence_for: Callable[[Iterable[str]], dict[str, NoteEvidence]],
        max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
        max_submits: int = DEFAULT_MAX_SUBMITS,
    ) -> None:
        self.tool_specs = [dict(s) for s in tool_specs if s["name"] in AGENT_TOOL_NAMES]
        self._call_door = call_door
        self._evidence_for = evidence_for
        self.max_tool_calls = max_tool_calls
        self.max_submits = max_submits
        self.calls: list[ToolCallRecord] = []
        self.seen: dict[str, int] = {}  # slug -> index of the call that first named it
        self.searched: set[str] = set()  # every slug any search returned
        self.citable: set[str] = set()
        self.opened: dict[str, None] = {}
        self.submissions: list[dict[str, Any]] = []

    @property
    def research_calls(self) -> int:
        return sum(1 for c in self.calls if c.tool != SUBMIT_TOOL and not c.over_budget)

    @property
    def final(self) -> GatedAnswer | None:
        return self.submissions[-1]["gated"] if self.submissions else None

    async def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == SUBMIT_TOOL:
            return await self._submit(arguments)
        started = time.monotonic()
        if self.research_calls >= self.max_tool_calls:
            message = (
                f"Tool budget used up ({self.max_tool_calls} research calls). Nothing was "
                "run. Call submit_answer now with what you have."
            )
            self.calls.append(
                ToolCallRecord(name, dict(arguments), 0.0, 0, is_error=True, over_budget=True)
            )
            return _content(message, is_error=True)
        result = await self._call_door(name, dict(arguments))
        seconds = time.monotonic() - started
        if name == "search":
            kind = "search"
        elif name == "open_note" and arguments.get("slug") in self.searched:
            kind = "land"
        else:
            kind = "hop"
        chars = sum(len(b) for b in result.blocks)
        record = ToolCallRecord(
            name, dict(arguments), seconds, chars, is_error=result.is_error, kind=kind
        )
        if not result.is_error:
            seen, citable, opened = slugs_in_result(name, result.payload)
            index = len(self.calls)
            for slug in seen:
                self.seen.setdefault(slug, index)
            if name == "search":
                self.searched.update(seen)
            self.citable.update(citable)
            for slug in opened:
                self.opened.setdefault(slug, None)
            record.returned = list(dict.fromkeys(seen))
            if not result.blocks:
                record.detail = "empty result (no content blocks)"
        else:
            record.detail = " ".join(result.blocks)[:300]
        self.calls.append(record)
        return _content(result.blocks, is_error=result.is_error)

    async def _submit(self, arguments: dict[str, Any]) -> dict[str, Any]:
        started = time.monotonic()
        if len(self.submissions) >= self.max_submits:
            self.calls.append(
                ToolCallRecord(
                    SUBMIT_TOOL, {}, 0.0, 0, is_error=True, over_budget=True, kind="submit"
                )
            )
            return _content("No more submissions: your last one stands. Reply DONE.", is_error=True)
        sentences = arguments.get("sentences")
        if not isinstance(sentences, list):
            sentences = []
        cited = {
            c.get("note_slug")
            for s in sentences
            if isinstance(s, Mapping)
            for c in (s.get("citations") or [])
            if isinstance(c, Mapping) and isinstance(c.get("note_slug"), str)
        }
        evidence = await asyncio.to_thread(self._evidence_for, cited & self.citable)
        gated = gate_answer(sentences, evidence, self.citable)
        self.submissions.append({"sentences": sentences, "gated": gated})
        summary = gated.summary()
        self.calls.append(
            ToolCallRecord(
                SUBMIT_TOOL,
                {"sentences": len(sentences)},
                time.monotonic() - started,
                0,
                detail=json.dumps(summary),
                kind="submit",
            )
        )
        if not gated.dropped:
            feedback: dict[str, Any] = {"status": "accepted", "kept": summary["kept"]}
        else:
            feedback = {
                "status": "partly_accepted" if gated.kept else "nothing_kept",
                "kept": summary["kept"],
                "dropped": [{"sentence": d["index"], "reason": d["reason"]} for d in gated.dropped],
                "reasons": {
                    "note_not_returned": "cite only notes a tool returned to you in this session",
                    "quote_not_supported": "the quote is not verbatim in that note",
                    "claim_not_in_note": "that claim_id belongs to a different note",
                    "numerals": "a number in the sentence is not in what it cites",
                    "no_citation": "every sentence needs a citation",
                },
                "next": (
                    "Call submit_answer once more with the fixes, or reply DONE to keep the "
                    "sentences that passed. A new submission replaces the whole answer: "
                    "include every sentence you want kept, not only the fixed ones."
                    if len(self.submissions) < self.max_submits
                    else "No submissions left; reply DONE."
                ),
            }
        return _content(json.dumps(feedback))

    def sdk_server(self) -> Any:
        """The in-process MCP server the agent session connects to: the door's
        six read tools (their descriptions and input schemas verbatim) and
        `submit_answer`."""
        from claude_agent_sdk import ToolAnnotations, create_sdk_mcp_server, tool

        annotations = ToolAnnotations(readOnlyHint=True, maxResultSizeChars=MAX_RESULT_SIZE_CHARS)
        sdk_tools = []
        for spec in self.tool_specs:
            name = spec["name"]

            async def handler(args: dict[str, Any], _name: str = name) -> dict[str, Any]:
                return await self.call(_name, args)

            sdk_tools.append(
                tool(name, spec.get("description") or "", spec["input_schema"], annotations)(
                    handler
                )
            )

        async def submit(args: dict[str, Any]) -> dict[str, Any]:
            return await self.call(SUBMIT_TOOL, args)

        sdk_tools.append(tool(SUBMIT_TOOL, SUBMIT_DESCRIPTION, ANSWER_SCHEMA)(submit))
        return create_sdk_mcp_server(name=MCP_SERVER_NAME, tools=sdk_tools)

    @property
    def allowed_tools(self) -> list[str]:
        names = [s["name"] for s in self.tool_specs] + [SUBMIT_TOOL]
        return [f"mcp__{MCP_SERVER_NAME}__{n}" for n in names]


def door_tool_specs(door: Any) -> list[dict[str, Any]]:
    """The door's own tool list (`MCPServer.list_tools`) as plain dicts."""
    listed = asyncio.run(door.list_tools())
    return [
        {"name": t.name, "description": t.description or "", "input_schema": t.input_schema}
        for t in listed
    ]


def door_caller(door: Any) -> Callable[[str, dict[str, Any]], Any]:
    """`call_door` for `DoorSession` over a real door: the blocks a client
    receives, or, for a refused call, the text the MCP SDK sends a client
    (`MCPServer._handle_call_tool` turns any tool exception into an `isError`
    result carrying `str(exc)`)."""
    from mcp.server.mcpserver.exceptions import ToolError

    async def call(name: str, arguments: dict[str, Any]) -> DoorResult:
        try:
            result = await door.call_tool(name, arguments)
        except ToolError as exc:
            return DoorResult([str(exc)], True)
        except Exception as exc:  # noqa: BLE001 -- a client sees any failure as an error result
            return DoorResult([str(exc)], True)
        blocks = [getattr(block, "text", "") for block in (result.content or [])]
        payload = result.structured_content
        # A list or `None` return comes back wrapped as {"result": ...}.
        if isinstance(payload, dict) and set(payload) == {"result"}:
            payload = payload["result"]
        return DoorResult(blocks, bool(result.is_error), payload)

    return call


def agent_system_prompt(door_instructions: str, max_calls: int) -> str:
    return AGENT_PROMPT_TEMPLATE.format(
        door_instructions=door_instructions.strip(),
        max_calls=max_calls,
        contract=ANSWER_CONTRACT,
    )


# ---------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------


# List prices, dollars per million tokens (input, output), from the claude-api skill's
# model table (cached 2026-06-24), read 2026-09-23. `portolan.pricing` has no Opus 5.5
# row and lists Sonnet 5 at $3 / $15; the write-up flags that difference.
BENCH_PRICES: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
}
CACHE_READ_MULTIPLIER = 0.1
CACHE_WRITE_MULTIPLIER = 1.25  # five-minute cache writes


def bench_cost(model: str, usage: Mapping[str, Any]) -> float | None:
    """Dollars at list price for one run's usage (uncached input, cache reads
    and writes, output); `None` for a model not in `BENCH_PRICES`."""
    price = BENCH_PRICES.get(model)
    if price is None:
        return None
    inp, out = price
    total = (
        int(usage.get("input_tokens", 0) or 0) * inp
        + int(usage.get("cache_read_input_tokens", 0) or 0) * inp * CACHE_READ_MULTIPLIER
        + int(usage.get("cache_creation_input_tokens", 0) or 0) * inp * CACHE_WRITE_MULTIPLIER
        + int(usage.get("output_tokens", 0) or 0) * out
    )
    return round(total / 1e6, 6)


@dataclass
class ModeRun:
    """One question answered one way: the judged text, what retrieval put in
    front of the model, the gate's counts, time, tokens and cost."""

    mode: str
    qid: str
    model: str
    answer_text: str
    retrieved: list[str]
    in_context: list[str]
    cited: list[str]
    gate: dict[str, Any]
    wall_s: float
    usage: dict[str, int]
    cost_usd: float | None
    tool_calls: int = 0
    raw: Any = None
    error: str | None = None

    def as_row(self, question: BenchQuestion) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "question_id": self.qid,
            "model": self.model,
            "recall_retrieved_pct": recall_pct(self.retrieved, question.expected),
            "recall_in_context_pct": recall_pct(self.in_context, question.expected),
            "recall_cited_pct": recall_pct(self.cited, question.expected),
            "gate": self.gate,
            "wall_s": round(self.wall_s, 2),
            "usage": self.usage,
            "cost_usd": self.cost_usd,
            "tool_calls": self.tool_calls,
            "answered": bool(self.answer_text.strip()),
            "error": self.error,
        }


@dataclass
class AgentRun(ModeRun):
    calls: list[ToolCallRecord] = field(default_factory=list)
    submissions: int = 0
    first_gate: dict[str, Any] | None = None
    stop: str = ""
    turns: int = 0
    sdk_cost_usd: float | None = None
    # Time inside model API calls (the CLI's `duration_api_ms`), apart from tools
    # and the CLI's own start-up.
    api_s: float | None = None
    tool_s: float = 0.0
    searched: list[str] = field(default_factory=list)

    def as_row(self, question: BenchQuestion) -> dict[str, Any]:
        row = super().as_row(question)
        gold = set(question.expected)
        searched = set(self.searched)
        ran = [c for c in self.calls if not c.over_budget]
        row.update(
            {
                "submissions": self.submissions,
                "first_gate": self.first_gate,
                "stop": self.stop,
                "turns": self.turns,
                "sdk_cost_usd": self.sdk_cost_usd,
                "api_s": round(self.api_s, 2) if self.api_s is not None else None,
                "tool_s": round(self.tool_s, 2),
                "calls_by_tool": _count_by_tool(self.calls),
                "calls_by_kind": {
                    k: sum(1 for c in ran if c.kind == k) for k in ("search", "land", "hop")
                },
                "over_budget_calls": sum(1 for c in self.calls if c.over_budget),
                "error_calls": sum(1 for c in ran if c.is_error),
                "gold_first_seen_call": _first_gold_call(self.calls, question.expected),
                "recall_searched_pct": recall_pct(searched, question.expected),
                # Evidence hopping reached that no search in the session returned.
                "gold_via_hop_only": sorted((gold & set(self.retrieved)) - searched),
                "cited_via_hop_only": sorted(set(self.cited) - searched),
            }
        )
        return row


def _count_by_tool(calls: Sequence[ToolCallRecord]) -> dict[str, int]:
    out: dict[str, int] = {}
    for c in calls:
        if not c.over_budget:
            out[c.tool] = out.get(c.tool, 0) + 1
    return out


def _first_gold_call(calls: Sequence[ToolCallRecord], expected: Sequence[str]) -> int | None:
    """1-based index, among research calls, of the call whose result first named
    a gold note; `None` if none did."""
    n = 0
    gold = set(expected)
    for c in calls:
        if c.tool == SUBMIT_TOOL or c.over_budget:
            continue
        n += 1
        if gold & set(c.returned):
            return n
    return None


async def run_agent(
    question: BenchQuestion,
    *,
    session: DoorSession,
    model: str,
    system_prompt: str,
    wall_seconds: float = DEFAULT_WALL_SECONDS,
    max_turns: int = DEFAULT_MAX_TURNS,
    cli_path: str | None = None,
    seed_search: bool = False,
) -> AgentRun:
    """One agent session on `question`. Stops when the agent ends its turn,
    at `wall_seconds`, or at `max_turns`; the answer is its last submission,
    gated (none: an empty answer).

    `seed_search=True` runs the door's `search` on the question's own words
    before the model starts (one of the research calls) and puts the result in
    the first message: today's single-shot retrieval as step one, so the loop
    only adds to it (mode `agent_seeded`)."""
    from claude_agent_sdk import (
        AssistantMessage,
        ClaudeAgentOptions,
        ResultMessage,
        ToolUseBlock,
        query,
    )

    options = ClaudeAgentOptions(
        system_prompt=system_prompt,
        tools=[],  # no built-in tools: the door and submit_answer only
        disallowed_tools=["Bash", "Read", "Write", "Edit", "WebFetch", "WebSearch", "Task"],
        allowed_tools=session.allowed_tools,
        mcp_servers={MCP_SERVER_NAME: session.sdk_server()},
        strict_mcp_config=True,
        setting_sources=[],
        skills=[],
        permission_mode="bypassPermissions",
        model=model,
        max_turns=max_turns,
        cli_path=cli_path,
    )
    started = time.monotonic()
    result_message: Any = None
    tool_uses = 0
    stop = "ended"
    error: str | None = None
    prompt = f"Question: {question.question}"
    if seed_search:
        seeded = await session.call("search", {"query": question.question[:500]})
        blocks = "\n".join(b["text"] for b in seeded["content"])
        prompt += (
            "\n\nA search with the question's own words (one of your research calls) "
            f"returned:\n{blocks}"
        )
    agen = query(prompt=prompt, options=options)

    async def consume() -> None:
        nonlocal result_message, tool_uses
        async for message in agen:
            if isinstance(message, AssistantMessage):
                tool_uses += sum(1 for b in message.content if isinstance(b, ToolUseBlock))
            elif isinstance(message, ResultMessage):
                result_message = message
                break

    try:
        await asyncio.wait_for(consume(), timeout=wall_seconds)
    except TimeoutError:
        stop = "wall_time"
    except Exception as exc:  # noqa: BLE001 -- one failed session must not end the run
        stop = "error"
        error = f"{type(exc).__name__}: {str(exc)[:300]}"
    finally:
        try:
            await agen.aclose()
        except Exception:  # noqa: BLE001, S110 -- closing a broken session is best effort
            pass
    wall = time.monotonic() - started

    usage: dict[str, int] = {}
    turns = 0
    sdk_cost = None
    api_s = None
    if result_message is not None:
        raw_usage = result_message.usage or {}
        usage = {
            k: int(raw_usage.get(k, 0) or 0)
            for k in (
                "input_tokens",
                "output_tokens",
                "cache_read_input_tokens",
                "cache_creation_input_tokens",
            )
        }
        turns = int(result_message.num_turns or 0)
        sdk_cost = result_message.total_cost_usd
        api_s = (result_message.duration_api_ms or 0) / 1000
        if stop == "ended" and getattr(result_message, "subtype", "") == "error_max_turns":
            stop = "max_turns"
    if stop == "ended" and not session.submissions:
        stop = "no_submission"

    final = session.final
    first = session.submissions[0]["gated"].summary() if session.submissions else None
    return AgentRun(
        mode="agent_seeded" if seed_search else "agent",
        qid=question.qid,
        model=model,
        answer_text=final.text if final else "",
        retrieved=list(session.seen),
        in_context=list(session.citable),
        cited=final.cited_slugs if final else [],
        gate=final.summary() if final else {},
        wall_s=wall,
        usage=usage,
        cost_usd=bench_cost(model, usage) if usage else None,
        tool_calls=session.research_calls,
        raw={"submissions": [s["sentences"] for s in session.submissions]},
        error=error,
        calls=list(session.calls),
        submissions=len(session.submissions),
        first_gate=first,
        stop=stop,
        turns=turns,
        sdk_cost_usd=sdk_cost,
        api_s=api_s,
        tool_s=sum(c.seconds for c in session.calls),
        searched=sorted(session.searched),
    )


def _llm_usage(llm: Any, start_index: int) -> dict[str, int]:
    """Summed token counts of the usage events `llm` recorded since
    `start_index` (an `AgentSDKClient` with an `InMemorySink`, whose
    `input_tokens` is uncached + cache reads + cache writes), split the way
    `bench_cost` reads them."""
    events = list(getattr(llm.sink, "events", []))[start_index:]
    total = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
    }
    for e in events:
        read = int(e.cached_tokens or 0)
        write = int((e.extra or {}).get("cache_creation_input_tokens", 0) or 0)
        total["input_tokens"] += max(0, int(e.input_tokens or 0) - read - write)
        total["cache_read_input_tokens"] += read
        total["cache_creation_input_tokens"] += write
        total["output_tokens"] += int(e.output_tokens or 0)
    return total


def run_chat_path(
    conn: psycopg.Connection,
    llm: Any,
    embedder: Any,
    question: BenchQuestion,
    *,
    model: str,
) -> ModeRun:
    """Today's chat path, unchanged (`portolan.web.answer.answer_question`,
    k=8, expand=1, its own citation and numbers checks). The notes that fit
    its context budget are recomputed with the same `_context_block`."""
    from portolan.web.answer import (
        INDEX_BROKEN_MESSAGE,
        NOTHING_RECORDED_MESSAGE,
        _context_block,
        answer_question,
    )

    start = len(getattr(llm.sink, "events", []))
    started = time.monotonic()
    try:
        answer = answer_question(
            conn, llm, embedder, question.question, k=CHAT_K, expand=CHAT_EXPAND, model=model
        )
    except Exception as exc:  # noqa: BLE001 -- recorded as an unanswered question
        usage = _llm_usage(llm, start)
        return ModeRun(
            mode="chat",
            qid=question.qid,
            model=model,
            answer_text="",
            retrieved=[],
            in_context=[],
            cited=[],
            gate={},
            wall_s=time.monotonic() - started,
            usage=usage,
            cost_usd=bench_cost(model, usage),
            error=f"{type(exc).__name__}: {str(exc)[:300]}",
        )
    wall = time.monotonic() - started
    usage = _llm_usage(llm, start)
    retrieved = [h.note_slug for h in answer.retrieved]
    _block, _claims, note_claim_ids, _syn = _context_block(conn, answer.retrieved)
    no_answer = {NOTHING_RECORDED_MESSAGE, INDEX_BROKEN_MESSAGE}
    kept = [s for s in answer.sentences if s.citations and s.text not in no_answer]
    text = " ".join(s.text if s.text.endswith((".", "!", "?")) else s.text + "." for s in kept)
    cited = list(dict.fromkeys(c.note_slug for s in kept for c in s.citations))
    return ModeRun(
        mode="chat",
        qid=question.qid,
        model=model,
        answer_text=text,
        retrieved=retrieved,
        in_context=list(note_claim_ids),
        cited=cited,
        gate={
            "kept": len(kept),
            "numerals_dropped": answer.numerals_dropped,
            "note_level_citations": sum(1 for s in kept for c in s.citations if c.is_note_level),
            "passage_citations": sum(1 for s in kept for c in s.citations if c.is_passage),
            "claim_citations": sum(1 for s in kept for c in s.citations if c.claim_id is not None),
            "uncited_dropped": answer.uncited_dropped,
        },
        wall_s=wall,
        usage=usage,
        cost_usd=bench_cost(model, usage),
        raw=[s.model_dump() for s in answer.sentences],
    )


WEB_MODES = ("web_single", "web_agent")


def run_web_chat(
    conn: psycopg.Connection,
    llm: Any,
    embedder: Any,
    question: BenchQuestion,
    *,
    mode: str,
    model: str,
    limits: Any = None,
) -> ModeRun:
    """The live chat path as the web app runs it, over a metered
    `OpenRouterClient`: `web_single` is `answer_question` (k=8, expand=1),
    `web_agent` is `portolan.web.chat_agent` (fallback to single included).
    Cost is what OpenRouter reported for this question's calls, from
    `llm.sink` (a fresh `InMemorySink` per question)."""
    from portolan.web.answer import (
        INDEX_BROKEN_MESSAGE,
        NOTHING_RECORDED_MESSAGE,
        _context_block,
        answer_question,
    )
    from portolan.web.chat_agent import AgentLimits, answer_question_agent

    started = time.monotonic()
    error = None
    answer = None
    try:
        if mode == "web_agent":
            answer = answer_question_agent(
                conn,
                llm,
                embedder,
                question.question,
                model=model,
                fallback_model=model,
                limits=limits or AgentLimits(),
                tenant="bench",
            )
        else:
            answer = answer_question(
                conn, llm, embedder, question.question, k=CHAT_K, expand=CHAT_EXPAND, model=model
            )
    except Exception as exc:  # noqa: BLE001 -- recorded as an unanswered question
        error = f"{type(exc).__name__}: {str(exc)[:300]}"
    wall = time.monotonic() - started
    events = list(getattr(llm.sink, "events", []))
    cost = round(sum(e.cost_estimate or 0.0 for e in events), 6)
    usage = {
        "input_tokens": sum(e.input_tokens for e in events),
        "output_tokens": sum(e.output_tokens for e in events),
        "cache_read_input_tokens": sum(e.cached_tokens for e in events),
        "model_calls": len(events),
    }
    if answer is None:
        return ModeRun(
            mode=mode,
            qid=question.qid,
            model=model,
            answer_text="",
            retrieved=[],
            in_context=[],
            cited=[],
            gate={},
            wall_s=wall,
            usage=usage,
            cost_usd=cost,
            error=error,
        )
    no_answer = {NOTHING_RECORDED_MESSAGE, INDEX_BROKEN_MESSAGE}
    kept = [s for s in answer.sentences if s.citations and s.text not in no_answer]
    text = " ".join(s.text if s.text.endswith((".", "!", "?")) else s.text + "." for s in kept)
    agent = answer.usage.get("agent") or {}
    if agent:
        in_context = list(agent.get("context_slugs") or [])
    else:
        in_context = list(_context_block(conn, answer.retrieved)[2])
    return ModeRun(
        mode=mode,
        qid=question.qid,
        model=model,
        answer_text=text,
        retrieved=[h.note_slug for h in answer.retrieved],
        in_context=in_context,
        cited=list(dict.fromkeys(c.note_slug for s in kept for c in s.citations)),
        gate={
            "kept": len(kept),
            "numerals_dropped": answer.numerals_dropped,
            "uncited_dropped": answer.uncited_dropped,
            "passage_citations": sum(1 for s in kept for c in s.citations if c.is_passage),
            "claim_citations": sum(1 for s in kept for c in s.citations if c.claim_id is not None),
            "note_level_citations": sum(1 for s in kept for c in s.citations if c.is_note_level),
        },
        wall_s=wall,
        usage=usage,
        cost_usd=cost,
        tool_calls=int(agent.get("tool_calls") or 0),
        raw={
            "sentences": [s.model_dump() for s in answer.sentences],
            "agent": {k: v for k, v in agent.items() if k != "context_slugs"},
        },
        error=error,
    )


def _note_block(index: int, slug: str, conn: psycopg.Connection) -> str:
    with conn.cursor() as cur:
        cur.execute("SELECT title, body_md FROM notes WHERE slug = %s", (slug,))
        row = cur.fetchone()
        cur.execute(
            "SELECT id, claim_text, quote FROM claims WHERE note_slug = %s AND archived = false "
            "ORDER BY recorded, id",
            (slug,),
        )
        claims = cur.fetchall()
    title, body = row if row else ("", "")
    lines = [f"--- Note {index} (note_slug: {slug}) ---", f"Title: {title}", "", body or ""]
    if claims:
        lines += ["", "Claims:"]
        lines += [
            f"- [claim_id={cid}] {text}" + (f' (quote: "{quote}")' if quote else "")
            for cid, text, quote in claims
        ]
    return "\n".join(lines)


def run_single_top10(
    conn: psycopg.Connection,
    llm: Any,
    embedder: Any,
    question: BenchQuestion,
    *,
    model: str,
) -> ModeRun:
    """One search (`k=10`), the ten notes whole (title, body, claims) in one
    prompt, the agent's answer contract and the same gate."""
    from portolan.search.query import search

    start = len(getattr(llm.sink, "events", []))
    started = time.monotonic()
    hits = search(conn, embedder, question.question, k=SINGLE_TOP_K).hits
    slugs = [h.note_slug for h in hits]
    context = "\n\n".join(_note_block(i, s, conn) for i, s in enumerate(slugs, 1))
    user = f"## Notes\n{context}\n\n## Question\n{question.question}"
    error = None
    sentences: list[dict[str, Any]] = []
    try:
        result = llm.complete(
            system=SINGLE_SYSTEM_PROMPT,
            user=user,
            json_schema=ANSWER_SCHEMA,
            model=model,
            surface="chat",
        )
        sentences = list((result.data or {}).get("sentences") or [])
    except Exception as exc:  # noqa: BLE001 -- recorded as an unanswered question
        error = f"{type(exc).__name__}: {str(exc)[:300]}"
    gated = gate_answer(sentences, load_note_evidence(conn, slugs), slugs)
    wall = time.monotonic() - started
    usage = _llm_usage(llm, start)
    return ModeRun(
        mode="single",
        qid=question.qid,
        model=model,
        answer_text=gated.text,
        retrieved=slugs,
        in_context=slugs,
        cited=gated.cited_slugs,
        gate=gated.summary(),
        wall_s=wall,
        usage=usage,
        cost_usd=bench_cost(model, usage),
        raw=sentences,
        error=error,
    )
