"""The chat stream as a browser reads it: server-sent events, the
sentence fragments, citations, and the checks that a citation resolves
(the note page holds the claim, the original page shows the cited block and
the claim's quote). Shared by `scripts/verify.py` and the `portolan` CLI."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Iterable, Iterator
from typing import Any
from urllib.parse import parse_qsl, urlsplit

import httpx

from portolan.client.htmltree import Node, Page, parse_html
from portolan.client.quotes import quote_match, row_fields, row_quote_match

# Openings of `portolan/web/answer.py` NOTHING_RECORDED_MESSAGE,
# GAPS_PREFIX (the can't-answer reply listing what the graph lacks) and
# INDEX_BROKEN_MESSAGE, pinned by `tests/test_verify_script.py`.
NOTHING_RECORDED_PREFIX = "Nothing recorded."
GAPS_PREFIX = "Not in the graph yet:"
INDEX_BROKEN_PREFIX = "The search index looks broken"
_CLAIM_FRAGMENT_RE = re.compile(r"^claim-(\d+)$")


def sse_events(chunks: Iterable[str]) -> Iterator[tuple[str, str]]:
    """`(event, data)` for each server-sent event in a text stream, split
    on blank lines the way `chat.html`'s `parseSseChunk` does."""

    def parse(block: str) -> tuple[str, str] | None:
        event, data = "message", []
        for line in block.split("\n"):
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data.append(line[5:].strip())
        return (event, "\n".join(data)) if data else None

    buffer = ""
    for chunk in chunks:
        buffer += chunk.replace("\r\n", "\n")
        *blocks, buffer = buffer.split("\n\n")
        for block in blocks:
            parsed = parse(block)
            if parsed is not None:
                yield parsed
    if buffer.strip():
        parsed = parse(buffer)
        if parsed is not None:
            yield parsed


def parse_sentence_html(html: str) -> dict[str, Any]:
    """One streamed `_chat_sentence.html`: the sentence's own text and its
    citations, each a chip (`a.citation-link`) followed by its optional
    note-level or note-text badge, a passage citation's quote
    (`q.passage-quote`) and "Open the original" button (`a.dig`)."""
    span = parse_html(html).find("span", "chat-sentence")
    parts: list[str] = []
    citations: list[dict[str, Any]] = []
    for child in span.children if span is not None else []:
        if isinstance(child, str):
            parts.append(child)
        elif child.tag == "a" and child.has_class("citation-link"):
            href = child.attrs.get("href", "")
            split = urlsplit(href)
            claim = _CLAIM_FRAGMENT_RE.match(split.fragment)
            citations.append(
                {
                    "label": child.text(),
                    "note_href": href,
                    "note_path": split.path,
                    "claim_id": int(claim.group(1)) if claim else None,
                    "note_level": False,
                    "passage_quote": None,
                    "original_href": None,
                }
            )
        elif child.has_class("note-level") and citations:
            citations[-1]["note_level"] = True
        elif child.has_class("passage") and citations:
            continue
        elif child.tag == "q" and child.has_class("passage-quote") and citations:
            citations[-1]["passage_quote"] = child.text().strip()
        elif child.tag == "a" and child.has_class("dig") and citations:
            citations[-1]["original_href"] = child.attrs.get("href")
        else:
            parts.append(child.text())
    return {"text": " ".join("".join(parts).split()), "citations": citations}


def claim_quote(claim: Node) -> str | None:
    quote = claim.find("p", "claim-quote")
    if quote is None:
        return None
    return quote.text().removeprefix("“").removesuffix("”").strip() or None


def check_original(citation: dict[str, Any], original: Page, problems: list[str]) -> None:
    """The cited block is on the original page, and the claim's quote is
    in that block or elsewhere on the page."""
    href = citation["original_href"]
    body = original.dom.find(attrs={"id": "original-body"})
    if body is None:
        if original.dom.find("iframe") is None:
            problems.append(f"original {href} has no document body")
        else:
            citation["quote_check"] = "skipped: a PDF original, its text is not in the page"
        return
    locator = dict(parse_qsl(urlsplit(href).query)).get("at", "")
    # Chat and survey originals render one block with no locator.
    blocks = [n for n in body.descendants() if "data-locator" in n.attrs]
    block = next((n for n in blocks if n.attrs["data-locator"] == locator), None)
    # A passage citation may name the original with no locator: the whole page.
    if blocks and locator:
        citation["cited_block_found"] = block is not None
        if block is None:
            problems.append(f"no block at locator {locator!r} on {href}")
    quote = citation["quote"]
    if quote is None:
        citation["quote_check"] = "skipped: the claim has no quote"
        return
    fields, label = row_fields(block) if block is not None else ({}, "")
    if fields:
        in_block = row_quote_match(quote, fields, label)
    else:
        in_block = quote_match(quote, block.text()) if block is not None else "none"
    citation["quote_in_cited_block"] = in_block != "none"
    citation["quote_match"] = in_block if in_block != "none" else quote_match(quote, body.text())
    if citation["quote_match"] == "none":
        problems.append(f"the claim's quote is not on {href}")


def resolve_citation(
    citation: dict[str, Any],
    get_note: Callable[[str], Page | None],
    get_original: Callable[[str], Page | None],
) -> None:
    """Follow one citation the way a reader clicks it; record on it what
    was found and every problem. `get_note` and `get_original` return the
    page at a path, or `None` unless it returned 200."""
    problems: list[str] = []
    citation["quote"] = None
    note = get_note(citation["note_path"])
    if note is None:
        problems.append(f"note page {citation['note_path']} did not return 200")
    elif citation["claim_id"] is not None:
        claim = note.dom.find(attrs={"id": f"claim-{citation['claim_id']}"})
        if claim is None:
            problems.append(f"claim {citation['claim_id']} is not on {citation['note_path']}")
        else:
            citation["quote"] = claim_quote(claim)
    elif citation.get("passage_quote"):
        # A passage is a chunk of the note (its title, body or claims), so
        # its quote must be somewhere on the note page.
        match = quote_match(citation["passage_quote"], note.dom.text())
        citation["passage_quote_match"] = match
        if match == "none":
            problems.append(f"the passage quote is not on {citation['note_path']}")
    href = citation["original_href"]
    if href:
        original = get_original(href)
        if original is None:
            problems.append(f"original {href} did not return 200")
        else:
            check_original(citation, original, problems)
    citation["ok"] = not problems
    citation["problems"] = problems


def stream_question(
    client: httpx.Client,
    base_url: str,
    question: str,
    conversation: str,
    *,
    timeout: float,
    chat_timeout: float,
) -> dict[str, Any]:
    """POST one question to the stream endpoint and read the reply to its
    end. Returns what arrived; nothing is judged here."""
    path = f"/chat/{conversation}/stream"
    out: dict[str, Any] = {
        "path": path,
        "status": None,
        "content_type": None,
        "sentences": [],
        "done": None,
        "error": None,
        "first_sentence_ms": None,
        "answer_ms": None,
        "bytes": 0,
    }
    started = time.perf_counter()
    try:
        # Multipart, as the browser's `FormData` sends it.
        with client.stream(
            "POST",
            f"{base_url}{path}",
            files={"question": (None, question)},
            timeout=httpx.Timeout(timeout, read=chat_timeout),
            follow_redirects=False,
        ) as response:
            out["status"] = response.status_code
            out["content_type"] = response.headers.get("content-type")
            if response.status_code != 200:
                response.read()
                out["error"] = f"status {response.status_code}: {response.text[:300]}"
            else:
                for event, data in sse_events(response.iter_text()):
                    payload = json.loads(data)
                    elapsed = round((time.perf_counter() - started) * 1000, 1)
                    if event == "sentence":
                        out["sentences"].append(parse_sentence_html(payload.get("html", "")))
                        if out["first_sentence_ms"] is None:
                            out["first_sentence_ms"] = elapsed
                    elif event == "done":
                        out["done"] = payload
                        out["answer_ms"] = elapsed
                    elif event == "error":
                        out["error"] = f"error event: {payload.get('detail')}"
            out["bytes"] = response.num_bytes_downloaded
    except (httpx.HTTPError, ValueError) as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    out["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 1)
    return out


def classify_outcome(streamed: dict[str, Any]) -> str:
    sentences = streamed["sentences"]
    if streamed["error"] or streamed["done"] is None:
        return "error"
    if not sentences:
        return "no_answer"
    if len(sentences) == 1 and not sentences[0]["citations"]:
        text = sentences[0]["text"]
        if text.startswith((NOTHING_RECORDED_PREFIX, GAPS_PREFIX)):
            return "nothing_recorded"
        if text.startswith(INDEX_BROKEN_PREFIX):
            return "index_broken"
    return "answered"
