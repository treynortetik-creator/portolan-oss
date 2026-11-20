"""Heading-aware chunking for note bodies.

Spec: 01-spec/portolan-spec-v2.md sec 9. Contract:
01-spec/build-contracts.md "Embeddings, index, search":
`chunk_note(body_md, max_chars=1200) -> list[Chunk]`.

Splits on headings first, then on paragraphs within each heading section,
packing consecutive paragraphs up to `max_chars`; a paragraph that alone
exceeds `max_chars` is split at sentence boundaries instead, and only a
single run-on sentence that still exceeds `max_chars` is hard-split by
character count (last resort -- "never mid-sentence when avoidable"). Every
chunk records the nearest heading above it and its `[start, end)` char
offsets into `body_md`.

Optional `title`/`description` keyword args add a synthetic chunk 0 holding
the title (and description), marked `is_title=True` with sentinel offsets
`(-1, -1)`, so the title gets its own embedding and full-text row.
`index_note` passes them from the `notes` row.

Three things in a rendered note are not chunks, because nobody retrieves
them and each one took a place among the vector leg's nearest chunks: the
YAML header `write_note` renders at the top of `body_md` (the title chunk
already holds the title and description), a chunk with no words of its own
(a "Related" section's list of `[[links]]`), and HTML comments. A list
longer than `max_chars` (a note's claims) splits between items, so a
passage never starts or ends inside a claim.
"""

from __future__ import annotations

import re

from pydantic import BaseModel

_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.*)$")
# A "sentence" ends at .!? followed by whitespace or end of string; DOTALL so
# one can span lines.
_SENTENCE_RE = re.compile(r".+?(?:(?<=[.!?])(?=\s|\Z)|\Z)", re.DOTALL)


# The YAML header at the very top of a rendered note: a `---` line, then
# non-blank lines up to the next `---` line. Anchored at offset 0 and closed
# before any blank line, so a rule further down the body is text.
_FRONTMATTER_RE = re.compile(r"\A---[ \t]*\n(?:[^\n]+\n)*?---[ \t]*(?:\n|\Z)")
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]*))?\]\]")
_WORD_RE = re.compile(r"[^\W_]{2,}")
# A list item, a numbered item or a table row: a line that is a unit of its own.
_ITEM_LINE_RE = re.compile(r"^[ \t]*(?:[-*+][ \t]|\d+[.)][ \t]|\|)")


class Chunk(BaseModel):
    text: str
    heading: str | None = None
    start: int
    end: int
    is_title: bool = False
    token_estimate: int


def _estimate_tokens(text: str) -> int:
    """chars/4 heuristic: no tokenizer dependency, good enough for
    `chunks.token_count`."""
    return max(1, len(text) // 4)


def _iter_paragraphs(text: str) -> list[tuple[str, int, int]]:
    """Blank-line-separated blocks of `text`, as (text, start, end) with
    offsets relative to the start of `text`."""
    out: list[tuple[str, int, int]] = []
    pos = 0
    buf_start: int | None = None
    buf_lines: list[str] = []
    for line in text.splitlines(keepends=True):
        if line.strip() == "":
            if buf_lines:
                para = "".join(buf_lines)
                out.append((para, buf_start, buf_start + len(para)))  # type: ignore[operator]
                buf_lines = []
                buf_start = None
        else:
            if buf_start is None:
                buf_start = pos
            buf_lines.append(line)
        pos += len(line)
    if buf_lines:
        para = "".join(buf_lines)
        out.append((para, buf_start, buf_start + len(para)))  # type: ignore[operator]
    return out


def _iter_sections(body_md: str) -> list[tuple[str | None, str, int]]:
    """Heading-delimited sections of `body_md`, as (heading_text_or_None,
    content, content_start_offset). Content before the first heading (if
    any) is its own section with heading=None."""
    out: list[tuple[str | None, str, int]] = []
    pos = 0
    heading: str | None = None
    buf_lines: list[str] = []
    buf_start = 0
    for line in body_md.splitlines(keepends=True):
        m = _HEADING_RE.match(line)
        if m:
            out.append((heading, "".join(buf_lines), buf_start))
            heading = m.group(2).strip()
            buf_lines = []
            buf_start = pos + len(line)
        else:
            buf_lines.append(line)
        pos += len(line)
    out.append((heading, "".join(buf_lines), buf_start))
    return out


def _item_units(text: str) -> list[tuple[str, int, int]] | None:
    """`text` as list items, each with the lines that continue it, when its
    first line is an item line and it holds more than one item; else None."""
    lines = text.splitlines(keepends=True)
    if len(lines) < 2 or not _ITEM_LINE_RE.match(lines[0]):
        return None
    units: list[tuple[str, int, int]] = []
    pos = 0
    for line in lines:
        if _ITEM_LINE_RE.match(line) or not units:
            units.append((line, pos, pos + len(line)))
        else:
            item, start, _end = units[-1]
            units[-1] = (item + line, start, pos + len(line))
        pos += len(line)
    return units if len(units) > 1 else None


def _has_words(text: str) -> bool:
    """False for text that is only links, comments and punctuation."""
    bare = _WIKILINK_RE.sub(" ", _HTML_COMMENT_RE.sub(" ", text))
    return _WORD_RE.search(bare) is not None


def _pack(
    parts: list[tuple[str, int, int]],
    max_chars: int,
    base_start: int,
    *,
    hard_fallback: bool = False,
) -> list[tuple[str, int, int]]:
    """Greedily pack consecutive (text, rel_start, rel_end) parts into
    chunks no longer than `max_chars`, never splitting a part across two
    chunks unless the part alone exceeds `max_chars`. In that case, when
    `hard_fallback` is False (packing paragraphs or list items) a list is
    packed item by item and anything else sentence by sentence; when True
    (already packing sentences) it falls back to a raw character split --
    the true last resort, only reached for a single run-on sentence with no
    punctuation to split on.
    """
    out: list[tuple[str, int, int]] = []
    cur: list[tuple[str, int, int]] = []
    cur_len = 0

    def flush() -> None:
        nonlocal cur, cur_len
        if not cur:
            return
        text = "".join(p[0] for p in cur)
        start = base_start + cur[0][1]
        end = base_start + cur[-1][2]
        stripped = text.strip()
        if stripped:
            out.append((stripped, start, end))
        cur = []
        cur_len = 0

    for part_text, rel_start, rel_end in parts:
        part_len = len(part_text)
        if part_len > max_chars:
            flush()
            if hard_fallback:
                for i in range(0, part_len, max_chars):
                    piece = part_text[i : i + max_chars]
                    if piece.strip():
                        out.append(
                            (
                                piece.strip(),
                                base_start + rel_start + i,
                                base_start + rel_start + min(i + max_chars, part_len),
                            )
                        )
                continue
            items = _item_units(part_text)
            if items is not None:
                out.extend(_pack(items, max_chars, base_start + rel_start))
            else:
                sentences = [
                    (m.group(0), m.start(), m.end()) for m in _SENTENCE_RE.finditer(part_text)
                ]
                out.extend(_pack(sentences, max_chars, base_start + rel_start, hard_fallback=True))
            continue
        if cur_len + part_len > max_chars and cur:
            flush()
        cur.append((part_text, rel_start, rel_end))
        cur_len += part_len

    flush()
    return out


def chunk_note(
    body_md: str,
    max_chars: int = 1200,
    *,
    title: str | None = None,
    description: str | None = None,
) -> list[Chunk]:
    """Split `body_md` into heading-aware chunks (see module docstring).

    `title`/`description`, when given, produce a synthetic chunk 0 ahead of
    the body chunks so the note's title always has its own well-matching
    embedding and full-text row.
    """
    chunks: list[Chunk] = []

    if title:
        title_text = f"{title}\n\n{description}".strip() if description else title.strip()
        chunks.append(
            Chunk(
                text=title_text,
                heading=None,
                start=-1,
                end=-1,
                is_title=True,
                token_estimate=_estimate_tokens(title_text),
            )
        )

    if not body_md:
        return chunks

    header = _FRONTMATTER_RE.match(body_md)
    offset = header.end() if header else 0
    for heading, content, content_start in _iter_sections(body_md[offset:]):
        paragraphs = [
            p for p in _iter_paragraphs(content) if _HTML_COMMENT_RE.sub("", p[0]).strip()
        ]
        for text, start, end in _pack(paragraphs, max_chars, offset + content_start):
            if not _has_words(text):
                continue
            chunks.append(
                Chunk(
                    text=text,
                    heading=heading,
                    start=start,
                    end=end,
                    token_estimate=_estimate_tokens(text),
                )
            )

    return chunks
