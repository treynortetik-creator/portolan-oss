"""`prepare_text`: strip quoted text and boilerplate, inject locators.

Per source type (build-contracts.md build item 3; formats in
`01-spec/provenance-chain.md`'s locator table, which pins each shape to the
planted locators in `corpus/out/manifest.json`). Every locator is injected
into the text as a `[locator]` marker, so the extractor can see and cite it.

- **call** — the transcript already carries `[HH:MM:SS]` line prefixes,
  kept as-is: one span per timestamped line (untimestamped lines extend the
  previous span). A preamble before the first timestamp gets its own
  `header` span so it stays citable.
- **ticket** — one locator per comment, `<source_id>#<n>`, from
  `raw.metadata["comments"]` (`{"number", "body", "author"?}` dicts); a
  comment missing `number` uses its 1-indexed position. With no structured
  comments, the whole `raw.text` is comment 1.
- **email** — `_strip_quoted_and_boilerplate` first drops forwarded-header
  blocks, a trailing quoted block (with its "On ... wrote:" line), a
  signature, and trailing footer paragraphs (counted in `removed_chars`;
  never emptying non-empty input). Each remaining paragraph is marked
  `<message-id>#p<n>`, the id from `raw.metadata["message_id"]`, else
  `raw.source_id`.
- **doc** — `p<page>¶<para>` when the text carries `[page N]` marker lines
  (case-insensitive, injected by the connector), else `L<line>` per line.
- **crm** — one locator for the row: `<path>#<crm_id>` when
  `raw.metadata["path"]` is set (e.g. `crm/opportunities.csv#O-3001`), else
  `<object>:<crm_id>`; `crm_id` and `object` from `raw.metadata`, defaulting
  to `raw.source_id` and `record`.
- **rss** — one locator for the item, `url:<link> <date>`, from
  `raw.metadata["link"]` and `raw.source_date`.

Anything else (`chat`, `survey`) falls back to the doc behaviour.
"""

from __future__ import annotations

import re

from portolan.ingest.models import LocatorSpan, PreparedText, RawSource, SourceType

# --------------------------------------------------------------------
# call
# --------------------------------------------------------------------

_CALL_LINE_RE = re.compile(r"^\[(\d{2}:\d{2}:\d{2})\]")


def _prepare_call(raw: RawSource) -> PreparedText:
    text = raw.text
    lines = text.splitlines(keepends=True)
    locators: list[LocatorSpan] = []
    offset = 0
    current_locator: str | None = None
    current_start = 0
    header_end: int | None = None
    for line in lines:
        match = _CALL_LINE_RE.match(line)
        if match:
            if current_locator is None and header_end is None:
                header_end = offset
            if current_locator is not None:
                locators.append(
                    LocatorSpan(locator=current_locator, start=current_start, end=offset)
                )
            current_locator = match.group(1)
            current_start = offset
        offset += len(line)
    if current_locator is not None:
        locators.append(LocatorSpan(locator=current_locator, start=current_start, end=offset))

    # A preamble before the first timestamp gets a `header` span. `header_end`
    # is falsy both with no timestamp at all and with no preamble.
    if header_end and text[:header_end].strip():
        locators.insert(0, LocatorSpan(locator="header", start=0, end=header_end))

    return PreparedText(text=text, locators=locators, removed_chars=0)


# --------------------------------------------------------------------
# ticket
# --------------------------------------------------------------------


def _prepare_ticket(raw: RawSource) -> PreparedText:
    comments = raw.metadata.get("comments")
    if not comments:
        comments = [{"number": 1, "body": raw.text, "author": None}]

    parts: list[str] = []
    locators: list[LocatorSpan] = []
    offset = 0
    for index, comment in enumerate(comments, start=1):
        number = comment.get("number", index)
        author = comment.get("author")
        locator = f"{raw.source_id}#{number}"
        marker = f"[{locator}]"
        header = f"{marker} {author}: " if author else f"{marker} "
        body = str(comment.get("body", "")).strip()
        block = header + body + "\n\n"
        start = offset
        parts.append(block)
        offset += len(block)
        locators.append(LocatorSpan(locator=locator, start=start, end=offset))

    text = "".join(parts).rstrip("\n") + ("\n" if parts else "")
    return PreparedText(text=text, locators=locators, removed_chars=0)


# --------------------------------------------------------------------
# email
# --------------------------------------------------------------------

# Any depth of quoting: ">", ">>", "> >", etc, optionally indented.
_QUOTE_LINE_RE = re.compile(r"^\s{0,3}>+")
_ON_WROTE_RE = re.compile(r"^On .+ wrote:\s*$")
_FORWARDED_MARKER_RE = re.compile(r"^-{2,}\s*Forwarded message\s*-{2,}", re.IGNORECASE)
_FORWARDED_HEADER_RE = re.compile(r"^(From|To|Cc|Date|Subject):", re.IGNORECASE)
_SIGNATURE_DELIMITER = "-- "
_DISCLAIMER_KEYWORDS = (
    "unsubscribe",
    "this email and any attachments",
    "intended only for the use of the individual",
    "confidential and may be legally privileged",
)
# A disclaimer keyword alone does not make a footer: "Please unsubscribe our
# night-shift nurses... I have asked three times" is a request. A first-person
# pronoun or a question mark keeps the paragraph.
_FIRST_PERSON_RE = re.compile(r"\b(i|i'm|i've|we|we're|we've|our|my|me|us)\b", re.IGNORECASE)


def _is_footer_paragraph(paragraph: str) -> bool:
    low = paragraph.lower()
    if not any(keyword in low for keyword in _DISCLAIMER_KEYWORDS):
        return False
    if "?" in paragraph or _FIRST_PERSON_RE.search(paragraph):
        return False
    return True


def _find_trailing_quote_cut(lines: list[str]) -> int | None:
    """Index of the first line of a quoted block that runs uninterrupted
    to the end of `lines` (blank lines allowed inside it), optionally
    preceded immediately by an "On ... wrote:" top-post marker that
    introduces it.

    `None` if no such trailing block exists. A `>` block followed by
    unquoted content is real content (a customer quoting their own
    complaint) and is left alone.
    """
    n = len(lines)
    i = n
    saw_quote_line = False
    while i > 0:
        line = lines[i - 1]
        stripped = line.strip()
        if stripped == "":
            i -= 1
            continue
        if _QUOTE_LINE_RE.match(line):
            saw_quote_line = True
            i -= 1
            continue
        break
    if not saw_quote_line:
        return None
    if i > 0 and _ON_WROTE_RE.match(lines[i - 1].strip()):
        i -= 1
    return i


def _strip_quoted_and_boilerplate(text: str) -> tuple[str, int, bool]:
    """Returns `(cleaned_text, removed_char_count, stripping_skipped)`."""
    original_len = len(text)
    if original_len == 0:
        return text, 0, False

    lines = text.splitlines()

    # Forwarded-message marker: drop the marker line and the contiguous
    # From/To/Date/Subject header block after it; keep the forwarded body.
    kept: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        stripped = lines[i].strip()
        if _FORWARDED_MARKER_RE.match(stripped):
            i += 1
            while i < n and _FORWARDED_HEADER_RE.match(lines[i].strip()):
                i += 1
            continue
        kept.append(lines[i])
        i += 1

    # Only a `>` run at the end of the body is dropped (`_find_trailing_quote_cut`).
    cut = _find_trailing_quote_cut(kept)
    if cut is not None:
        kept = kept[:cut]

    # Signature: a block opening with exactly "-- " (RFC 3676's delimiter,
    # trailing space included) anywhere, or with a bare "--" only in the last
    # block, so a mid-message "--" section break does not truncate content.
    blocks: list[tuple[int, int]] = []
    block_start: int | None = None
    for idx, line in enumerate(kept):
        if line.strip() == "":
            if block_start is not None:
                blocks.append((block_start, idx))
                block_start = None
        elif block_start is None:
            block_start = idx
    if block_start is not None:
        blocks.append((block_start, len(kept)))

    sig_cut: int | None = None
    last_block = blocks[-1] if blocks else None
    for start, _end in blocks:
        if kept[start] == _SIGNATURE_DELIMITER:
            sig_cut = start
            break
        if kept[start].strip() == "--" and (start, _end) == last_block:
            sig_cut = start
            break
    if sig_cut is not None:
        kept = kept[:sig_cut]

    cleaned = "\n".join(kept)

    # Footer paragraphs (unsubscribe, legal disclaimers): only a trailing run,
    # scanned from the end and stopped at the first non-footer paragraph.
    paragraphs = re.split(r"\n\s*\n", cleaned)
    cut = len(paragraphs)
    for paragraph in reversed(paragraphs):
        if paragraph.strip() == "":
            cut -= 1
            continue
        if _is_footer_paragraph(paragraph):
            cut -= 1
            continue
        break

    surviving = [p.strip() for p in paragraphs[:cut] if p.strip()]
    cleaned = "\n\n".join(surviving)

    # Never hand extraction empty text for non-empty input: a still-quoted
    # line beats a source that silently vanishes.
    if not cleaned.strip():
        return text, 0, True

    removed = max(original_len - len(cleaned), 0)
    return cleaned, removed, False


def _prepare_email(raw: RawSource) -> PreparedText:
    message_id = raw.metadata.get("message_id", raw.source_id)
    cleaned, removed, stripping_skipped = _strip_quoted_and_boilerplate(raw.text)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", cleaned) if p.strip()]

    parts: list[str] = []
    locators: list[LocatorSpan] = []
    offset = 0
    for index, paragraph in enumerate(paragraphs, start=1):
        locator = f"{message_id}#p{index}"
        block = f"[{locator}] {paragraph}\n\n"
        start = offset
        parts.append(block)
        offset += len(block)
        locators.append(LocatorSpan(locator=locator, start=start, end=offset))

    text = "".join(parts).rstrip("\n") + ("\n" if parts else "")
    return PreparedText(
        text=text,
        locators=locators,
        removed_chars=removed,
        stripping_skipped=stripping_skipped,
    )


# --------------------------------------------------------------------
# doc
# --------------------------------------------------------------------

_PAGE_MARKER_RE = re.compile(r"^\s*\[page\s+(\d+)\]\s*$", re.IGNORECASE)


def _ensure_trailing_newline(block: str) -> str:
    return block if block.endswith("\n") else block + "\n"


def _prepare_doc(raw: RawSource) -> PreparedText:
    lines = raw.text.splitlines(keepends=True)
    has_page_markers = any(_PAGE_MARKER_RE.match(line.strip("\n")) for line in lines)

    locators: list[LocatorSpan] = []
    parts: list[str] = []
    offset = 0

    if has_page_markers:
        page = 0
        para = 0
        para_lines: list[str] = []

        def _flush() -> None:
            nonlocal para, offset
            if not any(line.strip() for line in para_lines):
                para_lines.clear()
                return
            para += 1
            locator = f"p{page}¶{para}"
            block = _ensure_trailing_newline(f"[{locator}] " + "".join(para_lines))
            start = offset
            parts.append(block)
            offset += len(block)
            locators.append(LocatorSpan(locator=locator, start=start, end=offset))
            para_lines.clear()

        for line in lines:
            stripped = line.strip("\n")
            marker = _PAGE_MARKER_RE.match(stripped)
            if marker:
                _flush()
                page = int(marker.group(1))
                para = 0
                continue
            if not stripped.strip():
                _flush()
                continue
            para_lines.append(line)
        _flush()
    else:
        for line_number, line in enumerate(lines, start=1):
            if line.strip():
                locator = f"L{line_number}"
                block = _ensure_trailing_newline(f"[{locator}] {line}")
                start = offset
                parts.append(block)
                offset += len(block)
                locators.append(LocatorSpan(locator=locator, start=start, end=offset))
            else:
                block = _ensure_trailing_newline(line) if line else "\n"
                parts.append(block)
                offset += len(block)

    return PreparedText(text="".join(parts), locators=locators, removed_chars=0)


# --------------------------------------------------------------------
# crm
# --------------------------------------------------------------------


def _prepare_crm(raw: RawSource) -> PreparedText:
    crm_id = raw.metadata.get("crm_id", raw.source_id)
    path = raw.metadata.get("path")
    locator = f"{path}#{crm_id}" if path else f"{raw.metadata.get('object', 'record')}:{crm_id}"
    # The marker goes in the text: an extractor cannot cite a locator it never sees.
    text = f"[{locator}] {raw.text}"
    return PreparedText(
        text=text,
        locators=[LocatorSpan(locator=locator, start=0, end=len(text))],
        removed_chars=0,
    )


# --------------------------------------------------------------------
# rss
# --------------------------------------------------------------------


def _prepare_rss(raw: RawSource) -> PreparedText:
    link = raw.metadata.get("link", "")
    locator = f"url:{link} {raw.source_date.isoformat()}"
    text = f"[{locator}] {raw.text}"
    return PreparedText(
        text=text,
        locators=[LocatorSpan(locator=locator, start=0, end=len(text))],
        removed_chars=0,
    )


# --------------------------------------------------------------------
# dispatch
# --------------------------------------------------------------------

_HANDLERS = {
    SourceType.CALL: _prepare_call,
    SourceType.TICKET: _prepare_ticket,
    SourceType.EMAIL: _prepare_email,
    SourceType.DOC: _prepare_doc,
    SourceType.CRM: _prepare_crm,
    SourceType.RSS: _prepare_rss,
}


def prepare_text(raw: RawSource) -> PreparedText:
    """Strip quoted text/boilerplate and inject locators, per source type.
    Types without a handler (`chat`, `survey`) get `_prepare_doc`'s line
    numbers rather than an error.
    """
    handler = _HANDLERS.get(raw.source_type, _prepare_doc)
    return handler(raw)
