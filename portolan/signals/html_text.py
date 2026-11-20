"""Readable text out of a public web page, split into sections by heading.

Standard library only (`html.parser`), so the worker image needs no extra
package. The rules, in order:

- If the page has a `<main>` element, only text inside it counts; otherwise the
  whole `<body>` minus page chrome (`header`, `footer`, `nav`, `aside`, forms).
- Never text: `script`, `style`, `noscript`, `template`, `svg`, `iframe`,
  `canvas`, buttons and selects, anything `hidden` or `aria-hidden="true"`, and
  landmarks with role `navigation`, `banner`, `contentinfo`, `search` or
  `dialog`.
- `h1` to `h4` start a new section; block elements end a paragraph.

`needs_browser` is a guess that the page builds its content with JavaScript
(almost no text but an app shell), so the plain-HTTP snapshot is not worth
diffing and the page should be fetched with a browser instead.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from html.parser import HTMLParser

__all__ = ["ExtractedPage", "Section", "extract_page", "html_to_text"]

_SKIP_TAGS = frozenset(
    {
        "script",
        "style",
        "noscript",
        "template",
        "svg",
        "iframe",
        "canvas",
        "button",
        "select",
        "option",
        "nav",
        "aside",
        "form",
        "dialog",
        "head",
    }
)
# Page chrome, skipped only when the page has no `<main>` to scope to.
_CHROME_TAGS = frozenset({"header", "footer"})
_SKIP_ROLES = frozenset({"navigation", "banner", "contentinfo", "search", "dialog"})
_HEADING_TAGS = frozenset({"h1", "h2", "h3", "h4"})
_BLOCK_TAGS = frozenset(
    {
        "p",
        "div",
        "li",
        "ul",
        "ol",
        "section",
        "article",
        "main",
        "header",
        "footer",
        "tr",
        "td",
        "th",
        "table",
        "dd",
        "dt",
        "dl",
        "blockquote",
        "pre",
        "figcaption",
        "h5",
        "h6",
        "br",
        "hr",
        "address",
    }
)
_VOID_TAGS = frozenset(
    {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    }
)
_WS_RE = re.compile(r"\s+")
_MAIN_RE = re.compile(r"<main[\s>]", re.IGNORECASE)
_APP_SHELL_RE = re.compile(
    r'id=["\'](?:root|app|__next|__nuxt)["\']|ng-app|enable javascript|requires javascript',
    re.IGNORECASE,
)

# Below this much extracted text, an app-shell page is treated as needing a browser.
NEEDS_BROWSER_MAX_CHARS = 200
# Hard caps, so one runaway page cannot bloat a snapshot row.
MAX_PARAGRAPHS = 3000
MAX_PARAGRAPH_CHARS = 4000


@dataclass
class Section:
    """One heading and the paragraphs under it. `heading` is `""` for text
    before the first heading."""

    heading: str
    paragraphs: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        return {"heading": self.heading, "paragraphs": list(self.paragraphs)}

    @classmethod
    def from_json(cls, data: dict) -> Section:
        return cls(heading=data.get("heading", ""), paragraphs=list(data.get("paragraphs") or []))


@dataclass
class ExtractedPage:
    title: str
    sections: list[Section]
    text_chars: int
    needs_browser: bool


def _clean(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


class _Extractor(HTMLParser):
    def __init__(self, *, scope_to_main: bool) -> None:
        super().__init__(convert_charrefs=True)
        self.scope_to_main = scope_to_main
        self.stack: list[tuple[str, bool]] = []  # (tag, skips its subtree)
        self.main_depth = 0
        self.title_parts: list[str] = []
        self.in_title = False
        self.heading_parts: list[str] | None = None
        self.buffer: list[str] = []
        self.sections: list[Section] = [Section(heading="")]
        self.paragraph_count = 0

    def _skipping(self) -> bool:
        return any(skip for _tag, skip in self.stack)

    def _collecting(self) -> bool:
        if self._skipping():
            return False
        return self.main_depth > 0 if self.scope_to_main else True

    def _flush(self) -> None:
        text = _clean("".join(self.buffer))
        self.buffer = []
        if len(text) < 2 or self.paragraph_count >= MAX_PARAGRAPHS:
            return
        self.sections[-1].paragraphs.append(text[:MAX_PARAGRAPH_CHARS])
        self.paragraph_count += 1

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag == "title":
            # Only the document's own title: an inline SVG carries `<title>` too.
            self.in_title = not self.title_parts and not any(t == "svg" for t, _ in self.stack)
            return
        if tag in _VOID_TAGS:
            if tag in ("br", "hr") and self.heading_parts is None:
                self._flush()
            return
        attr = {k.lower(): (v or "") for k, v in attrs}
        skip = (
            tag in _SKIP_TAGS
            or (tag in _CHROME_TAGS and not self.scope_to_main)
            or "hidden" in attr
            or attr.get("aria-hidden", "").lower() == "true"
            or attr.get("role", "").lower() in _SKIP_ROLES
        )
        self.stack.append((tag, skip))
        if tag == "main":
            self.main_depth += 1
        if tag in _HEADING_TAGS and self._collecting():
            self._flush()
            self.heading_parts = []
        elif tag in _BLOCK_TAGS and self.heading_parts is None:
            self._flush()

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in ("br", "hr") and self.heading_parts is None:
            self._flush()

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag == "title":
            self.in_title = False
            return
        if tag in _VOID_TAGS:
            return
        # Pop back to the matching open tag; a stray end tag with no match is ignored,
        # so one malformed element cannot unbalance the rest of the page.
        if not any(t == tag for t, _ in self.stack):
            return
        while self.stack:
            open_tag, _skip = self.stack.pop()
            if open_tag == "main":
                self.main_depth = max(0, self.main_depth - 1)
            if open_tag in _HEADING_TAGS and self.heading_parts is not None:
                heading = _clean("".join(self.heading_parts))
                self.heading_parts = None
                if heading:
                    self.sections.append(Section(heading=heading[:300]))
            if open_tag == tag:
                break
        if tag in _BLOCK_TAGS:
            self._flush()

    def handle_data(self, data: str) -> None:
        if self.in_title:
            self.title_parts.append(data)
            return
        if not self._collecting():
            return
        if self.heading_parts is not None:
            self.heading_parts.append(data)
        else:
            self.buffer.append(data)

    def close(self) -> None:
        super().close()
        if self.heading_parts is not None:
            heading = _clean("".join(self.heading_parts))
            self.heading_parts = None
            if heading:
                self.sections.append(Section(heading=heading[:300]))
        self._flush()


def extract_page(html: str) -> ExtractedPage:
    """`html` as a title plus heading-delimited sections (module docstring)."""
    parser = _Extractor(scope_to_main=bool(_MAIN_RE.search(html)))
    parser.feed(html)
    parser.close()
    sections = [s for s in parser.sections if s.paragraphs or s.heading]
    # A heading with nothing under it and nothing after it is noise, not a section.
    while sections and not sections[-1].paragraphs:
        sections.pop()
    text_chars = sum(len(p) for s in sections for p in s.paragraphs)
    title = _clean("".join(parser.title_parts))[:300]
    needs_browser = text_chars < NEEDS_BROWSER_MAX_CHARS and bool(_APP_SHELL_RE.search(html))
    return ExtractedPage(
        title=title, sections=sections, text_chars=text_chars, needs_browser=needs_browser
    )


def html_to_text(fragment: str) -> str:
    """The plain text of an HTML fragment (a feed item's description), one
    paragraph per line."""
    if "<" not in fragment:
        return _clean(fragment)
    parser = _Extractor(scope_to_main=False)
    parser.feed(fragment)
    parser.close()
    lines = []
    for section in parser.sections:
        if section.heading:
            lines.append(section.heading)
        lines.extend(section.paragraphs)
    return "\n".join(lines)
