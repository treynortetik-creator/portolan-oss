"""Search-result snippets.

A hit's `best_chunk` is a raw chunk of the note's `body_md`, which includes
the YAML frontmatter and the claim markdown (`- ` bullets, `**date**`,
`^[source @ locator]` markers, `[[slug]]` links). `search_snippet` strips
both, falls back to the note's `description` when nothing is left,
truncates on a word boundary, and highlights the query terms.
"""

from __future__ import annotations

import re

from markupsafe import Markup, escape

__all__ = [
    "highlight_terms",
    "query_terms_for_highlight",
    "search_snippet",
    "strip_frontmatter",
    "truncate_on_word_boundary",
]

# A leading `---\n...\n---` block; non-greedy so it stops at the first
# closing `---`.
_FRONTMATTER_RE = re.compile(r"\A---\n.*?\n---\n?", re.DOTALL)

# Claim-line markdown (portolan.notes.render `_render_claim_line`), stripped
# for a plain-prose snippet.
_CITATION_MARKER_RE = re.compile(r"\^\[[^\]]*\]")
_WIKILINK_BARE_RE = re.compile(r"\[\[([a-z0-9_-]+)\]\]")
_HEADING_MARK_RE = re.compile(r"(?m)^#{1,6}[ \t]*")
_LEADING_BULLET_RE = re.compile(r"(?m)^[ \t]*[-*][ \t]+")
_MD_EMPHASIS_RE = re.compile(r"\*\*|~~|__")

# Highlighting only, not the search tokenizer: letters, digits and inner
# hyphens, 3+ chars, so "a" or "to" is never highlighted.
_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-]{2,}")


def strip_frontmatter(text: str) -> str:
    """`text` with a leading `---`-delimited frontmatter block removed, if
    it has one; only at the very start of `text`."""
    return _FRONTMATTER_RE.sub("", text or "", count=1).strip()


def _strip_markdown_noise(text: str) -> str:
    """`text` with citation markers dropped, `[[slug]]` reduced to the bare
    slug, and heading/bullet markers and bold/strikethrough/underscore
    delimiters removed."""
    out = _CITATION_MARKER_RE.sub("", text)
    out = _WIKILINK_BARE_RE.sub(r"\1", out)
    out = _HEADING_MARK_RE.sub("", out)
    out = _LEADING_BULLET_RE.sub("", out)
    out = _MD_EMPHASIS_RE.sub("", out)
    return out


def query_terms_for_highlight(query: str) -> list[str]:
    """Terms worth wrapping in `<mark>` for `query`: lowercase, 3+ chars,
    deduplicated, sorted."""
    return sorted({m.group(0).lower() for m in _WORD_RE.finditer(query or "")})


def truncate_on_word_boundary(text: str, limit: int = 200) -> str:
    """`text`, collapsed to single-spaced whitespace, cut at or before
    `limit` characters on a word boundary (never mid-token), with a
    trailing `…` when it was actually cut."""
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    cut = collapsed.rfind(" ", 0, limit)
    if cut <= 0:
        cut = limit
    return collapsed[:cut].rstrip() + "…"


def highlight_terms(text: str, terms: list[str]) -> Markup:
    """`text`, HTML-escaped, with every case-insensitive occurrence of any
    of `terms` wrapped in `<mark>`; the `<mark>` tags are the only
    unescaped markup."""
    if not terms:
        return Markup(escape(text))
    pattern = re.compile(
        "(" + "|".join(re.escape(t) for t in sorted(set(terms), key=len, reverse=True)) + ")",
        re.IGNORECASE,
    )
    pieces = pattern.split(text)
    out = Markup("")
    for i, piece in enumerate(pieces):
        if not piece:
            continue
        out += Markup("<mark>{}</mark>").format(piece) if i % 2 == 1 else Markup(escape(piece))
    return out


def search_snippet(
    best_chunk: str, description: str | None, query: str, *, limit: int = 200
) -> Markup:
    """The snippet for one search hit: frontmatter and claim markdown
    stripped, the note's `description` used when that leaves nothing,
    truncated on a word boundary, query terms highlighted."""
    body = _strip_markdown_noise(strip_frontmatter(best_chunk)).strip()
    # Descriptions carry wikilinks too (e.g. a signal note's
    # "Pricing signals for [[slug]].", portolan.merge.facts).
    text = body if body else _strip_markdown_noise(description or "").strip()
    snippet = truncate_on_word_boundary(text, limit)
    return highlight_terms(snippet, query_terms_for_highlight(query))
