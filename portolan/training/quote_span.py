"""Exact-span resolution for a kept fact's quote (`feature/labeling-
exact-quotes`, orchestrator review in `01-spec/training-data-2026-09-19.md`,
2026-09-22 06:10 MST). The overlap gate (`portolan.training.overlap_gate`)
only asks whether a quote is verbatim/near-verbatim SUPPORTED by the
document text (`portolan.extract.quote_gate.quote_is_supported`: exact OR
fuzzy token-subsequence containment) -- a fuzzy-but-not-exact quote still
passes that gate and gets kept. A student model trained to *copy* a quote
needs the quote to be an exact substring of the record `text` it will see
at inference, not a paraphrase the gate merely judged "close enough". This
module resolves a kept fact's quote down to the exact original-text span
it names, in three widening tiers, or reports that none exists.

Tiers, tried in order, cheapest/strictest first:

1. **exact** -- `quote` is already a literal substring of `text`.
2. **normalized** -- equal after glyph-folding (curly quotes/dashes/an
   ellipsis character, `portolan.extract.quote_gate`'s own fold table),
   NFKC normalization, casefolding, and whitespace-run collapse -- the
   same normalization `quote_gate._normalize_for_containment` uses for
   its OWN containment check, but here every output character is also
   mapped back to the original span of `text` it came from
   (`_build_normalized_index`), so a normalized match can be translated
   back into an exact `(start, end)` into the UN-normalized `text`.
3. **mime** -- same as `normalized`, but first strips the quoted-
   printable artifacts real Enron MIME source text carries (`=20` -> a
   space, `=0D` -> a carriage return, `=09` -> a tab, a soft line break
   `=\r\n`/`=\n` -> nothing) before the same character-mapped
   normalization.

**Stitched quotes.** A quote containing `"..."` (literally, or an
ellipsis character the fold table turns into `"..."`) is the model
splicing two spans of the document together with an elision marker --
`quote_is_supported`'s fuzzy token-subsequence check tolerates this (each
side's tokens are found in order), but neither side is a literal,
contiguous span of `text` as a whole. This resolver's choice
(orchestrator review, documented here rather than left implicit): keep
ONLY the FIRST spliced segment, resolved through the same three tiers,
and re-point `quote` at just that segment -- never both, never a
best-of-both-segments pick. Two segments stitched together straddle a
gap the student would have to invent to reproduce contiguously, and the
first segment is the one a fact claim most often anchors its own opening
clause on (the "..." trails off into supporting detail more often than
it leads with it, by inspection of the real Enron drops this fix was
written against). A fact whose first segment alone cannot be resolved
through any tier is dropped (`quote_not_exact`) even if a later segment
could be -- consistent, not "try every segment and take whichever
resolves".

Nothing here is network- or corpus-specific: `resolve_quote_span` is a
pure function over two strings, unit-tested with synthetic cases
(`tests/test_training_quote_span.py`), and used both live
(`portolan.training.pipeline.run_corpus_single`) and offline
(`scripts/data/backfill_exact_quotes.py`).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from portolan.extract.quote_gate import _fold_glyphs

__all__ = [
    "ResolvedQuote",
    "resolve_quote_span",
]


@dataclass(frozen=True)
class ResolvedQuote:
    """`quote` is the exact substring `text[span[0]:span[1]]` -- callers
    must overwrite the fact's own `quote` field with THIS string, not
    keep the model's original wording, so `text[start:end] == quote`
    holds by construction. `tier` is one of `"exact"`, `"normalized"`,
    `"mime"`, or a stitched variant (`"stitched:exact"`,
    `"stitched:normalized"`, `"stitched:mime"`) -- which widening tier
    resolved it, for stats/diagnostics only."""

    quote: str
    span: tuple[int, int]
    tier: str


_WHITESPACE_RE = re.compile(r"\s+")

# Quoted-printable artifacts real Enron MIME source text left undecoded
# (`01-spec/training-data-2026-09-19.md`'s own "Known quirks" line): a
# soft line break (trailing `=` before the newline -- joins the next
# line with NO whitespace introduced) and three codes that decode to
# whitespace control characters (`=20` space, `=0D` CR, `=09` tab).
_MIME_SOFT_BREAK_RE = re.compile(r"=\r\n|=\n")
_MIME_WHITESPACE_CODES: frozenset[str] = frozenset({"=20", "=0D", "=09"})


def _normalize_plain(s: str) -> str:
    """The same normalization `_build_normalized_index` produces,
    without the position map -- used on the (short, unpositioned) quote
    side of a comparison."""
    folded = _fold_glyphs(unicodedata.normalize("NFKC", s))
    return _WHITESPACE_RE.sub(" ", folded.strip().casefold())


def _strip_mime(s: str) -> str:
    s = _MIME_SOFT_BREAK_RE.sub("", s)
    for code in _MIME_WHITESPACE_CODES:
        s = s.replace(code, " ")
    return s


def _build_normalized_index(text: str, *, mime: bool) -> tuple[str, list[tuple[int, int]]]:
    """`(normalized, spans)`: `normalized[i]` was produced by
    `text[spans[i][0]:spans[i][1]]`. A whitespace run (including any
    MIME artifact decoded to whitespace, or skipped entirely for a soft
    line break, when `mime=True`) collapses to one mapped space spanning
    the whole run; every other output character maps back to exactly the
    one input character it came from, even when NFKC/glyph-folding/
    casefolding that one character expands it to more than one output
    character (an ellipsis glyph -> three dots, `ß` -> `ss`, ...)."""
    out_chars: list[str] = []
    out_spans: list[tuple[int, int]] = []
    n = len(text)
    i = 0
    ws_start: int | None = None

    def flush_ws(end: int) -> None:
        nonlocal ws_start
        if ws_start is not None:
            out_chars.append(" ")
            out_spans.append((ws_start, end))
            ws_start = None

    while i < n:
        if mime:
            if text[i : i + 3] == "=\r\n":
                i += 3
                continue
            if text[i : i + 2] == "=\n":
                i += 2
                continue
            if text[i : i + 3] in _MIME_WHITESPACE_CODES:
                if ws_start is None:
                    ws_start = i
                i += 3
                continue

        ch = text[i]
        if ch.isspace():
            if ws_start is None:
                ws_start = i
            i += 1
            continue

        flush_ws(i)
        for folded_ch in _fold_glyphs(unicodedata.normalize("NFKC", ch)):
            for cased_ch in folded_ch.casefold():
                out_chars.append(cased_ch)
                out_spans.append((i, i + 1))
        i += 1

    flush_ws(n)
    return "".join(out_chars), out_spans


def _span_for_match(
    normalized_quote: str, normalized_text: str, spans: list[tuple[int, int]]
) -> tuple[int, int] | None:
    if not normalized_quote:
        return None
    idx = normalized_text.find(normalized_quote)
    if idx == -1:
        return None
    end_idx = idx + len(normalized_quote) - 1
    return spans[idx][0], spans[end_idx][1]


def _resolve_single_span(quote: str, text: str) -> ResolvedQuote | None:
    """Resolve one (non-stitched) `quote` against `text` through the
    exact/normalized/mime tiers -- the building block `resolve_quote_
    span` also runs on the first segment of a stitched quote."""
    if not quote or not quote.strip() or not text:
        return None

    idx = text.find(quote)
    if idx != -1:
        return ResolvedQuote(quote=quote, span=(idx, idx + len(quote)), tier="exact")

    normalized_text, spans = _build_normalized_index(text, mime=False)
    span = _span_for_match(_normalize_plain(quote), normalized_text, spans)
    if span is not None:
        start, end = span
        return ResolvedQuote(quote=text[start:end], span=(start, end), tier="normalized")

    mime_text, mime_spans = _build_normalized_index(text, mime=True)
    span = _span_for_match(_normalize_plain(_strip_mime(quote)), mime_text, mime_spans)
    if span is not None:
        start, end = span
        return ResolvedQuote(quote=text[start:end], span=(start, end), tier="mime")

    return None


def resolve_quote_span(quote: str | None, text: str) -> ResolvedQuote | None:
    """The public entry point: resolve `quote` to an exact `(start, end)`
    span of `text`, or `None` when no tier -- including the stitched-
    quote fallback -- can. `text[span[0]:span[1]] == result.quote` always
    holds for a non-`None` result; callers assert this, they never trust
    it silently (`run_corpus_single`, `backfill_exact_quotes.py`)."""
    if not quote or not quote.strip() or not text:
        return None

    resolved = _resolve_single_span(quote, text)
    if resolved is not None:
        return resolved

    folded = _fold_glyphs(quote)
    if "..." not in folded:
        return None

    segments = [seg.strip() for seg in folded.split("...") if seg.strip()]
    if not segments:
        return None

    first_resolved = _resolve_single_span(segments[0], text)
    if first_resolved is None:
        return None
    return ResolvedQuote(
        quote=first_resolved.quote,
        span=first_resolved.span,
        tier=f"stitched:{first_resolved.tier}",
    )
