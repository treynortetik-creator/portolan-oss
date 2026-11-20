"""The mechanical quote gate: does a fact's quote actually appear in the
source text it cites, verbatim or near-verbatim?

Moved here from `portolan.bench.extraction` (2026-09-19, "mechanical
quote gate to production"). The two functions below (`_quote_supported_
by_text`, exact containment; `_fuzzy_quote_containment`, a tolerant
token-subsequence score) were written for, and are still used by, the
bench's own `_evidence_basis` -- they decide its `exact_quote`/`fuzzy_
quote` verdicts, the two strictest of its five-way evidence classification.
The production campaign's pick (`01-spec/extraction-final-2026-09-18.md`:
"Nova Lite, five samples, v6 prompt, mechanical quote gate, human
review") needs that exact same rule enforced live, before a quote-
bearing fact ever reaches merge -- not a new, second definition of
"supported" that could quietly drift from the one the bench scores
against. This module is now the one implementation both paths import:
`portolan.bench.extraction` (unchanged, byte-for-byte -- it re-exports
these same names) and `portolan.pipeline._finish_one` (via `quote_is_
supported`, the new public entry point below: a quote-bearing fact it
returns `False` for is dropped from merge and parked in `review_queue`
instead -- see `_finish_one`'s own docstring for exactly where).

`quote_is_supported` deliberately stops at containment. It does NOT run
the bench's fragment/window fallback (`_fragment_vote`/`_window_
supported`, `_salient_claim_fragments`, `EvidenceContext` -- all still
bench-only, unmoved). Those exist to avoid double-penalizing a model for
loose paraphrase when SCORING how "invented" a fact looks after the
fact, a softer question than the one this module answers. The gate's job
is narrower and stricter, and it is rule 5's own job, not scoring's: a
quote-bearing fact either has its own quoted words in the source or it
doesn't.
"""

from __future__ import annotations

import re
from bisect import bisect_left
from typing import NamedTuple

from portolan.util.prompt_safety import INVISIBLE_RE, strip_invisible

__all__ = [
    "FUZZY_QUOTE_THRESHOLD",
    "QuoteMatch",
    "locate_quote",
    "quote_is_supported",
]

# Glyph variants a model routinely substitutes for the source's own ASCII
# punctuation when it re-types rather than copy-pastes a quote (curly
# apostrophes/quotes, en/em dashes, a single-character ellipsis) -- none of
# these change what the quote *says*, so both the exact-containment check
# and the fuzzy tokenizer fold them to one canonical form first. 2026-09-18
# audit (`scripts/bench_audit_invented.py`, `01-spec/model-bench-2026-09-18.md`
# "Corrected scoring"): this was the single largest cause of a textually-
# supported quote failing containment and the fact being mislabeled
# `invented`.
_GLYPH_FOLD_TRANSLATION = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "“": '"',
        "”": '"',
        "–": "-",
        "—": "-",
        "…": "...",
    }
)


def _fold_glyphs(s: str) -> str:
    return s.translate(_GLYPH_FOLD_TRANSLATION)


def _normalize_for_containment(s: str) -> str:
    return re.sub(r"\s+", " ", _fold_glyphs(s).strip().lower())


def _quote_supported_by_text(quote: str | None, text: str) -> bool | None:
    """True/False when `quote` is given (rule 5's verbatim excerpt,
    required for the five evidentiary types); None when there's no quote to
    check, so the caller falls back to fuzzy containment and then the
    salient-fragment heuristic. Glyph-folded (curly quotes, dashes, an
    ellipsis character) before the substring check, but otherwise still
    exact -- word choice, word order and dropped words are NOT tolerated
    here; that's `_fuzzy_quote_containment`'s job."""
    if not quote or not quote.strip():
        return None
    return _normalize_for_containment(quote) in _normalize_for_containment(text)


# `_plausible_in_source`'s fuzzy-quote signal: what fraction of the quote's
# own normalized tokens are found, in order (skipping tokens that aren't --
# a changed word, a dropped filler word, a paraphrase seam), somewhere in
# the source text. Calibrated by hand-checking (`01-spec/model-bench-
# 2026-09-18.md`'s "Corrected scoring" section, 20 items) every quote-
# bearing fact the old classifier labeled `invented` in the two
# 2026-09-18-available runs on disk (`var/bench/company-context-
# 2026-09-17/{with,without}` -- see that section for why today's three
# model-comparison runs' own raw output was not available to check
# instead) plus one synthetic fully-fabricated control: all 10 real
# quote-bearing disputed items, read side by side with their source
# passage, were judged genuinely supported by a human read and scored
# 0.828-1.0; the synthetic fabricated quote (real words, wrong claim,
# stitched from scattered unrelated parts of the transcript) scored 0.467.
# 0.75 sits in that gap.
FUZZY_QUOTE_THRESHOLD = 0.75
# The bench's own private name for the same constant, kept so `portolan.
# bench.extraction` can import it unchanged and every reference inside
# that module (`_evidence_basis`, tests) keeps working without a rename.
_FUZZY_QUOTE_THRESHOLD = FUZZY_QUOTE_THRESHOLD

_QUOTE_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def _quote_tokens(s: str) -> list[str]:
    """Lowercase, glyph-folded word tokens for the fuzzy containment check
    -- deliberately simpler than `_normalize_claim_tokens` (no number-word
    folding, no `$`/`%` retention): a quote is copied prose, not a claim's
    own durable-language figures, and stray punctuation (an ellipsis, a
    dash) should just vanish rather than become its own token."""
    return _QUOTE_TOKEN_RE.findall(_fold_glyphs(s).lower())


def _fuzzy_quote_containment(quote: str, text: str) -> float:
    """Fraction of `quote`'s tokens matched, in order, as a (not
    necessarily contiguous) subsequence of `text`'s tokens -- a greedy
    left-to-right scan: for each quote token, find its next occurrence at
    or after the current text position; on a hit, advance past it and
    count it matched, on a miss, leave the position alone and move to the
    next quote token. This tolerates exactly the near-verbatim drift a
    model introduces when it re-types instead of copy-pasting (one
    substituted word, a dropped filler word, a paraphrase seam bridged by
    "...") without tolerating word-order shuffling or a quote assembled
    from unrelated parts of the text (those still score low: only
    forward-scanning, never backward)."""
    quote_tokens = _quote_tokens(quote)
    if not quote_tokens:
        return 0.0
    text_tokens = _quote_tokens(text)
    pos = 0
    matched = 0
    for qt in quote_tokens:
        try:
            idx = text_tokens.index(qt, pos)
        except ValueError:
            continue
        matched += 1
        pos = idx + 1
    return matched / len(quote_tokens)


# ---------------------------------------------------------------------------
# Locating a quote: where in the text it sits, not only whether it is there.
#
# `_fuzzy_quote_containment` scans the whole text, so on a long source it
# passes almost any sentence made of common words: the tokens of "we are
# going to cancel the contract with you next month because the price is too
# high" are found in order somewhere across a 20,000-word transcript (score
# 0.88) although nobody said it. `locate_quote` requires the matched tokens
# to sit together: one bounded window, with a cap on the gap between
# neighbours. That is what "near-verbatim" meant in the first place.
# ---------------------------------------------------------------------------

# 1:1 character folds only, so a token's offsets in the folded text are its
# offsets in the original.
_OFFSET_SAFE_FOLD = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "“": '"',
        "”": '"',
        "–": "-",
        "—": "-",
        "…": " ",
        # `portolan.util.prompt_safety` swaps these in for tag lookalikes in the
        # prompt copy, so a quote typed from that copy still finds the raw text.
        "‹": "<",
        "›": ">",
    }
)
# Tokens the window may hold beyond the quote's own: room for dropped filler
# words and one "..." seam, not for a second sentence from elsewhere.
_WINDOW_SLACK_TOKENS = 20
_MAX_TOKEN_GAP = 15
_MAX_ANCHORS_PER_TOKEN = 400


class QuoteMatch(NamedTuple):
    """Where a quote sits in a text: `text[start:end]` is the source's own
    wording. `exact` is glyph- and whitespace-folded containment; otherwise
    `score` is the share of the quote's tokens found, in order, inside one
    window."""

    start: int
    end: int
    exact: bool
    score: float


def _tokens_with_offsets(text: str) -> tuple[list[str], list[tuple[int, int]]]:
    folded = text.translate(_OFFSET_SAFE_FOLD).lower()
    tokens: list[str] = []
    offsets: list[tuple[int, int]] = []
    for match in _QUOTE_TOKEN_RE.finditer(folded):
        tokens.append(match.group(0))
        offsets.append((match.start(), match.end()))
    return tokens, offsets


def _locate_exact(quote: str, text: str) -> tuple[int, int] | None:
    """Offsets in `text` of `quote`, compared the way
    `_normalize_for_containment` does (glyphs folded, case and runs of
    whitespace ignored)."""
    wanted = _normalize_for_containment(strip_invisible(quote).translate(_OFFSET_SAFE_FOLD))
    if not wanted:
        return None
    folded = text.translate(_OFFSET_SAFE_FOLD).lower()
    # Collapse whitespace while remembering where each kept character was.
    # Invisible characters in the source are skipped: the prompt copy never
    # showed them, so a quote typed from it does not hold them.
    chars: list[str] = []
    origin: list[int] = []
    previous_space = True
    for index, char in enumerate(folded):
        if INVISIBLE_RE.match(char):
            continue
        if char.isspace():
            if previous_space:
                continue
            chars.append(" ")
            origin.append(index)
            previous_space = True
        else:
            chars.append(char)
            origin.append(index)
            previous_space = False
    found = "".join(chars).find(wanted)
    if found == -1:
        return None
    return origin[found], origin[found + len(wanted) - 1] + 1


def locate_quote(quote: str | None, text: str) -> QuoteMatch | None:
    """The best place `quote` sits in `text`, or `None` when it is neither
    there verbatim nor near-verbatim (module comment above).

    Near-verbatim: at least `FUZZY_QUOTE_THRESHOLD` of the quote's tokens,
    in order, inside one window of the quote's length plus
    `_WINDOW_SLACK_TOKENS`, no two neighbours more than `_MAX_TOKEN_GAP`
    tokens apart."""
    if not text or not quote or not quote.strip():
        return None
    exact = _locate_exact(quote, text)
    if exact is not None:
        return QuoteMatch(exact[0], exact[1], True, 1.0)

    quote_tokens = _quote_tokens(quote)
    if not quote_tokens:
        return None
    text_tokens, offsets = _tokens_with_offsets(text)
    positions: dict[str, list[int]] = {}
    for index, token in enumerate(text_tokens):
        positions.setdefault(token, []).append(index)

    needed = FUZZY_QUOTE_THRESHOLD * len(quote_tokens)
    if sum(1 for token in quote_tokens if token in positions) < needed:
        return None

    window = len(quote_tokens) + _WINDOW_SLACK_TOKENS
    # Anchor on the quote's rarest words: a real quote's window holds them.
    present = sorted({t for t in quote_tokens if t in positions}, key=lambda t: len(positions[t]))
    best: tuple[int, int, int] | None = None  # (matched, first index, last index)
    seen_starts: set[int] = set()
    for anchor_token in present[:5]:
        first_use = quote_tokens.index(anchor_token)
        for anchor in positions[anchor_token][:_MAX_ANCHORS_PER_TOKEN]:
            start = max(0, anchor - first_use - _WINDOW_SLACK_TOKENS // 2)
            if start in seen_starts:
                continue
            seen_starts.add(start)
            limit = start + window
            cursor = start
            matched = 0
            first = last = -1
            for token in quote_tokens:
                where = positions.get(token)
                if not where:
                    continue
                slot = bisect_left(where, cursor)
                if slot == len(where):
                    continue
                index = where[slot]
                if index >= limit or (last != -1 and index - last > _MAX_TOKEN_GAP):
                    continue
                matched += 1
                if first == -1:
                    first = index
                last = index
                cursor = index + 1
            if matched and (best is None or matched > best[0]):
                best = (matched, first, last)
                if matched == len(quote_tokens):
                    break
    if best is None or best[0] < needed:
        return None
    matched, first, last = best
    return QuoteMatch(offsets[first][0], offsets[last][1], False, matched / len(quote_tokens))


def quote_is_supported(quote: str | None, text: str) -> bool:
    """The gate: `True` when `quote` is verbatim (glyph-folded exact
    containment) or near-verbatim (`locate_quote`: enough of its tokens, in
    order, together in one place) in `text`; `False` for a missing/blank
    quote, empty `text`, or a quote that clears neither bar.

    Stricter than the bench's `fuzzy_quote` verdict, which still scores
    with the whole-text `_fuzzy_quote_containment`: on a long source that
    one passes sentences nobody said (see the comment above `locate_quote`). Callers must only call
    this for a fact whose type actually requires a quote (`portolan.
    extract.schema.QUOTE_REQUIRED_TYPES`); it has no opinion on whether a
    quote was required in the first place."""
    return locate_quote(quote, text) is not None
