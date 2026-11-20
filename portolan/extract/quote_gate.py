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

__all__ = [
    "FUZZY_QUOTE_THRESHOLD",
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


def quote_is_supported(quote: str | None, text: str) -> bool:
    """The gate: `True` when `quote` is verbatim (glyph-folded exact
    containment) or near-verbatim (fuzzy token-subsequence containment at
    or above `FUZZY_QUOTE_THRESHOLD`) present in `text`; `False` for a
    missing/blank quote, empty `text`, or a quote that clears neither bar.

    This is exactly `_evidence_basis`'s first two checks (the bench's own
    `exact_quote`/`fuzzy_quote` verdicts), exposed standalone so a caller
    that only needs the yes/no gate -- not the bench's full five-way
    evidence classification -- doesn't have to import the bench module or
    build an `EvidenceContext` it has no use for. Callers must only call
    this for a fact whose type actually requires a quote (`portolan.
    extract.schema.QUOTE_REQUIRED_TYPES`); it has no opinion on whether a
    quote was required in the first place."""
    if not text or not quote or not quote.strip():
        return False
    if _quote_supported_by_text(quote, text):
        return True
    return _fuzzy_quote_containment(quote, text) >= FUZZY_QUOTE_THRESHOLD
