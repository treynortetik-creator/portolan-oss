"""Mechanical keep gate for silver-labeled facts (`--mode single`).

A teacher's fact is kept only when, checked against the full record `text`
(the student sees the whole record, so a name or number from elsewhere in
the same record is grounded):

1. `quote_is_supported`: the quote is in the text.
2. Numerals: every number in the claim appears in the text, compared by
   value (`unsupported_numerals`, the shared normalizer in
   `portolan.extract.numerals`, whose docstring lists the equivalences).
3. Names: every capitalized word in the claim, other than its first word
   and generic role words, appears in the text (`unsupported_names`). A
   trailing legal or corporate suffix (Bank, Inc, LLC, Corp) is optional
   when a distinctive word of the same name is in the text ("Comerica
   Bank" is supported by "Comerica"; "Chase Bank" is not). Possessives,
   hyphenated words ("Fiat-led"; an identifier such as "FTS-2" stays
   whole), plural acronyms ("ATMs"), and a name the text writes as one
   word or two ("CareLink" / "care link") compare equal.
4. Content-word coverage, off by default (`scripts/data/silver_label.py`
   `DEFAULT_OVERLAP_THRESHOLD` is 0.0): the local quote-plus-sentence
   score clears `threshold`, or the full-text score clears
   `full_text_threshold`.

Tests with the real dropped examples and with invented numerals and names
that must still drop: `tests/test_training_overlap_gate.py`.
`sweep_thresholds`/`choose_threshold` are the calibration-table half
(`scripts/data/overlap_calibration.py`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from portolan.extract.numerals import STOPWORDS as _STOPWORDS
from portolan.extract.numerals import unsupported_numerals
from portolan.extract.quote_gate import _fold_glyphs, quote_is_supported

__all__ = [
    "DEFAULT_FULL_TEXT_THRESHOLD",
    "tokenize_claim",
    "ClaimTokens",
    "extract_surrounding_sentence",
    "content_word_score",
    "numerals_and_names_covered",
    "unsupported_numerals",
    "unsupported_names",
    "overlap_gate_reason",
    "passes_overlap_gate",
    "CalibrationCase",
    "ThresholdStats",
    "evaluate_threshold",
    "sweep_thresholds",
    "choose_threshold",
]

# Higher than a local `threshold`: the whole document is a bigger haystack
# for an invented claim to score well against by chance.
DEFAULT_FULL_TEXT_THRESHOLD = 0.8

_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\-]*")
_STARTS_WITH_DIGIT_RE = re.compile(r"^\d")

# A grouping comma ("$1,200") is removed so both spellings tokenize alike;
# an ordinary sentence comma ("pricing, but") has no digit group after it.
_THOUSANDS_COMMA_RE = re.compile(r"(?<=\d),(?=\d{3}(?:\D|$))")

# Role words name a function, not an entity, and appear in nearly any
# transcript, so they are never required as names.
_GENERIC_ROLE_WORDS = frozenset(
    {
        "customer",
        "prospect",
        "rep",
        "team",
        "company",
        "vendor",
        "user",
        "client",
        "buyer",
        "seller",
    }
)


def _strip_thousands_commas(text: str) -> str:
    return _THOUSANDS_COMMA_RE.sub("", text)


def _raw_words(text: str) -> list[str]:
    return _WORD_RE.findall(_strip_thousands_commas(_fold_glyphs(text or "")))


def _strip_possessive(word: str) -> str:
    """Strip a trailing possessive marker so a possessive form and its
    bare noun compare equal for the name-coverage check: `'s` ("Customer's"
    -> "Customer", "SmartOnboard's" -> "SmartOnboard") or a bare trailing
    `s'` ("the reps'" -> "the rep"). Glyph-folding (already applied
    before a word is tokenized, via `_fold_glyphs`) has already turned
    any curly apostrophe into the ASCII `'`, so only that form needs
    checking here. A word that is ONLY a possessive marker (`"'s"`
    alone) is returned unchanged rather than stripped to nothing."""
    lowered = word.lower()
    if len(word) > 2 and lowered.endswith("'s"):
        return word[:-2]
    if len(word) > 2 and lowered.endswith("s'"):
        return word[:-1]
    return word


@dataclass(frozen=True)
class ClaimTokens:
    """`claim`'s raw tokens: `content` (lowercased, stopwords removed; the
    content-coverage score), `numerals` (tokens starting with a digit) and
    `names` (capitalized tokens other than the first word and role words,
    possessive-stripped). The gate's numeral and name rules compare by
    value and by phrase instead: `unsupported_numerals`, `unsupported_names`."""

    content: tuple[str, ...]
    numerals: tuple[str, ...]
    names: tuple[str, ...]


def tokenize_claim(claim: str) -> ClaimTokens:
    content: list[str] = []
    numerals: list[str] = []
    names: list[str] = []
    for index, word in enumerate(_raw_words(claim)):
        lowered = word.lower()
        if _STARTS_WITH_DIGIT_RE.match(word):
            numerals.append(lowered)
            continue
        if lowered in _STOPWORDS:
            continue
        content.append(lowered)
        if not word[0].isupper():
            continue
        if index == 0:
            # Sentence-initial capitalization, not necessarily a proper
            # noun -- true regardless of whether this word opens a
            # longer proper-noun phrase (word 2+ of "Virtual Onboarding
            # Suite" still counts; only "Virtual" itself is exempted
            # when it opens the claim).
            continue
        stripped = _strip_possessive(word).lower()
        if stripped in _GENERIC_ROLE_WORDS:
            continue
        names.append(stripped)
    return ClaimTokens(content=tuple(content), numerals=tuple(numerals), names=tuple(names))


def _coverage_word_set(text: str) -> set[str]:
    return {w.lower() for w in _raw_words(text)}


def extract_surrounding_sentence(quote: str, doc_text: str) -> str:
    """The sentence(s) of `doc_text` that best overlap `quote`, unioned
    with `quote` itself -- a wider "coverage text" than the bare quote,
    standing in for "the quote plus the document's surrounding sentence"
    (the gate's own rule). Best-effort: `doc_text` is split on
    `.`/`!`/`?` boundaries and scored by raw token overlap with `quote`
    (not a real sentence tokenizer) -- good enough to widen the window
    without pulling in the whole document. Falls back to `quote` alone
    when either is empty."""
    if not quote or not doc_text:
        return quote or ""
    sentences = re.split(r"(?<=[.!?])\s+", doc_text)
    quote_tokens = _coverage_word_set(quote)
    if not quote_tokens:
        return quote
    best = ""
    best_overlap = -1
    for sentence in sentences:
        overlap = len(quote_tokens & _coverage_word_set(sentence))
        if overlap > best_overlap:
            best_overlap = overlap
            best = sentence
    return f"{best} {quote}" if best else quote


def content_word_score(claim: str, coverage_text: str) -> float:
    """Fraction of `claim`'s content words (stopwords removed) present in
    `coverage_text`. An empty `content` bucket scores 1.0 (nothing to
    fail to cover)."""
    claim_tokens = tokenize_claim(claim)
    if not claim_tokens.content:
        return 1.0
    coverage = _coverage_word_set(coverage_text)
    matched = sum(1 for t in claim_tokens.content if t in coverage)
    return matched / len(claim_tokens.content)


# --------------------------------------------------------------------------
# Names
# --------------------------------------------------------------------------

# Optional at the end of a name that keeps a distinctive word before it.
_ORG_SUFFIXES = frozenset(
    {
        "ag",
        "bancorp",
        "bank",
        "co",
        "company",
        "corp",
        "corporation",
        "gmbh",
        "group",
        "holdings",
        "inc",
        "incorporated",
        "limited",
        "llc",
        "llp",
        "lp",
        "ltd",
        "na",
        "plc",
        "pllc",
        "sa",
        "usa",
    }
)
_PLURAL_ACRONYM_RE = re.compile(r"^[A-Z]{2,}s$")


@dataclass
class _NamePart:
    surface: str
    token: str
    required: bool
    distinctive: bool
    trailing_suffix: bool = False


def _claim_name_phrases(claim: str) -> list[list[_NamePart]]:
    """Runs of capitalized words, split on hyphens, that sit next to each
    other in the claim with only whitespace between them."""
    folded = _strip_thousands_commas(_fold_glyphs(claim or ""))
    phrases: list[list[_NamePart]] = []
    current: list[_NamePart] = []
    previous_end: int | None = None
    for index, m in enumerate(_WORD_RE.finditer(folded)):
        adjacent = previous_end is not None and folded[previous_end : m.start()].isspace()
        previous_end = m.end()
        parts: list[_NamePart] = []
        word = m.group(0)
        # "Fiat-led" splits; an identifier such as "FTS-2" stays whole
        pieces = [word] if any(ch.isdigit() for ch in word) else word.split("-")
        for piece in pieces:
            if not piece or not piece[0].isupper():
                continue
            token = _strip_possessive(piece).lower()
            if token in _STOPWORDS:
                continue
            role = token in _GENERIC_ROLE_WORDS
            parts.append(
                _NamePart(
                    surface=piece,
                    token=token,
                    required=index != 0 and not role,
                    distinctive=not role and token not in _ORG_SUFFIXES,
                )
            )
        if current and (not parts or not adjacent):
            phrases.append(current)
            current = []
        current.extend(parts)
    if current:
        phrases.append(current)

    for phrase in phrases:
        for i in range(len(phrase) - 1, 0, -1):
            if phrase[i].token not in _ORG_SUFFIXES:
                break
            phrase[i].trailing_suffix = True
    return phrases


@dataclass(frozen=True)
class _DocumentNames:
    words: frozenset[str]
    joined_pairs: frozenset[str]


def _document_names(text: str) -> _DocumentNames:
    words: set[str] = set()
    sequence: list[str] = []
    for raw in _raw_words(text):
        token = _strip_possessive(raw).lower()
        sequence.append(token)
        words.add(token)
        if "-" in token:
            words.update(p for p in token.split("-") if p)
            words.add(token.replace("-", ""))
    joined = {a + b for a, b in zip(sequence, sequence[1:], strict=False)}
    return _DocumentNames(words=frozenset(words), joined_pairs=frozenset(joined))


def _name_part_supported(part: _NamePart, doc: _DocumentNames) -> bool:
    if part.token in doc.words or part.token in doc.joined_pairs:
        return True
    return bool(_PLURAL_ACRONYM_RE.match(part.surface)) and part.token[:-1] in doc.words


def unsupported_names(claim: str, text: str) -> list[str]:
    """Every required name word in `claim`, as written there, that `text`
    does not support."""
    doc = _document_names(text)
    missing: list[str] = []
    for phrase in _claim_name_phrases(claim):
        found = [_name_part_supported(p, doc) for p in phrase]
        # "Active Directory" is supported by "activedirectory"
        for i in range(len(phrase) - 1):
            if phrase[i].token + phrase[i + 1].token in doc.words:
                found[i] = found[i + 1] = True
        for i, part in enumerate(phrase):
            if not part.required or found[i]:
                continue
            # "Comerica Bank" is supported when the text has "Comerica"
            if part.trailing_suffix and any(found[j] and phrase[j].distinctive for j in range(i)):
                continue
            missing.append(part.surface)
    return missing


def numerals_and_names_covered(claim: str, coverage_text: str) -> tuple[bool, bool]:
    """`(numerals_all_covered, names_all_covered)` against `coverage_text`,
    which `overlap_gate_reason` always passes as the full record text."""
    return (
        not unsupported_numerals(claim, coverage_text),
        not unsupported_names(claim, coverage_text),
    )


def overlap_gate_reason(
    *,
    claim: str,
    quote: str | None,
    doc_text: str,
    threshold: float,
    full_text_threshold: float = DEFAULT_FULL_TEXT_THRESHOLD,
) -> str | None:
    """`None` when the gate passes; otherwise the first rule that failed:
    `"quote_gate"`, `"overlap_gate_numeral"`, `"overlap_gate_name"` or
    `"overlap_gate_content"` (the reasons `DroppedFact` logs)."""
    if not quote_is_supported(quote, doc_text):
        return "quote_gate"

    numerals_covered, names_covered = numerals_and_names_covered(claim, doc_text)
    if not numerals_covered:
        return "overlap_gate_numeral"
    if not names_covered:
        return "overlap_gate_name"

    coverage_text = extract_surrounding_sentence(quote or "", doc_text)
    local_score = content_word_score(claim, coverage_text)
    if local_score >= threshold:
        return None
    full_score = content_word_score(claim, doc_text)
    if full_score >= full_text_threshold:
        return None
    return "overlap_gate_content"


def passes_overlap_gate(
    *,
    claim: str,
    quote: str | None,
    doc_text: str,
    threshold: float,
    full_text_threshold: float = DEFAULT_FULL_TEXT_THRESHOLD,
) -> bool:
    """`overlap_gate_reason(...) is None`: the rules in the module docstring,
    in order."""
    return (
        overlap_gate_reason(
            claim=claim,
            quote=quote,
            doc_text=doc_text,
            threshold=threshold,
            full_text_threshold=full_text_threshold,
        )
        is None
    )


# --------------------------------------------------------------------------
# Calibration table (pure, generic -- no corpus/judge dependency)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CalibrationCase:
    """One calibration item: a `(claim, quote, doc_text)` triple with the
    label the caller wants grouped by (`"gold"`, `"wrong_type"`,
    `"quote_swapped"`, `"added_detail"`, or a test's own made-up label)
    and whether the gate is EXPECTED to keep it."""

    label: str
    claim: str
    quote: str
    doc_text: str
    expected_keep: bool


@dataclass(frozen=True)
class ThresholdStats:
    threshold: float
    gold_keep_rate: float
    corruption_drop_rate_by_label: dict[str, float] = field(default_factory=dict)
    overall_corruption_drop_rate: float = 0.0


def evaluate_threshold(cases: list[CalibrationCase], threshold: float) -> ThresholdStats:
    gold_cases = [c for c in cases if c.expected_keep]
    corruption_cases = [c for c in cases if not c.expected_keep]

    gold_kept = sum(
        1
        for c in gold_cases
        if passes_overlap_gate(
            claim=c.claim, quote=c.quote, doc_text=c.doc_text, threshold=threshold
        )
    )
    gold_keep_rate = (gold_kept / len(gold_cases)) if gold_cases else 1.0

    by_label: dict[str, list[CalibrationCase]] = {}
    for c in corruption_cases:
        by_label.setdefault(c.label, []).append(c)

    drop_rate_by_label: dict[str, float] = {}
    for label, group in by_label.items():
        dropped = sum(
            1
            for c in group
            if not passes_overlap_gate(
                claim=c.claim, quote=c.quote, doc_text=c.doc_text, threshold=threshold
            )
        )
        drop_rate_by_label[label] = dropped / len(group)

    overall_dropped = sum(
        1
        for c in corruption_cases
        if not passes_overlap_gate(
            claim=c.claim, quote=c.quote, doc_text=c.doc_text, threshold=threshold
        )
    )
    overall_rate = (overall_dropped / len(corruption_cases)) if corruption_cases else 0.0

    return ThresholdStats(
        threshold=threshold,
        gold_keep_rate=gold_keep_rate,
        corruption_drop_rate_by_label=drop_rate_by_label,
        overall_corruption_drop_rate=overall_rate,
    )


def sweep_thresholds(cases: list[CalibrationCase], thresholds: list[float]) -> list[ThresholdStats]:
    return [evaluate_threshold(cases, t) for t in thresholds]


def choose_threshold(
    rows: list[ThresholdStats], *, min_gold_keep_rate: float = 0.95
) -> ThresholdStats | None:
    """The coordinator's own rule: "pick the value that keeps at least 95
    percent of the 20 gold facts and drops the most corruptions" -- among
    thresholds clearing `min_gold_keep_rate`, the one with the highest
    overall corruption drop rate; ties broken toward the HIGHER threshold
    (stricter, so a tie doesn't silently prefer the looser of two
    equally-good options). `None` when no threshold clears the gold-keep
    floor at all."""
    eligible = [r for r in rows if r.gold_keep_rate >= min_gold_keep_rate]
    if not eligible:
        return None
    return max(eligible, key=lambda r: (r.overall_corruption_drop_rate, r.threshold))
