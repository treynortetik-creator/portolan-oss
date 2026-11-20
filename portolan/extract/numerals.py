"""Numbers compared by value: is every number a claim states also stated
by a text? One normalizer for every gate that asks, so a rule fixed for one
is fixed for all: the silver-label overlap gate
(`portolan.training.overlap_gate`) and the chat answer's numbers guard
(`portolan.report.numbers.unsupported_cited_numerals`).

`unsupported_numerals(claim, text)` compares by value: thousands commas,
currency symbols, k/M/bn suffixes and scale words ("$5k" == "$5,000",
"$25M" == "$25 million"), leading zeros, a number glued to a unit
("250ms"), each end of a range ("20-40"), a spelled-out number in the text
("five"), and a claim ISO date matched against the same day written any
common way in the text. A hyphenated unit in the claim ("5-year") also
needs that number next to that unit in the text ("5 year", "5 years").
Scale and percent are part of the value: a claim's "$3M", "3M" or "3
million" needs 3,000,000, not a 3; "37%", "37 percent" and "37-percent"
are one percentage, which a count of 37 does not support, nor it a count.
Other units are not compared: "250 ms" and "250 s" both read 250.
Tests both ways (false drops that must pass, invented numbers that must
fail): `tests/test_training_overlap_gate_normalization.py`,
`tests/test_report_numbers_by_value.py`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal

from portolan.extract.quote_gate import _fold_glyphs

__all__ = ["STOPWORDS", "unsupported_numerals"]

# A short, deliberately generic stopword list. Here: a hyphenated word after
# a number ("5-to-10") is never read as that number's unit. The overlap gate
# also uses it for content words and names.
STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "then",
        "than",
        "so",
        "because",
        "as",
        "of",
        "to",
        "in",
        "on",
        "at",
        "by",
        "for",
        "with",
        "about",
        "against",
        "between",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "from",
        "up",
        "down",
        "out",
        "off",
        "over",
        "under",
        "again",
        "further",
        "once",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "having",
        "do",
        "does",
        "did",
        "doing",
        "will",
        "would",
        "shall",
        "should",
        "can",
        "could",
        "may",
        "might",
        "must",
        "that",
        "this",
        "these",
        "those",
        "it",
        "its",
        "their",
        "his",
        "her",
        "he",
        "she",
        "they",
        "we",
        "you",
        "i",
        "not",
        "no",
        "which",
        "who",
        "whom",
        "what",
        "when",
        "where",
        "why",
        "how",
        "there",
        "here",
        "also",
        "on-",
    }
)

_SMALL_NUMBER_WORDS: dict[str, int] = {
    word: i
    for i, word in enumerate(
        (
            "zero one two three four five six seven eight nine ten eleven twelve "
            "thirteen fourteen fifteen sixteen seventeen eighteen nineteen"
        ).split()
    )
}
_TENS_WORDS: dict[str, int] = {
    word: 10 * (i + 2)
    for i, word in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split())
}
_SCALE_WORDS: dict[str, int] = {
    "thousand": 10**3,
    "million": 10**6,
    "billion": 10**9,
    "trillion": 10**12,
}
_SUFFIX_SCALES: dict[str, int] = {
    "k": 10**3,
    "m": 10**6,
    "mm": 10**6,
    "mn": 10**6,
    "b": 10**9,
    "bn": 10**9,
}
# "5m" alone can be metres or minutes, "5b" a list item; only a currency
# sign makes them millions and billions for sure, so both readings count.
# A claim's capital "5M"/"5B" is read as the scale only: 3 million is not 3.
_AMBIGUOUS_SUFFIXES = frozenset({"m", "b"})
_CURRENCY_SIGNS = "$€£¥"
# A number followed by "%", "percent", "per cent" or "pct", or joined to one
# by a hyphen ("37-percent"), is a percentage: a percent is not a count, so
# each supports only its own kind.
_PERCENT_AFTER_RE = re.compile(r"[ \t]?%|[ \t]+(?:per[ \t]?cent|pct)\b", re.IGNORECASE)
_PERCENT_UNITS = frozenset({"percent", "pct"})
# The first end of a range whose last end is a percentage ("25-30%") reads
# either way.
_RANGE_TO_PERCENT_RE = re.compile(
    r"[ \t]*(?:-|to)[ \t]*\d[\d,]*(?:\.\d+)?(?:[ \t]?%|[ \t]+(?:per[ \t]?cent|pct)\b)",
    re.IGNORECASE,
)

# A number with an optional scale suffix ("5k", "$1.5MM"), a spaced suffix
# that counts only after a currency sign ("$2.5 MM"), a scale word ("3
# million"), or a hyphenated unit ("5-year"). A grouping comma needs
# exactly three digits after it, so "February 28,2001" is 28 and 2001.
_NUMBER_BODY = (
    r"(?P<num>\d{1,3}(?:,\d{3})+(?!\d)(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(?:(?P<suffix>[kK]|[mM][mMnN]?|[bB][nN]?)(?![A-Za-z0-9])"
    r"|\s(?P<spaced>[kK]|[mM][mMnN]?|[bB][nN]?)(?![A-Za-z0-9])"
    r"|\s+(?P<word>thousand|million|billion|trillion)\b)?"
    r"(?P<unit>-[A-Za-z]+)?"
)
_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9])" + _NUMBER_BODY, re.IGNORECASE)
# Digits glued after letters ("Q4", "B2B", "W10") belong to a name, not a
# numeral, on both sides, unless the letters are a currency code ("EUR1,200").
_CURRENCY_CODES = frozenset(
    {"AUD", "CAD", "CHF", "CNY", "EUR", "GBP", "INR", "JPY", "RMB", "RS", "USD"}
)
_GLUED_NUMBER_RE = re.compile(r"(?<![A-Za-z])(?P<letters>[A-Za-z]+)(?P<digits>\d[\d,.]*)")
_SPELLED_NUMBER_RE = re.compile(
    r"\b(?:(?P<tens>"
    + "|".join(_TENS_WORDS)
    + r")(?:[- ](?P<ones>"
    + "|".join(w for w, v in _SMALL_NUMBER_WORDS.items() if 0 < v < 10)
    + r"))?|(?P<small>"
    + "|".join(_SMALL_NUMBER_WORDS)
    + r"))\b",
    re.IGNORECASE,
)
_NEXT_WORD_RE = re.compile(r"[ \t-]{1,2}([A-Za-z]+)")
_TWO_DIGIT_YEAR_RE = re.compile(r"(?<=[/'])(\d{2})(?!\d)")

_MONTH = (
    r"(?P<mon>jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|"
    r"aug(?:ust)?|sept?(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?"
)
_MONTH_NUMBERS: dict[str, int] = {
    name: i + 1 for i, name in enumerate("jan feb mar apr may jun jul aug sep oct nov dec".split())
}
_ISO_DATE_RE = re.compile(r"(?<![\d-])(\d{4})-(\d{1,2})-(\d{1,2})(?![\d-])")
_SLASH_DATE_RE = re.compile(r"(?<![\d/])(\d{1,2})/(\d{1,2})(?:/(\d{4}|\d{2}))?(?![\d/])")
_MONTH_DAY_RE = re.compile(
    r"\b" + _MONTH + r"\s+(?P<day>\d{1,2})(?:st|nd|rd|th)?\b(?:,?\s*(?P<year>\d{4})\b)?",
    re.IGNORECASE,
)
_DAY_MONTH_RE = re.compile(
    r"\b(?P<day>\d{1,2})(?:st|nd|rd|th)?[\s-]+"
    + _MONTH
    + r"(?![a-z])(?:,?[\s-]+(?P<year>\d{4}|\d{2})\b)?",
    re.IGNORECASE,
)


def _canonical(value: Decimal) -> str:
    """One string per numeric value: "5", "5.0" and "05" all give "5". An
    integer goes through `int`, so a long digit run (an id, a hash) keeps
    every digit instead of hitting Decimal's 28-digit context."""
    if value == value.to_integral_value():
        return str(int(value))
    return format(value.normalize(), "f")


def _unit_stem(word: str) -> str:
    lowered = word.lower()
    return lowered[:-1] if len(lowered) > 3 and lowered.endswith("s") else lowered


@dataclass(frozen=True)
class _NumberMention:
    surface: str
    forms: frozenset[str]
    unit: str | None
    end: int


def _unglue_numbers(text: str) -> str:
    """Blank digits glued after letters, and the letters themselves; keep
    the digits after a currency code. Offsets are unchanged."""

    def unglue(m: re.Match[str]) -> str:
        letters = m.group("letters")
        if letters.upper() in _CURRENCY_CODES:
            return " " * len(letters) + m.group("digits")
        return " " * len(m.group(0))

    return _GLUED_NUMBER_RE.sub(unglue, text)


def _percent_forms(forms: set[str]) -> set[str]:
    return {f"{form}%" for form in forms}


def _number_mentions(text: str, *, in_claim: bool) -> list[_NumberMention]:
    """Numbers in `text`, which must already be unglued. `in_claim` makes a
    scale strict: a claim's "5 million" or "5M" needs 5,000,000 in the text,
    while the text's "5 million" also offers a bare 5, as it always has. A
    percentage's forms carry a "%" (`_PERCENT_AFTER_RE`)."""
    mentions: list[_NumberMention] = []
    for m in _NUMBER_RE.finditer(text):
        raw = Decimal(m.group("num").replace(",", ""))
        before = text[max(0, m.start() - 2) : m.start()].strip()
        has_currency = bool(before) and before[-1] in _CURRENCY_SIGNS
        suffix = (m.group("suffix") or "").lower()
        spaced = (m.group("spaced") or "").lower() if has_currency else ""
        word = (m.group("word") or "").lower()
        if word:
            scaled_word = _canonical(raw * _SCALE_WORDS[word])
            forms = {scaled_word} if in_claim else {scaled_word, _canonical(raw)}
        elif suffix or spaced:
            key = suffix or spaced
            scaled = _canonical(raw * _SUFFIX_SCALES[key])
            capital = in_claim and (m.group("suffix") or "").isupper()
            ambiguous = key in _AMBIGUOUS_SUFFIXES and not has_currency and not capital
            forms = {scaled, _canonical(raw)} if ambiguous else {scaled}
        else:
            forms = {_canonical(raw)}
        unit = m.group("unit")
        unit_stem = _unit_stem(unit[1:]) if unit else None
        end = m.end()
        percent = unit_stem in _PERCENT_UNITS
        if percent:
            unit_stem = None
        elif not unit and (after := _PERCENT_AFTER_RE.match(text, end)) is not None:
            percent, end = True, after.end()
        if unit_stem is not None and (len(unit_stem) < 3 or unit_stem in STOPWORDS):
            unit_stem = None
        if percent:
            forms = _percent_forms(forms)
        elif _RANGE_TO_PERCENT_RE.match(text, end):
            forms |= _percent_forms(forms)
        mentions.append(
            _NumberMention(
                surface=text[m.start() : end],
                forms=frozenset(forms),
                unit=unit_stem,
                end=m.end("num"),
            )
        )
    return mentions


def _spelled_values(text: str) -> list[tuple[int, int]]:
    """`(value, end offset)` for every spelled-out number from zero to
    ninety-nine, and its scaled value when a scale word follows ("five
    million")."""
    out: list[tuple[int, int]] = []
    for m in _SPELLED_NUMBER_RE.finditer(text):
        if m.group("small"):
            value = _SMALL_NUMBER_WORDS[m.group("small").lower()]
        else:
            value = _TENS_WORDS[m.group("tens").lower()]
            if m.group("ones"):
                value += _SMALL_NUMBER_WORDS[m.group("ones").lower()]
        out.append((value, m.end()))
        nxt = _NEXT_WORD_RE.match(text, m.end())
        if nxt and nxt.group(1).lower() in _SCALE_WORDS:
            out.append((value * _SCALE_WORDS[nxt.group(1).lower()], nxt.end()))
    return out


def _full_year(year: str) -> int:
    value = int(year)
    if len(year) == 2:
        return 2000 + value if value < 50 else 1900 + value
    return value


@dataclass(frozen=True)
class _DocumentNumbers:
    forms: frozenset[str]
    unit_pairs: frozenset[tuple[str, str]]
    full_dates: frozenset[tuple[int, int, int]]
    month_days: frozenset[tuple[int, int]]


def _document_numbers(text: str) -> _DocumentNumbers:
    folded = _fold_glyphs(text or "")
    forms: set[str] = set()
    unit_pairs: set[tuple[str, str]] = set()
    for mention in _number_mentions(_unglue_numbers(folded), in_claim=False):
        forms |= mention.forms
        nxt = _NEXT_WORD_RE.match(folded, mention.end)
        if nxt:
            unit_pairs.update((form, _unit_stem(nxt.group(1))) for form in mention.forms)
    for value, end in _spelled_values(folded):
        form = _canonical(Decimal(value))
        if _PERCENT_AFTER_RE.match(folded, end):
            form = f"{form}%"
        forms.add(form)
        nxt = _NEXT_WORD_RE.match(folded, end)
        if nxt:
            unit_pairs.add((form, _unit_stem(nxt.group(1))))
    # "04/98" and "'98" name a year a claim may write in full
    forms.update(str(_full_year(m.group(1))) for m in _TWO_DIGIT_YEAR_RE.finditer(folded))

    full_dates: set[tuple[int, int, int]] = set()
    month_days: set[tuple[int, int]] = set()

    def add_date(year: int | None, month: int, day: int) -> None:
        if not (1 <= month <= 12 and 1 <= day <= 31):
            return
        month_days.add((month, day))
        if year is not None:
            full_dates.add((year, month, day))

    for m in _ISO_DATE_RE.finditer(folded):
        add_date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    for m in _SLASH_DATE_RE.finditer(folded):
        add_date(_full_year(m.group(3)) if m.group(3) else None, int(m.group(1)), int(m.group(2)))
    for regex in (_MONTH_DAY_RE, _DAY_MONTH_RE):
        for m in regex.finditer(folded):
            year = _full_year(m.group("year")) if m.group("year") else None
            add_date(year, _MONTH_NUMBERS[m.group("mon").lower()[:3]], int(m.group("day")))
    return _DocumentNumbers(
        forms=frozenset(forms),
        unit_pairs=frozenset(unit_pairs),
        full_dates=frozenset(full_dates),
        month_days=frozenset(month_days),
    )


def _date_supported(year: int, month: int, day: int, doc: _DocumentNumbers) -> bool:
    if (year, month, day) in doc.full_dates:
        return True
    return (month, day) in doc.month_days and str(year) in doc.forms


def unsupported_numerals(claim: str, text: str) -> list[str]:
    """Every number in `claim`, as written there, that `text` does not
    support. A spelled-out number in the claim is not required."""
    doc = _document_numbers(text)
    missing: list[str] = []

    def blank_date(m: re.Match[str]) -> str:
        if not _date_supported(int(m.group(1)), int(m.group(2)), int(m.group(3)), doc):
            missing.append(m.group(0))
        return " " * len(m.group(0))

    remainder = _ISO_DATE_RE.sub(blank_date, _fold_glyphs(claim or ""))
    for mention in _number_mentions(_unglue_numbers(remainder), in_claim=True):
        if not (mention.forms & doc.forms) or (
            mention.unit is not None
            and not any((form, mention.unit) in doc.unit_pairs for form in mention.forms)
        ):
            missing.append(mention.surface)
    return missing
