"""The numbers rule (`01-spec/decisions-log.md` "Numbers rule"): models never
compute a figure, only restate one the report's code computed.

`NumberFacts` holds every computed number under a name and the exact numeral
string the model may use. A prose caller hands the model
`NumberFacts.as_table()`, then runs `check_numerals` on the reply; a text
that fails is dropped whole and `queue_numeral_rejection` logs a
`review_queue` row naming the unsupplied numerals.

The allow-list: a candidate is a whitespace-delimited word containing a
digit, after citation markers, `[[wikilinks]]` and month-name date phrases
are removed and edge punctuation (incl. `$`, `%`, quotes) is stripped.

- dates (`2026-08-04`, `2026-08`, `09/30/2026`), times, and years 1900-2099
  are never flagged;
- identifiers (a letter before the first digit, or `-`/`#`/`:` between two
  alphanumeric runs: `A-1010`, `T-031#1`, `p3¶2`) and ordinals are never
  flagged;
- magnitudes (`$8.2M`, `450k`, `3x`, `12bn`) are checked whole, then by
  their numeric part; any other letter suffix is left unflagged;
- ranges (`3-4`) are checked as two endpoints;
- spelled-out counting words have no digit, so are never candidates.

Every other candidate must equal one of `facts`' renderings.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

import psycopg

from portolan.extract.numerals import unsupported_numerals
from portolan.notes.render import CITATION_RE, WIKILINK_RE
from portolan.report.util import citation_markers_in

__all__ = [
    "NUMERAL_NOT_SUPPLIED",
    "NumberFacts",
    "check_numerals",
    "queue_numeral_rejection",
    "unsupported_cited_numerals",
]

NUMERAL_NOT_SUPPLIED = "numeral_not_supplied"

# Stripped from a candidate's edges only, never its middle. Quote marks are
# included so a quoted, licensed number still matches.
_JUNK_LEADING = "([{$\"'‘’“”"
_JUNK_TRAILING = ")]}.,;:!?%\"'‘’“”"

_TIME_RE = re.compile(r"^\d{1,2}:\d{2}(:\d{2})?$")
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_YEAR_RE = re.compile(r"^(?:19|20)\d{2}$")
_HAS_LETTER_RE = re.compile(r"[A-Za-z]")
_HAS_DIGIT_RE = re.compile(r"\d")

# Dates are never numerals. A "Month Day[, Year]" or "Day Month [Year]"
# phrase is removed before splitting, so its day and year never become
# candidates.
_MONTH_NAMES = (
    r"Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?"
    r"|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?"
)
_MONTH_DAY_RE = re.compile(
    # `(?!\d)` stops `\d{1,2}` matching the "20" of "August 2026" as a day.
    rf"\b(?:{_MONTH_NAMES})\.?\s+\d{{1,2}}(?!\d)(?:st|nd|rd|th)?,?\s*(?:(?:19|20)\d{{2}}(?!\d))?"
    rf"|\b\d{{1,2}}(?!\d)(?:st|nd|rd|th)?\s+(?:{_MONTH_NAMES})\.?\s*(?:(?:19|20)\d{{2}}(?!\d))?",
    re.IGNORECASE,
)
# A short `YYYY-MM` month-year ("2026-08") and a US-style slash date
# ("09/30/2026") -- neither is `_ISO_DATE_RE`'s full `YYYY-MM-DD`.
_MONTH_YEAR_RE = re.compile(r"^(?:19|20)\d{2}-(?:0[1-9]|1[0-2])$")
_SLASH_DATE_RE = re.compile(r"^\d{1,2}/\d{1,2}/(?:19|20)\d{2}$")
# A hyphen/en-dash/em-dash numeric range ("3-4", "3–4"): not a single
# numeral at all, so it's checked as its two endpoints, each against
# `facts` on its own (see `check_numerals`).
_RANGE_RE = re.compile(r"^([+-]?\d[\d,]*(?:\.\d+)?)[-–—](\d[\d,]*(?:\.\d+)?)$")

# A word with both letters and digits is an ordinal, an identifier, a
# magnitude, or (unrecognised) left unflagged.
_ORDINAL_RE = re.compile(r"^\d+(st|nd|rd|th)$", re.IGNORECASE)
# An id/locator joins letters and digits either by putting a letter
# before the first digit (`A-1010`, `T-031#1`, `p3¶2`) or by threading a
# `-`/`#`/`:` between two alphanumeric runs (`_INTERNAL_ID_SEP_RE`) --
# `_is_identifier` below checks both.
_INTERNAL_ID_SEP_RE = re.compile(r"[A-Za-z0-9][-#:][A-Za-z0-9]")
# A magnitude: a number followed by exactly one of k, m, bn, x.
_MAGNITUDE_RE = re.compile(r"^([+-]?\d[\d,]*(?:\.\d+)?)(k|m|bn|x)$", re.IGNORECASE)


def _is_identifier(word: str) -> bool:
    """A letter before the first digit, or an internal `-`/`#`/`:` joining
    two alphanumeric runs -- see `_INTERNAL_ID_SEP_RE`'s comment. Only
    called on a word already known to contain both a letter and a digit."""
    first_digit = _HAS_DIGIT_RE.search(word)
    if first_digit and _HAS_LETTER_RE.search(word[: first_digit.start()]):
        return True
    return bool(_INTERNAL_ID_SEP_RE.search(word))


class NumberFacts:
    """An ordered table of `{name: (value, rendering)}`. `rendering` is the
    exact numeral the model may copy, always bare (no `$` or `%`, which
    `check_numerals` strips from candidates). It defaults to `str(value)`;
    pass it explicitly when the prompt shows the number formatted (`1,420,000`).
    """

    def __init__(self) -> None:
        self._facts: dict[str, tuple[Any, str]] = {}
        # Which citation markers license each rendering; `None` in the set
        # means licensed anywhere (`add()`, or `harvest()` without a marker).
        self._licenses: dict[str, set[tuple[str, str] | None]] = {}

    def add(self, name: str, value: Any, rendering: str | None = None) -> str:
        """Record one computed figure under `name`, licensed anywhere; returns
        the rendering so the prompt text can use the same string."""
        text = rendering if rendering is not None else str(value)
        self._facts[name] = (value, text)
        self._licenses.setdefault(text, set()).add(None)
        return text

    def __contains__(self, name: str) -> bool:
        return name in self._facts

    def __len__(self) -> int:
        return len(self._facts)

    def __bool__(self) -> bool:
        return bool(self._facts)

    def renderings(self) -> set[str]:
        """Every distinct numeral rendering recorded so far."""
        return {rendering for _, rendering in self._facts.values()}

    def as_table(self) -> str:
        """One `- name: rendering` line per fact, in insertion order. An empty
        table says explicitly that no number may be stated."""
        if not self._facts:
            return "(no figures were computed for this text -- state no numbers at all)"
        return "\n".join(
            f"- {name}: {rendering}" for name, (_value, rendering) in self._facts.items()
        )

    def harvest(self, text: str, *, marker: tuple[str, str] | None = None) -> NumberFacts:
        """Register every numeral already in `text` (the `_candidate_numerals`
        scan) as an allowed rendering, so the model may restate context it
        was shown; returns `self`. Computed aggregates still go in via `add()`.

        With `marker` (a claim's `(source_slug, locator)`), those numerals are
        licensed only in a text that cites that marker, so one claim's
        number cannot be restated under another citation. Without it they
        are licensed anywhere. A rendering licensed anywhere by any call
        stays licensed anywhere.
        """
        for word in _candidate_numerals(text):
            self._facts[f"_harvested:{len(self._facts)}"] = (word, word)
            self._licenses.setdefault(word, set()).add(marker)
        return self

    def is_licensed(self, rendering: str, present_markers: set[tuple[str, str]]) -> bool:
        """Whether `rendering` may appear in a text citing
        `present_markers`: licensed anywhere, or harvested under one of them."""
        licenses = self._licenses.get(rendering)
        if not licenses:
            return False
        return None in licenses or bool(licenses & present_markers)

    def items(self):
        return self._facts.items()


def _strip_junk(word: str) -> str:
    while word and word[0] in _JUNK_LEADING:
        word = word[1:]
    while word and word[-1] in _JUNK_TRAILING:
        word = word[:-1]
    return word


def _candidate_numerals(text: str) -> list[str]:
    """Every word in `text` the module docstring's allow-list does not
    exempt. Shared by `check_numerals` and `NumberFacts.harvest`."""
    stripped = CITATION_RE.sub(" ", text)
    stripped = WIKILINK_RE.sub(" ", stripped)
    stripped = _MONTH_DAY_RE.sub(" ", stripped)

    found: list[str] = []
    for raw_word in stripped.split():
        word = _strip_junk(raw_word)
        if not word or not _HAS_DIGIT_RE.search(word):
            continue  # no digit at all -- not a numeral (includes every spelled-out counting word)
        if _HAS_LETTER_RE.search(word):
            if _ORDINAL_RE.match(word):
                continue  # 21st, 3rd -- an ordinal, not a quantity
            if _is_identifier(word):
                continue  # an id/locator, e.g. A-1010, T-031#1, p3¶2
            if not _MAGNITUDE_RE.match(word):
                continue  # an unrecognised letter suffix -- left unflagged
            # A magnitude stays a candidate, suffix and all.
        elif (
            _TIME_RE.match(word)
            or _ISO_DATE_RE.match(word)
            or _YEAR_RE.match(word)
            or _MONTH_YEAR_RE.match(word)  # "2026-08"
            or _SLASH_DATE_RE.match(word)  # "09/30/2026"
        ):
            continue
        found.append(word)
    return found


def check_numerals(
    text: str,
    facts: NumberFacts,
    *,
    present_markers: set[tuple[str, str]] | None = None,
) -> list[str]:
    """Every candidate numeral in `text` (module docstring's allow-list) that
    is not a licensed rendering in `facts`, in order, repeats included;
    empty means `text` passes.

    A rendering harvested with a marker counts only if `text` cites
    that marker. `present_markers` supplies those markers when the caller's
    citations are not inline (`None` scans `text` for `^[source @ locator]`).
    A magnitude passes on its whole string or on its numeric part matching a
    bare rendering (commas ignored); a range needs both endpoints.
    """
    allowed = facts.renderings()
    allowed_bare = {rendering.replace(",", "") for rendering in allowed}
    if present_markers is None:
        present_markers = citation_markers_in(text)
    violations = []
    for word in _candidate_numerals(text):
        if word in allowed and facts.is_licensed(word, present_markers):
            continue
        magnitude = _MAGNITUDE_RE.match(word)
        if magnitude is not None:
            numeric_part = magnitude.group(1).replace(",", "")
            if numeric_part in allowed_bare and any(
                r.replace(",", "") == numeric_part and facts.is_licensed(r, present_markers)
                for r in allowed
            ):
                continue
        range_match = _RANGE_RE.match(word)
        if range_match is not None:
            lo, hi = (g.replace(",", "") for g in range_match.groups())
            if lo in allowed_bare and hi in allowed_bare:
                both_licensed = all(
                    any(
                        r.replace(",", "") == part and facts.is_licensed(r, present_markers)
                        for r in allowed
                    )
                    for part in (lo, hi)
                )
                if both_licensed:
                    continue
        violations.append(word)
    return violations


# A number joined to a unit by a hyphen ("5-year", "30-day") is a quantity,
# not an identifier, even though `_INTERNAL_ID_SEP_RE` matches it.
_NUMBER_UNIT_RE = re.compile(r"^[+-]?\d[\d,]*(?:\.\d+)?-[A-Za-z]+$")


def _exempt_from_value_check(word: str) -> bool:
    """`check_numerals`' allow-list for one whitespace word already stripped
    of edge punctuation: dates, times, years, ordinals and identifiers are
    never asserted quantities."""
    if _HAS_LETTER_RE.search(word):
        if _ORDINAL_RE.match(word):
            return True
        return _is_identifier(word) and not _NUMBER_UNIT_RE.match(word)
    return bool(
        _TIME_RE.match(word)
        or _ISO_DATE_RE.match(word)
        or _YEAR_RE.match(word)
        or _MONTH_YEAR_RE.match(word)
        or _SLASH_DATE_RE.match(word)
    )


def unsupported_cited_numerals(text: str, cited_texts: Sequence[str]) -> list[str]:
    """Every number `text` states that none of `cited_texts` states, compared
    by value with the shared normalizer (`portolan.extract.numerals`): "250
    ms" is supported by "250ms", "20 to 40" by "20–40", "$5,000" by "$5k",
    "about 200" by "~200"; "300 ms", "20–50" and "$5,500" are not, nor is
    "$3M" by a count of 3 or "37-percent" by a count of 37
    (`tests/test_report_numbers_by_value.py`).

    The chat answer's numbers guard (`portolan.web.answer`). The allow-list
    is `check_numerals`': citation markers, `[[wikilinks]]`, month-name
    dates, ISO/slash dates, times, years, ordinals and identifiers are
    blanked out before the check, so they are never required. Unlike
    `check_numerals`, a number glued to an unrecognized unit ("900ms") is
    checked, not skipped. Returns the offending numbers as the normalizer
    wrote them; empty means `text` passes.
    """
    stripped = CITATION_RE.sub(" ", text)
    stripped = WIKILINK_RE.sub(" ", stripped)
    stripped = _MONTH_DAY_RE.sub(" ", stripped)
    kept = [
        word
        for word in stripped.split()
        if not (_HAS_DIGIT_RE.search(word) and _exempt_from_value_check(_strip_junk(word)))
    ]
    return unsupported_numerals(" ".join(kept), "\n".join(cited_texts))


def queue_numeral_rejection(
    conn: psycopg.Connection,
    *,
    note_slug: str | None,
    surface: str,
    violations: list[str],
    text: str | None = None,
    facts: NumberFacts | None = None,
) -> None:
    """One `review_queue` row (`kind='other'`) for a text `check_numerals`
    rejected. `surface` names the call site (`"headline"`,
    `"account_synthesis"`, `"theme_synthesis"`, `"theme_name"`,
    `"standing_question"`). `text` and `facts.as_table()`, when given, are
    saved so a reviewer can see what was rejected and what was allowed."""
    payload = {
        "reason": NUMERAL_NOT_SUPPLIED,
        "surface": surface,
        "note_slug": note_slug,
        "detail": f"numeral(s) not in the supplied figures: {', '.join(violations)}",
        "violations": violations,
        "text": text,
        "allowed_figures": facts.as_table() if facts is not None else None,
    }
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO review_queue (kind, payload) VALUES ('other', %s::jsonb)",
            (json.dumps(payload),),
        )
    conn.commit()
