"""Locator-window matching, shared by `extraction.py` (produced fact vs.
planted fact) and `graph.py` (a `claims` row vs. a manifest placement).

Window rules, per the wave B4 build instructions and
`01-spec/extraction-schema-v0.md` ("Segmenting long sources"):
- calls: timestamps, within 90 seconds.
- tickets: same comment (an exact match on the comment number).
- emails: same paragraph, +/- 1.
- RSS: same item, by URL (the appended date is ignored -- see `_rss_url`).
- CRM: exact.

The comparison is deliberately format-tolerant rather than a literal string
equals for every type: `corpus_loader.py` is a stand-in for the real
`portolan.ingest.prepare_text` and does not necessarily spell a locator
identically to `manifest.json`'s own answer-key string (e.g. a CRM row is
`crm/accounts.csv#A-1010` in the manifest but `account:A-1010` out of
`corpus_loader`) — both name the same row, so the comparison extracts the
identifying token rather than requiring byte-identical strings.

2026-09-18 harness review (`01-spec/harness-review-2026-09-18.md`): RSS used
to be a literal `planted.lower() == produced.lower()` compare. The truth
table's own planted locator is `url:<link> <date>` (`corpus_loader.py`'s
`[url:<link> <date>]` line, copied verbatim into `manifest.json`), but
`portolan/extract/prompt.py`'s own locator instructions tell the model the
RSS locator is "the `[url:<link> <date>]` line's own url" -- url only, no
date. A model following that instruction to the letter produced a locator
that could never pass the old exact-string rule: all three non-distractor
RSS facts in the corpus (F18, F29, F59) failed `locator` in at least one
run even when type/entities/claim all matched (`01-spec/
extraction-miss-analysis-2026-09-18.md` section 3). That is a scorer bug,
not an extractor error -- the model did what the prompt (owned by a
different wave, not editable here) told it to -- so the rule now compares
the URL alone, the same "extract the identifying token" treatment CRM
already gets above.
"""

from __future__ import annotations

import re

CALL_WINDOW_SECONDS = 90.0
EMAIL_PARAGRAPH_WINDOW = 1

_TIMESTAMP_RE = re.compile(r"(\d{1,2}):(\d{2}):(\d{2})")
_TRAILING_INT_RE = re.compile(r"(\d+)\s*$")


def _strip_enclosing_brackets(locator: str) -> str:
    """Drop one layer of enclosing `[...]`, if present, before parsing.

    Every locator format in this corpus is injected into the source text
    wrapped in brackets (`[00:05:20] Felix Arden: ...`, `[S-ticket-031#1]
    Felix Arden: ...`, `[¶2] ...`) -- the bracket is punctuation marking
    where the locator sits in the text, not part of the locator value
    itself (`extraction-schema-v0.md`'s own example fact has
    `"locator": "00:14:32"`, no brackets). Diagnosis
    (`var/bench/diag/diagnosis.md`) found a real model copying the
    brackets along with the digits often enough to be worth guarding
    mechanically here, on top of the explicit worked examples now in
    `extract/prompt.py` telling it not to: `parse_hhmmss`'s and
    `trailing_int`'s regexes are anchored to the *whole* (`^...$`) or
    *trailing* (`...$`) string, so a stray `]` fails them outright even
    when every digit is otherwise correct."""
    s = locator.strip()
    if len(s) >= 2 and s[0] == "[" and s[-1] == "]":
        return s[1:-1].strip()
    return s


def parse_hhmmss(locator: str) -> float | None:
    """Parse the *first* `HH:MM:SS` timestamp found anywhere in `locator` to
    seconds, or None if it contains no such pattern at all (a different
    source type's locator format).

    Deliberately a search, not a whole-string match: diagnosis
    (`var/bench/diag/diagnosis.md`) found a real model emitting a locator
    as a *range* over the segment it read from, e.g.
    `[00:04:30-00:04:43]` -- a reasonable reading of "when did this get
    said" for a fact synthesized from several lines, but not the single
    point-in-time value `01-spec/extraction-schema-v0.md`'s own example
    fact (`"locator": "00:14:32"`) and this corpus's answer key both use.
    Taking the first timestamp in the string (the start of the range, if
    it is one) recovers a comparable point in time instead of failing the
    whole locator on format alone."""
    m = _TIMESTAMP_RE.search(_strip_enclosing_brackets(locator))
    if not m:
        return None
    hours, minutes, seconds = (int(part) for part in m.groups())
    return hours * 3600 + minutes * 60 + seconds


def trailing_int(locator: str) -> int | None:
    """The trailing integer in a locator string: the comment number out of
    `S-ticket-019#1` or `1`, the paragraph number out of `S-email-...#p2`
    or `¶2`. None if the string has no trailing digits at all."""
    m = _TRAILING_INT_RE.search(_strip_enclosing_brackets(locator))
    return int(m.group(1)) if m else None


def _rss_url(locator: str) -> str:
    """The URL identifying an RSS/feed item: `locator` stripped of one
    layer of enclosing brackets, then of the `[url:<link> <date>]` line's
    trailing ` <date>` (if present) and leading `url:` prefix (if present),
    lowercased. Handles the truth table's own planted form (`url:<link>
    <date>`), a model that follows the prompt literally and emits just
    `<link>`, and anything in between (a stray `url:` prefix with no date,
    say) the same way."""
    s = _strip_enclosing_brackets(locator).strip()
    first = s.split()[0] if s.split() else s
    if first.lower().startswith("url:"):
        first = first[4:]
    return first.strip().lower()


def crm_key(locator: str) -> str:
    """The row-identifying token of a CRM locator: the part after the last
    `#` if present, else after the last `:`, else the whole (lowercased,
    stripped) string. Reduces both `crm/accounts.csv#A-1010` and
    `account:A-1010` to `a-1010`."""
    s = locator.strip()
    if "#" in s:
        s = s.rsplit("#", 1)[-1]
    elif ":" in s:
        s = s.rsplit(":", 1)[-1]
    return s.strip().lower()


def locator_matches(source_type: str, planted_locator: str, produced_locator: str) -> bool:
    """Does `produced_locator` land within the planted fact's window, given
    `source_type`'s matching rule?"""
    planted = planted_locator.strip()
    produced = produced_locator.strip()
    if not planted or not produced:
        return False

    if source_type == "call":
        a, b = parse_hhmmss(planted), parse_hhmmss(produced)
        if a is not None and b is not None:
            return abs(a - b) <= CALL_WINDOW_SECONDS
        return planted == produced

    if source_type == "ticket":
        a, b = trailing_int(planted), trailing_int(produced)
        if a is not None and b is not None:
            return a == b
        return planted == produced

    if source_type == "email":
        a, b = trailing_int(planted), trailing_int(produced)
        if a is not None and b is not None:
            return abs(a - b) <= EMAIL_PARAGRAPH_WINDOW
        return planted == produced

    if source_type == "rss":
        return _rss_url(planted) == _rss_url(produced)

    if source_type == "crm":
        return crm_key(planted) == crm_key(produced)

    # doc / chat / survey / anything else not yet given a rule: exact match.
    return planted == produced


def locator_matches_any(
    source_type: str, planted_locators: list[str], produced_locator: str
) -> bool:
    """`locator_matches` against any one of several acceptable planted
    locators (a fact planted more than once within the same source)."""
    return any(locator_matches(source_type, loc, produced_locator) for loc in planted_locators)
