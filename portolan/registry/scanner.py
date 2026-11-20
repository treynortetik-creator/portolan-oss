"""A fast, case-aware multi-pattern scanner over registry names and aliases.

Compiled regex alternation with word boundaries, not an Aho-Corasick
automaton: two precompiled alternations, one case-insensitive and one
case-sensitive, built once per `Registry.load()` and reused for every
`scan()` call. Speed is pinned by
`tests/test_registry_scan.py::test_scan_is_fast_against_the_longest_real_call_transcript`
and `tests/test_registry.py::test_scan_is_fast_on_a_20k_character_text`.

Two patterns of the same text can legitimately point at different
entities (e.g. a shared brand alias registered on more than one product),
so a single text span can produce more than one `Match`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Names and aliases of at most this many characters match with exact case only ("T2",
# "AMC", "NGH" must not fire on stray lowercase words); longer ones match case-insensitively.
# The same rule applies to `scan()` and to `Registry.resolve()`'s alias tier.
SHORT_PATTERN_MAX_LEN = 3


@dataclass(frozen=True)
class PatternEntry:
    """One name or alias, and the entity it points back to."""

    text: str
    entity_id: str
    kind: str
    canonical_name: str


@dataclass
class CompiledScanner:
    ci_pattern: re.Pattern[str] | None
    cs_pattern: re.Pattern[str] | None
    ci_index: dict[str, list[PatternEntry]] = field(default_factory=dict)
    cs_index: dict[str, list[PatternEntry]] = field(default_factory=dict)


def _compile_alternation(texts: set[str], flags: int) -> re.Pattern[str] | None:
    if not texts:
        return None
    # Longest-first ordering so that, at a single starting position, the
    # regex engine's left-to-right alternation prefers the longer
    # alternative (e.g. "Larkspur Memorial" tried before "Larkspur").
    # Overlap across *different* starting positions is still resolved
    # afterwards in `scan_text`.
    ordered = sorted(texts, key=len, reverse=True)
    body = "|".join(re.escape(t) for t in ordered)
    return re.compile(rf"\b(?:{body})\b", flags)


def build_scanner(entries: list[PatternEntry]) -> CompiledScanner:
    """Compile every pattern once, when `Registry.load()` builds its indexes;
    never per `scan()` call."""
    ci_entries = [e for e in entries if len(e.text) > SHORT_PATTERN_MAX_LEN]
    cs_entries = [e for e in entries if len(e.text) <= SHORT_PATTERN_MAX_LEN]

    ci_index: dict[str, list[PatternEntry]] = {}
    for e in ci_entries:
        ci_index.setdefault(e.text.lower(), []).append(e)

    cs_index: dict[str, list[PatternEntry]] = {}
    for e in cs_entries:
        cs_index.setdefault(e.text, []).append(e)

    return CompiledScanner(
        ci_pattern=_compile_alternation(set(ci_index), re.IGNORECASE),
        cs_pattern=_compile_alternation(set(cs_index), 0),
        ci_index=ci_index,
        cs_index=cs_index,
    )


RawHit = tuple[int, int, str, list[PatternEntry]]


def scan_text(scanner: CompiledScanner, text: str) -> list[RawHit]:
    """Every non-overlapping hit, longest match wins, left to right."""
    raw: list[RawHit] = []
    if scanner.ci_pattern is not None:
        for m in scanner.ci_pattern.finditer(text):
            entries = scanner.ci_index.get(m.group(0).lower(), [])
            raw.append((m.start(), m.end(), m.group(0), entries))
    if scanner.cs_pattern is not None:
        for m in scanner.cs_pattern.finditer(text):
            entries = scanner.cs_index.get(m.group(0), [])
            raw.append((m.start(), m.end(), m.group(0), entries))

    # Longest match wins: sort by span length descending, then by start
    # position, and greedily accept hits that don't overlap one already
    # accepted.
    raw.sort(key=lambda h: (-(h[1] - h[0]), h[0]))
    accepted: list[RawHit] = []
    occupied: list[tuple[int, int]] = []
    for hit in raw:
        start, end, _text, _entries = hit
        if any(start < e and s < end for s, e in occupied):
            continue
        occupied.append((start, end))
        accepted.append(hit)

    accepted.sort(key=lambda h: h[0])
    return accepted
