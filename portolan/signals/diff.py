"""Section-by-section diff of two page snapshots, with noise ignored.

Paragraphs are compared on a normalised form: lower case, whitespace collapsed,
and the parts that change on every visit replaced by placeholders (dates,
times, "3 hours ago", "© 2026", counters such as "1,204 followers"). A page
whose only change is one of those produces no change record. A paragraph that
moved from one section to another is a move, not a change, and is dropped from
both sides. Prices and other plain numbers are kept: a changed price is exactly
what the judge should see.

`diff_snapshots` returns `None` when nothing meaningful changed, else a
`PageChange` whose `render()` is the text the judge reads and the change
record's original shows.
"""

from __future__ import annotations

import difflib
import hashlib
import re
from collections import Counter
from dataclasses import dataclass, field

from portolan.signals.html_text import Section

__all__ = [
    "MAX_CHANGE_CHARS",
    "PageChange",
    "SectionChange",
    "content_hash",
    "diff_snapshots",
    "normalize_paragraph",
]

_MONTHS = (
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|"
    r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?"
)
_WEEKDAYS = r"(?:mon|tue|tues|wed|thu|thur|thurs|fri|sat|sun)(?:day|sday|nesday|urday)?,?"
_NOISE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # "© 2026", "copyright 2025-2026"
    (re.compile(r"(?:©|&copy;|copyright)\s*\d{4}(?:\s*[-–]\s*\d{4})?"), " <copyright> "),
    # "last updated: ..." / "updated" stamps up to the end of the phrase
    (re.compile(r"(?:last\s+)?(?:updated|modified|reviewed)\s*(?:on|:)?\s*(?=\S*\d)"), " <upd> "),
    (re.compile(rf"\b{_WEEKDAYS}\s+"), " "),
    (re.compile(rf"\b{_MONTHS}\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s+\d{{4}})?"), " <date> "),
    (re.compile(rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+{_MONTHS}(?:,?\s+\d{{4}})?"), " <date> "),
    (re.compile(rf"\b{_MONTHS}\s+\d{{4}}\b"), " <date> "),
    (re.compile(r"\b\d{4}-\d{2}-\d{2}(?:t[\d:.]+z?)?\b"), " <date> "),
    (re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"), " <date> "),
    (
        re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?)?(?:\s*[a-z]{2,4}t)?\b"),
        " <time> ",
    ),
    (
        re.compile(
            r"\b(?:\d+|an?|one)\s+(?:second|minute|min|hour|hr|day|week|month|year)s?\s+ago\b"
        ),
        " <ago> ",
    ),
    (re.compile(r"\b(?:yesterday|today|just now)\b"), " <ago> "),
    (
        re.compile(
            r"\b\d[\d,.]*\s*[km]?\+?\s+(?:followers|views|likes|comments|shares|reactions|"
            r"reviews|ratings|results|visitors|subscribers|members|employees on linkedin|"
            r"open (?:jobs|positions|roles)|jobs|openings)\b"
        ),
        " <count> ",
    ),
]
_WS_RE = re.compile(r"\s+")

# Cap on the change text sent to the judge and stored as the change record.
MAX_CHANGE_CHARS = 6000
# Per paragraph, in the rendered change text.
_MAX_RENDERED_PARAGRAPH = 700


def normalize_paragraph(text: str) -> str:
    """The comparison form of one paragraph (module docstring)."""
    out = text.lower()
    for pattern, replacement in _NOISE_PATTERNS:
        out = pattern.sub(replacement, out)
    return _WS_RE.sub(" ", out).strip()


def content_hash(sections: list[Section]) -> str:
    """sha256 over every section's heading and normalised paragraphs: equal
    for two snapshots that differ only in noise."""
    digest = hashlib.sha256()
    for section in sections:
        digest.update(normalize_paragraph(section.heading).encode("utf-8") + b"\x1e")
        for paragraph in section.paragraphs:
            normalized = normalize_paragraph(paragraph)
            if normalized:
                digest.update(normalized.encode("utf-8") + b"\x1f")
    return digest.hexdigest()


@dataclass
class SectionChange:
    heading: str
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        return {"heading": self.heading, "added": self.added, "removed": self.removed}


@dataclass
class PageChange:
    url: str
    title: str
    changes: list[SectionChange]

    @property
    def added_count(self) -> int:
        return sum(len(c.added) for c in self.changes)

    @property
    def removed_count(self) -> int:
        return sum(len(c.removed) for c in self.changes)

    def render(self, *, observed: str, previous: str) -> str:
        """The change as plain text: what the judge reads and the change
        record's original shows. Capped at `MAX_CHANGE_CHARS`."""
        lines = [
            f"Website change on {self.url}" + (f' ("{self.title}")' if self.title else ""),
            f"Compared the scan of {observed} with the scan of {previous}.",
        ]
        for change in self.changes:
            lines.append("")
            lines.append(f"Section: {change.heading or '(top of page)'}")
            for paragraph in change.added:
                lines.append(f"Added: {_clip(paragraph)}")
            for paragraph in change.removed:
                lines.append(f"Removed: {_clip(paragraph)}")
        text = "\n".join(lines)
        if len(text) > MAX_CHANGE_CHARS:
            text = text[: MAX_CHANGE_CHARS - 20].rstrip() + "\n[change text cut]"
        return text


def _clip(paragraph: str) -> str:
    if len(paragraph) <= _MAX_RENDERED_PARAGRAPH:
        return paragraph
    return paragraph[: _MAX_RENDERED_PARAGRAPH - 3].rstrip() + "..."


def _keyed(sections: list[Section]) -> dict[str, Section]:
    """Sections by normalised heading; a repeated heading gets `#2`, `#3`."""
    out: dict[str, Section] = {}
    counts: Counter[str] = Counter()
    for section in sections:
        base = normalize_paragraph(section.heading)
        counts[base] += 1
        key = base if counts[base] == 1 else f"{base}#{counts[base]}"
        out[key] = section
    return out


def _paragraph_diff(old: list[str], new: list[str]) -> tuple[list[str], list[str]]:
    """`(added, removed)` paragraphs between two lists, compared on the
    normalised form, in page order. Noise-only paragraphs never count."""
    old_pairs = [(normalize_paragraph(p), p) for p in old]
    new_pairs = [(normalize_paragraph(p), p) for p in new]
    old_pairs = [pair for pair in old_pairs if pair[0]]
    new_pairs = [pair for pair in new_pairs if pair[0]]
    matcher = difflib.SequenceMatcher(
        a=[n for n, _ in old_pairs], b=[n for n, _ in new_pairs], autojunk=False
    )
    added: list[str] = []
    removed: list[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ("replace", "delete"):
            removed.extend(p for _, p in old_pairs[i1:i2])
        if tag in ("replace", "insert"):
            added.extend(p for _, p in new_pairs[j1:j2])
    return added, removed


def diff_snapshots(
    old_sections: list[Section], new_sections: list[Section], *, url: str, title: str
) -> PageChange | None:
    """What changed from `old_sections` to `new_sections`, or `None` if
    nothing meaningful did (module docstring)."""
    old_keyed = _keyed(old_sections)
    new_keyed = _keyed(new_sections)
    changes: list[SectionChange] = []
    for key, section in new_keyed.items():
        previous = old_keyed.get(key)
        added, removed = _paragraph_diff(
            previous.paragraphs if previous else [], section.paragraphs
        )
        if added or removed:
            changes.append(SectionChange(heading=section.heading, added=added, removed=removed))
    for key, section in old_keyed.items():
        if key not in new_keyed:
            _added, removed = _paragraph_diff(section.paragraphs, [])
            if removed:
                changes.append(SectionChange(heading=section.heading, removed=removed))

    # A paragraph removed in one place and added in another moved; drop both sides.
    added_norm = Counter(normalize_paragraph(p) for c in changes for p in c.added)
    removed_norm = Counter(normalize_paragraph(p) for c in changes for p in c.removed)
    moved = added_norm & removed_norm
    if moved:
        drop_added = Counter(moved)
        drop_removed = Counter(moved)
        for change in changes:
            kept_added = []
            for paragraph in change.added:
                norm = normalize_paragraph(paragraph)
                if drop_added[norm] > 0:
                    drop_added[norm] -= 1
                else:
                    kept_added.append(paragraph)
            kept_removed = []
            for paragraph in change.removed:
                norm = normalize_paragraph(paragraph)
                if drop_removed[norm] > 0:
                    drop_removed[norm] -= 1
                else:
                    kept_removed.append(paragraph)
            change.added, change.removed = kept_added, kept_removed
        changes = [c for c in changes if c.added or c.removed]

    if not changes:
        return None
    return PageChange(url=url, title=title, changes=changes)
