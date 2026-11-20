"""Name normalisation helpers shared by import and resolution.

Matching never guesses (`portolan-spec-v2.md` sec 3): an exact
canonical-name match, then an alias match, then a normalised match that
only strips legal suffixes and punctuation. Every function here is a
small, deterministic, side-effect-free string transform — no fuzzy
matching, no edit distance.
"""

from __future__ import annotations

import re

# The four legal suffixes the registry contract names, longest (most words) first so a
# two-word suffix is stripped as a unit rather than leaving a dangling first word behind.
LEGAL_SUFFIXES: tuple[str, ...] = ("Medical Center", "Hospital", "Health", "Inc")

_SUFFIX_RE = re.compile(
    r"\s+(?:" + "|".join(re.escape(s) for s in LEGAL_SUFFIXES) + r")\.?$",
    re.IGNORECASE,
)
_PUNCT_RE = re.compile(r"[.,'\"]")
_WHITESPACE_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z.'-]*")
_SLUG_RE = re.compile(r"[^a-z0-9]+")


def strip_legal_suffix(name: str) -> str:
    """Strip a trailing legal suffix (see `LEGAL_SUFFIXES`), repeatedly.

    "Ashgrove Medical Center" -> "Ashgrove". "Larkspur Regional Health" ->
    "Larkspur Regional". A name with no matching suffix is returned
    unchanged (stripped of surrounding whitespace only).
    """
    current = name.strip()
    for _ in range(3):  # defensive bound; no real name stacks 3 suffixes
        updated = _SUFFIX_RE.sub("", current).strip()
        if updated == current:
            return current
        current = updated
    return current


def normalize_name(name: str) -> str:
    """Lowercase, strip a legal suffix, strip punctuation, collapse whitespace.

    This is the third resolution tier in `Registry.resolve()`: "normalised
    (strip suffixes, punctuation)". Applied symmetrically to a mention and
    to every candidate's canonical name.
    """
    stripped = strip_legal_suffix(name)
    lowered = stripped.lower()
    no_punct = _PUNCT_RE.sub("", lowered)
    return _WHITESPACE_RE.sub(" ", no_punct).strip()


def light_normalize(name: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace — no suffix
    stripping.

    Used only for import-time duplicate-*row* detection (`importer.py`'s
    account dedup). `normalize_name`'s suffix stripping would wrongly
    collapse a parent system into a child facility sharing its domain:
    "Ashgrove Health" and "Ashgrove Medical Center" both reduce to
    "ashgrove" under it, but they are not the same account.
    """
    lowered = name.strip().lower()
    no_punct = _PUNCT_RE.sub("", lowered)
    return _WHITESPACE_RE.sub(" ", no_punct).strip()


def acronym_of(name: str) -> str | None:
    """The acronym of a 2-to-4 capitalised-word name, else None.

    "Ashgrove Medical Center" -> "AMC". "Cobalt Ridge Medical Center" ->
    "CRMC". "Northlake General Hospital" -> "NGH" — all three match aliases
    `corpus/truth-table.yaml` lists by hand.

    Every word must start with an uppercase letter (a stray lowercase
    connector like "of"/"the" disqualifies the whole name) and there must
    be 2 to 4 such words.
    """
    words = _WORD_RE.findall(name)
    if not (2 <= len(words) <= 4):
        return None
    if not all(w[0].isupper() for w in words):
        return None
    letters = "".join(w[0] for w in words)
    return letters.upper()


def slugify(text: str) -> str:
    """A stable, lowercase, hyphenated slug: "Tom Ridley" -> "tom-ridley"."""
    slug = _SLUG_RE.sub("-", text.lower()).strip("-")
    return slug or "entity"


def domain_of_email(email: str | None) -> str | None:
    """The lowercase domain part of an email address, or None."""
    if not email or "@" not in email:
        return None
    return email.rsplit("@", 1)[1].strip().lower() or None
