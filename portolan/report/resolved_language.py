"""The weekly report's resolved-language guard, shared by the headline
(`portolan.report.weekly._headline`) and the executive summary
(`portolan.report.executive._gate_sentence`): a model sentence that says a
problem is over is kept only when the report's own computed theme
movement backs it.

The rule, applied to one sentence:

1. Take out the words that use the vocabulary without asserting that
   anything is resolved: a customer's own words in quotation marks that
   appear verbatim in the claims the model was given (the sentence must
   also cite a source); negated forms ("not yet closed out", "no
   resolution"); forms that state a need ("needs to resolve", "worth
   resolving"); and "resolution proposed"/"proposed resolution" when the
   sentence names a theme whose movement is exactly that.
2. If no resolved-language is left, keep the sentence.
3. Otherwise keep it only when some theme in the report is `resolved`,
   the sentence names one, and it names no theme that is not resolved.

`tests/test_resolved_language.py` pins the sentences that must pass and
the attack sentences that must still be rejected.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from typing import Protocol

__all__ = [
    "RESOLVED_LANGUAGE_RE",
    "grounded_strings",
    "resolved_language_rejected",
    "themes_named_in",
]

# `fixed(?!-)` keeps "fixed-fee" out.
RESOLVED_LANGUAGE_RE = re.compile(
    r"\b(resolved|resolves|resolving|resolve|resolution|resolutions"
    r"|closed out|cleared up|behind us|fixed(?!-))\b",
    re.IGNORECASE,
)

_STATE = r"(?:resolved|fixed|closed\s+out|cleared\s+up)"
_NOT_ASSERTING_RE = re.compile(
    r"(?:\b(?:"
    r"to\s+resolve"
    r"|worth\s+resolving"
    rf"|(?:needs?|must|has|have)\s+to\s+be\s+{_STATE}"
    rf"|yet\s+to\s+be\s+{_STATE}"
    rf"|(?:not|never)\s+(?:yet\s+)?(?:been\s+|be\s+)?(?:confirmed\s+|fully\s+|actually\s+|truly\s+)?{_STATE}"
    rf"|rather\s+than\s+(?:being\s+)?(?:confirmed\s+)?{_STATE}"
    r"|none\s+(?:yet\s+)?(?:resolved|closed\s+out)"
    r"|no\s+(?:confirmed\s+)?resolution"
    r"|(?:a|any)\s+resolution\s+yet"
    rf")|n['’]t\s+(?:yet\s+)?(?:been\s+|be\s+)?{_STATE})\b",
    re.IGNORECASE,
)

_PROPOSED_RESOLUTION_RE = re.compile(
    r"\b(?:resolution\s+(?:is\s+)?proposed|proposed\s+(?:a\s+)?resolution"
    r"|proposed\s+resolving|proposed\s+to\s+resolve)\b",
    re.IGNORECASE,
)

# A quoted span: an opening mark not glued to a word on its left (so the
# apostrophe in "Meadowbrook's" never opens one), a closing mark not glued
# to a word on its right.
_QUOTED_RE = re.compile(r"(?<![\w])['‘\"“]([^'‘’\"“”]+?)['’\"”](?![\w])")
_CITATION_MARKER_RE = re.compile(r"\^\[[^\]]+\]")
_MIN_QUOTED_WORDS = 3
_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")

# Words that never identify a theme on their own; every theme slug starts
# with "theme-".
_SLUG_STOPWORDS = frozenset(
    {"a", "an", "and", "the", "of", "to", "in", "on", "for", "with", "at", "by", "or", "up"}
    | {"is", "its", "theme"}
)


class _Theme(Protocol):
    slug: str
    name: str
    movement: str


def _normalize(text: str) -> str:
    return _NORMALIZE_RE.sub(" ", text.lower()).strip()


def grounded_strings(value: object) -> list[str]:
    """Every string inside `value` (nested dicts/lists), for `grounded_texts`."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in grounded_strings(v)]
    if isinstance(value, list | tuple):
        return [s for v in value for s in grounded_strings(v)]
    return []


def themes_named_in(sentence: str, themes: Sequence[_Theme]) -> set[str]:
    """Slugs of the themes `sentence` names: the theme's full name, or any
    slug word outside the stopwords and outside every full theme name the
    sentence already contains (a word that belongs to another named theme
    says nothing about this one)."""
    lowered = sentence.lower()
    named: set[str] = set()
    remainder = lowered
    for theme in themes:
        name = theme.name.strip().lower()
        if name and name in lowered:
            named.add(theme.slug)
            remainder = remainder.replace(name, " ")
    for theme in themes:
        if theme.slug in named:
            continue
        for word in theme.slug.lower().split("-"):
            if not word or word in _SLUG_STOPWORDS:
                continue
            if re.search(rf"\b{re.escape(word)}\b", remainder):
                named.add(theme.slug)
                break
    return named


def _mask(text: str, pattern: re.Pattern[str]) -> str:
    return pattern.sub(lambda m: " " * len(m.group(0)), text)


def _mask_grounded_quotes(sentence: str, grounded: str) -> str:
    if not grounded or not _CITATION_MARKER_RE.search(sentence):
        return sentence

    def repl(match: re.Match[str]) -> str:
        inner = _normalize(match.group(1))
        if len(inner.split()) >= _MIN_QUOTED_WORDS and inner in grounded:
            return " " * len(match.group(0))
        return match.group(0)

    return _QUOTED_RE.sub(repl, sentence)


def resolved_language_rejected(
    sentence: str, themes: Sequence[_Theme], *, grounded_texts: Iterable[str] = ()
) -> bool:
    """True when `sentence` must be dropped (module docstring, the rule).

    `grounded_texts`: the claim sentences and quotes the model was handed;
    a quoted span counts as the customer's words only when it appears in
    one of them verbatim (compared lower-cased, punctuation collapsed)."""
    grounded = " | ".join(_normalize(t) for t in grounded_texts if t)
    text = _mask_grounded_quotes(sentence, grounded)
    text = _mask(text, _NOT_ASSERTING_RE)

    named = themes_named_in(sentence, themes)
    proposed = {t.slug for t in themes if t.movement == "resolution proposed"}
    if named & proposed:
        text = _mask(text, _PROPOSED_RESOLUTION_RE)

    if not RESOLVED_LANGUAGE_RE.search(text):
        return False

    resolved = {t.slug for t in themes if t.movement == "resolved"}
    if not resolved:
        return True
    return not (named & resolved) or bool(named - resolved)
