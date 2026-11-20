"""Fact marker handling: find ``⟦F01⟧ ... ⟦/F01⟧``, record the locator of the
line/paragraph/comment it sits on, verify one marker per planted fact, and
strip the markers to produce the clean file the corpus actually ships.

This is the one mechanical step every prose source goes through regardless
of how its words were produced (API call, hand-written, or a delegated
subagent writing plain text): the writer's only job is to wrap the sentence
carrying each fact in its marker; this module turns that into a locator and
removes the marker glyphs so the corpus reads like a normal transcript,
ticket, or email.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from difflib import SequenceMatcher

from corpus.generator.structured import call_locator, email_locator, ticket_locator

MARKER_RE = re.compile(r"⟦(F\d+[a-z]?)⟧(.*?)⟦/\1⟧", re.DOTALL)
CALL_LINE_RE = re.compile(r"^\[(\d{2}:\d{2}:\d{2})\]\s+([^:]+):\s?(.*)$")

# --------------------------------------------------------------------------
# World generator v2 extension (`01-spec/world-generator-v2-2026-09-22.md`
# section 3.1 and section 9: "Markers ... Reuse, extended with ⟦D⟧ decoys").
# A v2 writer wraps each scripted fact in ⟦F…⟧…⟦/F…⟧ (unchanged) and each
# decoy -- a near-miss statement that is NOT a fact -- in ⟦D…⟧…⟦/D…⟧. The
# helpers below see both kinds; the v1 functions above stay F-only and
# unchanged for their existing callers.
# --------------------------------------------------------------------------

# Every marker id a v2 writer is given: `F00041`, `D00004` (optionally a
# trailing letter, same as v1's multi-mention ids).
V2_MARKER_ID_RE = r"[FD]\d+[a-z]?"

# A balanced pair of either kind, same non-greedy, same-id-closes shape as
# `MARKER_RE`.
ANY_MARKER_RE = re.compile(rf"⟦({V2_MARKER_ID_RE})⟧(.*?)⟦/\1⟧", re.DOTALL)

# One marker TOKEN (an opener or a closer), used to find unbalanced or
# crossing markers that `ANY_MARKER_RE` alone would silently skip.
MARKER_TOKEN_RE = re.compile(rf"⟦(/?)({V2_MARKER_ID_RE})⟧")

# Glyph variants cheap models emit instead of the exact brackets they were
# shown: the CJK white square brackets, a doubled ASCII bracket, stray
# whitespace inside the brackets, or the wrong bracket first (observed live
# 2026-09-22 from qwen3 models: `⟧/F00041⟧` for the closer `⟦/F00041⟧`).
# Only a bracket pair wrapped around a marker id is rewritten, so ordinary
# `[[`...`]]` text is left alone.
_MARKER_VARIANT_RE = re.compile(
    rf"(?:[⟦⟧〚〛]|\[\[)\s*(/?)\s*({V2_MARKER_ID_RE})\s*(?:[⟦⟧〚〛]|\]\])"
)

# A lone bracket glyph left after every well-formed token is removed.
STRAY_MARKER_GLYPH_RE = re.compile(r"[⟦⟧〚〛]")


# An angle bracket typed next to a marker token is part of the token, not
# text (live 2026-09-22: `.⟩⟧/F00406⟧` left "⟩" at the end of a label).
_ADJACENT_ANGLE_RE = re.compile(rf"[⟨⟩]+(?=⟦/?{V2_MARKER_ID_RE}⟧)|(?<=⟧)[⟨⟩]+")


def normalize_marker_glyphs(text: str) -> str:
    """Rewrite marker-token variants to the canonical `⟦F00041⟧` /
    `⟦/F00041⟧` form. Structure only: the words between markers are
    untouched."""
    canonical = _MARKER_VARIANT_RE.sub(lambda m: f"⟦{m.group(1)}{m.group(2)}⟧", text)
    return _ADJACENT_ANGLE_RE.sub("", canonical)


def strip_all_markers(text: str) -> str:
    """Remove every v2 marker token (balanced or not), keeping the wrapped
    text. Unlike `strip_markers`, an unbalanced opener or closer is removed
    too, and so is a lone bracket glyph, so no marker glyph can survive into
    a shipped file."""
    return STRAY_MARKER_GLYPH_RE.sub("", MARKER_TOKEN_RE.sub("", text))


@dataclass(frozen=True)
class Placement:
    fact_id: str
    locator: str
    line_or_comment: str
    excerpt: str


class MarkerError(ValueError):
    """Raised when a source's markers don't match its brief exactly."""


def strip_markers(text: str) -> str:
    """Remove marker glyphs, keeping the wrapped text."""
    return MARKER_RE.sub(lambda m: m.group(2), text)


def find_markers(text: str) -> list[tuple[str, str]]:
    """Return [(fact_id, inner_text), ...] for every marker in `text`, in
    document order. Does not strip."""
    return [(m.group(1), m.group(2)) for m in MARKER_RE.finditer(text)]


def _excerpt(text: str, limit: int = 160) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def verify_exactly_one(
    expected_fact_ids: list[str],
    placements: list[Placement],
    allow_multiple: frozenset[str] = frozenset(),
) -> list[str]:
    """Return a list of human-readable problems: empty means the source's
    markers exactly match its brief (one marker per expected fact, no
    extras) — except fact ids in `allow_multiple`, which the brief
    deliberately plants twice in the same source (F04: "two separate
    mentions ... in the same call"; see briefs.MULTI_MENTION_FACTS), where
    two or more is fine and zero is still a problem."""
    problems: list[str] = []
    counts: dict[str, int] = {}
    for p in placements:
        counts[p.fact_id] = counts.get(p.fact_id, 0) + 1
    for fid in expected_fact_ids:
        n = counts.get(fid, 0)
        if n == 0:
            problems.append(f"missing marker for {fid}")
        elif n > 1 and fid not in allow_multiple:
            problems.append(f"{n} markers for {fid}, expected exactly 1")
    for fid in counts:
        if fid not in expected_fact_ids:
            problems.append(f"unexpected marker for {fid} (not in this source's brief)")
    return problems


def best_match_chunk_index(claim_text: str, candidates: list[str]) -> int | None:
    """Index of the item in `candidates` (a source's lines/paragraphs/comment
    bodies) most similar to `claim_text` by plain-text `difflib` ratio, or
    `None` if `candidates` is empty. Used only as a last-resort fallback
    (`insert_marker_around_best_match` below) when a live model omitted a
    fact's marker glyphs but did write the fact itself in its own words --
    the same fuzzy-match-then-trust-it approach `portolan.fake_extractor`
    already uses to identify which truth-table fact a rendered claim came
    from, applied here to locate it in raw generated text instead."""
    if not candidates:
        return None
    plain_claim = " ".join(claim_text.lower().split())
    best_index: int | None = None
    best_ratio = 0.0
    for index, candidate in enumerate(candidates):
        ratio = SequenceMatcher(None, plain_claim, " ".join(candidate.lower().split())).ratio()
        if ratio > best_ratio:
            best_ratio, best_index = ratio, index
    return best_index


def insert_marker_around_best_match(
    fact_id: str, claim_text: str, chunks: list[str]
) -> list[str] | None:
    """Best-effort repair for a live-model generation that stated a fact in
    its own words but forgot the ⟦F..⟧...⟦/F..⟧ marker glyphs (observed live,
    2026-09-13: `openai/gpt-5.6-luna` did this for a handful of unconstrained-
    phrasing facts even after several same-prompt retries -- see
    `corpus/generator/cli.py`'s `_generate_*` retry helpers, which try this
    only after every live-retry attempt is exhausted). Wraps the WHOLE
    best-matching chunk (a call line, a ticket comment body, or an email
    paragraph -- whatever unit `chunks` holds) in the marker, on the theory
    that a human/automated reviewer would rather cite one full sentence-or-
    two of real, un-invented model text than lose the source's only mention
    of this fact entirely. Returns the updated `chunks` list, or `None` if
    `chunks` is empty (nothing to wrap)."""
    index = best_match_chunk_index(claim_text, chunks)
    if index is None:
        return None
    updated = list(chunks)
    updated[index] = f"⟦{fact_id}⟧{updated[index]}⟦/{fact_id}⟧"
    return updated


# --------------------------------------------------------------------------
# Per-source-type processing
# --------------------------------------------------------------------------


def process_call(raw_text: str) -> tuple[str, list[Placement]]:
    """Parse `[HH:MM:SS] Speaker: text` lines, record the timestamp of any
    line carrying a marker, strip markers, and return the clean transcript
    plus the placements found."""
    placements: list[Placement] = []
    clean_lines: list[str] = []
    for line_number, line in enumerate(raw_text.splitlines(), start=1):
        markers_here = find_markers(line)
        if markers_here:
            m = CALL_LINE_RE.match(line)
            ts = m.group(1) if m else ""
            for fact_id, inner in markers_here:
                placements.append(
                    Placement(fact_id, call_locator(ts), str(line_number), _excerpt(inner))
                )
        clean_lines.append(strip_markers(line))
    return "\n".join(clean_lines), placements


def process_ticket_comments(
    ticket_id: str, comments: list[dict]
) -> tuple[list[dict], list[Placement]]:
    """Each comment dict has at least `number` (int) and `body` (str)."""
    placements: list[Placement] = []
    clean_comments: list[dict] = []
    for c in comments:
        markers_here = find_markers(c["body"])
        for fact_id, inner in markers_here:
            placements.append(
                Placement(
                    fact_id,
                    ticket_locator(ticket_id, c["number"]),
                    str(c["number"]),
                    _excerpt(inner),
                )
            )
        clean = dict(c)
        clean["body"] = strip_markers(c["body"])
        clean_comments.append(clean)
    return clean_comments, placements


def process_indexed_texts(
    texts: list[str], locator_fn: Callable[[int], str]
) -> tuple[list[str], list[Placement]]:
    """Generic per-unit marker scan, added 2026-09-19 for the synthetic-data
    pipeline's chat (Slack/Teams message text) and CRM-activity-note
    sources: `texts` is an ordered list of independent text units, each
    scanned for its own markers exactly like one ticket comment body or one
    email paragraph. `locator_fn(index)` (0-based, matching `texts`'s own
    order) builds that unit's locator string. Returns (clean_texts,
    placements) in the same order as `texts`.

    `process_ticket_comments`/`process_email_body` predate this and are not
    rewritten to call it — same behavior, kept as-is to avoid disturbing
    their existing callers/tests."""
    placements: list[Placement] = []
    clean: list[str] = []
    for idx, text in enumerate(texts):
        markers_here = find_markers(text)
        for fact_id, inner in markers_here:
            placements.append(Placement(fact_id, locator_fn(idx), str(idx), _excerpt(inner)))
        clean.append(strip_markers(text))
    return clean, placements


def process_email_body(source_id: str, raw_body: str) -> tuple[str, list[Placement]]:
    """Split the body into blank-line-separated paragraphs, record the
    1-based paragraph index of any paragraph carrying a marker."""
    placements: list[Placement] = []
    paragraphs = raw_body.split("\n\n")
    clean_paragraphs: list[str] = []
    for idx, para in enumerate(paragraphs, start=1):
        markers_here = find_markers(para)
        for fact_id, inner in markers_here:
            placements.append(
                Placement(fact_id, email_locator(source_id, idx), str(idx), _excerpt(inner))
            )
        clean_paragraphs.append(strip_markers(para))
    return "\n\n".join(clean_paragraphs), placements
