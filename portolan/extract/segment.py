"""Split a long source into extractable segments, and merge their per-segment
extraction results back into one result per source.

extraction-schema-v0.md, "Segmenting long sources": "A 25k-token transcript
is extracted in segments of roughly 4k tokens with a small overlap... Facts
are merged per source before the graph merge: same type, same entities,
locators within 90 seconds or 2 paragraphs, claim similarity above threshold
means one fact, keeping the earliest locator."

Splitting only ever happens on line boundaries. Locators (`[00:14:32]`,
`[msg:...]`, `[p7¶2]`, `[L123]`) appear at line start, so a boundary that
never falls inside a line never cuts a locator in half — there is no special
"don't split here" logic beyond "only split between lines".

**One exception (bug hunt 2026-09-18, finding 11).** RSS items and CRM
rows are prepared as a single "line" with one locator at its start
(`_prepare_rss`/`_prepare_crm`), as is any email paragraph — a line
longer than `max_tokens` used to become its own unbounded segment (the
`current and ...` split test below only ever fires when `current` is
non-empty, so a line that's already over budget the moment it's the
first thing added to an empty bucket never gets checked at all).
`_hard_split_line` now splits such a line on plain character count
before segmenting proceeds — only sub-chunks past the first lose their
line's own locator (nothing here can invent one), which is a real,
narrower gap than the unbounded segment it replaces.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from difflib import SequenceMatcher

from portolan.extract.schema import Action, Coverage, EntityMention, ExtractionResult, Fact
from portolan.ingest.locators import parse_locator

_TIMESTAMP_RE = re.compile(r"^(\d+):(\d{2}):(\d{2})$")

_CLAIM_SIMILARITY_THRESHOLD = 0.6
_LOCATOR_WINDOW_SECONDS = 90.0


def _default_token_count(text: str) -> int:
    """`len(text)//4`, per the build instructions, unless a real tokenizer
    is injected via `count_tokens`."""
    return max(1, len(text) // 4)


def _hard_split_line(line: str, max_tokens: int, counter: Callable[[str], int]) -> list[str]:
    """Bug hunt 2026-09-18, finding 11: split `line` on character count
    when it's over `max_tokens` on its own -- a no-op (single-element
    list) otherwise. Binary-searches the largest prefix that still fits
    the target on each cut, so this works against any injected `counter`,
    not just the default `len // 4` estimate.

    Each piece targets `max_tokens // 2`, not `max_tokens` -- `_seed_
    overlap` always carries at least one WHOLE line into the next bucket
    once a bucket closes, even one that alone reaches `max_tokens` (a
    guarantee this module's per-line overlap logic already relied on, it
    just never had a line big enough to expose it before this fix). A
    piece sized to a full `max_tokens` would make the next bucket start
    already AT budget before its own first new piece is even added --
    doubling that bucket's true size. Halving the target leaves enough
    headroom that one seeded piece plus one new piece still fits within
    `max_tokens`, matching every other bucket this function produces.
    """
    if counter(line) <= max_tokens or not line:
        return [line]

    target = max(1, max_tokens // 2)
    pieces: list[str] = []
    remaining = line
    while remaining and counter(remaining) > target:
        lo, hi, best = 1, len(remaining), 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if counter(remaining[:mid]) <= target:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        pieces.append(remaining[:best])
        remaining = remaining[best:]
    if remaining:
        pieces.append(remaining)
    return pieces


@dataclass(frozen=True)
class Segment:
    """One chunk of a segmented source."""

    index: int
    total: int
    text: str
    start_locator: str | None
    end_locator: str | None


def segment(
    text: str,
    max_tokens: int = 4000,
    overlap_tokens: int = 200,
    count_tokens: Callable[[str], int] | None = None,
) -> list[Segment]:
    """Split `text` into segments of roughly `max_tokens`, overlapping
    consecutive segments by roughly `overlap_tokens` so entity mentions near
    a boundary appear in both the tail of one segment and the head of the
    next. Always splits on line boundaries, except a single line already
    over `max_tokens` on its own -- see `_hard_split_line` and finding 11
    in the module docstring.
    """
    counter = count_tokens or _default_token_count
    raw_lines = text.splitlines()
    if not raw_lines:
        return [Segment(index=1, total=1, text=text, start_locator=None, end_locator=None)]

    lines = [
        piece for raw_line in raw_lines for piece in _hard_split_line(raw_line, max_tokens, counter)
    ]

    line_tokens = [counter(line) for line in lines]

    buckets: list[list[int]] = []
    current: list[int] = []
    current_tokens = 0

    for i, tok in enumerate(line_tokens):
        if current and current_tokens + tok > max_tokens:
            buckets.append(current)
            current, current_tokens = _seed_overlap(
                current, line_tokens, overlap_tokens, max_tokens
            )
        current.append(i)
        current_tokens += tok

    if current:
        buckets.append(current)

    total = len(buckets)
    segments: list[Segment] = []
    for seg_index, bucket in enumerate(buckets, start=1):
        seg_text = "\n".join(lines[idx] for idx in bucket)
        locators = [loc for idx in bucket if (loc := parse_locator(lines[idx])) is not None]
        segments.append(
            Segment(
                index=seg_index,
                total=total,
                text=seg_text,
                start_locator=locators[0] if locators else None,
                end_locator=locators[-1] if locators else None,
            )
        )
    return segments


def _seed_overlap(
    prior_bucket: list[int],
    line_tokens: list[int],
    overlap_tokens: int,
    max_tokens: int,
) -> tuple[list[int], int]:
    """Pick the trailing lines of `prior_bucket` to carry over as the start
    of the next segment: enough to cover `overlap_tokens`, capped at half of
    `max_tokens`. The cap is a defensive bound: without it, a caller who
    passes an `overlap_tokens` close to or above `max_tokens` (or just uses
    the default `overlap_tokens=200` with a small custom `max_tokens`) would
    get segments that each add only a sliver of new content over the last,
    ballooning the segment count instead of producing a small overlap."""
    if overlap_tokens <= 0:
        return [], 0
    target = min(overlap_tokens, max_tokens // 2)
    if target <= 0:
        return [], 0
    overlap: list[int] = []
    overlap_sum = 0
    for idx in reversed(prior_bucket):
        candidate_sum = overlap_sum + line_tokens[idx]
        if overlap and candidate_sum > target:
            break
        overlap.insert(0, idx)
        overlap_sum = candidate_sum
        if overlap_sum >= target:
            break
    return overlap, overlap_sum


def _parse_timestamp_seconds(locator: str) -> float | None:
    m = _TIMESTAMP_RE.match(locator.strip())
    if not m:
        return None
    hours, minutes, seconds = (int(part) for part in m.groups())
    return hours * 3600 + minutes * 60 + seconds


def _locators_within_window(
    a: str, b: str, window_seconds: float = _LOCATOR_WINDOW_SECONDS
) -> bool | None:
    """True/False when both locators parse as HH:MM:SS timestamps; None when
    they don't (a different locator format), so the caller falls back to
    claim similarity alone rather than treating "can't compare" as "far
    apart"."""
    ta, tb = _parse_timestamp_seconds(a), _parse_timestamp_seconds(b)
    if ta is None or tb is None:
        return None
    return abs(ta - tb) <= window_seconds


def _claim_similarity(a: str, b: str) -> float:
    """Simple similarity per the build instructions ("Use a simple
    similarity (difflib ratio) for now")."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _earlier_locator(a: str, b: str) -> str:
    """Keep the earliest locator. When both parse as timestamps, compare
    numerically; otherwise `a` is treated as earliest because facts are
    merged in segment/appearance order, so `a` was seen first."""
    ta, tb = _parse_timestamp_seconds(a), _parse_timestamp_seconds(b)
    if ta is not None and tb is not None:
        return a if ta <= tb else b
    return a


def _facts_match(a: Fact, b: Fact) -> bool:
    """Same type, overlapping entities (or both empty), locators within the
    90-second window when both are timestamps, and claim similarity above
    threshold — extraction-schema-v0.md's per-source dedup rule."""
    if a.type != b.type:
        return False
    shared_entities = set(a.entities) & set(b.entities)
    entities_ok = bool(shared_entities) or (not a.entities and not b.entities)
    if not entities_ok:
        return False

    similarity = _claim_similarity(a.claim, b.claim)
    within_window = _locators_within_window(a.locator, b.locator)
    if within_window is True:
        # Close in time is strong dedup evidence on its own; allow a lower
        # text-similarity bar.
        return similarity >= 0.4
    if within_window is False:
        # Far apart in time; only merge if the claim text is nearly
        # identical anyway.
        return similarity >= 0.85
    # Locators aren't comparable as timestamps (e.g. page/paragraph
    # locators) — fall back to claim similarity alone.
    return similarity >= _CLAIM_SIMILARITY_THRESHOLD


def _merge_entities(all_entities: list[EntityMention]) -> list[EntityMention]:
    merged: dict[tuple[str, str], EntityMention] = {}
    for ent in all_entities:
        key = (ent.mention.strip().lower(), ent.kind.value)
        existing = merged.get(key)
        if existing is None:
            merged[key] = ent
            continue
        existing_resolved = existing.registry_id is not None
        new_resolved = ent.registry_id is not None
        newly_resolved = new_resolved and not existing_resolved
        tie_but_more_confident = (
            new_resolved == existing_resolved and ent.confidence > existing.confidence
        )
        if newly_resolved or tie_but_more_confident:
            merged[key] = ent
    return list(merged.values())


def _merge_facts(all_facts: list[Fact]) -> list[Fact]:
    accepted: list[Fact] = []
    for fact in all_facts:
        match_idx = next(
            (i for i, existing in enumerate(accepted) if _facts_match(existing, fact)), None
        )
        if match_idx is None:
            accepted.append(fact)
            continue
        existing = accepted[match_idx]
        merged_entities = list(dict.fromkeys([*existing.entities, *fact.entities]))
        accepted[match_idx] = existing.model_copy(
            update={
                "entities": merged_entities,
                "locator": _earlier_locator(existing.locator, fact.locator),
                "quote": existing.quote or fact.quote,
                "confidence": max(existing.confidence, fact.confidence),
            }
        )
    # Each segment's extractor call restarts its own "f1", "f2", ... ids, so
    # ids collide across segments once merged. Renumber sequentially here.
    return [fact.model_copy(update={"id": f"f{i}"}) for i, fact in enumerate(accepted, start=1)]


def _actions_match(a: Action, b: Action) -> bool:
    same_owner = (a.owner_mention or "").strip().lower() == (b.owner_mention or "").strip().lower()
    return same_owner and _claim_similarity(a.what, b.what) >= _CLAIM_SIMILARITY_THRESHOLD


def _merge_actions(all_actions: list[Action]) -> list[Action]:
    accepted: list[Action] = []
    for action in all_actions:
        match_idx = next(
            (i for i, existing in enumerate(accepted) if _actions_match(existing, action)), None
        )
        if match_idx is None:
            accepted.append(action)
            continue
        existing = accepted[match_idx]
        accepted[match_idx] = existing.model_copy(
            update={
                "locator": _earlier_locator(existing.locator, action.locator),
                "due": existing.due or action.due,
            }
        )
    return accepted


def merge_segment_results(
    results: list[ExtractionResult], *, segments_total: int | None = None
) -> ExtractionResult:
    """Merge one ExtractionResult per segment of the same source into a
    single per-source result, applying the dedup rule from
    extraction-schema-v0.md "Segmenting long sources".

    Results are expected in segment order (index 1, 2, 3, ...) — "keep the
    earliest locator" and "each segment restarts f1, f2, ..." both rely on
    that order to mean what "earliest" and "renumber" should mean.

    `segments_total` should be the true number of segments the caller split
    the source into (review finding F10): trusting `results[0].coverage.
    segments_total`, the model's own self-reported count for its segment,
    lets one hallucinating segment corrupt the merged coverage ("3 of 1").
    When the caller can't supply it, `max()` across every result's
    self-reported total is a strictly better guess than segment 1's alone.
    """
    if not results:
        raise ValueError("merge_segment_results requires at least one ExtractionResult")

    all_entities = [e for r in results for e in r.entities]
    all_facts = [f for r in results for f in r.facts]
    all_actions = [a for r in results for a in r.actions]
    unknown_entities = list(dict.fromkeys(u for r in results for u in r.unknown_entities))
    # Bug hunt 2026-09-18, finding 6: each segment's own `parse_lenient`
    # call already computed `rejected`/`coercions` (fact rule 4: "a
    # rejected fact is counted in the coverage report") -- this rebuild
    # used to drop both on the floor, so nothing past this function ever
    # saw a rejected fact or a coercion that happened on any segment.
    all_rejected = [r for result in results for r in result.rejected]
    all_coercions = [c for result in results for c in result.coercions]

    # Prefer the longest non-empty summary as the most informative one-line
    # overview; segments after the first often add detail the first missed.
    summaries = [r.summary.strip() for r in results if r.summary.strip()]
    summary = max(summaries, key=len) if summaries else ""

    coverage = Coverage(
        segments_read=len(results),
        segments_total=(
            segments_total
            if segments_total is not None
            else max(r.coverage.segments_total for r in results)
        ),
    )

    return ExtractionResult(
        schema_version=results[0].schema_version,
        source_id=results[0].source_id,
        summary=summary,
        entities=_merge_entities(all_entities),
        facts=_merge_facts(all_facts),
        actions=_merge_actions(all_actions),
        unknown_entities=unknown_entities,
        coverage=coverage,
        rejected=all_rejected,
        coercions=all_coercions,
    )
