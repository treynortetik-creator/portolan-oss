"""`FakeBackend`-driven canned results for testing the bench's *scoring*
logic without a model call: `perfect_results` emits exactly what the truth
table planted (recall 1.0 / precision 1.0 by construction); `lossy_results`
derives from it by dropping, inventing, or mis-locating a controlled
number of facts, for tests with a known expected score.
"""

from __future__ import annotations

import random
import re

from portolan.bench.locators import parse_hhmmss, trailing_int
from portolan.bench.truth import PlantedFact, Truth
from portolan.extract.schema import (
    QUOTE_REQUIRED_TYPES,
    Coverage,
    ExtractionResult,
    Fact,
    FactType,
)

_QUOTE_WORD_LIMIT = 40
_QUOTE_REQUIRED_VALUES = {t.value for t in QUOTE_REQUIRED_TYPES}


def _quote_for(planted: PlantedFact) -> str | None:
    if planted.type not in _QUOTE_REQUIRED_VALUES:
        return None
    words = planted.excerpt.split()[:_QUOTE_WORD_LIMIT]
    return " ".join(words) if words else "no quote available in the excerpt"


def _fact_from_planted(planted: PlantedFact) -> Fact:
    # `id` is set to the truth-table fact_id (e.g. "F02") rather than the
    # "f1", "f2", ... a real extractor emits — nothing in scoring reads
    # `Fact.id`, and reusing the truth id lets `lossy_results` find and
    # drop/mislocate a specific planted fact by name.
    return Fact(
        id=planted.fact_id,
        type=FactType(planted.type),
        claim=planted.claim or planted.excerpt,
        entities=list(planted.entities),
        locator=planted.locator,
        quote=_quote_for(planted),
        speaker="customer",
        sentiment=0,
        confidence=0.9,
    )


def _result_for_source(source_id: str, planted_list: list[PlantedFact]) -> ExtractionResult:
    facts = [_fact_from_planted(p) for p in planted_list]
    return ExtractionResult(
        source_id=source_id,
        summary="bench fixture",
        entities=[],
        facts=facts,
        actions=[],
        unknown_entities=[],
        coverage=Coverage(segments_read=1, segments_total=1),
    )


def perfect_results(truth: Truth) -> dict[str, ExtractionResult]:
    """One `ExtractionResult` per manifest source, containing exactly the
    fact(s) planted in it (none, for a noise or distractor source). Scores
    recall 1.0 / precision 1.0 against `truth` by construction."""
    return {
        src.id: _result_for_source(src.id, truth.planted(src.id)) for src in truth.manifest.sources
    }


def _far_locator(planted: PlantedFact) -> str:
    """A locator in the same format family as `planted.locator` but well
    outside its matching window, so a fact carrying it fails the locator
    criterion specifically rather than disappearing."""
    loc = planted.locator
    if planted.source_type == "call":
        seconds = parse_hhmmss(loc)
        if seconds is not None:
            moved = int(seconds) + 3600  # one hour away; window is 90s
            hours, remainder = divmod(moved, 3600)
            minutes, secs = divmod(remainder, 60)
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    n = trailing_int(loc)
    if n is not None:
        return re.sub(r"\d+\s*$", str(n + 1000), loc)
    return loc + "-wrong"


def _invented_fact(planted_like: PlantedFact, suffix: str) -> Fact:
    """A fact with no corresponding planted fact at all: same shape as a
    real one so it passes `Fact`'s own validation, but a claim that cannot
    match anything planted in this source."""
    fact_type = FactType(planted_like.type) if planted_like.type != "unknown" else FactType.OTHER
    quote_needed = fact_type.value in _QUOTE_REQUIRED_VALUES
    return Fact(
        id=f"invented-{suffix}",
        type=fact_type,
        claim=f"Invented fact {suffix} with no planted counterpart in this source.",
        entities=[],
        locator=planted_like.locator,
        quote=("a fabricated quote for bench testing" if quote_needed else None),
        speaker="customer",
        sentiment=0,
        confidence=0.5,
    )


def lossy_results(
    truth: Truth,
    *,
    drop: int = 0,
    invent: int = 0,
    wrong_locator: int = 0,
    seed: int = 0,
) -> dict[str, ExtractionResult]:
    """`perfect_results`, mutated in three independent, non-overlapping
    ways so a test can assert an exact resulting score:

    - `drop`: this many *distinct planted fact_ids* are removed from every
      source they appear in (so a cross-source fact like F02 is fully
      gone, not just missing from one placement) — lowers the unique-
      fact-id recall count by exactly `drop`.
    - `wrong_locator`: this many other planted fact_ids are kept but given
      a locator far outside their window (`_far_locator`) in every source
      they appear in — fails the locator criterion specifically, so they
      show up in `unmatched` with a reason rather than just vanishing.
    - `invent`: this many extra, non-matching facts are appended, one per
      non-distractor source that already has at least one planted fact
      (distractor-source leakage is a deliberately different test) —
      raises `unplanted_count` by exactly `invent`.
    """
    rng = random.Random(seed)
    results = perfect_results(truth)

    all_planted = truth.all_planted()
    unique_fact_ids = sorted({p.fact_id for p in all_planted})
    rng.shuffle(unique_fact_ids)
    drop_ids = set(unique_fact_ids[:drop])
    mislocate_ids = set(unique_fact_ids[drop : drop + wrong_locator])

    planted_by_key = {(p.source_id, p.fact_id): p for p in all_planted}

    for source_id, result in list(results.items()):
        kept: list[Fact] = []
        for fact in result.facts:
            if fact.id in drop_ids:
                continue
            if fact.id in mislocate_ids:
                planted = planted_by_key[(source_id, fact.id)]
                fact = fact.model_copy(update={"locator": _far_locator(planted)})
            kept.append(fact)
        results[source_id] = result.model_copy(update={"facts": kept})

    if invent:
        distractor_ids = truth.distractor_source_ids()
        candidate_sources = [
            src_id
            for src_id, planted_list in (
                (s.id, truth.planted(s.id)) for s in truth.manifest.sources
            )
            if planted_list and src_id not in distractor_ids
        ]
        for i in range(invent):
            source_id = candidate_sources[i % len(candidate_sources)]
            reference = truth.planted(source_id)[0]
            new_fact = _invented_fact(reference, suffix=str(i))
            result = results[source_id]
            results[source_id] = result.model_copy(update={"facts": [*result.facts, new_fact]})

    return results
