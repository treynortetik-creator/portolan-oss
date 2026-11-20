"""Anchor each extracted fact to the place in the source it cites.

A fact reaches merge with a `locator` and, usually, a `quote`, both typed by
the model. Before this module the pipeline checked only that the quote's
words occurred somewhere in the source, and never looked at the locator at
all. So a claim could cite `00:14:32` while its quote was said at
`00:31:07`, cite a locator that exists nowhere in the source, or carry a
"verbatim" quote the model had re-typed with a word changed.

`anchor_fact` settles all three mechanically, from the source text and the
locator index `portolan.ingest.prepare` built. Nothing here is a judgement:

- the quote is located in the text (`quote_gate.locate_quote`), in the
  cited block first;
- a near-verbatim quote is replaced by the source's own words for it;
- a locator that disagrees with where the quote sits, or is not one of the
  source's locators, is replaced by the block the quote is in;
- a locator typed with its brackets, as a range, or shortened to a unique
  tail of a real one is read as the real one;
- a fact that still cannot be tied to a place (no supported quote, and a
  locator the source does not have) is reported as a `problem`, for the
  caller to park for a person. It is never dropped here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from portolan.extract.quote_gate import QuoteMatch, locate_quote
from portolan.extract.schema import Fact
from portolan.ingest.models import LocatorSpan

__all__ = [
    "PROBLEM_LOCATOR",
    "PROBLEM_QUOTE",
    "PROBLEM_REASONS",
    "Anchored",
    "anchor_fact",
]

PROBLEM_QUOTE = "quote_not_in_source"
PROBLEM_LOCATOR = "locator_not_in_source"

#: The one-line explanation a review card shows for each problem.
PROBLEM_REASONS = {
    PROBLEM_QUOTE: "The model's quote for this fact was not found in the source text.",
    PROBLEM_LOCATOR: (
        "This fact cites a place that is not in the source, and has no quote to find it by."
    ),
}

_MAX_QUOTE_WORDS = 40


@dataclass
class Anchored:
    """`fact` after anchoring; `changes` names what was corrected
    (`locator_repaired`, `quote_snapped`, `quote_dropped`); `problem` is set
    when the fact could not be tied to a place in the source."""

    fact: Fact
    changes: list[str] = field(default_factory=list)
    problem: str | None = None


def _span_at(spans: list[LocatorSpan], offset: int) -> LocatorSpan | None:
    for span in spans:
        if span.start <= offset < span.end:
            return span
    return None


def _resolve_locator(given: str, spans: list[LocatorSpan]) -> LocatorSpan | None:
    """The source block `given` names: exactly, without its brackets, by
    case, as the unique real locator it ends with (`p2` for
    `<message-id>#p2`), or as the first real locator inside it (a range)."""
    cleaned = given.strip()
    if cleaned.startswith("[") and cleaned.endswith("]"):
        cleaned = cleaned[1:-1].strip()
    if not cleaned:
        return None
    by_locator = {span.locator: span for span in spans}
    if cleaned in by_locator:
        return by_locator[cleaned]
    folded = cleaned.lower()
    same_case_folded = [s for s in spans if s.locator.lower() == folded]
    if len(same_case_folded) == 1:
        return same_case_folded[0]
    tails = [s for s in spans if s.locator.lower().endswith(folded)]
    if len(tails) == 1:
        return tails[0]
    inside = [(cleaned.find(s.locator), -len(s.locator), s) for s in spans if s.locator in cleaned]
    if inside:
        return min(inside, key=lambda item: item[:2])[2]
    return None


def anchor_fact(
    fact: Fact,
    *,
    text: str,
    spans: list[LocatorSpan],
    quote_required: bool,
) -> Anchored:
    """Tie `fact` to its place in `text` (module docstring).

    `spans`: the source's locator index (`PreparedText.locators`). Empty
    means the caller has none, and only the quote is checked.
    `quote_required`: whether the fact's type needs a supported quote
    (`extract.schema.QUOTE_REQUIRED_TYPES`); a quote that is optional and
    not in the source is removed rather than stored as if it were said.
    """
    changes: list[str] = []
    updates: dict[str, object] = {}

    cited = _resolve_locator(fact.locator, spans) if spans else None

    match: QuoteMatch | None = None
    match_span: LocatorSpan | None = None
    has_quote = bool(fact.quote and fact.quote.strip())
    if has_quote:
        if cited is not None:
            # The cited block first: the same words may be said twice.
            local = locate_quote(fact.quote, text[cited.start : cited.end])
            if local is not None:
                match = QuoteMatch(
                    local.start + cited.start, local.end + cited.start, local.exact, local.score
                )
                match_span = cited
        if match is None:
            match = locate_quote(fact.quote, text)
            if match is not None and spans:
                match_span = _span_at(spans, match.start)

    problem: str | None = None
    if has_quote and match is None:
        if quote_required:
            problem = PROBLEM_QUOTE
        else:
            updates["quote"] = None
            changes.append("quote_dropped")
    elif match is not None and not match.exact:
        # The source's own words, when they are one unbroken stretch that
        # still fits the quote rule; otherwise the model's stay (supported).
        snapped = text[match.start : match.end]
        if "\n" not in snapped and len(snapped.split()) <= _MAX_QUOTE_WORDS:
            updates["quote"] = snapped
            changes.append("quote_snapped")

    if spans and problem is None:
        target = match_span or cited
        if target is None and len(spans) == 1:
            target = spans[0]  # a one-block source (a CRM row, a feed item)
        if target is None:
            problem = PROBLEM_LOCATOR
        elif target.locator != fact.locator:
            updates["locator"] = target.locator
            changes.append("locator_repaired")

    anchored = fact.model_copy(update=updates) if updates else fact
    return Anchored(fact=anchored, changes=changes, problem=problem)
