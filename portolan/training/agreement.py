"""The agreement gate (silver-labels build instructions item 2): "A fact
is kept when both teachers produced a fact of the same `fact_type` whose
quotes overlap ... and the quote passes `quote_is_supported` against the
document text."

Quote normalization reuses `portolan.extract.quote_gate`'s own helpers
(`_normalize_for_containment`, `_fuzzy_quote_containment`,
`FUZZY_QUOTE_THRESHOLD`) rather than a second, possibly-drifting
definition of "the same quote, near enough" -- the build instructions
call this out explicitly ("normalize with `portolan/extract/quote_gate.py`
helpers"). `quote_is_supported` (the same module's public gate) is the
final check against the real document text: a fact whose quote is a
plausible paraphrase match between the two teachers but does not actually
appear in the source still gets dropped -- rule 5's own bar, applied here
before a fact ever reaches a training record.

Both inputs to `evaluate_document` are `portolan.extract.schema.Fact`
instances -- the real production type, not a duck-typed stand-in -- so a
test constructing `Fact(...)` directly exercises the exact validation
(quote length, closed type list) a live extraction result would already
have passed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from portolan.extract.quote_gate import (
    FUZZY_QUOTE_THRESHOLD,
    _fuzzy_quote_containment,
    _normalize_for_containment,
    quote_is_supported,
)
from portolan.extract.schema import Fact
from portolan.training.records import teacher_fact_fields

__all__ = ["AgreementOutcome", "quotes_overlap", "match_facts", "evaluate_document"]


def quotes_overlap(quote_a: str | None, quote_b: str | None) -> bool:
    """Two teachers' quotes "overlap" when, after `quote_gate`'s own
    glyph-fold/whitespace normalization, one is an exact substring of the
    other, or a fuzzy token-subsequence match in either direction clears
    `FUZZY_QUOTE_THRESHOLD` -- the same near-verbatim tolerance
    `quote_is_supported` grants a single quote against source text,
    applied here between two teachers' independently-written quotes.
    `None`/blank on either side never overlaps (a fact with no quote
    cannot be corroborated this way -- see module docstring: it will also
    fail `quote_is_supported` later, so it is dropped either way, just
    earlier)."""
    if not quote_a or not quote_a.strip() or not quote_b or not quote_b.strip():
        return False
    norm_a = _normalize_for_containment(quote_a)
    norm_b = _normalize_for_containment(quote_b)
    if norm_a == norm_b or norm_a in norm_b or norm_b in norm_a:
        return True
    return (
        _fuzzy_quote_containment(quote_a, quote_b) >= FUZZY_QUOTE_THRESHOLD
        or _fuzzy_quote_containment(quote_b, quote_a) >= FUZZY_QUOTE_THRESHOLD
    )


def match_facts(facts_a: list[Fact], facts_b: list[Fact]) -> list[tuple[Fact, Fact]]:
    """Greedy one-to-one pairing: for each fact in `facts_a` (in order),
    the first not-yet-used fact in `facts_b` with the same `type` whose
    quote overlaps it (`quotes_overlap`) is its match. A `facts_b` fact
    used in one pair is never reused for a later `facts_a` fact -- so two
    facts of the same type in `facts_a` that both plausibly match one
    `facts_b` fact only pair with it once, and the second one is counted
    as a disagreement rather than double-counted."""
    remaining_b = list(facts_b)
    pairs: list[tuple[Fact, Fact]] = []
    for fact_a in facts_a:
        for fact_b in remaining_b:
            if fact_a.type == fact_b.type and quotes_overlap(fact_a.quote, fact_b.quote):
                pairs.append((fact_a, fact_b))
                remaining_b.remove(fact_b)
                break
    return pairs


@dataclass
class AgreementOutcome:
    """`evaluate_document`'s result.

    `kept`: one dict per agreed-and-supported fact, already in
    `portolan.training.records.TrainingFact`'s field names (`fact_type`,
    `claim`, `quote`, `locator`, `entities`, and the teacher's own
    `TEACHER_FACT_FIELDS`) -- built from `fact_a` of
    each matched pair (the first teacher's own wording), so which
    teacher "wins" the record's claim/locator/entities text is
    deterministic and documented, not an arbitrary merge.
    `dropped_disagreement`: raw fact count (from both teachers combined)
    that never found a same-type, quote-overlapping partner -- a type
    mismatch, or a quote that didn't overlap either teacher's other
    facts.
    `dropped_quote_gate`: count of matched pairs (type agreed, quotes
    overlapped each other) whose quote still failed
    `quote_is_supported` against the real document text -- corroborated
    by both teachers, but not actually in the source.
    `agreement`: `len(kept) / max(len(facts_a), len(facts_b))`, or `1.0`
    when both teachers produced zero facts (build instructions item 2:
    "keep documents where both teachers returned zero facts as
    negatives" -- perfect agreement that there was nothing to extract).
    """

    kept: list[dict] = field(default_factory=list)
    dropped_disagreement: int = 0
    dropped_quote_gate: int = 0
    agreement: float = 0.0

    @property
    def is_negative(self) -> bool:
        return len(self.kept) == 0


def _to_training_fact(fact: Fact) -> dict:
    return {
        "fact_type": fact.type.value if hasattr(fact.type, "value") else str(fact.type),
        "claim": fact.claim,
        "quote": fact.quote,
        "locator": fact.locator,
        "entities": list(fact.entities),
        **teacher_fact_fields(fact),
    }


def evaluate_document(facts_a: list[Fact], facts_b: list[Fact], doc_text: str) -> AgreementOutcome:
    """The full gate for one document: match, then check each matched
    pair's quote against `doc_text`."""
    pairs = match_facts(facts_a, facts_b)
    matched_a_ids = {id(a) for a, _ in pairs}
    matched_b_ids = {id(b) for _, b in pairs}
    dropped_disagreement = sum(1 for f in facts_a if id(f) not in matched_a_ids) + sum(
        1 for f in facts_b if id(f) not in matched_b_ids
    )

    kept: list[dict] = []
    dropped_quote_gate = 0
    for fact_a, _fact_b in pairs:
        if quote_is_supported(fact_a.quote, doc_text):
            kept.append(_to_training_fact(fact_a))
        else:
            dropped_quote_gate += 1

    denom = max(len(facts_a), len(facts_b))
    agreement = 1.0 if denom == 0 else len(kept) / denom

    return AgreementOutcome(
        kept=kept,
        dropped_disagreement=dropped_disagreement,
        dropped_quote_gate=dropped_quote_gate,
        agreement=agreement,
    )
