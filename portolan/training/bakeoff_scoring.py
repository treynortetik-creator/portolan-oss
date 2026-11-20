"""Scoring for the local-teacher bake-off (`scripts/data/local_teacher_bakeoff.py`,
`01-spec/local-teachers-bakeoff-2026-09-22.md`): one labeler's final facts
against a reference labeler's, by quote span and fact type.

Two facts match when their quote spans (`TrainingFact.quote_span`, offsets
into the same record text) overlap by at least `MIN_OVERLAP` of the shorter
span and, unless `require_type=False`, their fact types are equal. A fact
with no span can only match one with no span, the same type and the same
locator. Matching is one-to-one and pairs up as many facts as it can,
each trying its largest overlap first.

A majority vote clusters the facts of several labelers the same way and
keeps a cluster that at least `min_support` different labelers reached.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "MIN_OVERLAP",
    "LabelFact",
    "overlap_ratio",
    "facts_match",
    "match_facts",
    "PRScore",
    "precision_recall",
    "vote_facts",
]

MIN_OVERLAP = 0.5


@dataclass(frozen=True)
class LabelFact:
    fact_type: str
    span: tuple[int, int] | None
    locator: str = ""
    claim: str = ""

    @classmethod
    def from_training_fact(cls, fact: Any) -> LabelFact:
        span = tuple(fact.quote_span) if fact.quote_span is not None else None
        return cls(fact.fact_type, span, fact.locator, fact.claim)  # type: ignore[arg-type]


def overlap_ratio(a: tuple[int, int], b: tuple[int, int]) -> float:
    """Characters shared by two `[start, end)` spans over the shorter span's
    length; 0.0 when either span is empty."""
    shorter = min(a[1] - a[0], b[1] - b[0])
    if shorter <= 0:
        return 0.0
    shared = min(a[1], b[1]) - max(a[0], b[0])
    return max(0, shared) / shorter


def facts_match(
    a: LabelFact, b: LabelFact, *, require_type: bool = True, min_overlap: float = MIN_OVERLAP
) -> float:
    """The match strength of two facts of the same record: their overlap
    ratio (1.0 for two span-less facts on the same locator), or 0.0 when
    they do not match."""
    if require_type and a.fact_type != b.fact_type:
        return 0.0
    if a.span is None or b.span is None:
        return 1.0 if a.span is None and b.span is None and a.locator == b.locator else 0.0
    ratio = overlap_ratio(a.span, b.span)
    return ratio if ratio >= min_overlap else 0.0


def match_facts(
    predicted: Sequence[LabelFact],
    reference: Sequence[LabelFact],
    *,
    require_type: bool = True,
    min_overlap: float = MIN_OVERLAP,
) -> list[tuple[int, int]]:
    """One-to-one `(predicted index, reference index)` pairs: as many as
    possible (augmenting paths), each fact trying its strongest match first,
    so the result is deterministic. Sorted by predicted index."""
    options: list[list[int]] = []
    for p in predicted:
        scored = [
            (-facts_match(p, r, require_type=require_type, min_overlap=min_overlap), j)
            for j, r in enumerate(reference)
        ]
        options.append([j for neg, j in sorted(scored) if neg < 0])
    owner: dict[int, int] = {}

    def assign(i: int, seen: set[int]) -> bool:
        for j in options[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in owner or assign(owner[j], seen):
                owner[j] = i
                return True
        return False

    for i in range(len(predicted)):
        assign(i, set())
    return sorted((i, j) for j, i in owner.items())


@dataclass(frozen=True)
class PRScore:
    matched: int
    predicted: int
    reference: int

    @property
    def precision(self) -> float | None:
        return self.matched / self.predicted if self.predicted else None

    @property
    def recall(self) -> float | None:
        return self.matched / self.reference if self.reference else None

    @property
    def f1(self) -> float | None:
        total = self.predicted + self.reference
        return 2 * self.matched / total if total else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "matched": self.matched,
            "predicted": self.predicted,
            "reference": self.reference,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


def precision_recall(
    predicted: Mapping[str, Sequence[LabelFact]],
    reference: Mapping[str, Sequence[LabelFact]],
    *,
    documents: Sequence[str],
    require_type: bool = True,
    min_overlap: float = MIN_OVERLAP,
) -> PRScore:
    """Micro-averaged over `documents`; a document missing from either side
    counts as zero facts there."""
    matched = n_pred = n_ref = 0
    for doc in documents:
        p = list(predicted.get(doc, ()))
        r = list(reference.get(doc, ()))
        n_pred += len(p)
        n_ref += len(r)
        matched += len(match_facts(p, r, require_type=require_type, min_overlap=min_overlap))
    return PRScore(matched, n_pred, n_ref)


def vote_facts(
    labelers: Sequence[Sequence[LabelFact]],
    *,
    min_support: int = 2,
    min_overlap: float = MIN_OVERLAP,
) -> list[LabelFact]:
    """Facts of one record that at least `min_support` labelers agree on.
    Each labeler's facts join the first cluster whose founding fact they
    match (same type, overlapping span), at most one fact per labeler per
    cluster; a cluster's founding fact stands for it."""
    clusters: list[tuple[LabelFact, set[int]]] = []
    for who, facts in enumerate(labelers):
        for fact in facts:
            best = None
            best_strength = 0.0
            for index, (founder, members) in enumerate(clusters):
                if who in members:
                    continue
                strength = facts_match(fact, founder, min_overlap=min_overlap)
                if strength > best_strength:
                    best, best_strength = index, strength
            if best is None:
                clusters.append((fact, {who}))
            else:
                clusters[best][1].add(who)
    return [founder for founder, members in clusters if len(members) >= min_support]
