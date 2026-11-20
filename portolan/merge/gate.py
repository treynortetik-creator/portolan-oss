"""The candidate gate and the compare call (`01-spec/build-contracts.md`
"Merge": "no candidate above threshold -> new note, no model call").

One threshold, `GATE_SKIP_THRESHOLD = 0.45`: a best score below it is `new`
with no model call; at or above it, one compare call. The contract's 0.80
describes a strong candidate; it is not a second branch.

Score: `(cosine + difflib_ratio) / 2` over `plain_text` of both sides, so a
stored, wikilinked claim compares fairly against a fresh fact's prose.
"""

from __future__ import annotations

import difflib
import re

import psycopg

from portolan.extract.schema import Fact
from portolan.notes.models import Claim
from portolan.notes.persist import read_note
from portolan.notes.render import cosine_similarity, plain_text
from portolan.search.embed import Embedder
from portolan.search.query import search

GATE_SKIP_THRESHOLD = 0.45

COMPARE_SYSTEM = (
    "You compare one existing recorded claim about a customer to one newly "
    "extracted fact about the same customer. Decide whether the new fact "
    "is the SAME underlying fact already on file (same event, possibly "
    "reworded or from a different source), EXTENDS it with new detail "
    "about the same event, or CONTRADICTS it (the new fact states "
    "something that cannot both be true alongside the existing claim). "
    "Reply with your verdict and a one-line reason."
)


def _score_against(
    embedder: Embedder,
    plain_new: str,
    new_vec: list[float],
    claims: list[Claim],
    vector_cache: dict[int, list[float]] | None,
) -> tuple[Claim | None, float]:
    """The best-scoring claim in `claims` against an already-embedded
    `new_vec`/`plain_new`, and its score."""
    if not claims:
        return None, 0.0

    # Cached vectors are keyed by `claims.id` and never refreshed, so a
    # cache must not outlive the merge call that built it.
    if vector_cache is None:
        fresh = embedder.embed([plain_text(c.claim_text) for c in claims])
        vecs_by_id = {c.id: vec for c, vec in zip(claims, fresh, strict=True) if c.id is not None}
    else:
        missing = [c for c in claims if c.id is not None and c.id not in vector_cache]
        if missing:
            fresh = embedder.embed([plain_text(c.claim_text) for c in missing])
            for claim, vec in zip(missing, fresh, strict=True):
                vector_cache[claim.id] = vec  # type: ignore[index]
        vecs_by_id = vector_cache

    best_claim: Claim | None = None
    best_score = -1.0
    for claim in claims:
        vec = vecs_by_id.get(claim.id) if claim.id is not None else None
        if vec is None:
            vec = embedder.embed([plain_text(claim.claim_text)])[0]
        cosine = cosine_similarity(new_vec, vec)
        ratio = difflib.SequenceMatcher(None, plain_new, plain_text(claim.claim_text)).ratio()
        combined = (cosine + ratio) / 2
        if combined > best_score:
            best_score = combined
            best_claim = claim
    return best_claim, max(best_score, 0.0)


def gate(
    embedder: Embedder,
    new_claim_text: str,
    existing: list[Claim],
    *,
    vector_cache: dict[int, list[float]] | None = None,
    conn: psycopg.Connection | None = None,
    entity_slugs: list[str] | None = None,
    exclude_note_slugs: list[str] | None = None,
    cross_note_k: int = 8,
) -> tuple[Claim | None, float]:
    """The best-scoring unsuperseded claim against `new_claim_text`, and
    its combined score; `(None, 0.0)` when there is nothing to compare.
    Returns a candidate and a score only; the caller applies the threshold.

    A superseded claim is never a candidate: a new fact must not extend,
    agree with, or supersede again a claim the graph already corrected.

    `vector_cache`, keyed by `claims.id`, saves re-embedding the same
    claims for every fact in one merge call.

    Cross-note candidates ("candidate gate via `search()`") only when both
    `conn` and `entity_slugs` are given: claims from notes `search` finds
    linked to those entities join the pool, minus `exclude_note_slugs`
    (normally the home note, already in `existing`). Otherwise the gate is
    home-note-only.
    """
    # `read_note` returns superseded claims too (they render struck through).
    candidate_pool = [c for c in existing if c.superseded_at is None]

    if conn is not None and entity_slugs:
        exclude = set(exclude_note_slugs or ())
        # No reranker: this runs once per extracted fact, and a CPU cross-encoder
        # costs about a second per candidate note.
        cross_result = search(
            conn,
            embedder,
            new_claim_text,
            k=cross_note_k,
            entity_slugs=list(entity_slugs),
            reranker=False,
        )
        for hit in cross_result.hits:
            if hit.note_slug in exclude:
                continue
            _note, other_claims = read_note(conn, hit.note_slug)
            candidate_pool.extend(c for c in other_claims if c.superseded_at is None)

    if not candidate_pool:
        return None, 0.0

    plain_new = plain_text(new_claim_text)
    new_vec = embedder.embed([plain_new])[0]
    return _score_against(embedder, plain_new, new_vec, candidate_pool, vector_cache)


def build_compare_prompt(*, candidate: Claim, fact: Fact) -> str:
    return (
        f"Existing claim on file, recorded {candidate.recorded:%Y-%m-%d}, "
        f"cited to {candidate.source_slug} @ {candidate.locator}:\n"
        f"{re.sub(r'~~|→ superseded.*$', '', candidate.claim_text)}\n\n"
        f"New fact, locator {fact.locator}"
        f"{f', dated {fact.valid_from:%Y-%m-%d}' if fact.valid_from else ''}:\n"
        f"{fact.claim}"
    )
