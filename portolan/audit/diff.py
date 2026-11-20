"""`diff_facts`: match a source's first-pass claims (already in the graph)
against its second-pass extraction, per `01-spec/quality-loop.md` ("diff
the two fact sets programmatically: a fact matches if type matches,
entities overlap, and locators are within a window. Claim text similarity
breaks ties. Unmatched facts from the strong pass are candidate misses;
unmatched facts from the cheap pass are candidate inventions").

Reuses the bench's own matching primitives rather than reimplementing them
(instructions for this wave): `portolan.bench.locators.locator_matches`
for the locator-window rule, and `portolan.bench.extraction`'s type-
compatibility and claim-similarity helpers, including its exact
`_CLAIM_SIMILARITY_THRESHOLD` -- the same number
`01-spec/build-contracts.md` specifies for this module, so it is a re-use,
not a coincidence. Both are underscore-prefixed in `bench.extraction`
(private to that module) but are pure, self-contained functions with no
`Truth`/`PlantedFact` dependency, so importing them here is the "reuse the
bench's matching logic" instruction taken literally rather than
copy-pasted.

The one piece of the bench's matching this module cannot reuse as-is is
`Truth.entities_overlap`: it works from the synthetic corpus's truth-table
entity alias index, which does not exist for production data. `_entities_
overlap` below is this module's own, much smaller equivalent: normalised
(case/whitespace-insensitive) string overlap between a first-pass claim's
wikilinked entity mentions (`[[slug]]` tokens in `claim_text`, de-slugified
back to space-separated words) and a second-pass fact's raw `entities`
mention strings.

**Review finding G23 (fix wave 2026-09-12): two permissive holes,
resolved together.** (1) `_entities_overlap` used to return `True`
whenever *either* side had zero entities -- so a second-pass fact with no
`entities` at all (a genuinely vague extraction, not "nothing to check")
auto-matched every claim regardless of content, and a claim's own bare
`crm_id` mention (`"A-1010"`, `portolan.fake_extractor`'s shape) shared no
substring with a note's slug (`"ashgrove-health"`) and so silently failed
what should have been an obvious match -- both push the reported miss
rate in the flattering direction (fewer real misses surfaced, or a real
match wrongly counted as one). `diff_facts` now takes a `Registry`
(the only way to close the second half: resolving a bare id needs a
registry lookup) and both fixes land in `_entities_overlap`: it resolves
each side's mentions through the registry to a canonical name before
comparing, and only auto-passes when BOTH sides are empty -- "nothing to
check" requires there to really be nothing on either side, not just one.
(2) `superseded_at IS NULL`: an unmatched first-pass claim that has since
been superseded (G17's second clock) is not a "candidate invention" --
it is dead, replaced by a newer claim the second pass may reasonably not
have restated in exactly the same words. `diff_facts` filters it out of
`first` before matching (and see `report.run_audit`'s `_claims_for_source`
query for the same filter on the read side); `report.py` also stops
counting a `same`-verdict corroboration (G4's `claim_citations`) as
either a match or an invention -- it is neither a first-pass claim in
this diff's sense nor a fact the second pass has to independently
surface, so it is reported as its own count instead.
"""

from __future__ import annotations

import re
from collections import Counter

from portolan.audit.models import DiffCandidate, FactDiff
from portolan.bench.extraction import _CLAIM_SIMILARITY_THRESHOLD as CLAIM_SIMILARITY_THRESHOLD
from portolan.bench.extraction import _claim_similarity as claim_similarity
from portolan.bench.extraction import _type_credit as type_credit
from portolan.bench.locators import locator_matches
from portolan.extract.schema import ExtractionResult, Fact
from portolan.notes.models import Claim
from portolan.notes.render import WIKILINK_RE
from portolan.registry import Registry

__all__ = ["CLAIM_SIMILARITY_THRESHOLD", "diff_facts"]

_NORM_WS_RE = re.compile(r"\s+")


def _norm(s: str) -> str:
    """Lowercase, whitespace-collapsed, hyphen-insensitive. Hyphens are
    folded to spaces (not stripped) so a `[[tidewater-t2]]` wikilink slug
    (recovered as `"tidewater-t2"`) and a raw `entities` string of
    `"tidewater-t2"` or `"Tidewater T2"` all normalise to the same
    `"tidewater t2"` -- both shapes appear in practice, since a second-pass
    `Fact.entities` string is sometimes a registry id/slug and sometimes a
    natural-language mention (`portolan.extract.schema.Fact.entities`
    docstring makes no promise either way)."""
    return _NORM_WS_RE.sub(" ", s.strip().lower().replace("-", " "))


def _claim_entity_names(claim: Claim) -> list[str]:
    """The entity mentions a first-pass claim carries, recovered from its
    `[[slug]]` wikilinks (`portolan.notes.render.link_mentions` is what put
    them there) rather than from a dedicated field -- `Claim` has none
    (`portolan/notes/models.py`)."""
    return [m.group(1) for m in WIKILINK_RE.finditer(claim.claim_text)]


def _resolved_names(registry: Registry, mentions: list[str]) -> set[str]:
    """`mentions`, normalised, PLUS each one's registry-resolved canonical
    name when it resolves to something (review finding G23): closes the
    bare-`crm_id` gap ("A-1010" now also contributes "ashgrove health",
    not just "a 1010") without dropping the raw string, which still
    matters for a mention the registry doesn't know (a competitor, a
    product not yet in this tenant's registry, ...)."""
    names: set[str] = set()
    for mention in mentions:
        names.add(_norm(mention))
        entity = registry.get(mention)
        if entity is not None:
            names.add(_norm(entity.canonical_name))
    return names


def _entities_overlap(
    registry: Registry, claim_entities: list[str], fact_entities: list[str]
) -> bool:
    """True when both sides name at least one entity and they share a
    normalised (registry-resolved) name, or NEITHER side names any entity
    at all (review finding G23: "nothing to check" now requires there to
    really be nothing on either side -- a fact with zero `entities` no
    longer auto-matches a claim that names one, or vice versa).

    "Share a name" allows substring containment as well as exact equality
    (`"halvardcare" in "halvardcare service contract"`), not just set
    equality: a second-pass fact's `entities` sometimes names an account
    by a short product/competitor id and sometimes by the fuller string a
    note's title (and so its `[[slug]]` wikilink) actually uses --
    `extraction-schema-v0.md`'s own example shows `Fact.entities` as
    slug-like strings derived from the mention (`"ashgrove-medical"`), not
    a promise that they equal any particular note's exact slug.
    """
    if not claim_entities and not fact_entities:
        return True
    a = _resolved_names(registry, claim_entities)
    b = _resolved_names(registry, fact_entities)
    if a & b:
        return True
    return any(x and y and (x in y or y in x) for x in a for y in b)


def _evaluate_pair(
    registry: Registry, claim: Claim, fact: Fact, source_type: str
) -> tuple[bool, float, float]:
    """(is_match, credit, similarity) for one (first-pass claim,
    second-pass fact) pair, the four AND'd criteria from quality-loop.md
    (module docstring)."""
    compatible, credit = type_credit(claim.fact_type, fact.type.value)
    entities_ok = _entities_overlap(registry, _claim_entity_names(claim), fact.entities)
    locator_ok = locator_matches(source_type, claim.locator, fact.locator)
    similarity = claim_similarity(claim.claim_text, fact.claim)
    similarity_ok = similarity >= CLAIM_SIMILARITY_THRESHOLD

    is_match = compatible and entities_ok and locator_ok and similarity_ok
    return is_match, (credit if is_match else 0.0), similarity


def diff_facts(
    first: list[Claim], second: ExtractionResult, source_type: str, registry: Registry
) -> FactDiff:
    """Match `first` (the unarchived claims already on file for one source,
    `claims.source_slug`-scoped) against `second.facts` (that same
    source's strong-model second pass). A greedy 1:1 assignment (strongest
    match claimed first) mirrors `bench.extraction.score_extraction`'s own
    bipartite matching so one fact on either side cannot double-count.

    Every unmatched second-pass fact becomes a `kind="miss"` candidate
    (something the first pass did not surface); every unmatched first-pass
    claim becomes a `kind="invention"` candidate (something the first pass
    surfaced that the more thorough second pass did not corroborate).
    Each candidate carries its closest match on the other side (whether or
    not that match actually passed) as context for `judge()`.

    `registry` (review finding G23) resolves a mention on either side to
    its canonical name before comparing. A superseded claim (G17's second
    clock) is dropped from `first` before matching at all (module
    docstring, G20): it is dead, not a candidate invention. Its
    `claim_citations` (G4 corroborations) are counted and reported
    separately, never folded into `misses`/`inventions`.
    """
    source_slug = second.source_id
    facts = list(second.facts)
    first = [c for c in first if c.superseded_at is None]
    corroborations_count = sum(len(c.citations) for c in first)

    pair_scores: list[tuple[bool, float, float, int, int]] = []
    for ci, claim in enumerate(first):
        for fi, fact in enumerate(facts):
            is_match, credit, similarity = _evaluate_pair(registry, claim, fact, source_type)
            pair_scores.append((is_match, credit, similarity, ci, fi))

    pair_scores.sort(key=lambda t: (not t[0], -t[1], -t[2]))

    matched_claims: set[int] = set()
    matched_facts: set[int] = set()
    for is_match, _credit, _similarity, ci, fi in pair_scores:
        if not is_match:
            continue
        if ci in matched_claims or fi in matched_facts:
            continue
        matched_claims.add(ci)
        matched_facts.add(fi)

    on_file = [(c.id, c.claim_text) for c in first if c.id is not None]
    misses: list[DiffCandidate] = []
    for fi, fact in enumerate(facts):
        if fi in matched_facts:
            continue
        best_similarity = 0.0
        best_claim: Claim | None = None
        for claim in first:
            similarity = claim_similarity(claim.claim_text, fact.claim)
            if similarity >= best_similarity:
                best_similarity = similarity
                best_claim = claim
        misses.append(
            DiffCandidate(
                kind="miss",
                source_slug=source_slug,
                fact_type=fact.type.value,
                claim_text=fact.claim,
                locator=fact.locator,
                best_match_claim_id=best_claim.id if best_claim else None,
                best_match_claim_text=best_claim.claim_text if best_claim else None,
                best_similarity=best_similarity,
                on_file=on_file,
            )
        )

    inventions: list[DiffCandidate] = []
    for ci, claim in enumerate(first):
        if ci in matched_claims:
            continue
        best_similarity = 0.0
        best_fact: Fact | None = None
        for fact in facts:
            similarity = claim_similarity(claim.claim_text, fact.claim)
            if similarity >= best_similarity:
                best_similarity = similarity
                best_fact = fact
        inventions.append(
            DiffCandidate(
                kind="invention",
                source_slug=source_slug,
                fact_type=claim.fact_type,
                claim_text=claim.claim_text,
                locator=claim.locator,
                claim_id=claim.id,
                best_match_claim_text=best_fact.claim if best_fact else None,
                best_similarity=best_similarity,
            )
        )

    return FactDiff(
        source_slug=source_slug,
        source_type=source_type,
        matched_count=len(matched_claims),
        first_pass_count=len(first),
        second_pass_count=len(facts),
        fact_type_totals=dict(Counter(f.type.value for f in facts)),
        misses=misses,
        inventions=inventions,
        corroborations_count=corroborations_count,
    )
