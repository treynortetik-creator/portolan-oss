"""Pydantic models for `portolan.audit` (`01-spec/quality-loop.md` "The
production audit", `01-spec/build-contracts.md` "Report, audit,
consolidation").

Shapes proposed by this wave (C2), per build-contracts.md's own note that
"the C agents propose signatures in their reports" for this module.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

DiffCandidateKind = Literal["miss", "invention"]
JudgeVerdictLabel = Literal["genuine_miss", "paraphrase_of_existing", "not_a_fact"]


class UsageTotals(BaseModel):
    """Mirrors `portolan.pipeline.UsageTotals`'s shape (not imported, to
    keep `portolan.audit` self-contained against a module this wave does
    not own): calls, tokens, cost, rolled up per surface or per source."""

    model_config = ConfigDict(extra="forbid")

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    cost_estimate: float = 0.0


class DiffCandidate(BaseModel):
    """One fact `diff_facts` could not match to the other pass.

    `kind="miss"`: a second-pass (strong-model) fact with no matching
    first-pass claim -- candidate evidence the cheap pass missed it.
    `kind="invention"`: a first-pass claim (`claim_id` set) with no
    matching second-pass fact -- candidate evidence the cheap pass
    invented or over-split it. `best_match_*` carries the closest
    candidate on the *other* side regardless of whether it actually
    passed the match criteria (mirrors `bench.extraction.UnmatchedFact.
    best_candidate_claim`), so `judge()` has something concrete to grade
    against even when nothing matched formally.
    """

    model_config = ConfigDict(extra="forbid")

    kind: DiffCandidateKind
    source_slug: str
    fact_type: str
    claim_text: str
    locator: str
    claim_id: int | None = None
    best_match_claim_id: int | None = None
    best_match_claim_text: str | None = None
    best_similarity: float = 0.0
    # For a miss: every first-pass claim on file for the source, as (id,
    # text). The judge reads all of them: a paraphrase is worded unlike the
    # claim it restates, so the closest by wording is often the wrong one.
    on_file: list[tuple[int, str]] = Field(default_factory=list)


class FactDiff(BaseModel):
    """`diff_facts`'s return: one source's first-pass claims diffed
    against its second-pass extraction."""

    model_config = ConfigDict(extra="forbid")

    source_slug: str
    source_type: str
    matched_count: int = 0
    first_pass_count: int = 0
    second_pass_count: int = 0
    # Count of second-pass facts by fact_type (matched and missed both) --
    # the denominator `run_audit` needs for a per-fact-type miss rate,
    # since `misses` alone only carries the numerator.
    fact_type_totals: dict[str, int] = Field(default_factory=dict)
    misses: list[DiffCandidate] = Field(default_factory=list)
    inventions: list[DiffCandidate] = Field(default_factory=list)
    # Review finding G20: a `same`-verdict corroboration (G4's
    # `claim_citations`) is neither a match nor an invention -- counted
    # here, separately, instead of inflating either.
    corroborations_count: int = 0


class Judgement(BaseModel):
    """`judge()`'s per-candidate verdict."""

    model_config = ConfigDict(extra="forbid")

    candidate: DiffCandidate
    verdict: JudgeVerdictLabel
    matched_claim_id: int | None = None
    reason: str = ""


class WorstSource(BaseModel):
    """One row of the report's "ten worst sources" table.

    `link` (review finding G22, this fix wave): plain text, `source_slug`
    verbatim, never a `[[wikilink]]` -- `sources` rows are not notes, so a
    wikilinked source slug was a `links` row the lint's `broken_link`
    check could never resolve."""

    model_config = ConfigDict(extra="forbid")

    source_slug: str
    source_type: str
    genuine_misses: int
    inventions: int
    link: str
    usage: UsageTotals = Field(default_factory=UsageTotals)


class TypeRate(BaseModel):
    """Denominator/numerator pair for a per-type (fact type or source
    type) miss rate, kept as counts rather than a bare float so the
    report can show "3 of 40" alongside the percentage."""

    model_config = ConfigDict(extra="forbid")

    total: int = 0
    misses: int = 0

    @property
    def miss_rate(self) -> float | None:
        return self.misses / self.total if self.total else None


class AuditReport(BaseModel):
    """Everything `run_audit` computed and rendered for one run."""

    model_config = ConfigDict(extra="forbid")

    note_slug: str
    since: date
    rate: float
    seed: str
    sources_sampled: int = 0
    sources_second_passed: int = 0
    first_pass_claims_total: int = 0
    second_pass_facts_total: int = 0
    genuine_misses_total: int = 0
    inventions_total: int = 0
    # Review finding 12: a miss candidate `judge()` never returned a
    # `Judgement` for at all (a gap in the judge call, not assumed to be
    # a genuine miss) -- see `run_audit`'s own docstring comment for why
    # this replaced a silent "default to genuine_miss".
    unjudged_misses_total: int = 0
    # Review finding G20: corroborations (G4 `claim_citations`), counted
    # separately from misses/inventions -- not a genuine miss, not a
    # candidate invention, just evidence the fact was seen more than once.
    corroborations_total: int = 0
    miss_rate_overall: float | None = None
    invention_rate_overall: float | None = None
    # Review finding G10: `miss_rate_overall` pools raw counts across
    # strata sampled at very different rates (a rare source type sampled
    # near 100%, a common one at 2%), over-representing the rare one by
    # orders of magnitude. `miss_rate_weighted` weights each source
    # type's rate by its real population share instead (`population_by_
    # source_type`, from `audit.sample.population_counts`) -- the more
    # honest headline number.
    miss_rate_weighted: float | None = None
    population_by_source_type: dict[str, int] = Field(default_factory=dict)
    miss_by_fact_type: dict[str, TypeRate] = Field(default_factory=dict)
    miss_by_source_type: dict[str, TypeRate] = Field(default_factory=dict)
    worst_sources: list[WorstSource] = Field(default_factory=list)
    judgements: list[Judgement] = Field(default_factory=list)
    skipped_sources: list[str] = Field(default_factory=list)
    tokens_input: int = 0
    tokens_output: int = 0
    tokens_cached: int = 0
    cost_estimate: float = 0.0
