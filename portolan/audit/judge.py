"""`judge`: grade the diff's unmatched remainder with a judge call
(`01-spec/quality-loop.md`: "a judge call grades only the unmatched
remainder (genuine miss, paraphrase of a matched fact, or not a fact)").

Scoped to `kind="miss"` candidates only (`run_audit` is what enforces
this, by only ever passing misses to `judge()`; `judge()` itself is
generic over whatever `DiffCandidate` list it is given). An "invention"
candidate -- something the cheap pass claimed that the more thorough
second pass did not corroborate -- does not fit this module's three-label
taxonomy (it was not "missed"; there is nothing to call a paraphrase of,
since the second pass is what would be doing the paraphrasing); the
production audit reports an invention rate as a plain count/rate instead
(`run_audit`, `01-spec/quality-loop.md`: "report: miss rate ..., invention
rate, ..."), same design bench's own `_run_judge` (`portolan.bench.
extraction`) follows for its `UnmatchedFact`s.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from portolan.audit.models import DiffCandidate, Judgement, JudgeVerdictLabel

# `JudgeClient` used to be a locally-defined `Protocol`, structurally
# identical to `portolan.llm.base.LLMClient` (build-contracts.md),
# "duplicated to avoid an import" (slop list, fix wave 2026-09-12) -- an
# import is exactly what a `Protocol` needs no more of a real dependency
# than a duck-typed test fake already tolerates, so the alias replaces it.
from portolan.llm.base import LLMClient as JudgeClient

__all__ = ["JUDGE_SYSTEM", "JudgeClient", "JudgeVerdict", "judge"]

JUDGE_SYSTEM = (
    "You are grading a candidate miss from a production audit of an "
    "extraction pipeline. A stronger model re-read the same source and "
    "produced a fact the cheaper model's pass did not surface. Decide "
    "whether that is a genuine miss, a paraphrase of a fact already on "
    "file (name which one), or not really a fact at all.\n\n"
    "- genuine_miss: the fact is real and not covered by anything on file.\n"
    "- paraphrase_of_existing: the fact is already on file, just worded "
    "differently -- name the matched claim id.\n"
    "- not_a_fact: neither text describes a real, checkable fact."
)


class JudgeVerdict(BaseModel):
    """The judge call's structured reply."""

    model_config = ConfigDict(extra="forbid")

    verdict: JudgeVerdictLabel
    matched_claim_id: int | None = None
    reason: str = ""


# Claims listed to the judge per candidate; a source with more keeps the
# closest by wording first and says how many were left out.
ON_FILE_LIMIT = 80


def _user_message(candidate: DiffCandidate) -> str:
    matched = (
        "Closest existing claim on file by wording"
        + (f" (id {candidate.best_match_claim_id})" if candidate.best_match_claim_id else "")
        + f":\n{candidate.best_match_claim_text or '(nothing close on file)'}"
    )
    message = (
        f"Second-pass fact ({candidate.fact_type}), source {candidate.source_slug} "
        f"@ {candidate.locator}:\n{candidate.claim_text}\n\n{matched}"
    )
    if candidate.on_file:
        shown = candidate.on_file[:ON_FILE_LIMIT]
        lines = [f"- (id {claim_id}) {text}" for claim_id, text in shown]
        left_out = len(candidate.on_file) - len(shown)
        if left_out:
            lines.append(f"({left_out} more claims on file are not shown.)")
        message += (
            "\n\nEvery claim on file for this source (a paraphrase may be worded quite "
            "unlike the fact; name its id if one of these already says it):\n" + "\n".join(lines)
        )
    return message


def _names_a_claim_on_file(candidate: DiffCandidate, claim_id: int | None) -> bool:
    known = {cid for cid, _text in candidate.on_file}
    if candidate.best_match_claim_id is not None:
        known.add(candidate.best_match_claim_id)
    return claim_id is not None and claim_id in known


def judge(llm: JudgeClient, candidates: list[DiffCandidate]) -> list[Judgement]:
    """One structured judge call per candidate (`surface="audit"`,
    `json_schema=JudgeVerdict.model_json_schema()`).

    The judge reads every first-pass claim on file for the candidate's source
    (`DiffCandidate.on_file`), not only the closest by wording. A
    `paraphrase_of_existing` verdict takes a miss off the count, so it stands
    only when it names a claim the judge was shown; one that names any other
    id, or none, returns no `Judgement` for that candidate, and the report
    counts the candidate as unjudged."""
    out: list[Judgement] = []
    for candidate in candidates:
        result = llm.complete(
            system=JUDGE_SYSTEM,
            user=_user_message(candidate),
            json_schema=JudgeVerdict.model_json_schema(),
            surface="audit",
        )
        verdict = JudgeVerdict.model_validate(result.data)
        if verdict.verdict == "paraphrase_of_existing" and not _names_a_claim_on_file(
            candidate, verdict.matched_claim_id
        ):
            continue
        out.append(
            Judgement(
                candidate=candidate,
                verdict=verdict.verdict,
                matched_claim_id=verdict.matched_claim_id,
                reason=verdict.reason,
            )
        )
    return out
