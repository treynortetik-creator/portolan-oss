"""Three-judge dispute resolution for extraction (`01-spec/world-generator-
v2-2026-09-22.md` section 7, ERB idea (a); `01-spec/research/enterpriserag-
bench-2026-09-22.md` borrowable 3).

A *dispute* is a produced fact the labels do not account for: an off-span
fact or a decoy hit on a v2 world, or an unplanted fact on the Halvard
exam. Three judges, three different models, each read the document and
vote on an ordinal scale: `supported`, `partial` or `unsupported`.

- A `supported` vote must quote the document; if its evidence does not
  resolve to a span of the text, the vote counts as `partial`.
- A judge call that fails counts as a non-supported vote (ties go to the
  labels' side).
- The labels' side wins as soon as two votes are not `supported`, so the
  third judge is skipped then. Otherwise all three vote, so every
  proposal carries a 3-0 or 2-1 count.
- Two or more `supported` votes make a proposed label correction, written
  to `corrections.jsonl` with the fact, the votes, the judge models and
  their rationales. Nothing is applied: a label set or truth table is
  never edited here.

Cost control: judges run on disputes only. Before any call,
`resolve_disputes` prints two numbers: the *expected* cost (every judge on
every dispute, one request each, `EXPECTED_OUTPUT_TOKENS` out) and the
*ceiling* (full `max_tokens` out and every request the client may bill for
one call: its JSON fallback can send up to `MAX_BILLED_REQUESTS_PER_CALL`).
It refuses to start when the expected cost exceeds `max_usd`, and it never
starts a call whose ceiling could take spend past `max_usd`, so the cap is
never crossed. Real spend is recorded per call in
`portolan.training.spend.SpendMeter` under corpus key `bench_judges`.
"""

from __future__ import annotations

import datetime as dt
import json
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from portolan.extract.schema import FACT_TYPES
from portolan.training.quote_span import resolve_quote_span
from portolan.training.spend import SpendMeter

__all__ = [
    "CORPUS_KEY",
    "DEFAULT_JUDGES",
    "BudgetEstimateExceeded",
    "Dispute",
    "DisputeOutcome",
    "DisputeRun",
    "JudgeBackend",
    "JudgePanel",
    "JudgeReply",
    "JudgeVote",
    "OpenRouterJudgeBackend",
    "build_judge_prompt",
    "estimate_usd",
    "fetch_openrouter_prices",
    "halvard_disputes",
    "resolve_disputes",
    "summarize",
    "write_dispute_logs",
]

CORPUS_KEY = "bench_judges"
# Three families, none of them the extractor under test (Nova), the
# labeling teacher (Kimi K2) or a writer-rotation family.
DEFAULT_JUDGES: tuple[str, ...] = (
    "anthropic/claude-sonnet-5",
    "google/gemini-3.8-flash",
    "openai/gpt-5.4-mini",
)
# Room for a judge that cannot turn reasoning off: Gemini 3.8 Flash spends
# about 400 reasoning tokens before its verdict, and 600 cut replies short.
JUDGE_MAX_TOKENS = 2_000
EXPECTED_OUTPUT_TOKENS = 500
# `OpenRouterClient.complete(json_schema=...)`: the native attempt, then the
# json_util fallback and its one retry.
MAX_BILLED_REQUESTS_PER_CALL = 3
# A document longer than this is sent as a window around the fact.
MAX_DOC_CHARS = 30_000
# Conservative characters per token for the pre-call estimate.
CHARS_PER_TOKEN = 3.0

Verdict = Literal["supported", "partial", "unsupported", "error"]
Decision = Literal["supported", "not_supported", "not_judged"]


class BudgetEstimateExceeded(RuntimeError):
    def __init__(self, estimate_usd: float, max_usd: float, disputes: int) -> None:
        super().__init__(
            f"expected judge cost ${estimate_usd:.4f} for {disputes} disputes "
            f"exceeds --max-usd ${max_usd:.2f}; no judge was called"
        )
        self.estimate_usd = estimate_usd
        self.max_usd = max_usd


class Dispute(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str
    produced_id: str
    category: str
    fact_type: str
    claim: str
    quote: str | None = None
    locator: str = ""
    entities: list[str] = Field(default_factory=list)
    span: tuple[int, int] | None = None
    decoy_class_id: str | None = None
    source_type: str | None = None
    source_date: str | None = None

    @property
    def dispute_id(self) -> str:
        return f"{self.source_id}/{self.produced_id}"


class JudgeVote(BaseModel):
    model_config = ConfigDict(extra="forbid")

    judge_model: str
    verdict: Verdict
    effective: Verdict
    type_fits: bool | None = None
    evidence: str | None = None
    evidence_resolves: bool = False
    rationale: str = ""
    cost_usd: float = 0.0
    error: str | None = None


class DisputeOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dispute: Dispute
    votes: list[JudgeVote] = Field(default_factory=list)
    decision: Decision = "not_judged"
    note: str | None = None

    @property
    def supported_votes(self) -> int:
        return sum(1 for v in self.votes if v.effective == "supported")

    @property
    def tally(self) -> str:
        return f"{self.supported_votes}-{len(self.votes) - self.supported_votes}"


@dataclass
class JudgeReply:
    data: dict[str, Any] | None
    cost_usd: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None


class JudgeBackend(Protocol):
    def judge(self, *, model: str, system: str, user: str, max_tokens: int) -> JudgeReply: ...


@dataclass
class JudgePanel:
    """Three distinct judge models, their per-token prices (prompt,
    completion) for the estimate, and the backend that calls them."""

    models: tuple[str, ...]
    prices: dict[str, tuple[float, float]]
    backend: JudgeBackend
    max_tokens: int = JUDGE_MAX_TOKENS

    def __post_init__(self) -> None:
        if len(self.models) != 3 or len(set(self.models)) != 3:
            raise ValueError(f"need three different judge models, got {list(self.models)}")
        missing = [m for m in self.models if m not in self.prices]
        if missing:
            raise ValueError(f"no price for judge model(s) {missing}; cannot estimate cost")


# --------------------------------------------------------------------------
# Prompt
# --------------------------------------------------------------------------

JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["supported", "partial", "unsupported"]},
        "type_fits": {"type": "boolean"},
        "evidence": {"type": "string"},
        "rationale": {"type": "string"},
    },
    "required": ["verdict", "type_fits", "evidence", "rationale"],
    "additionalProperties": False,
}


def _type_lines() -> str:
    return "\n".join(f"- {t.value}: {info.definition}" for t, info in FACT_TYPES.items())


JUDGE_SYSTEM = (
    "You check one fact that an automated extractor pulled from a business document "
    "about a vendor and its customer accounts. Decide whether the document itself "
    "states the fact.\n\n"
    "Verdicts:\n"
    "- supported: a participant says it, or the record states it, as real; the claim "
    "matches the text in every detail that matters (numbers, dates, names, who said "
    "it). Resolving a pronoun, first name or handle from the rest of the document is "
    "fine.\n"
    "- partial: the document says something close, but the claim adds, drops or "
    "changes a detail that matters, or the fact is only implied.\n"
    "- unsupported: the document does not state it, or states it only as a "
    "hypothetical, a question, a negation, a rumour about another company, general "
    "industry talk, or a past issue told as already resolved.\n\n"
    "Also say whether the fact type fits, by these definitions:\n"
    f"{_type_lines()}\n\n"
    "evidence: copy the one sentence or line of the document that best supports the "
    "fact, character for character, or an empty string if none does. rationale: one "
    "or two short sentences. Reply with the JSON object only."
)


def _window(text: str, span: tuple[int, int] | None) -> tuple[str, bool]:
    if len(text) <= MAX_DOC_CHARS:
        return text, False
    center = (span[0] + span[1]) // 2 if span else 0
    start = max(0, min(center - MAX_DOC_CHARS // 2, len(text) - MAX_DOC_CHARS))
    return text[start : start + MAX_DOC_CHARS], True


def build_judge_prompt(dispute: Dispute, text: str) -> tuple[str, str]:
    doc, excerpt = _window(text, dispute.span)
    header = f"Document {dispute.source_id}"
    if dispute.source_type:
        header += f", a {dispute.source_type}"
    if dispute.source_date:
        header += f", dated {dispute.source_date}"
    if excerpt:
        header += f" (an excerpt of {len(doc)} of its {len(text)} characters)"
    user = (
        f"Fact type: {dispute.fact_type}\n"
        f"Claim: {dispute.claim}\n"
        f"Quote the extractor gave: {dispute.quote or '(none)'}\n"
        f"Locator the extractor cited: {dispute.locator or '(none)'}\n\n"
        f"{header}:\n<<<\n{doc}\n>>>"
    )
    return JUDGE_SYSTEM, user


# --------------------------------------------------------------------------
# Cost
# --------------------------------------------------------------------------


def _call_cost(panel: JudgePanel, model: str, system: str, user: str, *, ceiling: bool) -> float:
    """One judge call: expected (one request, `EXPECTED_OUTPUT_TOKENS` out)
    or its ceiling (every billable request, full `max_tokens` out)."""
    prompt_price, completion_price = panel.prices[model]
    tokens_in = math.ceil((len(system) + len(user)) / CHARS_PER_TOKEN)
    if not ceiling:
        return tokens_in * prompt_price + EXPECTED_OUTPUT_TOKENS * completion_price
    one = tokens_in * prompt_price + panel.max_tokens * completion_price
    return MAX_BILLED_REQUESTS_PER_CALL * one


def estimate_usd(
    panel: JudgePanel, disputes: Iterable[Dispute], texts: dict[str, str]
) -> tuple[float, float]:
    """(expected, ceiling) for every judge on every dispute."""
    expected = ceiling = 0.0
    for d in disputes:
        system, user = build_judge_prompt(d, texts.get(d.source_id, ""))
        for m in panel.models:
            expected += _call_cost(panel, m, system, user, ceiling=False)
            ceiling += _call_cost(panel, m, system, user, ceiling=True)
    return expected, ceiling


def fetch_openrouter_prices(
    models: Iterable[str], *, timeout: float = 30.0
) -> dict[str, tuple[float, float]]:
    """Per-token (prompt, completion) prices from OpenRouter's public model
    list. A model missing from the list is left out, and `JudgePanel`
    refuses to start without it."""
    import httpx

    wanted = set(models)
    response = httpx.get("https://openrouter.ai/api/v1/models", timeout=timeout)
    response.raise_for_status()
    prices: dict[str, tuple[float, float]] = {}
    for row in response.json().get("data", []):
        if row.get("id") in wanted:
            pricing = row.get("pricing") or {}
            prices[row["id"]] = (float(pricing["prompt"]), float(pricing["completion"]))
    return prices


# --------------------------------------------------------------------------
# The OpenRouter judge backend
# --------------------------------------------------------------------------


class OpenRouterJudgeBackend:
    """One `OpenRouterClient` per judge model, each metering into its own
    in-memory sink so a call's real cost (OpenRouter's `usage.cost`) can be
    read back and recorded in the spend meter."""

    def __init__(self, api_key: str, *, client_factory: Callable[..., Any] | None = None) -> None:
        self._api_key = api_key
        self._clients: dict[str, Any] = {}
        self._factory = client_factory

    def _client(self, model: str) -> Any:
        if model not in self._clients:
            from portolan.llm.openrouter import OpenRouterClient
            from portolan.metering import InMemorySink

            factory = self._factory or OpenRouterClient
            self._clients[model] = factory(
                api_key=self._api_key, default_model=model, sink=InMemorySink(), tenant="bench"
            )
        return self._clients[model]

    def judge(self, *, model: str, system: str, user: str, max_tokens: int) -> JudgeReply:
        client = self._client(model)
        before = len(client.sink.events)
        data: dict[str, Any] | None = None
        error: str | None = None
        try:
            result = client.complete(
                system=system,
                user=user,
                json_schema=JUDGE_SCHEMA,
                model=model,
                surface="audit",
                max_tokens=max_tokens,
            )
            data = result.data if isinstance(result.data, dict) else None
            if data is None:
                error = "no JSON object in the reply"
        except Exception as exc:  # noqa: BLE001 - a failed judge is a vote, not a crash
            error = f"{type(exc).__name__}: {exc}"[:500]
        events = client.sink.events[before:]
        costs = [e.cost_estimate for e in events]
        return JudgeReply(
            data=data,
            cost_usd=sum(c for c in costs if c is not None) if costs else 0.0,
            input_tokens=sum(e.input_tokens for e in events),
            output_tokens=sum(e.output_tokens for e in events),
            error=error,
        )


# --------------------------------------------------------------------------
# Resolution
# --------------------------------------------------------------------------


def _vote(model: str, reply: JudgeReply, text: str, cost: float) -> JudgeVote:
    data = reply.data or {}
    raw = data.get("verdict")
    if reply.error is not None or raw not in ("supported", "partial", "unsupported"):
        return JudgeVote(
            judge_model=model,
            verdict="error",
            effective="error",
            cost_usd=cost,
            error=reply.error or f"bad verdict {raw!r}",
        )
    evidence = str(data.get("evidence") or "").strip() or None
    resolves = evidence is not None and resolve_quote_span(evidence, text) is not None
    effective: Verdict = raw
    if raw == "supported" and not resolves:
        effective = "partial"
    type_fits = data.get("type_fits")
    return JudgeVote(
        judge_model=model,
        verdict=raw,
        effective=effective,
        type_fits=type_fits if isinstance(type_fits, bool) else None,
        evidence=evidence,
        evidence_resolves=resolves,
        rationale=str(data.get("rationale") or "")[:600],
        cost_usd=cost,
    )


@dataclass
class DisputeRun:
    outcomes: list[DisputeOutcome]
    models: tuple[str, ...]
    estimate_usd: float
    ceiling_usd: float
    max_usd: float
    spent_usd: float = 0.0
    calls: int = 0
    stopped: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def decisions(self) -> dict[tuple[str, str], str]:
        return {(o.dispute.source_id, o.dispute.produced_id): o.decision for o in self.outcomes}


def resolve_disputes(
    disputes: list[Dispute],
    texts: dict[str, str],
    panel: JudgePanel,
    *,
    max_usd: float,
    meter: SpendMeter | None = None,
    corpus_key: str = CORPUS_KEY,
    progress: Callable[[str], None] | None = None,
) -> DisputeRun:
    expected, ceiling = estimate_usd(panel, disputes, texts)
    if progress is not None:
        progress(
            f"judge estimate: expected ${expected:.4f}, ceiling ${ceiling:.4f} for "
            f"{len(disputes)} disputes x {len(panel.models)} judges "
            f"({', '.join(panel.models)}); cap ${max_usd:.2f}"
        )
    if expected > max_usd:
        raise BudgetEstimateExceeded(expected, max_usd, len(disputes))
    run = DisputeRun(
        outcomes=[],
        models=panel.models,
        estimate_usd=expected,
        ceiling_usd=ceiling,
        max_usd=max_usd,
    )
    for dispute in disputes:
        outcome = DisputeOutcome(dispute=dispute)
        run.outcomes.append(outcome)
        if run.stopped is not None:
            outcome.note = run.stopped
            continue
        text = texts.get(dispute.source_id, "")
        system, user = build_judge_prompt(dispute, text)
        for model in panel.models:
            call_ceiling = _call_cost(panel, model, system, user, ceiling=True)
            if run.spent_usd + call_ceiling > max_usd:
                run.stopped = f"budget: ${run.spent_usd:.4f} spent of ${max_usd:.2f}"
                break
            reply = panel.backend.judge(
                model=model, system=system, user=user, max_tokens=panel.max_tokens
            )
            prompt_price, completion_price = panel.prices[model]
            cost = (
                reply.cost_usd
                if reply.cost_usd
                else reply.input_tokens * prompt_price + reply.output_tokens * completion_price
            )
            run.calls += 1
            run.spent_usd += cost
            if meter is not None and cost > 0:
                meter.record(model=model, corpus=corpus_key, amount_usd=cost)
            outcome.votes.append(_vote(model, reply, text, cost))
            if sum(1 for v in outcome.votes if v.effective != "supported") >= 2:
                break
        if outcome.supported_votes >= 2:
            outcome.decision = "supported"
        elif sum(1 for v in outcome.votes if v.effective != "supported") >= 2:
            outcome.decision = "not_supported"
        else:
            outcome.decision = "not_judged"
            outcome.note = run.stopped
        if progress is not None:
            progress(
                f"dispute {dispute.dispute_id} [{dispute.category}] {outcome.decision} "
                f"({outcome.tally}) spent=${run.spent_usd:.4f}"
            )
    return run


def summarize(run: DisputeRun) -> dict[str, dict[str, int]]:
    """category -> disputed, judged, overturned (supported), upheld, not_judged."""
    out: dict[str, dict[str, int]] = {}
    for o in run.outcomes:
        row = out.setdefault(
            o.dispute.category,
            {"disputed": 0, "judged": 0, "overturned": 0, "upheld": 0, "not_judged": 0},
        )
        row["disputed"] += 1
        if o.decision == "not_judged":
            row["not_judged"] += 1
            continue
        row["judged"] += 1
        row["overturned" if o.decision == "supported" else "upheld"] += 1
    return dict(sorted(out.items()))


def write_dispute_logs(
    run: DisputeRun, out_dir: Path | str, *, context: dict[str, Any]
) -> tuple[Path, Path]:
    """`corrections.jsonl`: one proposed label correction per `supported`
    decision, never applied. `disputes.jsonl`: every dispute and its votes."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    created = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    corrections: list[str] = []
    everything: list[str] = []
    for o in run.outcomes:
        d = o.dispute
        votes = [v.model_dump() for v in o.votes]
        row = {
            "created": created,
            **context,
            "source_id": d.source_id,
            "produced_id": d.produced_id,
            "category": d.category,
            "fact": {
                "type": d.fact_type,
                "claim": d.claim,
                "quote": d.quote,
                "locator": d.locator,
                "entities": d.entities,
                "span": list(d.span) if d.span else None,
                "decoy_class_id": d.decoy_class_id,
            },
            "votes": votes,
            "tally": o.tally,
            "judge_models": [v.judge_model for v in o.votes],
            "decision": o.decision,
        }
        everything.append(json.dumps({**row, "note": o.note}, sort_keys=True))
        if o.decision == "supported":
            rationale = " | ".join(
                f"{v.judge_model}: {v.rationale}" for v in o.votes if v.effective == "supported"
            )
            corrections.append(
                json.dumps(
                    {**row, "rationale": rationale, "status": "proposed", "applied": False},
                    sort_keys=True,
                )
            )
    corrections_path = out / "corrections.jsonl"
    disputes_path = out / "disputes.jsonl"
    corrections_path.write_text("".join(c + "\n" for c in corrections), encoding="utf-8")
    disputes_path.write_text("".join(e + "\n" for e in everything), encoding="utf-8")
    return corrections_path, disputes_path


# --------------------------------------------------------------------------
# The Halvard exam (report only)
# --------------------------------------------------------------------------


def halvard_disputes(
    score: Any,
    texts: dict[str, str],
    *,
    source_meta: dict[str, tuple[str | None, str | None]] | None = None,
) -> list[Dispute]:
    """One dispute per unplanted fact in a Halvard `ExtractionScore`. The
    category is the bench's own classification (`invented`,
    `unplanted_plausible`, `unverifiable`) or `distractor_leak`."""
    disputes: list[Dispute] = []
    for u in score.unplanted:
        record = next(
            (r for r in score.produced_by_source.get(u.source_id, []) if r.id == u.produced_id),
            None,
        )
        text = texts.get(u.source_id)
        if text is None:
            continue
        quote = record.quote if record else None
        resolved = resolve_quote_span(quote, text) if quote else None
        stype, sdate = (source_meta or {}).get(u.source_id, (None, None))
        disputes.append(
            Dispute(
                source_id=u.source_id,
                produced_id=u.produced_id or (record.id if record else ""),
                category="distractor_leak" if u.is_distractor_leak else u.classification,
                fact_type=u.fact_type,
                claim=u.claim,
                quote=quote,
                locator=u.locator,
                entities=list(record.entities) if record else [],
                span=resolved.span if resolved else None,
                source_type=stype,
                source_date=sdate,
            )
        )
    return disputes
