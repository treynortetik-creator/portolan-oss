"""Typed-decision judge for silver-label fact verification (coordinator
course-correction, 2026-09-21, replacing teacher two as a second
free-form extractor): for every fact the paid teacher
(`portolan.training.paid_teacher`) emits and that already passes
`quote_is_supported`, ask three typed questions of a Jev-compatible
endpoint (TypeSafe AI's `POST /v1/systemone`,
`var/jev-typesafe-brief-2026-09-16.md` secs 3-4; prior code
`scripts/jev_eval/judge_and_lint.py`, `scripts/jev_eval/classify.py`)
instead of running a second free-form extraction pass over the same
document:

  (a) noul  -- is `quote` a faithful, sufficient basis for `claim`?
  (b) choice -- which production `fact_type`
      (`portolan.extract.schema.FactType`) fits this claim best, judged
      independently of whatever type the extractor emitted
  (c) noul  -- does `claim` state anything `quote` does not?

A fact is kept when (a) is yes, (b) matches the extractor's own emitted
type, and (c) is no -- `FactVerdict.keep()` is that pure decision;
`JevClient` is the only network-touching piece here, thin enough that
every gate-logic test builds a fake `{"answers": {...}}` response
directly rather than mocking HTTP.

`Kev` (an Apache-2.0 local model, `docs/kev-local.md` once the wave
standing it up documents its URL) implements the same `/v1/systemone`
contract -- `JevClient(base_url=...)` is the one knob that points this
module at either endpoint; nothing else changes.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from portolan.extract.quote_gate import quote_is_supported
from portolan.extract.schema import FACT_TYPES, FactType

__all__ = [
    "JEV_URL_DEFAULT",
    "JEV_PRICE_PER_MILLION_INPUT_USD",
    "JevError",
    "fact_type_criteria",
    "build_fact_questions",
    "JevClient",
    "FactVerdict",
    "parse_verdict",
    "ask_typed_questions",
    "evaluate_fact",
]

JEV_URL_DEFAULT = "https://api.typesafe.ai/v1/systemone"
# var/jev-typesafe-brief-2026-09-16.md sec 4: "$0.042 per million input
# tokens. Output tokens are free (vendor: 'too cheap to meter')."
JEV_PRICE_PER_MILLION_INPUT_USD = 0.042


class JevError(RuntimeError):
    """A Jev/Kev call failed after retries, or returned an unusable
    response shape."""


def fact_type_criteria() -> dict[str, str]:
    """The production fact-type list as a Jev `choice` question's
    `criteria`: `{type_value: definition}` -- the exact definition text
    `portolan.extract.prompt` renders into the extractor's own prompt,
    so the judge is choosing from the identical type vocabulary the
    extractor was told to use (same approach as
    `scripts/jev_eval/classify.py::build_fact_type_criteria`)."""
    return {ft.value: FACT_TYPES[ft].definition for ft in FactType}


def build_fact_questions(criteria: dict[str, str] | None = None) -> dict[str, dict[str, Any]]:
    """The three typed questions, batched into one Jev request per fact
    (the brief's own rule: "Batch every question about the same input
    into one request"). `state` (the claim/quote pair) is built
    separately by `_state_for_fact` -- these questions never repeat it,
    per the brief's "the question id never reaches the model, put
    everything in instructions/criteria" rule."""
    return {
        "faithful": {
            "type": "noul",
            "instructions": (
                "The claim was written from the quoted source text below it. Is the quote a "
                "faithful, sufficient basis for the claim -- does the quote actually support "
                "what the claim says? Answer no if the claim goes beyond the quote, or "
                "misreads it."
            ),
        },
        "fact_type": {
            "type": "choice",
            "instructions": (
                "Which of these signal types best fits the claim, given its quote? Judge the "
                "claim and quote on their own merits, not by guessing what type someone else "
                "may have already assigned."
            ),
            "criteria": criteria or fact_type_criteria(),
        },
        "added_detail": {
            "type": "noul",
            "instructions": (
                "Does the claim state anything -- a name, a number, a date, a cause, an "
                "outcome -- that the quote does not itself say? Answer yes if the claim adds "
                "any detail beyond what the quote says, no if the claim only restates the "
                "quote."
            ),
        },
    }


def _state_for_fact(claim: str, quote: str) -> str:
    return f'Claim: "{claim}"\nQuote: "{quote}"'


@dataclass
class JevClient:
    """Thin HTTP client for one Jev-compatible endpoint (`POST
    .../v1/systemone`, `{"state", "model", "questions"}` in,
    `{"answers": {...}, "usage": {"input_tokens": ..., "output_tokens":
    ...}}` out -- confirmed against `scripts/jev_eval/judge_and_lint.py`'s
    real usage). `base_url` defaults to TypeSafe's own hosted Jev; pass
    Kev's local URL to talk to that instead. `api_key` may be `None` for
    a local Kev with no auth requirement."""

    api_key: str | None
    base_url: str = JEV_URL_DEFAULT
    model: str = "jev-latest"
    timeout: float = 60.0
    max_retries: int = 3
    sleep: Callable[[float], None] = time.sleep
    transport: Any = None  # httpx.BaseTransport, for tests

    def ask(self, state: str, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        import httpx

        client = httpx.Client(transport=self.transport, timeout=self.timeout)
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        body = {"state": state, "model": self.model, "questions": questions}
        try:
            last_exc: Exception | None = None
            for attempt in range(self.max_retries):
                try:
                    response = client.post(self.base_url, json=body, headers=headers)
                except httpx.HTTPError as exc:
                    last_exc = exc
                    if attempt < self.max_retries - 1:
                        self.sleep(2 * (attempt + 1))
                        continue
                    raise JevError(f"Jev call failed after retries: {exc}") from exc
                if response.status_code == 200:
                    return response.json()
                if response.status_code in (429, 529) and attempt < self.max_retries - 1:
                    self.sleep(2 * (attempt + 1))
                    continue
                raise JevError(f"Jev HTTP {response.status_code}: {response.text[:500]}")
            raise JevError(f"Jev call failed after retries: {last_exc}")
        finally:
            client.close()


@dataclass
class FactVerdict:
    """One judge's answer to the three typed questions for one fact."""

    faithful: bool
    faithful_probability: float
    predicted_type: str
    type_confidence: float
    added_detail: bool
    added_detail_probability: float
    input_tokens: int = 0
    output_tokens: int = 0

    def keep(self, emitted_type: str) -> bool:
        """The gate: faithful, type-matching, no added detail."""
        return self.faithful and not self.added_detail and self.predicted_type == emitted_type

    def cost_usd(self) -> float:
        """`var/jev-typesafe-brief-2026-09-16.md` sec 4: input tokens
        billed, output tokens free."""
        return self.input_tokens * JEV_PRICE_PER_MILLION_INPUT_USD / 1_000_000

    def to_record_dict(self) -> dict[str, Any]:
        """Compact shape for `TrainingFact.judge[<name>]` -- the verdict
        fields a training record wants, not the raw API payload."""
        return {
            "faithful": self.faithful,
            "faithful_probability": round(self.faithful_probability, 4),
            "predicted_type": self.predicted_type,
            "type_confidence": round(self.type_confidence, 4),
            "added_detail": self.added_detail,
            "added_detail_probability": round(self.added_detail_probability, 4),
        }


def parse_verdict(response: dict[str, Any]) -> FactVerdict:
    answers = response["answers"]
    faithful_p = float(answers["faithful"]["noul"])
    added_p = float(answers["added_detail"]["noul"])
    usage = response.get("usage") or {}
    return FactVerdict(
        faithful=faithful_p >= 0.5,
        faithful_probability=faithful_p,
        predicted_type=answers["fact_type"]["choice"],
        type_confidence=float(answers["fact_type"].get("confidence", 0.0)),
        added_detail=added_p >= 0.5,
        added_detail_probability=added_p,
        input_tokens=int(usage.get("input_tokens", 0) or 0),
        output_tokens=int(usage.get("output_tokens", 0) or 0),
    )


def ask_typed_questions(
    client: JevClient, claim: str, quote: str, *, criteria: dict[str, str] | None = None
) -> FactVerdict:
    response = client.ask(_state_for_fact(claim, quote), build_fact_questions(criteria))
    return parse_verdict(response)


@dataclass
class FactEvaluation:
    keep: bool
    verdict: FactVerdict | None
    quote_gate_passed: bool


def evaluate_fact(
    *,
    claim: str,
    quote: str | None,
    emitted_type: str,
    doc_text: str,
    client: JevClient,
    criteria: dict[str, str] | None = None,
) -> FactEvaluation:
    """The full gate for one fact: `quote_is_supported` first (the
    mechanical, free check every quote-bearing fact already has to
    clear), then the three typed questions -- only spent on a fact that
    already cleared the free gate. Returns a fact that never reached Jev
    (`verdict=None`) when the quote gate itself failed."""
    if not quote_is_supported(quote, doc_text):
        return FactEvaluation(keep=False, verdict=None, quote_gate_passed=False)
    verdict = ask_typed_questions(client, claim, quote or "", criteria=criteria)
    return FactEvaluation(keep=verdict.keep(emitted_type), verdict=verdict, quote_gate_passed=True)
