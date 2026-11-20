"""Cheapest qualifying open-weight PAID teacher discovery (coordinator
course-correction, 2026-09-21, item 1: "Teacher one is a cheap PAID
open-weight model on OpenRouter, not a free one ... pick the cheapest
open-weight model (Qwen, DeepSeek, Llama, Mistral, GLM, Kimi families)
that supports structured JSON output and has a context window over
32k").

Live-checked 2026-09-21 against `GET /api/v1/models` (443 models that
day): the cheapest qualifying candidate was `mistralai/mistral-nemo` --
$0.000000019/token prompt ($0.019/million), $0.00000003/token completion
($0.03/million), 131072-token context, `supported_parameters` includes
both `structured_outputs` and `response_format`. Not hardcoded here --
`discover_cheapest_paid_teacher` re-queries live every run, the same
"never hardcode an id you haven't seen in the response" discipline
`corpus.generator.prose.discover_free_models` already uses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = [
    "PAID_TEACHER_FAMILIES",
    "MIN_CONTEXT_LENGTH",
    "PaidModelCandidate",
    "candidates_from_models_response",
    "select_cheapest_paid_teacher",
    "select_two_cheapest_different_families",
    "discover_cheapest_paid_teacher",
    "discover_two_cheapest_paid_teachers",
]

# The coordinator's literal list for the PAID teacher -- narrower than
# `portolan.training.teachers.OPEN_WEIGHT_FAMILIES` (the free-teacher
# list, which also allows Gemma and gpt-oss): "Qwen, DeepSeek, Llama,
# Mistral, GLM, Kimi families". A deliberately separate constant, not a
# subset filter over the other module's list, so each list's own
# provenance (which instruction named it) stays traceable.
PAID_TEACHER_FAMILIES: tuple[str, ...] = (
    "qwen/",
    "deepseek/",
    "meta-llama/",
    "mistralai/",
    "z-ai/glm",
    "moonshotai/",  # Kimi / Moonshot
)

MIN_CONTEXT_LENGTH = 32_000
_STRUCTURED_OUTPUT_PARAMS = {"structured_outputs", "response_format"}


def _family_of(model_id: str) -> str | None:
    for family in PAID_TEACHER_FAMILIES:
        if family in model_id:
            return family
    return None


@dataclass(frozen=True)
class PaidModelCandidate:
    model_id: str
    family: str
    context_length: int
    prompt_price_per_token: float
    completion_price_per_token: float


def candidates_from_models_response(rows: list[dict[str, Any]]) -> list[PaidModelCandidate]:
    """Pure filter over a `GET /api/v1/models` `data` list -- no network,
    so this is the unit-testable half of discovery. Excludes: `:free`
    ids (this is the PAID teacher); ids outside `PAID_TEACHER_FAMILIES`;
    a context window at or under `MIN_CONTEXT_LENGTH` (32k); any id whose
    `supported_parameters` has neither `structured_outputs` nor
    `response_format` (no reliable JSON-schema output -- the same two
    parameter names `portolan.llm.openrouter._choose_json_strategy`
    already looks for); and any row with unparseable or non-positive
    pricing."""
    candidates: list[PaidModelCandidate] = []
    for row in rows:
        model_id = row.get("id")
        if not isinstance(model_id, str) or model_id.endswith(":free"):
            continue
        family = _family_of(model_id)
        if family is None:
            continue
        context_length = row.get("context_length") or 0
        if not isinstance(context_length, int | float) or context_length <= MIN_CONTEXT_LENGTH:
            continue
        supported = set(row.get("supported_parameters") or [])
        if not (supported & _STRUCTURED_OUTPUT_PARAMS):
            continue
        pricing = row.get("pricing") or {}
        try:
            prompt_price = float(pricing.get("prompt", "nan"))
            completion_price = float(pricing.get("completion", "nan"))
        except (TypeError, ValueError):
            continue
        if prompt_price != prompt_price or prompt_price <= 0:  # NaN or non-positive
            continue
        candidates.append(
            PaidModelCandidate(
                model_id=model_id,
                family=family,
                context_length=int(context_length),
                prompt_price_per_token=prompt_price,
                completion_price_per_token=completion_price,
            )
        )
    return candidates


def select_cheapest_paid_teacher(
    candidates: list[PaidModelCandidate],
) -> PaidModelCandidate | None:
    """Cheapest by prompt (input) price -- the primary cost driver for a
    call whose system prompt dwarfs its completion budget -- ties broken
    by completion price, then model id, for a deterministic pick when
    two candidates are priced identically."""
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda c: (c.prompt_price_per_token, c.completion_price_per_token, c.model_id),
    )


def select_two_cheapest_different_families(
    candidates: list[PaidModelCandidate],
) -> tuple[PaidModelCandidate, PaidModelCandidate] | None:
    """Two cheapest candidates from two DIFFERENT families (coordinator
    course-correction, 2026-09-21, item 3: "two different cheap PAID
    open-weight models" for the fallback two-extractor agreement mode).
    `None` when fewer than two families are represented at all."""
    if not candidates:
        return None
    first = select_cheapest_paid_teacher(candidates)
    if first is None:
        return None
    remaining = [c for c in candidates if c.family != first.family]
    second = select_cheapest_paid_teacher(remaining)
    if second is None:
        return None
    return first, second


def discover_cheapest_paid_teacher(
    api_key: str | None, *, timeout: float = 30.0
) -> PaidModelCandidate | None:
    """Live `GET /api/v1/models`, filtered and picked. Never raises --
    a network failure or unexpected shape returns `None`, same
    never-fatal contract `corpus.generator.prose.discover_free_models`
    already uses."""
    import httpx

    try:
        response = httpx.get(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
            timeout=timeout,
        )
        if response.status_code >= 400:
            return None
        payload = response.json()
    except Exception:  # noqa: BLE001 - best-effort discovery, never fatal
        return None

    rows = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return None
    return select_cheapest_paid_teacher(candidates_from_models_response(rows))


def discover_two_cheapest_paid_teachers(
    api_key: str | None, *, timeout: float = 30.0
) -> tuple[PaidModelCandidate, PaidModelCandidate] | None:
    """`discover_cheapest_paid_teacher`'s two-model sibling for the
    fallback two-extractor agreement mode (`--judge none` after a failed
    judge calibration)."""
    import httpx

    try:
        response = httpx.get(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
            timeout=timeout,
        )
        if response.status_code >= 400:
            return None
        payload = response.json()
    except Exception:  # noqa: BLE001 - best-effort discovery, never fatal
        return None

    rows = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return None
    return select_two_cheapest_different_families(candidates_from_models_response(rows))
