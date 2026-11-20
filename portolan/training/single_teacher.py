"""Single-teacher selection for `--mode single` (coordinator course-
correction, 2026-09-21, final design item 1): one extractor, no cross-
extractor agreement, no judge gate -- Mistral Nemo and Qwen3.7-Flash are
OUT (the two-paid-teacher smoke test found them disagreeing on nearly
every fact; see this task's own report).

Preference order, not a live price sort across every open-weight model
the way `portolan.training.paid_teacher` does for the two-extractor
design: DeepSeek V3.x, Qwen3-235B-A22B (or the current 235B id), Kimi K2
(any id) -- three families picked for their own reputation for reliable
structured-JSON extraction. The first family (in that order) with a
qualifying id priced under `MAX_COST_PER_DOCUMENT_USD` at an estimated
4k-in/0.5k-out document wins; within a family, the cheapest qualifying
id."""

from __future__ import annotations

from typing import Any

from portolan.training.paid_teacher import (
    _STRUCTURED_OUTPUT_PARAMS,
    MIN_CONTEXT_LENGTH,
    PaidModelCandidate,
)

__all__ = [
    "SINGLE_TEACHER_PREFERENCE",
    "ESTIMATED_INPUT_TOKENS",
    "ESTIMATED_OUTPUT_TOKENS",
    "MAX_COST_PER_DOCUMENT_USD",
    "estimate_document_cost",
    "candidates_matching_pattern",
    "select_single_teacher",
    "discover_single_teacher",
]

# Live-checked 2026-09-21 substrings, matched against `GET /api/v1/models`
# ids actually seen -- not hardcoded exact ids, so a version bump
# ("deepseek-v3.3") still matches: "any V3.x id" / "the current Qwen3
# 235B id" / "any K2 id", per the coordinator's own wording.
SINGLE_TEACHER_PREFERENCE: tuple[str, ...] = (
    "deepseek/deepseek-v3",
    "qwen/qwen3-235b-a22b",
    "moonshotai/kimi-k2",
)

ESTIMATED_INPUT_TOKENS = 4000
ESTIMATED_OUTPUT_TOKENS = 500
MAX_COST_PER_DOCUMENT_USD = 0.0015


def estimate_document_cost(
    prompt_price_per_token: float, completion_price_per_token: float
) -> float:
    return (
        ESTIMATED_INPUT_TOKENS * prompt_price_per_token
        + ESTIMATED_OUTPUT_TOKENS * completion_price_per_token
    )


def candidates_matching_pattern(
    rows: list[dict[str, Any]], pattern: str
) -> list[PaidModelCandidate]:
    """Like `portolan.training.paid_teacher.candidates_from_models_response`
    but matched against one literal `pattern` substring (one preference
    entry) instead of the fixed six-family list -- reuses that module's
    own context-length (>32k) and structured-output-support bar, so a
    candidate here has already cleared the same production-usability
    check as the two-extractor design's candidates."""
    candidates: list[PaidModelCandidate] = []
    for row in rows:
        model_id = row.get("id")
        if not isinstance(model_id, str) or model_id.endswith(":free"):
            continue
        if pattern not in model_id:
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
                family=pattern,
                context_length=int(context_length),
                prompt_price_per_token=prompt_price,
                completion_price_per_token=completion_price,
            )
        )
    return candidates


def select_single_teacher(rows: list[dict[str, Any]]) -> PaidModelCandidate | None:
    """First preference family (in `SINGLE_TEACHER_PREFERENCE` order)
    with a qualifying id under `MAX_COST_PER_DOCUMENT_USD` wins; within a
    family, the cheapest qualifying id. `None` when no family has a
    candidate under budget."""
    for pattern in SINGLE_TEACHER_PREFERENCE:
        candidates = candidates_matching_pattern(rows, pattern)
        if not candidates:
            continue
        cheapest = min(
            candidates,
            key=lambda c: (c.prompt_price_per_token, c.completion_price_per_token, c.model_id),
        )
        cost = estimate_document_cost(
            cheapest.prompt_price_per_token, cheapest.completion_price_per_token
        )
        if cost < MAX_COST_PER_DOCUMENT_USD:
            return cheapest
    return None


def discover_single_teacher(
    api_key: str | None, *, timeout: float = 30.0
) -> PaidModelCandidate | None:
    """Live `GET /api/v1/models`, filtered and picked. Never raises -- a
    network failure or unexpected shape returns `None`."""
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
    return select_single_teacher(rows)
