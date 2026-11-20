"""Per-model cooldown and rotation for silver-labeling teacher calls
(coordinator course-correction, 2026-09-21, replacing the original
retry-same-model-once-then-fail design): a 429/5xx from model X must not
be retried against X -- `portolan.llm.openrouter.OpenRouterClient`'s own
internal backoff already tried that (confirmed live 2026-09-21: ~70s
burned per exhausted attempt at the client's default 5-retry budget,
against a model that wasn't going to answer differently on retry 3, 4, or
5). Instead X goes on a per-model cooldown and the SAME logical request
moves to a different eligible open-weight model immediately. A document
is only recorded as a `failures.jsonl` congestion event when every
eligible model is cooling down at once -- and even then the run sleeps
until the earliest cooldown expires and retries, rather than abandoning
the document.

Kept separate from `portolan.training.pipeline` (which owns the real
network call, the rate limiter, and the free-tier budget) so the pure
state-machine logic here -- which model to try next, when a model counts
as "eligible", how fast a repeat offender's cooldown grows -- is
unit-tested with a fake `call_model` and a fake clock, no network, no
real sleep.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from portolan.training.teachers import family_of

__all__ = [
    "ModelCooldownTracker",
    "RotationCursor",
    "CongestionEvent",
    "HardFailure",
    "BudgetExhausted",
    "is_congestion_error",
    "retry_after_seconds",
    "select_two_teachers",
]


class BudgetExhausted(Exception):
    """Raised by the caller's `call_model` (`portolan.training.pipeline`)
    to signal the shared daily OpenRouter free-tier budget
    (`portolan.training.free_tier.FreeTierBudget`) is spent -- propagates
    straight out of `select_two_teachers` unmodified, so `run_corpus` can
    stop the run cleanly at exactly the point it happened, before ever
    spending another request."""


@dataclass
class HardFailure(Exception):
    """A `call_model` exception that is not a 429/5xx/timeout-shaped
    congestion signal (`is_congestion_error` said no), or a structural
    shortage of open-weight model diversity (fewer than two families
    discovered) -- not something a different model would plausibly answer
    differently either way. `select_two_teachers` raises this once,
    rather than rotating, so the caller's original "retry once on
    transient failure, then record the failure" path (silver-labels build
    instructions item 4) still applies to it."""

    message: str
    model: str
    teachers_tried: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return self.message


@dataclass
class CongestionEvent:
    """Every model still eligible for the slot being filled (right
    family, not already used for the other slot) is currently cooling
    down. `select_two_teachers` calls `on_congestion` with one of these
    immediately before sleeping until the earliest cooldown expires, so
    the caller can log exactly one `failures.jsonl` diagnostic line per
    pause -- not a silent stall, and not one line per busy-loop
    iteration."""

    candidates: list[str]
    wait_seconds: float


_RATE_LIMIT_OR_SERVER_STATUSES = {429, 500, 502, 503, 504}


def is_congestion_error(exc: Exception) -> bool:
    """True for a 429/5xx `portolan.llm.base.LLMError` (its `status_code`
    attribute, added 2026-09-21 for exactly this) or any other `LLMError`
    with no status at all (a timeout -- also "this model didn't answer",
    also worth trying a different one for instead of waiting it out).
    False for a non-`LLMError` exception (a local bug -- rotating models
    won't fix it) or an `LLMError` with a genuine non-congestion status
    (401/403/404 -- likely account- or request-shape-wide, not
    model-specific)."""
    from portolan.llm.base import LLMError

    if not isinstance(exc, LLMError):
        return False
    status = exc.status_code
    return status is None or status in _RATE_LIMIT_OR_SERVER_STATUSES


def retry_after_seconds(exc: Exception) -> float | None:
    """`Retry-After` (seconds) from an `LLMError`'s captured response
    headers (`exc.headers`), when present and numeric -- OpenRouter's own
    hint for how long to back this model off, honored over this module's
    own default cooldown schedule."""
    headers = getattr(exc, "headers", None) or {}
    for key in ("retry-after", "Retry-After"):
        value = headers.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return None


@dataclass
class ModelCooldownTracker:
    """Per-model exponential-backoff cooldown state: a model's first
    429/5xx puts it on a `base_seconds` (default 60s) cooldown; each
    consecutive failure (with no intervening success) doubles the next
    cooldown, capped at `max_seconds` (default 900s / 15 minutes) --
    exactly the schedule the coordinator specified. `now` is injected
    (default `time.monotonic`) so tests never sleep for real."""

    base_seconds: float = 60.0
    max_seconds: float = 900.0
    now: Callable[[], float] = time.monotonic
    _cooldown_until: dict[str, float] = field(default_factory=dict, init=False)
    _next_backoff: dict[str, float] = field(default_factory=dict, init=False)

    def is_cooling(self, model: str) -> bool:
        return self.now() < self._cooldown_until.get(model, 0.0)

    def available_at(self, model: str) -> float:
        return self._cooldown_until.get(model, 0.0)

    def mark_failure(self, model: str, *, retry_after: float | None = None) -> float:
        """Put `model` on cooldown; returns the cooldown length applied.
        `retry_after` (seconds, from `retry_after_seconds`), when given
        and positive, overrides this tracker's own schedule for THIS
        cooldown's length -- OpenRouter told us directly how long to
        wait, so its answer wins over a guess. The escalating internal
        backoff still advances underneath either way, so a model that
        keeps failing after its `retry_after` cooldown lapses gets a
        longer default next time."""
        if retry_after is not None and retry_after > 0:
            cooldown = retry_after
        else:
            cooldown = self._next_backoff.get(model, self.base_seconds)
        self._cooldown_until[model] = self.now() + cooldown
        self._next_backoff[model] = min(
            self._next_backoff.get(model, self.base_seconds) * 2, self.max_seconds
        )
        return cooldown

    def mark_success(self, model: str) -> None:
        """A model that just answered resets its own escalation -- a
        congestion window it recovered from shouldn't leave it
        permanently the slowest option in the rotation."""
        self._next_backoff[model] = self.base_seconds
        self._cooldown_until.pop(model, None)

    def earliest_available(self, models: Sequence[str]) -> float:
        if not models:
            return self.now()
        return min(self.available_at(m) for m in models)


@dataclass
class RotationCursor:
    """One shared round-robin position across every `select_two_teachers`
    call, so successive documents spread across the whole discovered
    model list (same spirit as `portolan.training.teachers.
    TeacherPairPicker`, but cooldown-aware -- a plain fixed-offset index
    into `models` isn't enough once some entries are temporarily
    ineligible)."""

    index: int = 0


def _pick_next(eligible: list[str], models: list[str], cursor: RotationCursor) -> str:
    """The first `eligible` model at or after `cursor`'s position in
    `models`' own order (wrapping) -- keeps successive picks spreading
    forward through the list instead of always returning `eligible[0]`."""
    n = len(models)
    for offset in range(n):
        candidate = models[(cursor.index + offset) % n]
        if candidate in eligible:
            cursor.index = (cursor.index + offset + 1) % n
            return candidate
    return eligible[0]  # unreachable given eligible is always a subset of models


def select_two_teachers[T](
    *,
    models: list[str],
    cooldown: ModelCooldownTracker,
    cursor: RotationCursor,
    call_model: Callable[[str], T],
    sleep: Callable[[float], None] = time.sleep,
    on_congestion: Callable[[CongestionEvent], None] | None = None,
    on_model_failure: Callable[[str, Exception], None] | None = None,
) -> tuple[T, T, str, str]:
    """Get results from two open-weight models of different families for
    one logical request (one envelope): `(result_a, result_b, model_a,
    model_b)`.

    On a congestion-classified failure (`is_congestion_error`), the
    failing model is put on cooldown and the search simply continues --
    cooldown state is the only "don't try me right now" signal, so a
    model that failed minutes ago and has since recovered becomes
    eligible again automatically, with no separate "already tried" memory
    to reset. When every remaining candidate for the current slot is
    cooling down, `on_congestion` fires once and the call sleeps until
    the earliest of their cooldowns expires, then keeps searching --
    never gives up on the document over congestion alone.

    Raises `HardFailure` for a non-congestion error, or when `models`
    structurally lacks two families the first slot's pick didn't already
    use (fewer than two open-weight families were ever discovered).
    `BudgetExhausted` (raised by `call_model` itself) propagates
    unmodified, uncaught.
    """
    results: list[T] = []
    used: list[str] = []
    used_family: str | None = None

    for _slot in range(2):
        while True:
            candidates = [m for m in models if m not in used and family_of(m) != used_family]
            if not candidates:
                raise HardFailure(
                    "no open-weight model of a different family available "
                    f"(discovered models: {models!r})",
                    model=used[-1] if used else "",
                    teachers_tried=list(used),
                )

            eligible = [m for m in candidates if not cooldown.is_cooling(m)]
            if not eligible:
                wait_until = cooldown.earliest_available(candidates)
                wait_seconds = max(0.0, wait_until - cooldown.now())
                if on_congestion is not None:
                    on_congestion(
                        CongestionEvent(candidates=list(candidates), wait_seconds=wait_seconds)
                    )
                sleep(wait_seconds + 0.05)
                continue

            model = _pick_next(eligible, models, cursor)
            try:
                result = call_model(model)
            except BudgetExhausted:
                raise
            except Exception as exc:  # noqa: BLE001 - classified via is_congestion_error
                if is_congestion_error(exc):
                    cooldown.mark_failure(model, retry_after=retry_after_seconds(exc))
                    if on_model_failure is not None:
                        on_model_failure(model, exc)
                    continue
                raise HardFailure(
                    f"{type(exc).__name__}: {exc}", model=model, teachers_tried=[*used, model]
                ) from exc
            else:
                cooldown.mark_success(model)
                results.append(result)
                used.append(model)
                if used_family is None:
                    used_family = family_of(model)
                break

    return results[0], results[1], used[0], used[1]
