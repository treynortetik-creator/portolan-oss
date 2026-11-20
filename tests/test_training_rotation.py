"""Tests for `portolan.training.rotation` -- the per-model cooldown and
rotation logic added 2026-09-21 (coordinator course-correction: "A 429 or
5xx from model X must put X on a per-model cooldown ... and move the same
request to the next eligible open-weight model immediately"). No
network: `call_model` is a plain fake in every test, and every clock is a
fake too, so cooldown math is exact and nothing here sleeps for real.
"""

from __future__ import annotations

import pytest

from portolan.llm.base import LLMError
from portolan.training.rotation import (
    BudgetExhausted,
    HardFailure,
    ModelCooldownTracker,
    RotationCursor,
    is_congestion_error,
    retry_after_seconds,
    select_two_teachers,
)

MODEL_A = "qwen/a:free"
MODEL_B = "meta-llama/b:free"
MODEL_C = "deepseek/c:free"


class _FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


# --------------------------------------------------------------------------
# is_congestion_error / retry_after_seconds
# --------------------------------------------------------------------------


def test_is_congestion_error_true_for_429() -> None:
    assert is_congestion_error(LLMError("x", status_code=429)) is True


@pytest.mark.parametrize("code", [500, 502, 503, 504])
def test_is_congestion_error_true_for_5xx(code: int) -> None:
    assert is_congestion_error(LLMError("x", status_code=code)) is True


def test_is_congestion_error_true_for_no_status_llmerror() -> None:
    # a timeout, or any LLMError with no captured HTTP response at all
    assert is_congestion_error(LLMError("timed out")) is True


def test_is_congestion_error_false_for_genuine_client_error() -> None:
    assert is_congestion_error(LLMError("bad auth", status_code=401)) is False
    assert is_congestion_error(LLMError("not found", status_code=404)) is False


def test_is_congestion_error_false_for_non_llmerror() -> None:
    assert is_congestion_error(RuntimeError("local bug")) is False


def test_retry_after_seconds_reads_header() -> None:
    exc = LLMError("x", status_code=429, headers={"retry-after": "12"})
    assert retry_after_seconds(exc) == 12.0


def test_retry_after_seconds_missing_is_none() -> None:
    assert retry_after_seconds(LLMError("x", status_code=429)) is None


def test_retry_after_seconds_non_numeric_is_none() -> None:
    exc = LLMError("x", status_code=429, headers={"retry-after": "soon"})
    assert retry_after_seconds(exc) is None


# --------------------------------------------------------------------------
# ModelCooldownTracker
# --------------------------------------------------------------------------


def test_cooldown_tracker_escalates_and_caps() -> None:
    clock = _FakeClock()
    tracker = ModelCooldownTracker(base_seconds=60, max_seconds=900, now=clock)

    assert tracker.is_cooling(MODEL_A) is False
    assert tracker.mark_failure(MODEL_A) == 60
    assert tracker.is_cooling(MODEL_A) is True

    clock.advance(60)
    assert tracker.is_cooling(MODEL_A) is False
    assert tracker.mark_failure(MODEL_A) == 120

    clock.advance(120)
    assert tracker.mark_failure(MODEL_A) == 240
    clock.advance(240)
    assert tracker.mark_failure(MODEL_A) == 480
    clock.advance(480)
    assert tracker.mark_failure(MODEL_A) == 900  # capped, not 960
    clock.advance(900)
    assert tracker.mark_failure(MODEL_A) == 900  # stays capped


def test_cooldown_tracker_retry_after_overrides_schedule() -> None:
    clock = _FakeClock()
    tracker = ModelCooldownTracker(now=clock)
    assert tracker.mark_failure(MODEL_A, retry_after=5.0) == 5.0
    assert tracker.is_cooling(MODEL_A) is True
    clock.advance(5.0)
    assert tracker.is_cooling(MODEL_A) is False


def test_cooldown_tracker_mark_success_resets_escalation() -> None:
    clock = _FakeClock()
    tracker = ModelCooldownTracker(base_seconds=60, now=clock)
    tracker.mark_failure(MODEL_A)
    clock.advance(60)
    tracker.mark_success(MODEL_A)
    assert tracker.is_cooling(MODEL_A) is False
    assert tracker.mark_failure(MODEL_A) == 60  # back to base, not 120


def test_cooldown_tracker_earliest_available() -> None:
    clock = _FakeClock()
    tracker = ModelCooldownTracker(now=clock)
    tracker.mark_failure(MODEL_A, retry_after=30)
    tracker.mark_failure(MODEL_B, retry_after=10)
    assert tracker.earliest_available([MODEL_A, MODEL_B]) == 10.0


def test_cooldown_tracker_earliest_available_empty_list_is_now() -> None:
    clock = _FakeClock(start=42.0)
    tracker = ModelCooldownTracker(now=clock)
    assert tracker.earliest_available([]) == 42.0


# --------------------------------------------------------------------------
# select_two_teachers
# --------------------------------------------------------------------------


def test_select_two_teachers_success_picks_two_distinct_families() -> None:
    models = [MODEL_A, MODEL_B]
    tracker = ModelCooldownTracker(now=_FakeClock())
    cursor = RotationCursor()

    def call_model(model: str) -> str:
        return f"result-{model}"

    a, b, ta, tb = select_two_teachers(
        models=models, cooldown=tracker, cursor=cursor, call_model=call_model
    )
    assert {ta, tb} == {MODEL_A, MODEL_B}
    assert a == f"result-{ta}"
    assert b == f"result-{tb}"


def test_select_two_teachers_rotates_away_from_429_model_without_retrying_it() -> None:
    models = [MODEL_A, MODEL_B, MODEL_C]
    tracker = ModelCooldownTracker(now=_FakeClock())
    cursor = RotationCursor()
    calls: list[str] = []

    def call_model(model: str) -> str:
        calls.append(model)
        if model == MODEL_A:
            raise LLMError("rate limited", status_code=429)
        return f"ok-{model}"

    _a, _b, ta, tb = select_two_teachers(
        models=models, cooldown=tracker, cursor=cursor, call_model=call_model
    )
    # MODEL_A tried exactly once -- rotation moved on, it was never
    # retried against itself the way the pre-2026-09-21 design would have.
    assert calls.count(MODEL_A) == 1
    assert MODEL_A not in (ta, tb)
    assert tracker.is_cooling(MODEL_A) is True


def test_select_two_teachers_honors_retry_after_header() -> None:
    # Only two models total, so slot 1 has exactly one structurally valid
    # candidate (MODEL_A, the other family) once slot 0 has used MODEL_B
    # -- MODEL_A must recover (fail once, then answer) for this to ever
    # resolve, exactly like a real model that was transiently congested,
    # not permanently gone.
    models = [MODEL_A, MODEL_B]
    clock = _FakeClock()
    tracker = ModelCooldownTracker(base_seconds=60, now=clock)
    cursor = RotationCursor()
    model_a_calls = 0

    def call_model(model: str) -> str:
        nonlocal model_a_calls
        if model == MODEL_A:
            model_a_calls += 1
            if model_a_calls == 1:
                raise LLMError("rate limited", status_code=429, headers={"retry-after": "5"})
        return "ok"

    def fake_sleep(seconds: float) -> None:
        # A real `time.sleep` would leave this test's fake clock frozen,
        # so the cooldown it just set would never lapse -- advance the
        # SAME clock `cooldown` reads instead.
        clock.advance(seconds)

    _a, _b, ta, tb = select_two_teachers(
        models=models, cooldown=tracker, cursor=cursor, call_model=call_model, sleep=fake_sleep
    )
    assert model_a_calls == 2  # one 429, one recovered answer
    assert {ta, tb} == {MODEL_A, MODEL_B}
    # mark_success cleared the cooldown once MODEL_A recovered -- no
    # lingering entry, not even the header's 5s.
    assert tracker.is_cooling(MODEL_A) is False


def test_select_two_teachers_congestion_sleeps_then_retries() -> None:
    models = [MODEL_A, MODEL_B]
    clock = _FakeClock()
    tracker = ModelCooldownTracker(base_seconds=60, now=clock)
    cursor = RotationCursor()
    tracker.mark_failure(MODEL_A)  # both pre-cooled
    tracker.mark_failure(MODEL_B)

    sleeps: list[float] = []

    def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock.advance(seconds)

    events = []

    def on_congestion(event) -> None:  # noqa: ANN001
        events.append(event)

    def call_model(model: str) -> str:
        return f"ok-{model}"

    _a, _b, ta, tb = select_two_teachers(
        models=models,
        cooldown=tracker,
        cursor=cursor,
        call_model=call_model,
        sleep=fake_sleep,
        on_congestion=on_congestion,
    )
    assert len(events) >= 1
    assert sleeps
    assert {ta, tb} == {MODEL_A, MODEL_B}


def test_select_two_teachers_hard_failure_on_non_congestion_error() -> None:
    models = [MODEL_A, MODEL_B]
    tracker = ModelCooldownTracker(now=_FakeClock())
    cursor = RotationCursor()

    def call_model(model: str) -> str:
        raise RuntimeError("local bug")

    with pytest.raises(HardFailure, match="local bug"):
        select_two_teachers(models=models, cooldown=tracker, cursor=cursor, call_model=call_model)


def test_select_two_teachers_hard_failure_when_only_one_family_discovered() -> None:
    models = [MODEL_A, "qwen/another:free"]  # same family as MODEL_A
    tracker = ModelCooldownTracker(now=_FakeClock())
    cursor = RotationCursor()

    def call_model(model: str) -> str:
        return "ok"

    with pytest.raises(HardFailure):
        select_two_teachers(models=models, cooldown=tracker, cursor=cursor, call_model=call_model)


def test_select_two_teachers_budget_exhausted_propagates_uncaught() -> None:
    models = [MODEL_A, MODEL_B]
    tracker = ModelCooldownTracker(now=_FakeClock())
    cursor = RotationCursor()

    def call_model(model: str) -> str:
        raise BudgetExhausted("daily budget spent")

    with pytest.raises(BudgetExhausted):
        select_two_teachers(models=models, cooldown=tracker, cursor=cursor, call_model=call_model)


def test_select_two_teachers_on_model_failure_callback_fires_per_congestion_hit() -> None:
    models = [MODEL_A, MODEL_B, MODEL_C]
    tracker = ModelCooldownTracker(now=_FakeClock())
    cursor = RotationCursor()
    failed_models: list[str] = []

    def call_model(model: str) -> str:
        if model == MODEL_A:
            raise LLMError("x", status_code=503)
        return "ok"

    select_two_teachers(
        models=models,
        cooldown=tracker,
        cursor=cursor,
        call_model=call_model,
        on_model_failure=lambda model, exc: failed_models.append(model),
    )
    assert failed_models == [MODEL_A]


def test_select_two_teachers_recovers_a_model_after_its_cooldown_expires() -> None:
    # A model that failed once and whose cooldown has since lapsed
    # becomes eligible again -- no permanent per-document blacklist.
    models = [MODEL_A, MODEL_B]
    clock = _FakeClock()
    tracker = ModelCooldownTracker(base_seconds=1, now=clock)
    cursor = RotationCursor()
    tracker.mark_failure(MODEL_A)  # cooling for 1s
    clock.advance(1.0)  # recovered

    def call_model(model: str) -> str:
        return f"ok-{model}"

    _a, _b, ta, tb = select_two_teachers(
        models=models, cooldown=tracker, cursor=cursor, call_model=call_model
    )
    assert {ta, tb} == {MODEL_A, MODEL_B}
