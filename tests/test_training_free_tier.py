"""Tests for `portolan.training.free_tier` -- the account-wide OpenRouter
`:free` daily budget tracker (coordinator course-correction, 2026-09-21).
No network: `fetch_daily_limit` tests use `httpx.MockTransport` isn't
needed since the function takes no injectable transport -- covered via
monkeypatching `httpx.get` isn't done either; instead these tests exercise
`FreeTierBudget`'s file-backed counting and `seconds_until_utc_midnight`'s
pure date math, the two pieces the pipeline actually depends on for
correctness. A live check of `fetch_daily_limit` against the real account
was done by hand (see `var/data/MANIFEST.md` "Silver labels" / this
task's own report), not re-asserted here as a unit test (it would be a
network test).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from portolan.training.free_tier import (
    FREE_DAILY_LIMIT_HIGH_CREDIT,
    FREE_DAILY_LIMIT_LOW_CREDIT,
    FreeTierBudget,
    default_budget_path,
    seconds_until_utc_midnight,
    today_utc,
)


def test_default_budget_path(tmp_path: Path) -> None:
    assert default_budget_path(tmp_path) == tmp_path / "data" / "free_tier_budget.json"


def test_documented_limits() -> None:
    # openrouter.ai/docs/api-reference/limits, fetched 2026-09-21.
    assert FREE_DAILY_LIMIT_LOW_CREDIT == 50
    assert FREE_DAILY_LIMIT_HIGH_CREDIT == 1000


def test_today_utc_matches_real_utc_date() -> None:
    assert today_utc() == datetime.now(UTC).date().isoformat()


def test_seconds_until_utc_midnight_at_exactly_midnight_is_a_full_day() -> None:
    midnight = datetime(2026, 9, 21, 0, 0, 0, tzinfo=UTC)
    assert seconds_until_utc_midnight(midnight) == 86400.0


def test_seconds_until_utc_midnight_one_second_before() -> None:
    almost = datetime(2026, 9, 21, 23, 59, 59, tzinfo=UTC)
    assert seconds_until_utc_midnight(almost) == 1.0


def test_seconds_until_utc_midnight_naive_datetime_treated_as_utc() -> None:
    naive = datetime(2026, 9, 21, 12, 0, 0)  # noqa: DTZ001
    aware = datetime(2026, 9, 21, 12, 0, 0, tzinfo=UTC)
    assert seconds_until_utc_midnight(naive) == seconds_until_utc_midnight(aware)


# --------------------------------------------------------------------------
# FreeTierBudget
# --------------------------------------------------------------------------


def test_budget_starts_at_zero(tmp_path: Path) -> None:
    budget = FreeTierBudget(path=default_budget_path(tmp_path), daily_limit=1000)
    assert budget.used_today() == 0
    assert budget.remaining_today() == 1000
    assert budget.is_exhausted() is False


def test_budget_record_requests_accumulates(tmp_path: Path) -> None:
    budget = FreeTierBudget(path=default_budget_path(tmp_path), daily_limit=1000)
    assert budget.record_requests(3) == 3
    assert budget.record_requests(2) == 5
    assert budget.used_today() == 5
    assert budget.remaining_today() == 995


def test_budget_record_requests_zero_or_negative_is_a_no_op(tmp_path: Path) -> None:
    budget = FreeTierBudget(path=default_budget_path(tmp_path), daily_limit=1000)
    budget.record_requests(3)
    assert budget.record_requests(0) == 3
    assert budget.used_today() == 3


def test_budget_is_exhausted_at_the_limit(tmp_path: Path) -> None:
    budget = FreeTierBudget(path=default_budget_path(tmp_path), daily_limit=3)
    budget.record_requests(3)
    assert budget.is_exhausted() is True
    assert budget.remaining_today() == 0


def test_budget_persists_across_instances_same_file(tmp_path: Path) -> None:
    path = default_budget_path(tmp_path)
    FreeTierBudget(path=path, daily_limit=1000).record_requests(7)
    reopened = FreeTierBudget(path=path, daily_limit=1000)
    assert reopened.used_today() == 7


def test_budget_shared_across_two_instances_same_file_accumulates(tmp_path: Path) -> None:
    # "so the synthetic batch runner being built in parallel can share it
    # later" -- two independent FreeTierBudget objects over the SAME file
    # must see each other's writes, the way two separate processes would.
    path = default_budget_path(tmp_path)
    silver_label_budget = FreeTierBudget(path=path, daily_limit=1000)
    synth_runner_budget = FreeTierBudget(path=path, daily_limit=1000)

    silver_label_budget.record_requests(4)
    synth_runner_budget.record_requests(6)

    assert silver_label_budget.used_today() == 10
    assert synth_runner_budget.used_today() == 10


def test_budget_resets_on_a_new_utc_day(tmp_path: Path) -> None:
    path = default_budget_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    import json

    path.write_text(json.dumps({"date": "2020-01-01", "requests_used": 999}), encoding="utf-8")
    budget = FreeTierBudget(path=path, daily_limit=1000)
    # stale date -- never carries yesterday's (or, here, six-years-ago's)
    # count into today's budget.
    assert budget.used_today() == 0


def test_budget_corrupt_file_treated_as_empty(tmp_path: Path) -> None:
    path = default_budget_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not json{{{", encoding="utf-8")
    budget = FreeTierBudget(path=path, daily_limit=1000)
    assert budget.used_today() == 0
    assert budget.record_requests(1) == 1
