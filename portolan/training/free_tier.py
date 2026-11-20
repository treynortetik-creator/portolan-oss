"""Account-wide OpenRouter free-model (`:free`) daily request budget
(coordinator course-correction, 2026-09-21).

**Confirmed live against this account**, `GET /api/v1/auth/key`
(`AUTH_KEY_URL` below), 2026-09-21:

```
"is_free_tier": false,
"free_model_daily_requests": {"used": 1, "limit": 1000, "remaining": 999}
```

**Confirmed against OpenRouter's own docs**
(openrouter.ai/docs/api-reference/limits, fetched 2026-09-21): free
(`:free`-suffixed) models are capped **account-wide, across every
`:free` model together, not per model**: 20 requests/minute regardless
of account status, and a daily cap of 50 requests if the account has
purchased fewer than 10 lifetime credits, 1000 if it has purchased 10 or
more -- this account is on the 1000/day tier (`is_free_tier: false`,
`free_model_daily_requests.limit: 1000` agree). The docs say the daily
window is "the current UTC day" -- `seconds_until_utc_midnight` below is
that reset point, not a rolling 24 hours.

Only the **daily** count is tracked in the shared file this module
writes (`var/data/free_tier_budget.json`, one file, gitignored like
everything under `var/`, `{"date": "YYYY-MM-DD", "requests_used": N}`)
-- the **minute** cap is already enforced in-process by
`corpus.generator.prose.TokenBucketLimiter`
(`portolan.training.pipeline.run_corpus`'s own `limiter` argument), which
doesn't need cross-process coordination: a minute window is short-lived
enough that two processes racing for it just means both wait out
whichever one filled it first, not a day's budget silently double-spent.
The daily count DOES need to survive a process restart and be shared --
across `portolan.training.pipeline`'s own run and, per the coordinator,
"the synthetic batch runner being built in parallel" -- so every write
here goes through a real OS file lock (`fcntl.flock`, exclusive, held for
the read-modify-write) rather than trusting two processes never to
interleave.
"""

from __future__ import annotations

import fcntl
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

__all__ = [
    "FREE_MINUTE_LIMIT",
    "FREE_DAILY_LIMIT_LOW_CREDIT",
    "FREE_DAILY_LIMIT_HIGH_CREDIT",
    "AUTH_KEY_URL",
    "FreeTierBudget",
    "default_budget_path",
    "today_utc",
    "seconds_until_utc_midnight",
    "fetch_daily_limit",
]

# openrouter.ai/docs/api-reference/limits, fetched 2026-09-21 -- account-
# wide across every :free model, not per model.
FREE_MINUTE_LIMIT = 20
FREE_DAILY_LIMIT_LOW_CREDIT = 50  # < 10 lifetime credits purchased
FREE_DAILY_LIMIT_HIGH_CREDIT = 1000  # >= 10 lifetime credits purchased

AUTH_KEY_URL = "https://openrouter.ai/api/v1/auth/key"


def default_budget_path(var_root: Path) -> Path:
    return var_root / "data" / "free_tier_budget.json"


def today_utc() -> str:
    return datetime.now(UTC).date().isoformat()


def seconds_until_utc_midnight(now: datetime | None = None) -> float:
    """Seconds from `now` (default: real UTC now) until the next UTC
    midnight -- when OpenRouter's daily `:free` budget resets ("the
    current UTC day", per the docs)."""
    current = now or datetime.now(UTC)
    if current.tzinfo is None:
        current = current.replace(tzinfo=UTC)
    next_midnight = datetime.combine(
        current.date() + timedelta(days=1), datetime.min.time(), tzinfo=UTC
    )
    return max(0.0, (next_midnight - current).total_seconds())


def fetch_daily_limit(api_key: str, *, timeout: float = 15.0) -> int | None:
    """Live `GET /api/v1/auth/key`, returns `data.free_model_daily_
    requests.limit` (50 or 1000 for this account, per the module
    docstring) -- best-effort: any network failure or unexpected shape
    returns `None`, letting the caller fall back to a documented default
    rather than crash a long-lived run over one flaky lookup."""
    import httpx

    try:
        response = httpx.get(
            AUTH_KEY_URL, headers={"Authorization": f"Bearer {api_key}"}, timeout=timeout
        )
        if response.status_code >= 400:
            return None
        payload = response.json()
    except Exception:  # noqa: BLE001 - best-effort, never fatal
        return None

    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, dict):
        return None
    daily = data.get("free_model_daily_requests")
    if not isinstance(daily, dict):
        return None
    limit = daily.get("limit")
    return int(limit) if isinstance(limit, int | float) else None


@dataclass
class FreeTierBudget:
    """The shared daily counter. `path` is normally
    `default_budget_path(var_root)`; `daily_limit` defaults to the
    documented high-credit tier (1000) but a caller that confirmed this
    account is on the low-credit tier (`fetch_daily_limit`, or
    `is_free_tier: true`) should pass `FREE_DAILY_LIMIT_LOW_CREDIT`
    instead.

    Every method takes and releases the file lock for exactly the one
    read-modify-write it needs -- no lock is held across a network call
    or between two separate method calls, so one process spending a long
    time between requests never blocks another process's own bookkeeping.
    """

    path: Path
    daily_limit: int = FREE_DAILY_LIMIT_HIGH_CREDIT

    def _read_locked(self, fh: Any) -> dict[str, Any]:
        fh.seek(0)
        raw = fh.read()
        today = today_utc()
        if not raw.strip():
            return {"date": today, "requests_used": 0}
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return {"date": today, "requests_used": 0}
        if not isinstance(data, dict) or data.get("date") != today:
            # a new UTC day (or a corrupt/unrecognized file) -- reset,
            # never carry yesterday's count into today's budget.
            return {"date": today, "requests_used": 0}
        try:
            used = int(data.get("requests_used", 0))
        except (TypeError, ValueError):
            used = 0
        return {"date": today, "requests_used": max(0, used)}

    def _write_locked(self, fh: Any, data: dict[str, Any]) -> None:
        fh.seek(0)
        fh.truncate()
        fh.write(json.dumps(data))
        fh.flush()

    def _with_lock(self, fn: Any) -> Any:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        with self.path.open("r+", encoding="utf-8") as fh:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            try:
                return fn(fh)
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)

    def used_today(self) -> int:
        return int(self._with_lock(lambda fh: self._read_locked(fh))["requests_used"])

    def remaining_today(self) -> int:
        return max(0, self.daily_limit - self.used_today())

    def is_exhausted(self) -> bool:
        return self.remaining_today() <= 0

    def record_requests(self, count: int = 1) -> int:
        """Atomically add `count` real HTTP requests to today's tally
        (resetting first if the stored date has rolled over past UTC
        midnight) and return the new total. Call once per REAL HTTP
        round-trip actually sent to a `:free` model -- a document that
        needed two teacher calls records two, not one."""
        if count <= 0:
            return self.used_today()

        def _op(fh: Any) -> int:
            data = self._read_locked(fh)
            data["requests_used"] += count
            self._write_locked(fh, data)
            return int(data["requests_used"])

        return self._with_lock(_op)
