"""Tests for `portolan.util.clock` (e2e-run follow-up wave, item 2: "one
clock"). No `db` fixture needed -- this module touches neither Postgres
nor the network.
"""

from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

from portolan.util import clock


def test_today_defaults_to_america_phoenix(monkeypatch) -> None:
    monkeypatch.delenv("PORTOLAN_TIMEZONE", raising=False)
    assert clock.DEFAULT_TIMEZONE == "America/Phoenix"
    assert clock.now().tzinfo.key == "America/Phoenix"  # type: ignore[union-attr]


def test_today_reads_portolan_timezone_env_var(monkeypatch) -> None:
    monkeypatch.setenv("PORTOLAN_TIMEZONE", "UTC")
    assert clock.now().tzinfo == ZoneInfo("UTC")

    monkeypatch.setenv("PORTOLAN_TIMEZONE", "America/Phoenix")
    assert clock.now().tzinfo == ZoneInfo("America/Phoenix")


def test_now_and_today_agree_with_each_other(monkeypatch) -> None:
    monkeypatch.setenv("PORTOLAN_TIMEZONE", "America/Phoenix")
    assert clock.today() == clock.now().date()


def test_phoenix_and_utc_calendar_dates_can_differ_near_midnight() -> None:
    """The whole point of a deployment-timezone clock (module docstring):
    America/Phoenix (UTC-7, no daylight saving) is behind UTC, so a fixed
    instant just after UTC's own midnight is still the PREVIOUS calendar
    day in Phoenix. This is exactly the gap that made `record_edit`'s old
    `datetime.now(UTC).date()` disagree with a test's stdlib `date.
    today()` (the system's OWN local timezone, whatever that happens to
    be) for part of every day -- deterministic here (no dependence on
    when the suite happens to run), unlike asserting on the live clock."""
    instant = datetime(2026, 9, 12, 3, 0, tzinfo=ZoneInfo("UTC"))  # 03:00 UTC
    utc_date = instant.astimezone(ZoneInfo("UTC")).date()
    phoenix_date = instant.astimezone(ZoneInfo("America/Phoenix")).date()
    assert utc_date == date(2026, 9, 12)
    assert phoenix_date == date(2026, 9, 11)


def test_returns_plain_date_and_aware_datetime() -> None:
    assert isinstance(clock.today(), date)
    assert not isinstance(clock.today(), datetime)
    aware = clock.now()
    assert isinstance(aware, datetime)
    assert aware.tzinfo is not None
