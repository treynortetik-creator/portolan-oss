"""One clock for the whole deployment (e2e-run follow-up wave, item 2:
"the two failing tests ... use the same helper so they pass at any hour").

Before this module existed, `record_edit` stamped its citation with
`datetime.now(UTC).date()` while its own tests compared against stdlib
`date.today()` (the *system's local* timezone, whatever that happens to
be on the machine running the suite). Those two clocks agree except in
the roughly-seven-hour window each day where America/Phoenix (the maintainer's
own timezone, `~/.claude/CLAUDE.md`: "no daylight saving") has already
turned over to a new calendar day but UTC has not yet, or vice versa --
a test run landing in that window sees `record_edit`'s stamped date and
the test's own `date.today()` disagree, for no reason a reader of either
file could see locally. `today()`/`now()` below are the one place either
"today" or "now" gets computed anywhere in this build's owned files
(`portolan/report/**`, `portolan/consolidate/edits.py`,
`portolan/notes/**`, `portolan/merge/**`) -- every prior `date.today()`/
`datetime.now(UTC)` in those files now calls through here instead, and so
does the test suite wherever it needs to assert against "today".

Deliberately NOT a new `Settings` field (task instruction: "add nothing
to Settings"): `PORTOLAN_TIMEZONE` is read straight from the environment
here, an IANA zone name for `zoneinfo`, defaulting to `America/Phoenix`.
A customer deployment elsewhere sets this one variable in its own `.env`;
nothing in this module hardcodes the maintainer's own timezone as anything more
than the default.

`now()` changes only which (equally valid) UTC offset a fresh, correct
instant is *labelled* with -- `datetime.now(UTC)` and `now()` name the
exact same moment, just with different `tzinfo` attached, and a
`timestamptz` column stores the instant, not the offset, so every
`note.updated = clock.now()`-style write in this codebase round-trips
through Postgres identically to before. `today()` is where the fix
actually bites: `now().date()` reads the calendar date in the
deployment's own zone, not whichever zone the process host happens to be
in.
"""

from __future__ import annotations

import os
from datetime import date, datetime
from zoneinfo import ZoneInfo

__all__ = ["DEFAULT_TIMEZONE", "now", "timezone_name", "today", "world_date", "world_date_sql"]

DEFAULT_TIMEZONE = "America/Phoenix"


def _zone() -> ZoneInfo:
    """Re-read `PORTOLAN_TIMEZONE` on every call, same reasoning as
    `portolan.config.get_settings` (`Settings` docstring): a test that
    does `monkeypatch.setenv("PORTOLAN_TIMEZONE", ...)` must see the
    override immediately, with nothing cached from an earlier import."""
    return ZoneInfo(os.environ.get("PORTOLAN_TIMEZONE", DEFAULT_TIMEZONE))


def now() -> datetime:
    """The current instant, timezone-aware, labelled in the deployment's
    own zone (`PORTOLAN_TIMEZONE`, default `America/Phoenix`)."""
    return datetime.now(_zone())


def today() -> date:
    """The current calendar date in the deployment's own zone -- the one
    "today" every owned module and its tests should agree on."""
    return now().date()


def timezone_name() -> str:
    """The deployment's configured IANA zone name (`PORTOLAN_TIMEZONE`),
    re-read every call like `_zone()`. Round-tripped through `ZoneInfo`
    first (`_zone().key`) so anything a caller embeds downstream --
    `world_date_sql` puts this straight into a SQL string literal -- is
    guaranteed to be a zone name `zoneinfo` itself already accepted, never
    an unvalidated environment string."""
    return _zone().key  # type: ignore[return-value]


def world_date_sql(column: str) -> str:
    """SQL fragment converting a `timestamptz` column to the calendar date
    in the deployment's own zone (`PORTOLAN_TIMEZONE`), regardless of
    whatever `TimeZone` the current database session happens to be set
    to.

    Review finding 1 (`01-spec/reviews/review-report-audit-consolidate.md`):
    a bare `column::date` cast resolves in the *session's* timezone, not
    the deployment's. Before this fix, `portolan.merge.util.recorded_dt`
    stamped `claims.recorded` at midnight UTC unconditionally, so a
    session whose own `TimeZone` wasn't UTC read that instant back as the
    *previous* calendar day (proven on this machine, session `TimeZone`
    `America/Phoenix`: a 2026-08-24 source read back as 2026-08-23) --
    `recorded_dt` now anchors at midnight in this same configured zone
    instead (its own docstring, "fix half 1 of 2"), which is what makes
    casting through the zone name explicitly here -- `column AT TIME ZONE
    '<zone>'` -- round-trip back to the original calendar day: the result
    no longer depends on the session's own setting at all, so the same
    query gives the same, correct date whether the session happens to be
    UTC, Phoenix, or anything else.

    Every report/audit/consolidate query that means "which calendar day,
    in the deployment's own reckoning, does this fact belong to" --
    whether that's `claims.recorded` (the world clock, the source's own
    date) or `notes.created`/`updated`/`claims.superseded_at` (the record
    clock, when Portolan itself wrote something) -- should build its cast
    through this helper rather than writing its own `::date`, so there is
    one place, not N, that decides what "today" means for a read.

    The zone name is embedded as a SQL string literal rather than a bound
    parameter, so callers can keep using positional `%s` placeholders for
    their own values without reordering around an extra parameter --
    `timezone_name()` guarantees it's never anything but a zone name
    `zoneinfo` accepted, so there is nothing here for a caller-controlled
    value to inject (`PORTOLAN_TIMEZONE` is a deployment environment
    variable, not user input, same trust boundary as `_zone()` already
    assumes).
    """
    return f"(({column}) AT TIME ZONE '{timezone_name()}')::date"


def world_date(value: datetime) -> date:
    """The Python-side equivalent of `world_date_sql`: convert a
    timezone-aware `datetime` -- e.g. a `timestamptz` value psycopg has
    already handed back, labelled in whatever zone the session happened
    to be in -- to the calendar date in the deployment's own zone, rather
    than calling `.date()` directly and silently inheriting the session's
    zone the same way a bare SQL `::date` cast does (finding 1)."""
    return value.astimezone(_zone()).date()
