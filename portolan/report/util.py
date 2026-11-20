"""Small helpers shared across `portolan.report`: the ISO-week string used
by `reports.week` / `standing_question_answers.week`, the report note's slug
(`report-<ISO week>`), and one citation-marker check for every place a model
writes prose over a `^[source @ locator]`-tagged payload.
"""

from __future__ import annotations

from datetime import date, timedelta

from portolan.notes.render import CITATION_RE

__all__ = [
    "citation_markers_in",
    "iso_week_str",
    "report_note_slug",
    "validate_citations",
    "week_bounds",
    "week_start",
]


def iso_week_str(d: date) -> str:
    """`"2026-W37"` -- the ISO year and week `d` falls in. The ISO year can
    differ from `d.year` around New Year, hence `isocalendar()`."""
    iso = d.isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


def week_start(d: date) -> date:
    """The Monday of the ISO week containing `d`."""
    return d - timedelta(days=d.weekday())


def week_bounds(week: str) -> tuple[date, date]:
    """`(monday, sunday)` for a `"YYYY-Www"` string; the inverse of
    `iso_week_str` + `week_start`."""
    year_str, week_str = week.split("-W")
    monday = date.fromisocalendar(int(year_str), int(week_str), 1)
    return monday, monday + timedelta(days=6)


def report_note_slug(week: str) -> str:
    return f"report-{week}"


def citation_markers_in(text: str) -> set[tuple[str, str]]:
    """Every `^[source-slug @ locator]` marker in `text`, as `(source_slug,
    locator)` pairs."""
    return {(m.group(1).strip(), m.group(2).strip()) for m in CITATION_RE.finditer(text)}


def validate_citations(
    text: str, allowed_markers: set[tuple[str, str]], *, max_lines: int | None = None
) -> str | None:
    """`None` if `text` passes: every citation marker in it is in
    `allowed_markers`, and (when `max_lines` is given) it has at most that
    many non-blank lines. Otherwise, the reason it failed."""
    if max_lines is not None:
        lines = [line for line in text.splitlines() if line.strip()]
        if len(lines) > max_lines:
            return f"text has {len(lines)} non-blank lines, over the {max_lines}-line limit"
    for marker in citation_markers_in(text):
        if marker not in allowed_markers:
            return f"citation marker ^[{marker[0]} @ {marker[1]}] is not in the given payload"
    return None
