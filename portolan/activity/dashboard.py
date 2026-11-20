"""The numbers behind `/admin/usage`: weekly active users, questions per
user, the thumbs ratio, citation click-through, email opens and clicks, and
the questions that got no answer. Weeks start Monday, in UTC.

"Active" means any in-app event (`APP_KINDS`); an email open alone does not
count. Click-through is the share of cited answers with at least one click
from that answer through to an original. An "unanswered" question is one
whose answer carried no citation (`mode` other than `answered`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Any

import psycopg

__all__ = ["APP_KINDS", "PilotDashboard", "pilot_dashboard"]

APP_KINDS = (
    "page_view",
    "chat_question",
    "citation_click",
    "review_decision",
    "action_confirm",
    "feedback",
)


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


@dataclass
class PilotDashboard:
    now: datetime
    window_days: int
    weeks: list[dict[str, Any]] = field(default_factory=list)
    totals: dict[str, Any] = field(default_factory=dict)
    per_user: list[dict[str, Any]] = field(default_factory=list)
    unanswered: list[dict[str, Any]] = field(default_factory=list)
    candidates_waiting: int = 0


def _week_start(day: date) -> date:
    return day - timedelta(days=day.weekday())


def _weekly(conn: psycopg.Connection, start: datetime, now: datetime) -> dict[date, dict]:
    rows: dict[date, dict[str, Any]] = {}
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT date_trunc('week', occurred_at AT TIME ZONE 'UTC')::date AS wk,
                   count(DISTINCT user_name) FILTER (WHERE kind = ANY(%(app)s)) AS active,
                   count(*) FILTER (WHERE kind = 'chat_question') AS questions,
                   count(*) FILTER (WHERE kind = 'email_open') AS opens,
                   count(DISTINCT user_name) FILTER (WHERE kind = 'email_open') AS openers,
                   count(*) FILTER (WHERE kind = 'email_click') AS clicks
            FROM activity_events
            WHERE occurred_at >= %(start)s AND occurred_at <= %(now)s
            GROUP BY 1
            """,
            {"app": list(APP_KINDS), "start": start, "now": now},
        )
        for wk, active, questions, opens, openers, clicks in cur:
            rows[wk] = {
                "active_users": active,
                "questions": questions,
                "email_opens": opens,
                "email_openers": openers,
                "email_clicks": clicks,
            }
        cur.execute(
            """
            SELECT date_trunc('week', q.occurred_at AT TIME ZONE 'UTC')::date AS wk,
                   count(*) AS cited,
                   count(*) FILTER (WHERE EXISTS (
                       SELECT 1 FROM activity_events c
                       WHERE c.kind = 'citation_click'
                         AND c.detail ->> 'message_id' = q.detail ->> 'message_id'
                   )) AS clicked
            FROM activity_events q
            WHERE q.kind = 'chat_question'
              AND coalesce((q.detail ->> 'citations')::int, 0) > 0
              AND q.occurred_at >= %(start)s AND q.occurred_at <= %(now)s
            GROUP BY 1
            """,
            {"start": start, "now": now},
        )
        for wk, cited, clicked in cur:
            rows.setdefault(wk, {}).update({"cited_answers": cited, "clicked_answers": clicked})
        cur.execute(
            """
            SELECT date_trunc('week', created_at AT TIME ZONE 'UTC')::date AS wk,
                   count(*) FILTER (WHERE rating = 'up'),
                   count(*) FILTER (WHERE rating = 'down')
            FROM answer_feedback
            WHERE created_at >= %(start)s AND created_at <= %(now)s
            GROUP BY 1
            """,
            {"start": start, "now": now},
        )
        for wk, up, down in cur:
            rows.setdefault(wk, {}).update({"thumbs_up": up, "thumbs_down": down})
    return rows


def _finish(row: dict[str, Any]) -> dict[str, Any]:
    out = {
        "active_users": 0,
        "questions": 0,
        "email_opens": 0,
        "email_openers": 0,
        "email_clicks": 0,
        "cited_answers": 0,
        "clicked_answers": 0,
        "thumbs_up": 0,
        "thumbs_down": 0,
        **row,
    }
    out["questions_per_user"] = _ratio(out["questions"], out["active_users"])
    out["thumbs_up_share"] = _ratio(out["thumbs_up"], out["thumbs_up"] + out["thumbs_down"])
    out["click_through"] = _ratio(out["clicked_answers"], out["cited_answers"])
    return out


def _totals(conn: psycopg.Connection, since: datetime, now: datetime) -> dict[str, Any]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT count(DISTINCT user_name) FILTER (WHERE kind = ANY(%(app)s)),
                   count(*) FILTER (WHERE kind = 'chat_question'),
                   count(*) FILTER (WHERE kind = 'email_open'),
                   count(DISTINCT user_name) FILTER (WHERE kind = 'email_open'),
                   count(*) FILTER (WHERE kind = 'email_click'),
                   count(*) FILTER (WHERE kind = 'chat_question'
                                    AND coalesce((detail ->> 'citations')::int, 0) > 0),
                   count(*) FILTER (WHERE kind = 'chat_question'
                                    AND coalesce((detail ->> 'citations')::int, 0) > 0
                                    AND EXISTS (
                                        SELECT 1 FROM activity_events c
                                        WHERE c.kind = 'citation_click'
                                          AND c.detail ->> 'message_id'
                                              = activity_events.detail ->> 'message_id'))
            FROM activity_events
            WHERE occurred_at >= %(since)s AND occurred_at <= %(now)s
            """,
            {"app": list(APP_KINDS), "since": since, "now": now},
        )
        active, questions, opens, openers, clicks, cited, clicked = cur.fetchone()
        cur.execute(
            """
            SELECT count(*) FILTER (WHERE rating = 'up'), count(*) FILTER (WHERE rating = 'down')
            FROM answer_feedback WHERE created_at >= %s AND created_at <= %s
            """,
            (since, now),
        )
        up, down = cur.fetchone()
    return _finish(
        {
            "active_users": active,
            "questions": questions,
            "email_opens": opens,
            "email_openers": openers,
            "email_clicks": clicks,
            "cited_answers": cited,
            "clicked_answers": clicked,
            "thumbs_up": up,
            "thumbs_down": down,
        }
    )


def _per_user(conn: psycopg.Connection, since: datetime, now: datetime) -> list[dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT e.user_name,
                   count(*) FILTER (WHERE e.kind = 'chat_question') AS questions,
                   count(*) FILTER (WHERE e.kind = 'page_view') AS page_views,
                   count(*) FILTER (WHERE e.kind = 'citation_click') AS citation_clicks,
                   count(DISTINCT (e.occurred_at AT TIME ZONE 'UTC')::date) AS active_days,
                   max(e.occurred_at) AS last_seen
            FROM activity_events e
            WHERE e.kind = ANY(%(app)s) AND e.user_name IS NOT NULL
              AND e.occurred_at >= %(since)s AND e.occurred_at <= %(now)s
            GROUP BY e.user_name
            ORDER BY questions DESC, last_seen DESC
            """,
            {"app": list(APP_KINDS), "since": since, "now": now},
        )
        cols = [d.name for d in cur.description]
        users = [dict(zip(cols, r, strict=True)) for r in cur]
        cur.execute(
            """
            SELECT user_name, count(*) FILTER (WHERE rating = 'up'),
                   count(*) FILTER (WHERE rating = 'down')
            FROM answer_feedback WHERE created_at >= %s AND created_at <= %s
            GROUP BY user_name
            """,
            (since, now),
        )
        thumbs = {name: (up, down) for name, up, down in cur}
    for user in users:
        user["thumbs_up"], user["thumbs_down"] = thumbs.get(user["user_name"], (0, 0))
        user["last_seen"] = user["last_seen"].astimezone(UTC)
    return users


def _unanswered(
    conn: psycopg.Connection, since: datetime, now: datetime, limit: int
) -> list[dict[str, Any]]:
    """Questions whose answer had no citation, grouped by their text with
    case and spacing ignored, most asked first."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT min(detail ->> 'question') AS question,
                   count(*) AS times_asked,
                   count(DISTINCT user_name) AS people,
                   max(occurred_at) AS last_asked,
                   array_agg(DISTINCT detail ->> 'mode') AS modes
            FROM activity_events
            WHERE kind = 'chat_question'
              AND coalesce(detail ->> 'mode', '') <> 'answered'
              AND occurred_at >= %(since)s AND occurred_at <= %(now)s
            GROUP BY lower(regexp_replace(btrim(detail ->> 'question'), '\\s+', ' ', 'g'))
            ORDER BY times_asked DESC, last_asked DESC
            LIMIT %(limit)s
            """,
            {"since": since, "now": now, "limit": limit},
        )
        cols = [d.name for d in cur.description]
        rows = [dict(zip(cols, r, strict=True)) for r in cur]
    for row in rows:
        row["last_asked"] = row["last_asked"].astimezone(UTC)
    return rows


def pilot_dashboard(
    conn: psycopg.Connection,
    *,
    now: datetime | None = None,
    weeks: int = 8,
    window_days: int = 30,
    top_unanswered: int = 20,
) -> PilotDashboard:
    now = now or datetime.now(UTC)
    this_monday = _week_start(now.astimezone(UTC).date())
    mondays = [this_monday - timedelta(weeks=i) for i in range(weeks)]
    start = datetime.combine(mondays[-1], datetime.min.time(), tzinfo=UTC)
    since = now - timedelta(days=window_days)

    weekly = _weekly(conn, start, now)
    dash = PilotDashboard(now=now, window_days=window_days)
    dash.weeks = [{"week_start": m, **_finish(weekly.get(m, {}))} for m in mondays]
    dash.totals = _totals(conn, since, now)
    dash.per_user = _per_user(conn, since, now)
    dash.unanswered = _unanswered(conn, since, now, top_unanswered)
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM eval_candidates WHERE status = 'candidate'")
        dash.candidates_waiting = cur.fetchone()[0]
    return dash
