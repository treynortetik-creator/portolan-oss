"""Regression tests for the engine bug hunt, wave 2, 2026-09-22 -- the
weekly report half (`01-spec/reviews/bughunt-wave2-2026-09-22.md`). Each
test names the finding it pins; each one failed against the code before
its fix. Uses `tests/test_report_helpers.py`'s shared synthetic graph and
fake model, no network, no paid calls.
"""

# ruff: noqa: F811  (pytest fixtures imported by name; see test_report_helpers.py)

from __future__ import annotations

from datetime import date

import pytest
from test_report_helpers import (  # noqa: F401
    FullGraph,
    arr_by_note_slug,
    full_graph,
    report_llm,
)
from test_report_researcher import _minimal_card

from portolan.report import weekly
from portolan.report.accounts import load_account_arr
from portolan.report.researcher import queue_action_card


def _standing_question(conn, question: str) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM standing_questions WHERE question = %s", (question,))
        row = cur.fetchone()
        if row is None:
            cur.execute(
                "INSERT INTO standing_questions (reader, question) VALUES (%s, %s) RETURNING id",
                ("ops", question),
            )
            row = cur.fetchone()
    conn.commit()
    return row[0]


def test_standing_answers_are_stored_under_the_reports_own_week(
    full_graph: FullGraph, report_llm, arr_by_note_slug: dict[str, float]
) -> None:
    """Finding 10. `--as-of 2026-09-21` (a Monday) windows 09-15..09-21,
    six days of W38, and the report is labelled W38 (the window's middle
    day). The standing answers were keyed by `as_of`'s own week, W39: the
    W38 row never existed and W39's row held W38's answer until W39's own
    run overwrote it."""
    question_id = _standing_question(
        full_graph.conn, "Which accounts mentioned a competitor this week?"
    )

    result = weekly.build_weekly(
        full_graph.conn,
        report_llm,
        full_graph.embedder,
        week=None,
        as_of=date(2026, 9, 21),
        arr_by_note_slug=arr_by_note_slug,
    )
    assert result.week == "2026-W38"

    with full_graph.conn.cursor() as cur:
        cur.execute(
            "SELECT week FROM standing_question_answers WHERE question_id = %s "
            "AND week IN ('2026-W38', '2026-W39')",
            (question_id,),
        )
        weeks = sorted(r[0] for r in cur.fetchall())
    assert weeks == ["2026-W38"]


@pytest.mark.parametrize("closed_status", ["dismissed", "resolved"])
def test_a_research_regen_does_not_repropose_a_card_already_closed_that_week(
    db, closed_status: str
) -> None:
    """Finding 11. `queue_action_card` skipped a candidate only while its
    card was still OPEN. Regenerating W36 with `--research` after a
    reviewer dismissed or confirmed the card queued it again: the W36
    report read "(dismissed)" and "(proposed ...)" for the same candidate,
    and a second Confirm would create a second action."""
    card = _minimal_card(candidate_slug="cand-closed", account="acc", what="Call the account")
    assert queue_action_card(db, card, week="2026-W36") is True
    with db.cursor() as cur:
        cur.execute(
            "UPDATE review_queue SET status = %s, resolved_by = 'Jordan Blake', "
            "resolved_at = now() WHERE kind = 'action_card'",
            (closed_status,),
        )
    db.commit()

    assert queue_action_card(db, card, week="2026-W36") is False
    with db.cursor() as cur:
        cur.execute("SELECT count(*) FROM review_queue WHERE kind = 'action_card'")
        assert cur.fetchone()[0] == 1


def test_a_new_week_may_still_propose_a_card_dismissed_last_week(db) -> None:
    """Finding 11 guard: the same situation can deserve a card again in a
    later week; only the same week is closed for good."""
    card = _minimal_card(candidate_slug="cand-again", account="acc", what="Call the account")
    queue_action_card(db, card, week="2026-W36")
    with db.cursor() as cur:
        cur.execute("UPDATE review_queue SET status = 'dismissed' WHERE kind = 'action_card'")
    db.commit()

    assert queue_action_card(db, card, week="2026-W37") is True


def test_an_unparseable_arr_cell_is_queued_once_not_once_per_report_run(db, tmp_path) -> None:
    """Finding 12. Every weekly run (and every research run) re-reads the
    CRM export and filed a fresh `arr_not_parseable` review row for the
    same dirty cell, so the review queue grew by one row per week per bad
    cell for as long as the export stayed dirty."""
    accounts = tmp_path / "accounts.csv"
    accounts.write_text("crm_id,name,arr\nA-1,Acme,N/A\nA-2,Bolt,1000\n")

    first = load_account_arr(accounts, db)
    second = load_account_arr(accounts, db)

    assert first == second == {"A-2": 1000.0}
    with db.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM review_queue WHERE payload ->> 'reason' = 'arr_not_parseable'"
        )
        assert cur.fetchone()[0] == 1
