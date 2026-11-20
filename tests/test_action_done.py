"""Auto-close actions from later evidence (`portolan.report.action_done`,
checklist item 24): a later claim by our own side on the same account that
answers an open action becomes one "Proposed done" review row; a person
confirms or dismisses it on `/review`.

The fixture graph is one open action for Ashgrove Medical Center (Sam
Okafor, from a call on 2026-08-20) plus whichever later claims a test adds.
The judge is a stub: it answers "supported" only for a claim carrying
`ANSWER_MARK`, and records every call, so a test can tell "filtered before
the judge" from "judged and refused".
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import psycopg
import pytest
from fastapi.testclient import TestClient

from portolan.config import Settings
from portolan.extract.schema import FactType
from portolan.llm.fake import FakeLLMClient
from portolan.merge.actions import merge_action
from portolan.merge.facts import merge_fact
from portolan.registry import Registry
from portolan.report import action_done, weekly
from portolan.report.action_done import (
    ANSWER_FACT_TYPES,
    done_evidence_errors,
    open_actions,
    propose_done_actions,
)
from portolan.report.researcher_tools import absence_check
from portolan.search.embed import FakeEmbedder
from portolan.store import LocalStore
from portolan.web.app import create_app
from portolan.web.deps import get_conn, get_embedder_dep, get_llm_dep, get_store_dep
from tests.merge_helpers import (
    ASHGROVE_MEDICAL,
    build_action,
    build_fact,
    loaded_registry_fixture,  # noqa: F401
    make_source_row,
)

ACTION_DATE = date(2026, 8, 20)
WEEK = "2026-W36"
AS_OF = date(2026, 9, 6)
ACTION_WHAT = "Follow up with the support team about the late Tidewater T2 delivery"
ANSWER_MARK = "emailed the support team"
FOLLOW_UP = (
    "Sam Okafor emailed the support team about Ashgrove Medical Center's late Tidewater T2 "
    "delivery."
)
ORG_KEY = "test-org-key-for-action-done-review"
SESSION_SECRET = "test-session-secret-for-action-done-review-not-a-real-secret"
EDITOR = "Test Editor"


class StubJudge:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def __call__(self, *, sentence: str, reference_text: str) -> str | None:
        self.calls.append((sentence, reference_text))
        return "supported" if ANSWER_MARK in reference_text else "not_supported"


def _open_action(db, registry: Registry) -> int:
    source = make_source_row(db, "s-done-call", source_date=ACTION_DATE)
    action = build_action(ACTION_WHAT, owner_mention="Sam Okafor", locator="00:10:00")
    outcome = merge_action(db, registry, action, source, entity_id=ASHGROVE_MEDICAL)
    assert outcome.status == "new"
    return outcome.action_id


def _claim(
    db,
    registry: Registry,
    *,
    slug: str,
    day: date,
    claim: str,
    speaker: str,
    quote: str = "per the email thread",
) -> int:
    source = make_source_row(db, slug, source_type="email", source_date=day)
    fact = build_fact(
        f"F-{slug}",
        FactType.COMMITMENT,
        claim,
        ["Ashgrove Medical Center", "Tidewater T2"],
        locator="p1",
        quote=quote,
        speaker=speaker,
    )
    llm = FakeLLMClient(lambda **_kwargs: {"verdict": "extends", "reason": "distinct"})
    outcome = merge_fact(db, llm, FakeEmbedder(), registry, fact, source)
    assert outcome.status in ("new", "extends")
    return outcome.claim_id


def _follow_up(db, registry: Registry, *, day: date = date(2026, 8, 21)) -> int:
    return _claim(
        db,
        registry,
        slug="s-done-followup",
        day=day,
        claim=FOLLOW_UP,
        speaker="Sam Okafor",
        quote="I emailed support this morning",
    )


def _propose(db, judge: StubJudge) -> list[action_done.DoneProposalOutcome]:
    return propose_done_actions(db, judge=judge, embedder=FakeEmbedder(), as_of=AS_OF, week=WEEK)


def _done_rows(db, status: str | None = None) -> list[tuple[int, dict, str | None, str]]:
    with db.cursor() as cur:
        cur.execute(
            "SELECT id, payload, source_slug, status FROM review_queue WHERE kind = 'action_done' "
            "AND (%s::text IS NULL OR status = %s) ORDER BY id",
            (status, status),
        )
        return cur.fetchall()


def _action_status(db, action_id: int) -> str:
    with db.cursor() as cur:
        cur.execute("SELECT status FROM actions WHERE id = %s", (action_id,))
        return cur.fetchone()[0]


# ---------------------------------------------------------------------
# The presence check.
# ---------------------------------------------------------------------


def test_the_fixture_action_resolves_to_its_account(db, loaded_registry: Registry) -> None:
    action_id = _open_action(db, loaded_registry)
    [action] = open_actions(db)
    assert action.id == action_id
    assert action.account_slug is not None
    assert action.action_date == ACTION_DATE


def test_a_real_follow_up_by_the_owner_proposes_done(db, loaded_registry: Registry) -> None:
    action_id = _open_action(db, loaded_registry)
    claim_id = _follow_up(db, loaded_registry)
    judge = StubJudge()

    [outcome] = _propose(db, judge)

    assert outcome.status == "proposed"
    assert outcome.claim_id == claim_id
    [(row_id, payload, source_slug, status)] = _done_rows(db)
    assert (row_id, status) == (outcome.row_id, "open")
    assert payload["action_id"] == action_id
    assert payload["claim_id"] == claim_id
    assert payload["speaker"] == "Sam Okafor"
    assert payload["claim_date"] == "2026-08-21"
    assert payload["week"] == WEEK
    with db.cursor() as cur:
        cur.execute(
            "SELECT claim_text, quote, source_slug, locator FROM claims WHERE id = %s",
            (claim_id,),
        )
        text, quote, claim_source, locator = cur.fetchone()
    assert payload["evidence"] == {
        "claim_id": claim_id,
        "text": text,
        "quote": quote,
        "source_slug": claim_source,
        "locator": locator,
    }
    assert source_slug == claim_source
    assert done_evidence_errors(db, payload, as_of=AS_OF) == []
    [(sentence, reference)] = judge.calls
    assert ACTION_WHAT in sentence
    assert ANSWER_MARK in reference


def test_the_recorded_finding_is_what_absence_check_computes(db, loaded_registry: Registry) -> None:
    _open_action(db, loaded_registry)
    _follow_up(db, loaded_registry)
    _propose(db, StubJudge())
    [(_row_id, payload, _source, _status)] = _done_rows(db)
    with db.cursor() as cur:
        cur.execute(
            "SELECT tool, arguments, result FROM researcher_findings WHERE id = %s",
            (payload["finding_id"],),
        )
        tool, arguments, recorded = cur.fetchone()

    _finding_id, rerun = absence_check(
        db,
        run_id="rerun",
        candidate_slug="rerun",
        account_slug=arguments["account_slug"],
        theme_slug=None,
        fact_types=arguments["fact_types"],
        after_date=date.fromisoformat(arguments["after_date"]),
    )

    assert tool == "absence_check"
    assert arguments["fact_types"] == ANSWER_FACT_TYPES
    assert arguments["after_date"] == ACTION_DATE.isoformat()
    assert payload["claim_id"] in recorded["claim_ids"]
    assert sorted(recorded["claim_ids"]) == sorted(rerun["claim_ids"])
    assert recorded["count"] == rerun["count"]
    assert recorded["count_before"] == rerun["count_before"]


def test_an_unrelated_later_claim_by_the_owner_does_not(db, loaded_registry: Registry) -> None:
    action_id = _open_action(db, loaded_registry)
    _claim(
        db,
        loaded_registry,
        slug="s-done-qbr",
        day=date(2026, 8, 22),
        claim="Sam Okafor scheduled a quarterly business review with Ashgrove Medical Center.",
        speaker="Sam Okafor",
    )
    judge = StubJudge()

    [outcome] = _propose(db, judge)

    assert outcome.status == "not_answered"
    assert len(judge.calls) == 1  # judged and refused, not filtered out
    assert _done_rows(db) == []
    assert _action_status(db, action_id) == "open"


def test_a_customer_side_claim_does_not(db, loaded_registry: Registry) -> None:
    _open_action(db, loaded_registry)
    _claim(
        db,
        loaded_registry,
        slug="s-done-customer",
        day=date(2026, 8, 22),
        claim="Rob Tallis emailed the support team about the late Tidewater T2 delivery.",
        speaker="Rob Tallis",
    )
    judge = StubJudge()

    [outcome] = _propose(db, judge)

    assert outcome.status == "no_candidates"
    assert judge.calls == []
    assert _done_rows(db) == []


def test_a_claim_dated_before_the_action_does_not(db, loaded_registry: Registry) -> None:
    _open_action(db, loaded_registry)
    _follow_up(db, loaded_registry, day=date(2026, 8, 19))
    judge = StubJudge()

    [outcome] = _propose(db, judge)

    assert outcome.status == "no_candidates"
    assert judge.calls == []
    assert _done_rows(db) == []


def test_a_claim_dated_after_the_week_does_not(db, loaded_registry: Registry) -> None:
    _open_action(db, loaded_registry)
    _follow_up(db, loaded_registry, day=date(2026, 9, 8))
    judge = StubJudge()

    [outcome] = _propose(db, judge)

    assert outcome.status == "no_candidates"
    assert judge.calls == []


def test_a_rerun_does_not_duplicate(db, loaded_registry: Registry) -> None:
    _open_action(db, loaded_registry)
    _follow_up(db, loaded_registry)
    judge = StubJudge()
    _propose(db, judge)
    _claim(
        db,
        loaded_registry,
        slug="s-done-second",
        day=date(2026, 8, 25),
        claim="Sam Okafor emailed the support team a second time about the Tidewater T2 delivery.",
        speaker="Sam Okafor",
    )

    [outcome] = _propose(db, judge)

    assert outcome.status == "already_proposed"
    assert len(judge.calls) == 1
    assert len(_done_rows(db)) == 1


def test_the_database_allows_one_open_proposal_per_action(db) -> None:
    insert = "INSERT INTO review_queue (kind, payload) VALUES ('action_done', %s::jsonb)"
    with db.cursor() as cur:
        cur.execute(insert, ('{"action_id": 7, "claim_id": 1}',))
        with pytest.raises(psycopg.errors.UniqueViolation):
            cur.execute(insert, ('{"action_id": 7, "claim_id": 2}',))


def test_a_dry_run_writes_nothing_and_calls_no_judge(db, loaded_registry: Registry) -> None:
    _open_action(db, loaded_registry)
    claim_id = _follow_up(db, loaded_registry)
    judge = StubJudge()

    [outcome] = propose_done_actions(
        db, judge=judge, embedder=FakeEmbedder(), as_of=AS_OF, week=WEEK, dry_run=True
    )

    assert outcome.status == "candidates"
    assert outcome.candidate_claim_ids == [claim_id]
    assert judge.calls == []
    assert _done_rows(db) == []
    with db.cursor() as cur:
        cur.execute("SELECT count(*) FROM researcher_findings")
        assert cur.fetchone()[0] == 0


# ---------------------------------------------------------------------
# The review queue: render, Confirm, Dismiss.
# ---------------------------------------------------------------------


@pytest.fixture()
def review_client(db: psycopg.Connection, db_url: str, tmp_path: Path):
    store_root = tmp_path / "store"
    store_root.mkdir()
    settings = Settings(
        database_url=db_url,
        tenant="test",
        store_root=store_root,
        llm_provider="fake",
        embed_provider="fake",
        org_key=ORG_KEY,
        session_secret=SESSION_SECRET,
        cookie_secure=False,
    )
    app = create_app(settings)
    app.dependency_overrides[get_conn] = lambda: db
    app.dependency_overrides[get_embedder_dep] = lambda: FakeEmbedder()
    app.dependency_overrides[get_llm_dep] = lambda: FakeLLMClient([])
    app.dependency_overrides[get_store_dep] = lambda: LocalStore(store_root)
    with TestClient(app) as client:
        response = client.post(
            "/login", data={"org_key": ORG_KEY, "editor": EDITOR}, follow_redirects=False
        )
        assert response.status_code == 303
        yield client


def _proposed(db, registry: Registry) -> tuple[int, int, int, str]:
    """(action id, review row id, claim id, action note slug)."""
    action_id = _open_action(db, registry)
    claim_id = _follow_up(db, registry)
    [outcome] = _propose(db, StubJudge())
    assert outcome.status == "proposed"
    with db.cursor() as cur:
        cur.execute("SELECT note_slug FROM actions WHERE id = %s", (action_id,))
        slug = cur.fetchone()[0]
    return action_id, outcome.row_id, claim_id, slug


def test_the_review_page_shows_the_proposal(
    review_client: TestClient, db, loaded_registry: Registry
) -> None:
    _action_id, _row_id, _claim_id, slug = _proposed(db, loaded_registry)

    page = review_client.get("/review?kind=Proposed%20done")

    assert page.status_code == 200
    assert "Proposed done" in page.text
    assert "Sam Okafor recorded something on 2026-08-21" in page.text
    assert FOLLOW_UP.split("Ashgrove")[0] in page.text
    assert "/originals/s-done-followup?at=p1" in page.text
    assert f'href="/notes/{slug}"' in page.text
    assert "Confirm done" in page.text


def test_confirm_marks_the_action_done_and_keeps_the_evidence(
    review_client: TestClient, db, loaded_registry: Registry
) -> None:
    action_id, row_id, claim_id, slug = _proposed(db, loaded_registry)

    response = review_client.post(f"/review/{row_id}/resolve", follow_redirects=False)

    assert response.status_code == 303
    assert response.headers["location"] == "/review?notice=action_done_confirmed"
    assert _action_status(db, action_id) == "done"
    with db.cursor() as cur:
        cur.execute(
            "SELECT editor, before_text, after_text FROM edit_records WHERE note_slug = %s "
            "AND after_text = 'done'",
            (slug,),
        )
        assert cur.fetchall() == [(EDITOR, "open", "done")]
        cur.execute(
            "SELECT claim_id, evidence ->> 'text', confirmed_by, review_row_id "
            "FROM action_done_evidence WHERE action_id = %s",
            (action_id,),
        )
        evidence_claim, evidence_text, confirmed_by, evidence_row = cur.fetchone()
        cur.execute("SELECT status, resolved_by FROM review_queue WHERE id = %s", (row_id,))
        assert cur.fetchone() == ("resolved", EDITOR)
    assert (evidence_claim, confirmed_by, evidence_row) == (claim_id, EDITOR, row_id)
    assert "emailed the support team" in evidence_text

    note = review_client.get(f"/notes/{slug}")
    assert note.status_code == 200
    assert '<p class="eyebrow">Done</p>' in note.text
    assert f"Marked done by {EDITOR}" in note.text
    assert "Open the original at p1" in note.text

    download = review_client.get(f"/notes/{slug}.md")
    assert "\n## Done\n" in download.text
    assert "[source: s-done-followup @ p1]" in download.text

    atlas = review_client.get(f"/atlas/{slug}")
    assert '<p class="eyebrow">Done</p>' in atlas.text


def test_a_second_confirm_changes_nothing(
    review_client: TestClient, db, loaded_registry: Registry
) -> None:
    _action_id, row_id, _claim_id, slug = _proposed(db, loaded_registry)
    review_client.post(f"/review/{row_id}/resolve", follow_redirects=False)

    again = review_client.post(f"/review/{row_id}/resolve", follow_redirects=False)

    assert again.headers["location"] == "/review?notice=already_handled"
    with db.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM edit_records WHERE note_slug = %s AND after_text = 'done'",
            (slug,),
        )
        assert cur.fetchone()[0] == 1


def test_confirm_refuses_when_the_evidence_changed(
    review_client: TestClient, db, loaded_registry: Registry
) -> None:
    action_id, row_id, claim_id, _slug = _proposed(db, loaded_registry)
    with db.cursor() as cur:
        cur.execute(
            "UPDATE claims SET claim_text = 'Sam Okafor has not contacted support.' WHERE id = %s",
            (claim_id,),
        )

    response = review_client.post(f"/review/{row_id}/resolve", follow_redirects=False)

    assert response.headers["location"] == "/review?notice=action_done_evidence_changed"
    assert _action_status(db, action_id) == "open"
    assert _done_rows(db, "open")[0][0] == row_id
    with db.cursor() as cur:
        cur.execute("SELECT count(*) FROM action_done_evidence")
        assert cur.fetchone()[0] == 0


def test_confirm_on_an_action_already_closed_changes_nothing(
    review_client: TestClient, db, loaded_registry: Registry
) -> None:
    action_id, row_id, _claim_id, _slug = _proposed(db, loaded_registry)
    review_client.post(f"/actions/{action_id}/dropped", follow_redirects=False)

    response = review_client.post(f"/review/{row_id}/resolve", follow_redirects=False)

    assert response.headers["location"] == "/review?notice=action_done_already_closed"
    assert _action_status(db, action_id) == "dropped"
    with db.cursor() as cur:
        cur.execute("SELECT count(*) FROM action_done_evidence")
        assert cur.fetchone()[0] == 0


def test_dismiss_records_the_reason_and_stops_that_claim_coming_back(
    review_client: TestClient, db, loaded_registry: Registry
) -> None:
    action_id, row_id, _claim_id, _slug = _proposed(db, loaded_registry)

    response = review_client.post(
        f"/review/{row_id}/dismiss", data={"reason": "that email was about a different order"}
    )

    assert response.status_code == 200
    assert _action_status(db, action_id) == "open"
    [(_id, payload, _source, status)] = _done_rows(db)
    assert status == "dismissed"
    assert payload["dismiss_reason"] == "that email was about a different order"

    judge = StubJudge()
    [outcome] = _propose(db, judge)
    assert outcome.status == "no_candidates"
    assert judge.calls == []
    assert len(_done_rows(db)) == 1

    later = _claim(
        db,
        loaded_registry,
        slug="s-done-later",
        day=date(2026, 8, 27),
        claim="Sam Okafor emailed the support team again about the Tidewater T2 delivery.",
        speaker="Sam Okafor",
    )
    [outcome] = _propose(db, judge)
    assert (outcome.status, outcome.claim_id) == ("proposed", later)


# ---------------------------------------------------------------------
# The weekly flag and the one-week CLI.
# ---------------------------------------------------------------------


def test_the_weekly_report_proposes_only_with_the_flag(db, loaded_registry: Registry) -> None:
    from tests.test_report_helpers import build_report_llm

    _open_action(db, loaded_registry)
    _follow_up(db, loaded_registry)
    judge = StubJudge()

    weekly.build_weekly(db, build_report_llm(), FakeEmbedder(), week=WEEK, as_of=AS_OF)
    assert judge.calls == []
    assert _done_rows(db) == []

    weekly.build_weekly(
        db,
        build_report_llm(),
        FakeEmbedder(),
        week=WEEK,
        as_of=AS_OF,
        propose_done=True,
        done_judge=judge,
    )
    [(_row_id, payload, _source, status)] = _done_rows(db)
    assert (status, payload["week"]) == ("open", WEEK)


def test_the_cli_flags_parse() -> None:
    parser = weekly._build_arg_parser()
    args = parser.parse_args(["propose-done", "--week", WEEK, "--action", "3", "--dry-run"])
    assert (args.week, args.action_ids, args.dry_run, args.embed) == (WEEK, [3], True, "fake")
    assert parser.parse_args(["weekly", "--week", WEEK]).propose_done is False
    assert parser.parse_args(["weekly", "--week", WEEK, "--propose-done"]).propose_done is True


def test_the_cli_runs_one_week(db, loaded_registry: Registry, capsys) -> None:
    action_id = _open_action(db, loaded_registry)
    _follow_up(db, loaded_registry)
    judge = StubJudge()
    args = argparse.Namespace(week=WEEK, embed="fake", action_ids=None, dry_run=False)

    assert weekly._run_propose_done_cli(args, judge=judge) == 0

    out = capsys.readouterr().out
    assert f"action {action_id}: proposed" in out
    assert "1 proposed, 1 judge call(s), 0 judge error(s)" in out
    assert len(_done_rows(db)) == 1
