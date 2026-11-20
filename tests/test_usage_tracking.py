"""Pilot usage tracking (`portolan.activity`, migration 0037,
`portolan/web/routes/usage.py`): what each route records, thumbs feedback
and the eval candidate a thumbs-down leaves, the email pixel and click
redirect, retention and per-person deletion, and the `/admin/usage`
numbers against a hand-counted event set.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import psycopg
import pytest
from fastapi.testclient import TestClient

from portolan.activity import __main__ as activity_cli
from portolan.activity import delete_user, purge_expired, record_activity
from portolan.activity.dashboard import pilot_dashboard
from portolan.activity.email_links import email_click_url, email_open_url
from portolan.activity.feedback import save_feedback
from portolan.config import Settings
from portolan.connectors.file_connector import FileConnector
from portolan.ingest.intake import intake
from portolan.llm.fake import FakeLLMClient
from portolan.merge.actions import merge_action
from portolan.registry import Registry
from portolan.search.embed import FakeEmbedder
from portolan.store import LocalStore
from portolan.web import answer as answer_module
from portolan.web.app import create_app
from portolan.web.deps import get_conn, get_embedder_dep, get_llm_dep, get_store_dep
from portolan.web.routes import chat as chat_routes
from tests.merge_helpers import (
    ASHGROVE_MEDICAL,
    build_action,
    loaded_registry_fixture,  # noqa: F401
    make_source_row,
)

CORPUS_OUT = Path(__file__).resolve().parent.parent / "corpus" / "out"
ORG_KEY = "test-org-key-for-usage-tracking"
SESSION_SECRET = "test-session-secret-for-usage-tracking-not-a-real-secret"
EDITOR = "Pilot Person"
SOURCE_ID = "S-call-002"
BASE = "https://portolan.example.test"


def _settings(db_url: str, store_root: Path) -> Settings:
    return Settings(
        database_url=db_url,
        tenant="test",
        store_root=store_root,
        llm_provider="fake",
        embed_provider="fake",
        org_key=ORG_KEY,
        session_secret=SESSION_SECRET,
        cookie_secure=False,
        public_base_url=BASE,
    )


@pytest.fixture()
def store(db, tmp_path: Path) -> LocalStore:
    """One real stored original, so `/originals/{slug}` renders."""
    store = LocalStore(tmp_path / "store")
    connector = FileConnector(CORPUS_OUT)
    [ref] = [r for r in connector.list_sources() if r.source_id == SOURCE_ID]
    intake(connector.fetch(ref), conn=db, store=store)
    return store


@pytest.fixture()
def client(db: psycopg.Connection, db_url: str, tmp_path: Path, store: LocalStore):
    app = create_app(_settings(db_url, tmp_path / "store"))
    app.dependency_overrides[get_conn] = lambda: db
    app.dependency_overrides[get_embedder_dep] = lambda: FakeEmbedder()
    app.dependency_overrides[get_llm_dep] = lambda: FakeLLMClient([])
    app.dependency_overrides[get_store_dep] = lambda: store
    with TestClient(app) as c:
        login = c.post(
            "/login", data={"org_key": ORG_KEY, "editor": EDITOR}, follow_redirects=False
        )
        assert login.status_code == 303
        yield c


def _events(db, kind: str) -> list[dict]:
    with db.cursor() as cur:
        cur.execute(
            "SELECT user_name, route, target, detail FROM activity_events "
            "WHERE kind = %s ORDER BY id",
            (kind,),
        )
        return [
            {"user_name": u, "route": r, "target": t, "detail": d} for u, r, t, d in cur.fetchall()
        ]


def _sse(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.strip().split("\n\n"):
        name = next(line[6:].strip() for line in block.splitlines() if line.startswith("event:"))
        data = next(line[5:].strip() for line in block.splitlines() if line.startswith("data:"))
        events.append((name, json.loads(data)))
    return events


def _cited_answer(**_kwargs) -> answer_module.Answer:
    citation = answer_module.Citation(
        note_slug="some-note", claim_id=7, source_slug=SOURCE_ID, locator="00:01:00"
    )
    return answer_module.Answer(
        sentences=[
            answer_module.Sentence(text="They asked about the renewal.", citations=[citation])
        ],
        retrieved=[],
        control_ok=True,
        usage={"model": "claude-haiku-4-5", "input_tokens": 1000, "output_tokens": 100},
        numerals_dropped=1,
        uncited_dropped=2,
    )


def _gaps_answer(**_kwargs) -> answer_module.Answer:
    return answer_module.Answer(
        sentences=[answer_module.Sentence(text=answer_module.gaps_message(["NPS"]))],
        retrieved=[],
        control_ok=True,
        usage={},
        gaps=["NPS"],
    )


def _ask(client, monkeypatch, fake, question="What did Ashgrove say about renewal?"):
    monkeypatch.setattr(chat_routes, "answer_question", lambda *a, **k: fake(**k))
    response = client.post("/chat/new/stream", data={"question": question})
    assert response.status_code == 200
    return _sse(response.text)


# ---------------------------------------------------------------------
# Chat questions, citation clicks, page views
# ---------------------------------------------------------------------


def test_a_chat_question_records_its_outcome_numbers(client, db, monkeypatch) -> None:
    events = _ask(client, monkeypatch, _cited_answer)
    done = next(p for name, p in events if name == "done")

    [row] = _events(db, "chat_question")
    detail = row["detail"]
    assert row["user_name"] == EDITOR
    assert row["target"] == str(done["message_id"])
    assert detail["question"] == "What did Ashgrove say about renewal?"
    assert detail["mode"] == "answered"
    assert (detail["cited_sentences"], detail["citations"], detail["withheld"]) == (1, 1, 3)
    assert detail["message_id"] == done["message_id"]
    assert detail["conversation_id"] == done["conversation_id"]
    assert detail["answer_chars"] > 0
    assert detail["latency_ms"] >= 0
    assert detail["cost_usd"] > 0
    assert detail["model"] == "claude-haiku-4-5"
    assert detail["regenerate"] is False


def test_an_uncited_answer_and_a_failed_answer_are_not_answered(client, db, monkeypatch) -> None:
    _ask(client, monkeypatch, _gaps_answer, question="What is our NPS?")

    def boom(**_kwargs):
        raise RuntimeError("model down")

    _ask(client, monkeypatch, boom, question="Anything at all?")

    modes = [(e["detail"]["question"], e["detail"]["mode"]) for e in _events(db, "chat_question")]
    assert modes == [("What is our NPS?", "gaps"), ("Anything at all?", "error")]


def test_the_original_chip_is_marked_and_a_click_is_recorded(client, db, monkeypatch) -> None:
    events = _ask(client, monkeypatch, _cited_answer)
    done = next(p for name, p in events if name == "done")
    sentence_html = next(p["html"] for name, p in events if name == "sentence")
    message_id = done["message_id"]

    assert (
        f"/originals/{SOURCE_ID}?at=00%3A01%3A00&amp;from=chat&amp;m={message_id}" in sentence_html
    )

    reloaded = client.get(f"/c/{done['conversation_id']}")
    assert f"from=chat&amp;m={message_id}" in reloaded.text

    opened = client.get(f"/originals/{SOURCE_ID}?at=00%3A01%3A00&from=chat&m={message_id}")
    assert opened.status_code == 200
    client.get(f"/originals/{SOURCE_ID}")  # opened directly: not a citation click

    clicks = _events(db, "citation_click")
    assert [(c["user_name"], c["target"], c["detail"]) for c in clicks] == [
        (EDITOR, SOURCE_ID, {"message_id": message_id})
    ]


def test_page_views_record_the_route_template_not_the_query(client, db) -> None:
    client.get("/actions")
    client.get(f"/originals/{SOURCE_ID}?at=secret-looking-query")
    client.get("/atlas/panel", headers={"HX-Request": "true"})

    views = [(v["route"], v["target"]) for v in _events(db, "page_view")]
    assert views == [("/actions", None), ("/originals/{slug}", SOURCE_ID)]


def test_a_signed_out_request_records_nothing(db, db_url, tmp_path, store) -> None:
    app = create_app(_settings(db_url, tmp_path / "store"))
    app.dependency_overrides[get_conn] = lambda: db
    with TestClient(app) as anonymous:
        assert anonymous.get("/actions", follow_redirects=False).status_code == 302
    assert _events(db, "page_view") == []


# ---------------------------------------------------------------------
# Thumbs feedback and eval candidates
# ---------------------------------------------------------------------


def _candidates(db) -> list[tuple]:
    with db.cursor() as cur:
        cur.execute(
            "SELECT question, answer_md, reason, status, user_name FROM eval_candidates ORDER BY id"
        )
        return cur.fetchall()


def test_thumbs_down_saves_an_eval_candidate_and_thumbs_up_removes_it(
    client, db, monkeypatch
) -> None:
    events = _ask(client, monkeypatch, _cited_answer, question="When does Ashgrove renew?")
    done = next(p for name, p in events if name == "done")
    message_id = done["message_id"]
    assert f'action="/messages/{message_id}/feedback"' in done["feedback_html"]

    down = client.post(
        f"/messages/{message_id}/feedback",
        data={"rating": "down", "reason": "Missed the date.\nIt is in June."},
        follow_redirects=False,
    )
    assert down.status_code == 303
    assert down.headers["location"] == f"/c/{done['conversation_id']}"
    assert _candidates(db) == [
        (
            "When does Ashgrove renew?",
            "They asked about the renewal.",
            "Missed the date. It is in June.",
            "candidate",
            EDITOR,
        )
    ]
    page = client.get(f"/c/{done['conversation_id']}")
    assert 'value="down" class="btn chip quiet"\n          aria-pressed="true"' in page.text
    assert "you marked this not helpful" in page.text

    client.post(f"/messages/{message_id}/feedback", data={"rating": "up"})
    assert _candidates(db) == []
    with db.cursor() as cur:
        cur.execute("SELECT user_name, rating, reason FROM answer_feedback")
        assert cur.fetchall() == [(EDITOR, "up", None)]
    assert [e["detail"]["rating"] for e in _events(db, "feedback")] == ["down", "up"]


def test_feedback_is_refused_on_someone_elses_answer(client, db, monkeypatch) -> None:
    done = next(p for n, p in _ask(client, monkeypatch, _cited_answer) if n == "done")

    assert (
        save_feedback(
            db, message_id=done["message_id"], user_name="Someone Else", rating="down", reason=None
        )
        is None
    )
    assert client.post("/messages/999999/feedback", data={"rating": "down"}).status_code == 404
    assert (
        client.post(f"/messages/{done['message_id']}/feedback", data={"rating": "meh"}).status_code
        == 400
    )
    assert _candidates(db) == []


# ---------------------------------------------------------------------
# Review decisions and action confirms
# ---------------------------------------------------------------------


def _review_row(db) -> int:
    with db.cursor() as cur:
        cur.execute("INSERT INTO review_queue (kind, payload) VALUES ('other', '{}') RETURNING id")
        return cur.fetchone()[0]


def test_review_decisions_are_recorded(client, db) -> None:
    kept, dropped = _review_row(db), _review_row(db)

    client.post(f"/review/{kept}/resolve", follow_redirects=False)
    client.post(f"/review/{dropped}/dismiss", follow_redirects=False)
    client.post(f"/review/{dropped}/dismiss", follow_redirects=False)  # already handled

    decisions = [
        (e["target"], e["detail"]["decision"], e["detail"]["row_kind"])
        for e in _events(db, "review_decision")
    ]
    assert decisions == [(str(kept), "resolve", "other"), (str(dropped), "dismiss", "other")]


def test_marking_an_action_done_is_an_action_confirm(client, db, loaded_registry: Registry) -> None:
    source = make_source_row(db, "s-usage-call")
    action = build_action("Send the renewal quote", owner_mention="Sam Okafor", locator="00:10:00")
    action_id = merge_action(
        db, loaded_registry, action, source, entity_id=ASHGROVE_MEDICAL
    ).action_id

    client.post(f"/actions/{action_id}/done", follow_redirects=False)
    client.post(f"/actions/{action_id}/done", follow_redirects=False)  # already closed

    confirms = [(e["target"], e["detail"]["how"]) for e in _events(db, "action_confirm")]
    assert confirms == [(str(action_id), "marked_done")]


# ---------------------------------------------------------------------
# Email pixel and click redirect
# ---------------------------------------------------------------------


def test_the_email_pixel_records_an_open_only_for_a_valid_token(client, db, db_url, tmp_path):
    settings = _settings(db_url, tmp_path / "store")
    url = email_open_url(settings, email_id="monday-2026-09-28", recipient=EDITOR)
    assert url.startswith(f"{BASE}/t/o/") and url.endswith(".gif")
    path = url.removeprefix(BASE)

    good = client.get(path)
    tampered = client.get(path.replace(".gif", "x.gif"))

    assert good.status_code == tampered.status_code == 200
    assert good.headers["content-type"] == "image/gif"
    assert good.content.startswith(b"GIF89a")
    assert [(e["user_name"], e["target"]) for e in _events(db, "email_open")] == [
        (EDITOR, "monday-2026-09-28")
    ]


def test_the_click_redirect_goes_only_where_it_was_signed_to(client, db, db_url, tmp_path):
    settings = _settings(db_url, tmp_path / "store")
    url = email_click_url(
        settings, email_id="monday-2026-09-28", recipient=EDITOR, target="/reports/weekly-2026-w39"
    )
    open_url = email_open_url(settings, email_id="monday-2026-09-28", recipient=EDITOR)
    token = url.removeprefix(f"{BASE}/t/c/")

    good = client.get(url.removeprefix(BASE), follow_redirects=False)
    forged = client.get(f"/t/c/{token[:-2]}xx", follow_redirects=False)
    wrong_kind = client.get(
        "/t/c/" + open_url.removeprefix(f"{BASE}/t/o/").removesuffix(".gif"),
        follow_redirects=False,
    )

    assert (good.status_code, good.headers["location"]) == (302, "/reports/weekly-2026-w39")
    assert forged.headers["location"] == wrong_kind.headers["location"] == "/"
    [click] = _events(db, "email_click")
    assert click["detail"] == {"email_id": "monday-2026-09-28", "url": "/reports/weekly-2026-w39"}

    for bad in ("javascript:alert(1)", "//evil.example/x", "ftp://x.example/y"):
        with pytest.raises(ValueError):
            email_click_url(settings, email_id="e", recipient=EDITOR, target=bad)


# ---------------------------------------------------------------------
# Retention and per-person deletion
# ---------------------------------------------------------------------


def _feedback_row(db, user: str, rating: str, when: datetime) -> int:
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO answer_feedback (user_name, rating, created_at, updated_at) "
            "VALUES (%s, %s, %s, %s) RETURNING id",
            (user, rating, when, when),
        )
        return cur.fetchone()[0]


def test_purge_deletes_only_rows_past_retention(db) -> None:
    now = datetime(2026, 9, 25, 12, tzinfo=UTC)
    record_activity(db, kind="page_view", user_name="A", occurred_at=now - timedelta(days=200))
    record_activity(db, kind="page_view", user_name="A", occurred_at=now - timedelta(days=10))
    old_feedback = _feedback_row(db, "A", "down", now - timedelta(days=200))
    _feedback_row(db, "A", "up", now - timedelta(days=10))
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO eval_candidates (feedback_id, user_name, question, answer_md) "
            "VALUES (%s, 'A', 'q', 'a')",
            (old_feedback,),
        )

    assert purge_expired(db, retention_days=0, now=now) == {
        "activity_events": 0,
        "answer_feedback": 0,
    }
    assert purge_expired(db, retention_days=180, now=now) == {
        "activity_events": 1,
        "answer_feedback": 1,
    }
    with db.cursor() as cur:
        cur.execute("SELECT count(*) FROM activity_events")
        assert cur.fetchone()[0] == 1
        cur.execute("SELECT count(*), count(feedback_id) FROM eval_candidates")
        assert cur.fetchone() == (1, 0)


def test_delete_user_removes_only_that_persons_rows(db, db_url, monkeypatch, capsys) -> None:
    for user in ("Alice", "Bob"):
        record_activity(db, kind="page_view", user_name=user)
        fid = _feedback_row(db, user, "down", datetime.now(UTC))
        with db.cursor() as cur:
            cur.execute(
                "INSERT INTO eval_candidates (feedback_id, user_name, question, answer_md) "
                "VALUES (%s, %s, 'q', 'a')",
                (fid, user),
            )

    assert delete_user(db, "Alice") == {
        "activity_events": 1,
        "answer_feedback": 1,
        "eval_candidates": 1,
    }

    monkeypatch.setenv("DATABASE_URL", db_url)
    assert activity_cli.main(["delete-user", "Bob"]) == 0
    assert json.loads(capsys.readouterr().out)["activity_events"] == 1
    with db.cursor() as cur:
        cur.execute(
            "SELECT (SELECT count(*) FROM activity_events), (SELECT count(*) FROM answer_feedback),"
            " (SELECT count(*) FROM eval_candidates)"
        )
        assert cur.fetchone() == (0, 0, 0)


# ---------------------------------------------------------------------
# The dashboard numbers
# ---------------------------------------------------------------------

NOW = datetime(2026, 9, 25, 12, tzinfo=UTC)  # a Friday


def _at(day: int, month: int = 9) -> datetime:
    return datetime(2026, month, day, 10, tzinfo=UTC)


def _question(db, user, day, question, mode, citations=0, message_id=None):
    record_activity(
        db,
        kind="chat_question",
        user_name=user,
        occurred_at=_at(day),
        detail={
            "question": question,
            "mode": mode,
            "citations": citations,
            "message_id": message_id,
        },
    )


def _seed_dashboard(db) -> None:
    record_activity(db, kind="page_view", user_name="Alice", occurred_at=_at(21))
    _question(db, "Alice", 22, "What did Ashgrove say?", "answered", 2, 11)
    record_activity(
        db,
        kind="citation_click",
        user_name="Alice",
        occurred_at=_at(22),
        detail={"message_id": 11},
    )
    _question(db, "Bob", 23, "What is our NPS?", "nothing_recorded")
    _question(db, "Alice", 23, "what is  our NPS? ", "gaps")
    _question(db, "Bob", 24, "Who owns Northlake?", "answered", 1, 12)
    record_activity(db, kind="email_open", user_name="Carol", occurred_at=_at(21))
    record_activity(db, kind="email_open", user_name="Carol", occurred_at=_at(22))
    record_activity(db, kind="email_click", user_name="Carol", occurred_at=_at(22))
    record_activity(db, kind="page_view", user_name="Dave", occurred_at=_at(15))
    record_activity(db, kind="page_view", user_name="Old", occurred_at=_at(1, month=6))
    _feedback_row(db, "Alice", "up", _at(22))
    _feedback_row(db, "Bob", "down", _at(24))


def test_the_dashboard_counts_match_a_hand_count(db) -> None:
    _seed_dashboard(db)

    dash = pilot_dashboard(db, now=NOW, weeks=3)

    this_week, last_week, before = dash.weeks
    assert this_week["week_start"].isoformat() == "2026-09-21"
    assert (this_week["active_users"], this_week["questions"]) == (2, 4)
    assert this_week["questions_per_user"] == 2.0
    assert (this_week["cited_answers"], this_week["clicked_answers"]) == (2, 1)
    assert this_week["click_through"] == 0.5
    assert (this_week["thumbs_up"], this_week["thumbs_down"]) == (1, 1)
    assert this_week["thumbs_up_share"] == 0.5
    assert (this_week["email_opens"], this_week["email_openers"], this_week["email_clicks"]) == (
        2,
        1,
        1,
    )
    assert (last_week["active_users"], last_week["questions"]) == (1, 0)
    assert last_week["click_through"] is None
    assert before["active_users"] == 0

    assert dash.totals["active_users"] == 3  # Alice, Bob, Dave; Carol only opened email
    assert dash.totals["questions"] == 4
    assert [(u["user_name"], u["questions"]) for u in dash.per_user] == [
        ("Bob", 2),
        ("Alice", 2),
        ("Dave", 0),
    ]
    alice = next(u for u in dash.per_user if u["user_name"] == "Alice")
    assert (alice["page_views"], alice["citation_clicks"], alice["thumbs_up"]) == (1, 1, 1)

    [nps] = dash.unanswered
    assert nps["times_asked"] == 2 and nps["people"] == 2
    assert sorted(nps["modes"]) == ["gaps", "nothing_recorded"]


def test_admin_usage_needs_a_login_and_shows_the_tables(client, db, db_url, tmp_path) -> None:
    record_activity(
        db,
        kind="chat_question",
        user_name="Bob",
        detail={"question": "What is our churn risk?", "mode": "nothing_recorded"},
    )
    record_activity(db, kind="email_open", user_name="Carol")

    page = client.get("/admin/usage")

    assert page.status_code == 200
    for table in ("usage-totals", "usage-weeks", "usage-people", "usage-unanswered"):
        assert f'id="{table}"' in page.text
    assert "What is our churn risk?" in page.text

    app = create_app(_settings(db_url, tmp_path / "store"))
    app.dependency_overrides[get_conn] = lambda: db
    with TestClient(app) as anonymous:
        assert anonymous.get("/admin/usage", follow_redirects=False).status_code == 302


@pytest.fixture(autouse=True)
def _single_chat_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    # These tests script one model call per answer; agent mode is covered in
    # tests/test_chat_agent.py.
    monkeypatch.setenv("PORTOLAN_CHAT_MODE", "single")
