"""Regression tests for the demo-path bug hunt, 2026-09-22
(`01-spec/reviews/bughunt-demo-path-2026-09-22.md`). Each test names the
finding it pins; each one failed against the code before its fix.

One fresh, migrated database per test (`tests/conftest.py`'s `db`
fixture) and the real app with fake model backends -- no network, no
paid calls.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import psycopg
import pytest
from fastapi.testclient import TestClient

from portolan.config import Settings
from portolan.ingest.ledger import record as record_coverage
from portolan.search.embed import FakeEmbedder
from portolan.store import LocalStore
from portolan.util import clock
from portolan.web.app import create_app
from portolan.web.deps import get_conn, get_embedder_dep, get_store_dep

ORG_KEY = "test-org-key-for-bughunt-demo-path"
SESSION_SECRET = "test-session-secret-for-bughunt-demo-path-not-a-real-secret"


@pytest.fixture()
def store_root(tmp_path: Path) -> Path:
    root = tmp_path / "store"
    root.mkdir()
    return root


@pytest.fixture()
def client(db: psycopg.Connection, db_url: str, store_root: Path):
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
    app.dependency_overrides[get_store_dep] = lambda: LocalStore(store_root)
    with TestClient(app) as c:
        r = c.post(
            "/login", data={"org_key": ORG_KEY, "editor": "Test Editor"}, follow_redirects=False
        )
        assert r.status_code == 303
        c.cookies.update(r.cookies)
        yield c


# ---------------------------------------------------------------------
# Coverage: the ledger's `day` is the SOURCE's own date, so a corpus whose
# newest source is older than the window read as "No coverage recorded
# yet" even with every source ingested.
# ---------------------------------------------------------------------


def test_coverage_shows_the_latest_recorded_days_when_nothing_is_newer_than_the_window(
    client: TestClient, db: psycopg.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    record_coverage(db, "call", date(2026, 8, 28), seen=7, extracted=6, failed=1)
    record_coverage(db, "email", date(2026, 3, 2), seen=4, extracted=4)
    monkeypatch.setattr(clock, "today", lambda: date(2026, 10, 27))

    r = client.get("/coverage")

    assert r.status_code == 200
    assert "No coverage recorded yet." not in r.text
    assert "2026-08-28" in r.text
    # Still a window, not the whole history: the March row stays out.
    assert "2026-03-02" not in r.text


def test_coverage_days_still_widens_the_window_back_from_the_newest_day(
    client: TestClient, db: psycopg.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    record_coverage(db, "call", date(2026, 8, 28), seen=7, extracted=6, failed=1)
    record_coverage(db, "email", date(2026, 3, 2), seen=4, extracted=4)
    monkeypatch.setattr(clock, "today", lambda: date(2026, 10, 27))

    r = client.get("/coverage", params={"days": 240})

    assert r.status_code == 200
    assert "2026-08-28" in r.text
    assert "2026-03-02" in r.text
    assert "last 240 days" in r.text


def test_coverage_empty_state_only_when_the_ledger_is_empty(client: TestClient) -> None:
    r = client.get("/coverage")

    assert r.status_code == 200
    assert "No coverage recorded yet." in r.text


# ---------------------------------------------------------------------
# Timestamps: `timestamptz` values come back labelled in the database
# session's zone (UTC on Railway and in CI), and the `date` filter
# formatted them as-is -- an edit made at 18:30 in Phoenix on 2026-09-22
# read "2026-09-23" (and "01:30" in the version history).
# ---------------------------------------------------------------------


def _write_plain_note(db: psycopg.Connection, slug: str) -> None:
    from datetime import UTC, datetime

    from portolan.notes.models import NoteRow, NoteStatus, NoteType
    from portolan.notes.persist import write_note

    now = datetime.now(UTC)
    note = NoteRow(
        slug=slug,
        type=NoteType.SIGNAL,
        title="Timezone fixture note",
        status=NoteStatus.LIVING,
        created=now,
        updated=now,
        synthesis="A note whose timestamps sit just past midnight UTC.",
    )
    write_note(db, note, [], changed_by="test", reason="first write")
    write_note(
        db, note, [], changed_by="Test Editor", reason="second write", embedder=FakeEmbedder()
    )


def test_note_page_shows_dates_and_times_in_the_org_zone_not_utc(
    client: TestClient, db: psycopg.Connection
) -> None:
    slug = "tz-fixture-note"
    _write_plain_note(db, slug)
    # 2026-09-23 01:30 UTC is 2026-09-22 18:30 in America/Phoenix.
    with db.cursor() as cur:
        cur.execute(
            "UPDATE notes SET created = %s, updated = %s WHERE slug = %s",
            ("2026-09-23 01:30:00+00", "2026-09-23 01:30:00+00", slug),
        )
        cur.execute(
            "UPDATE note_versions SET changed_at = %s WHERE slug = %s",
            ("2026-09-23 01:30:00+00", slug),
        )
    db.commit()

    r = client.get(f"/notes/{slug}")

    assert r.status_code == 200
    assert "updated: 2026-09-22" in r.text
    assert "created: 2026-09-22" in r.text
    assert "2026-09-22 18:30" in r.text
    assert "2026-09-23" not in r.text


def test_atlas_search_results_show_the_org_zone_date_not_utc(
    client: TestClient, db: psycopg.Connection
) -> None:
    slug = "tz-fixture-note"
    _write_plain_note(db, slug)
    with db.cursor() as cur:
        cur.execute(
            "UPDATE notes SET updated = %s WHERE slug = %s", ("2026-09-23 01:30:00+00", slug)
        )
    db.commit()

    r = client.get("/atlas", params={"q": "Timezone fixture note"})

    assert r.status_code == 200
    assert f'href="/atlas/{slug}?q=' in r.text
    assert "updated: 2026-09-22" in r.text
    assert "2026-09-23" not in r.text


# ---------------------------------------------------------------------
# Original viewer: `prepare_text` injects `[<locator>] ` at the start of
# every ticket, email, doc and RSS block, and the page prints the locator
# again in its own gutter -- "S-ticket-043#2 | [S-ticket-043#2] Sam Okafor:
# ...". Only call transcripts had the duplicate stripped.
# ---------------------------------------------------------------------


def _source_row(slug: str, source_type: str) -> dict:
    return {
        "slug": slug,
        "source_id": slug,
        "source_type": source_type,
        "s3_key": "unused",
        "source_date": date(2026, 9, 2),
        "participants": [],
        "summary": None,
    }


def test_original_viewer_ticket_blocks_do_not_repeat_the_locator() -> None:
    import json

    from portolan.web.routes.originals import _build_view

    raw = json.dumps(
        {"comments": [{"number": 2, "author": "Sam Okafor", "body": "Carrier review is booked."}]}
    ).encode()

    view = _build_view(_source_row("S-ticket-043", "ticket"), raw)

    assert view["blocks"] == [
        {"locator": "S-ticket-043#2", "text": "Sam Okafor: Carrier review is booked."}
    ]


def test_original_viewer_email_blocks_do_not_repeat_the_locator() -> None:
    from portolan.web.routes.originals import _build_view

    raw = (
        b"From: June Park <june@meadowbrook.example>\n"
        b"Message-ID: <m-1@meadowbrook.example>\n"
        b"Subject: Delays\n"
        b"\n"
        b"Please send the carrier review.\n"
        b"\n"
        b"And the delay-prevention plan.\n"
    )

    view = _build_view(_source_row("S-email-001", "email"), raw)

    assert [b["text"] for b in view["blocks"]] == [
        "Please send the carrier review.",
        "And the delay-prevention plan.",
    ]
    assert view["blocks"][0]["locator"] == "m-1@meadowbrook.example#p1"


def test_original_viewer_rss_block_does_not_repeat_the_locator() -> None:
    import json

    from portolan.web.routes.originals import _build_view

    raw = json.dumps(
        {"title": "Recall notice", "summary": "A pump recall.", "link": "https://x.example/a"}
    ).encode()

    view = _build_view(_source_row("S-rss-006", "rss"), raw)

    assert view["blocks"] == [
        {"locator": "url:https://x.example/a 2026-09-02", "text": "A pump recall."}
    ]


# ---------------------------------------------------------------------
# Weekly report: every theme line printed "N signals across M accounts"
# with no singular -- "1 signals across 1 accounts" for a new theme, and
# Ashgrove's rolled-up theme always read "1 accounts".
# ---------------------------------------------------------------------


def test_weekly_theme_line_uses_singular_for_one_signal_and_one_account() -> None:
    from portolan.report import weekly
    from portolan.report.models import (
        ActionsFeed,
        CoverageSummary,
        DiffFeed,
        QuietAccounts,
        ThemeReport,
        ThemeSummary,
        WavesFeed,
    )

    since, as_of = date(2026, 8, 31), date(2026, 9, 6)
    theme = ThemeSummary(
        slug="theme-one-signal",
        name="One-signal theme",
        signal_count=1,
        account_slugs=["larkspur-regional-health"],
        linked_arr=100.0,
        first_seen=since,
        last_seen=as_of,
        movement="new",
    )

    body = weekly._render_body(
        "2026-W36",
        [],
        DiffFeed(since=since, as_of=as_of, quiet_accounts=QuietAccounts(as_of=as_of)),
        ThemeReport(since=since, as_of=as_of, themes=[theme]),
        WavesFeed(since=since, as_of=as_of),
        [],
        [],
        ActionsFeed(),
        CoverageSummary(since=since, as_of=as_of),
        [],
    )

    assert "[[theme-one-signal]]: 1 signal across 1 account ([[larkspur-regional-health]])" in body
    assert "1 signals" not in body
    assert "1 accounts" not in body


# ---------------------------------------------------------------------
# /reports listed reports by last update, so regenerating an older week
# (a fix rerun, a hand edit) floated it above the newest one.
# ---------------------------------------------------------------------


def _write_report(db: psycopg.Connection, week: str) -> None:
    from datetime import UTC, datetime

    from portolan.notes.models import NoteRow, NoteStatus, NoteType
    from portolan.notes.persist import write_note

    now = datetime.now(UTC)
    slug = f"report-{week}"
    note = NoteRow(
        slug=slug,
        type=NoteType.REPORT,
        title=f"Weekly report — {week}",
        status=NoteStatus.LIVING,
        created=now,
        updated=now,
        synthesis="### Coverage\n- Seen 1\n",
    )
    write_note(db, note, [], changed_by="test", reason="test report")
    with db.cursor() as cur:
        cur.execute("INSERT INTO reports (week, note_slug) VALUES (%s, %s)", (week, slug))
    db.commit()


def test_reports_list_is_newest_week_first_even_after_an_older_week_is_regenerated(
    client: TestClient, db: psycopg.Connection
) -> None:
    for week in ("2026-W33", "2026-W34", "2026-W35", "2026-W36"):
        _write_report(db, week)
    # W34 regenerated last.
    with db.cursor() as cur:
        cur.execute(
            "UPDATE notes SET updated = now() + interval '1 hour' WHERE slug = %s",
            ("report-2026-W34",),
        )
    db.commit()

    r = client.get("/reports")

    assert r.status_code == 200
    positions = [r.text.index(f'href="/reports/report-2026-W3{n}"') for n in (6, 5, 4, 3)]
    assert positions == sorted(positions)


# ---------------------------------------------------------------------
# /coverage's dead-letter table printed raw exception text. The security
# review's S2 fix redacted the same text in the "Not ingested" table
# (`sources.last_error`) but not here, where a psycopg error carries the
# connection string.
# ---------------------------------------------------------------------


def test_coverage_dead_letter_errors_are_redacted(
    client: TestClient, db: psycopg.Connection
) -> None:
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO dead_letter (source_id, stage, attempt, error) VALUES (%s, %s, %s, %s)",
            (
                "S-call-900",
                "merge",
                3,
                'connection to "postgresql://portolan:hunter2-not-real@db.internal/app" failed',
            ),
        )
    db.commit()

    r = client.get("/coverage")

    assert r.status_code == 200
    assert "S-call-900" in r.text
    assert "hunter2-not-real" not in r.text
