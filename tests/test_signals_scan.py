"""`portolan.signals.scan.run_scan` end to end on a test database: feeds by GUID,
page snapshots and diffs, the judge (a `FakeLLMClient`), and what gets written.
Network is a `FakeTransport`; the store is a temp directory."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import psycopg
import pytest
from connector_helpers import FakeTransport, text_response

from portolan.config import Settings
from portolan.llm.fake import FakeLLMClient
from portolan.notes import read_note
from portolan.signals.fetch import PoliteFetcher
from portolan.signals.scan import SignalsDisabled, run_scan
from portolan.signals.targets import import_targets
from portolan.store import LocalStore
from portolan.web.routes.originals import _build_view

CRM_ID = "PUB-001"
FEED_URL = "https://rss.app/feeds/test-example-health.xml"
PAGE_URL = "https://www.example-health.test/about/leadership"
CFO_LINK = "https://www.linkedin.com/feed/update/urn:li:activity:111"
NOW_1 = datetime(2026, 9, 23, 12, 0, tzinfo=UTC)
# Labelled in the deployment's zone, as `clock.now()` is; 12:00 UTC.
NOW_2 = datetime(2026, 9, 30, 5, 0, tzinfo=ZoneInfo("America/Phoenix"))

FEED = f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel>
  <title>Example Health | LinkedIn</title>
  <link>https://www.linkedin.com/company/example-health/posts</link>
  <description>RSS.app feed</description>
  <item>
    <title>Example Health names Renata Solis chief financial officer</title>
    <link>{CFO_LINK}</link>
    <guid isPermaLink="false">urn:li:activity:111</guid>
    <pubDate>Sun, 20 Sep 2026 18:00:00 GMT</pubDate>
    <description>&lt;p&gt;Example Health announced Renata Solis as its new Chief Financial
Officer.&lt;/p&gt;</description>
  </item>
  <item>
    <title>Thank you to our nurses</title>
    <link>https://www.linkedin.com/feed/update/urn:li:activity:222</link>
    <guid isPermaLink="false">urn:li:activity:222</guid>
    <pubDate>Mon, 21 Sep 2026 18:00:00 GMT</pubDate>
    <description>A nurse appreciation lunch at the main campus.</description>
  </item>
  <item>
    <title>Spring gala recap</title>
    <link>https://www.linkedin.com/feed/update/urn:li:activity:000</link>
    <guid isPermaLink="false">urn:li:activity:000</guid>
    <pubDate>Mon, 01 Jun 2026 18:00:00 GMT</pubDate>
    <description>Old post, outside the window.</description>
  </item>
</channel></rss>"""

PAGE_V1 = """<html><head><title>Leadership | Example Health</title></head><body><main>
<h1>Leadership</h1><p>Our executive team, updated September 1, 2026.</p>
<h2>Executive team</h2><ul><li>Jordan Reyes, President and CEO</li>
<li>Sam Patel, Chief Operating Officer</li></ul></main></body></html>"""
PAGE_V2 = PAGE_V1.replace(
    "<li>Sam Patel, Chief Operating Officer</li>",
    "<li>Sam Patel, Chief Operating Officer</li><li>Alex Kim, Chief Nursing Officer</li>",
).replace("September 1, 2026", "September 29, 2026")
# Two events in one page change: a new leader and a new clinic.
PAGE_V2_TWO_EVENTS = PAGE_V2.replace(
    "</ul></main>", "</ul><h2>News</h2><p>Example Health opened its Mesa clinic.</p></main>"
)


def _settings(enabled: bool = True) -> Settings:
    return Settings(_env_file=None, external_signals_enabled=enabled)


CFO_FINDING = {
    "category": "leadership_change",
    "claim": "Example Health named Renata Solis its chief financial officer.",
    "quote": "Example Health announced Renata Solis as its new Chief Financial",
    "confidence": 0.9,
}


def _judge(calls: list[str]):
    def respond(*, system, user, json_schema, model, surface):
        calls.append(user)
        if "Renata Solis" in user:
            return {"reason": "new CFO", "relevant": True, "findings": [CFO_FINDING]}
        if "Alex Kim" in user:
            findings = [
                {
                    "category": "leadership_change",
                    "claim": "Example Health's leadership page now lists Alex Kim as chief "
                    "nursing officer.",
                    "quote": "Alex Kim, Chief Nursing Officer",
                    "confidence": 0.8,
                }
            ]
            if "Mesa" in user:
                findings.append(
                    {
                        "category": "expansion",
                        "claim": "Example Health opened a clinic in Mesa.",
                        "quote": "Example Health opened its Mesa clinic",
                        "confidence": 0.7,
                    }
                )
            return {"reason": "leadership page changed", "relevant": True, "findings": findings}
        return {"reason": "routine post", "relevant": False, "findings": []}

    return FakeLLMClient(respond)


def _account(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO registry_entities (kind, canonical_name, crm_id) "
            "VALUES ('account', 'Example Health', %s)",
            (CRM_ID,),
        )
    conn.commit()


def _targets(conn: psycopg.Connection, extra_feeds: list[dict] | None = None) -> None:
    import_targets(
        conn,
        {
            "accounts": [
                {
                    "crm_id": CRM_ID,
                    "feeds": [
                        {
                            "url": FEED_URL,
                            "kind": "linkedin",
                            "linkedin_company_url": "https://www.linkedin.com/company/example-health/",
                            "label": "Example Health on LinkedIn",
                        },
                        *(extra_feeds or []),
                    ],
                    "pages": [{"url": PAGE_URL, "label": "Leadership page"}],
                }
            ]
        },
    )


def _transport(
    page_html: str, *, page_robots: str | None = None, feed: str = FEED
) -> FakeTransport:
    transport = FakeTransport()
    transport.queue("https://rss.app/robots.txt", text_response("User-agent: *\nAllow: /\n"))
    transport.queue(FEED_URL, text_response(feed))
    transport.queue(
        "https://www.example-health.test/robots.txt",
        text_response(page_robots) if page_robots else text_response("", status_code=404),
    )
    transport.queue(PAGE_URL, text_response(page_html))
    return transport


def _scan(conn, store, llm, page_html: str, now: datetime, **kwargs):
    fetcher = PoliteFetcher(
        _transport(page_html), user_agent="PortolanSignals/0.1 (test)", sleep=lambda s: None
    )
    return run_scan(
        conn,
        settings=_settings(),
        store=store,
        llm=llm,
        fetcher=fetcher,
        now=now,
        **kwargs,
    )


def _source_view(conn, store, slug: str) -> dict:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT slug, source_id, source_type, s3_key, source_date, participants, summary "
            "FROM sources WHERE slug = %s",
            (slug,),
        )
        row = cur.fetchone()
    source_row = dict(
        zip(
            [
                "slug",
                "source_id",
                "source_type",
                "s3_key",
                "source_date",
                "participants",
                "summary",
            ],
            row,
            strict=True,
        )
    )
    return _build_view(source_row, store.get(source_row["s3_key"]))


def test_scan_refuses_to_run_when_the_flag_is_off(db, tmp_path) -> None:
    fetcher = PoliteFetcher(FakeTransport(), user_agent="t", sleep=lambda s: None)
    with pytest.raises(SignalsDisabled):
        run_scan(
            db,
            settings=_settings(enabled=False),
            store=LocalStore(tmp_path),
            llm=FakeLLMClient([]),
            fetcher=fetcher,
        )


def test_two_scans_write_a_feed_signal_then_a_page_change_signal(db, tmp_path) -> None:
    _account(db)
    _targets(db)
    store = LocalStore(tmp_path / "store")
    calls: list[str] = []
    llm = _judge(calls)

    first = _scan(db, store, llm, PAGE_V1, NOW_1)

    assert (first.feed_items_new, first.feed_items_too_old) == (2, 1)
    assert (first.page_baselines, first.page_changes) == (1, 0)
    assert (first.judged, first.relevant, first.not_relevant) == (2, 1, 1)
    with db.cursor() as cur:
        cur.execute(
            "SELECT c.id, c.note_slug, c.fact_type, c.claim_text, c.source_slug, c.locator, "
            "c.speaker FROM claims c ORDER BY c.id"
        )
        claims = cur.fetchall()
    assert len(claims) == 1
    _cid, note_slug, fact_type, claim_text, source_slug, locator, speaker = claims[0]
    assert note_slug == "signal-public_signal-example-health"
    assert fact_type == "public_signal"
    assert claim_text == (
        "Leadership change (public LinkedIn company post, Example Health on LinkedIn): "
        "Example Health named Renata Solis its chief financial officer."
    )
    assert speaker == "Example Health on LinkedIn"
    assert locator == f"url:{CFO_LINK} 2026-09-20"

    # "Open the original" lands on the cited block of the stored item.
    view = _source_view(db, store, source_slug)
    assert view["kind"] == "rss"
    assert view["item"]["link"] == CFO_LINK
    assert [b["locator"] for b in view["blocks"]] == [locator]
    assert "Renata Solis" in view["blocks"][0]["text"]

    account, _ = read_note(db, "example-health")
    assert "signal-public_signal-example-health" in account.related

    with db.cursor() as cur:
        cur.execute(
            "SELECT seen, extracted, filtered FROM coverage_ledger WHERE connector = 'signals_feed'"
        )
        assert cur.fetchone() == (2, 1, {"not_relevant": 1})

    # Second scan a week later: the feed is unchanged, the page gained a leader.
    second = _scan(db, store, llm, PAGE_V2, NOW_2)

    assert second.feed_items_new == 0
    assert (second.page_changes, second.judged, second.relevant) == (1, 1, 1)
    assert len(calls) == 3
    assert "Added: Alex Kim, Chief Nursing Officer" in calls[-1]
    assert "September 29" not in calls[-1]  # the date-only edit is noise, not a change
    assert (
        "Compared the scan of 2026-09-30 12:00 UTC with the scan of 2026-09-23 12:00 UTC"
        in (calls[-1])
    )

    _note, note_claims = read_note(db, "signal-public_signal-example-health")
    assert [c.fact_type for c in note_claims] == ["public_signal", "public_signal"]
    page_claim = note_claims[-1]
    assert page_claim.claim_text.startswith("Leadership change (public website change,")
    assert page_claim.locator == f"url:{PAGE_URL} 2026-09-30"
    view = _source_view(db, store, page_claim.source_slug)
    assert [b["locator"] for b in view["blocks"]] == [page_claim.locator]
    record = json.loads(store.get(f"originals/rss/{page_claim.source_slug}"))
    assert record["signal_kind"] == "page_change"
    assert record["previous_snapshot"]["html_key"] != record["snapshot"]["html_key"]
    assert store.get(record["previous_snapshot"]["html_key"]).decode() == PAGE_V1
    assert store.get(record["snapshot"]["html_key"]).decode() == PAGE_V2
    assert record["snapshot"]["html_key"].endswith("/20260930T120000Z.html")
    assert page_claim.source_slug.endswith("-20260930T120000Z")

    # A third scan with nothing new asks the judge nothing and writes nothing.
    third = _scan(db, store, llm, PAGE_V2, NOW_2)
    assert (third.feed_items_new, third.page_changes, third.judged) == (0, 0, 0)
    assert len(calls) == 3


def test_the_same_event_from_two_items_is_one_claim_with_two_citations(db, tmp_path) -> None:
    _account(db)
    _targets(db)
    translated = FEED.replace(
        "<title>Thank you to our nurses</title>",
        "<title>Example Health nombra a Renata Solis directora financiera</title>",
    ).replace(
        "A nurse appreciation lunch at the main campus.",
        "Example Health anuncia a Renata Solis como su nueva directora financiera.",
    )
    transport = _transport(PAGE_V1, feed=translated)
    calls: list[str] = []

    def respond(**kwargs):
        calls.append(kwargs["user"])
        quote = (
            "Example Health anuncia a Renata Solis"
            if "anuncia" in kwargs["user"]
            else CFO_FINDING["quote"]
        )
        return {
            "reason": "new CFO",
            "relevant": True,
            "findings": [{**CFO_FINDING, "quote": quote}],
        }

    fetcher = PoliteFetcher(transport, user_agent="PortolanSignals/0.1", sleep=lambda s: None)
    report = run_scan(
        db,
        settings=_settings(),
        store=LocalStore(tmp_path),
        llm=FakeLLMClient(respond),
        fetcher=fetcher,
        now=NOW_1,
        only="feeds",
    )

    assert (report.relevant, len(calls)) == (2, 2)
    with db.cursor() as cur:
        cur.execute("SELECT id, source_slug FROM claims")
        claims = cur.fetchall()
        cur.execute("SELECT claim_id, source_slug FROM claim_citations")
        citations = cur.fetchall()
        cur.execute("SELECT item_id, claim_id FROM signal_item_claims ORDER BY item_id")
        item_claims = cur.fetchall()
    assert len(claims) == 1
    assert len(citations) == 1
    assert citations[0][0] == claims[0][0] and citations[0][1] != claims[0][1]
    assert [claim_id for _item, claim_id in item_claims] == [claims[0][0], claims[0][0]]
    assert report.claims_written == 1


def test_one_page_change_with_two_events_writes_two_claims(db, tmp_path) -> None:
    _account(db)
    _targets(db)
    store = LocalStore(tmp_path)
    calls: list[str] = []
    _scan(db, store, _judge(calls), PAGE_V1, NOW_1, only="pages")

    report = _scan(db, store, _judge(calls), PAGE_V2_TWO_EVENTS, NOW_2, only="pages")

    assert (report.page_changes, report.relevant, report.claims_written) == (1, 1, 2)
    with db.cursor() as cur:
        cur.execute(
            "SELECT sic.category, c.source_slug FROM signal_item_claims sic "
            "JOIN claims c ON c.id = sic.claim_id ORDER BY c.id"
        )
        rows = cur.fetchall()
    assert [category for category, _ in rows] == ["leadership_change", "expansion"]
    assert rows[0][1] == rows[1][1]  # both claims cite the one change record


def test_a_feed_mirroring_a_personal_profile_is_refused(db, tmp_path) -> None:
    _account(db)
    personal = "https://rss.app/feeds/someone.xml"
    _targets(db, extra_feeds=[{"url": personal, "kind": "other"}])
    transport = _transport(PAGE_V1)
    transport.queue(
        personal,
        text_response(
            FEED.replace(
                "https://www.linkedin.com/company/example-health/posts",
                "https://www.linkedin.com/in/jane-doe-123/recent-activity",
            )
        ),
    )
    fetcher = PoliteFetcher(transport, user_agent="PortolanSignals/0.1", sleep=lambda s: None)
    report = run_scan(
        db,
        settings=_settings(),
        store=LocalStore(tmp_path),
        llm=_judge([]),
        fetcher=fetcher,
        now=NOW_1,
        skip_judge=True,
    )

    assert report.personal_profile_refused == 1
    with db.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM signal_items si JOIN signal_targets st ON st.id = si.target_id "
            "WHERE st.url = %s",
            (personal,),
        )
        assert cur.fetchone()[0] == 0
        cur.execute("SELECT last_error FROM signal_targets WHERE url = %s", (personal,))
        assert cur.fetchone()[0].startswith("PersonalProfileRefused")


def test_a_robots_disallowed_page_is_never_fetched(db, tmp_path) -> None:
    _account(db)
    _targets(db)
    transport = _transport(PAGE_V1, page_robots="User-agent: *\nDisallow: /about/\n")
    fetcher = PoliteFetcher(transport, user_agent="PortolanSignals/0.1", sleep=lambda s: None)

    report = run_scan(
        db,
        settings=_settings(),
        store=LocalStore(tmp_path),
        llm=_judge([]),
        fetcher=fetcher,
        now=NOW_1,
        only="pages",
    )

    assert report.robots_disallowed == 1
    assert PAGE_URL not in [c["url"] for c in transport.calls]
    with db.cursor() as cur:
        cur.execute("SELECT count(*) FROM page_snapshots")
        assert cur.fetchone()[0] == 0


def test_a_failed_judge_call_is_retried_on_the_next_scan(db, tmp_path) -> None:
    _account(db)
    _targets(db)
    store = LocalStore(tmp_path)

    def boom(**_kwargs):
        raise RuntimeError("provider down")

    first = _scan(db, store, FakeLLMClient(boom), PAGE_V1, NOW_1, only="feeds")
    assert (first.judged, first.judge_errors) == (2, 2)

    calls: list[str] = []
    second = _scan(db, store, _judge(calls), PAGE_V1, NOW_1, only="feeds")
    assert (second.judged, second.relevant) == (2, 1)
    with db.cursor() as cur:
        cur.execute("SELECT status, attempts FROM signal_items ORDER BY id")
        assert cur.fetchall() == [("relevant", 2), ("not_relevant", 2)]


def test_the_budget_stops_judging(db, tmp_path) -> None:
    _account(db)
    _targets(db)
    calls: list[str] = []

    report = _scan(db, LocalStore(tmp_path), _judge(calls), PAGE_V1, NOW_1, budget_usd=0.0)

    assert report.budget_hit is True
    assert (report.judged, len(calls)) == (0, 0)


def test_the_database_refuses_a_linkedin_feed_that_is_not_a_company_page(db) -> None:
    with pytest.raises(psycopg.errors.CheckViolation), db.cursor() as cur:
        cur.execute(
            "INSERT INTO signal_targets (account_crm_id, kind, url, feed_kind, source_page_url) "
            "VALUES ('X', 'feed', 'https://rss.app/feeds/x.xml', 'linkedin', "
            "'https://www.linkedin.com/in/jane-doe/')"
        )
    db.rollback()
