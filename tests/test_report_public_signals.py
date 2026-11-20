"""The weekly report's "Public signals this week" section (`portolan.report.public_signals`)
and the rule that public signals stay out of themes, waves and outliers. The signals
come from a real `run_scan` over a `FakeTransport` and a `FakeLLMClient` judge."""

from __future__ import annotations

from datetime import date, timedelta

from test_report_helpers import build_report_llm
from test_signals_scan import NOW_1, PAGE_V1, _account, _judge, _scan, _targets

from portolan.report import weekly
from portolan.report.models import (
    ActionsFeed,
    CoverageSummary,
    DiffFeed,
    PublicSignal,
    QuietAccounts,
    ThemeReport,
    WavesFeed,
)
from portolan.report.outliers import list_outliers
from portolan.report.public_signals import public_signals_feed
from portolan.report.themes import _fetch_candidates
from portolan.search.embed import FakeEmbedder
from portolan.store import LocalStore
from portolan.util import clock


def _one_public_signal(db, tmp_path) -> None:
    _account(db)
    _targets(db)
    report = _scan(db, LocalStore(tmp_path), _judge([]), PAGE_V1, NOW_1, only="feeds")
    assert report.claims_written == 1


def test_feed_follows_the_report_clock(db, tmp_path) -> None:
    _one_public_signal(db, tmp_path)
    today = clock.today()

    by_record = public_signals_feed(db, since=today - timedelta(days=6), as_of=today)
    by_world = public_signals_feed(
        db, since=date(2026, 9, 14), as_of=date(2026, 9, 20), clock="world"
    )
    before = public_signals_feed(db, since=date(2026, 9, 7), as_of=date(2026, 9, 13), clock="world")

    assert [(s.account_slug, s.category, s.item_kind) for s in by_record] == [
        ("example-health", "leadership_change", "feed_item")
    ]
    assert [s.claim_id for s in by_world] == [s.claim_id for s in by_record]
    assert before == []


def test_public_signals_are_not_outliers_or_theme_candidates(db, tmp_path) -> None:
    _one_public_signal(db, tmp_path)
    with db.cursor() as cur:
        cur.execute("SELECT count(*) FROM claims WHERE fact_type = 'public_signal'")
        assert cur.fetchone()[0] == 1

    assert list_outliers(db, since=date(2026, 9, 1), as_of=date(2026, 9, 30)) == []
    assert _fetch_candidates(db, date(2026, 9, 30)) == []


def test_build_weekly_renders_the_section_per_account(db, tmp_path) -> None:
    _one_public_signal(db, tmp_path)
    today = clock.today()

    result = weekly.build_weekly(
        db,
        build_report_llm(),
        FakeEmbedder(),
        as_of=today,
        arr_by_note_slug={},
        public_signals=True,
    )

    signal = result.public_signals[0]
    assert "### Public signals this week" in result.body_md
    assert "**[[example-health]]**" in result.body_md
    assert f"^[{signal.source_slug} @ {signal.locator}]" in result.body_md
    assert "Renata Solis" in result.body_md
    assert result.themes.themes == []
    assert "example-health" in result.body_md.split("### Public signals this week")[1]

    quiet = weekly.build_weekly(
        db,
        build_report_llm(),
        FakeEmbedder(),
        as_of=today,
        arr_by_note_slug={},
        public_signals=False,
    )
    assert "Public signals" not in quiet.body_md


def _empty_feeds():
    since, as_of = date(2026, 9, 21), date(2026, 9, 27)
    return (
        DiffFeed(since=since, as_of=as_of, quiet_accounts=QuietAccounts(as_of=as_of)),
        ThemeReport(since=since, as_of=as_of),
        WavesFeed(since=since, as_of=as_of),
        ActionsFeed(),
        CoverageSummary(since=since, as_of=as_of),
    )


def test_render_body_section_only_when_asked() -> None:
    diff, themes, waves, actions, coverage = _empty_feeds()
    signal = PublicSignal(
        claim_id=1,
        account_slug="example-health",
        category="expansion",
        item_kind="page_change",
        claim_text="Expansion (public website change, Newsroom): Example Health opened a clinic.",
        source_slug="signal-web-example-health-abc-20260923T120000Z",
        locator="url:https://example.test/news 2026-09-23",
        url="https://example.test/news",
        recorded=date(2026, 9, 23),
    )

    with_section = weekly._render_body(
        "2026-W39", [], diff, themes, waves, [], [], actions, coverage, [], public_signals=[signal]
    )
    empty_section = weekly._render_body(
        "2026-W39", [], diff, themes, waves, [], [], actions, coverage, [], public_signals=[]
    )
    without = weekly._render_body(
        "2026-W39", [], diff, themes, waves, [], [], actions, coverage, []
    )

    assert (
        "- Expansion (public website change, Newsroom): Example Health opened a clinic. "
        "^[signal-web-example-health-abc-20260923T120000Z @ "
        "url:https://example.test/news 2026-09-23] (2026-09-23)"
    ) in with_section
    assert "### Public signals this week" in empty_section
    assert "- none this window" in empty_section.split("### Public signals this week")[1]
    assert "Public signals" not in without
