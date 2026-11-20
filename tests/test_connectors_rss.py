"""Tests for `portolan.connectors.rss.RSSConnector`.

Fixtures under `tests/fixtures/connectors/rss/` are a hand-written RSS
2.0 feed (a LinkedIn page via RSS.app) and a hand-written Atom feed (an
industry newswire), sharing one deliberately duplicated link so the
dedup-by-link behaviour has something real to catch. Names are from
`corpus/truth-table.yaml` (Larkspur Regional Health) and its own
`corpus/out/rss/*.json` fixtures (the `s-rss-*` link scheme, reused here
verbatim)."""

from __future__ import annotations

from datetime import date

from connector_helpers import FakeTransport, load_text_fixture, text_response

from portolan.connectors.rss import FeedConfig, RSSConfig, RSSConnector
from portolan.extract.schema import SourceType

LINKEDIN_URL = "https://feeds.rss.app/mock/larkspur-linkedin.xml"
NEWS_URL = "https://feeds.example-newswire.test/mock/industry-news.xml"


def _connector() -> tuple[RSSConnector, FakeTransport]:
    transport = FakeTransport()
    transport.queue(
        LINKEDIN_URL, text_response(load_text_fixture("rss", "linkedin_larkspur_rss2.xml"))
    )
    transport.queue(NEWS_URL, text_response(load_text_fixture("rss", "industry_news_atom.xml")))
    config = RSSConfig(
        feeds=[
            FeedConfig(name="Larkspur Regional Health LinkedIn", url=LINKEDIN_URL, kind="linkedin"),
            FeedConfig(name="Industry News", url=NEWS_URL, kind="news"),
        ]
    )
    return RSSConnector(config, transport=transport), transport


def test_parses_both_rss_and_atom_feeds():
    connector, _ = _connector()

    refs = list(connector.list_sources(since=None))

    links = {r.ref["item"].link for r in refs}
    assert "https://feeds.example-newswire.test/portolan-corpus/s-rss-n002" in links  # RSS 2.0
    assert "https://feeds.example-newswire.test/portolan-corpus/s-rss-201" in links  # Atom
    assert all(r.source_type == SourceType.RSS for r in refs)


def test_dedups_by_link_across_feeds():
    connector, _ = _connector()

    refs = list(connector.list_sources(since=None))

    # s-rss-006 appears in both the RSS 2.0 feed and the Atom feed with the
    # exact same link; only the first occurrence (LinkedIn feed, listed
    # first) should survive.
    matching = [r for r in refs if r.ref["item"].link.endswith("s-rss-006")]
    assert len(matching) == 1
    assert matching[0].ref["feed"].kind == "linkedin"
    assert len(refs) == 3  # s-rss-006, s-rss-n002, s-rss-201 — not 4


def test_since_filters_on_published_date():
    connector, _ = _connector()

    refs = list(connector.list_sources(since=date(2026, 8, 1)))

    links = {r.ref["item"].link for r in refs}
    assert links == {
        "https://feeds.example-newswire.test/portolan-corpus/s-rss-006",
        "https://feeds.example-newswire.test/portolan-corpus/s-rss-201",
    }


def test_fetch_yields_rss_source_with_link_and_feed_metadata():
    connector, _ = _connector()
    refs = list(connector.list_sources(since=None))
    ref = next(r for r in refs if r.ref["item"].link.endswith("s-rss-n002"))

    raw = connector.fetch(ref)

    assert raw.source_type == SourceType.RSS
    assert raw.source_date == date(2026, 3, 2)
    assert raw.metadata["feed_kind"] == "linkedin"
    assert raw.metadata["link"] == "https://feeds.example-newswire.test/portolan-corpus/s-rss-n002"
    assert "nurse appreciation event" in raw.text
