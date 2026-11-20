"""Per-account targets (`portolan.signals.targets`): LinkedIn company pages only, never a
personal profile; import is an idempotent upsert. Plus the feed parser's GUID and
publisher fields the feed targets rely on (`portolan.connectors.rss.parse_feed`)."""

from __future__ import annotations

import pytest

from portolan.connectors.rss import parse_feed
from portolan.signals.targets import import_targets, load_targets, parse_targets_yaml

COMPANY = "https://www.linkedin.com/company/example-health/"


def _feed(**feed) -> dict:
    return {"accounts": [{"crm_id": "PUB-001", "feeds": [feed]}]}


def test_a_linkedin_feed_needs_its_company_page() -> None:
    rows = parse_targets_yaml(
        _feed(url="https://rss.app/feeds/a.xml", kind="linkedin", linkedin_company_url=COMPANY)
    )
    assert rows[0]["source_page_url"] == COMPANY

    with pytest.raises(ValueError, match="company page"):
        parse_targets_yaml(_feed(url="https://rss.app/feeds/a.xml", kind="linkedin"))


@pytest.mark.parametrize(
    "profile",
    [
        "https://www.linkedin.com/in/jane-doe/",
        "https://linkedin.com/in/jane-doe",
        "https://www.linkedin.com/pub/jane-doe/1/2/3",
    ],
)
def test_personal_profiles_are_refused_everywhere(profile: str) -> None:
    with pytest.raises(ValueError, match=r"company page|personal"):
        parse_targets_yaml(
            _feed(url="https://rss.app/feeds/a.xml", kind="linkedin", linkedin_company_url=profile)
        )
    with pytest.raises(ValueError, match="personal"):
        parse_targets_yaml(_feed(url=profile, kind="other"))
    with pytest.raises(ValueError, match="personal"):
        parse_targets_yaml({"accounts": [{"crm_id": "PUB-001", "pages": [profile]}]})


def test_linkedin_is_never_fetched_as_a_page() -> None:
    with pytest.raises(ValueError, match=r"RSS\.app company-page feed"):
        parse_targets_yaml({"accounts": [{"crm_id": "PUB-001", "pages": [COMPANY]}]})


def test_import_is_an_idempotent_upsert_that_deactivates_dropped_targets(db) -> None:
    data = {
        "accounts": [
            {
                "crm_id": "PUB-001",
                "feeds": [
                    {
                        "url": "https://rss.app/feeds/a.xml",
                        "kind": "linkedin",
                        "linkedin_company_url": COMPANY,
                    }
                ],
                "pages": ["https://www.example-health.test/newsroom"],
            }
        ]
    }
    first = import_targets(db, data)
    again = import_targets(db, data)
    data["accounts"][0]["pages"] = []
    dropped = import_targets(db, data)

    assert (first.added, again.unchanged, dropped.deactivated) == (2, 2, 1)
    targets = load_targets(db, active_only=False)
    assert [(t.kind, t.active) for t in targets] == [("feed", True), ("page", False)]


def test_parse_feed_reads_guid_and_publisher() -> None:
    xml = """<rss version="2.0"><channel><title>t</title><link>https://x.test</link>
      <item><title>Example Health opens Mesa clinic - Phoenix Business Journal</title>
        <link>https://news.test/r/abc</link><guid isPermaLink="false">CBMiabc</guid>
        <pubDate>Mon, 21 Sep 2026 18:00:00 GMT</pubDate>
        <source url="https://www.bizjournals.com">Phoenix Business Journal</source>
        <description>Example Health opens Mesa clinic</description></item>
      <item><title>No guid here</title><link>https://news.test/r/def</link></item>
    </channel></rss>"""
    first, second = parse_feed(xml)

    assert (first.guid, first.source_name, first.dated) == (
        "CBMiabc",
        "Phoenix Business Journal",
        True,
    )
    assert (second.guid, second.dated) == ("https://news.test/r/def", False)
