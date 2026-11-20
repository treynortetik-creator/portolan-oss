"""Tests for `portolan.connectors.tickets`: `ZendeskConnector` (an API
client, fixtures under `tests/fixtures/connectors/zendesk/`) and
`ExportTicketConnector` (a local CSV/JSON reader, fixtures under
`tests/fixtures/connectors/ticket_export/`). Names are Halvard Medical's
synthetic customers from `corpus/truth-table.yaml`.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from connector_helpers import FakeTransport, json_response, load_json_fixture, sleep_recorder

from portolan.connectors.base import ConnectorError
from portolan.connectors.tickets import (
    ExportTicketConfig,
    ExportTicketConnector,
    ZendeskConfig,
    ZendeskConnector,
)
from portolan.extract.schema import SourceType

FIXTURES_ROOT = Path(__file__).resolve().parent / "fixtures" / "connectors"

BASE = "https://halvardmedical.zendesk.com"
TICKETS_URL = f"{BASE}/api/v2/tickets.json"
COMMENTS_URL = f"{BASE}/api/v2/tickets/4031/comments.json"


def _zendesk_config() -> ZendeskConfig:
    return ZendeskConfig(
        subdomain="halvardmedical", email="sam@halvardmedical.com", api_token="mock-api-token"
    )


def _zendesk_connector() -> tuple[ZendeskConnector, FakeTransport]:
    transport = FakeTransport()
    transport.queue(
        TICKETS_URL,
        json_response(load_json_fixture("zendesk", "tickets_page1.json")),
        json_response(load_json_fixture("zendesk", "tickets_page2.json")),
    )
    transport.queue(COMMENTS_URL, json_response(load_json_fixture("zendesk", "comments_4031.json")))
    return ZendeskConnector(_zendesk_config(), transport=transport), transport


def test_zendesk_pagination_across_two_pages():
    connector, transport = _zendesk_connector()

    refs = list(connector.list_sources(since=None))

    assert [r.source_id for r in refs] == ["zendesk-4031", "zendesk-4038"]
    assert all(r.source_type == SourceType.TICKET for r in refs)
    calls = [c for c in transport.calls if c["url"] == TICKETS_URL]
    assert len(calls) == 2
    assert calls[1]["params"]["page[after]"] == "cursor-page-2"


def test_zendesk_fetch_preserves_comment_numbering():
    connector, _ = _zendesk_connector()
    refs = list(connector.list_sources(since=None))
    ref = next(r for r in refs if r.source_id == "zendesk-4031")

    raw = connector.fetch(ref)

    numbers = [c["number"] for c in raw.metadata["comments"]]
    assert numbers == [1, 2, 3]
    assert raw.metadata["comments"][0]["body"].startswith("Our Tidewater T2 consumable order")
    # created_at 2026-08-05T00:00:00Z is 2026-08-04 17:00 in the org's zone.
    assert raw.source_date == date(2026, 8, 4)
    assert "[1]" in raw.text and "[3]" in raw.text


def test_zendesk_auth_error_surfaces_as_connector_error_without_the_token():
    transport = FakeTransport()
    transport.queue(
        TICKETS_URL, json_response(load_json_fixture("zendesk", "auth_error.json"), status_code=401)
    )
    config = _zendesk_config()
    connector = ZendeskConnector(config, transport=transport)

    with pytest.raises(ConnectorError) as excinfo:
        list(connector.list_sources(since=None))

    message = str(excinfo.value)
    assert "authenticate" in message.lower()
    assert config.api_token not in message


def test_zendesk_a_429_then_success_retries_and_returns_data():
    transport = FakeTransport()
    transport.queue(
        TICKETS_URL,
        json_response({}, status_code=429, headers={"Retry-After": "0"}),
        json_response(load_json_fixture("zendesk", "tickets_page2.json")),
    )
    sleep, waits = sleep_recorder()
    connector = ZendeskConnector(_zendesk_config(), transport=transport, sleep=sleep)

    refs = list(connector.list_sources(since=None))

    assert refs[0].source_id == "zendesk-4038"
    assert waits == [0.0]


INCREMENTAL_URL = f"{BASE}/api/v2/incremental/tickets.json"


def test_zendesk_since_uses_incremental_export_not_the_plain_list():
    """Review finding F17: the plain ticket list has no date filter, so
    an incremental crawl used to page every ticket and filter client-
    side. since=<date> now uses Zendesk's Incremental Export API
    instead, cursor-paginated via after_url until end_of_stream."""
    transport = FakeTransport()
    transport.queue(
        INCREMENTAL_URL,
        json_response(
            {
                "tickets": [{"id": 4031, "updated_at": "2026-08-14T12:00:00Z"}],
                "end_of_stream": False,
                "after_url": f"{INCREMENTAL_URL}?cursor=abc",
            }
        ),
    )
    transport.queue(
        f"{INCREMENTAL_URL}?cursor=abc",
        json_response(
            {
                "tickets": [{"id": 4038, "updated_at": "2026-08-21T09:00:00Z"}],
                "end_of_stream": True,
            }
        ),
    )
    connector = ZendeskConnector(_zendesk_config(), transport=transport)

    refs = list(connector.list_sources(since=date(2026, 8, 1)))

    assert {r.source_id for r in refs} == {"zendesk-4031", "zendesk-4038"}
    assert not any(c["url"] == TICKETS_URL for c in transport.calls)


def test_zendesk_incremental_export_refuses_a_foreign_after_url():
    transport = FakeTransport()
    transport.queue(
        INCREMENTAL_URL,
        json_response(
            {
                "tickets": [],
                "end_of_stream": False,
                "after_url": "https://evil.example.com/steal-the-token",
            }
        ),
    )
    connector = ZendeskConnector(_zendesk_config(), transport=transport)

    with pytest.raises(ConnectorError, match=r"evil\.example\.com"):
        list(connector.list_sources(since=date(2026, 8, 1)))

    assert not any(c["url"].startswith("https://evil.example.com") for c in transport.calls)


def test_export_json_reader_preserves_comment_numbering():
    config = ExportTicketConfig(path=FIXTURES_ROOT / "ticket_export" / "export.json", format="json")
    connector = ExportTicketConnector(config)

    refs = list(connector.list_sources(since=None))
    assert refs[0].source_id == "export-T-9001"

    raw = connector.fetch(refs[0])
    numbers = [c["number"] for c in raw.metadata["comments"]]
    assert numbers == [1, 2]
    assert raw.source_type == SourceType.TICKET
    assert "Spanish-language" in raw.text


def test_export_csv_reader_groups_rows_by_ticket_and_preserves_numbering():
    config = ExportTicketConfig(path=FIXTURES_ROOT / "ticket_export" / "export.csv", format="csv")
    connector = ExportTicketConnector(config)

    refs = list(connector.list_sources(since=None))
    assert refs[0].source_id == "export-T-9002"

    raw = connector.fetch(refs[0])
    numbers = [c["number"] for c in raw.metadata["comments"]]
    assert numbers == [1, 2]
    assert "dosage-range alert" in raw.text


def test_zendesk_incremental_start_time_is_utc_midnight_whatever_the_host_zone():
    """`start_time` is a Unix timestamp; `since` is a calendar day, read as
    UTC like HubSpot's `since`. The suite runs with TZ=UTC, so the host zone
    is switched here to show the value does not depend on it."""
    import os
    import time
    from datetime import UTC, datetime

    transport = FakeTransport()
    transport.queue(INCREMENTAL_URL, json_response({"tickets": [], "end_of_stream": True}))
    connector = ZendeskConnector(_zendesk_config(), transport=transport)

    original_tz = os.environ.get("TZ")
    os.environ["TZ"] = "America/Phoenix"
    time.tzset()
    try:
        list(connector.list_sources(since=date(2026, 8, 1)))
    finally:
        if original_tz is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = original_tz
        time.tzset()

    (call,) = [c for c in transport.calls if c["url"] == INCREMENTAL_URL]
    assert call["params"]["start_time"] == int(datetime(2026, 8, 1, tzinfo=UTC).timestamp())
