"""Tests for `portolan.connectors.gong.GongConnector`.

Fixtures under `tests/fixtures/connectors/gong/` mirror Gong's `/v2/calls`,
`/v2/calls/transcript` and `/v2/calls/extensive` response shapes (module
docstring in `gong.py` cites the doc sources). The transcript's planted
line ("Two shipments showed up days late this week") echoes truth-table
fact F01's phrasing constraint (`corpus/truth-table.yaml`).
"""

from __future__ import annotations

import pytest
from connector_helpers import FakeTransport, json_response, load_json_fixture, sleep_recorder

from portolan.connectors.base import ConnectorError
from portolan.connectors.gong import GongConfig, GongConnector
from portolan.extract.schema import SourceType

BASE = "https://api.gong.io"
CALLS_URL = f"{BASE}/v2/calls"
TRANSCRIPT_URL = f"{BASE}/v2/calls/transcript"
EXTENSIVE_URL = f"{BASE}/v2/calls/extensive"

CALL_ID = "7782342871234567890"


def _config() -> GongConfig:
    return GongConfig(access_key="mock-access-key", access_key_secret="mock-access-key-secret")


def _connector_two_pages() -> tuple[GongConnector, FakeTransport]:
    transport = FakeTransport()
    transport.queue(
        CALLS_URL,
        json_response(load_json_fixture("gong", "calls_page1.json")),
        json_response(load_json_fixture("gong", "calls_page2.json")),
    )
    return GongConnector(_config(), transport=transport), transport


def test_list_sources_paginates_via_records_cursor():
    connector, transport = _connector_two_pages()

    refs = list(connector.list_sources(since=None))

    assert [r.source_id for r in refs] == [
        "gong-7782342871234567890",
        "gong-7782342871234567891",
    ]
    assert all(r.source_type == SourceType.CALL for r in refs)
    calls = [c for c in transport.calls if c["url"] == CALLS_URL]
    assert len(calls) == 2
    assert calls[1]["params"]["cursor"] == "eyJwYWdlIjoxfQ=="


def test_fetch_converts_transcript_to_timestamped_lines_with_a_specific_line():
    connector, _ = _connector_two_pages()
    refs = list(connector.list_sources(since=None))
    ref = next(r for r in refs if r.source_id == f"gong-{CALL_ID}")

    transport = FakeTransport()
    transport.queue(
        TRANSCRIPT_URL,
        json_response(load_json_fixture("gong", f"transcript_{CALL_ID}.json")),
    )
    transport.queue(
        EXTENSIVE_URL,
        json_response(load_json_fixture("gong", f"extensive_{CALL_ID}.json")),
    )
    connector._transport = transport  # swap in the fetch-only fixtures

    raw = connector.fetch(ref)

    lines = raw.text.splitlines()
    assert lines[0] == "[00:00:00] Sam Okafor: Hey Rob, thanks for jumping on today."
    assert (
        lines[1] == "[00:00:03] Rob Tallis: Oh, you know, Wednesday. "
        "We had a wild overnight but today's been quiet."
    )
    assert lines[2] == "[00:02:03] Sam Okafor: Two shipments showed up days late this week."
    assert raw.source_type == SourceType.CALL
    emails = {p.email for p in raw.participants}
    assert emails == {"s.okafor@halvardmedical.com", "rtallis@ashgrovehealth.org"}
    assert raw.metadata["call_id"] == CALL_ID


def test_auth_error_surfaces_as_connector_error_without_the_secret():
    transport = FakeTransport()
    transport.queue(
        CALLS_URL, json_response(load_json_fixture("gong", "auth_error.json"), status_code=401)
    )
    config = _config()
    connector = GongConnector(config, transport=transport)

    with pytest.raises(ConnectorError) as excinfo:
        list(connector.list_sources(since=None))

    message = str(excinfo.value)
    assert "invalid or has expired" in message
    assert config.access_key_secret not in message


def test_a_429_then_success_retries_and_returns_data():
    transport = FakeTransport()
    transport.queue(
        CALLS_URL,
        json_response({}, status_code=429, headers={"Retry-After": "0"}),
        # `calls_page2.json` carries no `records.cursor`, so pagination
        # stops after this one response — the point of this test is the
        # retry, not another round of pagination.
        json_response(load_json_fixture("gong", "calls_page2.json")),
    )
    sleep, waits = sleep_recorder()
    connector = GongConnector(_config(), transport=transport, sleep=sleep)

    refs = list(connector.list_sources(since=None))

    assert refs[0].source_id == "gong-7782342871234567891"
    assert waits == [0.0]
