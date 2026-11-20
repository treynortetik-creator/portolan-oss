"""Tests for `portolan.connectors.clari.ClariConnector` (EXPERIMENTAL —
see the module docstring for what is unverified against a real Clari
Copilot account). Fixtures under `tests/fixtures/connectors/clari/` are
this module's best reading of the community-documented shapes. Names are
Halvard Medical's synthetic customers from `corpus/truth-table.yaml`
(Northlake General Hospital, Derek Hollis, Priya Natarajan).
"""

from __future__ import annotations

import pytest
from connector_helpers import FakeTransport, json_response, load_json_fixture, sleep_recorder

from portolan.connectors.base import ConnectorError
from portolan.connectors.clari import ClariConfig, ClariConnector
from portolan.extract.schema import SourceType

CONVERSATIONS_URL = "https://api.clari.com/api/v1/conversations"
CALL_DETAILS_URL = "https://api.clari.com/call-details"


def _config() -> ClariConfig:
    return ClariConfig(api_key="mock-api-key", api_password="mock-api-password")


def _connector() -> tuple[ClariConnector, FakeTransport]:
    transport = FakeTransport()
    transport.queue(
        CONVERSATIONS_URL, json_response(load_json_fixture("clari", "conversations.json"))
    )
    transport.queue(
        CALL_DETAILS_URL, json_response(load_json_fixture("clari", "call_details.json"))
    )
    return ClariConnector(_config(), transport=transport), transport


def test_list_sources_yields_calls():
    connector, _ = _connector()

    refs = list(connector.list_sources(since=None))

    assert [r.source_id for r in refs] == ["clari-conv-90001"]
    assert refs[0].source_type == SourceType.CALL


def test_fetch_converts_turns_to_timestamped_lines():
    connector, _ = _connector()
    ref = next(iter(connector.list_sources(since=None)))

    raw = connector.fetch(ref)

    lines = raw.text.splitlines()
    assert lines[0] == "[00:00:00] Priya Natarajan: Hi Derek, thanks for making time today."
    assert (
        lines[2]
        == "[00:01:35] Derek Hollis: We are still evaluating a Corvane pump for the new tower."
    )
    emails = {p.email for p in raw.participants}
    assert emails == {"p.natarajan@halvardmedical.com", "dhollis@northlakehs.org"}
    assert raw.metadata["call_id"] == "conv-90001"


def test_auth_error_surfaces_as_connector_error_without_the_password():
    transport = FakeTransport()
    transport.queue(
        CONVERSATIONS_URL,
        json_response(load_json_fixture("clari", "auth_error.json"), status_code=401),
    )
    config = _config()
    connector = ClariConnector(config, transport=transport)

    with pytest.raises(ConnectorError) as excinfo:
        list(connector.list_sources(since=None))

    message = str(excinfo.value)
    assert "API key or password is invalid" in message
    assert config.api_password not in message


def test_a_429_then_success_retries_and_returns_data():
    transport = FakeTransport()
    transport.queue(
        CONVERSATIONS_URL,
        json_response({}, status_code=429, headers={"Retry-After": "0"}),
        json_response(load_json_fixture("clari", "conversations.json")),
    )
    sleep, waits = sleep_recorder()
    connector = ClariConnector(_config(), transport=transport, sleep=sleep)

    refs = list(connector.list_sources(since=None))

    assert refs[0].source_id == "clari-conv-90001"
    assert waits == [0.0]
