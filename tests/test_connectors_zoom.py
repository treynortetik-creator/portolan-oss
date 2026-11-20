"""Tests for `portolan.connectors.zoom.ZoomConnector` and
`convert_vtt_to_call_lines`.

Fixtures under `tests/fixtures/connectors/zoom/` mirror Zoom's
Server-to-Server OAuth token response, cloud-recordings list, a VTT
transcript, and the meeting-participants report (module docstring in
`zoom.py` cites the doc sources). The VTT's third cue echoes truth-table
fact F01's phrasing ("two shipments showed up days late",
`corpus/truth-table.yaml`).
"""

from __future__ import annotations

import copy
from datetime import date, timedelta

import pytest
from connector_helpers import (
    FakeTransport,
    json_response,
    load_json_fixture,
    load_text_fixture,
    sleep_recorder,
    text_response,
)

from portolan.connectors.base import ConnectorError
from portolan.connectors.zoom import ZoomConfig, ZoomConnector, convert_vtt_to_call_lines
from portolan.extract.schema import SourceType

TOKEN_URL = "https://zoom.us/oauth/token"
RECORDINGS_URL = "https://api.zoom.us/v2/users/me/recordings"
TRANSCRIPT_URL = "https://zoom.us/rec/download/transcript-mock"
PARTICIPANTS_URL = "https://api.zoom.us/v2/report/meetings/91234567890/participants"


def _config() -> ZoomConfig:
    return ZoomConfig(
        account_id="mock-account-id", client_id="mock-client-id", client_secret="mock-client-secret"
    )


def _connector() -> tuple[ZoomConnector, FakeTransport]:
    transport = FakeTransport()
    transport.queue(TOKEN_URL, json_response(load_json_fixture("zoom", "oauth_token.json")))
    transport.queue(RECORDINGS_URL, json_response(load_json_fixture("zoom", "recordings.json")))
    transport.queue(TRANSCRIPT_URL, text_response(load_text_fixture("zoom", "transcript.vtt")))
    transport.queue(PARTICIPANTS_URL, json_response(load_json_fixture("zoom", "participants.json")))
    return ZoomConnector(_config(), transport=transport), transport


def test_convert_vtt_to_call_lines_produces_a_specific_timestamped_line():
    lines = convert_vtt_to_call_lines(load_text_fixture("zoom", "transcript.vtt"))

    assert lines[0] == "[00:00:00] Sam Okafor: Hey Rob, thanks for jumping on today."
    assert lines[2] == "[00:02:03] Sam Okafor: Two shipments showed up days late this week."


def test_list_sources_only_yields_meetings_with_a_transcript_file():
    connector, _ = _connector()

    refs = list(connector.list_sources(since=None))

    assert len(refs) == 1
    # Bug hunt 2026-09-18, finding 12: `source_id` is now `_zoom_source_
    # id(meeting)` -- a hash of the meeting UUID, not the UUID passed
    # through verbatim (real Zoom UUIDs are base64 and routinely contain
    # "/" and "+", which broke `/originals/{slug}` routing -- see the
    # test below with a UUID that actually contains one).
    assert refs[0].source_id == "zoom-5de7cd4c562fbe56"
    assert refs[0].source_type == SourceType.CALL


def test_source_id_is_url_safe_even_when_the_uuid_contains_a_slash():
    # Zoom's own docs: meeting UUIDs are base64 and can contain "/" and
    # "+" -- passed straight through, a "/" nests `original_key`'s store
    # key and the FastAPI path param at `/originals/{slug}` can't match a
    # slug containing a slash at all, making the source unroutable.
    transport = FakeTransport()
    transport.queue(TOKEN_URL, json_response(load_json_fixture("zoom", "oauth_token.json")))
    recordings = copy.deepcopy(load_json_fixture("zoom", "recordings.json"))
    recordings["meetings"][0]["uuid"] = "abc/def+ghi=="
    transport.queue(RECORDINGS_URL, json_response(recordings))
    transport.queue(TRANSCRIPT_URL, text_response(load_text_fixture("zoom", "transcript.vtt")))
    transport.queue(PARTICIPANTS_URL, json_response(load_json_fixture("zoom", "participants.json")))
    connector = ZoomConnector(_config(), transport=transport)

    refs = list(connector.list_sources(since=None))

    assert len(refs) == 1
    assert "/" not in refs[0].source_id
    assert "+" not in refs[0].source_id
    assert "=" not in refs[0].source_id

    raw = connector.fetch(refs[0])
    # `list_sources`'s ref and `fetch`'s own RawSource must agree exactly
    # -- both call `_zoom_source_id` against the same meeting dict.
    assert raw.source_id == refs[0].source_id


def test_fetch_downloads_transcript_and_participants():
    connector, transport = _connector()
    ref = next(iter(connector.list_sources(since=None)))

    raw = connector.fetch(ref)

    lines = raw.text.splitlines()
    assert lines[0] == "[00:00:00] Sam Okafor: Hey Rob, thanks for jumping on today."
    assert lines[2] == "[00:02:03] Sam Okafor: Two shipments showed up days late this week."
    emails = {p.email for p in raw.participants}
    assert emails == {"s.okafor@halvardmedical.com", "rtallis@ashgrovehealth.org"}
    assert raw.metadata["meeting_id"] == 91234567890
    token_calls = [c for c in transport.calls if c["url"] == TOKEN_URL]
    assert len(token_calls) == 1  # authenticate once, cache the token


def test_auth_error_surfaces_as_connector_error_without_the_secret():
    transport = FakeTransport()
    transport.queue(
        TOKEN_URL, json_response(load_json_fixture("zoom", "auth_error.json"), status_code=401)
    )
    config = _config()
    connector = ZoomConnector(config, transport=transport)

    with pytest.raises(ConnectorError) as excinfo:
        list(connector.list_sources(since=None))

    message = str(excinfo.value)
    assert "Invalid access token" in message
    assert config.client_secret not in message


def test_a_429_then_success_retries_and_returns_data():
    transport = FakeTransport()
    transport.queue(TOKEN_URL, json_response(load_json_fixture("zoom", "oauth_token.json")))
    transport.queue(
        RECORDINGS_URL,
        json_response({}, status_code=429, headers={"Retry-After": "0"}),
        json_response(load_json_fixture("zoom", "recordings.json")),
    )
    sleep, waits = sleep_recorder()
    connector = ZoomConnector(_config(), transport=transport, sleep=sleep)

    refs = list(connector.list_sources(since=None))

    assert refs[0].source_id == "zoom-5de7cd4c562fbe56"
    assert waits == [0.0]


def test_fetch_refuses_a_download_url_on_a_foreign_host():
    """Review finding F8: a compromised or spoofed recordings response
    could point a transcript's download_url at an attacker's host and
    this connector would still attach its bearer token."""
    transport = FakeTransport()
    transport.queue(TOKEN_URL, json_response(load_json_fixture("zoom", "oauth_token.json")))
    hostile_recordings = copy.deepcopy(load_json_fixture("zoom", "recordings.json"))
    hostile_recordings["meetings"][0]["recording_files"][1]["download_url"] = (
        "https://evil.example.com/steal-the-bearer-token"
    )
    transport.queue(RECORDINGS_URL, json_response(hostile_recordings))
    connector = ZoomConnector(_config(), transport=transport)
    ref = next(iter(connector.list_sources(since=None)))

    with pytest.raises(ConnectorError, match=r"evil\.example\.com"):
        connector.fetch(ref)

    assert not any(c["url"].startswith("https://evil.example.com") for c in transport.calls)


def test_list_sources_with_a_wide_since_chunks_into_windows():
    """Review finding F17: Zoom's recordings list accepts roughly a
    month per request; a `since` far enough back used to silently return
    only the vendor's first window. A 75-day range should make three
    requests (30 + 30 + 15 days), each with its own from/to bound."""
    since = date.today() - timedelta(days=75)
    transport = FakeTransport()
    transport.queue(TOKEN_URL, json_response(load_json_fixture("zoom", "oauth_token.json")))
    empty_page = json_response({"meetings": []})
    transport.queue(RECORDINGS_URL, empty_page, empty_page, empty_page)
    connector = ZoomConnector(_config(), transport=transport)

    list(connector.list_sources(since=since))

    recordings_calls = [c for c in transport.calls if c["url"] == RECORDINGS_URL]
    assert len(recordings_calls) == 3
    for call in recordings_calls:
        assert "from" in call["params"]
        assert "to" in call["params"]
