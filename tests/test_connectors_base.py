"""Tests for `portolan.connectors.base`: the shared `request_with_retry`
retry/backoff/error path every connector's HTTP calls go through, and
`format_ms_offset` / `parse_datetime`.
"""

from __future__ import annotations

import pytest
from connector_helpers import FakeTransport, json_response, sleep_recorder

from portolan.connectors.base import (
    ConnectorError,
    format_ms_offset,
    parse_datetime,
    request_with_retry,
)

URL = "https://api.example.test/v1/things"


def test_success_passes_through_on_first_try():
    transport = FakeTransport()
    transport.queue(URL, json_response({"ok": True}))
    sleep, waits = sleep_recorder()

    response = request_with_retry(transport, "GET", URL, sleep=sleep, vendor="example")

    assert response.json() == {"ok": True}
    assert waits == []


def test_a_429_then_success_retries_once_with_backoff():
    transport = FakeTransport()
    transport.queue(
        URL,
        json_response({"error": "rate limited"}, status_code=429),
        json_response({"ok": True}),
    )
    sleep, waits = sleep_recorder()

    response = request_with_retry(transport, "GET", URL, sleep=sleep, vendor="example")

    assert response.json() == {"ok": True}
    assert waits == [1.0]  # backoff_seconds(1.0) * 2**0


def test_429_honours_a_numeric_retry_after_header():
    transport = FakeTransport()
    transport.queue(
        URL,
        json_response({"error": "rate limited"}, status_code=429, headers={"Retry-After": "5"}),
        json_response({"ok": True}),
    )
    sleep, waits = sleep_recorder()

    request_with_retry(transport, "GET", URL, sleep=sleep, vendor="example")

    assert waits == [5.0]


def test_exceeding_max_retries_raises_connector_error():
    transport = FakeTransport()
    transport.queue(
        URL,
        json_response({}, status_code=429),
        json_response({}, status_code=429),
        json_response({}, status_code=429),
        json_response({}, status_code=429),
    )
    sleep, waits = sleep_recorder()

    with pytest.raises(ConnectorError):
        request_with_retry(transport, "GET", URL, sleep=sleep, max_retries=3, vendor="example")

    assert len(waits) == 3


def test_non_429_error_raises_connector_error_with_vendor_message():
    transport = FakeTransport()
    transport.queue(URL, json_response({"message": "widget not found"}, status_code=404))
    sleep, _ = sleep_recorder()

    with pytest.raises(ConnectorError) as excinfo:
        request_with_retry(transport, "GET", URL, sleep=sleep, vendor="example")

    assert "widget not found" in str(excinfo.value)
    assert "404" in str(excinfo.value)


def test_error_message_never_reads_request_headers():
    """A connector's own secret lives in the *request* headers it sends,
    never in the response body — `request_with_retry` must build its
    error message only from the response, so passing a header containing
    a secret must not leak it into the raised message."""
    transport = FakeTransport()
    transport.queue(URL, json_response({"message": "forbidden"}, status_code=403))
    sleep, _ = sleep_recorder()

    with pytest.raises(ConnectorError) as excinfo:
        request_with_retry(
            transport,
            "GET",
            URL,
            headers={"Authorization": "Bearer super-secret-token-value"},
            sleep=sleep,
            vendor="example",
        )

    assert "super-secret-token-value" not in str(excinfo.value)


def test_format_ms_offset():
    assert format_ms_offset(0) == "00:00:00"
    assert format_ms_offset(3400) == "00:00:03"
    assert format_ms_offset(123000) == "00:02:03"
    assert format_ms_offset(3_661_000) == "01:01:01"


def test_parse_datetime_handles_z_and_offset_suffixes():
    assert parse_datetime("2026-08-05T12:00:00Z").year == 2026
    assert parse_datetime("2026-08-05T12:00:00.000+0000").day == 5
