"""Connector protocol and shared HTTP plumbing (build-contracts.md
"Connectors").

Every network connector in this package takes an injectable `Transport`
callable, so tests never touch the network ("Tests must not hit the
network. Use fakes.", build-contracts.md "Ground rules").
`request_with_retry` is the one path their HTTP calls go through: a `429`
is retried up to three times with exponential backoff (honouring a clamped
`Retry-After`); any other non-2xx raises `ConnectorError` with the vendor's
own error message, built from the response body only, never from local
config, so an error can never leak the connector's credentials.

`format_ms_offset` turns a transcript's milliseconds-from-call-start
(Gong, Clari) into the `[HH:MM:SS]` line prefix `portolan/ingest/prepare.py`
indexes as a call locator.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Protocol, runtime_checkable
from urllib.parse import urlsplit

from portolan.extract.schema import SourceType
from portolan.ingest.models import RawSource
from portolan.util import clock

__all__ = [
    "Connector",
    "ConnectorError",
    "SourceRef",
    "Transport",
    "TransportResponse",
    "format_ms_offset",
    "httpx_transport",
    "iter_date_windows",
    "parse_datetime",
    "request_with_reauth",
    "request_with_retry",
    "require_allowed_origin",
    "source_day",
]


class ConnectorError(Exception):
    """A connector's HTTP call failed (a non-2xx, non-retried-429
    response). Carries the vendor's own error message text — constructed
    only from the HTTP response body, never from request headers or the
    connector's own config, so a raised message can never contain a
    token, API key, or secret.

    `status_code` is the HTTP status, when known, so a caller can tell a
    401 (`request_with_reauth`) from other failures without parsing text.
    """

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


@runtime_checkable
class TransportResponse(Protocol):
    """Duck-typed to match `httpx.Response` exactly, so
    `httpx_transport` needs no wrapping."""

    status_code: int
    headers: Mapping[str, str]
    text: str
    content: bytes

    def json(self) -> Any: ...


class Transport(Protocol):
    """What every connector calls instead of an HTTP client directly.
    Tests inject a fake; production wraps an `httpx.Client` with
    `httpx_transport`."""

    def __call__(
        self,
        method: str,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        json: Any = None,
        data: Any = None,
        headers: Mapping[str, str] | None = None,
    ) -> TransportResponse: ...


@dataclass(frozen=True)
class SourceRef:
    """What `Connector.list_sources` yields and `Connector.fetch`
    consumes (build-contracts.md: "yielding SourceRef(source_id,
    source_type, updated_at, ref)"). `ref` is an opaque, connector-owned
    pointer: usually the full listed record (so `fetch` is a pure
    transform of data already in hand) or an id plus whatever a
    follow-up call needs, for vendors that only expose the full payload
    from a second endpoint (Gong and Zoom transcripts, Zendesk comments,
    Clari call details)."""

    source_id: str
    source_type: SourceType
    updated_at: datetime
    ref: Any


@runtime_checkable
class Connector(Protocol):
    """build-contracts.md "Connectors"."""

    name: str

    def list_sources(self, since: date | None) -> Iterable[SourceRef]: ...

    def fetch(self, ref: SourceRef) -> RawSource: ...


def _vendor_message(response: TransportResponse, fallback: str) -> str:
    """Best-effort extraction of a human-readable error message from a
    response *body*. Never reads `response.request` or any header — the
    only place a credential could leak into this string is the vendor's
    own body, which is out of this connector's control and is exactly
    what a caller of the real API would also see."""
    try:
        data = response.json()
    except Exception:
        text = (response.text or "").strip()
        return text[:500] if text else fallback

    if isinstance(data, dict):
        for key in ("message", "error_description", "error", "reason", "detail"):
            value = data.get(key)
            if isinstance(value, str) and value:
                return value
        errors = data.get("errors")
        if isinstance(errors, list) and errors:
            first = errors[0]
            if isinstance(first, dict):
                for key in ("message", "title", "description"):
                    if isinstance(first.get(key), str):
                        return first[key]
            elif isinstance(first, str):
                return first
    elif isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, dict) and isinstance(first.get("message"), str):
            return first["message"]

    text = (response.text or "").strip()
    return text[:500] if text else fallback


# Cap on a vendor's `Retry-After`, so a hostile or malformed header cannot
# force an arbitrarily long sleep.
MAX_RETRY_AFTER_SECONDS = 60.0


def request_with_retry(
    transport: Transport,
    method: str,
    url: str,
    *,
    params: Mapping[str, Any] | None = None,
    json: Any = None,
    data: Any = None,
    headers: Mapping[str, str] | None = None,
    max_retries: int = 3,
    backoff_seconds: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
    vendor: str = "connector",
) -> TransportResponse:
    """Call `transport`, retrying a `429` up to `max_retries` times with
    exponential backoff (`backoff_seconds * 2**attempt`, or a numeric
    `Retry-After` header when the response carries one and it parses as a
    sane, non-negative, clamped wait), and raising `ConnectorError`
    immediately for any other non-2xx status."""
    attempt = 0
    while True:
        response = transport(method, url, params=params, json=json, data=data, headers=headers)
        if 200 <= response.status_code < 300:
            return response

        if response.status_code == 429 and attempt < max_retries:
            retry_after = (response.headers or {}).get("Retry-After")
            try:
                wait = float(retry_after) if retry_after is not None else None
            except ValueError:
                wait = None
            if wait is None or wait < 0:
                wait = backoff_seconds * (2**attempt)
            wait = min(wait, MAX_RETRY_AFTER_SECONDS)
            sleep(wait)
            attempt += 1
            continue

        message = _vendor_message(
            response, f"{vendor} request failed with status {response.status_code}"
        )
        raise ConnectorError(
            f"{vendor}: HTTP {response.status_code}: {message}", status_code=response.status_code
        )


def request_with_reauth(
    transport: Transport,
    method: str,
    url: str,
    *,
    headers: Mapping[str, str],
    reauthenticate: Callable[[], Mapping[str, str]],
    params: Mapping[str, Any] | None = None,
    json: Any = None,
    data: Any = None,
    max_retries: int = 3,
    backoff_seconds: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
    vendor: str = "connector",
) -> TransportResponse:
    """`request_with_retry`, plus one re-authenticate-and-retry on a `401`
    (a cached OAuth token that expired).

    `reauthenticate()` must drop the cached token and return fresh headers
    for the retry. A second `401` propagates as `ConnectorError`; this only
    ever retries once.
    """
    try:
        return request_with_retry(
            transport,
            method,
            url,
            params=params,
            json=json,
            data=data,
            headers=headers,
            max_retries=max_retries,
            backoff_seconds=backoff_seconds,
            sleep=sleep,
            vendor=vendor,
        )
    except ConnectorError as exc:
        if exc.status_code != 401:
            raise
        fresh_headers = reauthenticate()
        return request_with_retry(
            transport,
            method,
            url,
            params=params,
            json=json,
            data=data,
            headers=fresh_headers,
            max_retries=max_retries,
            backoff_seconds=backoff_seconds,
            sleep=sleep,
            vendor=vendor,
        )


def require_allowed_origin(url: str, *, allowed_hosts: Iterable[str], vendor: str) -> str:
    """`url`, unchanged, if it is relative or its host is one of
    `allowed_hosts`; otherwise raises `ConnectorError` without ever
    calling the transport.

    A vendor-supplied URL (Salesforce `nextRecordsUrl`, Zendesk `after_url`,
    Zoom `download_url`) gets the connector's credentials attached when
    followed; a spoofed response could point it at another host. Check it
    here first, so no request is sent to a host outside the allow-list.
    """
    if "://" not in url:
        return url  # relative -- always resolved against our own configured host
    host = (urlsplit(url).hostname or "").lower()
    allowed = {h.lower() for h in allowed_hosts if h}
    if host not in allowed:
        raise ConnectorError(
            f"{vendor}: refusing to send credentials to {host!r} -- not one of this "
            f"connector's configured API host(s) ({sorted(allowed)!r})"
        )
    return url


def iter_date_windows(start: date, end: date, window_days: int) -> Iterator[tuple[date, date]]:
    """Split `[start, end]` into consecutive windows of at most
    `window_days` days each, for vendors that cap a list request's date
    span (Gong 90 days, Zoom about a month). Always yields at least one
    window, even when `start >= end`, so a same-day backfill still makes one
    request.
    """
    if start >= end:
        yield start, end
        return
    current = start
    while current < end:
        window_end = min(current + timedelta(days=window_days), end)
        yield current, window_end
        current = window_end


def httpx_transport(client: Any) -> Transport:
    """Adapt an `httpx.Client` to `Transport`, for production use.
    `httpx.Response` already satisfies `TransportResponse` by duck typing
    (`status_code`, `headers`, `text`, `content`, `.json()`), so no
    wrapping of the response is needed."""

    def _call(
        method: str,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        json: Any = None,
        data: Any = None,
        headers: Mapping[str, str] | None = None,
    ) -> TransportResponse:
        return client.request(method, url, params=params, json=json, data=data, headers=headers)

    return _call


def parse_datetime(value: str) -> datetime:
    """Parse a vendor ISO-8601 timestamp, `Z`-suffixed or with a numeric
    offset (Salesforce's `+0000`, no colon, included) into an aware
    `datetime`. Python 3.11+'s `datetime.fromisoformat` already accepts
    most of these directly; the `Z` replacement covers the rest."""
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def source_day(value: datetime) -> date:
    """The calendar day a vendor timestamp falls on in the deployment's
    zone (`PORTOLAN_TIMEZONE`), the day `claims.recorded` and every weekly
    window read back. A naive value (a date-only export column) is already
    a calendar day and is kept as is."""
    if value.tzinfo is None:
        return value.date()
    return clock.world_date(value)


def format_ms_offset(milliseconds: int) -> str:
    """`123456` -> `"00:02:03"`: milliseconds from the start of a call
    into the `[HH:MM:SS]` prefix `portolan/ingest/prepare.py` indexes as
    a call locator."""
    total_seconds = int(milliseconds) // 1000
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
