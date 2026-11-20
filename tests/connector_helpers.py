"""Shared fake `Transport` and response helpers for
`tests/test_connectors*.py` (build-contracts.md "Tests must not hit the
network. Use fakes."). Not a `test_*.py` module on purpose — pytest must
not collect it as a test file; it is imported by the connector test
files instead, the same pattern as `tests/ingest_helpers.py`.
"""

from __future__ import annotations

import json as json_module
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

FIXTURES_ROOT = Path(__file__).resolve().parent / "fixtures" / "connectors"


@dataclass
class FakeResponse:
    """Duck-types `portolan.connectors.base.TransportResponse`."""

    status_code: int = 200
    _json: Any = None
    text: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    content: bytes = b""

    def json(self) -> Any:
        return self._json


def json_response(
    data: Any, *, status_code: int = 200, headers: dict[str, str] | None = None
) -> FakeResponse:
    return FakeResponse(
        status_code=status_code,
        _json=data,
        text=json_module.dumps(data),
        headers=headers or {},
    )


def text_response(
    text: str, *, status_code: int = 200, headers: dict[str, str] | None = None
) -> FakeResponse:
    return FakeResponse(
        status_code=status_code,
        text=text,
        content=text.encode("utf-8"),
        headers=headers or {},
    )


def load_json_fixture(*parts: str) -> Any:
    return json_module.loads(FIXTURES_ROOT.joinpath(*parts).read_text(encoding="utf-8"))


def load_text_fixture(*parts: str) -> str:
    return FIXTURES_ROOT.joinpath(*parts).read_text(encoding="utf-8")


class FakeTransport:
    """Records every call made through it (`self.calls`) and serves
    queued `FakeResponse`s FIFO per exact URL — or a registered dynamic
    handler, for a response that must vary with `params`/`json`.

    A URL queued with more than one response is consumed in order (the
    pagination and retry tests rely on this); a URL queued with exactly
    one response keeps returning that same response to every subsequent
    call to it (so tests that don't care about a call's repeat count
    don't have to over-queue).
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self._queues: dict[str, list[FakeResponse]] = {}
        self._handlers: dict[str, Callable[..., FakeResponse]] = {}

    def queue(self, url: str, *responses: FakeResponse) -> None:
        self._queues.setdefault(url, []).extend(responses)

    def on(self, url: str, handler: Callable[..., FakeResponse]) -> None:
        self._handlers[url] = handler

    def __call__(
        self,
        method: str,
        url: str,
        *,
        params: Any = None,
        json: Any = None,
        data: Any = None,
        headers: Any = None,
    ) -> FakeResponse:
        self.calls.append(
            {
                "method": method,
                "url": url,
                "params": params,
                "json": json,
                "data": data,
                "headers": headers,
            }
        )
        if url in self._handlers:
            return self._handlers[url](
                method=method, url=url, params=params, json=json, data=data, headers=headers
            )
        queue = self._queues.get(url)
        if not queue:
            raise AssertionError(f"FakeTransport: no response queued for {method} {url}")
        return queue.pop(0) if len(queue) > 1 else queue[0]


def sleep_recorder() -> tuple[Callable[[float], None], list[float]]:
    """A `sleep` stand-in for `request_with_retry`'s `sleep=` param: never
    actually sleeps, just records how long it was asked to wait."""
    waits: list[float] = []

    def _sleep(seconds: float) -> None:
        waits.append(seconds)

    return _sleep, waits
