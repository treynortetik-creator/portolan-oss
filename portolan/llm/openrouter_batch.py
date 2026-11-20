"""OpenRouter Batch API client: submit many chat-completion requests at once,
poll, read the results back. Half the per-token price of the sync API for
models with a `:batch` endpoint, with a 24-hour completion window.

Contract, per https://openrouter.ai/docs/batch-quickstart (read 2026-09-23):

- `POST /api/v1/batches` with `endpoint`, `model`, optional `provider` and
  `completion_window` ("24h", the only value), then `requests`: a list of
  `{custom_id, body}`. The API stream-parses the body and returns 400 if
  `requests` comes before the other fields, so the body is serialized here in
  that order. Answers `202` with `status: "validating"`.
- `GET /api/v1/batches/{id}`: status `validating -> in_progress -> finalizing
  -> completed`, or `failed`, `expired`, `cancelled`. `results` is inline on
  the completed batch (no download endpoint), one item per `custom_id` with
  exactly one of `response` (`status_code`, `body`) or `error`.
- The batch-level `model` must be a `:batch` slug; a request body may omit
  `model`. `stream`, `speed` and per-line reasoning-effort updates are rejected.
- Billing: the completed batch's `usage.cost`, recorded once per batch on the
  spend meter under `spend_key`.

Tests inject `transport=httpx.MockTransport(...)`, `sleep` and `clock`.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import httpx

from portolan.llm.base import LLMError

if TYPE_CHECKING:
    from portolan.training.spend import SpendMeter

BATCHES_URL = "https://openrouter.ai/api/v1/batches"
CHAT_ENDPOINT = "/v1/chat/completions"
COMPLETION_WINDOW = "24h"
TERMINAL_STATUSES = frozenset({"completed", "failed", "expired", "cancelled"})
BATCH_SUFFIX = ":batch"

_DEFAULT_TIMEOUT = httpx.Timeout(30.0, read=300.0)


class BatchNotFinishedError(LLMError):
    """`wait()` gave up before the batch reached a terminal status."""


@dataclass(frozen=True)
class BatchResult:
    """One request's outcome: the chat-completion body, or an error."""

    custom_id: str
    body: dict[str, Any] | None
    error: Any = None

    @property
    def text(self) -> str:
        if not self.body:
            return ""
        choices = self.body.get("choices") or []
        if not choices:
            return ""
        content = (choices[0].get("message") or {}).get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(p.get("text", "") for p in content if isinstance(p, dict))
        return ""

    @property
    def usage(self) -> dict[str, Any]:
        return dict((self.body or {}).get("usage") or {})


def chat_request_body(
    *,
    system: str,
    user: str,
    max_tokens: int,
    json_schema: dict[str, Any] | None = None,
    reasoning: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One `/v1/chat/completions` body for a batch line: the same fields the
    sync client sends (`OpenRouterClient._base_body` plus a strict
    `json_schema` response format), minus `model`, which the batch sets."""
    body: dict[str, Any] = {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "reasoning": dict(reasoning) if reasoning is not None else {"enabled": False},
    }
    if json_schema is not None:
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "portolan_response", "strict": True, "schema": json_schema},
        }
    return body


class OpenRouterBatchClient:
    """Submit, poll and read OpenRouter batches. `spend_key` is required, as on
    `OpenRouterClient` (every paid call is metered under a named key)."""

    def __init__(
        self,
        *,
        spend_key: str,
        api_key: str | None,
        spend_meter: SpendMeter | None = None,
        transport: httpx.BaseTransport | None = None,
        timeout: float | httpx.Timeout = _DEFAULT_TIMEOUT,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not spend_key or not spend_key.strip():
            raise ValueError("OpenRouterBatchClient needs a spend_key naming what it pays for")
        self.spend_key = spend_key
        self.spend_meter = spend_meter
        self._api_key = api_key
        self._transport = transport
        self._timeout = timeout
        self._sleep = sleep
        self._clock = clock
        self._metered: set[str] = set()
        self._lock = threading.Lock()

    def _headers(self) -> dict[str, str]:
        if not self._api_key:
            raise RuntimeError("OpenRouterBatchClient has no API key (OPENROUTER_API_KEY)")
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "Portolan",
            "X-Title": "Portolan",
        }

    def _request(self, method: str, url: str, content: str | None = None) -> dict[str, Any]:
        with httpx.Client(transport=self._transport, timeout=self._timeout) as client:
            response = client.request(method, url, headers=self._headers(), content=content)
        if response.status_code >= 400:
            # Response text only; the request (and its key) is never echoed.
            raise LLMError(
                f"OpenRouterBatchClient: {method} {url} failed with status "
                f"{response.status_code}: {response.text[:2000]}",
                status_code=response.status_code,
                headers=dict(response.headers),
            )
        return response.json()

    def submit(
        self,
        model: str,
        requests: list[dict[str, Any]],
        *,
        endpoint: str = CHAT_ENDPOINT,
        provider_only: list[str] | None = None,
    ) -> dict[str, Any]:
        """Submit `requests` (`{custom_id, body}` items) as one batch on
        `model`, which must be a `:batch` slug. Returns the batch object."""
        if not model.endswith(BATCH_SUFFIX):
            raise ValueError(f"batch model must be a {BATCH_SUFFIX!r} slug, got {model!r}")
        if not requests:
            raise ValueError("a batch needs at least one request")
        ids = [r["custom_id"] for r in requests]
        if len(set(ids)) != len(ids):
            raise ValueError("custom_id must be unique within a batch")
        # Field order matters: `requests` last (the API stream-parses the body).
        payload: dict[str, Any] = {"endpoint": endpoint, "model": model}
        if provider_only:
            payload["provider"] = {"only": list(provider_only)}
        payload["completion_window"] = COMPLETION_WINDOW
        payload["requests"] = [{"custom_id": r["custom_id"], "body": r["body"]} for r in requests]
        return self._request("POST", BATCHES_URL, content=json.dumps(payload))

    def get(self, batch_id: str) -> dict[str, Any]:
        batch = self._request("GET", f"{BATCHES_URL}/{batch_id}")
        self._meter(batch)
        return batch

    def wait(
        self,
        batch_id: str,
        *,
        poll_seconds: float = 15.0,
        timeout_seconds: float = 24 * 3600.0,
        on_poll: Callable[[dict[str, Any]], None] | None = None,
        not_found_grace_seconds: float = 300.0,
    ) -> dict[str, Any]:
        """Poll until the batch is terminal; `BatchNotFinishedError` after
        `timeout_seconds` (the batch keeps running server-side). A new batch
        can answer 404 for a short while after its 202,
        so 404 is retried for `not_found_grace_seconds`."""
        start = self._clock()
        while True:
            try:
                batch = self.get(batch_id)
            except LLMError as exc:
                if exc.status_code != 404 or self._clock() - start >= not_found_grace_seconds:
                    raise
                self._sleep(poll_seconds)
                continue
            if on_poll is not None:
                on_poll(batch)
            if batch.get("status") in TERMINAL_STATUSES:
                return batch
            if self._clock() - start >= timeout_seconds:
                raise BatchNotFinishedError(
                    f"batch {batch_id} still {batch.get('status')!r} after {timeout_seconds:.0f}s"
                )
            self._sleep(poll_seconds)

    def _meter(self, batch: dict[str, Any]) -> None:
        """Record a finished batch's `usage.cost` once, whatever its status."""
        batch_id = str(batch.get("id") or "")
        if batch.get("status") not in TERMINAL_STATUSES or self.spend_meter is None:
            return
        cost = (batch.get("usage") or {}).get("cost")
        if not isinstance(cost, int | float) or cost <= 0:
            return
        with self._lock:
            if batch_id in self._metered:
                return
            self._metered.add(batch_id)
        self.spend_meter.record(
            model=str(batch.get("model") or "unknown"), corpus=self.spend_key, amount_usd=cost
        )


def batch_results(batch: dict[str, Any]) -> dict[str, BatchResult]:
    """`{custom_id: BatchResult}` from a completed batch object."""
    out: dict[str, BatchResult] = {}
    for item in batch.get("results") or []:
        custom_id = str(item.get("custom_id"))
        response = item.get("response") or None
        error = item.get("error")
        body = None
        if isinstance(response, dict):
            status = response.get("status_code")
            if status is None or int(status) < 400:
                body = response.get("body")
            elif error is None:
                error = response.get("body") or {"status_code": status}
        out[custom_id] = BatchResult(custom_id=custom_id, body=body, error=error)
    return out
