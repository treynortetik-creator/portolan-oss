"""`portolan.llm.openrouter_batch`: the OpenRouter Batch API client. No network:
every test injects an `httpx.MockTransport`. Shapes from
https://openrouter.ai/docs/batch-quickstart (read 2026-09-23)."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from portolan.llm.base import LLMError
from portolan.llm.openrouter_batch import (
    BATCHES_URL,
    BatchNotFinishedError,
    OpenRouterBatchClient,
    batch_results,
    chat_request_body,
)
from portolan.training.spend import SpendMeter


def _completed(batch_id: str = "batch_1", cost: float = 0.0123) -> dict[str, Any]:
    return {
        "id": batch_id,
        "object": "batch",
        "model": "openai/gpt-6-luna-pro:batch",
        "status": "completed",
        "request_counts": {"total": 2, "completed": 1, "failed": 1},
        "usage": {"prompt_tokens": 20, "completion_tokens": 40, "cost": cost},
        "results": [
            {
                "custom_id": "req-a",
                "response": {
                    "status_code": 200,
                    "body": {
                        "choices": [{"message": {"content": '{"ok": true}'}}],
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 20,
                            "completion_tokens_details": {"reasoning_tokens": 15},
                        },
                    },
                },
                "error": None,
            },
            {"custom_id": "req-b", "response": None, "error": {"message": "bad line"}},
        ],
        "error": None,
    }


def _client(handler, meter: SpendMeter | None = None, **kwargs: Any) -> OpenRouterBatchClient:
    return OpenRouterBatchClient(
        spend_key="test_batch",
        api_key="or-test-key",
        spend_meter=meter,
        transport=httpx.MockTransport(handler),
        sleep=kwargs.pop("sleep", lambda s: None),
        **kwargs,
    )


def test_submit_puts_requests_after_the_batch_fields() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["raw"] = request.content.decode()
        return httpx.Response(202, json={"id": "batch_1", "status": "validating"})

    body = chat_request_body(system="s", user="u", max_tokens=100)
    result = _client(handler).submit(
        "openai/gpt-6-luna-pro:batch", [{"custom_id": "req-a", "body": body}]
    )
    assert result["status"] == "validating"
    assert seen["url"] == BATCHES_URL
    payload = json.loads(seen["raw"])
    assert list(payload) == ["endpoint", "model", "completion_window", "requests"]
    assert payload["endpoint"] == "/v1/chat/completions"
    assert payload["completion_window"] == "24h"
    assert payload["requests"][0]["custom_id"] == "req-a"
    assert "model" not in payload["requests"][0]["body"]


def test_submit_refuses_a_non_batch_slug_and_duplicate_ids() -> None:
    client = _client(lambda r: httpx.Response(500))
    with pytest.raises(ValueError, match=":batch"):
        client.submit("openai/gpt-6-luna-pro", [{"custom_id": "a", "body": {}}])
    with pytest.raises(ValueError, match="unique"):
        client.submit(
            "openai/gpt-6-luna-pro:batch",
            [{"custom_id": "a", "body": {}}, {"custom_id": "a", "body": {}}],
        )


def test_chat_request_body_sends_reasoning_and_a_strict_schema() -> None:
    schema = {"type": "object", "properties": {}}
    body = chat_request_body(
        system="s", user="u", max_tokens=32000, json_schema=schema, reasoning={"effort": "max"}
    )
    assert body["reasoning"] == {"effort": "max"}
    assert body["max_tokens"] == 32000
    assert body["response_format"]["json_schema"]["strict"] is True
    assert body["response_format"]["json_schema"]["schema"] == schema
    assert chat_request_body(system="s", user="u", max_tokens=10)["reasoning"] == {"enabled": False}


def test_wait_polls_to_completion_and_meters_the_cost_once(tmp_path) -> None:
    statuses = iter(["validating", "in_progress"])
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        status = next(statuses, None)
        if status is None:
            return httpx.Response(200, json=_completed())
        return httpx.Response(200, json={"id": "batch_1", "status": status, "results": None})

    meter = SpendMeter(tmp_path / "spend.json")
    client = _client(handler, meter)
    batch = client.wait("batch_1", poll_seconds=0)
    assert batch["status"] == "completed"
    assert calls["n"] == 3
    client.get("batch_1")  # a second read of the same finished batch
    assert meter.corpus_usd("test_batch") == pytest.approx(0.0123)


def test_wait_retries_a_404_right_after_submit() -> None:
    answers = iter([httpx.Response(404, json={"error": {"message": "not found"}})])

    def handler(request: httpx.Request) -> httpx.Response:
        return next(answers, None) or httpx.Response(200, json=_completed())

    batch = _client(handler, clock=lambda: 0.0).wait("batch_1", poll_seconds=0)
    assert batch["status"] == "completed"


def test_wait_raises_a_404_that_outlasts_the_grace() -> None:
    ticks = iter([0.0, 400.0])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": {"message": "not found"}})

    with pytest.raises(LLMError, match="404"):
        _client(handler, clock=lambda: next(ticks)).wait("batch_1", poll_seconds=0)


def test_wait_gives_up_after_the_timeout() -> None:
    ticks = iter([0.0, 5.0, 11.0])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": "batch_1", "status": "in_progress"})

    client = _client(handler, clock=lambda: next(ticks))
    with pytest.raises(BatchNotFinishedError, match="in_progress"):
        client.wait("batch_1", poll_seconds=0, timeout_seconds=10)


def test_batch_results_maps_answers_and_errors_by_custom_id() -> None:
    results = batch_results(_completed())
    assert results["req-a"].text == '{"ok": true}'
    assert results["req-a"].usage["completion_tokens_details"]["reasoning_tokens"] == 15
    assert results["req-b"].body is None
    assert results["req-b"].error == {"message": "bad line"}
    assert results["req-b"].text == ""


def test_http_error_raises_without_the_key() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "no :batch endpoint"}})

    with pytest.raises(LLMError) as info:
        _client(handler).get("batch_1")
    assert "400" in str(info.value)
    assert "or-test-key" not in str(info.value)


def test_client_needs_a_spend_key() -> None:
    with pytest.raises(ValueError, match="spend_key"):
        OpenRouterBatchClient(spend_key=" ", api_key="k")
