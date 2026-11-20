"""Tests for portolan.llm.fake.FakeLLMClient."""

from __future__ import annotations

import pytest

from portolan.llm.base import LLMResult
from portolan.llm.fake import FakeLLMClient
from portolan.metering import InMemorySink


def test_fake_llm_client_list_response_returns_text_and_meters_once() -> None:
    sink = InMemorySink()
    client = FakeLLMClient(["Hello there."], sink=sink, tenant="acme")

    result = client.complete(system="be terse", user="say hi", surface="chat")

    assert result.text == "Hello there."
    assert result.data is None
    assert len(sink.events) == 1
    event = sink.events[0]
    assert event.tenant == "acme"
    assert event.surface == "chat"
    assert event.model_key == "nova-lite"
    assert event.input_tokens > 0
    assert event.output_tokens > 0
    assert event.extra["backend"] == "fake"


def test_fake_llm_client_dict_response_populates_data_and_text() -> None:
    sink = InMemorySink()
    canned = {"greeting": "hi", "count": 3}
    client = FakeLLMClient([canned], sink=sink)

    result = client.complete(
        system="sys", user="usr", json_schema={"type": "object"}, surface="extract"
    )

    assert result.data == canned
    assert result.text is not None
    import json

    assert json.loads(result.text) == canned


def test_fake_llm_client_llmresult_response_passed_through() -> None:
    sink = InMemorySink()
    canned_result = LLMResult(
        text="raw text", data={"a": 1}, input_tokens=10, output_tokens=5, cached_tokens=2, model="x"
    )
    client = FakeLLMClient([canned_result], sink=sink)

    result = client.complete(system="sys", user="usr", surface="merge")

    assert result is canned_result
    assert len(sink.events) == 1
    assert sink.events[0].input_tokens == 10
    assert sink.events[0].output_tokens == 5
    assert sink.events[0].cached_tokens == 2


def test_fake_llm_client_list_consumed_in_call_order() -> None:
    client = FakeLLMClient(["first", "second"])
    r1 = client.complete(system="s", user="u", surface="chat")
    r2 = client.complete(system="s", user="u", surface="chat")
    assert r1.text == "first"
    assert r2.text == "second"


def test_fake_llm_client_list_exhausted_raises_index_error() -> None:
    client = FakeLLMClient([])
    with pytest.raises(IndexError):
        client.complete(system="s", user="u", surface="chat")


def test_fake_llm_client_callable_receives_request_kwargs() -> None:
    seen: dict = {}

    def responder(*, system, user, json_schema, model, surface):
        seen.update(system=system, user=user, json_schema=json_schema, model=model, surface=surface)
        return "ok"

    client = FakeLLMClient(responder)
    result = client.complete(
        system="the system",
        user="the user",
        json_schema={"type": "object"},
        model="claude-haiku-4-5",
        surface="audit",
    )

    assert result.text == "ok"
    assert seen == {
        "system": "the system",
        "user": "the user",
        "json_schema": {"type": "object"},
        "model": "claude-haiku-4-5",
        "surface": "audit",
    }


def test_fake_llm_client_model_override_used_as_metering_key() -> None:
    sink = InMemorySink()
    client = FakeLLMClient(["hi"], model_key="nova-lite", sink=sink)
    client.complete(system="s", user="u", model="claude-haiku-4-5", surface="chat")
    assert sink.events[0].model_key == "claude-haiku-4-5"
