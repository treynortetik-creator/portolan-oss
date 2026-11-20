"""Tests for `portolan.training.typed_judge` (coordinator course-
correction, 2026-09-21, item 2: the typed-judge keep gate). No real
network: `JevClient` is exercised through `httpx.MockTransport`, the
same seam `portolan.llm.openrouter`'s own tests use.
"""

from __future__ import annotations

import httpx
import pytest

from portolan.extract.schema import FactType
from portolan.training.typed_judge import (
    FactVerdict,
    JevClient,
    JevError,
    ask_typed_questions,
    evaluate_fact,
    fact_type_criteria,
    parse_verdict,
)

_DOC_TEXT = "[L1] the pump beeped three times and then shut off completely\n"
_QUOTE = "the pump beeped three times and then shut off completely"


def _jev_response(
    *,
    faithful: float = 0.9,
    fact_type: str = "complaint",
    type_confidence: float = 0.8,
    added_detail: float = 0.1,
    input_tokens: int = 100,
) -> dict:
    return {
        "answers": {
            "faithful": {"type": "noul", "noul": faithful},
            "fact_type": {
                "type": "choice",
                "choice": fact_type,
                "confidence": type_confidence,
                "probabilities": {fact_type: type_confidence},
            },
            "added_detail": {"type": "noul", "noul": added_detail},
        },
        "usage": {"input_tokens": input_tokens, "output_tokens": 0},
    }


def _client(handler) -> JevClient:
    return JevClient(
        api_key="test-key", transport=httpx.MockTransport(handler), sleep=lambda s: None
    )


# --------------------------------------------------------------------------
# fact_type_criteria
# --------------------------------------------------------------------------


def test_fact_type_criteria_covers_every_production_type() -> None:
    criteria = fact_type_criteria()
    assert set(criteria.keys()) == {ft.value for ft in FactType}
    for definition in criteria.values():
        assert isinstance(definition, str) and definition


# --------------------------------------------------------------------------
# parse_verdict / FactVerdict.keep
# --------------------------------------------------------------------------


def test_parse_verdict_reads_answers_and_usage() -> None:
    verdict = parse_verdict(_jev_response())
    assert verdict.faithful is True
    assert verdict.faithful_probability == 0.9
    assert verdict.predicted_type == "complaint"
    assert verdict.type_confidence == 0.8
    assert verdict.added_detail is False
    assert verdict.input_tokens == 100


def test_verdict_keep_true_when_faithful_type_matches_no_added_detail() -> None:
    verdict = parse_verdict(_jev_response(faithful=0.9, fact_type="complaint", added_detail=0.1))
    assert verdict.keep("complaint") is True


def test_verdict_keep_false_when_not_faithful() -> None:
    verdict = parse_verdict(_jev_response(faithful=0.2, fact_type="complaint", added_detail=0.1))
    assert verdict.keep("complaint") is False


def test_verdict_keep_false_when_type_mismatch() -> None:
    verdict = parse_verdict(_jev_response(faithful=0.9, fact_type="request", added_detail=0.1))
    assert verdict.keep("complaint") is False


def test_verdict_keep_false_when_added_detail() -> None:
    verdict = parse_verdict(_jev_response(faithful=0.9, fact_type="complaint", added_detail=0.9))
    assert verdict.keep("complaint") is False


def test_verdict_cost_usd_input_tokens_only() -> None:
    verdict = parse_verdict(_jev_response(input_tokens=1_000_000))
    assert verdict.cost_usd() == pytest.approx(0.042)


def test_verdict_to_record_dict_shape() -> None:
    verdict = parse_verdict(_jev_response())
    d = verdict.to_record_dict()
    assert set(d.keys()) == {
        "faithful",
        "faithful_probability",
        "predicted_type",
        "type_confidence",
        "added_detail",
        "added_detail_probability",
    }


# --------------------------------------------------------------------------
# JevClient / ask_typed_questions
# --------------------------------------------------------------------------


def test_ask_typed_questions_posts_state_and_batched_questions() -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json as _json

        captured["body"] = _json.loads(request.content)
        return httpx.Response(200, json=_jev_response())

    client = _client(handler)
    verdict = ask_typed_questions(client, "the device broke", _QUOTE)
    assert isinstance(verdict, FactVerdict)
    body = captured["body"]
    assert body["model"] == "jev-latest"
    assert set(body["questions"].keys()) == {"faithful", "fact_type", "added_detail"}
    assert "the device broke" in body["state"]
    assert _QUOTE in body["state"]


def test_jev_client_retries_on_429_then_succeeds() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, text="rate limited")
        return httpx.Response(200, json=_jev_response())

    client = _client(handler)
    verdict = ask_typed_questions(client, "claim", "quote")
    assert calls["n"] == 2
    assert isinstance(verdict, FactVerdict)


def test_jev_client_raises_jev_error_after_exhausting_retries() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="still limited")

    client = _client(handler)
    with pytest.raises(JevError):
        ask_typed_questions(client, "claim", "quote")


def test_jev_client_raises_on_non_retryable_status() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="bad key")

    client = _client(handler)
    with pytest.raises(JevError, match="401"):
        ask_typed_questions(client, "claim", "quote")


# --------------------------------------------------------------------------
# evaluate_fact
# --------------------------------------------------------------------------


def test_evaluate_fact_quote_gate_blocks_before_judge_is_called() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json=_jev_response())

    client = _client(handler)
    result = evaluate_fact(
        claim="the device broke",
        quote="a sentence never present in the document",
        emitted_type="complaint",
        doc_text=_DOC_TEXT,
        client=client,
    )
    assert result.keep is False
    assert result.verdict is None
    assert result.quote_gate_passed is False
    assert calls["n"] == 0  # never spent a Jev call on an unsupported quote


def test_evaluate_fact_kept_when_judge_agrees() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json=_jev_response(faithful=0.95, fact_type="complaint", added_detail=0.05)
        )

    client = _client(handler)
    result = evaluate_fact(
        claim="the pump beeped and shut off",
        quote=_QUOTE,
        emitted_type="complaint",
        doc_text=_DOC_TEXT,
        client=client,
    )
    assert result.keep is True
    assert result.quote_gate_passed is True
    assert result.verdict is not None


def test_evaluate_fact_dropped_when_judge_disagrees_on_type() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json=_jev_response(faithful=0.95, fact_type="request", added_detail=0.05)
        )

    client = _client(handler)
    result = evaluate_fact(
        claim="the pump beeped and shut off",
        quote=_QUOTE,
        emitted_type="complaint",
        doc_text=_DOC_TEXT,
        client=client,
    )
    assert result.keep is False
    assert result.verdict is not None
    assert result.verdict.predicted_type == "request"
