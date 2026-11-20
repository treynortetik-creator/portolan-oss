"""`scripts/bench/luna_eval.py`: the spend cap, the batch record/replay path,
and the label matching. No network and no model call."""

from __future__ import annotations

import importlib.util
import json
import sys
from typing import ClassVar

import pytest

from portolan.extract.schema import SourceEnvelope
from portolan.metering import InMemorySink
from portolan.training.spend import SpendMeter
from tests._paths import require_repo_path

_spec = importlib.util.spec_from_file_location(
    "luna_eval", require_repo_path("scripts", "bench", "luna_eval.py")
)
luna_eval = importlib.util.module_from_spec(_spec)
sys.modules["luna_eval"] = luna_eval  # dataclasses resolve their module by name
_spec.loader.exec_module(luna_eval)


def _envelope(source_id: str = "S-1") -> SourceEnvelope:
    return SourceEnvelope.model_validate(
        {
            "source_id": source_id,
            "source_type": "ticket",
            "source_date": "2026-08-14",
            "text": "[L1] the pump alarm failed twice this week",
        }
    )


def _answer(source_id: str = "S-1", **fact_overrides) -> dict:
    fact = {
        "id": "f1",
        "type": "complaint",
        "claim": "The pump alarm failed twice.",
        "entities": [],
        "locator": "L1",
        "quote": "the pump alarm failed twice this week",
        "sentiment": -1,
        "confidence": 0.8,
    }
    fact.update(fact_overrides)
    return {
        "source_id": source_id,
        "summary": "Alarm failures.",
        "facts": [fact],
        "coverage": {"segments_read": 1, "segments_total": 1},
    }


def test_budget_refuses_a_call_that_could_pass_the_cap(tmp_path) -> None:
    meter = SpendMeter(tmp_path / "spend.json")
    meter.record(model="m", corpus=luna_eval.SPEND_KEY, amount_usd=3.90)
    budget = luna_eval.Budget(meter, 4.00)
    held = budget.reserve(0.05)
    with pytest.raises(luna_eval.BudgetExceeded):
        budget.reserve(0.06)  # 3.90 + 0.05 in flight + 0.06 > 4.00
    budget.release(held)
    assert budget.reserve(0.09) == pytest.approx(0.09)


def test_budget_counts_only_its_own_key(tmp_path) -> None:
    meter = SpendMeter(tmp_path / "spend.json")
    meter.record(model="m", corpus="other_key", amount_usd=100.0)
    assert luna_eval.Budget(meter, 1.0).reserve(0.5) == pytest.approx(0.5)


def test_prices_worst_case_uses_the_output_cap() -> None:
    prices = luna_eval.Prices(prompt=1e-7, completion=5e-7, cache_read=1e-8)
    assert prices.worst_case(3000, 32000) == pytest.approx(1000 * 1e-7 + 32000 * 5e-7)


def test_batch_price_cost_reprices_the_tokens(monkeypatch) -> None:
    monkeypatch.setitem(luna_eval._price_cache, "x/m:batch", luna_eval.Prices(5e-8, 2.5e-7, 5e-9))
    calls = [{"input_tokens": 1000, "cached_tokens": 200, "output_tokens": 400}]
    expected = 800 * 5e-8 + 200 * 5e-9 + 400 * 2.5e-7
    assert luna_eval.batch_price_cost("x/m", 0.02, calls) == pytest.approx(expected)
    assert luna_eval.batch_price_cost("x/m:batch", 0.02, calls) == pytest.approx(expected)
    assert luna_eval.batch_price_cost("x/m", None, calls) is None


class _FakeInner:
    """Just enough of `OpenRouterClient` for `GuardedClient`."""

    default_model = "x/m"
    reasoning: ClassVar[dict[str, str]] = {"effort": "max"}
    service_tier = None
    tenant = "t"

    def __init__(self) -> None:
        self.sink = InMemorySink()
        self.calls = 0

    def using_sink(self, sink, tenant):
        from contextlib import contextmanager

        @contextmanager
        def scope():
            previous, self.sink = self.sink, sink
            try:
                yield
            finally:
                self.sink = previous

        return scope()

    def complete(self, **kwargs):
        from portolan.llm.base import LLMResult
        from portolan.metering import UsageEvent

        self.calls += 1
        self.sink.record(
            UsageEvent("t", "extract", "x/m", 10, 20, 0, 0.003, "r", "2026-09-23T00:00:00Z")
        )
        return LLMResult(
            text="{}", data={}, input_tokens=10, output_tokens=20, cached_tokens=0, model="x/m"
        )


def test_guarded_client_caches_finished_calls_across_runs(tmp_path) -> None:
    meter = SpendMeter(tmp_path / "spend.json")
    budget = luna_eval.Budget(meter, 1.0)
    prices = luna_eval.Prices(1e-7, 5e-7, 1e-8)
    cache_path = tmp_path / "cache.jsonl"
    inner = _FakeInner()
    run_sink = InMemorySink()
    inner.sink = run_sink
    guarded = luna_eval.GuardedClient(inner, budget, prices, luna_eval.ResponseCache(cache_path))
    kwargs = {"system": "s", "user": "u", "surface": "extract", "max_tokens": 100}
    guarded.complete(**kwargs)
    assert inner.calls == 1 and len(run_sink.events) == 1  # the call's cost reaches the run

    # A new run (new cache object from the same file) replays instead of calling.
    inner2 = _FakeInner()
    replay_sink = InMemorySink()
    inner2.sink = replay_sink
    again = luna_eval.GuardedClient(inner2, budget, prices, luna_eval.ResponseCache(cache_path))
    result = again.complete(**kwargs)
    assert inner2.calls == 0
    assert result.output_tokens == 20
    assert replay_sink.events[0].cost_estimate == pytest.approx(0.003)
    assert replay_sink.events[0].extra["cache_replay"] is True
    again.complete(**{**kwargs, "user": "different"})
    assert inner2.calls == 1


def test_recording_then_replay_returns_the_batch_answer() -> None:
    recorder = luna_eval.RecordingBackend()
    envelope = _envelope()
    empty = recorder.extract(envelope, "system", InMemorySink(), "t")
    assert empty.facts == []
    (request,) = recorder.requests.values()
    assert request.source_id == "S-1"
    assert request.custom_id == luna_eval.request_key("system", request.user)

    sink = InMemorySink()
    replay = luna_eval.ReplayBackend(
        {request.custom_id: json.dumps(_answer())},
        {request.custom_id: {"prompt_tokens": 50, "completion_tokens": 70}},
        model="openai/gpt-6-luna-pro:batch",
    )
    result = replay.extract(envelope, "system", sink, "t")
    assert [f.claim for f in result.facts] == ["The pump alarm failed twice."]
    assert sink.events[0].output_tokens == 70
    assert replay.strict_failures == []


def test_replay_salvages_an_invalid_answer_with_parse_lenient() -> None:
    envelope = _envelope()
    key = luna_eval.request_key("system", luna_eval.build_user_message(envelope))
    bad = _answer(confidence=7.0)  # out of range: strict validation fails
    replay = luna_eval.ReplayBackend({key: json.dumps(bad)}, {}, model="m:batch")
    result = replay.extract(envelope, "system", InMemorySink(), "t")
    assert replay.strict_failures == ["S-1"]
    assert result.source_id == "S-1"


def test_replay_raises_when_the_batch_has_no_answer() -> None:
    replay = luna_eval.ReplayBackend({}, {}, model="m:batch")
    with pytest.raises(ValueError, match="no batch answer"):
        replay.extract(_envelope(), "system", InMemorySink(), "t")


TEXT = "Hello. The alarm failed twice this week. We will switch vendors next year. Thanks!"


def _verdict(eval_id: str, quote: str, verdict: str, ok_types: list[str]) -> dict:
    start = TEXT.index(quote)
    return {
        "eval_id": eval_id,
        "quote": quote,
        "quote_span": [start, start + len(quote)],
        "verdict": verdict,
        "ok_types": ok_types,
    }


def test_match_label_facts_by_quote_overlap() -> None:
    verdicts = [
        _verdict("v1", "The alarm failed twice this week", "correct", ["complaint"]),
        _verdict("v2", "Thanks!", "not_a_fact", ["praise"]),
    ]
    produced = [
        {"fact_type": "risk", "claim": "c", "quote": "alarm failed twice"},
        {"fact_type": "praise", "claim": "c", "quote": "Thanks"},
        {"fact_type": "risk", "claim": "c", "quote": "We will switch vendors next year"},
        {"fact_type": "risk", "claim": "c", "quote": "not in the text"},
    ]
    matched = luna_eval.match_label_facts(produced, verdicts, TEXT)
    assert [m["eval_id"] for m in matched] == ["v1", "v2", None, None]
    assert matched[0]["type_ok"] is False
    assert matched[1]["verdict"] == "not_a_fact"
    assert matched[3]["span"] is None


def test_span_folds_curly_quotes_dashes_and_spaces() -> None:
    text = "Cariel:  we’ve got T2 running—no  hiccups."
    span = luna_eval._span(text, "we've got T2 running-no hiccups", "")
    assert span is not None
    assert text[span[0] : span[1]] == "we’ve got T2 running—no  hiccups"


def test_pooling_counts_a_by_eye_fact_once_across_models() -> None:
    refs = {"S-1": [_verdict("v1", "The alarm failed twice this week", "correct", ["complaint"])]}
    texts = {"S-1": TEXT}
    fact = {"fact_type": "decision", "claim": "c", "quote": "We will switch vendors next year"}
    runs = {
        "a": [{"source_key": "S-1", "facts": [dict(fact)]}],
        "b": [{"source_key": "S-1", "facts": [dict(fact)]}],
    }
    luna_eval._rematch(runs, refs, texts)
    extra = {"a": {("S-1", 0): {"verdict": "correct", "ok_types": ["risk"]}}}
    assert luna_eval.pool_extra_verdicts(runs, refs, extra) == 1
    luna_eval._rematch(runs, refs, texts)
    all_refs = [r for rows in refs.values() for r in rows]
    a = luna_eval.label_metrics(runs["a"], all_refs, {})
    b = luna_eval.label_metrics(runs["b"], all_refs, {})
    assert a["real_facts"] == b["real_facts"] == 2  # v1 plus the pooled fact
    assert a["real_found"] == b["real_found"] == 1  # model b's copy matches without a reading
    assert a["on_real_wrong_type"] == 1  # "decision" is not an acceptable type


def test_same_as_maps_a_restated_fact_to_the_reference() -> None:
    refs = {"S-1": [_verdict("v1", "The alarm failed twice this week", "correct", ["complaint"])]}
    runs = {"a": [{"source_key": "S-1", "facts": [
        {"fact_type": "complaint", "claim": "c", "quote": "We will switch vendors next year"}
    ]}]}  # fmt: skip
    luna_eval._rematch(runs, refs, {"S-1": TEXT})
    extra = {"a": {("S-1", 0): {"verdict": "correct", "same_as": "v1"}}}
    luna_eval.pool_extra_verdicts(runs, refs, extra)
    luna_eval._rematch(runs, refs, {"S-1": TEXT})
    all_refs = [r for rows in refs.values() for r in rows]
    m = luna_eval.label_metrics(runs["a"], all_refs, {})
    assert m["real_facts"] == 1 and m["real_found"] == 1 and m["on_real_ok_type"] == 1


def test_label_metrics_counts_wrong_real_and_unverdicted() -> None:
    verdicts = [
        {"eval_id": "v1", "verdict": "correct"},
        {"eval_id": "v2", "verdict": "not_a_fact"},
        {"eval_id": "v3", "verdict": "wrong_type"},
    ]
    rows = [
        {
            "source_key": "S-1",
            "facts": [
                {"eval_id": "v1", "verdict": "correct", "type_ok": True},
                {"eval_id": "v2", "verdict": "not_a_fact", "type_ok": True},
                {"eval_id": None, "verdict": None, "type_ok": None, "span": [0, 5]},
                {"eval_id": None, "verdict": None, "type_ok": None, "span": [6, 9]},
                {"eval_id": None, "verdict": None, "type_ok": None, "span": None},
            ],
        }
    ]
    extra = {("S-1", 2): {"verdict": "correct"}, ("S-1", 3): {"verdict": "not_a_fact"}}
    m = luna_eval.label_metrics(rows, verdicts, extra)
    assert m["produced"] == 5
    assert m["on_human_wrong"] == 1
    assert m["on_real_ok_type"] == 1
    assert m["unverdicted_correct"] == 1 and m["unverdicted_wrong"] == 1
    assert m["quote_not_in_text"] == 1  # the gate would drop it; counts against precision
    assert m["precision"] == pytest.approx(2 / 5)
    assert m["real_facts"] == 2
    assert m["real_found"] == 1
    assert m["recall_of_real"] == pytest.approx(0.5)
