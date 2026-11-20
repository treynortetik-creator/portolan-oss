"""Tests for `portolan.training.pipeline.run_corpus_judged` -- the
paid-teacher-plus-typed-judge orchestration (coordinator course-
correction, 2026-09-21). No network: the paid teacher is a
`FakeBackend`, the judge is a `JevClient` over `httpx.MockTransport`
(the same seams `test_training_pipeline.py`/`test_training_typed_judge.py`
already use), so `sink.events[i].cost_estimate` and `FactVerdict.
cost_usd()` are real numbers computed by the real pricing/cost code
paths, not hand-picked constants.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import httpx
import pytest

from portolan.extract.backends import FakeBackend
from portolan.extract.schema import Coverage, ExtractionResult, Fact, SourceType
from portolan.ingest.models import RawSource
from portolan.metering import InMemorySink
from portolan.training.pipeline import Document, run_corpus_judged
from portolan.training.records import failures_path, progress_path, read_records, records_path
from portolan.training.spend import SpendMeter, default_spend_path
from portolan.training.typed_judge import JevClient

PAID_MODEL = "mistralai/mistral-nemo"

_QUOTE = "the device broke during use"


def _doc(source_id: str) -> Document:
    raw = RawSource(
        source_id=source_id,
        source_type=SourceType.DOC,
        source_date=date(2026, 1, 1),
        participants=[],
        text=_QUOTE,
        metadata={},
    )
    return Document(source_id=source_id, raw=raw, kind="ticket")


def _result(source_id: str, facts: list[Fact]) -> ExtractionResult:
    return ExtractionResult(
        source_id=source_id,
        summary="",
        entities=[],
        facts=facts,
        actions=[],
        unknown_entities=[],
        coverage=Coverage(segments_read=1, segments_total=1),
    )


def _fact(fid: str, fact_type: str = "complaint") -> Fact:
    return Fact(
        id=fid,
        type=fact_type,
        claim="device broke",
        locator="L1",
        quote=_QUOTE,
        sentiment=-1,
        confidence=0.9,
    )


def _no_sleep(_seconds: float) -> None:
    pass


class _CountingLimiter:
    def acquire(self) -> None:
        pass


def _jev_response(faithful: float, fact_type: str, added_detail: float) -> dict:
    return {
        "answers": {
            "faithful": {"type": "noul", "noul": faithful},
            "fact_type": {"type": "choice", "choice": fact_type, "confidence": 0.8},
            "added_detail": {"type": "noul", "noul": added_detail},
        },
        "usage": {"input_tokens": 100, "output_tokens": 0},
    }


def _agreeing_jev_client() -> JevClient:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_jev_response(0.95, "complaint", 0.05))

    return JevClient(api_key="k", transport=httpx.MockTransport(handler), sleep=_no_sleep)


def _disagreeing_jev_client() -> JevClient:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_jev_response(0.95, "request", 0.05))

    return JevClient(api_key="k", transport=httpx.MockTransport(handler), sleep=_no_sleep)


def _run(
    tmp_path: Path,
    documents: list[Document],
    backend_factory,
    judges: dict,
    *,
    judge_mode: str = "jev",
    budget_usd: float = 100.0,
    max_docs: int | None = None,
):
    corpus = "fda_maude"
    return run_corpus_judged(
        corpus=corpus,
        kind="ticket",
        license_text="public domain",
        documents=documents,
        system_prompt="SYSTEM",
        prompt_variant="v6-type-precision",
        paid_model=PAID_MODEL,
        backend_factory=backend_factory,
        limiter=_CountingLimiter(),
        sink=InMemorySink(),
        tenant="test",
        records_file=records_path(tmp_path, corpus),
        failures_file=failures_path(tmp_path, corpus),
        progress_file=progress_path(tmp_path, corpus),
        judges=judges,
        judge_mode=judge_mode,
        spend=SpendMeter(path=default_spend_path(tmp_path)),
        budget_usd=budget_usd,
        max_docs=max_docs,
        sleep=_no_sleep,
        now=lambda: "2026-09-21T00:00:00-07:00",
    )


def test_kept_fact_when_judge_agrees(tmp_path: Path) -> None:
    backend = FakeBackend(
        canned=lambda envelope: _result("S-1", [_fact("f1")]), model_key="nova-lite"
    )
    stats = _run(tmp_path, [_doc("S-1")], lambda model: backend, {"jev": _agreeing_jev_client()})

    assert stats.facts_emitted == 1
    assert stats.facts_sent_to_judge == 1
    assert stats.facts_kept == 1
    assert stats.records_written == 1

    written = list(read_records(records_path(tmp_path, "fda_maude")))
    assert len(written) == 1
    record = written[0]
    assert record.mode == "judge"
    assert record.teachers == [PAID_MODEL]
    assert record.facts[0].fact_type == "complaint"
    assert record.facts[0].judge["jev"].faithful is True
    assert record.agreement == 1.0
    assert not record.is_negative


def test_dropped_fact_when_judge_disagrees_on_type(tmp_path: Path) -> None:
    backend = FakeBackend(
        canned=lambda envelope: _result("S-1", [_fact("f1")]), model_key="nova-lite"
    )
    stats = _run(tmp_path, [_doc("S-1")], lambda model: backend, {"jev": _disagreeing_jev_client()})

    assert stats.facts_sent_to_judge == 1
    assert stats.facts_kept == 0
    assert stats.facts_dropped_judge == 1

    written = list(read_records(records_path(tmp_path, "fda_maude")))
    assert len(written) == 1
    assert written[0].is_negative
    assert written[0].agreement == 0.0


def test_quote_gate_drops_fact_before_spending_a_judge_call(tmp_path: Path) -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json=_jev_response(0.95, "complaint", 0.05))

    judge = JevClient(api_key="k", transport=httpx.MockTransport(handler), sleep=_no_sleep)
    unsupported = Fact(
        id="f1",
        type="complaint",
        claim="x",
        locator="L1",
        quote="a sentence never present in the document",
        sentiment=0,
        confidence=0.5,
    )
    backend = FakeBackend(
        canned=lambda envelope: _result("S-1", [unsupported]), model_key="nova-lite"
    )
    stats = _run(tmp_path, [_doc("S-1")], lambda model: backend, {"jev": judge})

    assert stats.facts_dropped_quote_gate == 1
    assert stats.facts_sent_to_judge == 0
    assert calls["n"] == 0


def test_both_judge_mode_requires_both_to_agree(tmp_path: Path) -> None:
    backend = FakeBackend(
        canned=lambda envelope: _result("S-1", [_fact("f1")]), model_key="nova-lite"
    )
    judges = {"jev": _agreeing_jev_client(), "kev": _disagreeing_jev_client()}
    stats = _run(tmp_path, [_doc("S-1")], lambda model: backend, judges, judge_mode="both")
    assert stats.facts_kept == 0  # kev disagreed, "both" requires unanimous keep

    written = list(read_records(records_path(tmp_path, "fda_maude")))
    assert written[0].facts == []
    assert written[0].is_negative


def test_both_judge_mode_records_both_verdicts_when_kept(tmp_path: Path) -> None:
    backend = FakeBackend(
        canned=lambda envelope: _result("S-1", [_fact("f1")]), model_key="nova-lite"
    )
    judges = {"jev": _agreeing_jev_client(), "kev": _agreeing_jev_client()}
    stats = _run(tmp_path, [_doc("S-1")], lambda model: backend, judges, judge_mode="both")
    assert stats.facts_kept == 1

    written = list(read_records(records_path(tmp_path, "fda_maude")))
    fact = written[0].facts[0]
    assert set(fact.judge.keys()) == {"jev", "kev"}


def test_negative_when_extractor_emits_no_facts(tmp_path: Path) -> None:
    backend = FakeBackend(canned=lambda envelope: _result("S-1", []), model_key="nova-lite")
    stats = _run(tmp_path, [_doc("S-1")], lambda model: backend, {"jev": _agreeing_jev_client()})
    assert stats.facts_emitted == 0
    assert stats.records_written == 1
    written = list(read_records(records_path(tmp_path, "fda_maude")))
    assert written[0].is_negative
    assert written[0].agreement == 1.0  # nothing to disagree about


def test_spend_meter_tracks_extractor_and_judge_cost(tmp_path: Path) -> None:
    backend = FakeBackend(
        canned=lambda envelope: _result("S-1", [_fact("f1")]), model_key="nova-lite"
    )
    stats = _run(tmp_path, [_doc("S-1")], lambda model: backend, {"jev": _agreeing_jev_client()})
    assert stats.extractor_spend_usd > 0
    assert stats.judge_spend_usd > 0
    assert stats.total_spend_usd == pytest.approx(stats.extractor_spend_usd + stats.judge_spend_usd)

    snapshot = SpendMeter(path=default_spend_path(tmp_path)).snapshot()
    assert PAID_MODEL in snapshot["per_model_usd"]
    assert "judge:jev" in snapshot["per_model_usd"]


def test_run_stops_cleanly_when_budget_exhausted(tmp_path: Path) -> None:
    backend = FakeBackend(
        canned=lambda envelope: _result("S-1", [_fact("f1")]), model_key="nova-lite"
    )
    docs = [_doc("S-1"), _doc("S-2"), _doc("S-3")]
    # budget_usd=0.0 -- over_budget() is True even before the first
    # document, so nothing is processed and the run stops immediately.
    stats = _run(
        tmp_path, docs, lambda model: backend, {"jev": _agreeing_jev_client()}, budget_usd=0.0
    )
    assert stats.stopped_reason == "budget_usd"
    assert stats.documents_processed == 0
    assert stats.records_written == 0
