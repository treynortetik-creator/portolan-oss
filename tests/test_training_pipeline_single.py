"""Tests for `portolan.training.pipeline.run_corpus_single` -- the
single-teacher, mechanical-overlap-gate orchestration (coordinator
course-correction, 2026-09-21, final design). No network: the teacher is
a `FakeBackend`, the informational judge is a `JevClient` over
`httpx.MockTransport`.
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
from portolan.training.pipeline import Document, run_corpus_single
from portolan.training.records import (
    dropped_path,
    failures_path,
    progress_path,
    read_dropped,
    read_records,
    records_path,
)
from portolan.training.spend import SpendMeter, default_spend_path
from portolan.training.typed_judge import JevClient

TEACHER = "deepseek/deepseek-v3.2"
_DOC_TEXT = "the pump beeped three times and shut off completely during the site visit"


def _doc(source_id: str, text: str = _DOC_TEXT) -> Document:
    raw = RawSource(
        source_id=source_id,
        source_type=SourceType.DOC,
        source_date=date(2026, 1, 1),
        participants=[],
        text=text,
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


def _fact(fid: str, claim: str, quote: str, fact_type: str = "complaint") -> Fact:
    return Fact(
        id=fid,
        type=fact_type,
        claim=claim,
        locator="L1",
        quote=quote,
        sentiment=-1,
        confidence=0.9,
    )


def _no_sleep(_seconds: float) -> None:
    pass


class _CountingLimiter:
    def acquire(self) -> None:
        pass


def _jev_response(
    faithful: float = 0.9, fact_type: str = "complaint", added_detail: float = 0.1
) -> dict:
    return {
        "answers": {
            "faithful": {"type": "noul", "noul": faithful},
            "fact_type": {"type": "choice", "choice": fact_type, "confidence": 0.8},
            "added_detail": {"type": "noul", "noul": added_detail},
        },
        "usage": {"input_tokens": 100, "output_tokens": 0},
    }


def _kev_client() -> JevClient:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_jev_response())

    return JevClient(api_key=None, transport=httpx.MockTransport(handler), sleep=_no_sleep)


def _run(
    tmp_path: Path,
    documents: list[Document],
    backend,
    *,
    overlap_threshold: float = 0.5,
    judges: dict | None = None,
    budget_usd: float = 100.0,
    max_docs: int | None = None,
):
    corpus = "fda_maude"
    return run_corpus_single(
        corpus=corpus,
        kind="ticket",
        license_text="public domain",
        documents=documents,
        system_prompt="SYSTEM",
        prompt_variant="v6-type-precision",
        teacher_model=TEACHER,
        overlap_threshold=overlap_threshold,
        backend_factory=lambda model: backend,
        limiter=_CountingLimiter(),
        sink=InMemorySink(),
        tenant="test",
        records_file=records_path(tmp_path, corpus),
        failures_file=failures_path(tmp_path, corpus),
        progress_file=progress_path(tmp_path, corpus),
        dropped_file=dropped_path(tmp_path, corpus),
        spend=SpendMeter(path=default_spend_path(tmp_path)),
        budget_usd=budget_usd,
        judges=judges,
        max_docs=max_docs,
        sleep=_no_sleep,
        now=lambda: "2026-09-21T00:00:00-07:00",
    )


def test_kept_fact_when_overlap_gate_passes(tmp_path: Path) -> None:
    fact = _fact(
        "f1",
        "The pump beeped three times and shut off.",
        "the pump beeped three times and shut off",
    )
    backend = FakeBackend(canned=lambda envelope: _result("S-1", [fact]), model_key="nova-lite")
    stats = _run(tmp_path, [_doc("S-1")], backend)

    assert stats.facts_emitted == 1
    assert stats.facts_dropped_quote_gate == 0
    assert stats.facts_dropped_overlap == 0
    assert stats.facts_kept == 1
    assert stats.facts_kept_by_type["complaint"] == 1
    assert stats.records_written == 1

    written = list(read_records(records_path(tmp_path, "fda_maude")))
    assert len(written) == 1
    record = written[0]
    assert record.mode == "single"
    assert record.teachers == [TEACHER]
    assert not record.is_negative
    assert record.facts[0].judge == {}
    # feature/labeling-exact-quotes: an already-exact quote gets a
    # quote_span pointing at itself in the record text.
    kept = record.facts[0]
    start, end = kept.quote_span
    assert record.text[start:end] == kept.quote == "the pump beeped three times and shut off"


def test_dropped_by_overlap_when_claim_adds_unsupported_numeral(tmp_path: Path) -> None:
    fact = _fact(
        "f1",
        "The pump beeped three times, costing $9,900 in damages.",
        "the pump beeped three times and shut off",
    )
    backend = FakeBackend(canned=lambda envelope: _result("S-1", [fact]), model_key="nova-lite")
    stats = _run(tmp_path, [_doc("S-1")], backend)

    assert stats.facts_dropped_overlap == 1
    assert stats.facts_kept == 0
    written = list(read_records(records_path(tmp_path, "fda_maude")))
    assert written[0].is_negative

    dropped = list(read_dropped(dropped_path(tmp_path, "fda_maude")))
    assert len(dropped) == 1
    assert dropped[0].reason == "overlap_gate_numeral"
    assert dropped[0].source_id == "S-1"
    assert dropped[0].teacher == TEACHER
    assert dropped[0].fact["claim"] == "The pump beeped three times, costing $9,900 in damages."


def test_dropped_by_quote_gate_before_overlap_check(tmp_path: Path) -> None:
    fact = _fact("f1", "x", "a sentence never present anywhere in the document")
    backend = FakeBackend(canned=lambda envelope: _result("S-1", [fact]), model_key="nova-lite")
    stats = _run(tmp_path, [_doc("S-1")], backend)

    assert stats.facts_dropped_quote_gate == 1
    assert stats.facts_dropped_overlap == 0
    assert stats.facts_kept == 0

    dropped = list(read_dropped(dropped_path(tmp_path, "fda_maude")))
    assert len(dropped) == 1
    assert dropped[0].reason == "quote_gate"


# --------------------------------------------------------------------------
# feature/labeling-exact-quotes: exact-span resolution after the gate
# --------------------------------------------------------------------------


def test_kept_fact_resolves_quote_needing_normalization(tmp_path: Path) -> None:
    """A quote that differs from the doc only by case/whitespace clears
    both gates (quote_gate's own normalization already tolerates this)
    and gets resolved to the exact original-case span, tier
    `"normalized"`."""
    fact = _fact(
        "f1",
        "The pump beeped three times and shut off.",
        "The Pump  beeped THREE times and shut off",
    )
    backend = FakeBackend(canned=lambda envelope: _result("S-1", [fact]), model_key="nova-lite")
    stats = _run(tmp_path, [_doc("S-1")], backend)

    assert stats.facts_dropped_quote_not_exact == 0
    assert stats.facts_kept == 1
    record = list(read_records(records_path(tmp_path, "fda_maude")))[0]
    kept = record.facts[0]
    assert kept.quote == "the pump beeped three times and shut off"
    start, end = kept.quote_span
    assert record.text[start:end] == kept.quote


def test_kept_fact_resolves_quote_with_mime_artifact(tmp_path: Path) -> None:
    """Real Enron source text leaves quoted-printable `=20` (a soft
    space) undecoded (`01-spec/training-data-2026-09-19.md`'s "Known
    quirks"). A quote copied without the artifact still clears both
    gates and resolves via the `"mime"` tier."""
    doc_text = "the pump=20 beeped three times and shut off completely during the site visit"
    fact = _fact(
        "f1",
        "The pump beeped three times and shut off.",
        "the pump beeped three times and shut off",
    )
    backend = FakeBackend(canned=lambda envelope: _result("S-1", [fact]), model_key="nova-lite")
    stats = _run(tmp_path, [_doc("S-1", text=doc_text)], backend)

    assert stats.facts_kept == 1
    record = list(read_records(records_path(tmp_path, "fda_maude")))[0]
    kept = record.facts[0]
    start, end = kept.quote_span
    assert record.text[start:end] == kept.quote
    assert kept.quote == "the pump=20 beeped three times and shut off"


def test_kept_fact_resolves_stitched_quote_to_first_segment(tmp_path: Path) -> None:
    """A quote that splices two spans with `"..."` passes the fuzzy
    quote gate (each side's tokens found in order) but is not itself a
    contiguous span -- resolved to just its first segment, tier
    `"stitched:exact"`."""
    fact = _fact(
        "f1",
        "The pump beeped three times and shut off.",
        "the pump beeped three times ... shut off completely",
    )
    backend = FakeBackend(canned=lambda envelope: _result("S-1", [fact]), model_key="nova-lite")
    stats = _run(tmp_path, [_doc("S-1")], backend)

    assert stats.facts_kept == 1
    record = list(read_records(records_path(tmp_path, "fda_maude")))[0]
    kept = record.facts[0]
    assert kept.quote == "the pump beeped three times"
    start, end = kept.quote_span
    assert record.text[start:end] == kept.quote


def test_dropped_as_quote_not_exact_when_no_tier_resolves(tmp_path: Path) -> None:
    """A quote that clears the fuzzy overlap gate (one substituted word,
    still a valid forward token subsequence) but cannot be resolved to
    any exact/normalized/mime/stitched span is dropped, not kept with a
    paraphrase a student model could not learn to copy verbatim."""
    fact = _fact(
        "f1",
        "The pump beeped three times and shut off.",
        "the pump chimed three times and shut off",
    )
    backend = FakeBackend(canned=lambda envelope: _result("S-1", [fact]), model_key="nova-lite")
    stats = _run(tmp_path, [_doc("S-1")], backend)

    assert stats.facts_dropped_quote_gate == 0
    assert stats.facts_dropped_overlap == 0
    assert stats.facts_dropped_quote_not_exact == 1
    assert stats.facts_kept == 0
    written = list(read_records(records_path(tmp_path, "fda_maude")))
    assert written[0].is_negative

    dropped = list(read_dropped(dropped_path(tmp_path, "fda_maude")))
    assert len(dropped) == 1
    assert dropped[0].reason == "quote_not_exact"
    assert dropped[0].fact["quote"] == "the pump chimed three times and shut off"


def test_kev_verdict_attached_informationally_without_gating(tmp_path: Path) -> None:
    fact = _fact(
        "f1",
        "The pump beeped three times and shut off.",
        "the pump beeped three times and shut off",
    )
    backend = FakeBackend(canned=lambda envelope: _result("S-1", [fact]), model_key="nova-lite")
    stats = _run(tmp_path, [_doc("S-1")], backend, judges={"kev": _kev_client()})

    assert stats.facts_kept == 1
    assert stats.judge_spend_usd > 0
    written = list(read_records(records_path(tmp_path, "fda_maude")))
    assert set(written[0].facts[0].judge.keys()) == {"kev"}


def test_negative_when_no_facts_emitted(tmp_path: Path) -> None:
    backend = FakeBackend(canned=lambda envelope: _result("S-1", []), model_key="nova-lite")
    stats = _run(tmp_path, [_doc("S-1")], backend)
    assert stats.facts_emitted == 0
    assert stats.records_written == 1
    written = list(read_records(records_path(tmp_path, "fda_maude")))
    assert written[0].is_negative
    assert written[0].agreement == 1.0


def test_spend_tracks_extractor_cost(tmp_path: Path) -> None:
    fact = _fact(
        "f1",
        "The pump beeped three times and shut off.",
        "the pump beeped three times and shut off",
    )
    backend = FakeBackend(canned=lambda envelope: _result("S-1", [fact]), model_key="nova-lite")
    stats = _run(tmp_path, [_doc("S-1")], backend)
    assert stats.extractor_spend_usd > 0
    assert stats.total_spend_usd == pytest.approx(stats.extractor_spend_usd)


def test_run_stops_cleanly_when_budget_exhausted(tmp_path: Path) -> None:
    fact = _fact(
        "f1",
        "The pump beeped three times and shut off.",
        "the pump beeped three times and shut off",
    )
    backend = FakeBackend(canned=lambda envelope: _result("S-1", [fact]), model_key="nova-lite")
    docs = [_doc("S-1"), _doc("S-2"), _doc("S-3")]
    stats = _run(tmp_path, docs, backend, budget_usd=0.0)
    assert stats.stopped_reason == "budget_usd"
    assert stats.documents_processed == 0
    assert stats.records_written == 0
