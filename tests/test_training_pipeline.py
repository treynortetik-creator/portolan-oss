"""Tests for `portolan.training.pipeline` -- hold-out guard, resumability,
the negative cap, and the one-retry-then-record-failure behavior
(silver-labels build instructions items 4-6). No network: every backend
is a fake (`portolan.extract.backends.FakeBackend` or the small
`_FlakyBackend` double below), so these tests exercise the real
orchestration logic (`run_corpus`) without ever calling OpenRouter.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from portolan.extract.backends import Backend, FakeBackend
from portolan.extract.schema import Coverage, ExtractionResult, Fact, SourceType
from portolan.ingest.models import RawSource
from portolan.metering import InMemorySink
from portolan.training.pipeline import (
    Document,
    SilverRunStats,
    assert_not_holdout,
    load_progress,
    run_corpus,
    save_progress,
)
from portolan.training.records import failures_path, progress_path, read_records, records_path

MODEL_A = "qwen/test-a:free"
MODEL_B = "meta-llama/test-b:free"

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


def _positive_fact(fid: str) -> Fact:
    return Fact(
        id=fid,
        type="complaint",
        claim="device broke",
        locator="L1",
        quote=_QUOTE,
        sentiment=-1,
        confidence=0.9,
    )


def _no_sleep(_seconds: float) -> None:
    pass


class _CountingLimiter:
    """Stand-in for `corpus.generator.prose.TokenBucketLimiter` that
    never blocks -- these tests care about orchestration, not real rate
    limiting."""

    def __init__(self) -> None:
        self.acquisitions = 0

    def acquire(self) -> None:
        self.acquisitions += 1


def _make_backend_factory(responses: dict[tuple[str, str], ExtractionResult]):
    def factory(model: str) -> Backend:
        def canned(envelope):
            return responses[(model, envelope.source_id)]

        return FakeBackend(canned=canned, model_key="nova-lite")

    return factory


def _run(
    tmp_path: Path,
    documents: list[Document],
    responses: dict[tuple[str, str], ExtractionResult],
    *,
    negative_cap_ratio: float = 0.30,
    max_docs: int | None = None,
) -> SilverRunStats:
    corpus = "fda_maude"
    return run_corpus(
        corpus=corpus,
        kind="ticket",
        license_text="public domain",
        documents=documents,
        system_prompt="SYSTEM",
        prompt_variant="v6-type-precision",
        models=[MODEL_A, MODEL_B],
        backend_factory=_make_backend_factory(responses),
        limiter=_CountingLimiter(),
        sink=InMemorySink(),
        tenant="test",
        records_file=records_path(tmp_path, corpus),
        failures_file=failures_path(tmp_path, corpus),
        progress_file=progress_path(tmp_path, corpus),
        max_docs=max_docs,
        negative_cap_ratio=negative_cap_ratio,
        sleep=_no_sleep,
        now=lambda: "2026-09-21T00:00:00-07:00",
    )


# --------------------------------------------------------------------------
# Hold-out guard
# --------------------------------------------------------------------------


def test_assert_not_holdout_raises_for_corpus_out(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="hold-out violation"):
        assert_not_holdout(tmp_path / "corpus" / "out" / "calls")


def test_assert_not_holdout_raises_for_bare_corpus_out(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="hold-out violation"):
        assert_not_holdout(tmp_path / "corpus" / "out")


def test_assert_not_holdout_allows_converted_data(tmp_path: Path) -> None:
    assert_not_holdout(tmp_path / "var" / "data" / "converted" / "fda_maude")  # no raise


# --------------------------------------------------------------------------
# Resumability
# --------------------------------------------------------------------------


def test_load_save_progress_round_trip(tmp_path: Path) -> None:
    path = progress_path(tmp_path, "fda_maude")
    assert load_progress(path) == set()
    save_progress(path, {"S-1", "S-2"})
    assert load_progress(path) == {"S-1", "S-2"}


def test_run_corpus_skips_already_processed_sources(tmp_path: Path) -> None:
    docs = [_doc("S-1"), _doc("S-2")]
    responses = {
        (MODEL_A, "S-1"): _result("S-1", []),
        (MODEL_B, "S-1"): _result("S-1", []),
        (MODEL_A, "S-2"): _result("S-2", []),
        (MODEL_B, "S-2"): _result("S-2", []),
    }
    stats = _run(tmp_path, docs, responses)
    assert stats.documents_processed == 2
    assert stats.documents_skipped_resumed == 0

    # Re-run with a backend_factory that must never be called -- every
    # source is already in the progress file.
    def _forbidden_factory(model: str) -> Backend:
        raise AssertionError(f"backend_factory should not be called for {model!r}")

    stats2 = run_corpus(
        corpus="fda_maude",
        kind="ticket",
        license_text="public domain",
        documents=docs,
        system_prompt="SYSTEM",
        prompt_variant="v6-type-precision",
        models=[MODEL_A, MODEL_B],
        backend_factory=_forbidden_factory,
        limiter=_CountingLimiter(),
        sink=InMemorySink(),
        tenant="test",
        records_file=records_path(tmp_path, "fda_maude"),
        failures_file=failures_path(tmp_path, "fda_maude"),
        progress_file=progress_path(tmp_path, "fda_maude"),
        sleep=_no_sleep,
    )
    assert stats2.documents_processed == 0
    assert stats2.documents_skipped_resumed == 2


# --------------------------------------------------------------------------
# Negative cap
# --------------------------------------------------------------------------


def test_run_corpus_caps_negatives_at_30_percent(tmp_path: Path) -> None:
    # pos, neg, pos, neg, pos, neg, neg, neg, neg, neg
    kinds = ["pos", "neg", "pos", "neg", "pos", "neg", "neg", "neg", "neg", "neg"]
    docs = [_doc(f"S-{i}") for i in range(1, len(kinds) + 1)]
    responses: dict[tuple[str, str], ExtractionResult] = {}
    for doc, kind in zip(docs, kinds, strict=True):
        if kind == "pos":
            facts_a = [_positive_fact(f"{doc.source_id}-a")]
            facts_b = [_positive_fact(f"{doc.source_id}-b")]
        else:
            facts_a, facts_b = [], []
        responses[(MODEL_A, doc.source_id)] = _result(doc.source_id, facts_a)
        responses[(MODEL_B, doc.source_id)] = _result(doc.source_id, facts_b)

    stats = _run(tmp_path, docs, responses, negative_cap_ratio=0.30)

    assert stats.documents_processed == 10
    assert stats.records_written == 4  # 3 positive + 1 allowed negative
    assert stats.negatives_written == 1
    assert stats.negatives_capped == 6
    written = list(read_records(records_path(tmp_path, "fda_maude")))
    assert len(written) == 4
    negative_ratio = sum(1 for r in written if r.is_negative) / len(written)
    assert negative_ratio <= 0.30


def test_run_corpus_negative_cap_never_blocks_first_record(tmp_path: Path) -> None:
    # A single negative document must still be written even though 100%
    # negative trivially exceeds any cap ratio -- the cap only bites once
    # there is at least one record on the books to weigh against.
    docs = [_doc("S-1")]
    responses = {
        (MODEL_A, "S-1"): _result("S-1", []),
        (MODEL_B, "S-1"): _result("S-1", []),
    }
    stats = _run(tmp_path, docs, responses, negative_cap_ratio=0.30)
    assert stats.records_written == 1
    assert stats.negatives_written == 1
    assert stats.negatives_capped == 0


def test_run_corpus_negative_cap_respects_prior_run(tmp_path: Path) -> None:
    # Seed records.jsonl with 3 positives already on disk (from an
    # earlier run); a resumed run's negative cap must be computed against
    # that existing total, not reset to zero.
    from portolan.training.records import TrainingFact, TrainingRecord, append_record

    records_file = records_path(tmp_path, "fda_maude")
    for i in range(3):
        append_record(
            records_file,
            TrainingRecord(
                record_id=f"seed-{i}",
                corpus="fda_maude",
                source_id=f"seed-{i}",
                kind="ticket",
                license="public domain",
                text="x",
                facts=[TrainingFact(fact_type="complaint", claim="c", quote="q", locator="L1")],
                teachers=[MODEL_A, MODEL_B],
                agreement=1.0,
                created="2026-09-20T00:00:00-07:00",
            ),
        )

    docs = [_doc("S-1")]
    responses = {
        (MODEL_A, "S-1"): _result("S-1", []),
        (MODEL_B, "S-1"): _result("S-1", []),
    }
    # total_written starts at 3 (all positive); projected = (0+1)/(3+1) = 0.25 <= 0.30 -> allowed
    stats = _run(tmp_path, docs, responses, negative_cap_ratio=0.30)
    assert stats.negatives_written == 1
    assert stats.negatives_capped == 0


# --------------------------------------------------------------------------
# Retry-once-then-record-failure
# --------------------------------------------------------------------------


class _FlakyBackend:
    """Raises for every entry in `exceptions_then_result[:-1]`, returns
    `exceptions_then_result[-1]` on the final call. `.calls` records how
    many times `.extract()` was invoked."""

    def __init__(self, sequence: list[Exception | ExtractionResult]) -> None:
        self._sequence = sequence
        self.calls = 0

    def extract(self, envelope, system_prompt, sink, tenant):
        item = self._sequence[self.calls]
        self.calls += 1
        if isinstance(item, Exception):
            raise item
        return item


def test_run_corpus_retries_once_then_succeeds(tmp_path: Path) -> None:
    flaky = _FlakyBackend([RuntimeError("transient"), _result("S-1", [_positive_fact("a1")])])
    steady = FakeBackend(
        canned=lambda envelope: _result("S-1", [_positive_fact("b1")]), model_key="nova-lite"
    )

    def factory(model: str) -> Backend:
        return flaky if model == MODEL_A else steady

    stats = run_corpus(
        corpus="fda_maude",
        kind="ticket",
        license_text="public domain",
        documents=[_doc("S-1")],
        system_prompt="SYSTEM",
        prompt_variant="v6-type-precision",
        models=[MODEL_A, MODEL_B],
        backend_factory=factory,
        limiter=_CountingLimiter(),
        sink=InMemorySink(),
        tenant="test",
        records_file=records_path(tmp_path, "fda_maude"),
        failures_file=failures_path(tmp_path, "fda_maude"),
        progress_file=progress_path(tmp_path, "fda_maude"),
        sleep=_no_sleep,
    )
    assert flaky.calls == 2  # one failure, one retry
    assert stats.failures == 0
    assert stats.records_written == 1
    written = list(read_records(records_path(tmp_path, "fda_maude")))
    assert written[0].facts[0].fact_type == "complaint"
    assert not failures_path(tmp_path, "fda_maude").exists()


def test_run_corpus_records_failure_after_second_exception(tmp_path: Path) -> None:
    flaky = _FlakyBackend([RuntimeError("boom 1"), RuntimeError("boom 2")])

    def factory(model: str) -> Backend:
        return flaky  # only ever used for MODEL_A -- MODEL_B's call never happens

    stats = run_corpus(
        corpus="fda_maude",
        kind="ticket",
        license_text="public domain",
        documents=[_doc("S-1")],
        system_prompt="SYSTEM",
        prompt_variant="v6-type-precision",
        models=[MODEL_A, MODEL_B],
        backend_factory=factory,
        limiter=_CountingLimiter(),
        sink=InMemorySink(),
        tenant="test",
        records_file=records_path(tmp_path, "fda_maude"),
        failures_file=failures_path(tmp_path, "fda_maude"),
        progress_file=progress_path(tmp_path, "fda_maude"),
        sleep=_no_sleep,
        now=lambda: "2026-09-21T00:00:00-07:00",
    )
    assert flaky.calls == 2
    assert stats.failures == 1
    assert stats.records_written == 0

    failures_file = failures_path(tmp_path, "fda_maude")
    lines = failures_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["source_id"] == "S-1"
    assert entry["corpus"] == "fda_maude"
    assert "boom 2" in entry["error"]

    # "moves on": the doc is still marked processed, not retried forever.
    assert load_progress(progress_path(tmp_path, "fda_maude")) == {"S-1"}
