"""`portolan.pipeline`'s `--samples N` (2026-09-19): reseeds the live
graph with the extractor the extraction accuracy campaign proved
(`01-spec/HANDOFF-2026-09-18.md`: production pick is Nova Lite, five
samples, the v6-type-precision prompt). Runs a source's whole (possibly
segmented) extraction N independent times and unions the results with
`portolan.extract.segment.merge_segment_results` -- the exact same dedup
rule `portolan.bench.extraction._extract_one_source` already uses to
union its own samples (see `tests/test_bench_extraction.py`'s
`test_samples_unions_facts_across_independent_extraction_passes`, the
model for these tests).

Reuses `_FakeConcurrencyConnector`/`_settings` from `test_pipeline.py`
(established cross-test-file import pattern -- see `test_report_helpers`'s
own `_`-prefixed exports) rather than redefining them: one source, no
corpus files, no CRM import required, `account_hints=["TEST-ACCOUNT"]` is
enough for `customer_filter` to include it.
"""

from __future__ import annotations

from pathlib import Path

from test_pipeline import _FakeConcurrencyConnector, _settings

from portolan.llm.fake import FakeLLMClient
from portolan.metering import InMemorySink
from portolan.pipeline import run
from portolan.registry import Registry
from portolan.search.embed import FakeEmbedder
from portolan.store import LocalStore


def _zero_fact_counting_responder(counts: dict[str, int]):
    """One envelope, no facts: isolates the CALL COUNT (`--samples N`
    should multiply it) from anything about what the facts merge into.
    Mirrors `test_pipeline.py`'s own `_counting_zero_fact_responder`."""

    def _responder(*, system, user, json_schema, model, surface):
        del system, user, json_schema, model
        if surface == "extract":
            counts["extract_calls"] = counts.get("extract_calls", 0) + 1
            return {
                "source_id": "unknown",
                "summary": "no facts in this fake run",
                "coverage": {"segments_read": 1, "segments_total": 1},
            }
        return {"verdict": "new", "reason": "nothing to compare against"}  # pragma: no cover

    return _responder


def _identical_fact_responder(counts: dict[str, int]):
    """Every call returns the exact same fact (same type, claim, locator)
    -- proves the union DEDUPS identical passes down to one fact, not
    N copies of it."""

    def _responder(*, system, user, json_schema, model, surface):
        del system, user, json_schema, model
        if surface == "extract":
            counts["extract_calls"] = counts.get("extract_calls", 0) + 1
            return {
                "source_id": "unknown",
                "summary": "the same fact, every pass",
                "facts": [
                    {
                        "id": "f1",
                        "type": "risk",
                        "claim": "The customer is worried about renewal timing.",
                        "entities": [],
                        "locator": "00:00:05",
                        "sentiment": -1,
                        "confidence": 0.9,
                    }
                ],
                "coverage": {"segments_read": 1, "segments_total": 1},
            }
        return {"verdict": "new", "reason": "nothing to compare against"}  # pragma: no cover

    return _responder


def test_samples_n_makes_n_backend_calls_per_envelope(db, tmp_path: Path) -> None:
    """`--samples N` over one unsegmented source (one envelope) makes
    exactly N `backend.extract()` calls, not 1."""
    connector = _FakeConcurrencyConnector(1)
    settings = _settings(tmp_path)
    counts: dict[str, int] = {}
    llm = FakeLLMClient(_zero_fact_counting_responder(counts), sink=InMemorySink())

    report = run(
        db,
        connector=connector,
        llm=llm,
        embedder=FakeEmbedder(),
        registry=Registry.load(db),
        store=LocalStore(settings.store_root),
        settings=settings,
        samples=3,
        sink=InMemorySink(),
    )

    assert report.sources_extracted == 1
    assert counts["extract_calls"] == 3


def test_identical_passes_dedup_to_one_fact_set(db, tmp_path: Path) -> None:
    """Three independent passes that each find the SAME fact union down to
    one fact, not three -- `merge_segment_results`'s dedup rule, applied
    across samples the same way it's already applied across segments."""
    connector = _FakeConcurrencyConnector(1)
    settings = _settings(tmp_path)
    counts: dict[str, int] = {}
    llm = FakeLLMClient(_identical_fact_responder(counts), sink=InMemorySink())

    report = run(
        db,
        connector=connector,
        llm=llm,
        embedder=FakeEmbedder(),
        registry=Registry.load(db),
        store=LocalStore(settings.store_root),
        settings=settings,
        samples=3,
        sink=InMemorySink(),
    )

    assert counts["extract_calls"] == 3
    assert report.sources_extracted == 1
    assert report.facts_extracted == 1


def test_usage_event_count_equals_pass_count(db, tmp_path: Path) -> None:
    """Every pass meters its own usage event through the sink -- N passes
    bill N times, never double-counted and never collapsed into one."""
    connector = _FakeConcurrencyConnector(1)
    settings = _settings(tmp_path)
    counts: dict[str, int] = {}
    llm = FakeLLMClient(_identical_fact_responder(counts), sink=InMemorySink())

    report = run(
        db,
        connector=connector,
        llm=llm,
        embedder=FakeEmbedder(),
        registry=Registry.load(db),
        store=LocalStore(settings.store_root),
        settings=settings,
        samples=4,
        sink=InMemorySink(),
    )

    assert report.usage_by_surface["extract"].calls == 4
    assert counts["extract_calls"] == 4


def test_sample_count_change_reextracts_then_skips(db, tmp_path: Path) -> None:
    """A source extracted at samples=1 is re-extracted (not skipped) the
    first time a run over the same unchanged content asks for samples=5
    -- `extraction_samples` (migration 0025) is part of the stored
    extraction signature `_prepare_one`'s skip check compares. Re-running
    again at the same samples=5 then skips, exactly like an unchanged
    `extraction_model`/`extraction_prompt_variant` already does (bug hunt
    2026-09-18, finding 3)."""
    connector = _FakeConcurrencyConnector(1)
    settings = _settings(tmp_path)
    counts: dict[str, int] = {}
    llm = FakeLLMClient(_zero_fact_counting_responder(counts), sink=InMemorySink())

    first_report = run(
        db,
        connector=connector,
        llm=llm,
        embedder=FakeEmbedder(),
        registry=Registry.load(db),
        store=LocalStore(settings.store_root),
        settings=settings,
        samples=1,
        sink=InMemorySink(),
    )
    assert first_report.sources_extracted == 1
    assert first_report.sources_skipped == 0
    calls_after_first = counts["extract_calls"]
    assert calls_after_first == 1

    # Same unchanged corpus, but a different sample count: re-extracted,
    # not skipped, without --force.
    second_report = run(
        db,
        connector=connector,
        llm=llm,
        embedder=FakeEmbedder(),
        registry=Registry.load(db),
        store=LocalStore(settings.store_root),
        settings=settings,
        samples=5,
        sink=InMemorySink(),
    )
    assert second_report.sources_extracted == 1
    assert second_report.sources_skipped == 0
    assert counts["extract_calls"] == calls_after_first + 5

    # Re-run again at the SAME samples=5: now skipped, like any other
    # unchanged extraction signature.
    third_report = run(
        db,
        connector=connector,
        llm=llm,
        embedder=FakeEmbedder(),
        registry=Registry.load(db),
        store=LocalStore(settings.store_root),
        settings=settings,
        samples=5,
        sink=InMemorySink(),
    )
    assert third_report.sources_extracted == 0
    assert third_report.sources_skipped == 1
    assert counts["extract_calls"] == calls_after_first + 5


def test_default_samples_makes_exactly_one_call(db, tmp_path: Path) -> None:
    """Omitting `--samples` entirely (the CLI default, and `run()`'s own
    default) must leave today's single-pass behavior unchanged -- exactly
    one `backend.extract()` call per envelope, not `max(1, None)`-style
    surprises."""
    connector = _FakeConcurrencyConnector(1)
    settings = _settings(tmp_path)
    counts: dict[str, int] = {}
    llm = FakeLLMClient(_identical_fact_responder(counts), sink=InMemorySink())

    report = run(
        db,
        connector=connector,
        llm=llm,
        embedder=FakeEmbedder(),
        registry=Registry.load(db),
        store=LocalStore(settings.store_root),
        settings=settings,
        sink=InMemorySink(),
    )

    assert counts["extract_calls"] == 1
    assert report.facts_extracted == 1
