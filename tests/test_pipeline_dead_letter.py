"""Dead-letter and redrive tests for `portolan.pipeline` (build-contracts.md
"Pipeline runner"): a stage that raises on every attempt gets dead-lettered
after three tries, the run continues with the next source, and `redrive`
clears the row once the underlying problem is fixed.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

from portolan.config import Settings
from portolan.connectors.base import SourceRef
from portolan.extract.schema import Coverage, ExtractionResult, SourceType
from portolan.ingest.models import RawSource
from portolan.llm.fake import FakeLLMClient
from portolan.pipeline import redrive, run
from portolan.registry import Registry
from portolan.search.embed import FakeEmbedder
from portolan.store import LocalStore

BOOM_ID = "boom-1"
OK_ID = "ok-1"


def _refs() -> tuple[SourceRef, SourceRef]:
    now = datetime.now(UTC)
    boom_ref = SourceRef(source_id=BOOM_ID, source_type=SourceType.EMAIL, updated_at=now, ref={})
    ok_ref = SourceRef(source_id=OK_ID, source_type=SourceType.EMAIL, updated_at=now, ref={})
    return boom_ref, ok_ref


def _ok_raw_source() -> RawSource:
    return RawSource(
        source_id=OK_ID,
        source_type=SourceType.EMAIL,
        source_date=date(2026, 1, 1),
        participants=[],
        # Forces inclusion through the filter's account_hint rule without
        # needing a populated registry.
        account_hints=["forced-account"],
        text="hello from a working source",
        metadata={},
    )


class _AlwaysFailsFetchConnector:
    """`fetch()` raises unconditionally for BOOM_ID, succeeds for OK_ID."""

    name = "flaky"

    def __init__(self) -> None:
        self.fetch_attempts = 0
        self._boom_ref, self._ok_ref = _refs()

    def list_sources(self, since=None):
        return [self._boom_ref, self._ok_ref]

    def fetch(self, ref: SourceRef) -> RawSource:
        if ref.source_id == BOOM_ID:
            self.fetch_attempts += 1
            raise RuntimeError(f"vendor timeout (attempt {self.fetch_attempts})")
        return _ok_raw_source()


class _NowFixedConnector:
    """Same two refs, but BOOM_ID's `fetch()` now succeeds -- simulating
    the underlying problem being fixed before a redrive."""

    name = "flaky"

    def __init__(self) -> None:
        self._boom_ref, self._ok_ref = _refs()

    def list_sources(self, since=None):
        return [self._boom_ref, self._ok_ref]

    def fetch(self, ref: SourceRef) -> RawSource:
        return RawSource(
            source_id=ref.source_id,
            source_type=SourceType.EMAIL,
            source_date=date(2026, 1, 2),
            participants=[],
            account_hints=["forced-account"],
            text="now working",
            metadata={},
        )


def _canned_extraction(source_id: str) -> dict:
    return ExtractionResult(
        source_id=source_id,
        summary="fine",
        entities=[],
        facts=[],
        actions=[],
        unknown_entities=[],
        coverage=Coverage(segments_read=1, segments_total=1),
    ).model_dump(mode="json")


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        database_url="postgresql://localhost/unused",
        tenant="test",
        store_root=tmp_path / "store",
    )


def test_a_stage_that_fails_three_times_is_dead_lettered_and_the_run_continues(
    db, tmp_path
) -> None:
    connector = _AlwaysFailsFetchConnector()
    llm = FakeLLMClient([_canned_extraction(OK_ID)])
    embedder = FakeEmbedder()
    store = LocalStore(tmp_path / "store")
    registry = Registry.load(db)
    settings = _settings(tmp_path)

    report = run(
        db,
        connector=connector,
        llm=llm,
        embedder=embedder,
        registry=registry,
        store=store,
        settings=settings,
        sleep=lambda _seconds: None,
    )

    assert connector.fetch_attempts == 3
    assert report.sources_failed == 1
    assert report.dead_letter_count == 1
    # the OTHER source in the same run still got processed -- one bad
    # source never aborts the run.
    assert report.sources_extracted + report.sources_filtered + report.sources_partial == 1

    with db.cursor() as cur:
        cur.execute("SELECT source_id, stage, attempt, redriven_at FROM dead_letter")
        rows = cur.fetchall()
    assert rows == [(BOOM_ID, "fetch", 3, None)]

    with db.cursor() as cur:
        cur.execute("SELECT slug FROM sources WHERE slug = %s", (BOOM_ID,))
        # fetch failed before intake ever ran, so no sources row exists.
        assert cur.fetchone() is None


def test_redrive_clears_the_dead_letter_row_once_the_source_works(db, tmp_path) -> None:
    settings = _settings(tmp_path)
    embedder = FakeEmbedder()
    store = LocalStore(tmp_path / "store")
    registry = Registry.load(db)

    # First, produce a dead-lettered row the same way the previous test does.
    failing_connector = _AlwaysFailsFetchConnector()
    run(
        db,
        connector=failing_connector,
        llm=FakeLLMClient([_canned_extraction(OK_ID)]),
        embedder=embedder,
        registry=registry,
        store=store,
        settings=settings,
        sleep=lambda _seconds: None,
    )
    with db.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM dead_letter WHERE redriven_at IS NULL")
        (pending_before,) = cur.fetchone()
    assert pending_before == 1

    # Now redrive against a connector where the problem is fixed.
    fixed_connector = _NowFixedConnector()
    redrive_report = redrive(
        db,
        connector=fixed_connector,
        llm=FakeLLMClient([_canned_extraction(BOOM_ID)]),
        embedder=embedder,
        registry=registry,
        store=store,
        settings=settings,
        sleep=lambda _seconds: None,
    )

    assert redrive_report.attempted == 1
    assert redrive_report.succeeded == 1
    assert redrive_report.still_failing == 0

    with db.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM dead_letter WHERE redriven_at IS NULL")
        (pending_after,) = cur.fetchone()
    assert pending_after == 0

    with db.cursor() as cur:
        cur.execute("SELECT extraction_status FROM sources WHERE slug = %s", (BOOM_ID,))
        (status,) = cur.fetchone()
    assert status == "done"


def test_redrive_with_nothing_pending_is_a_no_op(db, tmp_path) -> None:
    settings = _settings(tmp_path)
    report = redrive(
        db,
        connector=_NowFixedConnector(),
        llm=FakeLLMClient([]),
        embedder=FakeEmbedder(),
        registry=Registry.load(db),
        store=LocalStore(tmp_path / "store"),
        settings=settings,
    )
    assert report == type(report)()


# --------------------------------------------------------------------
# Failures outside the retried stages: envelope building, the quote gate
# and the status/coverage writes after merge. One bad source is
# dead-lettered; the run finishes the others.
# --------------------------------------------------------------------


class _TwoIncludedSourcesConnector:
    """`env-boom` then `env-ok`, both included through `account_hints`."""

    name = "two-included"

    def __init__(self) -> None:
        now = datetime.now(UTC)
        self._refs = [
            SourceRef(source_id=sid, source_type=SourceType.EMAIL, updated_at=now, ref={})
            for sid in ("env-boom", "env-ok")
        ]

    def list_sources(self, since=None):
        return list(self._refs)

    def fetch(self, ref: SourceRef) -> RawSource:
        return RawSource(
            source_id=ref.source_id,
            source_type=SourceType.EMAIL,
            source_date=date(2026, 1, 3),
            participants=[],
            account_hints=["forced-account"],
            text=f"hello from {ref.source_id}",
            metadata={},
        )


def _run_two(db, tmp_path, llm):
    return run(
        db,
        connector=_TwoIncludedSourcesConnector(),
        llm=llm,
        embedder=FakeEmbedder(),
        registry=Registry.load(db),
        store=LocalStore(tmp_path / "store"),
        settings=_settings(tmp_path),
        sleep=lambda _seconds: None,
    )


def _dead_letters(db) -> list[tuple[str, str]]:
    with db.cursor() as cur:
        cur.execute("SELECT source_id, stage FROM dead_letter WHERE redriven_at IS NULL")
        return sorted(cur.fetchall())


def _statuses(db) -> dict[str, str]:
    with db.cursor() as cur:
        cur.execute("SELECT slug, extraction_status FROM sources WHERE slug LIKE 'env-%%'")
        return dict(cur.fetchall())


def test_a_database_error_building_envelopes_dead_letters_the_source(
    db, tmp_path, monkeypatch
) -> None:
    """A real SQL error on a non-autocommit connection, as the CLI opens:
    the failed transaction is rolled back before the dead letter is
    written."""
    import portolan.pipeline as pipeline_module

    real_build_account_context = pipeline_module.build_account_context
    calls = {"n": 0}

    def _account_context(conn, registry_context):
        calls["n"] += 1
        if calls["n"] == 1:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 / 0")
        return real_build_account_context(conn, registry_context)

    monkeypatch.setattr(pipeline_module, "build_account_context", _account_context)
    db.autocommit = False
    try:
        report = _run_two(db, tmp_path, FakeLLMClient([_canned_extraction("env-ok")]))
    finally:
        db.rollback()
        db.autocommit = True

    assert report.sources_failed == 1
    assert report.sources_extracted == 1
    assert _dead_letters(db) == [("env-boom", "envelope")]
    assert _statuses(db) == {"env-boom": "failed", "env-ok": "done"}


def test_a_quote_gate_failure_dead_letters_the_source(db, tmp_path, monkeypatch) -> None:
    import portolan.pipeline as pipeline_module

    monkeypatch.setenv("PORTOLAN_EXTRACT_CONCURRENCY", "1")
    real_gate = pipeline_module._gate_quotes
    calls = {"n": 0}

    def _gate(facts, *, source_text):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("quote gate broke")
        return real_gate(facts, source_text=source_text)

    monkeypatch.setattr(pipeline_module, "_gate_quotes", _gate)

    report = _run_two(
        db,
        tmp_path,
        FakeLLMClient([_canned_extraction("env-boom"), _canned_extraction("env-ok")]),
    )

    assert report.sources_failed == 1
    assert report.sources_extracted == 1
    assert _dead_letters(db) == [("env-boom", "quote_gate")]
    assert _statuses(db) == {"env-boom": "failed", "env-ok": "done"}


def test_a_status_write_failure_after_merge_dead_letters_the_source(
    db, tmp_path, monkeypatch
) -> None:
    import portolan.pipeline as pipeline_module

    monkeypatch.setenv("PORTOLAN_EXTRACT_CONCURRENCY", "1")
    real_record = pipeline_module.record_coverage
    calls = {"n": 0}

    def _record(conn, source_type, day, **counts):
        if "extracted" in counts:
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("coverage write broke")
        return real_record(conn, source_type, day, **counts)

    monkeypatch.setattr(pipeline_module, "record_coverage", _record)

    report = _run_two(
        db,
        tmp_path,
        FakeLLMClient([_canned_extraction("env-boom"), _canned_extraction("env-ok")]),
    )

    assert report.sources_failed == 1
    assert report.sources_extracted == 1
    assert _dead_letters(db) == [("env-boom", "status")]
    assert _statuses(db) == {"env-boom": "failed", "env-ok": "done"}
