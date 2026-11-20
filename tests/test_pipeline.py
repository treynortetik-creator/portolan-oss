"""End-to-end `portolan.pipeline` tests: a full fake-LLM run over the real
`corpus/out` corpus, and the CLI's argument parsing / fake path.
"""

from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from conftest import ensure_database_exists, with_database_name

import portolan.pipeline as pipeline_module
from portolan.config import Settings
from portolan.connectors.base import SourceRef
from portolan.connectors.file_connector import FileConnector
from portolan.extract.backends import FakeBackend, LLMClientBackend
from portolan.extract.schema import Coverage, ExtractionResult, Fact, SourceType, Vocabulary
from portolan.fake_extractor import FakePlacements, build_fake_llm_client
from portolan.ingest.models import RawSource
from portolan.llm.fake import FakeLLMClient
from portolan.merge import merge_extraction
from portolan.metering import InMemorySink
from portolan.pipeline import RunReport, main, run
from portolan.registry import Registry
from portolan.search.embed import FakeEmbedder
from portolan.store import LocalStore

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_OUT = REPO_ROOT / "corpus" / "out"
TRUTH_TABLE_PATH = REPO_ROOT / "corpus" / "truth-table.yaml"


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        database_url="postgresql://localhost/unused",
        tenant="test",
        store_root=tmp_path / "store",
    )


def _planted_fact_source_ids() -> set[str]:
    """Every non-distractor source_id `corpus/truth-table.yaml` plants at
    least one real fact on, per its own `facts:` list."""
    import yaml

    truth_table = yaml.safe_load(TRUTH_TABLE_PATH.read_text(encoding="utf-8"))
    source_ids: set[str] = set()
    for fact_def in truth_table.get("facts", []):
        if fact_def.get("type") == "distractor":
            continue
        for source_entry in fact_def.get("sources", []):
            source_ids.add(source_entry["id"])
    return source_ids


@pytest.fixture(scope="module")
def full_run_report_and_db_url(tmp_path_factory):
    """Runs the full fake pipeline once per test session (it's not cheap:
    ~100 fake extraction calls plus merge), and hands every test in this
    module the report plus a fresh connection to inspect afterwards.

    Scoped to the module so `db`'s function-scoped schema reset can't run
    underneath it -- this fixture manages its own connection/schema
    directly rather than depending on the shared `db` fixture.
    """
    import os

    import psycopg

    from portolan import db as db_module

    database_url = os.environ.get("DATABASE_URL", "postgresql://localhost/portolan_test_pipeline")
    try:
        conn = db_module.connect(database_url)
    except psycopg.OperationalError as exc:
        pytest.skip(f"Postgres unreachable at {database_url!r}: {exc}")

    with conn.cursor() as cur:
        cur.execute("DROP SCHEMA public CASCADE")
        cur.execute("CREATE SCHEMA public")
    conn.commit()
    db_module.apply_migrations(conn)
    conn.autocommit = True

    tmp_path = tmp_path_factory.mktemp("pipeline_e2e")
    settings = _settings(tmp_path)
    connector = FileConnector(CORPUS_OUT)
    placements = FakePlacements(CORPUS_OUT)
    llm = build_fake_llm_client(placements, tenant=settings.tenant)
    embedder = FakeEmbedder()
    store = LocalStore(settings.store_root)
    registry = Registry.load(conn)

    report = run(
        conn,
        connector=connector,
        llm=llm,
        embedder=embedder,
        registry=registry,
        store=store,
        settings=settings,
    )

    yield report, conn
    conn.close()


@pytest.fixture(scope="module")
def full_run_report(full_run_report_and_db_url) -> RunReport:
    return full_run_report_and_db_url[0]


@pytest.fixture(scope="module")
def full_run_conn(full_run_report_and_db_url):
    return full_run_report_and_db_url[1]


def test_full_run_has_no_failures(full_run_report: RunReport) -> None:
    assert full_run_report.sources_failed == 0
    assert full_run_report.dead_letter_count == 0


def test_truth_table_yaml_parses(full_run_report: RunReport) -> None:
    """Review finding G1: the truth table must be valid YAML, or every
    `test_report_*`/`test_audit*` test and `pipeline._maybe_import_
    registry`'s own real-run path go dark. `full_run_report` isn't used
    for anything here beyond confirming a full run already succeeded
    (the corpus is what this test is really about, not the pipeline)."""
    import yaml

    data = yaml.safe_load(TRUTH_TABLE_PATH.read_text(encoding="utf-8"))
    assert data.get("facts"), "expected a non-empty facts: list"


def test_full_run_lint_reports_zero_orphans_and_zero_broken_links(full_run_conn) -> None:
    """Review findings G3/G6 (task item 6): after a full fake pipeline run
    over `corpus/out`, `run_lint` must report zero `orphan` and zero
    `broken_link` errors. Before the fix, `WIKILINK_RE`/`parse_related`
    rejected `_` (dropping `signal-competitor_mention-*`/`signal-
    contact_change-*` links on write and losing already-written `##
    Related` entries on the next read/write round-trip) and contact notes
    got no link to their account at all -- five orphans (`maria-chen`,
    `tom-ridley`, and three `signal-competitor_mention-*` notes) on this
    exact corpus."""
    from portolan.lint import checks

    broken_links = checks.check_broken_link(full_run_conn)
    orphans = checks.check_orphan(full_run_conn)

    assert broken_links == [], [f.message for f in broken_links]
    assert orphans == [], [(f.note_slug, f.message) for f in orphans]


def test_full_run_ends_with_two_superseded_claims_and_three_action_notes(full_run_conn) -> None:
    """Review finding, task item 9: `fake_extractor` must emit the truth
    table's actions (F09, F32, F33) and `contradicts` verdicts for the
    planted reversals (F11 -> F11b, F15 -> F15b), so a full fake run ends
    with exactly 2 superseded claims and 3 action notes -- before the fix,
    `fake_extractor` never emitted an action or a `contradicts` verdict at
    all, so `merge_action` and the whole supersession path had zero
    end-to-end coverage from any fake pipeline run.

    v1.1 (2026-09-13) added a third reversal (F40 -> F40b, the Wrenfield
    Tidewater Link/Epic two-clocks case) and two more actions with an
    owner and a due date (F57, F58) -- see `corpus/truth-table.yaml`
    `expectations.merge.supersessions` / `expectations.report.actions_open`
    -- so the counts below are 3 and 5, not the v0 2 and 3 the function
    name (kept for git-blame continuity) still names."""
    with full_run_conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM claims WHERE superseded_at IS NOT NULL")
        (superseded_count,) = cur.fetchone()
        cur.execute("SELECT COUNT(*) FROM notes WHERE type = 'action'")
        (action_note_count,) = cur.fetchone()

    assert superseded_count == 3
    assert action_note_count == 5


def test_full_run_filters_the_junk_and_internal_only_emails(full_run_conn) -> None:
    with full_run_conn.cursor() as cur:
        cur.execute(
            "SELECT slug, extraction_status FROM sources WHERE slug IN "
            "('S-email-040', 'S-email-061')"
        )
        rows = dict(cur.fetchall())
    assert rows["S-email-040"] == "filtered"
    assert rows["S-email-061"] == "filtered"


def test_full_run_extracts_every_source_with_a_planted_fact(full_run_conn) -> None:
    # S-email-061 carries a planted fact (F07) but is *deliberately*
    # excluded by the filter's internal-only carve-out
    # (portolan/ingest/filter.py's module docstring, and this same
    # truth-table fact is what the coverage ledger's sample audit exists
    # to catch) -- it belongs in the "filtered" assertion above, not here.
    # S-email-065 (v1.1, F46) is the same shape: an internal-only ops email
    # about the Dayton branch, deliberately excluded the same way.
    planted = _planted_fact_source_ids() - {"S-email-061", "S-email-065"}
    assert planted  # sanity: the truth table actually has planted facts

    with full_run_conn.cursor() as cur:
        cur.execute(
            "SELECT slug, extraction_status FROM sources WHERE slug = ANY(%s)",
            (list(planted),),
        )
        rows = dict(cur.fetchall())

    missing = planted - set(rows)
    assert not missing, f"planted-fact sources never even reached the sources table: {missing}"

    not_extracted = {
        slug: status for slug, status in rows.items() if status not in ("done", "partial")
    }
    assert not not_extracted, f"planted-fact sources did not end up extracted: {not_extracted}"


def test_full_run_produced_facts_and_ran_the_merge_stage(full_run_report: RunReport) -> None:
    assert full_run_report.facts_extracted > 0
    assert full_run_report.usage_by_surface["extract"].calls > 0


def test_full_run_writes_usage_events_for_extract_and_merge_matching_the_report(
    full_run_report: RunReport, full_run_conn
) -> None:
    """Task 7: `run()` was never given a `sink=` here, so it defaults to
    `portolan.metering.PostgresSink(conn)` -- a real run's metering must
    land in `usage_events`, not an `InMemorySink` that dies with the
    process, and `RunReport.usage_by_surface` (computed from that very
    sink) must match what actually got written."""
    with full_run_conn.cursor() as cur:
        cur.execute(
            "SELECT surface, COUNT(*), SUM(input_tokens), SUM(output_tokens) "
            "FROM usage_events GROUP BY surface"
        )
        rows = {
            surface: (count, in_tok, out_tok) for surface, count, in_tok, out_tok in cur.fetchall()
        }

    assert "extract" in rows
    assert "merge" in rows, "F02 (ticket + call) should have triggered at least one compare call"

    for surface, totals in full_run_report.usage_by_surface.items():
        assert surface in rows
        db_count, db_input, db_output = rows[surface]
        assert db_count == totals.calls
        assert db_input == totals.input_tokens
        assert db_output == totals.output_tokens


def test_full_run_merge_notes_and_claims_present_for_f02_with_two_citations(
    full_run_conn,
) -> None:
    """Review finding G4: F02 (ticket + call, the truth table's own
    `cross_source_single_claim` case) lands as ONE claim, never two --
    the second sighting is a `claim_citations` row, not a duplicate
    `claims` row."""
    pytest.importorskip("portolan.merge")

    with full_run_conn.cursor() as cur:
        cur.execute(
            "SELECT id, note_slug, source_slug, locator FROM claims "
            "WHERE claim_text ILIKE %s ORDER BY id",
            ("%four days late%",),
        )
        claim_rows = cur.fetchall()

    assert len(claim_rows) == 1, f"F02 must be exactly one claim, got {claim_rows}"
    claim_id, note_slug, primary_source, primary_locator = claim_rows[0]

    with full_run_conn.cursor() as cur:
        cur.execute(
            "SELECT source_slug, locator FROM claim_citations WHERE claim_id = %s",
            (claim_id,),
        )
        citation_rows = cur.fetchall()

    assert len(citation_rows) == 1, f"expected exactly one extra citation, got {citation_rows}"
    extra_source, extra_locator = citation_rows[0]

    source_slugs = {primary_source, extra_source}
    assert source_slugs == {"S-call-009", "S-ticket-031"}
    # Each citation's locator points into its own source (provenance-
    # chain.md) -- never the same locator repeated under both slugs.
    assert primary_locator != extra_locator

    with full_run_conn.cursor() as cur:
        cur.execute("SELECT slug FROM notes WHERE slug = %s", (note_slug,))
        assert cur.fetchone() is not None


# --------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------


def test_cli_run_parses_arguments_and_completes_the_fake_path(
    db, db_url, tmp_path, monkeypatch
) -> None:
    pipeline_db_url = with_database_name(db_url, "portolan_test_pipeline")
    ensure_database_exists(pipeline_db_url)
    monkeypatch.setenv("DATABASE_URL", pipeline_db_url)
    monkeypatch.setenv("STORE_ROOT", str(tmp_path / "store"))
    monkeypatch.setenv("PORTOLAN_TENANT", "cli-test")

    exit_code = main(
        [
            "--connector",
            "file",
            "--path",
            str(CORPUS_OUT),
            "--llm",
            "fake",
            "--embed",
            "fake",
            "--limit",
            "5",
            "--dry-run",
        ]
    )
    assert exit_code == 0


def test_cli_redrive_subcommand_parses_and_runs(db, db_url, tmp_path, monkeypatch) -> None:
    pipeline_db_url = with_database_name(db_url, "portolan_test_pipeline")
    ensure_database_exists(pipeline_db_url)
    monkeypatch.setenv("DATABASE_URL", pipeline_db_url)
    monkeypatch.setenv("STORE_ROOT", str(tmp_path / "store"))

    exit_code = main(["redrive", "--connector", "file", "--path", str(CORPUS_OUT), "--llm", "fake"])
    assert exit_code == 0


def test_cli_run_via_subprocess(db_url, tmp_path) -> None:
    import os
    import subprocess
    import sys

    pipeline_db_url = with_database_name(db_url, "portolan_test_pipeline")
    ensure_database_exists(pipeline_db_url)

    env = dict(os.environ)
    env["DATABASE_URL"] = pipeline_db_url
    env["STORE_ROOT"] = str(tmp_path / "store")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "portolan.pipeline",
            "--connector",
            "file",
            "--path",
            str(CORPUS_OUT),
            "--llm",
            "fake",
            "--embed",
            "fake",
            "--limit",
            "3",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["sources_seen"] == 3


# --------------------------------------------------------------------
# Extraction fan-out (polish wave 2026-09-12, task 6): the extract stage
# runs across a ThreadPoolExecutor with settings-driven concurrency;
# merge/write/index stay strictly sequential in source order.
# --------------------------------------------------------------------


class _FakeConcurrencyConnector:
    """A minimal `Connector` (no corpus files, no registry) for timing
    tests: every source carries `account_hints`, so `customer_filter`
    includes it (`portolan.ingest.filter.customer_filter`'s second rule)
    with no CRM import required at all."""

    name = "fake-concurrency"

    def __init__(self, n: int) -> None:
        self._refs = [
            SourceRef(
                source_id=f"concurrency-src-{i}",
                source_type=SourceType.CALL,
                updated_at=datetime(2026, 8, 1, tzinfo=UTC),
                ref=i,
            )
            for i in range(n)
        ]

    def list_sources(self, since):
        del since
        return list(self._refs)

    def fetch(self, ref: SourceRef) -> RawSource:
        return RawSource(
            source_id=ref.source_id,
            source_type=SourceType.CALL,
            source_date=date(2026, 8, 1),
            participants=[],
            account_hints=["TEST-ACCOUNT"],
            text=(
                f"[00:00:01] Speaker: This is fake concurrency-test source "
                f"{ref.ref}, with enough text to look like a real call."
            ),
        )


def test_run_overlaps_extraction_but_merges_in_source_order(db, tmp_path, monkeypatch) -> None:
    """The proof task 6 asks for: with concurrency 4 and a fake backend
    that sleeps and records wall-clock windows, at least 4 extraction
    calls genuinely overlap in time -- and `_finish_one` (merge/write)
    still runs in exactly the order the sources were listed in, regardless
    of which order their (concurrent, unordered-completion) extraction
    calls happen to finish in."""
    monkeypatch.setenv("PORTOLAN_EXTRACT_CONCURRENCY", "4")
    n = 4
    connector = _FakeConcurrencyConnector(n)
    settings = _settings(tmp_path)
    embedder = FakeEmbedder()
    store = LocalStore(settings.store_root)
    registry = Registry.load(db)

    call_windows: list[tuple[float, float]] = []
    windows_lock = threading.Lock()

    def _responder(*, system, user, json_schema, model, surface):
        del system, user, json_schema, model
        if surface == "extract":
            start = time.time()
            time.sleep(0.15)  # long enough that 4 concurrent calls clearly overlap
            end = time.time()
            with windows_lock:
                call_windows.append((start, end))
            return {
                "source_id": "unknown",
                "summary": "no facts in this fake run",
                "coverage": {"segments_read": 1, "segments_total": 1},
            }
        # merge-surface compare calls shouldn't happen (every extraction
        # here returns zero facts), but answer safely if one ever does.
        return {"verdict": "new", "reason": "nothing to compare against"}

    llm = FakeLLMClient(_responder, sink=InMemorySink())

    finish_order: list[str] = []
    real_finish_one = pipeline_module._finish_one

    def _tracking_finish_one(extracted, **kwargs):
        outcome = real_finish_one(extracted, **kwargs)
        finish_order.append(extracted.prepared.ref.source_id)
        return outcome

    monkeypatch.setattr(pipeline_module, "_finish_one", _tracking_finish_one)

    report = run(
        db,
        connector=connector,
        llm=llm,
        embedder=embedder,
        registry=registry,
        store=store,
        settings=settings,
        sink=InMemorySink(),
    )

    assert report.sources_seen == n
    assert report.sources_extracted == n
    assert report.sources_failed == 0

    # Merge order equals source order, even though extraction ran
    # concurrently and could have finished in any order.
    assert finish_order == [f"concurrency-src-{i}" for i in range(n)]

    # Overlap proof: at the instant any one call started, count how many
    # of the n windows were already open -- with real concurrency, that
    # peaks at n; run sequentially (concurrency=1), it would never exceed 1.
    assert len(call_windows) == n
    max_concurrent = max(
        sum(1 for (s2, e2) in call_windows if s2 <= s < e2) for (s, _e) in call_windows
    )
    assert max_concurrent >= n


# --------------------------------------------------------------------
# `_usage_by_surface` / `RunReport.usage_pricing_errors`: a `None`
# `cost_estimate` (`metered()`'s review finding F3 -- an unrecognised
# model_key records the event anyway rather than losing it) must not
# crash the rollup, and must be visible somewhere other than a silently
# too-low total.
# --------------------------------------------------------------------


def _usage_event(*, surface: str, cost_estimate, pricing_error: bool = False):
    from portolan.metering import UsageEvent

    return UsageEvent(
        tenant="t",
        surface=surface,
        model_key="nova-lite",
        input_tokens=10,
        output_tokens=5,
        cached_tokens=0,
        cost_estimate=cost_estimate,
        request_id="req_test",
        timestamp="2026-09-12T00:00:00+00:00",
        extra={"pricing_error": "unknown model"} if pricing_error else {},
    )


def test_usage_by_surface_treats_a_none_cost_estimate_as_zero() -> None:
    events = [
        _usage_event(surface="extract", cost_estimate=0.05),
        _usage_event(surface="extract", cost_estimate=None, pricing_error=True),
    ]
    totals = pipeline_module._usage_by_surface(events)
    assert totals["extract"].calls == 2
    assert totals["extract"].cost_estimate == 0.05  # the None contributed 0, not a crash


def test_run_counts_pricing_errors_separately_from_usage_by_surface(
    db, tmp_path, monkeypatch
) -> None:
    """An extraction model unknown to `portolan.pricing.PRICING` must not
    crash `run()` -- its events still land in `usage_by_surface` (cost
    0.0), and `RunReport.usage_pricing_errors` counts them so an unpriced
    run is visibly flagged rather than silently under-billed."""
    monkeypatch.setenv("PORTOLAN_EXTRACT_CONCURRENCY", "1")
    connector = _FakeConcurrencyConnector(2)
    settings = _settings(tmp_path)
    settings = settings.model_copy(update={"extraction_model": "totally-unpriced-model-xyz"})
    embedder = FakeEmbedder()
    store = LocalStore(settings.store_root)
    registry = Registry.load(db)

    def _responder(*, system, user, json_schema, model, surface):
        del system, user, json_schema, model, surface
        return {
            "source_id": "unknown",
            "summary": "no facts",
            "coverage": {"segments_read": 1, "segments_total": 1},
        }

    report = run(
        db,
        connector=connector,
        llm=FakeLLMClient(_responder),
        embedder=embedder,
        registry=registry,
        store=store,
        settings=settings,
        sink=InMemorySink(),
    )

    assert report.sources_extracted == 2
    assert report.usage_pricing_errors >= 2  # at least one per extraction call
    assert report.usage_by_surface["extract"].cost_estimate == 0.0


def test_extract_concurrency_env_var_defaults_to_four(monkeypatch) -> None:
    monkeypatch.delenv("PORTOLAN_EXTRACT_CONCURRENCY", raising=False)
    assert pipeline_module._extract_concurrency() == 4

    monkeypatch.setenv("PORTOLAN_EXTRACT_CONCURRENCY", "2")
    assert pipeline_module._extract_concurrency() == 2

    monkeypatch.setenv("PORTOLAN_EXTRACT_CONCURRENCY", "not-a-number")
    assert pipeline_module._extract_concurrency() == 4


# --------------------------------------------------------------------
# Fix wave 2026-09-12: killed-run-against-portolan_demo2 defect.
#
# Observed: two sources came back `done`, with real extraction summaries,
# from a run that got killed mid-flight -- yet the database held zero
# notes and zero claims. Static reading of `portolan.merge.pipeline.
# merge_extraction` and `portolan.notes.persist.write_note` shows every
# claim/note write commits synchronously, well before any status write --
# so a genuinely empty commit trail can't itself explain a `done` source
# with no graph footprint. What the reading DID find: `merge_extraction`
# was ALSO unconditionally writing `sources.extraction_status = 'done'`
# itself, in the same statement as the summary update, duplicating (and,
# for a source that should end up `partial`, briefly contradicting)
# `portolan.pipeline._finish_one`'s own, authoritative status write --
# exactly the kind of double-write a kill lands badly inside. Fixed by
# making `_finish_one`'s call to `_mark_source_status` the ONLY place
# `extraction_status` is ever written once a source reaches merge.
#
# This test proves the fix two ways, using `portolan.extract.backends.
# FakeBackend` (a real `Backend`, not a mocked LLM) plumbed straight
# through the actual `_prepare_one`/`_extract_segments`/`_finish_one`
# chain against a real Postgres connection wrapped to record the order
# SQL statements execute in:
#
# 1. `extraction_status` is written exactly once for the source (the
#    duplicate premature write is gone).
# 2. That one write comes strictly after every claim/note write already
#    recorded for the source -- "done" is never visible before the notes
#    and claims it describes are.
# --------------------------------------------------------------------


class _OrderTrackingCursor:
    """Wraps a real psycopg cursor, classifying every `execute()` call's
    SQL text into `order` (shared with the connection wrapper below) so a
    test can assert on the relative order two kinds of writes happened in,
    without needing to reach into `portolan.merge`/`portolan.notes`
    internals or monkeypatch across several modules' own `from ... import`
    bindings (`write_note`/`_insert_claim` are imported by name into three
    different modules -- patching the DB boundary once here is simpler and
    proves the thing that actually matters: what really executed, and in
    what order)."""

    def __init__(self, real_cursor, order: list[str]) -> None:
        self._real = real_cursor
        self._order = order

    def __enter__(self) -> _OrderTrackingCursor:
        self._real.__enter__()
        return self

    def __exit__(self, *exc_info) -> None:
        return self._real.__exit__(*exc_info)

    def execute(self, query, params=None):
        text = query if isinstance(query, str) else str(query)
        upper = " ".join(text.upper().split())
        if upper.startswith("INSERT INTO CLAIMS"):
            self._order.append("claim_insert")
        elif upper.startswith("INSERT INTO NOTES") or upper.startswith("UPDATE NOTES SET"):
            self._order.append("note_write")
        elif (
            upper.startswith("UPDATE SOURCES SET")
            and "EXTRACTION_STATUS" in upper
            and "FILTER_REASON" not in upper
        ):
            # Any post-filter write to `sources.extraction_status` -- this
            # is deliberately broad enough to catch BOTH
            # `_mark_source_status`'s real done/partial/failed write AND
            # the bug this test guards against (`merge_extraction` used to
            # ALSO write `extraction_status = 'done'` itself, a second,
            # differently-shaped statement with no "attempts" clause).
            # Excludes `portolan.ingest.filter.apply_filter_decision`'s own
            # `extraction_status` write (always paired with `filter_reason`
            # in the same statement) -- that one happens during
            # `_prepare_one`, well before merge, and isn't what this test
            # is checking the ordering of.
            self._order.append("status_write")
        if params is None:
            return self._real.execute(query)
        return self._real.execute(query, params)

    def __getattr__(self, name):
        return getattr(self._real, name)


class _OrderTrackingConnection:
    """Same real connection, `cursor()` handing back `_OrderTrackingCursor`
    instead of the plain one -- every other attribute (`commit`,
    `autocommit`, ...) passes straight through to the real connection."""

    def __init__(self, real_conn, order: list[str]) -> None:
        self._real = real_conn
        self._order = order

    def cursor(self, *args, **kwargs):
        return _OrderTrackingCursor(self._real.cursor(*args, **kwargs), self._order)

    def __getattr__(self, name):
        return getattr(self._real, name)


def test_status_done_is_written_only_after_merge_commits_notes_and_claims(db, tmp_path) -> None:
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO registry_entities (kind, crm_id, canonical_name) "
            "VALUES ('account', 'A-ORDER-1', 'Order Test Co')"
        )

    registry = Registry.load(db)
    connector = _FakeConcurrencyConnector(1)
    ref = connector.list_sources(None)[0]
    settings = _settings(tmp_path)
    store = LocalStore(settings.store_root)
    embedder = FakeEmbedder()

    canned = ExtractionResult(
        source_id=ref.source_id,
        summary="Order Test Co raised a real risk in this call.",
        facts=[
            Fact(
                id="fact-1",
                type="risk",
                claim="Order Test Co is worried about renewal.",
                entities=["Order Test Co"],
                locator="00:00:05",
                sentiment=-1,
                confidence=0.9,
            )
        ],
        actions=[],
        coverage=Coverage(segments_read=1, segments_total=1),
    )
    backend = FakeBackend({ref.source_id: canned})

    def _unexpected_compare_call(**kwargs):
        raise AssertionError(
            "no existing claims yet -- gate() should skip straight to 'new' "
            "without ever calling the compare LLM"
        )

    llm = FakeLLMClient(_unexpected_compare_call)

    order: list[str] = []
    wrapped_conn = _OrderTrackingConnection(db, order)

    prepared = pipeline_module._prepare_one(
        conn=wrapped_conn,
        connector=connector,
        ref=ref,
        registry=registry,
        store=store,
        settings=settings,
        backend=backend,
        system_prompt="system prompt (unused by FakeBackend)",
        vocabulary=Vocabulary(),
        sink=InMemorySink(),
        sleep=lambda _seconds: None,
    )
    assert isinstance(prepared, pipeline_module._PreparedSource), prepared

    extracted = pipeline_module._extract_segments(
        prepared,
        backend=backend,
        system_prompt="system prompt (unused by FakeBackend)",
        settings=settings,
        sleep=lambda _seconds: None,
    )
    assert not extracted.segment_failures

    outcome = pipeline_module._finish_one(
        extracted,
        conn=wrapped_conn,
        llm=llm,
        embedder=embedder,
        registry=registry,
        merge_extraction=merge_extraction,
        sink=InMemorySink(),
        sleep=lambda _seconds: None,
    )

    assert outcome.status == "done"
    assert outcome.facts_count == 1

    # The graph really did get written -- this isn't a degenerate
    # zero-write run.
    claim_writes = [i for i, event in enumerate(order) if event == "claim_insert"]
    note_writes = [i for i, event in enumerate(order) if event == "note_write"]
    status_writes = [i for i, event in enumerate(order) if event == "status_write"]
    assert claim_writes, order
    assert note_writes, order

    # The fix: exactly one `extraction_status` write for this source (the
    # duplicate, premature write inside `merge_extraction` is gone) --
    assert len(status_writes) == 1, (
        f"expected exactly one extraction_status write, got {len(status_writes)}: {order}"
    )
    # -- and it comes strictly after every claim/note write already
    # recorded: 'done' is never visible before the notes and claims it
    # describes are.
    assert status_writes[0] > max(claim_writes + note_writes), order

    with db.cursor() as cur:
        cur.execute(
            "SELECT extraction_status, summary FROM sources WHERE slug = %s",
            (prepared.source_row.slug,),
        )
        db_status, db_summary = cur.fetchone()
        cur.execute(
            "SELECT COUNT(*) FROM claims WHERE source_slug = %s", (prepared.source_row.slug,)
        )
        (claim_count,) = cur.fetchone()

    assert db_status == "done"
    assert db_summary == canned.summary
    assert claim_count == 1


def test_merge_empty_is_logged_when_extraction_had_facts_but_nothing_resolved(db, tmp_path) -> None:
    """The other half of the fix: a non-empty extraction (real facts, real
    summary) whose facts' entities resolve against nothing in the registry
    legitimately merges to zero notes/claims (`merge_fact` -> every fact
    `queued_unknown`). Not a bug -- but it must be visible in the
    changelog with the counts, not indistinguishable from the ordering
    defect this fix wave confirmed and fixed above."""
    registry = Registry.load(db)  # deliberately empty: nothing for "Nobody Inc" to resolve to
    connector = _FakeConcurrencyConnector(1)
    ref = connector.list_sources(None)[0]
    settings = _settings(tmp_path)
    store = LocalStore(settings.store_root)
    embedder = FakeEmbedder()

    canned = ExtractionResult(
        source_id=ref.source_id,
        summary="A real summary about a company not in the registry.",
        facts=[
            Fact(
                id="fact-1",
                type="risk",
                claim="Nobody Inc is worried about renewal.",
                entities=["Nobody Inc"],
                locator="00:00:05",
                sentiment=-1,
                confidence=0.9,
            )
        ],
        actions=[],
        coverage=Coverage(segments_read=1, segments_total=1),
    )
    backend = FakeBackend({ref.source_id: canned})
    llm = FakeLLMClient(lambda **kwargs: (_ for _ in ()).throw(AssertionError("unexpected")))

    prepared = pipeline_module._prepare_one(
        conn=db,
        connector=connector,
        ref=ref,
        registry=registry,
        store=store,
        settings=settings,
        backend=backend,
        system_prompt="system prompt (unused by FakeBackend)",
        vocabulary=Vocabulary(),
        sink=InMemorySink(),
        sleep=lambda _seconds: None,
    )
    assert isinstance(prepared, pipeline_module._PreparedSource), prepared

    extracted = pipeline_module._extract_segments(
        prepared,
        backend=backend,
        system_prompt="system prompt (unused by FakeBackend)",
        settings=settings,
        sleep=lambda _seconds: None,
    )

    outcome = pipeline_module._finish_one(
        extracted,
        conn=db,
        llm=llm,
        embedder=embedder,
        registry=registry,
        merge_extraction=merge_extraction,
        sink=InMemorySink(),
        sleep=lambda _seconds: None,
    )

    # Legitimately merged nothing -- still `done` (extraction succeeded;
    # merge correctly had nothing resolvable to write), never `failed`.
    assert outcome.status == "done"

    with db.cursor() as cur:
        cur.execute(
            "SELECT extraction_status, summary FROM sources WHERE slug = %s",
            (prepared.source_row.slug,),
        )
        db_status, db_summary = cur.fetchone()
        cur.execute(
            "SELECT COUNT(*) FROM claims WHERE source_slug = %s", (prepared.source_row.slug,)
        )
        (claim_count,) = cur.fetchone()
        cur.execute(
            "SELECT summary, details FROM changelog WHERE kind = 'merge_empty' "
            "ORDER BY id DESC LIMIT 1"
        )
        changelog_row = cur.fetchone()

    assert db_status == "done"
    assert db_summary == canned.summary
    assert claim_count == 0, "this fact was never resolvable -- it must not have landed a claim"

    assert changelog_row is not None, "expected a merge_empty changelog row"
    changelog_summary, changelog_details = changelog_row
    assert prepared.source_row.slug in changelog_summary
    assert "1 fact" in changelog_summary
    assert changelog_details["facts_extracted"] == 1
    assert changelog_details["actions_extracted"] == 0


# --------------------------------------------------------------------
# Fix wave 2026-09-12, second defect (found running this pipeline end to
# end against a real `OpenRouterClient`, reported live by the coordinator
# watching the run's `usage_events` table): `_extract_segments` built its
# own per-source `InMemorySink` and handed it to `backend.extract()`, but
# never read it back -- for a `LLMClient` that provides `using_sink()`
# (today, only `OpenRouterClient`, the backend this project actually
# extracts against), THAT sink is where the real usage lands during the
# call (see `LLMClientBackend.extract`'s docstring), and it was thrown
# away the moment `_extract_segments` returned. A live run's
# `usage_events` showed rows for `surface='merge'` (compare calls, made
# straight through the client, unaffected) but zero for `surface='extract'`
# despite dozens of real, billed extraction calls having already
# happened. Fixed by carrying that sink on `_ExtractedSource` and having
# `_finish_one` replay its events into the run's real sink before doing
# anything else.
# --------------------------------------------------------------------


class _UsingSinkFakeClient:
    """A minimal stand-in for `OpenRouterClient`'s one distinguishing
    feature here: `using_sink()` temporarily redirects this client's
    metering to a caller-supplied sink/tenant, exactly like the real
    thing (`portolan/llm/openrouter.py`) -- everything else about talking
    to a real model is irrelevant to this test, so it's not reproduced.
    `sink=` at construction stands in for "whatever this client's own
    persistent sink is" (never actually receives an event in this test,
    same as production: `using_sink` always wins while it's active)."""

    def __init__(self, canned_data: dict, *, sink=None, tenant: str = "dev") -> None:
        self.sink = sink if sink is not None else InMemorySink()
        self.tenant = tenant
        self._canned_data = canned_data

    def complete(self, *, system, user, json_schema, surface, model=None):
        del system, user, json_schema, model
        from portolan.llm.base import LLMResult
        from portolan.metering import metered

        with metered(self.sink, self.tenant, surface, "fake-model") as usage:
            usage.input_tokens = 10
            usage.output_tokens = 5
        return LLMResult(
            text="{}",
            data=self._canned_data,
            input_tokens=10,
            output_tokens=5,
            cached_tokens=0,
            model="fake-model",
        )

    @contextmanager
    def using_sink(self, sink, tenant):
        previous_sink, previous_tenant = self.sink, self.tenant
        self.sink, self.tenant = sink, tenant
        try:
            yield
        finally:
            self.sink, self.tenant = previous_sink, previous_tenant


def test_extract_usage_events_survive_into_the_runs_real_sink(db, tmp_path) -> None:
    registry = Registry.load(db)  # empty: this source's (zero) facts need nothing
    connector = _FakeConcurrencyConnector(1)
    ref = connector.list_sources(None)[0]
    settings = _settings(tmp_path)
    store = LocalStore(settings.store_root)
    embedder = FakeEmbedder()

    canned_data = {
        "schema_version": "0.1",
        "source_id": ref.source_id,
        "summary": "nothing notable",
        "entities": [],
        "facts": [],
        "actions": [],
        "unknown_entities": [],
        "coverage": {"segments_read": 1, "segments_total": 1},
    }
    using_sink_client = _UsingSinkFakeClient(canned_data)
    backend = LLMClientBackend(using_sink_client, model=settings.extraction_model)

    prepared = pipeline_module._prepare_one(
        conn=db,
        connector=connector,
        ref=ref,
        registry=registry,
        store=store,
        settings=settings,
        backend=backend,
        system_prompt="system prompt (unused)",
        vocabulary=Vocabulary(),
        sink=InMemorySink(),
        sleep=lambda _seconds: None,
    )
    assert isinstance(prepared, pipeline_module._PreparedSource), prepared

    extracted = pipeline_module._extract_segments(
        prepared,
        backend=backend,
        system_prompt="system prompt (unused)",
        settings=settings,
        sleep=lambda _seconds: None,
    )
    # The bug this test guards against: before the fix, this event existed
    # only here, on a sink `_finish_one` never saw.
    assert any(e.surface == "extract" for e in extracted.sink.events), extracted.sink.events

    run_sink = InMemorySink()
    outcome = pipeline_module._finish_one(
        extracted,
        conn=db,
        llm=using_sink_client,
        embedder=embedder,
        registry=registry,
        merge_extraction=merge_extraction,
        sink=run_sink,
        sleep=lambda _seconds: None,
    )

    assert outcome.status == "done"
    extract_events = [e for e in run_sink.events if e.surface == "extract"]
    assert extract_events, (
        f"expected the extract-surface usage event to survive into the run's real "
        f"sink; run_sink had {run_sink.events}"
    )
    # The client's OWN persistent sink must be untouched -- `using_sink`
    # always diverted metering away from it, exactly like production.
    assert using_sink_client.sink is not run_sink
    assert using_sink_client.sink.events == []


# --------------------------------------------------------------------
# Bug hunt 2026-09-18, finding 8: `_run_filter_audit` is "the exact bug
# the fix wave closed for `_extract_segments` [above], left in place one
# function above" -- it built its own throwaway `InMemorySink` and handed
# it to `backend.extract()`, never reading it back. For any client with
# `using_sink()` (`OpenRouterClient`, the backend this project extracts
# against), the sampled 1-2% of excluded sources were extracted and
# billed with the record of it going nowhere.
# --------------------------------------------------------------------


def test_filter_audit_replays_its_extraction_usage_into_the_runs_real_sink(db, tmp_path) -> None:
    from portolan.ingest.filter import FilterDecision
    from portolan.ingest.models import PreparedText

    registry = Registry.load(db)
    settings = _settings(tmp_path)

    canned_data = {
        "schema_version": "0.1",
        "source_id": "S-excluded-sampled",
        "summary": "an internal thread that happened to use a signal word",
        "entities": [],
        "facts": [],
        "actions": [],
        "unknown_entities": [],
        "coverage": {"segments_read": 1, "segments_total": 1},
    }
    using_sink_client = _UsingSinkFakeClient(canned_data)
    backend = LLMClientBackend(using_sink_client, model=settings.extraction_model)

    raw = RawSource(
        source_id="S-excluded-sampled",
        source_type=SourceType.EMAIL,
        source_date=date(2026, 8, 1),
        text="an excluded source, sampled for the filter audit",
    )
    prepared = PreparedText(text=raw.text, locators=[], removed_chars=0)
    decision = FilterDecision(
        include=False, reason="excluded_sampled", matched_entities=[], sampled=True
    )
    run_sink = InMemorySink()

    pipeline_module._run_filter_audit(
        db,
        raw=raw,
        prepared=prepared,
        registry=registry,
        decision=decision,
        vocabulary=Vocabulary(),
        backend=backend,
        system_prompt="system prompt (unused)",
        settings=settings,
        sink=run_sink,
        sleep=lambda _seconds: None,
    )

    extract_events = [e for e in run_sink.events if e.surface == "extract"]
    assert extract_events, (
        f"expected the filter-audit's extraction usage event to survive into the "
        f"run's real sink; run_sink had {run_sink.events}"
    )
    # Same discipline as the extract-segments fix above: the client's OWN
    # persistent sink is untouched -- `using_sink` diverted metering away
    # from it for the duration of the audit's call.
    assert using_sink_client.sink.events == []


# --------------------------------------------------------------------
# Bug hunt 2026-09-18, finding 2: `--dry-run` used to run the ENTIRE
# pipeline (real model calls, notes, claims, the index) and only skip the
# closing lint/changelog write.
# --------------------------------------------------------------------


def test_dry_run_never_calls_the_extractor_or_writes_notes_claims_or_index(db, tmp_path) -> None:
    n = 2
    connector = _FakeConcurrencyConnector(n)
    settings = _settings(tmp_path)
    embedder = FakeEmbedder()
    store = LocalStore(settings.store_root)
    registry = Registry.load(db)

    def _must_not_be_called(**kwargs):
        raise AssertionError(f"--dry-run must never call the extractor: {kwargs.get('surface')!r}")

    llm = FakeLLMClient(_must_not_be_called, sink=InMemorySink())

    report = run(
        db,
        connector=connector,
        llm=llm,
        embedder=embedder,
        registry=registry,
        store=store,
        settings=settings,
        dry_run=True,
        sink=InMemorySink(),
    )

    assert report.sources_seen == n
    assert report.sources_would_extract == n
    assert report.sources_extracted == 0
    assert report.facts_extracted == 0
    assert report.index_chunks_written == 0

    with db.cursor() as cur:
        cur.execute("SELECT count(*) FROM notes")
        (notes_count,) = cur.fetchone()
        cur.execute("SELECT count(*) FROM claims")
        (claims_count,) = cur.fetchone()
    assert notes_count == 0
    assert claims_count == 0

    # Intake and filter still ran -- a dry run previews what's there (the
    # fix's own "stop after the filter stage" instruction) -- so the
    # sources rows exist, but stay `pending`: extraction never happened.
    with db.cursor() as cur:
        cur.execute(
            "SELECT DISTINCT extraction_status FROM sources WHERE slug LIKE 'concurrency-src-%'"
        )
        statuses = {row[0] for row in cur.fetchall()}
    assert statuses == {"pending"}


# --------------------------------------------------------------------
# Bug hunt 2026-09-18, finding 3: re-running the pipeline over an
# unchanged corpus must not re-extract (and re-bill) every source; a
# source already `done`/`partial` at the current extractor model and
# prompt variant is skipped. `--force` bypasses the skip.
# --------------------------------------------------------------------


def _counting_zero_fact_responder(counts: dict[str, int]):
    # `FakeLLMClient`'s own `_call_count` is only bumped on its LIST-of-
    # responses path, not its callable path (`_next_raw` returns before
    # reaching the `finally` that increments it) -- so this test counts
    # extract-surface calls itself instead of relying on that attribute.
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


def test_rerun_on_unchanged_corpus_skips_extraction_and_force_overrides(db, tmp_path) -> None:
    n = 2
    connector = _FakeConcurrencyConnector(n)
    settings = _settings(tmp_path)
    embedder = FakeEmbedder()
    store = LocalStore(settings.store_root)
    registry = Registry.load(db)
    counts: dict[str, int] = {}
    llm = FakeLLMClient(_counting_zero_fact_responder(counts), sink=InMemorySink())

    first_report = run(
        db,
        connector=connector,
        llm=llm,
        embedder=embedder,
        registry=registry,
        store=store,
        settings=settings,
        sink=InMemorySink(),
    )
    assert first_report.sources_extracted == n
    assert first_report.sources_skipped == 0
    calls_after_first = counts["extract_calls"]
    assert calls_after_first == n

    # Re-run over the exact same (unchanged) corpus: every source is
    # skipped, not re-extracted -- and the extractor makes no new calls.
    second_report = run(
        db,
        connector=connector,
        llm=llm,
        embedder=embedder,
        registry=registry,
        store=store,
        settings=settings,
        sink=InMemorySink(),
    )
    assert second_report.sources_extracted == 0
    assert second_report.sources_skipped == n
    assert counts["extract_calls"] == calls_after_first

    # `--force` bypasses the skip and re-extracts regardless.
    third_report = run(
        db,
        connector=connector,
        llm=llm,
        embedder=embedder,
        registry=registry,
        store=store,
        settings=settings,
        force=True,
        sink=InMemorySink(),
    )
    assert third_report.sources_extracted == n
    assert third_report.sources_skipped == 0
    assert counts["extract_calls"] == calls_after_first + n


def test_rerun_with_a_different_extraction_model_does_not_skip(db, tmp_path) -> None:
    # The skip check compares the CURRENT run's extraction_model against
    # what a source was actually extracted with -- a model change (a
    # bench promoting a new default) must re-extract, not skip.
    n = 1
    connector = _FakeConcurrencyConnector(n)
    settings = _settings(tmp_path)
    embedder = FakeEmbedder()
    store = LocalStore(settings.store_root)
    registry = Registry.load(db)
    counts: dict[str, int] = {}
    llm = FakeLLMClient(_counting_zero_fact_responder(counts), sink=InMemorySink())

    run(
        db,
        connector=connector,
        llm=llm,
        embedder=embedder,
        registry=registry,
        store=store,
        settings=settings,
        sink=InMemorySink(),
    )
    calls_after_first = counts["extract_calls"]

    other_settings = settings.model_copy(update={"extraction_model": "a-different-model"})
    second_report = run(
        db,
        connector=connector,
        llm=llm,
        embedder=embedder,
        registry=registry,
        store=store,
        settings=other_settings,
        sink=InMemorySink(),
    )
    assert second_report.sources_extracted == n
    assert second_report.sources_skipped == 0
    assert counts["extract_calls"] == calls_after_first + n
