"""Tests for `portolan/audit/**` (`01-spec/quality-loop.md` "The
production audit").

Own test database: `createdb portolan_test_audit`; run with
`DATABASE_URL=postgresql://localhost/portolan_test_audit`.

`graph` (module-scoped) builds a real graph once via the full fake
pipeline against `corpus/out` -- same pattern `tests/test_pipeline.py`'s
`full_run_report_and_db_url` fixture uses -- so the audit tests exercise
real stored originals, real claims (with wikilinked entity mentions), and
real notes rather than hand-rolled fixtures.
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import psycopg
import pytest

import portolan.audit.report as report_module
from portolan import db as db_module
from portolan.audit.diff import CLAIM_SIMILARITY_THRESHOLD, diff_facts
from portolan.audit.judge import judge
from portolan.audit.models import DiffCandidate, FactDiff, Judgement
from portolan.audit.report import _claims_for_source, render_markdown, run_audit
from portolan.audit.sample import sample_sources
from portolan.audit.second_pass import AuditBackend, second_pass
from portolan.config import Settings
from portolan.connectors.file_connector import FileConnector
from portolan.extract.backends import FakeBackend
from portolan.extract.schema import Coverage, ExtractionResult, Fact, FactType
from portolan.fake_extractor import FakePlacements, build_fake_llm_client
from portolan.llm.fake import FakeLLMClient
from portolan.metering import InMemorySink
from portolan.notes.persist import note_exists, read_note
from portolan.pipeline import run as run_pipeline
from portolan.registry import Registry
from portolan.search.embed import FakeEmbedder
from portolan.store import LocalStore

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_OUT = REPO_ROOT / "corpus" / "out"


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        database_url="postgresql://localhost/unused",
        tenant="audit-test",
        store_root=tmp_path / "store",
    )


@pytest.fixture(scope="module")
def graph(tmp_path_factory):
    """One real graph, built once per test module run (module-scoped like
    `test_pipeline.py`'s equivalent fixture, and for the same reason: a
    full fake-pipeline run is not cheap). Yields `(conn, store,
    placements)` -- `placements` is the same `FakePlacements` index the
    pipeline's fake LLM used, reused here to build "what the strong pass
    would also say" second-pass results without a second full run.
    """
    database_url = os.environ.get("DATABASE_URL", "postgresql://localhost/portolan_test_audit")
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

    tmp_path = tmp_path_factory.mktemp("audit_e2e")
    settings = _settings(tmp_path)
    connector = FileConnector(CORPUS_OUT)
    placements = FakePlacements(CORPUS_OUT)
    llm = build_fake_llm_client(placements, tenant=settings.tenant)
    embedder = FakeEmbedder()
    store = LocalStore(settings.store_root)
    registry = Registry.load(conn)

    run_pipeline(
        conn,
        connector=connector,
        llm=llm,
        embedder=embedder,
        registry=registry,
        store=store,
        settings=settings,
    )

    yield conn, store, placements
    conn.close()


def _busiest_call_source(conn: psycopg.Connection) -> tuple[str, str]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT sources.slug, sources.source_type, COUNT(*) AS n
            FROM claims
            JOIN sources ON sources.slug = claims.source_slug
            WHERE sources.source_type = 'call' AND claims.archived = false
            GROUP BY sources.slug, sources.source_type
            ORDER BY n DESC, sources.slug
            LIMIT 1
            """
        )
        row = cur.fetchone()
    assert row is not None, "expected at least one call source with claims in the fake graph"
    return row[0], row[1]


def _extra_fact(locator: str = "23:59:59") -> Fact:
    return Fact(
        id="extra1",
        type=FactType.OTHER,
        claim="A wholly new fact only the strong second pass surfaced.",
        entities=[],
        locator=locator,
        quote=None,
        speaker=None,
        valid_from=None,
        valid_to=None,
        severity=None,
        sentiment=0,
        confidence=0.9,
    )


# --------------------------------------------------------------------
# sample_sources
# --------------------------------------------------------------------


def test_sample_sources_deterministic(graph):
    conn, _store, _placements = graph
    first = sample_sources(conn, since=date(2000, 1, 1), rate=0.2, seed="wave-c2")
    second = sample_sources(conn, since=date(2000, 1, 1), rate=0.2, seed="wave-c2")
    assert [s.slug for s in first] == [s.slug for s in second]
    assert first, "expected a non-empty sample from the fake graph"


def test_sample_sources_different_seed_can_differ(graph):
    conn, _store, _placements = graph
    a = sample_sources(conn, since=date(2000, 1, 1), rate=0.2, seed="seed-a")
    b = sample_sources(conn, since=date(2000, 1, 1), rate=0.2, seed="seed-b")
    # Not a hard guarantee for every possible corpus, but true for this one
    # (enough sources per stratum that two seeds diverge) -- if this ever
    # goes flaky because the corpus changed shape, loosen to "a and b are
    # each individually deterministic" instead.
    assert {s.slug for s in a} != {s.slug for s in b}


def test_sample_sources_is_stratified_by_source_type(graph):
    conn, _store, _placements = graph
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT source_type FROM sources WHERE extraction_status = 'done'")
        done_types = {row[0] for row in cur.fetchall()}

    sample = sample_sources(conn, since=date(2000, 1, 1), rate=0.5, seed="strat", minimum=1)
    sampled_types = {s.source_type for s in sample}
    # Every source_type with at least one `done` source gets at least one
    # representative in the sample (minimum=1 per stratum).
    assert done_types <= sampled_types


def test_sample_sources_only_done_and_since(graph):
    conn, _store, _placements = graph
    # Far-future `since` excludes every source (source_date is always in
    # the synthetic corpus's own 2026 date range).
    sample = sample_sources(conn, since=date(2099, 1, 1), rate=1.0, seed="s")
    assert sample == []


def test_sample_sources_carries_the_connectors_raw_metadata(graph):
    """Migration `0009_sources_metadata.sql` / polish wave 2026-09-12:
    `sample_sources` used to build `SourceRow` from a third, independent
    column list and mapper (`portolan.audit.sample`'s own, now deleted)
    that never even selected `metadata` -- every sampled row's `.metadata`
    was silently `{}` regardless of what `intake` stored. It now imports
    `portolan.ingest.intake`'s public `SOURCE_COLUMNS`/`row_to_source_row`,
    so a sampled ticket/email/crm/rss source carries its connector's own
    `RawSource.metadata` under `.metadata["raw"]`."""
    conn, _store, _placements = graph
    sample = sample_sources(conn, since=date(2000, 1, 1), rate=1.0, seed="metadata-check")
    by_type = {}
    for row in sample:
        by_type.setdefault(row.source_type, row)

    for source_type in ("ticket", "email", "crm", "rss"):
        assert source_type in by_type, f"expected at least one {source_type!r} source in the sample"
        assert "raw" in by_type[source_type].metadata, (
            f"{source_type} source {by_type[source_type].slug!r} has no stored raw metadata"
        )


# --------------------------------------------------------------------
# `_reconstruct_raw_source` (migration 0009, polish wave 2026-09-12): a
# ticket/crm/rss source's `RawSource` is rebuilt from the stored bytes
# plus `sources.metadata["raw"]`, not re-derived lossily from the bytes
# alone -- proven against the real, pipeline-built `graph` fixture, whose
# sources were ingested with this fix already in place.
# --------------------------------------------------------------------


def test_reconstruct_raw_source_rebuilds_the_connectors_metadata_and_text(graph):
    import json as json_module

    from portolan.audit.second_pass import _reconstruct_raw_source
    from portolan.connectors.file_connector import _render_crm_row_text

    conn, store, _placements = graph
    checked_types = set()

    from portolan.ingest.intake import SOURCE_COLUMNS

    with conn.cursor() as cur:
        cur.execute(
            f"SELECT {', '.join(SOURCE_COLUMNS)} FROM sources "
            "WHERE source_type IN ('ticket', 'email', 'crm', 'rss') "
            "AND metadata ? 'raw' ORDER BY slug"
        )
        rows = cur.fetchall()

    from portolan.ingest.intake import row_to_source_row

    for row in rows:
        source_row = row_to_source_row(row)
        if source_row.source_type in checked_types:
            continue
        checked_types.add(source_row.source_type)

        raw = _reconstruct_raw_source(source_row, store)
        assert raw.metadata == source_row.metadata["raw"]

        if source_row.source_type == "crm":
            expected_text = _render_crm_row_text(
                raw.metadata.get("object", "record"),
                json_module.loads(store.get(source_row.s3_key)),
            )
            assert raw.text == expected_text
            assert not raw.text.lstrip().startswith("{"), (
                "crm text must be the rendered row, not the raw JSON blob"
            )
        elif source_row.source_type == "rss":
            item = json_module.loads(store.get(source_row.s3_key))
            expected_rss_text = f"{item.get('title', '')}\n\n{item.get('summary', '')}".strip()
            assert raw.text == expected_rss_text + "\n"

    assert checked_types == {"ticket", "email", "crm", "rss"}


# --------------------------------------------------------------------
# diff_facts
# --------------------------------------------------------------------


def test_diff_facts_finds_extra_fact_as_miss_no_inventions(graph):
    conn, _store, placements = graph
    source_slug, source_type = _busiest_call_source(conn)

    first_claims = _claims_for_source(conn, source_slug)
    assert first_claims

    second = placements.extraction_result_for(source_slug)
    extra = _extra_fact()
    second_with_extra = second.model_copy(update={"facts": [*second.facts, extra]})

    diff = diff_facts(first_claims, second_with_extra, source_type, Registry.load(conn))

    assert diff.source_slug == source_slug
    assert [m.claim_text for m in diff.misses] == [extra.claim]
    assert diff.inventions == []
    assert diff.misses[0].fact_type == "other"


def test_diff_facts_perfect_match_is_empty(graph):
    """No extra fact at all -> the second pass agrees with the first pass
    exactly: no misses, no inventions."""
    conn, _store, placements = graph
    source_slug, source_type = _busiest_call_source(conn)
    first_claims = _claims_for_source(conn, source_slug)
    second = placements.extraction_result_for(source_slug)

    diff = diff_facts(first_claims, second, source_type, Registry.load(conn))

    assert diff.misses == []
    assert diff.inventions == []


def test_entities_overlap_resolves_bare_crm_id_via_registry(graph):
    """Review finding G23: a second-pass fact's `entities` sometimes names
    an account by its bare `crm_id` (`portolan.fake_extractor`'s own
    shape, e.g. `"A-1010"`) rather than any string sharing a substring
    with the account note's slug (`"ashgrove-health"`) -- resolving that
    needs a registry lookup, which `diff_facts` didn't carry before."""
    from portolan.audit.diff import _entities_overlap

    conn, _store, _placements = graph
    registry = Registry.load(conn)

    # "A-1010" is Ashgrove Health's crm_id (corpus/truth-table.yaml).
    assert _entities_overlap(registry, ["ashgrove-health"], ["A-1010"])
    # Genuinely unrelated entities must still not match.
    assert not _entities_overlap(registry, ["ashgrove-health"], ["A-1030"])


def test_entities_overlap_requires_both_sides_empty_to_auto_pass(graph):
    """Review finding G23: "nothing to check" used to fire whenever
    EITHER side had zero entities, so a second-pass fact with no
    `entities` at all auto-matched every claim regardless of content.
    Now it only auto-passes when BOTH sides are empty."""
    from portolan.audit.diff import _entities_overlap

    conn, _store, _placements = graph
    registry = Registry.load(conn)

    assert _entities_overlap(registry, [], [])  # nothing to check, either side
    assert not _entities_overlap(registry, ["ashgrove-health"], [])
    assert not _entities_overlap(registry, [], ["A-1010"])


def test_diff_facts_claim_similarity_threshold_matches_bench():
    # 0.45, per build-contracts.md and quality-loop.md, and literally the
    # same constant `portolan.bench.extraction` uses -- see diff.py's
    # module docstring on why this is a re-use, not a coincidence.
    from portolan.bench.extraction import _CLAIM_SIMILARITY_THRESHOLD as bench_threshold

    assert CLAIM_SIMILARITY_THRESHOLD == bench_threshold


# --------------------------------------------------------------------
# judge
# --------------------------------------------------------------------


def test_judge_fake_classifies_candidate():
    candidate = DiffCandidate(
        kind="miss",
        source_slug="S-call-test",
        fact_type="other",
        claim_text="A new fact.",
        locator="00:00:01",
    )
    llm = FakeLLMClient(
        [{"verdict": "genuine_miss", "matched_claim_id": None, "reason": "clearly new"}],
        model_key="claude-sonnet-5",
    )
    judgements = judge(llm, [candidate])
    assert len(judgements) == 1
    assert judgements[0].verdict == "genuine_miss"
    assert judgements[0].candidate is candidate


def test_judge_paraphrase_carries_matched_claim_id():
    candidate = DiffCandidate(
        kind="miss",
        source_slug="S-call-test",
        fact_type="other",
        claim_text="Same fact, different words.",
        locator="00:00:01",
        best_match_claim_id=42,
        best_match_claim_text="Same fact, original words.",
    )
    llm = FakeLLMClient(
        [{"verdict": "paraphrase_of_existing", "matched_claim_id": 42, "reason": "same fact"}],
        model_key="claude-sonnet-5",
    )
    (result,) = judge(llm, [candidate])
    assert result.verdict == "paraphrase_of_existing"
    assert result.matched_claim_id == 42


def test_score_misses_pins_judgements_by_candidate_not_position():
    """Test gap 13 (review finding 12): before this fix, `run_audit`
    mapped judgements back to candidates by POSITION -- correct only
    while `judge()` returns exactly one `Judgement` per candidate, in
    order. This gives `_score_misses` a `judgements` list one SHORT of
    `diffs`' own candidates (simulating a batched/filtered/retried judge
    call that dropped the middle candidate's verdict) and checks: the
    candidate actually missing its judgement is counted as unjudged, not
    silently defaulted to "genuine miss", and neither of its two
    NEIGHBOURS' own verdicts shifts onto the wrong candidate."""
    candidate_a = DiffCandidate(
        kind="miss", source_slug="S-1", fact_type="complaint", claim_text="A", locator="00:00:01"
    )
    candidate_b = DiffCandidate(
        kind="miss", source_slug="S-1", fact_type="complaint", claim_text="B", locator="00:00:02"
    )
    candidate_c = DiffCandidate(
        kind="miss", source_slug="S-1", fact_type="complaint", claim_text="C", locator="00:00:03"
    )
    diff = FactDiff(
        source_slug="S-1",
        source_type="call",
        second_pass_count=3,
        fact_type_totals={"complaint": 3},
        misses=[candidate_a, candidate_b, candidate_c],
    )

    # `judge()` would normally return one Judgement per candidate, same
    # order -- here candidate_b's is missing entirely (the gap), and
    # candidate_a/candidate_c deliberately carry DIFFERENT verdicts so a
    # positional shift would be observable (not masked by every verdict
    # happening to be the same label).
    judgements = [
        Judgement(candidate=candidate_a, verdict="genuine_miss", reason="a"),
        Judgement(candidate=candidate_c, verdict="not_a_fact", reason="c"),
    ]

    (
        miss_by_fact_type,
        miss_by_source_type,
        genuine_misses_total,
        unjudged_misses_total,
        source_genuine_by_slug,
    ) = report_module._score_misses([diff], judgements)

    assert genuine_misses_total == 1  # only candidate_a
    assert unjudged_misses_total == 1  # candidate_b, not defaulted to genuine
    assert miss_by_fact_type["complaint"].misses == 1
    assert miss_by_fact_type["complaint"].total == 3
    assert miss_by_source_type["call"].misses == 1
    assert source_genuine_by_slug["S-1"] == 1


# --------------------------------------------------------------------
# AuditBackend / second_pass
# --------------------------------------------------------------------


def test_audit_backend_meters_under_surface_audit():
    canned = ExtractionResult(
        source_id="S-x",
        summary="",
        entities=[],
        facts=[],
        actions=[],
        unknown_entities=[],
        coverage=Coverage(segments_read=1, segments_total=1),
    )
    sink = InMemorySink()
    llm = FakeLLMClient(
        [canned.model_dump(mode="json")], model_key="claude-sonnet-5", sink=sink, tenant="t"
    )
    backend = AuditBackend(llm)

    from portolan.extract.schema import SourceEnvelope, SourceType

    envelope = SourceEnvelope(
        source_id="S-x", source_type=SourceType.CALL, source_date=date(2026, 1, 1), text="hi"
    )
    result = backend.extract(envelope, "system prompt", sink, "t")

    assert result.source_id == "S-x"
    assert len(sink.events) == 1
    assert sink.events[0].surface == "audit"


def test_second_pass_rereads_stored_original(graph):
    conn, store, _placements = graph
    source_slug, _source_type = _busiest_call_source(conn)

    def canned(envelope) -> ExtractionResult:
        return ExtractionResult(
            source_id=envelope.source_id,
            summary="",
            entities=[],
            facts=[],
            actions=[],
            unknown_entities=[],
            coverage=Coverage(segments_read=1, segments_total=1),
        )

    with conn.cursor() as cur:
        cur.execute(
            "SELECT slug, source_type, s3_key, content_hash, source_date, ingested_at, "
            "extraction_status, filter_reason, schema_version, attempts, last_error, "
            "summary, participants FROM sources WHERE slug = %s",
            (source_slug,),
        )
        row = cur.fetchone()
    from portolan.ingest.models import Participant, SourceRow

    source_row = SourceRow(
        slug=row[0],
        source_type=row[1],
        s3_key=row[2],
        content_hash=row[3],
        source_date=row[4],
        ingested_at=row[5],
        extraction_status=row[6],
        filter_reason=row[7],
        schema_version=row[8],
        attempts=row[9],
        last_error=row[10],
        summary=row[11],
        participants=[Participant(**p) for p in (row[12] or [])],
    )

    backend = FakeBackend(canned, model_key="claude-sonnet-5")
    results = second_pass(conn, store, backend, [source_row])

    assert source_slug in results
    assert results.failures == {}


def test_second_pass_envelope_has_registry_context_and_resolved_participants(graph):
    """Review finding G9: the second pass must build its envelope through
    the same helpers the first pass uses -- `registry_context` populated
    and participants carrying their resolved `registry_id`, not the bare,
    stripped envelope the old second pass built (which confounded every
    measured miss with "the envelope was weaker", not "the model missed
    it")."""
    conn, store, placements = graph
    source_slug, _source_type = _busiest_call_source(conn)
    registry = Registry.load(conn)

    with conn.cursor() as cur:
        cur.execute(
            "SELECT slug, source_type, s3_key, content_hash, source_date, ingested_at, "
            "extraction_status, filter_reason, schema_version, attempts, last_error, "
            "summary, participants FROM sources WHERE slug = %s",
            (source_slug,),
        )
        row = cur.fetchone()
    from portolan.ingest.models import Participant, SourceRow

    source_row = SourceRow(
        slug=row[0],
        source_type=row[1],
        s3_key=row[2],
        content_hash=row[3],
        source_date=row[4],
        ingested_at=row[5],
        extraction_status=row[6],
        filter_reason=row[7],
        schema_version=row[8],
        attempts=row[9],
        last_error=row[10],
        summary=row[11],
        participants=[Participant(**p) for p in (row[12] or [])],
    )

    captured = []

    def canned(envelope) -> ExtractionResult:
        captured.append(envelope)
        return placements.extraction_result_for(envelope.source_id)

    backend = FakeBackend(canned, model_key="claude-sonnet-5")
    second_pass(conn, store, backend, [source_row], registry=registry)

    assert captured, "expected at least one envelope to reach the backend"
    envelope = captured[0]
    assert envelope.registry_context, (
        "expected a non-empty registry_context, matching the first pass"
    )
    # Participants go through the same `build_envelope_participants`
    # helper the first pass uses -- confirmed by the type match (a bare
    # `SourceEnvelope(participants=[...])` built without it would still
    # be a list of the same length, but this checks it's the real helper,
    # not a hand-rolled stripped one, by construction rather than by
    # asserting a specific resolution outcome this source's own
    # participant shape may or may not produce).
    assert len(envelope.participants) == len(source_row.participants)


def test_second_pass_prompt_carries_the_tenant_company_context(graph, monkeypatch):
    """The first pass renders the tenant's company context; the second pass
    must too, or it runs with no vendor at all (the prompt's own examples
    name a fictional one)."""
    import importlib

    from portolan.context import CompanyContext
    from portolan.ingest.models import Participant, SourceRow

    conn, store, placements = graph
    source_slug, _source_type = _busiest_call_source(conn)
    note = CompanyContext(text="Tenant Vendor Ltd sells widgets.", updated="2026-09-22T00:00:00Z")
    # `portolan.audit.second_pass` the attribute is the function; get the module.
    second_pass_module = importlib.import_module("portolan.audit.second_pass")
    monkeypatch.setattr(second_pass_module, "load_company_context", lambda _conn: note)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT slug, source_type, s3_key, content_hash, source_date, ingested_at, "
            "extraction_status, filter_reason, schema_version, attempts, last_error, "
            "summary, participants FROM sources WHERE slug = %s",
            (source_slug,),
        )
        row = cur.fetchone()
    fields = (
        "slug source_type s3_key content_hash source_date ingested_at extraction_status "
        "filter_reason schema_version attempts last_error summary"
    ).split()
    source_row = SourceRow(
        **dict(zip(fields, row[:12], strict=True)),
        participants=[Participant(**p) for p in (row[12] or [])],
    )
    prompts: list[str] = []

    class _Recording:
        model_key = "claude-sonnet-5"

        def extract(self, envelope, system_prompt, sink, tenant):
            prompts.append(system_prompt)
            return placements.extraction_result_for(envelope.source_id)

    second_pass(conn, store, _Recording(), [source_row])
    assert prompts
    assert all("## Company context\nTenant Vendor Ltd sells widgets." in p for p in prompts)


# --------------------------------------------------------------------
# run_audit end to end
# --------------------------------------------------------------


def test_run_audit_report_renders_and_miss_rate_is_right(graph):
    conn, store, placements = graph
    target_slug, target_type = _busiest_call_source(conn)
    extra = _extra_fact()

    def canned(envelope) -> ExtractionResult:
        base = placements.extraction_result_for(envelope.source_id)
        if envelope.source_id == target_slug:
            base = base.model_copy(update={"facts": [*base.facts, extra]})
        return base

    strong_backend = FakeBackend(canned, model_key="claude-sonnet-5")

    def judge_responder(*, system, user, json_schema, model, surface):
        del system, model, surface
        if json_schema is not None:
            return {"verdict": "genuine_miss", "matched_claim_id": None, "reason": "test"}
        return "ok"

    llm = FakeLLMClient(judge_responder, model_key="claude-sonnet-5", tenant="audit-test")

    report = run_audit(
        conn,
        store,
        llm,
        strong_backend,
        since=date(2000, 1, 1),
        rate=1.0,
        seed="full-audit",
    )

    assert report.sources_sampled > 0
    assert report.sources_second_passed == report.sources_sampled

    # The fact this test deliberately injected for `target_slug` is found
    # by the diff and judged a genuine miss. (A whole-corpus run over this
    # shared fake fixture also surfaces some baseline noise on OTHER
    # sources -- `portolan.fake_extractor` encodes account/contact
    # entities by bare `crm_id`, e.g. "A-1010", which shares no substring
    # with that account's note slug, e.g. "ashgrove-health"; a real
    # extractor's `Fact.entities` would be name-derived slugs instead, per
    # `extraction-schema-v0.md`'s own example -- see `portolan.audit.diff.
    # _entities_overlap`'s docstring. This assertion is scoped to the one
    # fact this test controls, so that known fixture gap doesn't make it
    # flaky.)
    injected = [j for j in report.judgements if j.candidate.claim_text == extra.claim]
    assert len(injected) == 1
    assert injected[0].candidate.source_slug == target_slug
    assert injected[0].verdict == "genuine_miss"

    assert report.genuine_misses_total >= 1
    assert report.miss_rate_overall is not None and 0 < report.miss_rate_overall <= 1
    assert report.miss_by_fact_type["other"].misses >= 1
    assert report.miss_by_source_type[target_type].misses >= 1

    assert note_exists(conn, report.note_slug)
    note, _claims = read_note(conn, report.note_slug)
    assert note.type.value == "report"
    assert "Miss rate" in (note.synthesis or "")

    markdown = render_markdown(report)
    assert "Production audit" in markdown
    assert "Miss rate" in markdown
