"""Regression tests for the engine bug hunt, wave 2, 2026-09-22
(`01-spec/reviews/bughunt-wave2-2026-09-22.md`). Each test names the
finding it pins; each one failed against the code before its fix.

Local test databases only (`tests/conftest.py`'s `db` fixture), fake
model backends, no network, no paid calls.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest
from connector_helpers import FakeTransport, json_response, text_response

import portolan.connectors.hubspot as hubspot_module
import portolan.connectors.salesforce as salesforce_module
import portolan.pipeline as pipeline_module
from portolan.config import Settings
from portolan.connectors.base import SourceRef, parse_datetime
from portolan.connectors.clari import ClariConfig, ClariConnector
from portolan.connectors.file_connector import FileConnector
from portolan.connectors.gong import GongConfig, GongConnector
from portolan.connectors.rss import FeedConfig, RSSConfig, RSSConnector
from portolan.connectors.tickets import (
    ExportTicketConfig,
    ExportTicketConnector,
    ZendeskConfig,
    ZendeskConnector,
)
from portolan.extract.schema import FactType, SourceType
from portolan.ingest.intake import intake
from portolan.ingest.models import RawSource, SourceRow
from portolan.llm.fake import FakeLLMClient
from portolan.merge.actions import merge_action
from portolan.merge.facts import merge_fact
from portolan.metering import InMemorySink
from portolan.notes import ensure_entity_note
from portolan.pipeline import redrive, run
from portolan.registry import Registry
from portolan.search.embed import FakeEmbedder
from portolan.store import LocalStore
from portolan.util import clock
from tests.merge_helpers import (  # noqa: F401
    ASHGROVE,
    build_action,
    build_fact,
    loaded_registry_fixture,
)


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        database_url="postgresql://localhost/unused",
        tenant="test",
        store_root=tmp_path / "store",
    )


# --------------------------------------------------------------------
# Finding 1: a re-run over unchanged sources re-counts them in the
# coverage ledger and re-bills the filter audit.
# --------------------------------------------------------------------

INCLUDED_ID = "wave2-included"
EXCLUDED_ID = "wave2-excluded"


class _OneIncludedOneExcludedConnector:
    """One source the filter includes (an account hint) and one it excludes
    (no participant, hint, registry name or signal word)."""

    name = "wave2"

    def __init__(self) -> None:
        when = datetime(2026, 8, 3, tzinfo=UTC)
        self._refs = [
            SourceRef(source_id=INCLUDED_ID, source_type=SourceType.CALL, updated_at=when, ref=0),
            SourceRef(source_id=EXCLUDED_ID, source_type=SourceType.EMAIL, updated_at=when, ref=1),
        ]

    def list_sources(self, since):
        del since
        return list(self._refs)

    def fetch(self, ref: SourceRef) -> RawSource:
        if ref.source_id == INCLUDED_ID:
            return RawSource(
                source_id=INCLUDED_ID,
                source_type=SourceType.CALL,
                source_date=date(2026, 8, 3),
                account_hints=["TEST-ACCOUNT"],
                text="[00:00:01] Speaker: a customer call with enough text to extract.",
            )
        return RawSource(
            source_id=EXCLUDED_ID,
            source_type=SourceType.EMAIL,
            source_date=date(2026, 8, 3),
            text="Lunch is in the kitchen at noon. Bring a plate.",
        )


def _counting_responder(counts: dict[str, int]):
    def _responder(*, system, user, json_schema, model, surface):
        del system, json_schema, model
        counts[surface] = counts.get(surface, 0) + 1
        return {
            "source_id": "unknown",
            "summary": "nothing to extract",
            "coverage": {"segments_read": 1, "segments_total": 1},
        }

    return _responder


def _ledger(conn) -> list[tuple]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT connector, day, seen, filtered, extracted, partial, failed, sampled "
            "FROM coverage_ledger ORDER BY connector, day"
        )
        return cur.fetchall()


def _filter_audit_rows(conn) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM changelog WHERE kind = 'filter_audit'")
        return cur.fetchone()[0]


def _run(conn, tmp_path, llm, **kwargs):
    settings = _settings(tmp_path)
    return run(
        conn,
        connector=_OneIncludedOneExcludedConnector(),
        llm=llm,
        embedder=FakeEmbedder(),
        registry=Registry.load(conn),
        store=LocalStore(settings.store_root),
        settings=settings,
        sink=InMemorySink(),
        **kwargs,
    )


def test_rerun_over_unchanged_sources_leaves_the_coverage_ledger_and_audit_alone(
    db, tmp_path, monkeypatch
) -> None:
    """Finding 1. The first run sees two sources: one extracted, one
    excluded and sampled for the filter audit (one paid extraction). A
    second run over the same, unchanged sources has nothing new to do: the
    ledger must read the same (it is keyed by the source's own date, so a
    re-count inflates that day's "seen" and "filtered" for good) and the
    audit must not re-extract the excluded source."""
    monkeypatch.setattr(pipeline_module, "_SAMPLE_RATE", 1.0)
    counts: dict[str, int] = {}
    llm = FakeLLMClient(_counting_responder(counts), sink=InMemorySink())

    _run(db, tmp_path, llm)
    ledger_after_first = _ledger(db)
    calls_after_first = dict(counts)
    assert calls_after_first == {"extract": 2}  # one real extraction, one audit
    assert _filter_audit_rows(db) == 1
    assert ledger_after_first == [
        ("call", date(2026, 8, 3), 1, {}, 1, 0, 0, 0),
        ("email", date(2026, 8, 3), 1, {"excluded_sampled": 1}, 0, 0, 0, 1),
    ]

    second = _run(db, tmp_path, llm)

    assert _ledger(db) == ledger_after_first
    assert counts == calls_after_first
    assert _filter_audit_rows(db) == 1
    assert second.sources_skipped == 2


def test_dry_run_then_real_run_counts_each_source_once(db, tmp_path, monkeypatch) -> None:
    """Finding 1, dry-run half. A dry run is a preview: it must not write
    the coverage ledger or stamp the filter decision on the source row, or
    the real run that follows either counts every source twice or (with
    the re-run fix) mistakes the preview's decision for an already-counted
    one and never audits the excluded source."""
    monkeypatch.setattr(pipeline_module, "_SAMPLE_RATE", 1.0)
    counts: dict[str, int] = {}
    llm = FakeLLMClient(_counting_responder(counts), sink=InMemorySink())

    _run(db, tmp_path, llm, dry_run=True)
    assert _ledger(db) == []
    assert counts == {}

    _run(db, tmp_path, llm)

    rows = _ledger(db)
    assert sum(r[2] for r in rows) == 2  # seen
    assert sum(r[4] for r in rows) == 1  # extracted
    assert sum(r[7] for r in rows) == 1  # sampled
    assert sum(sum(r[3].values()) for r in rows) == 1  # filtered
    assert counts == {"extract": 2}
    assert _filter_audit_rows(db) == 1


def test_force_rerun_still_counts_and_audits_again(db, tmp_path, monkeypatch) -> None:
    """Finding 1 guard: `--force` means "do it all again", so the re-run
    fix must not swallow a forced run."""
    monkeypatch.setattr(pipeline_module, "_SAMPLE_RATE", 1.0)
    counts: dict[str, int] = {}
    llm = FakeLLMClient(_counting_responder(counts), sink=InMemorySink())

    _run(db, tmp_path, llm)
    _run(db, tmp_path, llm, force=True)

    assert counts == {"extract": 4}
    assert _filter_audit_rows(db) == 2


# --------------------------------------------------------------------
# Finding 2: a re-crawled, changed source (`<source_id>--v2`) re-creates
# every action it carries -- including ones already marked done.
# --------------------------------------------------------------------


def _source_version(conn, slug: str, source_id: str) -> SourceRow:
    """A `sources` row that is one version of the connector source
    `source_id` (intake names the second version `<source_id>--v2`)."""
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO sources (slug, source_id, source_type, source_date, s3_key) "
            "VALUES (%s, %s, 'ticket', %s, %s)",
            (slug, source_id, date(2026, 8, 20), f"originals/ticket/{slug}"),
        )
    conn.commit()
    return SourceRow(
        slug=slug,
        source_id=source_id,
        source_type="ticket",
        source_date=date(2026, 8, 20),
        ingested_at=datetime.now(UTC),
    )


def _action_rows(conn) -> list[tuple]:
    with conn.cursor() as cur:
        cur.execute("SELECT id, status, source_slug FROM actions ORDER BY id")
        return cur.fetchall()


@pytest.mark.parametrize("entity_id", [ASHGROVE, None])
def test_a_new_version_of_a_source_does_not_recreate_an_action_already_done(
    loaded_registry: Registry, db, entity_id
) -> None:
    """Finding 2. A ticket gains a comment; the re-crawl ingests it as
    `<id>--v2` and re-extracts the commitment in comment 2, which is still
    there, word for word, at the same locator. The action from v1 was
    marked done. Before the fix the v2 merge created a fresh open action
    (the entity pool only holds open actions, and the no-account dedup
    keyed on the exact slug), so a finished commitment came back to the
    Actions page on every re-crawl."""
    v1 = _source_version(db, "S-ticket-900", "S-ticket-900")
    action = build_action(
        "Send Ashgrove Health the corrected May invoice",
        due=date(2026, 8, 28),
        locator="S-ticket-900#2",
    )
    first = merge_action(db, loaded_registry, action, v1, entity_id=entity_id)
    assert first.status == "new"
    with db.cursor() as cur:
        cur.execute("UPDATE actions SET status = 'done' WHERE id = %s", (first.action_id,))
    db.commit()

    v2 = _source_version(db, "S-ticket-900--v2", "S-ticket-900")
    second = merge_action(db, loaded_registry, action, v2, entity_id=entity_id)

    assert second.status == "duplicate"
    assert second.action_id == first.action_id
    assert _action_rows(db) == [(first.action_id, "done", "S-ticket-900")]


def test_a_new_version_does_not_duplicate_an_open_action_with_no_account(
    loaded_registry: Registry, db
) -> None:
    """Finding 2, open half: with no account resolved, the only dedup was
    keyed on the exact source slug, so `--v2` duplicated the open action."""
    v1 = _source_version(db, "S-doc-901", "S-doc-901")
    action = build_action("Send the CSV export spec to the customer", locator="p2")
    first = merge_action(db, loaded_registry, action, v1, entity_id=None)

    v2 = _source_version(db, "S-doc-901--v2", "S-doc-901")
    second = merge_action(db, loaded_registry, action, v2, entity_id=None)

    assert second.status == "duplicate"
    assert [row[0] for row in _action_rows(db)] == [first.action_id]


def test_a_different_source_with_the_same_wording_is_still_a_new_action(
    loaded_registry: Registry, db
) -> None:
    """Finding 2 guard: the family match is by the connector's own source
    id, so an unrelated source saying the same words at the same locator
    is still its own action."""
    a = _source_version(db, "S-doc-902", "S-doc-902")
    b = _source_version(db, "S-doc-903", "S-doc-903")
    action = build_action("Send the CSV export spec to the customer", locator="p2")
    first = merge_action(db, loaded_registry, action, a, entity_id=None)
    with db.cursor() as cur:
        cur.execute("UPDATE actions SET status = 'done' WHERE id = %s", (first.action_id,))
    db.commit()

    second = merge_action(db, loaded_registry, action, b, entity_id=None)

    assert second.status == "new"
    assert len(_action_rows(db)) == 2


# --------------------------------------------------------------------
# Finding 3: a re-crawled version of a source re-merges every fact it
# still carries; the gate never sees superseded or archived claims, so
# an old fact comes back as a live claim, and a flaky "extends" verdict
# duplicates a live one.
# --------------------------------------------------------------------

PRICING_OLD = "Ashgrove Health renewal is quoted at $50,000 for the Tidewater fleet."
PRICING_NEW = "Ashgrove Health renewal is quoted at $40,000 for the Tidewater fleet."


def _pricing_fact(fact_id: str, claim: str, quote: str, locator: str, valid_from: date):
    return build_fact(
        fact_id,
        FactType.PRICING,
        claim,
        ["Ashgrove Health"],
        locator=locator,
        quote=quote,
        valid_from=valid_from,
    )


def _live_claims(conn) -> list[tuple]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, claim_text, source_slug FROM claims "
            "WHERE superseded_at IS NULL AND archived = false ORDER BY id"
        )
        return cur.fetchall()


def test_a_new_version_of_a_source_does_not_bring_a_superseded_claim_back(
    loaded_registry: Registry, db
) -> None:
    """Finding 3. The July email quoted $50,000; an August call superseded
    it with $40,000. The email thread then gains a reply and is re-crawled
    as `--v2`: the $50,000 sentence is still in it, same locator, same
    quote. The gate only offered the live $40,000 claim, the model called
    it `contradicts`, the July date is not later, so the merge downgraded
    to `extends` and inserted $50,000 as a live claim again."""
    embedder = FakeEmbedder()
    email_v1 = _source_version(db, "S-email-910", "S-email-910")
    call = _source_version(db, "S-call-911", "S-call-911")
    old = _pricing_fact("F1", PRICING_OLD, "quoted at $50,000", "S-email-910#p2", date(2026, 7, 1))
    new = _pricing_fact("F2", PRICING_NEW, "quoted at $40,000", "00:12:00", date(2026, 8, 1))

    merge_fact(db, FakeLLMClient([]), embedder, loaded_registry, old, email_v1)
    llm = FakeLLMClient([{"verdict": "contradicts", "reason": "the price moved"}])
    merge_fact(db, llm, embedder, loaded_registry, new, call)
    assert [row[2] for row in _live_claims(db)] == ["S-call-911"]

    email_v2 = _source_version(db, "S-email-910--v2", "S-email-910")
    llm = FakeLLMClient([{"verdict": "contradicts", "reason": "the price moved"}])
    outcome = merge_fact(db, llm, embedder, loaded_registry, old, email_v2)

    assert [row[2] for row in _live_claims(db)] == ["S-call-911"]
    assert outcome.claim_inserted is False


def test_a_new_version_of_a_source_does_not_duplicate_a_live_claim(
    loaded_registry: Registry, db
) -> None:
    """Finding 3, live half: same sentence, same locator, same quote in
    `--v2`. Before the fix this cost a compare call and, on an `extends`
    verdict, wrote a second copy of the claim."""
    embedder = FakeEmbedder()
    v1 = _source_version(db, "S-email-920", "S-email-920")
    fact = _pricing_fact("F1", PRICING_OLD, "quoted at $50,000", "S-email-920#p2", date(2026, 7, 1))
    merge_fact(db, FakeLLMClient([]), embedder, loaded_registry, fact, v1)

    v2 = _source_version(db, "S-email-920--v2", "S-email-920")
    llm = FakeLLMClient([{"verdict": "extends", "reason": "adds nothing, really"}])
    outcome = merge_fact(db, llm, embedder, loaded_registry, fact, v2)

    assert len(_live_claims(db)) == 1
    assert outcome.compare_calls == 0


def test_the_same_quote_from_an_unrelated_source_still_goes_through_the_gate(
    loaded_registry: Registry, db
) -> None:
    """Finding 3 guard: the shortcut is only for versions of one source
    (same `sources.source_id`). A different source repeating the quote is
    a corroboration the compare call decides, as before."""
    embedder = FakeEmbedder()
    a = _source_version(db, "S-email-930", "S-email-930")
    b = _source_version(db, "S-email-931", "S-email-931")
    fact = _pricing_fact("F1", PRICING_OLD, "quoted at $50,000", "p2", date(2026, 7, 1))
    merge_fact(db, FakeLLMClient([]), embedder, loaded_registry, fact, a)

    llm = FakeLLMClient([{"verdict": "same", "reason": "same quote"}])
    outcome = merge_fact(db, llm, embedder, loaded_registry, fact, b)

    assert outcome.compare_calls == 1
    assert outcome.citation_inserted is True


class _TicketThatGainsAComment:
    """One ticket; `grow()` appends a comment, as a re-crawl would see."""

    name = "wave2-ticket"

    def __init__(self) -> None:
        self.comments = ["[1] Ashgrove Health: the renewal is quoted at $50,000 for us."]

    def grow(self) -> None:
        self.comments.append("[2] Halvard Support: thanks, looking into it.")

    def list_sources(self, since):
        del since
        return [
            SourceRef(
                source_id="S-ticket-940",
                source_type=SourceType.TICKET,
                updated_at=datetime(2026, 8, 20, tzinfo=UTC),
                ref=None,
            )
        ]

    def fetch(self, ref: SourceRef) -> RawSource:
        return RawSource(
            source_id="S-ticket-940",
            source_type=SourceType.TICKET,
            source_date=date(2026, 8, 20),
            account_hints=[ASHGROVE],
            text="\n\n".join(self.comments),
        )


def test_a_recrawled_ticket_keeps_one_claim_per_fact_through_the_whole_pipeline(
    loaded_registry: Registry, db, tmp_path
) -> None:
    """Finding 3 end to end: run, the ticket gains a comment, run again.
    The second run ingests `S-ticket-940--v2` and re-extracts the same
    fact. It must not write a second claim, spend a compare call, or log
    a `merge_empty` false alarm for a version whose facts were all known."""
    calls: dict[str, int] = {}

    def responder(*, system, user, json_schema, model, surface):
        del system, user, json_schema, model
        calls[surface] = calls.get(surface, 0) + 1
        if surface == "merge":
            return {"verdict": "extends", "reason": "a flaky verdict"}
        return {
            "source_id": "S-ticket-940",
            "summary": "Ashgrove pricing.",
            "facts": [
                {
                    "id": "f1",
                    "type": "pricing",
                    "claim": PRICING_OLD,
                    "entities": ["Ashgrove Health"],
                    "locator": "S-ticket-940#1",
                    "quote": "the renewal is quoted at $50,000",
                    "sentiment": 0,
                    "confidence": 0.9,
                }
            ],
            "coverage": {"segments_read": 1, "segments_total": 1},
        }

    connector = _TicketThatGainsAComment()
    llm = FakeLLMClient(responder, sink=InMemorySink())
    settings = _settings(tmp_path)

    def _run_once():
        return run(
            db,
            connector=connector,
            llm=llm,
            embedder=FakeEmbedder(),
            registry=Registry.load(db),
            store=LocalStore(settings.store_root),
            settings=settings,
            sink=InMemorySink(),
        )

    _run_once()
    assert len(_live_claims(db)) == 1
    connector.grow()
    second = _run_once()

    assert second.sources_extracted == 1
    assert len(_live_claims(db)) == 1
    assert calls.get("merge", 0) == 0
    with db.cursor() as cur:
        cur.execute("SELECT count(*) FROM changelog WHERE kind = 'merge_empty'")
        assert cur.fetchone()[0] == 0


# --------------------------------------------------------------------
# Finding 4: a `same` verdict against a claim from the same source at the
# same locator writes that claim's own primary citation a second time.
# --------------------------------------------------------------------


def test_a_same_verdict_at_the_claims_own_source_and_locator_adds_no_citation(
    loaded_registry: Registry, db
) -> None:
    """Finding 4. Five-sample extraction unions near-duplicate facts from
    one source: same locator, a slightly different quote and wording. The
    second one gates against the first, the model says `same`, and the
    merge wrote `(this source, this locator)` into `claim_citations`, a
    copy of the claim's own primary citation: the note rendered the marker
    twice and the audit counted the claim as corroborated by itself."""
    embedder = FakeEmbedder()
    source = _source_version(db, "S-email-950", "S-email-950")
    first = _pricing_fact("F1", PRICING_OLD, "quoted at $50,000", "S-email-950#p2", None)
    again = _pricing_fact(
        "F1b",
        "Ashgrove Health renewal was quoted at $50,000 for the Tidewater fleet.",
        "renewal is quoted at $50,000",
        "S-email-950#p2",
        None,
    )
    merge_fact(db, FakeLLMClient([]), embedder, loaded_registry, first, source)
    llm = FakeLLMClient([{"verdict": "same", "reason": "same sentence"}])
    outcome = merge_fact(db, llm, embedder, loaded_registry, again, source)

    assert outcome.status == "same"
    assert outcome.citation_inserted is False
    with db.cursor() as cur:
        cur.execute("SELECT count(*) FROM claim_citations")
        assert cur.fetchone()[0] == 0


# --------------------------------------------------------------------
# Finding 5: vendor connectors date a source by the UTC calendar day of
# its timestamp. The org is America/Phoenix (UTC-7): everything from
# 17:00 to midnight local lands on the next day, and `claims.recorded`,
# the weekly windows and the coverage ledger all read that date.
# --------------------------------------------------------------------

LATE_EVENING = "2026-09-10T01:30:00Z"  # 2026-09-09 18:30 in America/Phoenix
LATE_EVENING_SF = "2026-09-10T01:30:00.000+0000"
LATE_EVENING_MS = str(int(datetime(2026, 9, 10, 1, 30, tzinfo=UTC).timestamp() * 1000))
ORG_DAY = date(2026, 9, 9)


@pytest.fixture()
def phoenix(monkeypatch):
    monkeypatch.setenv("PORTOLAN_TIMEZONE", "America/Phoenix")


def test_gong_and_clari_calls_are_dated_by_the_org_day(phoenix) -> None:
    gong_transport = FakeTransport()
    gong_transport.queue(
        "https://api.gong.io/v2/calls/transcript", json_response({"callTranscripts": []})
    )
    gong_transport.queue("https://api.gong.io/v2/calls/extensive", json_response({"calls": []}))
    gong = GongConnector(GongConfig("key", "secret"), transport=gong_transport)
    gong_ref = SourceRef(
        source_id="gong-1",
        source_type=SourceType.CALL,
        updated_at=parse_datetime(LATE_EVENING),
        ref={"call": {"id": "1", "started": LATE_EVENING}},
    )

    clari_transport = FakeTransport()
    clari_transport.queue("https://api.clari.com/call-details", json_response({"turns": []}))
    clari = ClariConnector(ClariConfig("key", "password"), transport=clari_transport)
    clari_ref = SourceRef(
        source_id="clari-1",
        source_type=SourceType.CALL,
        updated_at=parse_datetime(LATE_EVENING),
        ref={"conversation": {"id": "1", "startTime": LATE_EVENING}},
    )

    assert gong.fetch(gong_ref).source_date == ORG_DAY
    assert clari.fetch(clari_ref).source_date == ORG_DAY


def test_zendesk_and_exported_tickets_are_dated_by_the_org_day(phoenix, tmp_path) -> None:
    transport = FakeTransport()
    transport.queue(
        "https://acme.zendesk.com/api/v2/tickets/7/comments.json",
        json_response({"comments": [], "meta": {"has_more": False}}),
    )
    zendesk = ZendeskConnector(ZendeskConfig("acme", "a@b.c", "token"), transport=transport)
    ticket = {"id": 7, "created_at": LATE_EVENING, "updated_at": LATE_EVENING, "subject": "s"}
    zendesk_ref = SourceRef(
        source_id="zendesk-7",
        source_type=SourceType.TICKET,
        updated_at=parse_datetime(LATE_EVENING),
        ref={"ticket": ticket},
    )

    export_path = tmp_path / "export.json"
    export_path.write_text(
        json.dumps([{"id": "E1", "subject": "s", "created": LATE_EVENING, "comments": []}])
    )
    export = ExportTicketConnector(ExportTicketConfig(path=export_path))
    (export_ref,) = list(export.list_sources(None))

    assert zendesk.fetch(zendesk_ref).source_date == ORG_DAY
    assert export.fetch(export_ref).source_date == ORG_DAY


def test_crm_emails_and_feed_items_are_dated_by_the_org_day(phoenix) -> None:
    salesforce_email = salesforce_module._email_message_to_raw_source(
        "sf-email-1",
        {"Id": "02s1", "MessageDate": LATE_EVENING_SF, "LastModifiedDate": LATE_EVENING_SF},
    )
    hubspot_email = hubspot_module._email_object_to_raw_source(
        "hs-email-1", {"id": "1", "properties": {"hs_timestamp": LATE_EVENING_MS}}
    )
    atom = (
        '<feed xmlns="http://www.w3.org/2005/Atom"><entry><title>t</title>'
        '<link href="https://example.com/a"/>'
        f"<published>{LATE_EVENING}</published><updated>{LATE_EVENING}</updated>"
        "</entry></feed>"
    )
    feed_transport = FakeTransport()
    feed_transport.queue("https://example.com/feed", text_response(atom))
    feed = RSSConnector(
        RSSConfig(feeds=[FeedConfig(name="news", url="https://example.com/feed")]),
        transport=feed_transport,
    )
    (feed_ref,) = list(feed.list_sources(None))

    assert salesforce_email.source_date == ORG_DAY
    assert hubspot_email.source_date == ORG_DAY
    assert feed.fetch(feed_ref).source_date == ORG_DAY


def test_a_utc_deployment_still_gets_the_utc_day(monkeypatch) -> None:
    """Finding 5 guard: the day follows `PORTOLAN_TIMEZONE`, not a
    hard-coded offset."""
    monkeypatch.setenv("PORTOLAN_TIMEZONE", "UTC")
    raw = salesforce_module._email_message_to_raw_source(
        "sf-email-1",
        {"Id": "02s1", "MessageDate": LATE_EVENING_SF, "LastModifiedDate": LATE_EVENING_SF},
    )
    assert raw.source_date == date(2026, 9, 10)


# --------------------------------------------------------------------
# Finding 6: the file connector reads an email's date only when the Date
# header starts with an ISO date. A real mail export (`.eml`) carries RFC
# 2822 ("Tue, 08 Sep 2026 18:30:00 -0700"), so every such email was dated
# the day it was ingested.
# --------------------------------------------------------------------


def _write_eml(root: Path, name: str, date_header: str) -> None:
    (root / "email").mkdir(parents=True, exist_ok=True)
    (root / "email" / name).write_text(
        "Message-ID: <m1@example.com>\n"
        "From: Rob Tallis <rob@example.org>\n"
        "To: Sam Okafor <sam@example.com>\n"
        f"Date: {date_header}\n"
        "Subject: Delivery\n"
        "\n"
        "The second shipment was late again.\n"
    )


@pytest.mark.parametrize(
    ("date_header", "expected"),
    [
        ("Tue, 08 Sep 2026 18:30:00 -0700", date(2026, 9, 8)),
        ("Wed, 09 Sep 2026 01:30:00 +0000", date(2026, 9, 8)),
        ("2026-03-04 09:00:00 -0700", date(2026, 3, 4)),
    ],
)
def test_file_connector_reads_an_rfc_2822_email_date(
    phoenix, tmp_path, date_header, expected
) -> None:
    _write_eml(tmp_path, "export-1.eml", date_header)
    connector = FileConnector(tmp_path)

    (ref,) = list(connector.list_sources(None))
    assert ref.updated_at.date() == expected
    assert connector.fetch(ref).source_date == expected
    assert list(connector.list_sources(expected + timedelta(days=1))) == []


# --------------------------------------------------------------------
# Finding 7: an incremental re-crawl of a ticket export (`since=`) skips a
# ticket opened before `since` that gained a comment after it.
# --------------------------------------------------------------------


def test_a_ticket_export_recrawl_lists_an_old_ticket_with_a_new_comment(phoenix, tmp_path) -> None:
    """Finding 7. The filter compared `since` with the ticket's `created`
    date only, so the new comment (a changed document, exactly what the
    re-crawl exists to pick up) was silently never fetched."""
    path = tmp_path / "export.json"
    path.write_text(
        json.dumps(
            [
                {
                    "id": "E7",
                    "subject": "Late delivery",
                    "created": "2026-08-01T16:00:00Z",
                    "comments": [
                        {"number": 1, "created": "2026-08-01T16:00:00Z", "body": "x"},
                        {"number": 2, "created": "2026-09-10T16:00:00Z", "body": "y"},
                    ],
                },
                {
                    "id": "E8",
                    "subject": "Old and quiet",
                    "created": "2026-08-01T16:00:00Z",
                    "comments": [
                        {"number": 1, "author": "a", "created": "2026-08-01T16:00:00Z", "body": "z"}
                    ],
                },
            ]
        )
    )
    connector = ExportTicketConnector(ExportTicketConfig(path=path))

    refs = list(connector.list_sources(date(2026, 9, 1)))

    assert [ref.source_id for ref in refs] == ["export-E7"]
    assert connector.fetch(refs[0]).source_date == date(2026, 8, 1)


# --------------------------------------------------------------------
# Finding 8: Gong's /v2/calls caps one request at 90 days (gong.py's own
# module docstring), and `iter_date_windows` says Gong loops windows --
# it never did, so a backfill from further back got only one window.
# --------------------------------------------------------------------


def test_gong_backfill_from_further_back_than_90_days_asks_for_every_window(monkeypatch) -> None:
    monkeypatch.setattr(clock, "today", lambda: date(2026, 9, 22))
    requests: list[dict] = []

    def handler(*, method, url, params, json, data, headers):
        del method, url, json, data, headers
        requests.append(dict(params or {}))
        day = params["fromDateTime"][:10]
        return json_response(
            {"calls": [{"id": f"call-{day}", "started": f"{day}T16:00:00Z"}], "records": {}}
        )

    transport = FakeTransport()
    transport.on("https://api.gong.io/v2/calls", handler)
    connector = GongConnector(GongConfig("key", "secret"), transport=transport)

    refs = list(connector.list_sources(date(2026, 1, 1)))

    starts = [date.fromisoformat(r["fromDateTime"][:10]) for r in requests]
    ends = [date.fromisoformat(r["toDateTime"][:10]) for r in requests if "toDateTime" in r]
    assert starts[0] == date(2026, 1, 1)
    assert len(requests) >= 3
    assert all((end - start).days <= 90 for start, end in zip(starts, ends, strict=False))
    assert starts[1:] == ends  # contiguous, no gap between windows
    assert "toDateTime" not in requests[-1]  # the last window runs to now
    assert len({r.source_id for r in refs}) == len(refs) == len(requests)


# --------------------------------------------------------------------
# Finding 9: intake finds a source's earlier versions with
# `slug = X OR slug LIKE 'X--v%'`. Under the database's en_US collation
# that is a sequential scan of `sources` for every new source (measured:
# 90 ms per lookup at 200,000 sources, 0.4 ms by `source_id`), and `_`
# and `%` in a source id are LIKE wildcards.
# --------------------------------------------------------------------


def test_intake_does_not_version_a_new_source_after_an_unrelated_one(db, tmp_path) -> None:
    """Finding 9. `S_call-1` is new. `SXcall-1--v2` (a changed version of
    an unrelated source) matched the pattern `S_call-1--v%`, so the new
    source was named `S_call-1--v2` and recorded as superseding it."""
    store = LocalStore(tmp_path)

    def raw(source_id: str, text: str) -> RawSource:
        return RawSource(
            source_id=source_id,
            source_type=SourceType.CALL,
            source_date=date(2026, 8, 1),
            text=text,
        )

    intake(raw("SXcall-1", "first version"), conn=db, store=store)
    changed = intake(raw("SXcall-1", "second version"), conn=db, store=store)
    assert changed.slug == "SXcall-1--v2"

    new = intake(raw("S_call-1", "an unrelated call"), conn=db, store=store)

    assert new.slug == "S_call-1"
    assert "supersedes_source" not in new.metadata


# --------------------------------------------------------------------
# Finding 13: a dead-letter row stays pending after the source succeeds
# in a later normal run, and redrive processes a source once per row.
# --------------------------------------------------------------------


class _FlakyThenFixed:
    """`wave2-flaky`'s fetch raises until `fixed` is set; counts fetches."""

    name = "wave2-flaky"

    def __init__(self) -> None:
        self.fixed = False
        self.fetches = 0

    def list_sources(self, since):
        del since
        return [
            SourceRef(
                source_id="wave2-flaky",
                source_type=SourceType.EMAIL,
                updated_at=datetime(2026, 8, 4, tzinfo=UTC),
                ref=None,
            )
        ]

    def fetch(self, ref: SourceRef) -> RawSource:
        self.fetches += 1
        if not self.fixed:
            raise RuntimeError("vendor timeout")
        return RawSource(
            source_id="wave2-flaky",
            source_type=SourceType.EMAIL,
            source_date=date(2026, 8, 4),
            account_hints=["forced-account"],
            text="now working",
        )


def _pending_dead_letters(conn) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM dead_letter WHERE redriven_at IS NULL")
        return cur.fetchone()[0]


def _zero_fact_llm() -> FakeLLMClient:
    return FakeLLMClient(_counting_responder({}), sink=InMemorySink())


def test_a_later_successful_run_clears_the_sources_dead_letter_rows(db, tmp_path) -> None:
    """Finding 13. The source failed on Monday and was ingested cleanly on
    Tuesday's run, but its dead-letter row stayed pending: /coverage kept
    listing it as not ingested, the run report's dead_letter_count kept
    counting it, and the next redrive re-extracted it with `force`."""
    connector = _FlakyThenFixed()
    settings = _settings(tmp_path)

    def _run_once():
        return run(
            db,
            connector=connector,
            llm=_zero_fact_llm(),
            embedder=FakeEmbedder(),
            registry=Registry.load(db),
            store=LocalStore(settings.store_root),
            settings=settings,
            sink=InMemorySink(),
            sleep=lambda _seconds: None,
        )

    assert _run_once().dead_letter_count == 1
    connector.fixed = True
    second = _run_once()

    assert second.sources_extracted == 1
    assert second.dead_letter_count == 0
    assert _pending_dead_letters(db) == 0


def test_redrive_processes_a_source_once_however_many_rows_it_has(db, tmp_path) -> None:
    """Finding 13, redrive half. Two failed runs leave two rows for one
    source; redrive re-ran the whole pipeline (a paid extraction) once per
    row, `force` on."""
    connector = _FlakyThenFixed()
    settings = _settings(tmp_path)
    for _ in range(2):
        run(
            db,
            connector=connector,
            llm=_zero_fact_llm(),
            embedder=FakeEmbedder(),
            registry=Registry.load(db),
            store=LocalStore(settings.store_root),
            settings=settings,
            sink=InMemorySink(),
            sleep=lambda _seconds: None,
        )
    assert _pending_dead_letters(db) == 2

    connector.fixed = True
    connector.fetches = 0
    counts: dict[str, int] = {}
    report = redrive(
        db,
        connector=connector,
        llm=FakeLLMClient(_counting_responder(counts), sink=InMemorySink()),
        embedder=FakeEmbedder(),
        registry=Registry.load(db),
        store=LocalStore(settings.store_root),
        settings=settings,
        sleep=lambda _seconds: None,
    )

    assert connector.fetches == 1
    assert counts == {"extract": 1}
    assert (report.succeeded, report.still_failing) == (1, 0)
    assert _pending_dead_letters(db) == 0


# --------------------------------------------------------------------
# Finding 14: a parent loop in a dirty CRM export (an account listed as
# its own parent, or two accounts listed as each other's) was linked
# as-is, and `ensure_entity_note` recursed up the chain forever: every
# fact about that account dead-lettered with a RecursionError.
# --------------------------------------------------------------------


@pytest.mark.parametrize(
    "rows",
    [
        ["A-1,Loop Health,system,A-1,loop.org"],
        ["A-1,Loop Health,system,A-2,loop.org", "A-2,Loop Group,system,A-1,loopgroup.org"],
    ],
    ids=["self-parent", "two-account-loop"],
)
def test_a_parent_loop_in_the_crm_export_does_not_break_the_account_note(
    db, tmp_path, rows
) -> None:
    accounts = tmp_path / "accounts.csv"
    accounts.write_text("crm_id,name,kind,parent_crm_id,domain\n" + "\n".join(rows) + "\n")
    contacts = tmp_path / "contacts.csv"
    contacts.write_text("crm_id,name,title,account_crm_id,email\n")

    Registry().import_crm(db, accounts, contacts)
    registry = Registry.load(db)

    note = ensure_entity_note(db, registry, "A-1")

    assert note.title == "Loop Health"
    assert "A-1" not in registry.ancestors("A-1")


# --------------------------------------------------------------------
# Finding 15: the pipeline names its lint file by the host's date while
# the report inside it is dated by the org clock, so after 17:00 Phoenix
# on a UTC host `lint-<tomorrow>.md` holds "Lint report — <today>".
# --------------------------------------------------------------------


def test_the_pipeline_lint_file_is_named_by_the_org_day(db, tmp_path, monkeypatch) -> None:
    org_day = date(2026, 9, 1)
    monkeypatch.setattr(clock, "today", lambda: org_day)

    _run(db, tmp_path, _zero_fact_llm())

    lint_dir = _settings(tmp_path).store_root.parent / "lint"
    assert [p.name for p in lint_dir.iterdir()] == ["lint-2026-09-01.md"]
    assert "Lint report — 2026-09-01" in (lint_dir / "lint-2026-09-01.md").read_text()
