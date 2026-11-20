"""The mechanical quote gate in `portolan.pipeline` (build item, 2026-09-19:
`01-spec/extraction-final-2026-09-18.md`'s production pick -- "Nova Lite,
five samples, v6 prompt, mechanical quote gate, human review"). Before
this, the pipeline only checked that a quote-requiring fact HAD a quote
and that it was forty words or fewer (`Fact._validate_against_effective_
schema`); it never checked the quote's words were actually IN the source,
the way `portolan.bench.extraction._evidence_basis` already did for
scoring. `_gate_quotes` is the pure split function; `_finish_one` wires
it in between "passes unioned" and "merge".

Reuses `_FakeConcurrencyConnector`/`_settings` from `test_pipeline.py`
(established cross-test-file import pattern, same as `test_pipeline_
samples.py`) and `portolan.extract.backends.FakeBackend` plumbed through
the real `_prepare_one`/`_extract_segments`/`_finish_one` chain against a
real Postgres connection -- `test_pipeline.py`'s own `test_status_done_
is_written_only_after_merge_commits_notes_and_claims` is the model for
this style.
"""

from __future__ import annotations

from pathlib import Path

from test_pipeline import _FakeConcurrencyConnector, _settings

import portolan.pipeline as pipeline_module
from portolan.extract.backends import FakeBackend
from portolan.extract.quote_gate import quote_is_supported
from portolan.extract.schema import Coverage, ExtractionResult, Fact, Vocabulary
from portolan.llm.fake import FakeLLMClient
from portolan.merge import merge_extraction
from portolan.metering import InMemorySink
from portolan.registry import Registry
from portolan.search.embed import FakeEmbedder
from portolan.store import LocalStore
from tests.merge_helpers import make_source_row

# `_FakeConcurrencyConnector.fetch`'s own fixed text -- a "call" source,
# whose `_prepare_call` handler passes `raw.text` through completely
# unchanged (`portolan.ingest.prepare._prepare_call`: `PreparedText(text=
# text, ...)`), so this is exactly `_PreparedSource.text` too.
_SOURCE_TEXT = (
    "[00:00:01] Speaker: This is fake concurrency-test source 0, with enough "
    "text to look like a real call."
)
_SUPPORTED_QUOTE = "enough text to look like a real call"
_UNSUPPORTED_QUOTE = "this exact sentence never appears anywhere in the source text"


def test_supported_quote_is_actually_in_the_fixture_text() -> None:
    """Sanity check on the fixture itself, not the gate -- if this ever
    goes false the test below is checking the wrong thing."""
    assert quote_is_supported(_SUPPORTED_QUOTE, _SOURCE_TEXT) is True
    assert quote_is_supported(_UNSUPPORTED_QUOTE, _SOURCE_TEXT) is False


# --------------------------------------------------------------------
# `_gate_quotes` -- pure function, no database
# --------------------------------------------------------------------


def test_gate_quotes_keeps_supported_drops_unsupported_passes_through_no_quote_type() -> None:
    facts = [
        Fact(
            id="f-supported",
            type="complaint",
            claim="A supported-quote complaint.",
            entities=["Acme"],
            locator="00:00:01",
            quote=_SUPPORTED_QUOTE,
            sentiment=-1,
            confidence=0.9,
        ),
        Fact(
            id="f-unsupported",
            type="praise",
            claim="An unsupported-quote praise.",
            entities=["Acme"],
            locator="00:00:01",
            quote=_UNSUPPORTED_QUOTE,
            sentiment=1,
            confidence=0.9,
        ),
        Fact(
            id="f-no-quote-type",
            type="risk",
            claim="A risk fact -- risk doesn't require a quote at all.",
            entities=["Acme"],
            locator="00:00:01",
            sentiment=-1,
            confidence=0.8,
        ),
    ]

    kept, gated = pipeline_module._gate_quotes(facts, source_text=_SOURCE_TEXT)

    assert [f.id for f in kept] == ["f-supported", "f-no-quote-type"]
    assert len(gated) == 1
    assert gated[0]["id"] == "f-unsupported"
    assert gated[0]["type"] == "praise"
    assert gated[0]["claim"] == "An unsupported-quote praise."
    assert gated[0]["quote"] == _UNSUPPORTED_QUOTE
    assert gated[0]["locator"] == "00:00:01"
    assert gated[0]["entities"] == ["Acme"]


def test_gate_quotes_empty_source_text_gates_every_quote_requiring_fact() -> None:
    facts = [
        Fact(
            id="f1",
            type="commitment",
            claim="A commitment fact.",
            entities=["Acme"],
            locator="00:00:01",
            quote="anything at all",
            sentiment=0,
            confidence=0.5,
        )
    ]
    kept, gated = pipeline_module._gate_quotes(facts, source_text="")
    assert kept == []
    assert len(gated) == 1


# --------------------------------------------------------------------
# End to end: `_prepare_one` -> `_extract_segments` -> `_finish_one`,
# a real Postgres connection, `FakeBackend` standing in for the model.
# --------------------------------------------------------------------


def test_quote_gate_wired_into_finish_one_drops_unsupported_fact_before_merge(
    db, tmp_path: Path
) -> None:
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO registry_entities (kind, crm_id, canonical_name) "
            "VALUES ('account', 'A-QGATE-1', 'Quote Gate Test Co')"
        )
    registry = Registry.load(db)
    connector = _FakeConcurrencyConnector(1)
    ref = connector.list_sources(None)[0]
    settings = _settings(tmp_path)
    store = LocalStore(settings.store_root)
    embedder = FakeEmbedder()

    canned = ExtractionResult(
        source_id=ref.source_id,
        summary="one supported quote, one unsupported quote, one no-quote-required fact",
        facts=[
            Fact(
                id="fact-supported",
                type="complaint",
                claim="Quote Gate Test Co had a real problem.",
                entities=["Quote Gate Test Co"],
                locator="00:00:01",
                quote=_SUPPORTED_QUOTE,
                sentiment=-1,
                confidence=0.9,
            ),
            Fact(
                id="fact-unsupported",
                type="praise",
                claim="Quote Gate Test Co loved the onboarding.",
                entities=["Quote Gate Test Co"],
                locator="00:00:01",
                quote=_UNSUPPORTED_QUOTE,
                sentiment=1,
                confidence=0.9,
            ),
            Fact(
                id="fact-no-quote-type",
                type="risk",
                claim="Quote Gate Test Co is evaluating a competitor.",
                entities=["Quote Gate Test Co"],
                locator="00:00:01",
                sentiment=-1,
                confidence=0.8,
            ),
        ],
        actions=[],
        coverage=Coverage(segments_read=1, segments_total=1),
    )
    backend = FakeBackend({ref.source_id: canned})

    def _unexpected_compare_call(**kwargs):
        raise AssertionError(
            "each fact lands on its own brand-new signal note -- gate() should "
            "skip straight to 'new' without ever calling the compare LLM"
        )

    llm = FakeLLMClient(_unexpected_compare_call)

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
    assert prepared.text == _SOURCE_TEXT

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
        conn=db,
        llm=llm,
        embedder=embedder,
        registry=registry,
        merge_extraction=merge_extraction,
        sink=InMemorySink(),
        sleep=lambda _seconds: None,
    )

    assert outcome.status == "done"
    # 3 facts produced, 1 gated -- the merge/report layer only ever sees
    # the 2 that survived the gate.
    assert outcome.facts_count == 2

    with db.cursor() as cur:
        cur.execute("SELECT claim_text FROM claims WHERE source_slug = %s", (ref.source_id,))
        claim_texts = {row[0] for row in cur.fetchall()}
    # Entity mentions get wikilinked when a claim is written
    # (`[[quote-gate-test-co]]` in place of "Quote Gate Test Co") -- check
    # the stable, non-entity part of each claim's own wording instead of
    # the exact fact text.
    assert any("had a real problem" in t for t in claim_texts)
    assert any("is evaluating a competitor" in t for t in claim_texts)
    assert not any("loved the onboarding" in t for t in claim_texts)
    assert len(claim_texts) == 2

    with db.cursor() as cur:
        cur.execute(
            "SELECT kind, payload, source_slug, status FROM review_queue WHERE source_slug = %s",
            (ref.source_id,),
        )
        rows = cur.fetchall()
    assert len(rows) == 1
    kind, payload, source_slug, status = rows[0]
    assert kind == "quote_unsupported"
    assert source_slug == ref.source_id
    assert status == "open"
    # `merge_segment_results` renumbers every fact "f1, f2, ..." in
    # encounter order regardless of the id the backend supplied (its own
    # docstring: "each segment restarts f1, f2, ...") -- the second of our
    # three facts, so "f2", not the "fact-unsupported" id `canned` gave it.
    assert payload["fact"]["id"] == "f2"
    assert payload["fact"]["claim"] == "Quote Gate Test Co loved the onboarding."
    assert payload["fact"]["quote"] == _UNSUPPORTED_QUOTE
    assert payload["fact"]["locator"] == "00:00:01"
    assert "reason" in payload


def test_quote_gate_redrive_does_not_duplicate_the_review_row(db, tmp_path: Path) -> None:
    """`_queue_quote_unsupported`'s own dedup (by `(source_slug, locator,
    quote)`), matching `portolan.registry.registry.queue_unknown`'s
    convention: running the exact same gated fact through `_finish_one`
    a second time (a redrive re-running extraction from scratch after a
    dead-lettered merge, or simply calling this twice) must not open a
    second, indistinguishable review row."""
    connector = _FakeConcurrencyConnector(1)
    ref = connector.list_sources(None)[0]
    # `review_queue.source_slug` has a real FK to `sources.slug` -- a
    # minimal row is enough (`_queue_quote_unsupported` never reads it).
    make_source_row(db, ref.source_id)

    fact_payload = {
        "id": "fact-1",
        "type": "praise",
        "entities": ["Quote Gate Redrive Co"],
        "claim": "Quote Gate Redrive Co loved it.",
        "quote": _UNSUPPORTED_QUOTE,
        "locator": "00:00:01",
        "speaker": None,
        "valid_from": None,
        "valid_to": None,
        "severity": None,
        "sentiment": 1,
        "confidence": 0.9,
    }
    pipeline_module._queue_quote_unsupported(
        db, source_slug=ref.source_id, fact_payload=fact_payload
    )
    pipeline_module._queue_quote_unsupported(
        db, source_slug=ref.source_id, fact_payload=fact_payload
    )

    with db.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM review_queue "
            "WHERE source_slug = %s AND kind = 'quote_unsupported'",
            (ref.source_id,),
        )
        (count,) = cur.fetchone()
    assert count == 1
