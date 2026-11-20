"""Tests for `portolan.web.routes.review`'s non-HTTP helper functions.

`_keep_quote_unsupported_fact` (build item 3, mechanical quote gate,
2026-09-19's "Keep as claim" choice) takes a plain `(conn, llm, embedder,
row)` -- no FastAPI/HTTP surface at all -- so it's tested directly here,
against the same lightweight real-registry `db`/`loaded_registry`
fixtures `tests/test_merge_facts.py` uses, rather than `tests/test_web.
py`'s heavy session-scoped `demo_conn` (a full `pipeline.run` over
`corpus/out`), which exists for full HTTP-route coverage this function
doesn't need.
"""

from __future__ import annotations

from portolan.extract.schema import FactType
from portolan.llm.fake import FakeLLMClient
from portolan.registry import Registry
from portolan.search.embed import FakeEmbedder
from portolan.web.routes.review import _keep_quote_unsupported_fact
from tests.merge_helpers import (
    loaded_registry_fixture,  # noqa: F401 - registers the `loaded_registry` fixture
    make_source_row,
)


def _quote_unsupported_row(*, row_id: int, source_slug: str, fact: dict) -> dict:
    """The shape `_get_row` returns for a `kind='quote_unsupported'` row
    -- built by hand here rather than through a real INSERT, since
    `_keep_quote_unsupported_fact` only ever reads `row['payload']`/
    `row['source_slug']`."""
    return {
        "id": row_id,
        "kind": "quote_unsupported",
        "payload": {
            "reason": "The model's quote for this fact was not found in the source text.",
            "fact": fact,
        },
        "source_slug": source_slug,
        "status": "open",
    }


def _fact_payload(**overrides) -> dict:
    base = {
        "id": "F-keep-1",
        "type": FactType.COMPLAINT.value,
        "claim": ("Ashgrove Medical Center reported a Zyphon Quantum-Widget outage on 2026-08-05."),
        "entities": ["Ashgrove Medical Center"],
        "locator": "00:01:00",
        "quote": "the widget just stopped responding",
        "speaker": None,
        "valid_from": None,
        "valid_to": None,
        "severity": None,
        "sentiment": -1,
        "confidence": 0.8,
    }
    base.update(overrides)
    return base


def test_keep_quote_unsupported_fact_inserts_a_claim(loaded_registry: Registry, db) -> None:
    """The happy path: entities resolve, the claim text shares no
    vocabulary with anything already in this fresh test database, so the
    gate (`portolan.merge.gate.gate`) has nothing to compare against --
    `new`, zero compare calls, `FakeLLMClient([])`'s empty canned-answer
    list is never actually needed. `merge_extraction` inserts a real
    `claims` row through the exact entry point `portolan.pipeline` itself
    uses."""
    source = make_source_row(db, "s-keep-001")
    row = _quote_unsupported_row(row_id=1, source_slug=source.slug, fact=_fact_payload())

    notice = _keep_quote_unsupported_fact(db, FakeLLMClient([]), FakeEmbedder(), row)

    assert notice == "kept_as_claim"
    with db.cursor() as cur:
        cur.execute(
            "SELECT claim_text, locator, quote FROM claims WHERE source_slug = %s",
            (source.slug,),
        )
        rows = cur.fetchall()
    assert len(rows) == 1
    claim_text, locator, quote = rows[0]
    assert "Zyphon Quantum-Widget" in claim_text
    assert locator == "00:01:00"
    assert quote == "the widget just stopped responding"


def test_keep_quote_unsupported_fact_with_unresolvable_entity_is_queued_unknown(
    loaded_registry: Registry, db
) -> None:
    """A fact whose only entity mention resolves to nothing real is a
    legitimate `queued_unknown` merge outcome -- exactly what would
    happen to any other fact reaching merge with the same entities. No
    claim is inserted; the caller still reports it as handled
    (`"kept_unresolved"`, not an error)."""
    source = make_source_row(db, "s-keep-002")
    fact = _fact_payload(
        id="F-keep-2",
        claim="Nonexistent Fictional Company reported an outage on 2026-08-05.",
        entities=["Nonexistent Fictional Company"],
        locator="00:02:00",
    )
    row = _quote_unsupported_row(row_id=2, source_slug=source.slug, fact=fact)

    notice = _keep_quote_unsupported_fact(db, FakeLLMClient([]), FakeEmbedder(), row)

    assert notice == "kept_unresolved"
    with db.cursor() as cur:
        cur.execute("SELECT count(*) FROM claims WHERE source_slug = %s", (source.slug,))
        (count,) = cur.fetchone()
    assert count == 0


def test_keep_quote_unsupported_fact_with_missing_fact_payload_is_malformed(
    loaded_registry: Registry, db
) -> None:
    source = make_source_row(db, "s-keep-003")
    row = {
        "id": 3,
        "kind": "quote_unsupported",
        "payload": {"reason": "test"},
        "source_slug": source.slug,
        "status": "open",
    }
    notice = _keep_quote_unsupported_fact(db, FakeLLMClient([]), FakeEmbedder(), row)
    assert notice == "malformed"


def test_keep_quote_unsupported_fact_with_unknown_source_is_malformed(
    loaded_registry: Registry, db
) -> None:
    row = _quote_unsupported_row(
        row_id=4, source_slug="s-does-not-exist", fact=_fact_payload(id="F-keep-4")
    )
    notice = _keep_quote_unsupported_fact(db, FakeLLMClient([]), FakeEmbedder(), row)
    assert notice == "malformed"


def test_keep_quote_unsupported_fact_already_on_the_graph_says_so(
    loaded_registry: Registry, db
) -> None:
    """A kept fact that matches a claim already on file inserts no claim;
    the notice says it matched, not that it was parked as unknown."""
    first = make_source_row(db, "s-keep-010")
    second = make_source_row(db, "s-keep-010--v2")
    with db.cursor() as cur:
        cur.execute(
            "UPDATE sources SET source_id = 's-keep-010' WHERE slug IN (%s, %s)",
            (first.slug, second.slug),
        )
    db.commit()
    fact = _fact_payload(id="F-keep-10", locator="00:10:00")

    kept = _keep_quote_unsupported_fact(
        db,
        FakeLLMClient([]),
        FakeEmbedder(),
        _quote_unsupported_row(row_id=10, source_slug=first.slug, fact=fact),
    )
    again = _keep_quote_unsupported_fact(
        db,
        FakeLLMClient([]),
        FakeEmbedder(),
        _quote_unsupported_row(row_id=11, source_slug=second.slug, fact=fact),
    )

    assert kept == "kept_as_claim"
    assert again == "kept_matched_existing"
    with db.cursor() as cur:
        cur.execute("SELECT count(*) FROM review_queue WHERE kind = 'unknown_entity'")
        assert cur.fetchone() == (0,)
