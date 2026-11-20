"""The compare call can say two gated claims are unrelated (`different`).

Before this, the verdict schema was `same | extends | contradicts`, so a pair
the similarity gate put together by shared words alone had to be forced into
one of them: `same` turned a new fact into a citation on an unrelated claim
(the fact was lost), and `contradicts` either retired an unrelated claim or
wrote a review row a person then had to dismiss.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

from portolan.extract.schema import FactType
from portolan.llm.fake import FakeLLMClient
from portolan.merge.facts import merge_fact
from portolan.merge.gate import COMPARE_SYSTEM, build_compare_prompt
from portolan.merge.models import CompareVerdict, compare_verdict_json_schema
from portolan.notes import read_note
from portolan.notes.models import Claim
from portolan.registry import Registry
from portolan.search.embed import FakeEmbedder
from tests.merge_helpers import (
    build_fact,
    loaded_registry_fixture,  # noqa: F401 - registers the `loaded_registry` fixture
    make_source_row,
)


def _two_similar_complaints():
    first = build_fact(
        "f1",
        FactType.COMPLAINT,
        "Larkspur Memorial Hospital reported a Tidewater T2 consumable delivery "
        "arriving four days late on 2026-08-04.",
        ["Larkspur Memorial Hospital", "Tidewater T2"],
        locator="T-031#1",
        quote="4 days late",
        valid_from=date(2026, 8, 4),
    )
    second = build_fact(
        "f2",
        FactType.COMPLAINT,
        "Larkspur Memorial Hospital reported a Tidewater T2 consumable delivery "
        "arriving with a damaged outer carton on 2026-08-19.",
        ["Larkspur Memorial Hospital", "Tidewater T2"],
        locator="00:04:12",
        quote="the carton was crushed",
        valid_from=date(2026, 8, 19),
    )
    return first, second


def test_schema_offers_the_different_verdict():
    schema = compare_verdict_json_schema()
    assert set(schema["properties"]["verdict"]["enum"]) == {
        "same",
        "extends",
        "contradicts",
        "different",
    }
    assert CompareVerdict.model_validate({"verdict": " Different ", "reason": "x"}).verdict == (
        "different"
    )
    assert "different" in COMPARE_SYSTEM


def test_different_verdict_files_a_new_claim_and_touches_nothing_else(
    loaded_registry: Registry, db
):
    embedder = FakeEmbedder()
    llm = FakeLLMClient([{"verdict": "different", "reason": "two separate deliveries"}])
    first, second = _two_similar_complaints()

    s1 = make_source_row(db, "s-ticket-031", source_type="ticket", source_date=date(2026, 8, 5))
    o1 = merge_fact(db, llm, embedder, loaded_registry, first, s1)
    s2 = make_source_row(db, "s-call-020", source_type="call", source_date=date(2026, 8, 20))
    o2 = merge_fact(db, llm, embedder, loaded_registry, second, s2)

    assert o2.compare_calls == 1
    assert o2.compare_verdict == "different"
    assert o2.status == "new"
    assert o2.claim_inserted and o2.claim_id != o1.claim_id
    assert o2.queue_rows == 0

    _note, claims = read_note(db, o1.note_slug)
    assert len(claims) == 2
    # The older claim is neither retired nor given the new fact's citation.
    assert all(c.superseded_at is None for c in claims)
    assert all(not c.citations for c in claims)
    with db.cursor() as cur:
        cur.execute("SELECT count(*) FROM review_queue")
        assert cur.fetchone()[0] == 0


def test_compare_prompt_shows_both_quotes_and_dates():
    first, second = _two_similar_complaints()
    candidate = Claim(
        id=7,
        note_slug="signal-complaint-x",
        fact_type="complaint",
        claim_text=first.claim,
        source_slug="s-ticket-031",
        locator="T-031#1",
        quote=first.quote,
        valid_from=first.valid_from,
        recorded=datetime(2026, 8, 5, tzinfo=UTC),
    )
    prompt = build_compare_prompt(candidate=candidate, fact=second, source_date=date(2026, 8, 20))
    assert 'Verbatim quote: "4 days late"' in prompt
    assert 'Verbatim quote: "the carton was crushed"' in prompt
    assert "dated 2026-08-04" in prompt and "dated 2026-08-19" in prompt
    assert "source dated 2026-08-20" in prompt
