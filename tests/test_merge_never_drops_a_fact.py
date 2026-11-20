"""A fact the merge cannot file is kept whole, and a person is asked only
when the answer matters.

Three ways a fact used to vanish with no row anywhere:
- it named no entity at all (`entities: []`);
- it was the second fact in one source to name the same unknown name (the
  review row kept the first fact's words only);
- a person mapped the unknown name, and the parked facts were never merged.

And one way the queue flooded: every unknown name opened a row, including
names on facts that were filed anyway. Those now wait as `watching` until
the name blocks a fact or keeps coming back.
"""

from __future__ import annotations

from portolan.extract.schema import Coverage, ExtractionResult, FactType
from portolan.ingest.models import Participant
from portolan.llm.fake import FakeLLMClient
from portolan.merge import merge_extraction
from portolan.merge.replay import replay_parked_facts
from portolan.notes import read_note
from portolan.registry import Registry, add_alias
from portolan.search.embed import FakeEmbedder
from tests.merge_helpers import (
    LARKSPUR_MEMORIAL,
    build_fact,
    loaded_registry_fixture,  # noqa: F401 - registers the `loaded_registry` fixture
    make_source_row,
)


def _result(source_id: str, facts) -> ExtractionResult:
    return ExtractionResult(
        source_id=source_id,
        summary="a call",
        facts=facts,
        coverage=Coverage(segments_read=1, segments_total=1),
    )


def _rows(db, kind: str = "unknown_entity"):
    with db.cursor() as cur:
        cur.execute(
            "SELECT id, status, payload, source_slug FROM review_queue WHERE kind = %s ORDER BY id",
            (kind,),
        )
        return cur.fetchall()


def _llm() -> FakeLLMClient:
    return FakeLLMClient(lambda **_: {"verdict": "different", "reason": "unrelated"})


def _larkspur_contact_email(registry: Registry) -> str:
    for entity in registry._entities.values():
        if entity.parent_id == LARKSPUR_MEMORIAL and entity.email:
            return entity.email
    raise AssertionError("fixture registry has no contact with an email at the account")


def test_a_fact_naming_no_entity_is_filed_on_the_sources_account(loaded_registry: Registry, db):
    source = make_source_row(
        db,
        "s-call-100",
        participants=[Participant(name="x", email=_larkspur_contact_email(loaded_registry))],
    )
    fact = build_fact(
        "f1", FactType.REQUEST, "The customer asked for a quiet-hours alarm setting.", []
    )
    report = merge_extraction(
        db, _llm(), FakeEmbedder(), loaded_registry, _result("s-call-100", [fact]), source
    )
    assert report.claims_inserted == 1
    assert report.outcome_counts == {"new": 1}
    # The account the source's participant belongs to (its email domain).
    assert report.placements[0].note_slug.startswith("signal-request-larkspur-")


def test_a_fact_naming_no_entity_on_a_source_with_no_account_is_parked(
    loaded_registry: Registry, db
):
    source = make_source_row(db, "s-call-101")
    fact = build_fact("f1", FactType.REQUEST, "Someone asked for a quiet-hours setting.", [])
    report = merge_extraction(
        db, _llm(), FakeEmbedder(), loaded_registry, _result("s-call-101", [fact]), source
    )
    assert report.claims_inserted == 0
    rows = _rows(db, "other")
    assert len(rows) == 1
    assert rows[0][2]["reason"] == "no_home_note"
    assert rows[0][2]["fact"]["claim"] == fact.claim


def test_every_fact_blocked_by_one_unknown_name_is_kept_on_its_row(loaded_registry: Registry, db):
    source = make_source_row(db, "s-call-102")
    facts = [
        build_fact(
            "f1",
            FactType.COMPLAINT,
            "Zeta Works had a late delivery.",
            ["Zeta Works"],
            locator="00:01:00",
            quote="late again",
        ),
        build_fact(
            "f2",
            FactType.REQUEST,
            "Zeta Works asked for a loaner pump.",
            ["Zeta Works"],
            locator="00:02:00",
        ),
    ]
    merge_extraction(
        db, _llm(), FakeEmbedder(), loaded_registry, _result("s-call-102", facts), source
    )
    rows = _rows(db)
    assert len(rows) == 1
    _id, status, payload, _slug = rows[0]
    assert status == "open"  # the name blocks facts: a person is asked now
    kept = payload["facts"]
    assert [entry["fact"]["claim"] for entry in kept] == [f.claim for f in facts]
    assert all(entry["blocking"] and entry["source_slug"] == "s-call-102" for entry in kept)
    assert kept[0]["fact"]["quote"] == "late again"

    # Merging the same source again adds nothing.
    merge_extraction(
        db, _llm(), FakeEmbedder(), loaded_registry, _result("s-call-102", facts), source
    )
    assert len(_rows(db)[0][2]["facts"]) == 2


def test_an_unknown_name_on_a_filed_fact_waits_until_it_recurs(loaded_registry: Registry, db):
    def one(slug: str):
        source = make_source_row(db, slug)
        fact = build_fact(
            "f1",
            FactType.COMPLAINT,
            f"Larkspur Memorial Hospital's night charge nurse reported a jam ({slug}).",
            ["Larkspur Memorial Hospital", "night charge nurse"],
            quote="it jammed",
        )
        return merge_extraction(
            db, _llm(), FakeEmbedder(), loaded_registry, _result(slug, [fact]), source
        )

    report = one("s-call-110")
    assert report.claims_inserted == 1  # the fact is filed; the name is not blocking
    rows = _rows(db)
    assert [(r[1]) for r in rows] == ["watching"]

    one("s-call-111")
    assert [(r[1]) for r in _rows(db)] == ["watching"]
    one("s-call-112")  # third distinct source: worth a person's answer
    rows = _rows(db)
    assert len(rows) == 1 and rows[0][1] == "open"
    assert len(rows[0][2]["occurrences"]) == 2


def test_a_blocking_fact_opens_a_watching_row(loaded_registry: Registry, db):
    s1 = make_source_row(db, "s-call-120")
    filed = build_fact(
        "f1",
        FactType.COMPLAINT,
        "Larkspur Memorial Hospital and Zeta Works share a dock.",
        ["Larkspur Memorial Hospital", "Zeta Works"],
        quote="share a dock",
    )
    merge_extraction(
        db, _llm(), FakeEmbedder(), loaded_registry, _result("s-call-120", [filed]), s1
    )
    assert _rows(db)[0][1] == "watching"

    s2 = make_source_row(db, "s-call-121")
    blocked = build_fact("f1", FactType.REQUEST, "Zeta Works asked for a quote.", ["Zeta Works"])
    merge_extraction(
        db, _llm(), FakeEmbedder(), loaded_registry, _result("s-call-121", [blocked]), s2
    )
    rows = _rows(db)
    assert len(rows) == 1 and rows[0][1] == "open"


def test_mapping_the_name_merges_the_parked_facts(loaded_registry: Registry, db):
    source = make_source_row(db, "s-call-130")
    facts = [
        build_fact(
            "f1",
            FactType.COMPLAINT,
            "Zeta Works had a late delivery.",
            ["Zeta Works"],
            locator="00:01:00",
            quote="late again",
        ),
        build_fact(
            "f2",
            FactType.REQUEST,
            "Zeta Works asked for a loaner pump.",
            ["Zeta Works"],
            locator="00:02:00",
        ),
    ]
    merge_extraction(
        db, _llm(), FakeEmbedder(), loaded_registry, _result("s-call-130", facts), source
    )
    with db.cursor() as cur:
        cur.execute("UPDATE sources SET summary = 'the real summary' WHERE slug = 's-call-130'")
    db.commit()
    row_id = _rows(db)[0][0]

    add_alias(db, LARKSPUR_MEMORIAL, "Zeta Works", source="human")
    report = replay_parked_facts(db, _llm(), FakeEmbedder(), Registry.load(db), row_id)

    assert (report.replayed, report.still_parked, report.malformed) == (2, 0, 0)
    _note, claims = read_note(db, "signal-complaint-larkspur-memorial-hospital")
    assert [c.locator for c in claims] == ["00:01:00"]
    assert claims[0].source_slug == "s-call-130" and claims[0].quote == "late again"
    _note, claims = read_note(db, "signal-request-larkspur-memorial-hospital")
    assert len(claims) == 1

    # A second replay writes nothing more, and the source keeps its summary.
    again = replay_parked_facts(db, _llm(), FakeEmbedder(), Registry.load(db), row_id)
    assert again.replayed == 0
    with db.cursor() as cur:
        cur.execute("SELECT count(*) FROM claims WHERE source_slug = 's-call-130'")
        assert cur.fetchone()[0] == 2
        cur.execute("SELECT summary FROM sources WHERE slug = 's-call-130'")
        assert cur.fetchone()[0] == "the real summary"
