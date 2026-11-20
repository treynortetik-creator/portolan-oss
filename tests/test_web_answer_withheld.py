"""The chat answer counts every sentence it withholds, by check, so the
footnote can say so (`portolan.web.answer`, `Answer.uncited_dropped` /
`Answer.numerals_dropped`). Model calls are stubbed (`FakeLLMClient`); the
notes are hand-built rows in the test database, as in
`tests/test_web_answer.py`."""

from __future__ import annotations

from datetime import date

import pytest

from portolan.llm.fake import FakeLLMClient
from portolan.search.embed import FakeEmbedder
from portolan.search.query import SearchHit, SearchResult
from portolan.web.answer import NOTHING_RECORDED_MESSAGE, answer_question
from tests.test_web_answer import _insert_note_with_one_claim


def _hit(slug: str) -> SearchHit:
    return SearchHit(
        note_slug=slug,
        title=slug,
        score=1.0,
        via=["fts"],
        best_chunk="irrelevant",
        heading=None,
        updated=date(2026, 8, 1),
        source_slugs=[],
    )


@pytest.fixture
def renewal_claim(db, monkeypatch: pytest.MonkeyPatch) -> int:
    claim_id = _insert_note_with_one_claim(
        db,
        slug="withheld-renewal",
        description="d",
        claim_text="The account renewed for $42,000 this quarter.",
    )

    def fake_search(conn, embedder, query, *, k, expand):
        return SearchResult(
            hits=[_hit("withheld-renewal")], control_ok=True, query_terms_unmatched=[]
        )

    monkeypatch.setattr("portolan.web.answer.search", fake_search)
    return claim_id


def _s(text, slugs, claims):
    return {"text": text, "note_slugs": slugs, "claim_ids": claims}


def test_each_withheld_sentence_is_counted_under_the_check_that_withheld_it(db, renewal_claim):
    canned = {
        "sentences": [
            _s("They renewed for $42,000.", ["withheld-renewal"], [renewal_claim]),
            _s("Onboarding took three weeks.", [], []),
            _s("They renewed for $99,000.", ["withheld-renewal"], [renewal_claim]),
            _s("A made-up note is cited.", ["no-such-note"], [999_999_999]),
            _s("   ", [], []),
        ]
    }
    answer = answer_question(db, FakeLLMClient([canned]), FakeEmbedder(), "how did it go?")

    assert [s.text for s in answer.sentences] == ["They renewed for $42,000."]
    assert (answer.uncited_dropped, answer.numerals_dropped) == (2, 1)
    assert answer.usage["uncited_dropped"] == 2 and answer.usage["numerals_dropped"] == 1


def test_nothing_withheld_leaves_no_counts_in_usage(db, renewal_claim):
    canned = {"sentences": [_s("They renewed.", ["withheld-renewal"], [renewal_claim])]}
    answer = answer_question(db, FakeLLMClient([canned]), FakeEmbedder(), "q")
    assert (answer.uncited_dropped, answer.numerals_dropped) == (0, 0)
    assert "uncited_dropped" not in answer.usage and "numerals_dropped" not in answer.usage


def test_the_model_abstains_with_an_empty_list_which_withholds_nothing(db, renewal_claim):
    calls = []

    def reply(*, system, user, json_schema, model, surface):
        calls.append(system)
        return {"sentences": []}

    answer = answer_question(db, FakeLLMClient(reply), FakeEmbedder(), "q")
    assert [s.text for s in answer.sentences] == [NOTHING_RECORDED_MESSAGE]
    assert answer.uncited_dropped == 0 and "uncited_dropped" not in answer.usage
    assert "return an empty sentences list" in calls[0]


def test_an_all_uncited_answer_falls_back_to_the_message_and_still_counts(db, renewal_claim):
    canned = {"sentences": [_s("Uncited claim.", [], [])]}
    answer = answer_question(db, FakeLLMClient([canned]), FakeEmbedder(), "q")
    assert [s.text for s in answer.sentences] == [NOTHING_RECORDED_MESSAGE]
    assert answer.uncited_dropped == 1
