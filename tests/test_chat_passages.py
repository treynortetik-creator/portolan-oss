"""Chat sends each note's best-matching passage and lets a sentence cite it
(`portolan.web.answer`; 01-spec/gbrain-borrow-knowledge-2026-09-23.md rank
1). A passage quote is checked against the exact passage text sent, and the
numbers guard reads it. When nothing survives, the reply names what the
graph lacks. Model calls are stubbed; notes are hand-built rows."""

from __future__ import annotations

import re
from datetime import date

import psycopg
import pytest

from portolan.llm.fake import FakeLLMClient
from portolan.search.embed import FakeEmbedder
from portolan.search.query import SearchHit, SearchResult
from portolan.web.answer import (
    GAPS_PREFIX,
    NOTHING_RECORDED_MESSAGE,
    PASSAGE_CHAR_CAP,
    SYNTHESIS_CHAR_CAP,
    SYNTHESIS_CUT_MARK,
    _build_context,
    answer_question,
    gate_answer,
    passage_id_for,
)
from portolan.web.conversations import sentence_to_dict
from portolan.web.markdown import render_note_body
from portolan.web.templating import build_templates
from tests.test_verify_script import verify

PILOT = (
    "Q3 pilot review. The biomed team said the pilot cut nuisance alarms by "
    "30 percent across 4 wards, and the charge nurses want it on every floor."
)


def _insert_note(conn: psycopg.Connection, slug: str, *, description: str = "d") -> None:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO notes (slug, type, title, description, body_md, updated) "
            "VALUES (%s, 'source', %s, %s, %s, now())",
            (slug, slug.replace("-", " ").title(), description, PILOT),
        )
    conn.commit()


def _insert_claim(
    conn: psycopg.Connection, slug: str, *, source: str, quote: str, locator: str
) -> int:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO sources (slug, source_type, s3_key, content_hash, source_date, "
            "extraction_status) VALUES (%s, 'call', %s, %s, '2026-08-01', 'done') "
            "ON CONFLICT (slug) DO NOTHING",
            (source, f"originals/call/{source}", f"hash-{source}"),
        )
        cur.execute(
            "INSERT INTO claims (note_slug, fact_type, claim_text, source_slug, locator, "
            "quote, severity, sentiment, confidence) "
            "VALUES (%s, 'praise', 'The pilot was praised.', %s, %s, %s, 'low', 1, 0.9) "
            "RETURNING id",
            (slug, source, locator, quote),
        )
        (claim_id,) = cur.fetchone()
    conn.commit()
    return claim_id


def _hit(slug: str, chunk: str = PILOT, sources: list[str] | None = None) -> SearchHit:
    return SearchHit(
        note_slug=slug,
        title=slug,
        score=1.0,
        via=["fts"],
        best_chunk=chunk,
        heading=None,
        updated=date(2026, 8, 1),
        source_slugs=sources or [],
    )


@pytest.fixture
def pilot_note(db, monkeypatch: pytest.MonkeyPatch) -> str:
    """A note with no claims whose matched passage is `PILOT`."""
    _insert_note(db, "pilot-review")

    def fake_search(conn, embedder, query, *, k, expand):
        return SearchResult(hits=[_hit("pilot-review")], control_ok=True, query_terms_unmatched=[])

    monkeypatch.setattr("portolan.web.answer.search", fake_search)
    return passage_id_for("pilot-review", PILOT)


def _answer(db, data: dict, question: str = "How did the pilot go?", seen: list | None = None):
    def respond(*, user: str, **_kwargs):
        if seen is not None:
            seen.append(user)
        return {"gaps": [], **data}

    return answer_question(db, FakeLLMClient(respond), FakeEmbedder(), question)


def _s(text: str, passages: list[dict], slugs: tuple[str, ...] = ("pilot-review",)) -> dict:
    return {"text": text, "note_slugs": list(slugs), "claim_ids": [], "passages": passages}


def test_a_note_with_no_claims_is_quoted_and_cited(db, pilot_note):
    seen: list[str] = []
    quote = "the pilot cut nuisance alarms by 30 percent across 4 wards"
    answer = _answer(
        db,
        {
            "sentences": [
                _s(
                    "The pilot cut nuisance alarms by 30 percent.",
                    [{"passage_id": pilot_note, "quote": quote}],
                )
            ]
        },
        seen=seen,
    )

    (prompt,) = seen
    assert f"[passage_id={pilot_note}]" in prompt
    assert PILOT in prompt
    (sentence,) = answer.sentences
    assert sentence.text == "The pilot cut nuisance alarms by 30 percent."
    (citation,) = sentence.citations
    assert (citation.note_slug, citation.passage_id, citation.quote) == (
        "pilot-review",
        pilot_note,
        quote,
    )
    assert citation.is_passage and not citation.is_note_level
    assert answer.usage["passage_citations"] == 1
    assert (answer.uncited_dropped, answer.numerals_dropped) == (0, 0)


def test_a_fabricated_quote_against_a_passage_is_dropped(db, pilot_note):
    answer = _answer(
        db,
        {
            "sentences": [
                _s(
                    "The pilot was cancelled.",
                    [
                        {
                            "passage_id": pilot_note,
                            "quote": "the charge nurses asked to cancel the pilot",
                        }
                    ],
                )
            ]
        },
    )

    # No note-level fallback: a failed quote must not ship under the note's name.
    assert [s.text for s in answer.sentences] == [NOTHING_RECORDED_MESSAGE]
    assert answer.uncited_dropped == 1


def test_a_passage_that_was_not_sent_or_not_this_notes_is_refused(db, monkeypatch):
    _insert_note(db, "pilot-review")
    _insert_note(db, "other-note")

    def fake_search(conn, embedder, query, *, k, expand):
        return SearchResult(
            hits=[_hit("pilot-review"), _hit("other-note", chunk="Nothing about alarms here.")],
            control_ok=True,
            query_terms_unmatched=[],
        )

    monkeypatch.setattr("portolan.web.answer.search", fake_search)
    quote = "the pilot cut nuisance alarms"
    pilot = passage_id_for("pilot-review", PILOT)
    answer = _answer(
        db,
        {
            "sentences": [
                _s("Unsent passage.", [{"passage_id": "p-0000000000", "quote": quote}]),
                # The quote is real, but the passage is another note's.
                _s("Wrong note.", [{"passage_id": pilot, "quote": quote}], slugs=("other-note",)),
                _s("Kept.", [{"passage_id": pilot, "quote": quote}]),
                _s("Bare note citation.", []),
            ]
        },
    )

    # A sentence whose passage fails gets no note-level fallback either.
    assert [s.text for s in answer.sentences] == ["Kept.", "Bare note citation."]
    assert [[c.is_passage for c in s.citations] for s in answer.sentences] == [[True], [False]]
    assert answer.uncited_dropped == 2


def test_the_numbers_guard_reads_passage_quotes(db, pilot_note):
    good = "the pilot cut nuisance alarms by 30 percent"
    answer = _answer(
        db,
        {
            "sentences": [
                _s("Nuisance alarms fell 30 percent.", [{"passage_id": pilot_note, "quote": good}]),
                # Its number is in no quote it cites: the guard drops it.
                _s("Nuisance alarms fell 45 percent.", [{"passage_id": pilot_note, "quote": good}]),
                # A near-verbatim quote with a changed number clears the
                # fuzzy quote match but not the number check: refused.
                _s(
                    "Alarms fell 45 percent across 4 wards.",
                    [
                        {
                            "passage_id": pilot_note,
                            "quote": "the pilot cut nuisance alarms by 45 percent across 4 wards",
                        }
                    ],
                ),
            ]
        },
    )

    assert [s.text for s in answer.sentences] == ["Nuisance alarms fell 30 percent."]
    assert (answer.numerals_dropped, answer.uncited_dropped) == (1, 1)


def test_a_passage_citation_links_to_the_original(db, monkeypatch):
    _insert_note(db, "one-source")
    _insert_note(db, "two-sources")
    claim_quote = "the pilot cut nuisance alarms by 30 percent across 4 wards"
    _insert_claim(db, "two-sources", source="call-b", quote=claim_quote, locator="@ 00:04:10")

    def fake_search(conn, embedder, query, *, k, expand):
        return SearchResult(
            hits=[
                _hit("one-source", sources=["call-a"]),
                _hit("two-sources", sources=["call-b", "call-c"]),
            ],
            control_ok=True,
            query_terms_unmatched=[],
        )

    monkeypatch.setattr("portolan.web.answer.search", fake_search)
    quote = "cut nuisance alarms by 30 percent"
    answer = _answer(
        db,
        {
            "sentences": [
                _s(
                    "Alarms fell.",
                    [{"passage_id": passage_id_for("one-source", PILOT), "quote": quote}],
                    slugs=("one-source",),
                ),
                _s(
                    "Alarms fell again.",
                    [{"passage_id": passage_id_for("two-sources", PILOT), "quote": quote}],
                    slugs=("two-sources",),
                ),
            ]
        },
    )

    first, second = (s.citations[0] for s in answer.sentences)
    # The note's only source, no locator.
    assert (first.source_slug, first.locator) == ("call-a", None)
    # Several sources: the sent claim whose own quote holds the passage quote.
    assert (second.source_slug, second.locator) == ("call-b", "@ 00:04:10")


def test_a_passage_the_synthesis_already_holds_is_not_sent_twice(db):
    _insert_note(db, "short-note", description=PILOT)
    ctx = _build_context(db, [_hit("short-note")])
    assert ctx.text.count("nuisance alarms") == 1
    assert ctx.passages == {}


def test_a_title_chunk_match_sends_the_notes_first_body_chunk_instead(db):
    _insert_note(db, "title-hit", description="")
    with db.cursor() as cur:
        cur.executemany(
            "INSERT INTO chunks (note_slug, heading, text) VALUES (%s, %s, %s)",
            [("title-hit", None, "Title Hit"), ("title-hit", "Notes", PILOT)],
        )
    db.commit()

    ctx = _build_context(db, [_hit("title-hit", chunk="Title Hit")])

    (passage,) = ctx.passages.values()
    assert (passage.text, passage.heading) == (PILOT, "Notes")
    assert passage.passage_id == passage_id_for("title-hit", PILOT)


def test_passages_are_capped_and_never_crowd_out_a_note(db):
    long_chunk = "word " * 1000
    _insert_note(db, "first-note")
    _insert_note(db, "second-note")
    ctx = _build_context(
        db, [_hit("first-note", chunk=long_chunk), _hit("second-note")], char_budget=1_550
    )

    # Both notes fit without passages, so both are sent; the budget left
    # holds the first note's capped passage and not the second's.
    assert set(ctx.note_claim_ids) == {"first-note", "second-note"}
    (passage,) = ctx.passages.values()
    assert passage.note_slug == "first-note"
    assert PASSAGE_CHAR_CAP - 5 <= len(passage.text) <= PASSAGE_CHAR_CAP
    assert passage.text in ctx.text
    assert "nuisance alarms" not in ctx.text


def test_cannot_answer_lists_what_the_graph_lacks(db, pilot_note):
    answer = _answer(
        db,
        {
            "sentences": [],
            "gaps": [
                "the pilot's cost",
                "which wards took part in the 2027 rollout of 12 floors",
                "  who approved it.  ",
                "a fourth gap",
                "a fifth gap",
            ],
        },
    )

    (sentence,) = answer.sentences
    assert sentence.citations == []
    # Uncited text: a gap with a number the question does not state is dropped.
    assert sentence.text == f"{GAPS_PREFIX} the pilot's cost; who approved it; a fourth gap."
    assert answer.gaps == ["the pilot's cost", "who approved it", "a fourth gap"]
    assert (
        verify._outcome(
            {"error": None, "done": {}, "sentences": [{"text": sentence.text, "citations": []}]}
        )
        == "nothing_recorded"
    )


def test_a_passage_citation_renders_its_quote_and_verify_reads_it(db, pilot_note):
    quote = "the pilot cut nuisance alarms by 30 percent"
    answer = _answer(
        db,
        {
            "sentences": [
                _s("Alarms fell 30 percent.", [{"passage_id": pilot_note, "quote": quote}])
            ]
        },
    )
    stored = sentence_to_dict(answer.sentences[0])
    assert stored["citations"][0]["is_passage"] is True
    html = (
        build_templates()
        .env.get_template("_chat_sentence.html")
        .render(sentence=stored, note_summaries={})
    )

    parsed = verify.parse_sentence_html(html)
    assert parsed["text"] == "Alarms fell 30 percent."
    (citation,) = parsed["citations"]
    assert (citation["passage_quote"], citation["note_level"], citation["note_path"]) == (
        quote,
        False,
        "/notes/pilot-review",
    )


# A synthesis line as stored: a citation marker and a wikilink, which the note
# page shows as a dated link and as the linked note's title.
MARKED = (
    "Open request: a test plan by September 10, 2026 ^[S-call-020 @ 09:15:16]. "
    "No status update since, per [[pierce-contact]]."
)


def _page_text(db, body: str) -> str:
    """The note page's text for `body`, tags dropped."""
    return re.sub(r"<[^>]+>", " ", str(render_note_body(db, body)))


def _gate_quotes(ctx, passage_id: str, quotes: dict[str, str]):
    data = {
        "sentences": [
            _s(text, [{"passage_id": passage_id, "quote": quote}], slugs=("marked-note",))
            for text, quote in quotes.items()
        ],
        "gaps": [],
    }
    return gate_answer(data, ctx, question="q", control_ok=True, retrieved=[], usage={})


def test_a_passage_reads_as_the_note_page_shows_it(db):
    _insert_note(db, "marked-note")
    _insert_note(db, "pierce-contact")
    ctx = _build_context(db, [_hit("marked-note", chunk=MARKED)])

    (passage,) = ctx.passages.values()
    assert passage.text == (
        "Open request: a test plan by September 10, 2026. No status update since, per "
        "Pierce Contact."
    )
    kept_quote = "2026. No status update since, per Pierce Contact."
    answer = _gate_quotes(
        ctx,
        passage.passage_id,
        {"kept": kept_quote, "dropped": "2026 ^[S-call-020 @ 09:15:16]. No status update since"},
    )

    assert [s.text for s in answer.sentences] == ["kept"]
    assert answer.uncited_dropped == 1
    # The quote the gate kept is on the note page the citation links to.
    assert verify.quote_match(kept_quote, _page_text(db, MARKED)) != "none"


def test_front_matter_is_never_a_passage(db):
    _insert_note(db, "fm-note", description="")
    front_matter = '---\ntitle: "Fm Note"\ntype: account\ncrm_id: A-1090\n---'
    with db.cursor() as cur:
        cur.executemany(
            "INSERT INTO chunks (note_slug, heading, text) VALUES (%s, %s, %s)",
            [("fm-note", None, front_matter), ("fm-note", "Notes", PILOT)],
        )
    db.commit()

    ctx = _build_context(db, [_hit("fm-note", chunk=front_matter)])

    (passage,) = ctx.passages.values()
    assert passage.text == PILOT


def test_one_long_synthesis_leaves_room_for_the_next_notes_claims(db):
    line = "Northlake logged a late delivery ^[S-ticket-041 @ S-ticket-041#1].\n"
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO notes (slug, type, title, description, body_md, updated) "
            "VALUES ('weekly', 'report', 'Weekly', 'd', %s, now())",
            ("## Synthesis\n\n" + line * 600,),
        )
    db.commit()
    _insert_note(db, "signal-note")
    claim_id = _insert_claim(db, "signal-note", source="call-a", quote="late", locator="L1")

    ctx = _build_context(db, [_hit("weekly"), _hit("signal-note")])

    assert set(ctx.note_claim_ids) == {"weekly", "signal-note"}
    assert claim_id in ctx.claim_index
    assert SYNTHESIS_CUT_MARK in ctx.text
    # A marker is not a passage id; the model is shown none to confuse with one.
    assert "^[" not in ctx.text
    assert len(ctx.note_synthesis["weekly"]) <= SYNTHESIS_CHAR_CAP + len(SYNTHESIS_CUT_MARK) + 1
