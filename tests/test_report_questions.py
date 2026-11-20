"""`portolan.report.questions` tests (wave C1)."""

# ruff: noqa: F811  (pytest fixtures imported by name; see test_report_helpers.py)

from __future__ import annotations

import json
import re
from datetime import date

import pytest
from test_report_helpers import FullGraph, full_graph, report_llm  # noqa: F401

from portolan.llm.fake import FakeLLMClient
from portolan.report import questions
from portolan.search.query import search
from portolan.util import clock as clock_module

WINDOW_START = date(2026, 3, 2)
# Should-fix 12 (review finding 12): under the DEFAULT record clock,
# `answer_standing_questions` now bounds its search to notes updated on
# or before `as_of` (`_record_touched_note_slugs`), closing the "a past
# week's report answers from later material" bug. This file's graph is
# built once, by `full_graph`, at REAL wall-clock time ("now") -- every
# note's own `updated` is stamped then, regardless of which historical
# date the corpus's own content pretends to be from -- so `as_of` must be
# "now" too for that bound to include anything at all. A fixed past date
# (this used to be `date(2026, 9, 4)`, the corpus's own content window)
# worked only because the old, buggy code had no upper bound to begin
# with; these tests are about citation/numeral behaviour, not clock
# semantics, so `clock.today()` -- not a `clock="world"` switch -- keeps
# them testing what they always meant to.
AS_OF = clock_module.today()


@pytest.fixture()
def competitor_question(full_graph: FullGraph):
    """One standing question, inserted idempotently -- other test modules
    may have already added it to this session's shared graph."""
    with full_graph.conn.cursor() as cur:
        cur.execute(
            "SELECT id FROM standing_questions WHERE question = %s",
            ("Which accounts mentioned a competitor this week?",),
        )
        row = cur.fetchone()
        if row is not None:
            return row[0]
        cur.execute(
            "INSERT INTO standing_questions (reader, question) VALUES (%s, %s) RETURNING id",
            ("ops", "Which accounts mentioned a competitor this week?"),
        )
        (question_id,) = cur.fetchone()
    full_graph.conn.commit()
    return question_id


def test_answer_cites_only_retrieved_notes(
    full_graph: FullGraph, report_llm: FakeLLMClient, competitor_question: int
) -> None:
    answers = questions.answer_standing_questions(
        full_graph.conn, report_llm, full_graph.embedder, since=WINDOW_START, as_of=AS_OF
    )
    ours = next(a for a in answers if a.question_id == competitor_question)

    retrieved = search(
        full_graph.conn, full_graph.embedder, ours.question, k=12, since=WINDOW_START
    )
    retrieved_slugs = {h.note_slug for h in retrieved.hits}
    assert ours.citations
    assert set(ours.citations).issubset(retrieved_slugs)


def test_answer_is_persisted_in_standing_question_answers(
    full_graph: FullGraph, report_llm: FakeLLMClient, competitor_question: int
) -> None:
    questions.answer_standing_questions(
        full_graph.conn, report_llm, full_graph.embedder, since=WINDOW_START, as_of=AS_OF
    )
    with full_graph.conn.cursor() as cur:
        cur.execute(
            "SELECT answer_md, citations FROM standing_question_answers WHERE question_id = %s",
            (competitor_question,),
        )
        row = cur.fetchone()
    assert row is not None
    answer_md, citations = row
    assert answer_md
    assert isinstance(json.loads(json.dumps(citations)), list)


def test_a_sentence_citing_outside_the_retrieved_set_is_dropped(
    full_graph: FullGraph, competitor_question: int
) -> None:
    """Build a `FakeLLMClient` that always answers with one sentence citing
    a note slug guaranteed never to be retrieved for this question --
    `answer_standing_questions` must drop that sentence whole, not just
    trim its citation list."""

    def _bad_responder(*, system, user, json_schema, model, surface):
        del system, user, json_schema, model, surface
        return {
            "sentences": [
                {"text": "This cites a note nobody retrieved.", "cites": ["not-a-real-note-slug"]}
            ]
        }

    bad_llm = FakeLLMClient(_bad_responder)
    answers = questions.answer_standing_questions(
        full_graph.conn, bad_llm, full_graph.embedder, since=WINDOW_START, as_of=AS_OF
    )
    ours = next(a for a in answers if a.question_id == competitor_question)
    assert ours.citations == []
    assert "not-a-real-note-slug" not in ours.answer_md
    assert ours.answer_md  # the "no evidence" fallback text, not an empty string


def test_a_sentence_with_an_unsupplied_numeral_is_dropped(
    full_graph: FullGraph, competitor_question: int
) -> None:
    """The numbers rule (polish wave 2026-09-12): a sentence that cites a
    genuinely retrieved note but states a number nothing in the retrieved
    claims backs up is dropped whole, exactly like a bad citation, and
    logged to `review_queue` with `reason='numeral_not_supplied'`."""
    with full_graph.conn.cursor() as cur:
        cur.execute("SELECT question FROM standing_questions WHERE id = %s", (competitor_question,))
        (question_text,) = cur.fetchone()
    retrieved = search(
        full_graph.conn, full_graph.embedder, question_text, k=12, since=WINDOW_START
    )
    assert retrieved.hits, "test needs at least one retrieved note to cite"
    real_slug = retrieved.hits[0].note_slug

    def _fabricating_responder(*, system, user, json_schema, model, surface):
        del system, model, surface
        assert "Figures you may state" in user
        return {
            "sentences": [
                {
                    "text": "This account had 39 competitor mentions this window.",
                    "cites": [real_slug],
                }
            ]
        }

    fabricating_llm = FakeLLMClient(_fabricating_responder)
    answers = questions.answer_standing_questions(
        full_graph.conn, fabricating_llm, full_graph.embedder, since=WINDOW_START, as_of=AS_OF
    )
    ours = next(a for a in answers if a.question_id == competitor_question)
    assert "39" not in ours.answer_md
    assert ours.citations == []

    with full_graph.conn.cursor() as cur:
        cur.execute(
            "SELECT payload->>'detail' FROM review_queue "
            "WHERE kind = 'other' AND payload->>'reason' = 'numeral_not_supplied' "
            "AND payload->>'surface' = 'standing_question' "
            "ORDER BY id DESC LIMIT 1"
        )
        row = cur.fetchone()
    assert row is not None
    assert "39" in row[0]


def test_a_sentence_stating_a_cited_claims_own_figure_is_kept(
    full_graph: FullGraph, competitor_question: int
) -> None:
    """Bughunt-2026-09-18 finding 1: `_claims_context` harvests each
    claim's numbers under a marker (review finding 9's own fix); before
    this fix that marker was the claim's `(source_slug, locator)`, only
    licensed by an inline `^[source @ locator]` marker literally present
    in the ANSWER'S OWN TEXT (`citation_markers_in(text)`). But
    `STANDING_QUESTION_SYSTEM` never asks the model to write one -- it
    asks for a `cites` list of NOTE SLUGS alongside each sentence -- so
    `present_markers` was always empty and every genuinely-cited figure
    was rejected as if fabricated, collapsing the answer to "No claim
    ... answers this question" even though the prompt told the model the
    figure was allowed. A sentence that cites a note via `cites` and
    restates a number straight out of that note's own claim text must
    now be KEPT, not dropped."""
    with full_graph.conn.cursor() as cur:
        cur.execute("SELECT question FROM standing_questions WHERE id = %s", (competitor_question,))
        (question_text,) = cur.fetchone()
    retrieved = search(
        full_graph.conn, full_graph.embedder, question_text, k=12, since=WINDOW_START
    )
    assert retrieved.hits, "test needs at least one retrieved note to cite"
    retrieved_slugs = [h.note_slug for h in retrieved.hits]

    with full_graph.conn.cursor() as cur:
        cur.execute(
            "SELECT note_slug, claim_text FROM claims "
            "WHERE note_slug = ANY(%s) AND archived = false AND claim_text ~ '[0-9]' "
            "ORDER BY note_slug, recorded DESC LIMIT 1",
            (retrieved_slugs,),
        )
        row = cur.fetchone()
    assert row is not None, "test needs a retrieved claim with a numeral in it"
    note_slug, claim_text = row
    number = re.search(r"\d+", claim_text).group(0)

    def _restating_responder(*, system, user, json_schema, model, surface):
        del system, model, json_schema, surface
        assert "Figures you may state" in user
        assert number in user  # the figure really was handed over, not just asserted here
        return {
            "sentences": [{"text": f"A claim on this note states {number}.", "cites": [note_slug]}]
        }

    llm = FakeLLMClient(_restating_responder)
    answers = questions.answer_standing_questions(
        full_graph.conn, llm, full_graph.embedder, since=WINDOW_START, as_of=AS_OF
    )
    ours = next(a for a in answers if a.question_id == competitor_question)
    # Kept (not the "No claim ... answers this question" fallback) is
    # only possible if the numeral check licensed `number` -- the whole
    # point of this test.
    assert number in ours.answer_md
    assert note_slug in ours.citations


def test_deactivated_question_is_skipped(full_graph: FullGraph, report_llm: FakeLLMClient) -> None:
    with full_graph.conn.cursor() as cur:
        cur.execute(
            "INSERT INTO standing_questions (reader, question, active) "
            "VALUES ('ops', 'A retired question nobody should answer', false) RETURNING id",
        )
        (retired_id,) = cur.fetchone()
    full_graph.conn.commit()

    answers = questions.answer_standing_questions(
        full_graph.conn, report_llm, full_graph.embedder, since=WINDOW_START, as_of=AS_OF
    )
    assert all(a.question_id != retired_id for a in answers)


def test_record_touched_note_slugs_bounds_on_both_ends(full_graph: FullGraph) -> None:
    """Should-fix 12 / test gap 12 (review finding 12): the record-clock
    sibling of the bound `_world_touched_note_slugs` already had. Before
    this fix, `answer_standing_questions`'s record-clock path used
    `search`'s own `since=`, which has NO upper bound at all -- re-running
    a PAST week (the exact thing a multi-report backfill does) drew in
    every note updated AFTER that week too, so a W33 report regenerated
    today answered its standing questions from W36 material."""
    conn = full_graph.conn
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO notes (slug, type, title, description, created, updated) "
            "VALUES ('record-clock-bound-in-window', 'signal', 'In window', 'fixture', "
            "'2026-08-10 12:00:00+00', '2026-08-10 12:00:00+00') "
            "ON CONFLICT (slug) DO UPDATE SET updated = EXCLUDED.updated"
        )
        cur.execute(
            "INSERT INTO notes (slug, type, title, description, created, updated) "
            "VALUES ('record-clock-bound-after-window', 'signal', 'After window', 'fixture', "
            "'2026-08-10 12:00:00+00', '2026-09-10 12:00:00+00') "
            "ON CONFLICT (slug) DO UPDATE SET updated = EXCLUDED.updated"
        )
    conn.commit()

    touched = questions._record_touched_note_slugs(
        conn, since=date(2026, 8, 1), as_of=date(2026, 8, 20)
    )
    assert "record-clock-bound-in-window" in touched
    assert "record-clock-bound-after-window" not in touched


def test_world_touched_note_slugs_excludes_archived_claims(full_graph: FullGraph) -> None:
    """Nit 17 fix: `_world_touched_note_slugs` used to select from every
    claim regardless of `archived`, unlike `_claims_context` and
    `_fetch_candidates` -- an archived claim could still pull its note
    into the world-clock search scope."""
    conn = full_graph.conn
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO notes (slug, type, title, description) VALUES "
            "('world-touched-archived-fixture', 'signal', 'Archived fixture', 'fixture') "
            "ON CONFLICT (slug) DO NOTHING"
        )
        cur.execute("SELECT slug FROM sources LIMIT 1")
        (any_source,) = cur.fetchone()
        cur.execute(
            "INSERT INTO claims (note_slug, fact_type, claim_text, source_slug, locator, "
            "recorded, archived) VALUES (%s, 'complaint', 'an archived-only claim', %s, "
            "'world-touched-archived#1', %s, true)",
            ("world-touched-archived-fixture", any_source, date(2026, 8, 15)),
        )
    conn.commit()

    touched = questions._world_touched_note_slugs(
        conn, since=date(2026, 8, 1), as_of=date(2026, 8, 20)
    )
    assert "world-touched-archived-fixture" not in touched
