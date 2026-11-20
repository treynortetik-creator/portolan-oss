"""Feed 3: standing questions (`01-spec/surfacing-and-weekly-report.md`
"3. Standing questions").

For every active question in `standing_questions`: run the fused search
over notes touched in the window, hand the retrieved notes' claims to one
`llm.complete` call, and keep each answer sentence only if every note slug
it cites was retrieved and every numeral passes `check_numerals`. A failing
sentence is dropped whole. Answers go to `standing_question_answers`.
"""

from __future__ import annotations

import json
from datetime import date

import psycopg
from pydantic import BaseModel, ConfigDict, Field

from portolan.llm.base import LLMClient
from portolan.report.models import StandingQuestionAnswer
from portolan.report.numbers import NumberFacts, check_numerals, queue_numeral_rejection
from portolan.report.util import iso_week_str
from portolan.search.embed import Embedder
from portolan.search.query import search
from portolan.util import clock as clock_module

__all__ = ["STANDING_QUESTION_SYSTEM", "answer_standing_questions"]

STANDING_QUESTION_SYSTEM = (
    "You answer one reader's standing question from a set of retrieved claims "
    "about customer accounts. Write one to five sentences. Every sentence must "
    "list the exact note slugs (from the ones given) it draws on -- never a "
    "slug that wasn't given, never an empty list. If nothing in the retrieved "
    "claims answers the question, say so in one sentence citing whichever "
    "notes you checked. Use absolute dates, never 'this week' or 'recently'. "
    "This feed computes no figures of its own -- never state a number "
    "(a count, a sum, a percentage) unless a claim you cite states it "
    "verbatim; never compute, count, or estimate one yourself. Reply as "
    "JSON."
)

_MAX_CLAIMS_PER_NOTE = 5
_MAX_NOTES = 12


class _AnswerSentence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    cites: list[str] = Field(default_factory=list)


class _StandingAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sentences: list[_AnswerSentence]


def _standing_answer_json_schema() -> dict:
    return _StandingAnswer.model_json_schema()


def _claims_context(conn: psycopg.Connection, note_slugs: list[str]) -> tuple[str, NumberFacts]:
    """The retrieved notes' claims rendered for the prompt, and a
    `NumberFacts` with each claim's numerals harvested under the marker
    `(note_slug, "cites")`. The answer cites note slugs in a `cites` list,
    not inline markers, so a numeral is licensed per cited note."""
    facts = NumberFacts()
    if not note_slugs:
        return "(no notes retrieved)", facts
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT note_slug, claim_text, source_slug, locator
            FROM claims
            WHERE note_slug = ANY(%s) AND archived = false
            ORDER BY note_slug, recorded DESC
            """,
            (note_slugs,),
        )
        rows = cur.fetchall()

    by_note: dict[str, list[str]] = {}
    for note_slug, claim_text, source_slug, locator in rows:
        lines = by_note.setdefault(note_slug, [])
        if len(lines) < _MAX_CLAIMS_PER_NOTE:
            lines.append(f"  - {claim_text} ^[{source_slug} @ {locator}]")
            facts.harvest(claim_text, marker=(note_slug, "cites"))

    blocks = [f"[[{slug}]]\n" + "\n".join(lines) for slug, lines in by_note.items()]
    context = "\n\n".join(blocks) if blocks else "(no claims on the retrieved notes)"
    return context, facts


def _active_questions(conn: psycopg.Connection) -> list[tuple[int, str, str]]:
    """Every active standing question. Not filtered by `created`: a question
    is a standing watchlist entry, and a backfill plants questions before
    answering past weeks."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, reader, question FROM standing_questions WHERE active = true ORDER BY id"
        )
        return cur.fetchall()


def _store_answer(
    conn: psycopg.Connection, *, question_id: int, week: str, answer_md: str, citations: list[str]
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO standing_question_answers (question_id, week, answer_md, citations)
            VALUES (%s, %s, %s, %s::jsonb)
            ON CONFLICT (question_id, week) DO UPDATE SET
                answer_md = EXCLUDED.answer_md,
                citations = EXCLUDED.citations,
                created = now()
            """,
            (question_id, week, answer_md, _to_jsonb(citations)),
        )
    conn.commit()


def _to_jsonb(citations: list[str]) -> str:
    return json.dumps(citations)


def _world_touched_note_slugs(conn: psycopg.Connection, *, since: date, as_of: date) -> list[str]:
    """Every note with a non-archived claim `recorded` (the source's date) in
    `[since, as_of]`: "notes touched this week" on the world clock, where
    `notes.updated` would be the backfill's write time."""
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT DISTINCT note_slug FROM claims "
            f"WHERE archived = false "
            f"AND {clock_module.world_date_sql('recorded')} BETWEEN %s AND %s",
            (since, as_of),
        )
        return [r[0] for r in cur.fetchall()]


def _record_touched_note_slugs(conn: psycopg.Connection, *, since: date, as_of: date) -> list[str]:
    """Every note whose `updated` (deployment zone) falls in `[since, as_of]`.
    Bounded on both ends because `search(since=)` has no upper bound, so a
    re-run past week would draw in later material."""
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT slug FROM notes "
            f"WHERE {clock_module.world_date_sql('updated')} BETWEEN %s AND %s",
            (since, as_of),
        )
        return [r[0] for r in cur.fetchall()]


def answer_standing_questions(
    conn: psycopg.Connection,
    llm: LLMClient,
    embedder: Embedder,
    *,
    since: date,
    as_of: date | None = None,
    clock: str = "record",
    prose_model: str | None = None,
    week: str | None = None,
) -> list[StandingQuestionAnswer]:
    """Answer every active standing question for `[since, as_of]` and store
    the answers.

    `week`: the label answers are stored under (default: `as_of`'s ISO week,
    which is the next week when `as_of` is a Monday, so callers may pass it).
    `clock`: `"record"` searches notes whose `updated` is in the window;
    `"world"` searches notes with a claim `recorded` in it. A `"record"` run
    for a closed past week finds nothing (a backfill writes `updated` as
    today); use `"world"` for backfills. `prose_model`: model for the one
    `llm.complete()` per question (`None` = the client's default)."""
    as_of = as_of or clock_module.today()
    week = week or iso_week_str(as_of)

    if clock == "world":
        touched_note_slugs = _world_touched_note_slugs(conn, since=since, as_of=as_of)
    else:
        touched_note_slugs = _record_touched_note_slugs(conn, since=since, as_of=as_of)

    results: list[StandingQuestionAnswer] = []
    for question_id, reader, question in _active_questions(conn):
        search_result = search(
            conn, embedder, question, k=_MAX_NOTES, note_slugs=touched_note_slugs
        )
        retrieved_slugs = {hit.note_slug for hit in search_result.hits}
        # No computed figures here: a sentence may restate a number only from
        # a claim on a note it cites (licensed per note, not per claim).
        context, facts = _claims_context(conn, [hit.note_slug for hit in search_result.hits])

        user = (
            f"Question: {question}\n\nRetrieved notes and their claims:\n\n{context}"
            "\n\nFigures you may state (use no other numbers):\n" + facts.as_table()
        )
        result = llm.complete(
            system=STANDING_QUESTION_SYSTEM,
            user=user,
            json_schema=_standing_answer_json_schema(),
            model=prose_model,
            surface="report",
        )
        parsed = _StandingAnswer.model_validate(result.data)

        kept_sentences: list[str] = []
        citations_used: list[str] = []
        for sentence in parsed.sentences:
            if not sentence.cites:
                continue
            if any(cite not in retrieved_slugs for cite in sentence.cites):
                continue  # cites outside the retrieved set -- drop the whole sentence
            # Same marker shape `_claims_context` harvested under.
            present_markers = {(cite, "cites") for cite in sentence.cites}
            numeral_violations = check_numerals(
                sentence.text, facts, present_markers=present_markers
            )
            if numeral_violations:
                queue_numeral_rejection(
                    conn,
                    note_slug=None,
                    surface="standing_question",
                    violations=numeral_violations,
                    text=sentence.text,
                    facts=facts,
                )
                continue  # a fabricated number rejects the whole sentence, like a bad citation
            kept_sentences.append(sentence.text)
            for cite in sentence.cites:
                if cite not in citations_used:
                    citations_used.append(cite)

        answer_md = (
            " ".join(kept_sentences)
            if kept_sentences
            else ("No claim in the notes touched this window answers this question.")
        )
        _store_answer(
            conn, question_id=question_id, week=week, answer_md=answer_md, citations=citations_used
        )
        results.append(
            StandingQuestionAnswer(
                question_id=question_id,
                reader=reader,
                question=question,
                week=week,
                answer_md=answer_md,
                citations=citations_used,
            )
        )
    return results
