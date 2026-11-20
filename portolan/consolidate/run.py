"""`consolidate_note` and `run_consolidation` (`01-spec/quality-loop.md`
"The consolidation pass").

    read note + claims
      -> select_claims_to_archive (oldest-first, skipping load-bearing ones)
      -> draft (cheap model, over the REMAINING, LIVE claims only) -> check
           -> ok?   write_note, THEN commit_archive
           -> fail? retry once (strong model) -> check
                -> ok?   write_note, THEN commit_archive
                -> fail? review_queue, note completely untouched

Archiving is *selected* before drafting (so the draft only ever needs to
cite what will remain -- see `portolan.consolidate.archive`'s module
docstring for why) but only *committed* to the database after a
successful write, so a note that fails both attempts is left exactly as
it was (`build-contracts.md`: "leave the note untouched").
"""

from __future__ import annotations

import json

import psycopg

from portolan.consolidate.archive import commit_archive, select_claims_to_archive
from portolan.consolidate.draft import check, draft_synthesis
from portolan.consolidate.models import CheckResult, ConsolidationOutcome
from portolan.consolidate.trigger import CONSOLIDATION_CLAIMS_THRESHOLD, needs_consolidation
from portolan.notes.persist import read_note, write_note

__all__ = ["CONSOLIDATION_REASON", "consolidate_note", "run_consolidation"]

CONSOLIDATION_REASON = "consolidation"


def _queue_failed_consolidation(
    conn: psycopg.Connection, note_slug: str, result: CheckResult
) -> int:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO review_queue (kind, payload) VALUES ('failed_consolidation', %s::jsonb) "
            "RETURNING id",
            (json.dumps({"note_slug": note_slug, "reasons": result.reasons}),),
        )
        (review_id,) = cur.fetchone()
    conn.commit()
    return review_id


def _record_consolidation(conn: psycopg.Connection, note_slug: str, version: int) -> None:
    """One `changelog` row this pass's `needs_consolidation` reads back as
    "the last consolidation" (`portolan.consolidate.trigger` module
    docstring: no dedicated table, `changelog.details` carries it)."""
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO changelog (kind, summary, details) VALUES (%s, %s, %s::jsonb)",
            (
                "consolidate",
                f"consolidated {note_slug} (now version {version})",
                json.dumps({"note_slug": note_slug, "version": version}),
            ),
        )
    conn.commit()


def consolidate_note(
    conn: psycopg.Connection,
    llm,
    note_slug: str,
    *,
    strong_llm=None,
    claims_threshold: int = CONSOLIDATION_CLAIMS_THRESHOLD,
    embedder=None,
) -> ConsolidationOutcome:
    """Draft with `llm` (the cheap model), check mechanically, retry once
    with `strong_llm` on failure, queue for a human on a second failure
    (`01-spec/quality-loop.md` "The consolidation pass").

    `embedder` (review finding G21, fix wave 2026-09-12): passed through
    to `write_note`/`commit_archive` so the note's `chunks` (the
    searchable top `write_note` renders) get re-embedded in the same
    call, not left stale. Consolidation exists specifically to rewrite
    that searchable top -- leaving it unindexed defeated the point.
    `None` is still accepted (search then updates on its own next full
    `index_pending` pass, same as before this fix).
    """
    note, claims = read_note(conn, note_slug)
    if not claims:
        return ConsolidationOutcome(note_slug=note_slug, status="skipped_no_claims")

    to_archive = select_claims_to_archive(note, claims, threshold=claims_threshold)
    archive_ids = {c.id for c in to_archive}
    remaining = [c for c in claims if c.id not in archive_ids]
    # The synthesis is the note's current state, so it is drafted from live
    # claims only. A superseded claim still renders (struck through) in the
    # claim list, but handing it to the model unmarked let the synthesis
    # state a retired fact (the old renewal date) as current.
    live = [c for c in remaining if c.superseded_at is None]
    if not live:
        return ConsolidationOutcome(note_slug=note_slug, status="skipped_no_claims")

    draft = draft_synthesis(llm, note, live)
    result = check(draft, live)
    attempts = 1
    used_strong = False

    if not result.ok and strong_llm is not None:
        draft = draft_synthesis(strong_llm, note, live)
        result = check(draft, live)
        attempts = 2
        used_strong = True

    if not result.ok:
        review_id = _queue_failed_consolidation(conn, note_slug, result)
        return ConsolidationOutcome(
            note_slug=note_slug,
            status="failed",
            draft_attempts=attempts,
            used_strong=used_strong,
            check=result,
            review_queue_id=review_id,
        )

    note.synthesis = draft
    write_note(
        conn,
        note,
        remaining,
        changed_by="consolidate",
        reason=CONSOLIDATION_REASON,
        embedder=embedder,
    )

    archive_slug = commit_archive(conn, note, to_archive, embedder=embedder)

    _record_consolidation(conn, note_slug, note.version)

    return ConsolidationOutcome(
        note_slug=note_slug,
        status="consolidated",
        draft_attempts=attempts,
        used_strong=used_strong,
        check=result,
        new_version=note.version,
        archived_claim_ids=sorted(archive_ids),
        archive_note_slug=archive_slug,
    )


def run_consolidation(
    conn: psycopg.Connection,
    llm,
    strong_llm,
    *,
    limit: int = 20,
    embedder=None,
) -> list[ConsolidationOutcome]:
    """Consolidate up to `limit` notes `needs_consolidation` names, in
    order."""
    slugs = needs_consolidation(conn)[:limit]
    return [
        consolidate_note(conn, llm, slug, strong_llm=strong_llm, embedder=embedder)
        for slug in slugs
    ]
