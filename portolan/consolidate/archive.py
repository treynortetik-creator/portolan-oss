"""Archiving the growth-threshold's excess claims into `<slug>-archive`
(`01-spec/quality-loop.md` "The consolidation pass": "when the claim list
passes the growth threshold, the older entries move to `<slug>-
archive.md`, linked from the note"; `02-reference-implementation/
wiki-tooling/wiki-schema.md`'s dossier growth guard: "Archive what is
superseded or no longer load-bearing -- NOT simply the oldest ... the
earliest entries are often the most-cited").

**Two steps, deliberately split, and in this order** (see `run.py`):

1. `select_claims_to_archive` (pure, no DB writes): oldest first, but
   never a claim the note's *current* (pre-this-round) synthesis still
   cites by `[claim:<id>]` tag -- this wave's mechanical fix for the
   wiki-schema warning above ("evicting by age alone removes exactly the
   facts people open the note for": a claim a human or an earlier
   consolidation round specifically called out stays put). `run.py` calls
   this *before* drafting, so the new synthesis is drafted and checked
   against exactly the claims that will remain -- the ones actually
   archived never need to appear in it. Checking the *old* synthesis
   (not the fresh draft) for "still cited" is what makes this ordering
   possible: the fresh draft's own citations wouldn't exist yet to check
   against, and `build-contracts.md`'s completeness rule ("every claim id
   must appear ... no missing, no unknown") is only satisfiable when the
   claim list handed to `draft_synthesis`/`check` already excludes
   whatever this round intends to archive.
2. `commit_archive` (the DB writes: `claims.archived = true`, the
   `<slug>-archive` note, the parent's `## Related` link) -- called only
   *after* `run.py`'s draft has actually passed `check()` and been
   written, so a note that fails consolidation twice is left completely
   untouched (`build-contracts.md`: "leave the note untouched"), not
   partially archived.

**No new table.** The closed note-type list (`portolan-spec-v2.md` sec 2)
has no dedicated "archive" type, and this wave owns no migrations. The
archive note is written as `type: report` (title `"<Title> — archive"`),
with its `description` prefixed `[archive]` per the build instructions --
"the search layer should down-rank it" is a change for whoever next
touches `portolan/search` (this wave does not edit it; called out again
in the wave report). `claims.archived = true` (already a real column,
`0001_init.sql`) is the actual exclusion mechanism today: `read_note`
already filters `archived = false`, so an archived claim already
disappears from its original note's rendered body and from anything
built on `read_note`'s claim list -- the archive note is a durable,
human-readable place those claims still live, not the only place
`archived` takes effect.
"""

from __future__ import annotations

from datetime import UTC, datetime

import psycopg

from portolan.consolidate.draft import CLAIM_ID_TAG_RE
from portolan.notes.models import Claim, NoteRow, NoteStatus, NoteType
from portolan.notes.persist import (
    _citations_for_claims,
    link_related,
    note_exists,
    read_note,
    write_note,
)

__all__ = ["archive_note_slug", "commit_archive", "select_claims_to_archive"]


def archive_note_slug(note_slug: str) -> str:
    return f"{note_slug}-archive"


def _cited_claim_ids(text: str) -> set[int]:
    return {int(m.group(1)) for m in CLAIM_ID_TAG_RE.finditer(text)}


def select_claims_to_archive(note: NoteRow, claims: list[Claim], *, threshold: int) -> list[Claim]:
    """`len(claims) - threshold` claims to archive, oldest first, skipping
    any claim `note.synthesis` (the current, pre-this-round prose) already
    cites via `[claim:<id>]` (module docstring, step 1). `[]` when
    `claims` is already at or under `threshold`."""
    excess = len(claims) - threshold
    if excess <= 0:
        return []

    load_bearing = _cited_claim_ids(note.synthesis or "")
    ordered = sorted(claims, key=lambda c: (c.recorded, c.id or 0))

    to_archive: list[Claim] = []
    for claim in ordered:
        if len(to_archive) >= excess:
            break
        if claim.id is not None and claim.id in load_bearing:
            continue
        to_archive.append(claim)

    return to_archive


def _mark_archived(conn: psycopg.Connection, claim_ids: list[int]) -> None:
    with conn.cursor() as cur:
        cur.execute("UPDATE claims SET archived = true WHERE id = ANY(%s)", (claim_ids,))
    conn.commit()


def _archived_claims_for(conn: psycopg.Connection, note_slug: str) -> list[Claim]:
    """Every already-archived claim for `note_slug`, WITH its extra
    citations (bug hunt 2026-09-18, finding 5).

    Before this fix, this SELECT omitted `claim_citations` entirely --
    `read_note`'s equivalent query joins them in
    (`persist._citations_for_claims`, review finding G4), but this one
    didn't, so `_write_archive_note`'s second and later consolidation
    rounds re-rendered an already-archived claim from scratch and
    silently dropped every corroborating citation it had picked up (a
    `same`-verdict sighting from a second source, or the same source at a
    second locator): the claim survives in `claim_citations` itself (this
    function just never reads it), but the rendered `<slug>-archive`
    note -- the only thing a reader or the lint ever sees -- loses that
    provenance chain link for good on the very next archive pass.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, note_slug, fact_type, claim_text, source_slug, locator, quote,
                   speaker, valid_from, valid_to, recorded, superseded_at, severity,
                   sentiment, confidence, superseded_by
            FROM claims
            WHERE note_slug = %s AND archived = true
            ORDER BY recorded, id
            """,
            (note_slug,),
        )
        rows = cur.fetchall()
        citations_by_claim = _citations_for_claims(cur, [r[0] for r in rows])
        return [
            Claim(
                id=r[0],
                note_slug=r[1],
                fact_type=r[2],
                claim_text=r[3],
                source_slug=r[4],
                locator=r[5],
                quote=r[6],
                speaker=r[7],
                valid_from=r[8],
                valid_to=r[9],
                recorded=r[10],
                superseded_at=r[11],
                severity=r[12],
                sentiment=r[13],
                confidence=r[14],
                superseded_by=r[15],
                citations=citations_by_claim.get(r[0], []),
            )
            for r in rows
        ]


def _write_archive_note(
    conn: psycopg.Connection, note: NoteRow, new_claims: list[Claim], *, embedder=None
) -> str:
    """(Re-)render `<note.slug>-archive` from every currently-archived
    claim for `note.slug` PLUS `new_claims` -- always the full set, not
    just this pass's newly-archived batch, so a second consolidation
    later adds to the same durable archive rather than needing to merge
    with it.

    `new_claims` (review finding 4): this round's own batch, passed in
    explicitly rather than re-read from `claims.archived = true` --
    `commit_archive` now calls this function *before* flipping that flag
    (see its own docstring), so a DB query for "everything already
    archived" would miss exactly the claims this call exists to write.
    Merged in here, keyed by id, so a claim already archived by an
    earlier consolidation round (already in `_archived_claims_for`'s
    result) is never duplicated if it somehow also appeared in
    `new_claims`.
    """
    slug = archive_note_slug(note.slug)
    already_archived = _archived_claims_for(conn, note.slug)
    seen_ids = {c.id for c in already_archived if c.id is not None}
    archived_claims = already_archived + [c for c in new_claims if c.id not in seen_ids]
    archived_claims.sort(key=lambda c: (c.recorded, c.id or 0))
    now = datetime.now(UTC)

    if note_exists(conn, slug):
        archive_note, _existing_claims = read_note(conn, slug)
    else:
        archive_note = NoteRow(
            slug=slug,
            type=NoteType.REPORT,
            title=f"{note.title} — archive",
            description=(
                f"[archive] Superseded claims from {note.title}, moved here by "
                "consolidation. Excluded from default search; included on "
                "history queries -- see 01-spec/quality-loop.md."
            ),
            status=NoteStatus.LIVING,
            created=now,
            updated=now,
        )
    archive_note.related = [note.slug]
    archive_note.synthesis = (
        f"Claims archived from [[{note.slug}]] during consolidation, oldest first, "
        "excluding anything the prior synthesis still cited."
    )
    write_note(
        conn,
        archive_note,
        archived_claims,
        changed_by="consolidate",
        reason="archive",
        embedder=embedder,
    )
    return slug


def commit_archive(
    conn: psycopg.Connection, note: NoteRow, to_archive: list[Claim], *, embedder=None
) -> str | None:
    """Persist an archiving decision already made by `select_claims_to_
    archive`: (re-)write `<slug>-archive`, THEN flip `claims.archived`,
    then link the archive from the parent note's `## Related` (module
    docstring, step 2). Called only after the parent note's own
    consolidated write has succeeded. `None`, no-op, when `to_archive` is
    empty.

    **Ordering (review finding 4, blocker-adjacent should-fix).** Before
    this fix, `claims.archived` flipped FIRST and committed immediately --
    `read_note` filters `archived = false`, so from that instant those
    claims were gone from the parent note's body, from search, from
    everywhere `read_note` feeds, while the note that is supposed to hold
    them (`_write_archive_note`, which reaches the embedding model when
    `embedder` is given) did not exist yet. A failure in between left the
    claims hidden with nothing linking to them, and nothing rolled it
    back -- `_mark_archived`'s `UPDATE` was already committed. Writing the
    archive note first means a failure here now leaves `to_archive`
    exactly as it was: still visible on the parent note, not archived,
    safely retried on the next consolidation pass. The ordering comment
    at this module's own docstring already argued for atomicity ("not
    partially archived") for draft-versus-archive; this makes it true of
    these two steps too.

    `embedder` (review finding G21): forwarded to the archive note's own
    `write_note` call, so its `chunks` get re-embedded too, not just the
    parent note's."""
    if not to_archive:
        return None

    archive_slug = _write_archive_note(conn, note, to_archive, embedder=embedder)
    ids = [c.id for c in to_archive if c.id is not None]
    _mark_archived(conn, ids)
    link_related(
        conn,
        note.slug,
        archive_slug,
        changed_by="consolidate",
        reason="archive created during consolidation",
    )
    return archive_slug
