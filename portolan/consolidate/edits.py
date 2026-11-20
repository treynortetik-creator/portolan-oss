"""`record_edit` and `edits_for` (`01-spec/quality-loop.md` "Human
edits": "a human edit is a source. It produces an edit record (note,
before, after, who, when, reason) and the changed claim carries a
citation of the form `^[edit:maintainer 2026-09-12]`").

Grouped with consolidation, per the build instructions, because it is the
same "who changed the synthesis" concern -- `record_edit` bumps the note
through the same `write_note` path consolidation does, just with
`changed_by=editor` instead of `changed_by="consolidate"`.

The `^[edit:<editor> <date>]` marker deliberately has no `@` in it, unlike
an ordinary `^[source-slug @ locator]` citation
(`portolan.notes.render.CITATION_RE`, `portolan.lint.checks.
_CITATION_RE`) -- both require an `@` to match at all, so an edit marker
never trips the lint's `unresolved_citation` check looking for a `sources`
row named `edit:<editor>` (there isn't one, and per this wave's build
instructions there is no `sources` row created for an edit either -- only
`edit_records` + the citation text; full "human edits are sources"
provenance, e.g. a `sources` row per edit, is out of this wave's scope and
flagged in the report). This is exactly the spec's own rendering, not a
new convention this module invents.

**Scope note**: the build instructions' signature bumps the note with
`write_note` and appends the citation "to the changed claim text or
synthesis line" -- `claim_id=None` means the synthesis; `claim_id=<id>`
means that specific claim's `claim_text`. Editing the append-only claims
list this way is a deliberate, human-attributed exception to "claims are
append-only" (the citation is *appended* to the existing text, not a
rewrite of it, and the change itself is fully attributed and dated in
`edit_records` -- nothing about the original text is lost).
"""

from __future__ import annotations

from datetime import date

import psycopg

from portolan.consolidate.models import EditRecord
from portolan.notes.persist import read_note, write_note
from portolan.util import clock

__all__ = ["StaleEditError", "edit_citation", "edits_for", "record_edit"]


def edit_citation(editor: str, when: date | None = None) -> str:
    """`^[edit:<editor> <date>]` (spec's own exact rendering)."""
    day = when or clock.today()
    return f"^[edit:{editor} {day.isoformat()}]"


class StaleEditError(ValueError):
    """`record_edit` refused a write because the caller's `before` did not
    match what is actually stored (review finding 5)."""


def _insert_edit_record(
    conn: psycopg.Connection,
    *,
    note_slug: str,
    claim_id: int | None,
    editor: str,
    before: str,
    after: str,
    reason: str,
) -> int:
    """Insert the `edit_records` row -- NOT committed here (review finding
    5): `record_edit` is the one place that commits, once, after
    `write_note` has also succeeded, so all three writes land or fail
    together (see `record_edit`'s own docstring)."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO edit_records (note_slug, claim_id, editor, before_text, after_text, reason)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (note_slug, claim_id, editor, before, after, reason),
        )
        (edit_id,) = cur.fetchone()
    return edit_id


def _update_claim_text(conn: psycopg.Connection, claim_id: int, text: str, edit_id: int) -> None:
    """Not committed here -- see `_insert_edit_record`'s docstring."""
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE claims SET claim_text = %s, human_edit_id = %s WHERE id = %s",
            (text, edit_id, claim_id),
        )


def record_edit(
    conn: psycopg.Connection,
    *,
    note_slug: str,
    editor: str,
    before: str,
    after: str,
    reason: str,
    claim_id: int | None = None,
) -> int:
    """Record a human edit AND apply it -- an `edit_records` row, the
    changed claim/synthesis actually set to `after` (plus the edit's
    citation marker), and one `write_note` (`changed_by=editor`).

    Review finding G11 (fix wave 2026-09-12): this used to only ever
    APPEND the citation marker to whatever text was *already* stored,
    never write `after` into the claim/synthesis at all -- so an edit
    record could claim "before X, after Y" while the note kept showing X
    (with a marker tacked on) forever. That silently relied on the caller
    (`portolan/web/routes/notes.py`) applying the change itself first, an
    unstated precondition the docstring called out but nothing enforced;
    call this from anywhere else and the record and the note would
    disagree, and the rebuild path (quality-loop.md: "the rebuild
    reapplies edit records") would then write text that was never
    actually approved. `record_edit` is now the single writer: it applies
    `after` (with the citation marker appended, unless already present)
    to the real claim/synthesis text itself, so `edit_records.after_text`
    and what the note actually shows can never diverge. One `write_note`
    call either way -- one note version per edit, not two.

    **Review finding 5, two more fixes.** (a) *Stale `before` refused.*
    Before this fix, `before` was written into `edit_records` verbatim,
    trusted, and never compared to what `read_note` was holding two lines
    later in `target.claim_text`/`note.synthesis` -- so an edit posted
    against a stale view (a UI tab left open while someone else, or the
    consolidation pass, changed the same text) silently overwrote
    whatever had landed in between, and `edit_records.before_text` then
    recorded a "before" that was never actually there. `record_edit` now
    captures the REAL current text first and raises `StaleEditError`
    (refusing the write entirely -- nothing is inserted or updated) when
    it doesn't match the caller's `before`, the same optimistic-
    concurrency discipline `write_note`'s own `FOR UPDATE` already uses
    one layer down. (b) *One transaction.* `_insert_edit_record` and
    `_update_claim_text` no longer commit on their own; this function's
    `write_note` call is the one that finally commits, so a failure
    partway through (e.g. `write_note` itself raising) leaves nothing
    committed at all -- not an edit record with no applied change, not a
    changed claim row whose note body still shows the old text.
    """
    note, claims = read_note(conn, note_slug)

    if claim_id is not None:
        target = next((c for c in claims if c.id == claim_id), None)
        if target is None:
            raise ValueError(f"record_edit: no such claim {claim_id!r} on note {note_slug!r}")
        actual_before = target.claim_text
    else:
        target = None
        actual_before = note.synthesis or ""

    if before != actual_before:
        raise StaleEditError(
            f"record_edit: stale `before` for note {note_slug!r}"
            f"{f' claim {claim_id!r}' if claim_id is not None else ' synthesis'} -- "
            f"the caller's `before` does not match what is actually stored; "
            f"re-read the note and retry with the current text"
        )

    citation = edit_citation(editor, clock.today())
    applied_text = after if citation in after else f"{after.rstrip()} {citation}".strip()

    if target is not None:
        target.claim_text = applied_text
    else:
        note.synthesis = applied_text

    edit_id = _insert_edit_record(
        conn,
        note_slug=note_slug,
        claim_id=claim_id,
        editor=editor,
        before=actual_before,
        after=applied_text,
        reason=reason,
    )
    if claim_id is not None:
        _update_claim_text(conn, claim_id, applied_text, edit_id)

    write_note(conn, note, claims, changed_by=editor, reason=reason)
    return edit_id


def edits_for(conn: psycopg.Connection, note_slug: str) -> list[EditRecord]:
    """Every edit recorded against `note_slug`, oldest first."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, note_slug, claim_id, editor, edited_at, before_text, after_text, reason
            FROM edit_records
            WHERE note_slug = %s
            ORDER BY edited_at, id
            """,
            (note_slug,),
        )
        return [
            EditRecord(
                id=r[0],
                note_slug=r[1],
                claim_id=r[2],
                editor=r[3],
                edited_at=r[4],
                before_text=r[5],
                after_text=r[6],
                reason=r[7],
            )
            for r in cur.fetchall()
        ]
