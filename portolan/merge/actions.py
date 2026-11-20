"""`merge_action` (`01-spec/build-contracts.md` "Merge": "dedup against
open actions on the same note by difflib >= 0.7, else insert into
`actions` and create/refresh an action note linked to the entity").

`entity_id` is passed in because an `Action` carries no entities list of
its own; `merge_extraction` resolves it once per source.

Dedup runs twice: exactly against actions the same document already
produced (`_duplicate_action_for_source`), then fuzzily against the open
actions linked from the entity's note.

The note is written before the `actions` row because `actions.note_slug`
is a NOT NULL foreign key into `notes`, so the row's id is drawn from its
sequence first (`reserve_action_note_slug`) and the slug's number is that id.
"""

from __future__ import annotations

import difflib

import psycopg

from portolan.extract.schema import Action, EntityKind
from portolan.ingest.models import SourceRow
from portolan.merge.models import ActionOutcome
from portolan.merge.util import recorded_dt
from portolan.notes import (
    Claim,
    NoteRow,
    NoteStatus,
    NoteType,
    action_note_slug,
    ensure_entity_note,
    link_mentions,
    link_related,
    note_exists,
    write_note,
)
from portolan.registry import Registry
from portolan.search.embed import Embedder
from portolan.util import clock

DEDUP_THRESHOLD = 0.7
_NOTE_BEARING_KINDS = {
    EntityKind.ACCOUNT,
    EntityKind.CONTACT,
    EntityKind.PRODUCT,
    EntityKind.COMPETITOR,
}


def _open_actions_for_entity(
    conn: psycopg.Connection, entity_note_slug: str
) -> list[tuple[int, str, str]]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT a.id, a.what, a.note_slug
            FROM actions a
            JOIN links l ON l.to_slug = a.note_slug AND l.kind = 'body'
            WHERE l.from_slug = %s AND a.status = 'open'
            ORDER BY a.id
            """,
            (entity_note_slug,),
        )
        return cur.fetchall()


def _find_duplicate(what: str, open_actions: list[tuple[int, str, str]]) -> tuple[int, str] | None:
    best: tuple[int, str] | None = None
    best_ratio = 0.0
    for action_id, existing_what, note_slug in open_actions:
        ratio = difflib.SequenceMatcher(None, what, existing_what).ratio()
        if ratio >= DEDUP_THRESHOLD and ratio > best_ratio:
            best_ratio = ratio
            best = (action_id, note_slug)
    return best


def _normalize_action_text(what: str) -> str:
    return " ".join(what.split()).lower()


def _duplicate_action_for_source(
    conn: psycopg.Connection, source_slug: str, locator: str, what: str
) -> tuple[int, str] | None:
    """The action this same document already produced at this locator
    with this wording, in any status: a retry, a redrive, or a re-crawled
    version (same `sources.source_id`). Exact on `(source family, locator,
    normalized what)`; any status, so a re-crawl never re-opens a done or
    dropped action (`tests/test_bughunt_wave2.py`)."""
    normalized = _normalize_action_text(what)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT a.id, a.what, a.note_slug FROM actions a
            WHERE a.locator = %s
              AND a.source_slug IN (
                  SELECT s.slug FROM sources s
                  WHERE s.slug = %s
                     OR s.source_id = (SELECT source_id FROM sources WHERE slug = %s)
              )
            ORDER BY a.id
            """,
            (locator, source_slug, source_slug),
        )
        rows = cur.fetchall()
    for action_id, existing_what, note_slug in rows:
        if _normalize_action_text(existing_what) == normalized:
            return action_id, note_slug
    return None


def reserve_action_note_slug(conn: psycopg.Connection, what: str) -> tuple[int, str]:
    """`(actions.id, note_slug)` for a new action: the id is drawn from the
    table's own sequence now, and the caller inserts the row with it
    (`OVERRIDING SYSTEM VALUE`), so the slug's number is the row's id. A
    slug some older note already holds draws the next id."""
    while True:
        with conn.cursor() as cur:
            cur.execute("SELECT nextval(pg_get_serial_sequence('actions', 'id'))")
            action_id = cur.fetchone()[0]
        slug = action_note_slug(action_id, what)
        if not note_exists(conn, slug):
            return action_id, slug


def _resolve_owner(
    conn: psycopg.Connection,
    registry: Registry,
    owner_mention: str | None,
    *,
    commit: bool = True,
) -> tuple[int | None, str | None]:
    """`(owner_entity_dbid, owner_note_slug)` for the action's named owner.

    Vendor-side only (`01-spec/actions-and-owners-2026-09-19.md`: "Customer
    contacts can never own an action"): a mention that resolves to anything
    other than an `internal_person` returns `(None, None)`, like an
    unresolved one, so `merge_action` falls back to `_inherited_owner`."""
    if not owner_mention:
        return None, None
    resolution = registry.resolve(owner_mention)
    if resolution.status != "resolved" or not resolution.entity_id:
        return None, None
    owner_entity = registry.get(resolution.entity_id)
    if owner_entity is None or owner_entity.kind != EntityKind.INTERNAL_PERSON:
        return None, None
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id FROM registry_entities WHERE crm_id = %s",
            (owner_entity.crm_id or owner_entity.id,),
        )
        row = cur.fetchone()
    owner_dbid = row[0] if row else None
    owner_slug = ensure_entity_note(conn, registry, owner_entity.id, commit=commit).slug
    return owner_dbid, owner_slug


def _inherited_owner(
    conn: psycopg.Connection,
    registry: Registry,
    entity,
    *,
    commit: bool = True,
) -> tuple[int | None, str | None]:
    """`(owner_entity_dbid, owner_note_slug)` of the account's owner, when
    `entity` is an account or a contact with a parent account ("Actions
    inherit the owner", `01-spec/actions-and-owners-2026-09-19.md`).
    `(None, None)` for a product/competitor or an account with no owner.
    """
    if entity.kind == EntityKind.ACCOUNT:
        account_crm_id = entity.id
    elif entity.kind == EntityKind.CONTACT and entity.parent_id:
        account_crm_id = entity.parent_id
    else:
        return None, None

    with conn.cursor() as cur:
        cur.execute(
            "SELECT owner.crm_id, owner.note_slug FROM registry_entities account "
            "JOIN registry_entities owner ON owner.id = account.owner_entity_id "
            "WHERE account.crm_id = %s AND account.kind = 'account'",
            (account_crm_id,),
        )
        row = cur.fetchone()
    if row is None:
        return None, None
    owner_crm_id, owner_note_slug = row
    owner_entity = registry.get(owner_crm_id)
    if owner_entity is None:
        return None, None
    if owner_note_slug is None:
        owner_note_slug = ensure_entity_note(conn, registry, owner_crm_id, commit=commit).slug
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM registry_entities WHERE crm_id = %s", (owner_crm_id,))
        owner_row = cur.fetchone()
    return (owner_row[0] if owner_row else None), owner_note_slug


def merge_action(
    conn: psycopg.Connection,
    registry: Registry,
    action: Action,
    source: SourceRow,
    *,
    embedder: Embedder | None = None,
    entity_id: str | None = None,
    commit: bool = True,
) -> ActionOutcome:
    """Dedup `action` or insert it with its own action note.

    `commit=False` when the caller holds the source's writes in one open
    transaction (see `merge_fact`)."""
    entity_note: NoteRow | None = None
    entity = registry.get(entity_id) if entity_id else None
    if entity is not None and entity.kind in _NOTE_BEARING_KINDS:
        entity_note = ensure_entity_note(conn, registry, entity.id, commit=commit)

    duplicate = _duplicate_action_for_source(conn, source.slug, action.locator, action.what)
    if duplicate is None and entity_note is not None:
        duplicate = _find_duplicate(action.what, _open_actions_for_entity(conn, entity_note.slug))
    if duplicate is not None:
        action_id, note_slug = duplicate
        return ActionOutcome(status="duplicate", action_id=action_id, note_slug=note_slug)

    owner_dbid, owner_slug = _resolve_owner(conn, registry, action.owner_mention, commit=commit)
    if owner_dbid is None and entity is not None:
        # No vendor-side owner named: inherit the account's owner. The claim
        # text below still quotes only `action.owner_mention`.
        owner_dbid, owner_slug = _inherited_owner(conn, registry, entity, commit=commit)

    claim_text = action.what
    if action.owner_mention:
        claim_text = f"{action.what} (owner: {action.owner_mention})"
        if owner_slug:
            claim_text = link_mentions(claim_text, [(action.owner_mention, owner_slug)])

    action_id, slug = reserve_action_note_slug(conn, action.what)
    now = clock.now()
    note = NoteRow(
        slug=slug,
        type=NoteType.ACTION,
        title=action.what[:120],
        description=(
            f"Open action for [[{entity_note.slug}]]."
            if entity_note is not None
            else "Open action."
        ),
        status=NoteStatus.LIVING,
        created=now,
        updated=now,
        related=[entity_note.slug] if entity_note is not None else [],
    )
    # The note has to exist before `claims.note_slug` (a NOT NULL FK into
    # `notes`) can reference it, so: create the note first with no claims,
    # insert the real claims row, then re-render with it included.
    write_note(conn, note, [], changed_by="merge", reason="action created", commit=commit)

    claim = Claim(
        note_slug=slug,
        fact_type="other",
        claim_text=claim_text,
        source_slug=source.slug,
        locator=action.locator,
        recorded=recorded_dt(source),
        valid_from=action.due,
    )
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO claims
                (note_slug, fact_type, claim_text, source_slug, locator, recorded, valid_from)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                claim.note_slug,
                claim.fact_type,
                claim.claim_text,
                claim.source_slug,
                claim.locator,
                claim.recorded,
                claim.valid_from,
            ),
        )
        claim.id = cur.fetchone()[0]
    if commit:
        conn.commit()
    write_note(
        conn,
        note,
        [claim],
        changed_by="merge",
        reason="action citation",
        embedder=embedder,
        commit=commit,
    )

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO actions
                (id, note_slug, what, owner_entity_id, due, status, source_slug, locator)
            OVERRIDING SYSTEM VALUE
            VALUES (%s, %s, %s, %s, %s, 'open', %s, %s)
            """,
            (action_id, slug, action.what, owner_dbid, action.due, source.slug, action.locator),
        )
    if commit:
        conn.commit()

    notes_updated: list[str] = []
    if entity_note is not None and link_related(
        conn,
        entity_note.slug,
        slug,
        changed_by="merge",
        reason="new action",
        embedder=embedder,
        commit=commit,
    ):
        notes_updated.append(entity_note.slug)

    return ActionOutcome(
        status="new",
        action_id=action_id,
        note_slug=slug,
        notes_created=[slug],
        notes_updated=notes_updated,
    )
