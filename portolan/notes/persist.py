"""`write_note`, `read_note`, `ensure_entity_note`, `append_changelog`, and
`link_related` (`build-contracts.md` "Notes and claims").

`write_note` is the one place a note's `body_md` gets computed
(`render_note`) and persisted: it versions the previous body into
`note_versions`, upserts `notes`, rewrites that note's `links` rows from
the freshly rendered body, and (when given an embedder) reindexes it.
Every other function in this module funnels through it rather than writing
`notes`/`links` directly, so those two invariants -- "every body change is
versioned" and "links always match the current body" -- never drift.
"""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING

import psycopg
from psycopg.types.json import Jsonb

from portolan.notes.models import (
    STALE_AFTER_DAYS,
    Claim,
    ClaimCitation,
    NoteRow,
    NoteStatus,
    NoteType,
)
from portolan.notes.render import (
    CITATION_RE,
    WIKILINK_RE,
    parse_related,
    parse_synthesis,
    render_note,
    slugify,
)
from portolan.util import clock

if TYPE_CHECKING:
    from portolan.registry import Registry
    from portolan.search.embed import Embedder

__all__ = [
    "append_changelog",
    "ensure_entity_note",
    "link_related",
    "note_exists",
    "read_note",
    "write_note",
]

_ENTITY_KIND_TO_NOTE_TYPE = {
    "account": NoteType.ACCOUNT,
    "contact": NoteType.CONTACT,
    "product": NoteType.PRODUCT,
    "competitor": NoteType.COMPETITOR,
    # Owners need a note to link to (01-spec/actions-and-owners-2026-09-19.md),
    # and the closed `NoteType` list has no person type for vendor staff.
    "internal_person": NoteType.CONTACT,
}


def note_exists(conn: psycopg.Connection, slug: str) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM notes WHERE slug = %s", (slug,))
        return cur.fetchone() is not None


def _row_to_note(row: tuple) -> NoteRow:
    (
        slug,
        type_,
        title,
        description,
        status,
        created,
        updated,
        stale_after,
        schema_version,
        crm_id,
        parent_slug,
        superseded_by,
        body_md,
        version,
    ) = row
    return NoteRow(
        slug=slug,
        type=NoteType(type_),
        title=title,
        description=description,
        status=NoteStatus(status),
        created=created,
        updated=updated,
        stale_after=stale_after,
        schema_version=schema_version,
        crm_id=crm_id,
        parent_slug=parent_slug,
        superseded_by=superseded_by,
        body_md=body_md or "",
        version=version,
        synthesis=parse_synthesis(body_md or ""),
        related=parse_related(body_md or ""),
    )


_NOTE_COLUMNS = (
    "slug, type, title, description, status, created, updated, stale_after, "
    "schema_version, crm_id, parent_slug, superseded_by, body_md, version"
)


def read_note(conn: psycopg.Connection, slug: str) -> tuple[NoteRow, list[Claim]]:
    """The note plus its unarchived claims, oldest-recorded first.

    Raises `ValueError` if no such note exists -- callers that want
    create-or-fetch semantics should use `ensure_entity_note` (for entity
    notes) or check `note_exists` first.
    """
    with conn.cursor() as cur:
        cur.execute(f"SELECT {_NOTE_COLUMNS} FROM notes WHERE slug = %s", (slug,))
        row = cur.fetchone()
        if row is None:
            raise ValueError(f"no such note: {slug!r}")
        note = _row_to_note(row)

        cur.execute(
            """
            SELECT id, note_slug, fact_type, claim_text, source_slug, locator, quote,
                   speaker, valid_from, valid_to, recorded, superseded_at, severity,
                   sentiment, confidence, superseded_by, company_context_updated_at
            FROM claims
            WHERE note_slug = %s AND archived = false
            ORDER BY recorded, id
            """,
            (slug,),
        )
        claim_rows = cur.fetchall()
        citations_by_claim = _citations_for_claims(cur, [r[0] for r in claim_rows])
        claims = [
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
                company_context_updated_at=r[16],
                citations=citations_by_claim.get(r[0], []),
            )
            for r in claim_rows
        ]
    return note, claims


def claims_by_ids(conn: psycopg.Connection, claim_ids: list[int]) -> list[Claim]:
    """The `Claim` rows for `claim_ids`, in the order given (evidence is
    cited in a deliberate order).

    Unlike `read_note`, not filtered by note or `archived`: cited evidence
    usually lives on another note, and the caller wants its current state,
    `superseded_at` included. An id with no row is skipped, not raised.
    """
    if not claim_ids:
        return []
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, note_slug, fact_type, claim_text, source_slug, locator, quote,
                   speaker, valid_from, valid_to, recorded, superseded_at, severity,
                   sentiment, confidence, superseded_by, company_context_updated_at
            FROM claims
            WHERE id = ANY(%s)
            """,
            (claim_ids,),
        )
        rows = cur.fetchall()
        citations_by_claim = _citations_for_claims(cur, [r[0] for r in rows])
    by_id = {
        r[0]: Claim(
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
            company_context_updated_at=r[16],
            citations=citations_by_claim.get(r[0], []),
        )
        for r in rows
    }
    return [by_id[cid] for cid in claim_ids if cid in by_id]


def _citations_for_claims(
    cur: psycopg.Cursor, claim_ids: list[int]
) -> dict[int, list[ClaimCitation]]:
    """`{claim_id: [ClaimCitation, ...]}`, oldest first: the extra citations
    `same` merge verdicts added beyond each claim's own `source_slug`/`locator`."""
    if not claim_ids:
        return {}
    cur.execute(
        """
        SELECT id, claim_id, source_slug, locator, quote, recorded
        FROM claim_citations
        WHERE claim_id = ANY(%s)
        ORDER BY claim_id, recorded, id
        """,
        (claim_ids,),
    )
    out: dict[int, list[ClaimCitation]] = {}
    for cid, claim_id, source_slug, locator, quote, recorded in cur.fetchall():
        out.setdefault(claim_id, []).append(
            ClaimCitation(
                id=cid, source_slug=source_slug, locator=locator, quote=quote, recorded=recorded
            )
        )
    return out


def _extract_link_targets(note: NoteRow, body_md: str) -> list[tuple[str, str]]:
    """`(to_slug, kind)` pairs to write to `links` for `note.slug`: every
    `[[slug]]` in the body as `body` (minus the note itself and its parent),
    the parent as `parent`, `superseded_by` as `supersession`, and one
    `provenance` row per source cited in the body."""
    body_targets = {m.group(1) for m in WIKILINK_RE.finditer(body_md)}
    body_targets.discard(note.slug)
    if note.parent_slug:
        body_targets.discard(note.parent_slug)

    pairs = [(slug, "body") for slug in sorted(body_targets)]
    if note.parent_slug:
        pairs.append((note.parent_slug, "parent"))
    if note.superseded_by:
        pairs.append((note.superseded_by, "supersession"))

    source_slugs = sorted({m.group(1).strip() for m in CITATION_RE.finditer(body_md)})
    pairs.extend((slug, "provenance") for slug in source_slugs)
    return pairs


def write_note(
    conn: psycopg.Connection,
    note: NoteRow,
    claims: list[Claim],
    *,
    changed_by: str,
    reason: str,
    embedder: Embedder | None = None,
    commit: bool = True,
    preserve_stale_after: bool = False,
) -> None:
    """Render, persist, version, relink, and (optionally) reindex one note.

    `note.updated` is stamped to now before rendering, so the frontmatter's
    `updated:` matches the stored row.

    `stale_after` is recomputed from `updated` on every write, per type
    (`STALE_AFTER_DAYS`); `preserve_stale_after=True` keeps a value already
    set on `note`.

    `commit=False` only from inside a caller's open `conn.transaction()`:
    psycopg raises `ProgrammingError` on `conn.commit()` there. Reindexing
    opens its own nested transaction (a savepoint), so it is safe either way.
    """
    note.updated = clock.now()
    if not preserve_stale_after or note.stale_after is None:
        note.stale_after = note.updated + timedelta(days=STALE_AFTER_DAYS.get(note.type, 180))
    body_md = render_note(note, claims)
    note.body_md = body_md

    with conn.cursor() as cur:
        cur.execute("SELECT body_md, version FROM notes WHERE slug = %s FOR UPDATE", (note.slug,))
        existing = cur.fetchone()

        if existing is None:
            note.version = 1
            cur.execute(
                f"""
                INSERT INTO notes ({_NOTE_COLUMNS})
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    note.slug,
                    note.type.value,
                    note.title,
                    note.description,
                    note.status.value,
                    note.created,
                    note.updated,
                    note.stale_after,
                    note.schema_version,
                    note.crm_id,
                    note.parent_slug,
                    note.superseded_by,
                    note.body_md,
                    note.version,
                ),
            )
        else:
            old_body, old_version = existing
            cur.execute(
                """
                INSERT INTO note_versions (slug, version, body_md, changed_by, reason)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (note.slug, old_version, old_body, changed_by, reason),
            )
            note.version = old_version + 1
            cur.execute(
                """
                UPDATE notes SET
                    type = %s, title = %s, description = %s, status = %s,
                    updated = %s, stale_after = %s, schema_version = %s,
                    crm_id = %s, parent_slug = %s, superseded_by = %s,
                    body_md = %s, version = %s
                WHERE slug = %s
                """,
                (
                    note.type.value,
                    note.title,
                    note.description,
                    note.status.value,
                    note.updated,
                    note.stale_after,
                    note.schema_version,
                    note.crm_id,
                    note.parent_slug,
                    note.superseded_by,
                    note.body_md,
                    note.version,
                    note.slug,
                ),
            )

        cur.execute("DELETE FROM links WHERE from_slug = %s", (note.slug,))
        for to_slug, kind in _extract_link_targets(note, note.body_md):
            cur.execute(
                """
                INSERT INTO links (from_slug, to_slug, kind) VALUES (%s, %s, %s)
                ON CONFLICT (from_slug, to_slug, kind) DO NOTHING
                """,
                (note.slug, to_slug, kind),
            )
    if commit:
        conn.commit()

    if embedder is not None:
        from portolan.search.index import index_note

        index_note(conn, embedder, note.slug)


def link_related(
    conn: psycopg.Connection,
    hub_slug: str,
    target_slug: str,
    *,
    changed_by: str,
    reason: str,
    embedder: Embedder | None = None,
    commit: bool = True,
) -> bool:
    """Ensure `hub_slug` carries a `## Related` link to `target_slug`
    (`render.py` module docstring on why). A no-op (returns `False`) if the
    link already exists or `target_slug == hub_slug`; `True` when it wrote a
    new version of `hub_slug`. `commit` as in `write_note`."""
    if target_slug == hub_slug:
        return False
    note, claims = read_note(conn, hub_slug)
    if target_slug in note.related:
        return False
    note.related = [*note.related, target_slug]
    write_note(
        conn,
        note,
        claims,
        changed_by=changed_by,
        reason=reason,
        embedder=embedder,
        commit=commit,
    )
    return True


def _entity_note_slug_for(conn: psycopg.Connection, dbid: int) -> str | None:
    with conn.cursor() as cur:
        cur.execute("SELECT note_slug FROM registry_entities WHERE id = %s", (dbid,))
        row = cur.fetchone()
    return row[0] if row else None


def _unique_note_slug(conn: psycopg.Connection, base_slug: str, crm_id: str | None) -> str:
    """`base_slug`, or `base_slug-2`, `-3`, ... if a note with that slug
    already exists and belongs to a different entity (two CRM ids whose
    names slugify the same)."""
    candidate = base_slug
    n = 2
    while True:
        with conn.cursor() as cur:
            cur.execute("SELECT crm_id FROM notes WHERE slug = %s", (candidate,))
            row = cur.fetchone()
        if row is None or row[0] == crm_id:
            return candidate
        candidate = f"{base_slug}-{n}"
        n += 1


def _entity_description(entity, parent_slug: str | None) -> str:
    if entity.kind.value == "account":
        kind_txt = (entity.account_kind or "account").replace("_", " ")
        if parent_slug:
            return f"A {kind_txt} account, part of [[{parent_slug}]]."
        return f"A {kind_txt} account."
    if entity.kind.value == "contact":
        return f"A contact tied to [[{parent_slug}]]." if parent_slug else "A contact."
    if entity.kind.value == "internal_person":
        return "A Halvard team member."
    if entity.kind.value == "competitor":
        return "A named competitor."
    return "A product Halvard sells."


def ensure_entity_note(
    conn: psycopg.Connection, registry: Registry, entity_id: str, *, commit: bool = True
) -> NoteRow:
    """Create (idempotently) the account/contact/product/competitor/
    internal-person note for `entity_id`, or return the existing one.

    For an account or contact with a parent, the parent chain is created
    first (recursively), so `[[parent]]` always resolves. The slug is cached
    on `registry_entities.note_slug`, so a second call is a plain lookup.
    `commit` as in `write_note`.
    """
    entity = registry.get(entity_id)
    if entity is None:
        raise ValueError(f"no registry entity with id {entity_id!r}")
    if entity.kind.value not in _ENTITY_KIND_TO_NOTE_TYPE:
        raise ValueError(
            f"entity kind {entity.kind.value!r} has no note type "
            "(only account/contact/product/competitor/internal_person do)"
        )

    with conn.cursor() as cur:
        cur.execute("SELECT id, note_slug FROM registry_entities WHERE crm_id = %s", (entity_id,))
        row = cur.fetchone()
    if row is None:
        raise ValueError(f"registry entity {entity_id!r} has no registry_entities row")
    dbid, cached_slug = row

    if cached_slug:
        note, _claims = read_note(conn, cached_slug)
        return note

    # A contact's parent is its account. `notes.parent_slug` stays
    # account-only (`notes_parent_only_on_account` CHECK), so a contact links
    # to its account through its description and the account's `## Related`.
    parent_slug: str | None = None
    parent_note: NoteRow | None = None
    if entity.kind.value in ("account", "contact") and entity.parent_id:
        parent_note = ensure_entity_note(conn, registry, entity.parent_id, commit=commit)
        parent_slug = parent_note.slug

    base_slug = slugify(entity.canonical_name)
    slug = _unique_note_slug(conn, base_slug, entity.crm_id or entity.id)

    now = clock.now()
    note = NoteRow(
        slug=slug,
        type=_ENTITY_KIND_TO_NOTE_TYPE[entity.kind.value],
        title=entity.canonical_name,
        description=_entity_description(entity, parent_slug),
        status=NoteStatus.LIVING,
        created=now,
        updated=now,
        crm_id=entity.crm_id or entity.id,
        parent_slug=parent_slug if entity.kind.value == "account" else None,
    )
    if entity.kind.value in ("contact", "internal_person"):
        # Set explicitly so a vendor rep never renders the CONTACT default
        # synthesis ("X is a contact on file").
        note.synthesis = _entity_description(entity, parent_slug)

    write_note(
        conn,
        note,
        [],
        changed_by="merge",
        reason=f"entity note created from registry ({entity.kind.value} {entity_id})",
        commit=commit,
    )

    with conn.cursor() as cur:
        cur.execute("UPDATE registry_entities SET note_slug = %s WHERE id = %s", (slug, dbid))
    if commit:
        conn.commit()

    # The account's `## Related` list gives the contact an inbound edge, so
    # it is not a lint orphan.
    if entity.kind.value == "contact" and parent_note is not None:
        link_related(
            conn,
            parent_note.slug,
            slug,
            changed_by="merge",
            reason=f"contact {entity_id} added to account",
            commit=commit,
        )

    return note


def append_changelog(
    conn: psycopg.Connection,
    kind: str,
    summary: str,
    details: dict | None = None,
    *,
    commit: bool = True,
) -> None:
    """One append-only row in `changelog` (spec sec 5.1 write step, sec 7).
    `commit` as in `write_note`."""
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO changelog (kind, summary, details) VALUES (%s, %s, %s)",
            (kind, summary, Jsonb(details or {})),
        )
    if commit:
        conn.commit()
