"""Page-shaped read queries shared by the web routes (note summaries,
backlinks, version history, the registry picker, the review queue, actions).
"""

from __future__ import annotations

import re
from typing import Any

import psycopg

from portolan.util.redact import redact_dsn

__all__ = [
    "account_owner",
    "account_owners_by_crm_ids",
    "actions_feed",
    "backfill_fact_evidence",
    "backlinks",
    "claims_citing_source",
    "dead_letter_rows",
    "escape_ilike",
    "list_notes",
    "not_ingested_sources",
    "note_summaries",
    "note_type",
    "note_versions",
    "record_action_status_change",
    "registry_alias_conflict",
    "registry_entities_by_crm_ids",
    "registry_entities_search",
    "report_notes",
    "report_week",
    "review_queue_open_index",
    "review_queue_rows",
    "review_queue_rows_by_ids",
    "sources_brief",
]


def escape_ilike(text: str) -> str:
    """`text` with every ILIKE wildcard (`%`, `_`) and the escape
    character itself (`\\`) backslash-escaped, for use inside a `LIKE`/
    `ILIKE` pattern with an explicit `ESCAPE '\\'` clause, so a typed `%`
    or `_` matches literally. Callers add their own leading/trailing `%`.
    """
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def note_type(conn: psycopg.Connection, slug: str) -> str | None:
    """The `notes.type` for `slug`, or `None` if it doesn't exist."""
    with conn.cursor() as cur:
        cur.execute("SELECT type FROM notes WHERE slug = %s", (slug,))
        row = cur.fetchone()
    return row[0] if row else None


def note_summaries(conn: psycopg.Connection, slugs: list[str]) -> dict[str, dict[str, Any]]:
    """`{slug: {slug, title, type, updated, description}}` for a list of
    note slugs, in one round trip (every page shows the `updated:` date of
    any note it references).
    """
    if not slugs:
        return {}
    with conn.cursor() as cur:
        cur.execute(
            "SELECT slug, title, type, updated, description FROM notes WHERE slug = ANY(%s)",
            (list(dict.fromkeys(slugs)),),
        )
        rows = cur.fetchall()
    return {
        row[0]: {
            "slug": row[0],
            "title": row[1],
            "type": row[2],
            "updated": row[3],
            "description": row[4],
        }
        for row in rows
    }


_BACKLINK_KINDS = ("body", "parent", "supersession")


def backlinks(conn: psycopg.Connection, slug: str) -> list[dict[str, Any]]:
    """Notes that link TO `slug` (the inverse of `note.related`/body
    links) -- excludes `provenance` edges, whose `to_slug` is a source,
    not a note.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT l.from_slug, l.kind, n.title, n.type, n.updated
            FROM links l
            JOIN notes n ON n.slug = l.from_slug
            WHERE l.to_slug = %s AND l.kind = ANY(%s)
            ORDER BY n.updated DESC, l.from_slug
            """,
            (slug, list(_BACKLINK_KINDS)),
        )
        rows = cur.fetchall()
    return [
        {"slug": r[0], "kind": r[1], "title": r[2], "type": r[3], "updated": r[4]} for r in rows
    ]


def note_versions(conn: psycopg.Connection, slug: str) -> list[dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT version, changed_at, changed_by, reason
            FROM note_versions
            WHERE slug = %s
            ORDER BY version DESC
            """,
            (slug,),
        )
        rows = cur.fetchall()
    return [{"version": r[0], "changed_at": r[1], "changed_by": r[2], "reason": r[3]} for r in rows]


_ENTITY_KIND_LABELS: dict[str, str] = {
    "account": "Account",
    "contact": "Contact",
    "product": "Product",
    "competitor": "Competitor",
    "internal_person": "Internal person",
    "other": "Other",
}


def _entity_description(
    kind: str,
    account_kind: str | None,
    email: str | None,
    email_domain: str | None,
    parent_name: str | None,
) -> str:
    """A one-line picker description. `registry_entities` has no description
    column, so it is built from kind, account kind, parent name and email."""
    bits = [_ENTITY_KIND_LABELS.get(kind, kind.title())]
    if account_kind:
        bits.append(account_kind)
    if parent_name:
        bits.append(f"part of {parent_name}")
    if email:
        bits.append(email)
    elif email_domain:
        bits.append(email_domain)
    return " · ".join(bits)


def registry_entities_search(
    conn: psycopg.Connection, q: str, *, limit: int
) -> list[dict[str, Any]]:
    """Up to `limit` active registry entities, as `{crm_id, canonical_name,
    kind, description}`, whose canonical name contains `q`,
    case-insensitively, ordered by name; ordered by kind then name when `q`
    is blank. Backs `GET /review/registry?q=`, one small query per keystroke.
    """
    q = q.strip()
    like = f"%{escape_ilike(q)}%" if q else None
    with conn.cursor() as cur:
        if like is not None:
            cur.execute(
                """
                SELECT e.crm_id, e.canonical_name, e.kind, e.account_kind, e.email,
                       e.email_domain, p.canonical_name
                FROM registry_entities e
                LEFT JOIN registry_entities p ON p.id = e.parent_id
                WHERE e.crm_id IS NOT NULL AND e.active
                  AND e.canonical_name ILIKE %s ESCAPE '\\'
                ORDER BY e.canonical_name
                LIMIT %s
                """,
                (like, limit),
            )
        else:
            cur.execute(
                """
                SELECT e.crm_id, e.canonical_name, e.kind, e.account_kind, e.email,
                       e.email_domain, p.canonical_name
                FROM registry_entities e
                LEFT JOIN registry_entities p ON p.id = e.parent_id
                WHERE e.crm_id IS NOT NULL AND e.active
                ORDER BY e.kind, e.canonical_name
                LIMIT %s
                """,
                (limit,),
            )
        rows = cur.fetchall()
    return [
        {
            "crm_id": r[0],
            "canonical_name": r[1],
            "kind": r[2],
            "description": _entity_description(r[2], r[3], r[4], r[5], r[6]),
        }
        for r in rows
    ]


def registry_entities_by_crm_ids(
    conn: psycopg.Connection, crm_ids: list[str]
) -> dict[str, dict[str, Any]]:
    """`{crm_id: {canonical_name, kind}}` for exactly the given ids. Not
    filtered to `active`, so a candidate that was since deactivated still
    shows its name rather than a bare id."""
    if not crm_ids:
        return {}
    with conn.cursor() as cur:
        cur.execute(
            "SELECT crm_id, canonical_name, kind FROM registry_entities WHERE crm_id = ANY(%s)",
            (crm_ids,),
        )
        rows = cur.fetchall()
    return {r[0]: {"canonical_name": r[1], "kind": r[2]} for r in rows}


def registry_alias_conflict(conn: psycopg.Connection, mention: str) -> dict[str, Any] | None:
    """The registry entity `mention` (case-insensitively) already maps to
    through `registry_aliases`, if any -- `{crm_id, canonical_name}`, or
    `None` when this mention has never been aliased.

    `registry_aliases`' primary key `(alias, entity_id)` does not stop a
    second entity for the same alias text, which would make the name
    ambiguous (spec sec 3: "a human maps a name once; it sticks").
    """
    mention = mention.strip()
    if not mention:
        return None
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT e.crm_id, e.canonical_name
            FROM registry_aliases a
            JOIN registry_entities e ON e.id = a.entity_id
            WHERE lower(a.alias) = lower(%s)
            ORDER BY a.created
            LIMIT 1
            """,
            (mention,),
        )
        row = cur.fetchone()
    return {"crm_id": row[0], "canonical_name": row[1]} if row else None


def sources_brief(conn: psycopg.Connection, slugs: list[str]) -> dict[str, dict[str, Any]]:
    """`{slug: {source_type, source_date, participants, summary,
    account_name}}` for a review card's source label (`sources` has no
    `title` column).

    `account_name`: an active account whose `email_domain` matches one of
    the participants' email domains, or `None`. Used only to name the
    source, never to resolve anything; with several matches, which one wins
    is not defined.
    """
    if not slugs:
        return {}
    with conn.cursor() as cur:
        cur.execute(
            "SELECT slug, source_type, source_date, participants, summary "
            "FROM sources WHERE slug = ANY(%s)",
            (slugs,),
        )
        rows = cur.fetchall()

    domains: set[str] = set()
    by_slug: dict[str, dict[str, Any]] = {}
    for slug, source_type, source_date, participants, summary in rows:
        emails = [p.get("email") for p in (participants or []) if p.get("email")]
        source_domains = {e.rsplit("@", 1)[-1].lower() for e in emails if "@" in e}
        domains |= source_domains
        by_slug[slug] = {
            "source_type": source_type,
            "source_date": source_date,
            "participants": participants or [],
            "summary": summary,
            "_domains": source_domains,
            "account_name": None,
        }

    if domains:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT email_domain, canonical_name FROM registry_entities "
                "WHERE kind = 'account' AND active AND email_domain = ANY(%s) "
                "ORDER BY canonical_name",
                (list(domains),),
            )
            name_by_domain = dict(cur.fetchall())
        for brief in by_slug.values():
            for domain in brief.pop("_domains"):
                if domain in name_by_domain:
                    brief["account_name"] = name_by_domain[domain]
                    break
    else:
        for brief in by_slug.values():
            brief.pop("_domains", None)

    return by_slug


def backfill_fact_evidence(
    conn: psycopg.Connection, *, source_slug: str, mention: str | None, locator: str | None
) -> dict[str, Any] | None:
    """Best-effort recovery of a fact's claim/quote/fact type/locator for an
    `unknown_entity`/`ambiguous_entity` review row whose payload has no
    `claim` saved, from a `claims` row on the same source (the fact may
    have been merged under another of its mentions).

    Two tiers, first hit wins, lowest claim id within a tier:
    1. `source_slug` + `locator`, when the row carries a locator.
    2. `source_slug` + `mention` as a whole word (`~*` with `\\y`) in
       `claim_text` or `quote`: a text match, not an identity match, so on
       a source naming the same entity in two claims it may pick the wrong
       one.

    `None` when neither tier finds anything.
    """
    with conn.cursor() as cur:
        if locator:
            cur.execute(
                "SELECT claim_text, quote, fact_type, locator FROM claims "
                "WHERE source_slug = %s AND locator = %s ORDER BY id LIMIT 1",
                (source_slug, locator),
            )
            row = cur.fetchone()
            if row is not None:
                return {"claim": row[0], "quote": row[1], "fact_type": row[2], "locator": row[3]}
        if mention and mention.strip():
            word = re.escape(mention.strip())
            pattern = rf"\y{word}\y"
            cur.execute(
                "SELECT claim_text, quote, fact_type, locator FROM claims "
                "WHERE source_slug = %s AND (claim_text ~* %s OR quote ~* %s) "
                "ORDER BY id LIMIT 1",
                (source_slug, pattern, pattern),
            )
            row = cur.fetchone()
            if row is not None:
                return {"claim": row[0], "quote": row[1], "fact_type": row[2], "locator": row[3]}
    return None


def review_queue_rows(conn: psycopg.Connection, *, status: str = "open") -> list[dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, kind, payload, source_slug, status, created, resolved_at, resolved_by
            FROM review_queue
            WHERE status = %s
            ORDER BY created
            """,
            (status,),
        )
        rows = cur.fetchall()
    return [
        {
            "id": r[0],
            "kind": r[1],
            "payload": r[2],
            "source_slug": r[3],
            "status": r[4],
            "created": r[5],
            "resolved_at": r[6],
            "resolved_by": r[7],
        }
        for r in rows
    ]


def review_queue_open_index(
    conn: psycopg.Connection, *, status: str = "open"
) -> list[dict[str, Any]]:
    """`[{id, kind, reason, created}]` for every `status` row, oldest first:
    four small columns (never the JSONB `payload`), cheap enough to read for
    the whole queue, so `/review` can order, count and filter across all of
    it while loading only one page in full. `reason` is `payload ->>
    'reason'`, which categorises a `kind='other'` row.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, kind, payload ->> 'reason', created FROM review_queue "
            "WHERE status = %s ORDER BY created",
            (status,),
        )
        rows = cur.fetchall()
    return [{"id": r[0], "kind": r[1], "reason": r[2], "created": r[3]} for r in rows]


def review_queue_rows_by_ids(conn: psycopg.Connection, ids: list[int]) -> list[dict[str, Any]]:
    """Full `review_queue_rows`-shaped rows for exactly `ids`, in the order
    given (one `/review` page)."""
    if not ids:
        return []
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, kind, payload, source_slug, status, created, resolved_at, resolved_by "
            "FROM review_queue WHERE id = ANY(%s)",
            (ids,),
        )
        rows = cur.fetchall()
    by_id = {
        r[0]: {
            "id": r[0],
            "kind": r[1],
            "payload": r[2],
            "source_slug": r[3],
            "status": r[4],
            "created": r[5],
            "resolved_at": r[6],
            "resolved_by": r[7],
        }
        for r in rows
    }
    return [by_id[i] for i in ids if i in by_id]


def not_ingested_sources(conn: psycopg.Connection) -> list[dict[str, Any]]:
    """Sources with status `failed`/`partial` -- the coverage page's
    "not ingested" list (portolan-spec-v2.md sec 6.2). `last_error` is raw
    pipeline exception text, so it goes through `redact_dsn`.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT slug, source_type, source_date, extraction_status, attempts, last_error
            FROM sources
            WHERE extraction_status IN ('failed', 'partial')
            ORDER BY source_date DESC NULLS LAST, slug
            """
        )
        rows = cur.fetchall()
    return [
        {
            "slug": r[0],
            "source_type": r[1],
            "source_date": r[2],
            "extraction_status": r[3],
            "attempts": r[4],
            "last_error": redact_dsn(r[5]) if r[5] else r[5],
        }
        for r in rows
    ]


def dead_letter_rows(conn: psycopg.Connection) -> list[dict[str, Any]]:
    """Pending (`redriven_at IS NULL`) dead-letter rows, or `[]` when the
    `dead_letter` table does not exist.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT to_regclass('public.dead_letter')",
        )
        (exists,) = cur.fetchone()
    if not exists:
        return []
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, source_id, stage, attempt, error, created
            FROM dead_letter
            WHERE redriven_at IS NULL
            ORDER BY created DESC
            """
        )
        rows = cur.fetchall()
    return [
        {
            "id": r[0],
            "source_id": r[1],
            "stage": r[2],
            "attempt": r[3],
            # Raw pipeline exception text, redacted as in `not_ingested_sources`.
            "error": redact_dsn(r[4]) if r[4] else r[4],
            "created": r[5],
        }
        for r in rows
    ]


def report_notes(conn: psycopg.Connection) -> list[dict[str, Any]]:
    """Report notes, newest report week first (not newest `updated`, which
    a regenerated older week would win). ISO week strings (`2026-W36`) sort
    correctly as text; a note with no `reports` row falls to the end."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT n.slug, n.title, n.description, n.updated
            FROM notes n
            LEFT JOIN reports r ON r.note_slug = n.slug
            WHERE n.type = 'report'
            ORDER BY r.week DESC NULLS LAST, n.updated DESC
            """
        )
        rows = cur.fetchall()
    return [{"slug": r[0], "title": r[1], "description": r[2], "updated": r[3]} for r in rows]


def report_week(conn: psycopg.Connection, note_slug: str) -> str | None:
    """The ISO week string (`"2026-W37"`) in `reports.week` for this report
    note, or `None` if it has no `reports` row. Notes don't carry the
    covered window; `portolan.report.util.week_bounds(week)` derives it."""
    with conn.cursor() as cur:
        cur.execute("SELECT week FROM reports WHERE note_slug = %s", (note_slug,))
        row = cur.fetchone()
    return row[0] if row else None


NOTES_INDEX_PAGE_SIZE = 50


def list_notes(
    conn: psycopg.Connection,
    *,
    type_filter: str | None = None,
    q: str | None = None,
    page: int = 1,
    page_size: int = NOTES_INDEX_PAGE_SIZE,
) -> tuple[list[dict[str, Any]], int]:
    """`(rows, total_count)` for the `/notes` index: a type filter and a
    title/slug substring match (`q`'s `%` and `_` match literally),
    newest-updated first, `page_size` per page. A directory listing, not the
    fused search.
    """
    where = []
    params: list[Any] = []
    if type_filter:
        where.append("type = %s")
        params.append(type_filter)
    if q:
        where.append("(title ILIKE %s ESCAPE '\\' OR slug ILIKE %s ESCAPE '\\')")
        like = f"%{escape_ilike(q)}%"
        params.extend([like, like])
    clause = f"WHERE {' AND '.join(where)}" if where else ""

    with conn.cursor() as cur:
        cur.execute(f"SELECT count(*) FROM notes {clause}", params)  # noqa: S608 - clause is a fixed allowlist, params bound
        (total,) = cur.fetchone()

        offset = max(page - 1, 0) * page_size
        cur.execute(
            f"""
            SELECT slug, title, type, status, updated
            FROM notes
            {clause}
            ORDER BY updated DESC, slug
            LIMIT %s OFFSET %s
            """,  # noqa: S608
            (*params, page_size, offset),
        )
        rows = cur.fetchall()

    return (
        [
            {"slug": r[0], "title": r[1], "type": r[2], "status": r[3], "updated": r[4]}
            for r in rows
        ],
        total,
    )


def claims_citing_source(conn: psycopg.Connection, source_slug: str) -> list[dict[str, Any]]:
    """Every claim that cites `source_slug`, as its primary citation
    (`claims.source_slug`/`locator`) or a corroborating one
    (`claim_citations`), oldest recorded first."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT c.note_slug, c.claim_text, c.locator, c.fact_type, c.recorded
            FROM claims c
            WHERE c.source_slug = %s
            UNION ALL
            SELECT c.note_slug, c.claim_text, cc.locator, c.fact_type, cc.recorded
            FROM claim_citations cc
            JOIN claims c ON c.id = cc.claim_id
            WHERE cc.source_slug = %s
            ORDER BY recorded
            """,
            (source_slug, source_slug),
        )
        rows = cur.fetchall()
    return [
        {
            "note_slug": r[0],
            "claim_text": r[1],
            "locator": r[2],
            "fact_type": r[3],
            "recorded": r[4],
        }
        for r in rows
    ]


# ---------------------------------------------------------------------
# Owners and the /actions page (01-spec/actions-and-owners-2026-09-19.md,
# Phase 1).
# ---------------------------------------------------------------------


def account_owner(conn: psycopg.Connection, account_crm_id: str) -> dict[str, Any] | None:
    """`{crm_id, name, note_slug}` for `account_crm_id`'s registry owner, or
    `None` when unset."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT owner.crm_id, owner.canonical_name, owner.note_slug "
            "FROM registry_entities account "
            "JOIN registry_entities owner ON owner.id = account.owner_entity_id "
            "WHERE account.crm_id = %s AND account.kind = 'account'",
            (account_crm_id,),
        )
        row = cur.fetchone()
    return {"crm_id": row[0], "name": row[1], "note_slug": row[2]} if row else None


def account_owners_by_crm_ids(
    conn: psycopg.Connection, account_crm_ids: list[str]
) -> dict[str, dict[str, Any]]:
    """`account_owner`, batched -- `{account_crm_id: {crm_id, name,
    note_slug}}`, one round trip for a list page (e.g. `/notes?type=account`)
    rather than one query per row."""
    if not account_crm_ids:
        return {}
    with conn.cursor() as cur:
        cur.execute(
            "SELECT account.crm_id, owner.crm_id, owner.canonical_name, owner.note_slug "
            "FROM registry_entities account "
            "JOIN registry_entities owner ON owner.id = account.owner_entity_id "
            "WHERE account.crm_id = ANY(%s) AND account.kind = 'account'",
            (account_crm_ids,),
        )
        rows = cur.fetchall()
    return {r[0]: {"crm_id": r[1], "name": r[2], "note_slug": r[3]} for r in rows}


def _accounts_for_action_notes(
    conn: psycopg.Connection, action_note_slugs: list[str]
) -> dict[str, dict[str, Any]]:
    """`{action_note_slug: {slug, title, crm_id}}`: the account each action
    belongs to -- the `account` note that body-links to it, or a linking
    `contact`'s parent account. An account link wins over a contact link.
    An action with neither (or whose contact has no parent account note)
    has no entry. Two queries in total.
    """
    if not action_note_slugs:
        return {}
    with conn.cursor() as cur:
        cur.execute(
            "SELECT DISTINCT ON (l.to_slug) l.to_slug, n.type, n.slug, n.title, n.crm_id "
            "FROM links l JOIN notes n ON n.slug = l.from_slug "
            "WHERE l.to_slug = ANY(%s) AND l.kind = 'body' "
            "AND n.type IN ('account', 'contact') "
            "ORDER BY l.to_slug, (n.type = 'account') DESC, n.slug",
            (action_note_slugs,),
        )
        linked = cur.fetchall()
    out: dict[str, dict[str, Any]] = {}
    contact_crm_ids: dict[str, list[str]] = {}
    for action_slug, note_type, slug, title, crm_id in linked:
        if note_type == "account":
            out[action_slug] = {"slug": slug, "title": title, "crm_id": crm_id}
        elif crm_id:
            contact_crm_ids.setdefault(crm_id, []).append(action_slug)
    if contact_crm_ids:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT contact.crm_id, parent.note_slug, parent.canonical_name, parent.crm_id "
                "FROM registry_entities contact "
                "JOIN registry_entities parent ON parent.id = contact.parent_id "
                "WHERE contact.crm_id = ANY(%s)",
                (list(contact_crm_ids),),
            )
            for contact_crm_id, parent_slug, parent_name, parent_crm_id in cur.fetchall():
                if parent_slug is None:
                    continue
                for action_slug in contact_crm_ids[contact_crm_id]:
                    out[action_slug] = {
                        "slug": parent_slug,
                        "title": parent_name,
                        "crm_id": parent_crm_id,
                    }
    return out


def _action_why_summaries(conn: psycopg.Connection, action_ids: list[int]) -> dict[int, str]:
    """`{action_id: why_summary}` in one query: the first inference sentence
    of the action's `action_why` row, else its first evidence text. An
    action with no `action_why` row has no entry."""
    if not action_ids:
        return {}
    with conn.cursor() as cur:
        cur.execute(
            "SELECT action_id, inference, evidence FROM action_why WHERE action_id = ANY(%s)",
            (action_ids,),
        )
        rows = cur.fetchall()
    summaries: dict[int, str] = {}
    for action_id, inference, evidence in rows:
        sentence = (inference or [{}])[0].get("sentence") if inference else None
        if sentence:
            summaries[action_id] = sentence
            continue
        text = (evidence or [{}])[0].get("text") if evidence else None
        if text:
            summaries[action_id] = text
    return summaries


def actions_feed(conn: psycopg.Connection) -> list[dict[str, Any]]:
    """Every action, with owner, account and "why" resolved in a fixed
    number of queries (not paginated). Each row: `{id, note_slug, what,
    due, status, source_slug, owner_crm_id, owner_name, owner_note_slug,
    account_slug, account_title, account_crm_id, closed_at, why_summary}`.
    `closed_at` is the latest `edit_records` change to done/dropped, else
    `None`. Overdue is not computed here: it depends on "now", which the
    caller supplies.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT a.id, a.note_slug, a.what, a.due, a.status, a.source_slug,
                   owner.crm_id, owner.canonical_name, owner.note_slug, er.edited_at
            FROM actions a
            LEFT JOIN registry_entities owner ON owner.id = a.owner_entity_id
            LEFT JOIN LATERAL (
                SELECT edited_at FROM edit_records
                WHERE note_slug = a.note_slug AND after_text IN ('done', 'dropped')
                ORDER BY edited_at DESC LIMIT 1
            ) er ON true
            ORDER BY a.id
            """
        )
        rows = cur.fetchall()

    why_summaries = _action_why_summaries(conn, [r[0] for r in rows])
    accounts = _accounts_for_action_notes(conn, [r[1] for r in rows])

    out: list[dict[str, Any]] = []
    for (
        action_id,
        note_slug,
        what,
        due,
        status,
        source_slug,
        owner_crm_id,
        owner_name,
        owner_note_slug,
        closed_at,
    ) in rows:
        account = accounts.get(note_slug)
        out.append(
            {
                "id": action_id,
                "note_slug": note_slug,
                "what": what,
                "due": due,
                "status": status,
                "source_slug": source_slug,
                "owner_crm_id": owner_crm_id,
                "owner_name": owner_name,
                "owner_note_slug": owner_note_slug,
                "account_slug": account["slug"] if account else None,
                "account_title": account["title"] if account else None,
                "account_crm_id": account["crm_id"] if account else None,
                "closed_at": closed_at,
                "why_summary": why_summaries.get(action_id),
            }
        )
    return out


def record_action_status_change(
    conn: psycopg.Connection,
    *,
    action_id: int,
    note_slug: str,
    from_status: str,
    to_status: str,
    editor: str,
    reason: str,
) -> None:
    """Set `actions.status` and write an `edit_records` row (who, when, from
    what to what); a human edit is itself a source (spec sec 4)."""
    with conn.cursor() as cur:
        # Conditional on the status the caller read, so a second click that
        # read the same status changes nothing and writes no second record.
        cur.execute(
            "UPDATE actions SET status = %s WHERE id = %s AND status = %s",
            (to_status, action_id, from_status),
        )
        if cur.rowcount != 1:
            conn.commit()
            return
        cur.execute(
            "INSERT INTO edit_records (note_slug, editor, before_text, after_text, reason) "
            "VALUES (%s, %s, %s, %s, %s)",
            (note_slug, editor, from_status, to_status, reason),
        )
    conn.commit()
