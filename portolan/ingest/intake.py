"""`intake`: the first pipeline step (spec sec 5.1).

"Source note created with status `pending`; original stored; ... locators
injected" — this module does the first two; `prepare_text` (`prepare.py`)
does the locator injection.

Idempotency is keyed on `(source_id, content_hash)` together, enforced by
the `sources_source_id_content_hash_uq` unique index through `INSERT ... ON
CONFLICT DO NOTHING`. The same source with unchanged bytes is a no-op; two
different sources with identical bytes (one alert text filed as two
tickets) each get their own row, original and facts.

A changed original under the same `source_id` (a ticket that gained a
comment) is versioned as `<source_id>--v<n>`, with
`metadata.supersedes_source` naming the prior slug
(`01-spec/provenance-chain.md` "Source versioning"); each version's
original is stored under its own slug.

`raw.metadata` (the connector's extras: ticket comments, email headers,
CRM object and id, RSS link) is stored verbatim under
`sources.metadata["raw"]`, nested so its keys never collide with
`supersedes_source`: for ticket/crm/rss, `raw.text` is a rendering that
cannot be turned back into that structure.
"""

from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

from portolan.extract.schema import SCHEMA_VERSION
from portolan.ingest.models import Participant, RawSource, SourceRow
from portolan.store import Store

__all__ = ["SOURCE_COLUMNS", "content_hash", "intake", "original_key", "row_to_source_row"]


def content_hash(raw: RawSource) -> str:
    data = raw.original_bytes if raw.original_bytes is not None else raw.text.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def original_key(raw: RawSource, slug: str | None = None) -> str:
    """The store key for `raw`'s original. `slug` defaults to
    `raw.source_id` (the common case — no prior version exists), but
    `intake` passes the actual *resolved* slug for a versioned re-ingest,
    so `<source_id>--v2`'s original never overwrites `<source_id>`'s."""
    return f"originals/{raw.source_type.value}/{slug or raw.source_id}"


# The one column list for reading `sources` into a `SourceRow`.
SOURCE_COLUMNS = (
    "slug",
    "source_id",
    "source_type",
    "s3_key",
    "content_hash",
    "source_date",
    "ingested_at",
    "extraction_status",
    "filter_reason",
    "schema_version",
    "attempts",
    "last_error",
    "summary",
    "participants",
    "metadata",
    "extraction_model",
    "extraction_prompt_variant",
    "extraction_samples",
)

_VERSION_SUFFIX_RE = re.compile(r"^(?P<base>.+)--v(?P<version>\d+)$")


def row_to_source_row(row: tuple) -> SourceRow:
    """A `sources` row (columns in `SOURCE_COLUMNS` order) as a `SourceRow`."""
    values = dict(zip(SOURCE_COLUMNS, row, strict=True))
    values["participants"] = [Participant(**p) for p in (values["participants"] or [])]
    values["metadata"] = values["metadata"] or {}
    return SourceRow(**values)


def _find_by_source_and_hash(
    conn: psycopg.Connection, source_id: str, hash_value: str
) -> SourceRow | None:
    """The idempotency check: already ingested only when both `source_id`
    and the content hash match (module docstring)."""
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT {', '.join(SOURCE_COLUMNS)} FROM sources "
            "WHERE source_id = %s AND content_hash = %s LIMIT 1",
            (source_id, hash_value),
        )
        row = cur.fetchone()
    return row_to_source_row(row) if row is not None else None


def _slug_version(slug: str, base_slug: str) -> int:
    if slug == base_slug:
        return 1
    match = _VERSION_SUFFIX_RE.match(slug)
    if match and match.group("base") == base_slug:
        return int(match.group("version"))
    return 1


def _resolve_slug(conn: psycopg.Connection, base_slug: str) -> tuple[str, str | None]:
    """`(slug_to_use, supersedes_source)`. `supersedes_source` is the most
    recent existing slug for this `source_id` family, or `None` when
    `base_slug` itself is free (the common, non-versioned case).

    Found by the indexed `source_id` (every version carries its base id),
    not `slug LIKE '<id>--v%'`: under a non-C collation that LIKE is a
    sequential scan per new source, and `_`/`%` in an id are wildcards."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT slug FROM sources WHERE source_id = %s OR slug = %s",
            (base_slug, base_slug),
        )
        existing = [row[0] for row in cur.fetchall()]

    if not existing:
        return base_slug, None

    existing.sort(key=lambda slug: _slug_version(slug, base_slug))
    latest = existing[-1]
    next_version = _slug_version(latest, base_slug) + 1
    return f"{base_slug}--v{next_version}", latest


def intake(raw: RawSource, *, conn: psycopg.Connection, store: Store) -> SourceRow:
    hash_value = content_hash(raw)

    existing = _find_by_source_and_hash(conn, raw.source_id, hash_value)
    if existing is not None:
        return existing

    slug, supersedes = _resolve_slug(conn, raw.source_id)

    key = original_key(raw, slug=slug)
    if raw.original_bytes is not None:
        data = raw.original_bytes
        content_type = raw.original_content_type or "application/octet-stream"
    else:
        data = raw.text.encode("utf-8")
        content_type = raw.original_content_type or "text/plain; charset=utf-8"
    store.put(key, data, content_type)

    participants_json = [p.model_dump(mode="json") for p in raw.participants]
    metadata: dict[str, Any] = {}
    if supersedes:
        metadata["supersedes_source"] = supersedes
    if raw.metadata:
        metadata["raw"] = raw.metadata
    now = datetime.now(UTC)

    with conn.cursor() as cur:
        cur.execute(
            f"""
            INSERT INTO sources
                (slug, source_id, source_type, s3_key, content_hash, source_date,
                 ingested_at, extraction_status, schema_version, participants, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s, 'pending', %s, %s, %s)
            ON CONFLICT (source_id, content_hash)
                WHERE content_hash IS NOT NULL AND source_id IS NOT NULL
                DO NOTHING
            RETURNING {", ".join(SOURCE_COLUMNS)}
            """,
            (
                slug,
                raw.source_id,
                raw.source_type.value,
                key,
                hash_value,
                raw.source_date,
                now,
                SCHEMA_VERSION,
                Jsonb(participants_json),
                Jsonb(metadata),
            ),
        )
        row = cur.fetchone()
    conn.commit()

    if row is None:
        # Lost a race to another writer of the same (source_id, content_hash);
        # its row is authoritative.
        existing = _find_by_source_and_hash(conn, raw.source_id, hash_value)
        if existing is None:  # pragma: no cover - defensive, shouldn't happen
            raise RuntimeError(
                f"intake: INSERT for source_id={raw.source_id!r} content_hash={hash_value!r} "
                "hit ON CONFLICT DO NOTHING but no row with that (source_id, content_hash) "
                "exists to return -- this should be unreachable"
            )
        return existing

    return row_to_source_row(row)
