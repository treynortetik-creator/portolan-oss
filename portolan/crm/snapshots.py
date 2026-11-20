"""CRM snapshots: one exported CRM record on one export date (`crm_snapshots`,
migration 0036), so code can compare a record across exports.

A snapshot is recorded for every CRM source that `intake` stores, from the
pipeline (`portolan.pipeline._prepare_one`) or from `portolan.crm.ingest`.
The export date (`as_of`) is, in order: the row's `export_date` column; the
date in the export's file name (`opportunities-2026-08-21.csv`); for a
connector with no file path (an API pull), the source's own date. A file
export with neither date cannot be placed in the week sequence and is not
recorded (`snapshot_date` returns `None`).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

from portolan.extract.schema import SourceType
from portolan.ingest.models import RawSource, SourceRow
from portolan.ingest.prepare import prepare_text

__all__ = [
    "SNAPSHOT_OBJECTS",
    "Snapshot",
    "load_snapshots",
    "record_snapshot",
    "snapshot_date",
]

SNAPSHOT_OBJECTS = ("account", "contact", "opportunity")

# A date just before the file extension: `contacts-2026-08-28.csv`.
_FILE_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})\.[A-Za-z0-9]+$")


@dataclass(frozen=True)
class Snapshot:
    id: int
    object: str
    crm_id: str
    account_crm_id: str | None
    as_of: date
    export_key: str
    fields: dict[str, str]
    source_slug: str
    locator: str


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value.strip())
    except ValueError:
        return None


def snapshot_date(
    fields: dict[str, Any], metadata: dict[str, Any], source_date: date
) -> date | None:
    """The export date for one row (module docstring), or `None`."""
    from_column = _parse_date(str(fields.get("export_date") or ""))
    if from_column is not None:
        return from_column
    path = metadata.get("path")
    if path:
        match = _FILE_DATE_RE.search(str(path))
        return _parse_date(match.group(1)) if match else None
    return source_date


def _row_fields(raw: RawSource) -> dict[str, str] | None:
    if raw.original_bytes is None:
        return None
    try:
        row = json.loads(raw.original_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(row, dict):
        return None
    return {str(k): "" if v is None else str(v) for k, v in row.items()}


def _stored_locator(source_row: SourceRow, raw: RawSource) -> str:
    """The locator the originals viewer shows for the STORED source. An
    unchanged row in a later export dedupes to the earlier source, whose
    metadata names the earlier file, so the current export's path would
    not match any block on that page."""
    stored_meta = (source_row.metadata or {}).get("raw") or raw.metadata
    stored = RawSource(
        source_id=source_row.source_id or raw.source_id,
        source_type=SourceType.CRM,
        source_date=source_row.source_date or raw.source_date,
        text="",
        metadata=stored_meta,
    )
    [span] = prepare_text(stored).locators
    return span.locator


def record_snapshot(conn: psycopg.Connection, raw: RawSource, source_row: SourceRow) -> int | None:
    """Record `raw` (a CRM row just taken in as `source_row`) as a snapshot.
    Returns the snapshot id, or `None` when the row is not an account,
    contact or opportunity, has no parseable fields, or has no export date.
    Idempotent: the same row from the same source on the same date is one
    snapshot. Commits."""
    if raw.source_type != SourceType.CRM:
        return None
    obj = str(raw.metadata.get("object") or "").lower()
    if obj not in SNAPSHOT_OBJECTS:
        return None
    fields = _row_fields(raw)
    if fields is None:
        return None
    crm_id = str(raw.metadata.get("crm_id") or "")
    if not crm_id:
        return None
    as_of = snapshot_date(fields, raw.metadata, raw.source_date)
    if as_of is None:
        return None
    account_crm_id = crm_id if obj == "account" else (fields.get("account_crm_id") or None)
    export_key = str(raw.metadata.get("path") or f"{obj}@{as_of.isoformat()}")
    locator = _stored_locator(source_row, raw)

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO crm_snapshots
                (object, crm_id, account_crm_id, as_of, export_key, fields, source_slug, locator)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (object, crm_id, as_of, source_slug) DO NOTHING
            RETURNING id
            """,
            (
                obj,
                crm_id,
                account_crm_id,
                as_of,
                export_key,
                Jsonb(fields),
                source_row.slug,
                locator,
            ),
        )
        row = cur.fetchone()
        if row is None:
            cur.execute(
                "SELECT id FROM crm_snapshots WHERE object = %s AND crm_id = %s "
                "AND as_of = %s AND source_slug = %s",
                (obj, crm_id, as_of, source_row.slug),
            )
            row = cur.fetchone()
    conn.commit()
    return row[0] if row else None


def load_snapshots(conn: psycopg.Connection, *, until: date | None = None) -> list[Snapshot]:
    """Every snapshot (with `as_of <= until` when given), in the order
    comparisons read them: record, then export date, then insertion."""
    sql = (
        "SELECT id, object, crm_id, account_crm_id, as_of, export_key, fields, source_slug, "
        "locator FROM crm_snapshots"
    )
    params: tuple[Any, ...] = ()
    if until is not None:
        sql += " WHERE as_of <= %s"
        params = (until,)
    sql += " ORDER BY object, crm_id, as_of, id"
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    return [
        Snapshot(
            id=r[0],
            object=r[1],
            crm_id=r[2],
            account_crm_id=r[3],
            as_of=r[4],
            export_key=r[5],
            fields=dict(r[6] or {}),
            source_slug=r[7],
            locator=r[8],
        )
        for r in rows
    ]
