"""Take in CRM exports as snapshots only: store each row as a source (so a
citation opens it), record the snapshot, and write the changes. No model sees
the rows (`01-spec/red-team-2026-09-24.md` sec 1).

The pipeline records snapshots too (`portolan.pipeline._prepare_one`), for
CRM rows that also go to the extractor; this path is for an export that only
needs comparing, such as the demo's weekly CRM exports
(`corpus/demo/weekly-crm/`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import psycopg

from portolan.connectors.base import Connector
from portolan.crm.changes import RefreshReport, refresh_crm_changes
from portolan.crm.snapshots import record_snapshot
from portolan.extract.schema import SourceType
from portolan.ingest.intake import intake
from portolan.registry import Registry
from portolan.search.embed import Embedder
from portolan.store import Store

__all__ = ["CRM_SNAPSHOT_MODEL", "IngestReport", "ingest_crm_snapshots"]

# Stamped as the source's extraction model, so a pipeline run with a real
# model id never counts these rows as already extracted by it.
CRM_SNAPSHOT_MODEL = "crm-snapshot-code"


@dataclass
class IngestReport:
    rows_seen: int = 0
    snapshots_recorded: int = 0
    rows_not_recorded: int = 0
    changes: RefreshReport = field(default_factory=RefreshReport)


def ingest_crm_snapshots(
    conn: psycopg.Connection,
    connector: Connector,
    store: Store,
    *,
    embedder: Embedder | None = None,
) -> IngestReport:
    """Every CRM row `connector` lists: intake, snapshot, then one
    `refresh_crm_changes` over everything recorded so far."""
    report = IngestReport()
    for ref in connector.list_sources():
        if ref.source_type != SourceType.CRM:
            continue
        report.rows_seen += 1
        raw = connector.fetch(ref)
        source_row = intake(raw, conn=conn, store=store)
        if source_row.extraction_status == "pending":
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE sources SET extraction_status = 'done', extraction_model = %s, "
                    "summary = %s WHERE slug = %s AND extraction_status = 'pending'",
                    (
                        CRM_SNAPSHOT_MODEL,
                        "CRM export row, compared by code (portolan.crm)",
                        source_row.slug,
                    ),
                )
            conn.commit()
        if record_snapshot(conn, raw, source_row) is None:
            report.rows_not_recorded += 1
        else:
            report.snapshots_recorded += 1
    report.changes = refresh_crm_changes(conn, Registry.load(conn), embedder=embedder)
    return report
