"""Merge the facts a review row was holding, once a person has answered it.

A fact that named only an unknown (or undecidable) name is parked whole on
that name's `review_queue` row (`payload.facts`, written by
`portolan.registry.registry.park_mention`). When a person maps the name,
`replay_parked_facts` runs each such fact through `merge_extraction`, the
pipeline's own merge entry point, against its own source. Before this, the
alias was saved and the facts stayed in the queue's payload for good.

Only `blocking` entries are replayed: a non-blocking one was filed when it
was extracted and only lacked a link for the unknown name.
"""

from __future__ import annotations

import json

import psycopg
from pydantic import BaseModel, ConfigDict, ValidationError

from portolan.extract.schema import Coverage, ExtractionResult, Fact
from portolan.ingest.intake import SOURCE_COLUMNS, row_to_source_row
from portolan.ingest.models import SourceRow
from portolan.llm.base import LLMClient
from portolan.merge.pipeline import merge_extraction
from portolan.registry import Registry
from portolan.search.embed import Embedder

__all__ = ["ReplayReport", "replay_parked_facts"]


class ReplayReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Facts that now sit on a note: a new claim, or a citation on one.
    replayed: int = 0
    # Still without a home (another name in the fact is unknown too).
    still_parked: int = 0
    # The stored fact or its source can no longer be rebuilt.
    malformed: int = 0


def _source_row(conn: psycopg.Connection, slug: str) -> SourceRow | None:
    with conn.cursor() as cur:
        cur.execute(f"SELECT {', '.join(SOURCE_COLUMNS)} FROM sources WHERE slug = %s", (slug,))
        row = cur.fetchone()
    return row_to_source_row(row) if row is not None else None


def replay_parked_facts(
    conn: psycopg.Connection,
    llm: LLMClient,
    embedder: Embedder,
    registry: Registry,
    row_id: int,
) -> ReplayReport:
    """Merge every blocking fact parked on `review_queue` row `row_id` that
    has not been merged yet. `registry` must already hold the person's
    answer (load it after `add_alias`). Each merged entry is marked
    `replayed` on the row, so a second call writes nothing.

    Works inside a caller's open transaction or outside one. Changes no
    row status: closing the row stays the caller's decision.
    """
    report = ReplayReport()
    with conn.cursor() as cur:
        cur.execute("SELECT payload FROM review_queue WHERE id = %s", (row_id,))
        row = cur.fetchone()
    if row is None:
        return report
    payload = dict(row[0] or {})
    entries = list(payload.get("facts") or [])

    for entry in entries:
        if not entry.get("blocking") or entry.get("replayed"):
            continue
        source = _source_row(conn, entry.get("source_slug") or "")
        try:
            fact = Fact.model_validate(entry.get("fact") or {})
        except ValidationError:
            fact = None
        if source is None or fact is None:
            report.malformed += 1
            continue
        result = ExtractionResult(
            source_id=source.source_id or source.slug,
            summary="",
            facts=[fact],
            coverage=Coverage(segments_read=1, segments_total=1),
        )
        merge_report = merge_extraction(conn, llm, embedder, registry, result, source)
        placement = merge_report.placements[0] if merge_report.placements else None
        if placement is not None and placement.claim_id is not None:
            entry["replayed"] = True
            entry["claim_id"] = placement.claim_id
            report.replayed += 1
        else:
            report.still_parked += 1

    if report.replayed:
        # Re-read: the merges above may have appended to this same row.
        with conn.cursor() as cur:
            cur.execute("SELECT payload FROM review_queue WHERE id = %s FOR UPDATE", (row_id,))
            current = dict(cur.fetchone()[0] or {})
            done = {
                (
                    e.get("source_slug"),
                    (e.get("fact") or {}).get("locator"),
                    (e.get("fact") or {}).get("claim"),
                ): e
                for e in entries
                if e.get("replayed")
            }
            for stored in current.get("facts") or []:
                key = (
                    stored.get("source_slug"),
                    (stored.get("fact") or {}).get("locator"),
                    (stored.get("fact") or {}).get("claim"),
                )
                if key in done:
                    stored["replayed"] = True
                    stored["claim_id"] = done[key]["claim_id"]
            cur.execute(
                "UPDATE review_queue SET payload = %s::jsonb WHERE id = %s",
                (json.dumps(current), row_id),
            )
        if conn.info.transaction_status == psycopg.pq.TransactionStatus.INTRANS:
            conn.commit()
    return report
