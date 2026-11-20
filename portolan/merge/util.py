"""Small helpers shared by `portolan.merge.facts` and
`portolan.merge.actions`."""

from __future__ import annotations

from datetime import datetime, time
from zoneinfo import ZoneInfo

from portolan.extract.schema import Fact
from portolan.ingest.models import SourceRow
from portolan.util import clock


def recorded_dt(source: SourceRow) -> datetime:
    """`claims.recorded` ("Every inserted claim carries `recorded =
    source.source_date`"): midnight of `source.source_date` in the
    deployment's zone (`PORTOLAN_TIMEZONE`), or now for an undated source.

    Midnight is anchored in the same zone the read side casts into
    (`clock.world_date_sql`), so the date round-trips whatever the database
    session's own `TimeZone`; a UTC midnight would read back as the previous
    day in a zone behind UTC.
    """
    if source.source_date is not None:
        return datetime.combine(
            source.source_date, time.min, tzinfo=ZoneInfo(clock.timezone_name())
        )
    return clock.now()


def fact_payload(fact: Fact) -> dict:
    """`fact` as plain JSON, every field: what a `review_queue` row keeps so
    the fact can be rebuilt (`Fact.model_validate`) and merged later."""
    return {
        "id": fact.id,
        "type": str(fact.type),
        "entities": list(fact.entities),
        "claim": fact.claim,
        "quote": fact.quote,
        "locator": fact.locator,
        "speaker": fact.speaker,
        "valid_from": fact.valid_from.isoformat() if fact.valid_from else None,
        "valid_to": fact.valid_to.isoformat() if fact.valid_to else None,
        "severity": fact.severity,
        "sentiment": fact.sentiment,
        "confidence": fact.confidence,
    }
