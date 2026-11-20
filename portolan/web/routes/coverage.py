"""`/coverage`: the ledger summary and the "not ingested" list (build
instructions "Routes"; portolan-spec-v2.md sec 6.2).
"""

from __future__ import annotations

from datetime import timedelta

import psycopg
from fastapi import APIRouter, Depends, Request

from portolan.ingest.ledger import latest_day as latest_ledger_day
from portolan.ingest.ledger import summary as coverage_summary
from portolan.util import clock
from portolan.web.auth import Session, require_session
from portolan.web.bounds import bounded_int
from portolan.web.deps import get_conn
from portolan.web.queries import dead_letter_rows, not_ingested_sources
from portolan.web.tracking import track_page_view

router = APIRouter()

DEFAULT_COVERAGE_DAYS = 14
# A huge `days` overflows `timedelta`; ten years is still generous.
MAX_COVERAGE_DAYS = 3_650


@router.get("/coverage", dependencies=[Depends(track_page_view)])
def coverage_page(
    request: Request,
    days: int = DEFAULT_COVERAGE_DAYS,
    session: Session = Depends(require_session),
    conn: psycopg.Connection = Depends(get_conn),
) -> object:
    templates = request.app.state.templates
    days = bounded_int(days, name="days", minimum=1, maximum=MAX_COVERAGE_DAYS)
    # The ledger's `day` is each source's own date, so the window ends at the
    # newest recorded day when that is before today; an older corpus would
    # otherwise show nothing.
    through = clock.today()
    latest = latest_ledger_day(conn)
    if latest is not None and latest < through:
        through = latest
    since = through - timedelta(days=days)
    return templates.TemplateResponse(
        request,
        "coverage.html",
        {
            "ledger": coverage_summary(conn, since),
            "not_ingested": not_ingested_sources(conn),
            "dead_letter": dead_letter_rows(conn),
            "since": since,
            "through": through,
            "days": days,
        },
    )


__all__ = ["router"]
