"""`/healthz`: unauthenticated liveness/readiness probe.

Runs `SELECT 1` through `app.state.pool`, the pool every request uses, and
returns 503 when that fails, so the deploy health check (`railway.json`
`deploy.healthcheckPath`) sees a dead database.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter()
logger = logging.getLogger("uvicorn.error")


@router.get("/healthz", include_in_schema=False)
def healthz(request: Request) -> JSONResponse:
    pool = request.app.state.pool
    try:
        with pool.connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT 1")
            cur.fetchone()
    except Exception:
        logger.exception("healthz: database check failed")
        return JSONResponse({"status": "error"}, status_code=503)
    return JSONResponse({"status": "ok"})


__all__ = ["router"]
