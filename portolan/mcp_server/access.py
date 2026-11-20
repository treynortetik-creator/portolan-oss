"""The per-tenant AI-access switch for the MCP door (checklist item 26(b),
`01-spec/app-checklist-2026-09-18.md`; idea from `01-spec/research/
openworkgraph-2026-09-22.md` sec 4, reimplemented, nothing copied).

One row per tenant in `ai_access` (migration `0031_ai_access.sql`) says
whether that tenant's MCP door may answer at all. The door
(`server.PortolanMCPServer`) calls `require_ai_access` before every tool
call, `tools/list`, `resources/list`, `resources/templates/list`,
`resources/read`, `prompts/list` and `prompts/get` -- a fresh read every
time, no cache -- so turning a tenant off closes its door on the very next
request, with no redeploy and no key rotation.

Per tenant, not per user: no role-based access in v1 (CLAUDE.md, "No RBAC
in v1"). Everyone the door admits sees everything, or nobody does.

Lookup order: the tenant's own row, else the database-default row
`tenant = '*'`, else CLOSED. A brand-new database has neither row, so its
door is off until someone turns it on; migration 0031 seeds `'*'` = ON
only on a database that already existed when it ran (see that file's
header; `tests/test_mcp_door.py`).

The in-process action researcher calls `tools` directly and never passes
this gate.

CLI (names only, never a DSN -- `CLAUDE.md` "never print deployment
variables"):

    python -m portolan.mcp_server access status [--tenant T]
    python -m portolan.mcp_server access on  [--tenant T] [--by NAME] [--reason TEXT]
    python -m portolan.mcp_server access off [--tenant T] [--by NAME] [--reason TEXT]

`--tenant` defaults to `Settings.tenant` (`PORTOLAN_TENANT`); `--tenant
'*'` addresses the database default.
"""

from __future__ import annotations

import argparse
import getpass
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import psycopg
from mcp.shared.exceptions import MCPError

DATABASE_DEFAULT_TENANT = "*"

# JSON-RPC reserves -32000..-32099 for implementation-defined server
# errors; this one means "the tenant's AI-access switch is off".
AI_ACCESS_CLOSED_CODE = -32010

MAX_TENANT_CHARS = 200
MAX_CHANGED_BY_CHARS = 200
MAX_REASON_CHARS = 2_000

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AccessState:
    """The effective switch for one tenant, and which row decided it."""

    tenant: str
    enabled: bool
    # "tenant": the tenant's own row; "database_default": the '*' row;
    # "none": no row at all (closed); "unmigrated": `ai_access` does not
    # exist yet (closed -- fail shut, never open).
    decided_by: Literal["tenant", "database_default", "none", "unmigrated"]
    changed_at: datetime | None = None
    changed_by: str | None = None
    reason: str | None = None


class AIAccessClosedError(MCPError):
    """Raised at the door when the tenant's switch is off. An `MCPError`,
    so the SDK answers with a top-level JSON-RPC error carrying this
    message rather than an empty list or an `is_error` tool result the
    client might mistake for "nothing found"."""

    def __init__(self, state: AccessState) -> None:
        if state.decided_by == "unmigrated":
            why = (
                "the AI-access switch table does not exist yet -- run the database "
                "migrations (0031_ai_access.sql) first"
            )
        elif state.decided_by == "none":
            why = "it has never been turned on for this tenant"
        else:
            who = f" by {state.changed_by}" if state.changed_by else ""
            when = f" at {state.changed_at.isoformat()}" if state.changed_at else ""
            why = f"it was turned off{who}{when}"
        super().__init__(
            code=AI_ACCESS_CLOSED_CODE,
            message=(
                f"Portolan AI access is OFF for tenant {state.tenant!r}: {why}. Nothing was "
                "read. This is a deliberate switch, not an empty result -- do not report it "
                "as 'no data'. An administrator can turn it on with "
                f"`python -m portolan.mcp_server access on --tenant {state.tenant}`."
            ),
            data={"reason": "ai_access_closed", "tenant": state.tenant},
        )
        self.state = state


def get_ai_access(conn: psycopg.Connection, tenant: str) -> AccessState:
    """The effective switch for `tenant` (see the module docstring for the
    lookup order). Never raises for a missing table -- reports it as
    closed (`decided_by="unmigrated"`) instead, so a door pointed at an
    unmigrated database fails shut."""
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT tenant, enabled, changed_at, changed_by, reason FROM ai_access "
                "WHERE tenant = %s OR tenant = %s",
                (tenant, DATABASE_DEFAULT_TENANT),
            )
            rows = {row[0]: row for row in cur.fetchall()}
    except psycopg.errors.UndefinedTable:
        if not conn.autocommit:
            conn.rollback()
        return AccessState(tenant=tenant, enabled=False, decided_by="unmigrated")

    row = rows.get(tenant)
    decided_by: Literal["tenant", "database_default"] = "tenant"
    if row is None:
        row = rows.get(DATABASE_DEFAULT_TENANT)
        decided_by = "database_default"
    if row is None:
        return AccessState(tenant=tenant, enabled=False, decided_by="none")
    _, enabled, changed_at, changed_by, reason = row
    return AccessState(
        tenant=tenant,
        enabled=bool(enabled),
        decided_by=decided_by,
        changed_at=changed_at,
        changed_by=changed_by,
        reason=reason,
    )


def require_ai_access(conn: psycopg.Connection, tenant: str, *, method: str) -> AccessState:
    """Raise `AIAccessClosedError` unless `tenant`'s door is open. `method`
    (e.g. `"tools/call:search"`) is logged on a denial -- never an
    argument, a query, or anything a client sent beyond the method name."""
    state = get_ai_access(conn, tenant)
    if not state.enabled:
        _logger.warning(
            "mcp ai access closed: tenant=%s method=%s decided_by=%s",
            tenant,
            method,
            state.decided_by,
        )
        raise AIAccessClosedError(state)
    return state


def set_ai_access(
    conn: psycopg.Connection,
    tenant: str,
    *,
    enabled: bool,
    changed_by: str,
    reason: str | None = None,
) -> AccessState:
    """Turn `tenant`'s door on or off, and append the change to
    `ai_access_log`, atomically. Returns the new effective state."""
    tenant = tenant.strip()
    changed_by = changed_by.strip()
    if not tenant or len(tenant) > MAX_TENANT_CHARS:
        raise ValueError(f"tenant must be 1-{MAX_TENANT_CHARS} characters")
    if not changed_by or len(changed_by) > MAX_CHANGED_BY_CHARS:
        raise ValueError(f"changed_by must be 1-{MAX_CHANGED_BY_CHARS} characters")
    if reason is not None and len(reason) > MAX_REASON_CHARS:
        raise ValueError(f"reason must be at most {MAX_REASON_CHARS:,} characters")

    with conn.transaction(), conn.cursor() as cur:
        cur.execute(
            """
                INSERT INTO ai_access (tenant, enabled, changed_at, changed_by, reason)
                VALUES (%s, %s, now(), %s, %s)
                ON CONFLICT (tenant) DO UPDATE
                    SET enabled = EXCLUDED.enabled,
                        changed_at = EXCLUDED.changed_at,
                        changed_by = EXCLUDED.changed_by,
                        reason = EXCLUDED.reason
                RETURNING changed_at
                """,
            (tenant, enabled, changed_by, reason),
        )
        (changed_at,) = cur.fetchone()
        cur.execute(
            "INSERT INTO ai_access_log (tenant, enabled, changed_at, changed_by, reason) "
            "VALUES (%s, %s, %s, %s, %s)",
            (tenant, enabled, changed_at, changed_by, reason),
        )
    return get_ai_access(conn, tenant)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def _describe(state: AccessState) -> str:
    status = "ON" if state.enabled else "OFF"
    source = {
        "tenant": "its own switch",
        "database_default": "the database default ('*')",
        "none": "no switch set -- closed until turned on",
        "unmigrated": "ai_access table missing -- run migrations",
    }[state.decided_by]
    line = f"tenant {state.tenant!r}: AI access {status} ({source})"
    if state.changed_at is not None:
        line += f"; last changed {state.changed_at.isoformat()} by {state.changed_by}"
    if state.reason:
        line += f"; reason: {state.reason}"
    return line


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m portolan.mcp_server access")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("status", "show the effective switch"),
        ("on", "open the MCP door for a tenant"),
        ("off", "close the MCP door for a tenant"),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--tenant", default=None, help="default: PORTOLAN_TENANT; '*' = database")
        if name != "status":
            p.add_argument("--by", default=None, help="who is making the change (default: $USER)")
            p.add_argument("--reason", default=None, help="why (recorded in ai_access_log)")
    return parser


def main(argv: list[str] | None = None) -> int:
    from portolan import db as db_module
    from portolan.config import get_settings
    from portolan.util.redact import redact_dsn

    args = build_parser().parse_args(argv)
    settings = get_settings()
    tenant = args.tenant if args.tenant is not None else settings.tenant
    try:
        conn = db_module.connect(settings.database_url.get_secret_value(), autocommit=True)
    except psycopg.Error as exc:
        print(f"error: {type(exc).__name__}: {redact_dsn(str(exc))}", file=sys.stderr)
        return 1
    try:
        if args.command == "status":
            state = get_ai_access(conn, tenant)
        else:
            changed_by = args.by or getpass.getuser() or "cli"
            state = set_ai_access(
                conn,
                tenant,
                enabled=args.command == "on",
                changed_by=changed_by,
                reason=args.reason,
            )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except psycopg.Error as exc:
        print(f"error: {type(exc).__name__}: {redact_dsn(str(exc))}", file=sys.stderr)
        return 1
    finally:
        conn.close()
    print(_describe(state))
    return 0
