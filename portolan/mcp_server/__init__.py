"""Portolan's MCP server.

    from portolan.config import settings
    from portolan.mcp_server import create_server

    server = create_server(settings)
    server.run(transport="stdio")

`README.md` in this package covers pointing Claude Code or Claude Desktop at
it over stdio; `server.py` has the streamable-HTTP transport and its auth gates.
"""

from __future__ import annotations

from portolan.mcp_server.server import (
    ORG_KEY_ENV_VAR,
    TOOL_NAMES,
    build_http_app,
    create_server,
    get_org_key,
)

__all__ = [
    "ORG_KEY_ENV_VAR",
    "TOOL_NAMES",
    "build_http_app",
    "create_server",
    "get_org_key",
]
