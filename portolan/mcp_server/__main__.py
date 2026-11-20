"""`python -m portolan.mcp_server` -- CLI entry point for the MCP server.

`--stdio` (the default) is what a local client -- Claude Code, Claude
Desktop -- spawns directly; see `README.md` for the client config
snippet. `--http` runs the streamable-HTTP transport for a hosted
deployment and requires `PORTOLAN_MCP_ORG_KEY` or the OAuth settings
(`oauth.py`, docs/mcp-hosted.md): without either, `build_http_app` refuses
to build the app and this CLI exits with a one-line error.
"""

from __future__ import annotations

import argparse
import sys

from portolan.config import get_settings
from portolan.mcp_server.oauth import load_oauth_config
from portolan.mcp_server.server import build_http_app, create_server


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    # `access status|on|off` manages the per-tenant AI-access switch
    # (`access.py`); dispatched before the transport parser so `--stdio`/`--http` still parse.
    if argv and argv[0] == "access":
        from portolan.mcp_server.access import main as access_main

        raise SystemExit(access_main(argv[1:]))

    parser = argparse.ArgumentParser(prog="python -m portolan.mcp_server")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--stdio", action="store_true", help="Run over stdio (default; for a local client)."
    )
    mode.add_argument(
        "--http",
        action="store_true",
        help="Run the streamable-HTTP transport (for a hosted deployment).",
    )
    parser.add_argument("--host", default="127.0.0.1", help="--http only.")
    parser.add_argument("--port", type=int, default=8000, help="--http only.")
    parser.add_argument("--path", default="/mcp", help="--http only: the mount path.")
    args = parser.parse_args(argv)

    settings = get_settings()
    # `hosted` switches `open_source`'s `original_url` to the web viewer route (`create_server`).
    server = create_server(settings, hosted=args.http)

    if args.http:
        import uvicorn

        try:
            app = build_http_app(
                server,
                oauth=load_oauth_config(settings),
                streamable_http_path=args.path,
                host=args.host,
            )
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc
        uvicorn.run(app, host=args.host, port=args.port)
    else:
        server.run(transport="stdio")


if __name__ == "__main__":
    main()
