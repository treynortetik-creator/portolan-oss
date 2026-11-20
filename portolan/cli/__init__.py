"""`portolan`: use and test a running Portolan app from the shell.

    uv run portolan <command> [args] [--json] [--session FILE] [--target local|URL]

See `docs/CLI.md`. The entry point is `main`, wired in `pyproject.toml`
under `[project.scripts]`.
"""

from __future__ import annotations

from portolan.cli.app import main

__all__ = ["main"]
