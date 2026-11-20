"""Portolan's web app: chat, note/source/original viewers, graph, reports,
coverage and review (build-contracts.md "Web and MCP";
adr-001-language-and-stack.md: FastAPI + Jinja2 + HTMX, served from the
same process as the API).

Public surface: `create_app(settings)` builds the FastAPI application.
`python -m portolan.web` runs it under uvicorn; `python -m portolan.web.demo`
builds a demo graph from `corpus/out` first so there's something to click
through.
"""

from __future__ import annotations

from portolan.web.app import create_app

__all__ = ["create_app"]
