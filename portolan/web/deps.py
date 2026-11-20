"""FastAPI dependencies pulling the shared pool/settings/llm/embedder/store
off `app.state` (set up in `create_app`'s lifespan, `app.py`).

Plain functions so tests can override each one independently
(`app.dependency_overrides[get_conn] = lambda: db_conn`) and share their
fixture's connection.
"""

from __future__ import annotations

from collections.abc import Iterator

import psycopg
from fastapi import Request

from portolan.config import Settings
from portolan.llm.base import LLMClient
from portolan.search.embed import Embedder
from portolan.store import Store

__all__ = [
    "get_conn",
    "get_embedder_dep",
    "get_llm_dep",
    "get_settings_dep",
    "get_store_dep",
]


def get_conn(request: Request) -> Iterator[psycopg.Connection]:
    """A connection checked out of `app.state.pool` for the life of this
    request. The pool is built with `check=ConnectionPool.check_connection`
    (`app.py`), so a dead connection is replaced rather than handed out.

    A generator so FastAPI returns the connection to the pool when the
    request ends, success or error. A test override may be a plain
    callable; FastAPI dispatches on the override's own shape.
    """
    pool = request.app.state.pool
    with pool.connection() as conn:
        yield conn


def get_settings_dep(request: Request) -> Settings:
    return request.app.state.settings


def get_embedder_dep(request: Request) -> Embedder:
    return request.app.state.embedder


def get_llm_dep(request: Request) -> LLMClient:
    return request.app.state.llm


def get_store_dep(request: Request) -> Store:
    return request.app.state.store
