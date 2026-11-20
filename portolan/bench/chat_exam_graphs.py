"""The chat exam's two local graphs, built from local data only (evaluation only).

- `demo`: the demo graph, the fake extractor over the synthetic Halvard corpus
  (`demo_search.seed_demo_graph`), the graph the live app was seeded from.
- `world`: a held-out v2 world (`WORLD_SPEC`, simulated and planned by code, no model call),
  put through the real pipeline with perfect extraction (`graph_world.seed_world_graph`).

Both are indexed with the production embedder (fastembed, `BAAI/bge-large-en-v1.5`) into
their own local `portolan_test_chat_exam_*` database; any other database is refused.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import psycopg

from portolan import db as db_module
from portolan.util.redact import redact_dsn

__all__ = [
    "GRAPHS",
    "WORLD_SPEC",
    "GraphSpec",
    "check_local_test_url",
    "connect_graph",
    "graph_counts",
    "seed_graph",
]

# `pack:split:seed`: a test-split world no bench or tuning run has read (the graph retrieval
# bench used `medical_devices:test:7`).
WORLD_SPEC = "medical_devices:test:11"
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", ""}


@dataclass(frozen=True)
class GraphSpec:
    name: str
    database: str

    @property
    def default_url(self) -> str:
        return f"postgresql://localhost:5432/{self.database}"


GRAPHS: dict[str, GraphSpec] = {
    "demo": GraphSpec("demo", "portolan_test_chat_exam_demo"),
    "world": GraphSpec("world", "portolan_test_chat_exam_world"),
}


def check_local_test_url(url: str) -> None:
    """Refuse anything but a local `portolan_test*` database: the exam never reads a dev,
    copied-live or live graph."""
    parts = urlsplit(url)
    name = parts.path.lstrip("/")
    if not name.startswith("portolan_test") or (parts.hostname or "") not in _LOCAL_HOSTS:
        raise SystemExit(f"refusing {redact_dsn(url)}: a local portolan_test* database only")


def connect_graph(url: str) -> psycopg.Connection:
    check_local_test_url(url)
    conn = db_module.connect(url)
    conn.autocommit = True
    return conn


def graph_counts(conn: psycopg.Connection) -> dict[str, int]:
    counts: dict[str, int] = {}
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass('public.notes') IS NOT NULL")
        if not cur.fetchone()[0]:
            return {"notes": 0}
        for table in ("notes", "claims", "links", "sources", "chunks"):
            cur.execute(f"SELECT count(*) FROM {table}")
            counts[table] = cur.fetchone()[0]
    return counts


def seed_graph(name: str, url: str, embedder: Any) -> dict[str, int]:
    """Drop the database's public schema, migrate, and seed graph `name`. Returns row counts."""
    check_local_test_url(url)
    conn = db_module.connect(url)
    try:
        with conn.cursor() as cur:
            cur.execute("SET synchronous_commit = off")
            cur.execute("DROP SCHEMA IF EXISTS public CASCADE")
            cur.execute("CREATE SCHEMA public")
        conn.commit()
        db_module.apply_migrations(conn)
        conn.commit()
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("SET synchronous_commit = off")
        workdir = Path(tempfile.mkdtemp(prefix=f"chat_exam_{name}_"))
        if name == "demo":
            from portolan.bench.demo_search import seed_demo_graph

            seed_demo_graph(conn, workdir, embedder)
        elif name == "world":
            from portolan.bench import graph_world

            world, plan = graph_world.build_world(WORLD_SPEC)
            graph_world.seed_world_graph(conn, world, plan, workdir, embedder)
        else:
            raise ValueError(f"unknown graph {name!r}; known: {sorted(GRAPHS)}")
        return graph_counts(conn)
    finally:
        conn.close()
