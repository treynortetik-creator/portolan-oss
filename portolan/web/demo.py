"""Builds the demo graph: `pipeline.run` with the fake extractor over
`corpus/out`, into the database `get_settings()` points at, so
`python -m portolan.web.demo && python -m portolan.web` gives a graph to
click through.

`build_fake_llm_client` answers each extraction call with the facts
`corpus/truth-table.yaml` planted for that source: no model call, no
network (the same client `portolan.pipeline --llm fake` builds).

Then the demo's weekly CRM exports (`corpus/demo/weekly-crm/`, synthetic,
outside the Halvard exam directory) go
in as snapshots only (`portolan.crm.ingest`), so the Monday email
(`python -m portolan.report monday`) has week-to-week CRM changes to show.
"""

from __future__ import annotations

from pathlib import Path

from portolan import db
from portolan.config import get_settings
from portolan.connectors.file_connector import FileConnector
from portolan.crm.ingest import ingest_crm_snapshots
from portolan.fake_extractor import FakePlacements, build_fake_llm_client
from portolan.pipeline import run
from portolan.registry import Registry
from portolan.search.embed import FakeEmbedder
from portolan.store import LocalStore
from portolan.util.redact import redact_dsn

DEFAULT_CORPUS_PATH = Path(__file__).resolve().parent.parent.parent / "corpus" / "out"
DEFAULT_WEEKLY_CRM_PATH = DEFAULT_CORPUS_PATH.parent / "demo" / "weekly-crm"

__all__ = ["DEFAULT_CORPUS_PATH", "DEFAULT_WEEKLY_CRM_PATH", "build_demo_graph", "redact_dsn"]


def build_demo_graph(
    corpus_path: Path = DEFAULT_CORPUS_PATH,
    weekly_crm_path: Path | None = DEFAULT_WEEKLY_CRM_PATH,
) -> None:
    settings = get_settings()
    conn = db.connect(settings.database_url.get_secret_value())
    try:
        db.apply_migrations(conn)
        connector = FileConnector(corpus_path)
        llm = build_fake_llm_client(FakePlacements(corpus_path), tenant=settings.tenant)
        embedder = FakeEmbedder()
        store = LocalStore(settings.store_root)
        registry = Registry.load(conn)

        report = run(
            conn,
            connector=connector,
            llm=llm,
            embedder=embedder,
            registry=registry,
            store=store,
            settings=settings,
        )
        print(f"Demo graph built into {redact_dsn(settings.database_url.get_secret_value())}:")
        print(report.model_dump_json(indent=2))
        if weekly_crm_path is not None and weekly_crm_path.is_dir():
            crm = ingest_crm_snapshots(
                conn, FileConnector(weekly_crm_path), store, embedder=embedder
            )
            print(
                f"Weekly CRM exports: {crm.snapshots_recorded} snapshots, "
                f"{crm.changes.changes_written} changes written"
            )
    finally:
        conn.close()


def main() -> None:
    build_demo_graph()


if __name__ == "__main__":
    main()
