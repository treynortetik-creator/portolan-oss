"""Write-path run on a real model: the synthetic corpus through the full
pipeline with the configured provider (default: the local Claude Code login via
the Agent SDK, no API key), then the same placement checks `noisy_extractor.py`
makes, without the planted damage.

The noisy run knows every fact's fate because it tags each one. A live model's
facts are not tagged, so this reports aggregates: how many facts became claims,
citations or review rows, and whether the claims it wrote cite a locator the
source has, quote the source's own words, and quote them at the place they cite.

    createdb portolan_test_live
    DATABASE_URL=postgresql://localhost/portolan_test_live STORE_ROOT=/tmp/live/store \\
        EXTRACTION_MODEL=claude-haiku-5-5 COMPARE_MODEL=claude-haiku-5-5 \\
        uv run python scripts/bench_write/live_stress.py --path corpus/out

Needs an empty database whose name starts with `portolan_test`. Spends model
calls. Prints one JSON object.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

from noisy_extractor import _norm

from portolan import db
from portolan.config import get_settings
from portolan.connectors.file_connector import FileConnector
from portolan.ingest.prepare import prepare_text
from portolan.llm import get_llm
from portolan.metering import InMemorySink
from portolan.pipeline import run
from portolan.registry import Registry
from portolan.search.embed import FakeEmbedder
from portolan.store import LocalStore


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--path", type=Path, default=Path("corpus/out"))
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args(argv)

    settings = get_settings()
    url = settings.database_url.get_secret_value()
    if not url.rsplit("/", 1)[-1].startswith("portolan_test"):
        print("refusing: DATABASE_URL must name a portolan_test* database", file=sys.stderr)
        return 2
    os.environ.setdefault("PORTOLAN_EXTRACT_CONCURRENCY", str(args.concurrency))

    conn = db.connect()
    db.apply_migrations(conn)
    connector = FileConnector(args.path)
    sink = InMemorySink()
    report = run(
        conn,
        connector=connector,
        llm=get_llm(settings),
        embedder=FakeEmbedder(),
        registry=Registry.load(conn),
        store=LocalStore(settings.store_root),
        settings=settings,
        sink=sink,
    )

    texts: dict[str, str] = {}
    spans: dict[str, dict[str, tuple[int, int]]] = {}
    for ref in connector.list_sources(None):
        prepared = prepare_text(connector.fetch(ref))
        texts[ref.source_id] = prepared.text
        spans[ref.source_id] = {s.locator: (s.start, s.end) for s in prepared.locators}

    with conn.cursor() as cur:
        cur.execute("SELECT source_slug, locator, quote FROM claims")
        claims = cur.fetchall()
        cur.execute("SELECT status, kind FROM review_queue")
        queue = cur.fetchall()
        cur.execute("SELECT count(*) FROM claim_citations")
        citations = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM claims WHERE superseded_at IS NOT NULL")
        superseded = cur.fetchone()[0]

    dead = wrong_place = quoted = not_verbatim = 0
    for source_slug, locator, quote in claims:
        base = source_slug.split("--v")[0]
        source_spans = spans.get(base, {})
        if source_spans and locator not in source_spans:
            dead += 1
            continue
        if quote:
            quoted += 1
            text = texts.get(base, "")
            if _norm(quote) not in _norm(text):
                not_verbatim += 1
            elif locator in source_spans:
                start, end = source_spans[locator]
                if _norm(quote) not in _norm(text[start:end]):
                    wrong_place += 1

    cost = sum(e.cost_estimate or 0 for e in sink.events) if hasattr(sink, "events") else None
    print(
        json.dumps(
            {
                "extraction_model": settings.extraction_model,
                "compare_model": settings.compare_model,
                "sources_extracted": report.sources_extracted,
                "sources_failed": report.sources_failed,
                "claims": len(claims),
                "claims_superseded": superseded,
                "extra_citations": citations,
                "claims_citing_a_locator_not_in_the_source": dead,
                "claims_with_a_quote": quoted,
                "claim_quotes_not_verbatim_in_source": not_verbatim,
                "claim_quotes_not_at_the_cited_place": wrong_place,
                "review_rows_by_status": dict(Counter(s for s, _ in queue)),
                "open_review_rows_by_kind": dict(Counter(k for s, k in queue if s == "open")),
                "cost_estimate_usd": cost,
            },
            indent=2,
        )
    )
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
