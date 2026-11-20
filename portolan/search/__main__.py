"""CLI: `python -m portolan.search "query" [--expand N] [--k N] [--graph MODE] ...`

Thin wrapper over `portolan.search.query.search()` / `explain()`, for manual
testing. `--graph expand|entity|ppr` (and `--window SINCE:UNTIL`) goes through
`portolan.search.graph.graph_search` instead; the default is plain search.
"""

from __future__ import annotations

import argparse
from datetime import date

from portolan.db import connect
from portolan.search.embed import get_embedder
from portolan.search.graph import DEFAULT_HOPS, GRAPH_MODES, graph_search
from portolan.search.query import explain, search


def _window(value: str) -> tuple[date, date]:
    since, _, until = value.partition(":")
    return date.fromisoformat(since), date.fromisoformat(until)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m portolan.search")
    parser.add_argument("query")
    parser.add_argument("--k", type=int, default=12)
    parser.add_argument("--expand", type=int, default=0)
    parser.add_argument("--type", dest="types", action="append", default=None)
    parser.add_argument("--since", type=date.fromisoformat, default=None)
    parser.add_argument("--graph", choices=GRAPH_MODES, default="off")
    parser.add_argument("--hops", type=int, default=DEFAULT_HOPS)
    parser.add_argument(
        "--window", type=_window, default=None, help="SINCE:UNTIL, claims recorded in range"
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    from portolan.config import settings  # local: importing this module loads no settings

    conn = connect()
    embedder = get_embedder(settings)
    if args.graph != "off" or args.window is not None:
        result = graph_search(
            conn,
            embedder,
            args.query,
            mode=args.graph,
            k=args.k,
            hops=args.hops,
            window=args.window,
        )
    else:
        result = search(
            conn,
            embedder,
            args.query,
            k=args.k,
            expand=args.expand,
            types=args.types,
            since=args.since,
        )

    if args.json:
        print(result.model_dump_json(indent=2))
    else:
        print(explain(result))


if __name__ == "__main__":
    main()
