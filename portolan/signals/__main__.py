"""`python -m portolan.signals ...`: the external-signals CLI.

    python -m portolan.signals targets import FILE.yaml
    python -m portolan.signals targets list
    python -m portolan.signals scan [--llm openrouter] [--model ID] [--only feeds|pages]
        [--account CRM_ID] [--feed-days 14] [--judge-limit 200] [--budget-usd 1.0]
        [--no-judge] [--embed fake|fastembed|none]
    python -m portolan.signals status

`scan` refuses to run unless `PORTOLAN_EXTERNAL_SIGNALS=true`. With `--llm
openrouter` the judge's spend is recorded in the machine's spend meter under
the key `external_signals` as well as in `usage_events`.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import httpx

from portolan import db
from portolan.config import Settings, get_settings, secret_value
from portolan.connectors.base import httpx_transport
from portolan.llm.base import LLMClient
from portolan.metering import PostgresSink
from portolan.signals.fetch import PoliteFetcher
from portolan.signals.scan import SignalsDisabled, run_scan
from portolan.signals.targets import import_targets, load_targets
from portolan.store import LocalStore

SPEND_KEY = "external_signals"
_REPO_VAR = Path(__file__).resolve().parents[2] / "var"


def _build_llm(kind: str, settings: Settings, conn) -> LLMClient:
    sink = PostgresSink(conn)
    if kind == "openrouter":
        from portolan.llm.openrouter import OpenRouterClient
        from portolan.training.spend import SpendMeter, default_spend_path

        api_key = secret_value(settings.openrouter_api_key)
        if not api_key:
            raise SystemExit("--llm openrouter needs OPENROUTER_API_KEY set")
        return OpenRouterClient(
            spend_key=SPEND_KEY,
            spend_meter=SpendMeter(default_spend_path(_REPO_VAR)),
            api_key=api_key,
            default_model=settings.external_signals_model,
            provider_order=settings.openrouter_provider_order,
            sink=sink,
            tenant=settings.tenant,
        )
    from portolan.llm.factory import get_llm

    llm = get_llm(settings, provider=kind)
    llm.sink = sink  # type: ignore[attr-defined]
    return llm


def _build_embedder(kind: str):
    if kind == "none":
        return None
    from portolan.search.embed import FakeEmbedder, FastEmbedEmbedder

    return FakeEmbedder() if kind == "fake" else FastEmbedEmbedder()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m portolan.signals")
    sub = parser.add_subparsers(dest="command", required=True)

    targets = sub.add_parser("targets")
    targets_sub = targets.add_subparsers(dest="targets_command", required=True)
    imp = targets_sub.add_parser("import")
    imp.add_argument("file", type=Path)
    targets_sub.add_parser("list")

    scan = sub.add_parser("scan")
    scan.add_argument(
        "--llm",
        choices=["openrouter", "agent_sdk", "bedrock", "anthropic", "fake"],
        default="openrouter",
    )
    scan.add_argument("--model", default=None)
    scan.add_argument("--only", choices=["feeds", "pages"], default=None)
    scan.add_argument("--account", default=None)
    scan.add_argument("--feed-days", type=int, default=14)
    scan.add_argument("--judge-limit", type=int, default=200)
    scan.add_argument("--budget-usd", type=float, default=None)
    scan.add_argument("--no-judge", action="store_true")
    scan.add_argument("--embed", choices=["fake", "fastembed", "none"], default="fastembed")

    sub.add_parser("status")
    return parser


def _status(conn) -> dict:
    with conn.cursor() as cur:
        cur.execute("SELECT kind, active, count(*) FROM signal_targets GROUP BY 1, 2 ORDER BY 1, 2")
        targets = [{"kind": k, "active": a, "count": n} for k, a, n in cur.fetchall()]
        cur.execute(
            "SELECT item_kind, status, count(*) FROM signal_items GROUP BY 1, 2 ORDER BY 1, 2"
        )
        items = [{"kind": k, "status": s, "count": n} for k, s, n in cur.fetchall()]
        cur.execute("SELECT count(*) FROM page_snapshots")
        (snapshots,) = cur.fetchone()
    return {"targets": targets, "items": items, "page_snapshots": snapshots}


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    settings = get_settings()
    if args.command == "scan" and not settings.external_signals_enabled:
        print(SignalsDisabled.MESSAGE, file=sys.stderr)
        return 2
    conn = db.connect()
    try:
        db.apply_migrations(conn)
        if args.command == "targets":
            if args.targets_command == "import":
                print(json.dumps(asdict(import_targets(conn, args.file))))
            else:
                for t in load_targets(conn, active_only=False):
                    state = "active" if t.active else "inactive"
                    print(f"{t.id}\t{t.account_crm_id}\t{t.kind}\t{state}\t{t.url}")
            return 0
        if args.command == "status":
            print(json.dumps(_status(conn), indent=2))
            return 0

        client = httpx.Client(timeout=30.0, follow_redirects=True)
        fetcher = PoliteFetcher(
            httpx_transport(client), user_agent=settings.external_signals_user_agent
        )
        try:
            report = run_scan(
                conn,
                settings=settings,
                store=LocalStore(settings.store_root),
                llm=_build_llm(args.llm, settings, conn),
                fetcher=fetcher,
                model=args.model,
                only=args.only,
                account=args.account,
                feed_days=args.feed_days,
                judge_limit=args.judge_limit,
                budget_usd=args.budget_usd,
                skip_judge=args.no_judge,
                embedder=_build_embedder(args.embed),
            )
        except SignalsDisabled as exc:
            print(str(exc), file=sys.stderr)
            return 2
        finally:
            client.close()
        print(json.dumps(asdict(report), indent=2))
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
