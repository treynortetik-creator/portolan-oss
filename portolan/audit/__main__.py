"""CLI: `python -m portolan.audit --since 2026-08-01 --rate 0.05
[--llm fake|agent_sdk|bedrock|anthropic|openrouter]`
(`01-spec/build-contracts.md` "Report, audit, consolidation").
"""

from __future__ import annotations

import argparse
import sys
from datetime import date

from portolan import db
from portolan.audit.second_pass import AuditBackend
from portolan.config import get_settings
from portolan.extract.schema import SCHEMA_VERSION
from portolan.llm.base import LLMClient
from portolan.llm.fake import FakeLLMClient
from portolan.store import LocalStore

__all__ = ["main"]


def _fake_responder(*, system, user, json_schema, model, surface):
    """Trivial fake for `--llm fake`: the strong pass finds nothing beyond
    what the first pass already has (a safe, valid, uninformative
    extraction) and the judge never calls anything a genuine miss. Good
    enough to exercise the CLI's plumbing end to end without a live model;
    not a substitute for `--llm agent_sdk` on real data. Both the second-
    pass extraction and the judge call go through `surface="audit"`
    (`portolan.audit.second_pass.AuditBackend`), so the two are told apart
    by which schema was requested, not by `surface`.
    """
    del system, model, surface
    if json_schema is not None:
        properties = json_schema.get("properties", {})
        if "verdict" in properties:
            return {"verdict": "not_a_fact", "matched_claim_id": None, "reason": "fake --llm path"}
        if "facts" in properties and "coverage" in properties:
            return {
                "schema_version": SCHEMA_VERSION,
                "source_id": "unknown",
                "summary": "",
                "entities": [],
                "facts": [],
                "actions": [],
                "unknown_entities": [],
                "coverage": {"segments_read": 1, "segments_total": 1},
            }
    return "ok"


def _build_strong_llm(kind: str, settings) -> LLMClient:
    """Review finding 11: every real provider now goes through
    `portolan.llm.factory.get_llm` -- the same factory `report.weekly`'s
    own `_build_llm` already uses (superseding this CLI's old hand-wired
    `AgentSDKClient(default_model=settings.strong_model, ...)`, which
    bypassed `get_llm`'s `PostgresUsageSink` entirely: the audit's judge
    call, one of the more expensive recurring jobs, never landed a row in
    `usage_events`) -- and can be pointed at any provider `Settings.
    llm_provider` names (`bedrock`, `anthropic`, `openrouter`, `agent_sdk`),
    not just `agent_sdk`. `get_llm` always uses `settings.extraction_model`
    as its client's default model; the audit's second pass and judge call
    both need the STRONG model instead, so this builds a `Settings` copy
    with `extraction_model` overridden to `settings.strong_model` (every
    other field -- tenant, that provider's own API key -- still comes from
    the real environment/`.env`, exactly like `Settings(llm_provider=...)`
    already works today)."""
    if kind == "fake":
        return FakeLLMClient(_fake_responder, model_key="claude-sonnet-5", tenant=settings.tenant)

    from portolan.config import Settings
    from portolan.llm.factory import get_llm

    return get_llm(Settings(llm_provider=kind, extraction_model=settings.strong_model))


def _parser() -> argparse.ArgumentParser:
    from typing import get_args

    from portolan.config import Settings

    parser = argparse.ArgumentParser(prog="python -m portolan.audit")
    parser.add_argument("--since", required=True, help="YYYY-MM-DD")
    parser.add_argument("--rate", type=float, default=0.02)
    parser.add_argument("--seed", default="audit")
    parser.add_argument(
        "--llm",
        # Review finding 11: every one of `Settings.llm_provider`'s own
        # choices (`fake` included) read off the Literal itself rather
        # than a second, separately-maintained list -- the same pattern
        # `report.weekly._build_arg_parser` already uses.
        choices=list(get_args(Settings.model_fields["llm_provider"].annotation)),
        default="fake",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    from portolan.audit.report import run_audit

    args = _parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    settings = get_settings()
    conn = db.connect()
    try:
        db.apply_migrations(conn)
        store = LocalStore(settings.store_root)
        strong_llm = _build_strong_llm(args.llm, settings)
        strong_backend = AuditBackend(strong_llm)
        since = date.fromisoformat(args.since)
        report = run_audit(
            conn,
            store,
            strong_llm,
            strong_backend,
            since=since,
            rate=args.rate,
            seed=args.seed,
        )
        print(report.model_dump_json(indent=2))
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
