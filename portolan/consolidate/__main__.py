"""CLI: `python -m portolan.consolidate [--limit 20]
[--llm fake|agent_sdk|bedrock|anthropic|openrouter]`
(`01-spec/build-contracts.md` "Report, audit, consolidation").
"""

from __future__ import annotations

import argparse
import re
import sys

from portolan import db
from portolan.config import get_settings
from portolan.consolidate.run import run_consolidation
from portolan.llm.base import LLMClient
from portolan.llm.fake import FakeLLMClient

__all__ = ["main"]

# Matches one line of `portolan.consolidate.draft._claim_line`:
# "- claim 12 [complaint] (2026-08-04): text here ^[slug @ locator]"
_CLAIM_LINE_RE = re.compile(
    r"^- claim (\d+) \[([^\]]+)\](?: \([^)]*\))?: (.*?) (\^\[[^\]]+\])\s*$", re.M
)


def _fake_draft_responder(*, system, user, json_schema, model, surface):
    """`--llm fake`'s cheap-model stand-in: reconstructs a mechanically
    valid synthesis directly from the claim list `draft_synthesis` put in
    the user message, so the fake path exercises `check()`'s success case
    without a live model. Good enough for a CLI smoke run; a real
    consolidation should use `--llm agent_sdk`."""
    del system, model, json_schema, surface
    lines = [
        f"{text.strip()} {marker} [claim:{claim_id}]"
        for claim_id, _fact_type, text, marker in _CLAIM_LINE_RE.findall(user)
    ]
    return "\n".join(lines) or "Nothing on file yet."


def _build_llms(kind: str, settings) -> tuple[LLMClient, LLMClient]:
    """Review finding 11: every real provider now goes through
    `portolan.llm.factory.get_llm` -- the same factory `report.weekly`'s
    own `_build_llm` already uses -- superseding this CLI's old hand-wired
    `AgentSDKClient(...)` pair, which bypassed `get_llm`'s
    `PostgresUsageSink` entirely: consolidation's strong-model retries,
    one of the more expensive recurring jobs, never landed a row in
    `usage_events`. Can now be pointed at any provider `Settings.
    llm_provider` names, not just `agent_sdk`. `get_llm` always uses
    `settings.extraction_model` as its client's default model; the cheap
    and strong passes need `compare_model`/`strong_model` instead, so
    each builds its own `Settings` copy with `extraction_model`
    overridden accordingly (every other field -- tenant, that provider's
    own API key -- still comes from the real environment/`.env`)."""
    if kind == "fake":
        cheap = FakeLLMClient(
            _fake_draft_responder, model_key="claude-haiku-4-5", tenant=settings.tenant
        )
        strong = FakeLLMClient(
            _fake_draft_responder, model_key="claude-sonnet-5", tenant=settings.tenant
        )
        return cheap, strong

    from portolan.config import Settings
    from portolan.llm.factory import get_llm

    cheap = get_llm(Settings(llm_provider=kind, extraction_model=settings.compare_model))
    strong = get_llm(Settings(llm_provider=kind, extraction_model=settings.strong_model))
    return cheap, strong


def _build_embedder(kind: str):
    if kind == "fake":
        from portolan.search.embed import FakeEmbedder

        return FakeEmbedder()
    from portolan.search.embed import FastEmbedEmbedder

    return FastEmbedEmbedder()


def _parser() -> argparse.ArgumentParser:
    from typing import get_args

    from portolan.config import Settings

    parser = argparse.ArgumentParser(prog="python -m portolan.consolidate")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument(
        "--llm",
        # Review finding 11: every one of `Settings.llm_provider`'s own
        # choices (`fake` included) read off the Literal itself -- the
        # same pattern `report.weekly._build_arg_parser` already uses.
        choices=list(get_args(Settings.model_fields["llm_provider"].annotation)),
        default="fake",
    )
    parser.add_argument("--embed", choices=["fake", "fastembed"], default="fake")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    settings = get_settings()
    conn = db.connect()
    try:
        db.apply_migrations(conn)
        llm, strong_llm = _build_llms(args.llm, settings)
        # Review finding G21: without an embedder, a consolidated note's
        # rewritten synthesis (or a freshly (re-)written archive note)
        # left `chunks` stale until some later, separate full reindex.
        embedder = _build_embedder(args.embed)
        outcomes = run_consolidation(conn, llm, strong_llm, limit=args.limit, embedder=embedder)
        print("[" + ",\n".join(o.model_dump_json() for o in outcomes) + "]")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
