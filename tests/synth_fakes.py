"""Shared fakes for the synthetic-data-pipeline tests
(`test_training_export.py`, `test_synth_batches.py` when present):
a no-network `FakeLLMClient` teacher that reads `corpus.generator.prose.
build_user_prompt`'s own format back out of the `user` string it is given
and plants every listed fact, marker glyphs and all, in the right shape
per source type (call transcript line / ticket JSON / email paragraph /
one short chat-or-note message).

Not a `test_*.py` module on purpose (mirrors `tests/merge_helpers.py`) --
pytest must not collect it as a test file.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from corpus.generator import cli as cli_mod
from corpus.generator import prose
from portolan.llm.fake import FakeLLMClient

_SOURCE_TYPE_RE = re.compile(r"^source_type: (\S+)$", re.MULTILINE)
_FACT_BLOCK_RE = re.compile(
    r"- fact (\S+) \([^)]*\), position=\S+, speaker=\S+:\n  claim: (.+)\n  phrasing:",
)


def _marked(fact_id: str, claim: str) -> str:
    return f"⟦{fact_id}⟧{claim}⟦/{fact_id}⟧"


def fake_teacher_response(
    *, system: str, user: str, json_schema=None, model=None, surface=None
) -> str:
    """One canned reply per call, deterministic from `user` alone -- every
    fact block `build_user_prompt` printed gets planted verbatim, in the
    marker glyphs `markers.py` expects, positioned per `source_type`."""
    m = _SOURCE_TYPE_RE.search(user)
    source_type = m.group(1) if m else "call"
    facts = _FACT_BLOCK_RE.findall(user)  # [(fact_id, claim), ...]
    marked = [_marked(fid, claim) for fid, claim in facts]

    if source_type == "ticket":
        comments = [
            {"number": i + 1, "author": "Customer", "body": text} for i, text in enumerate(marked)
        ] or [{"number": 1, "author": "Customer", "body": "Nothing to report this week."}]
        return json.dumps({"subject": "Test ticket", "comments": comments})

    if source_type == "email":
        body = "\n\n".join(marked) if marked else "Nothing to report this week."
        return f"Hi there,\n\n{body}\n\nBest,\nThe team"

    if source_type == "call":
        if marked:
            lines = [f"[00:00:{(i * 5) % 60:02d}] Speaker: {text}" for i, text in enumerate(marked)]
        else:
            lines = ["[00:00:00] Speaker: Nothing much to report this week."]
        lines.append(f"[00:00:{(len(lines) * 5) % 60:02d}] Speaker: Thanks, talk soon.")
        return "\n".join(lines)

    # slack / teams / crm activity notes: one short message, at most one
    # fact per brief (worlds.py never plants two facts in the same chat
    # message or note).
    return marked[0] if marked else "nothing new to report"


def build_fake_client() -> FakeLLMClient:
    return FakeLLMClient(
        fake_teacher_response, model_key="nova-lite", tenant="corpus-generator-test"
    )


FAKE_TEACHERS = [
    "faketeacher/alpha:free",
    "faketeacher/beta:free",
    "faketeacher/gamma:free",
]


def run_fake_synth(
    seed: int,
    out_dir: Path,
    *,
    num_accounts: int = 2,
    per_kind: int = 4,
    type_mix: dict[str, int] | None = None,
    client: FakeLLMClient | None = None,
    rotation: prose.TeacherRotation | None = None,
    limiter: prose.TokenBucketLimiter | None = None,
) -> dict[str, Any]:
    """Run `cli.synth()` against `out_dir` with a fake, no-network teacher.
    Returns the same report dict `cli.synth()` returns (also written to
    `<out_dir>/synth_report.json`)."""
    return cli_mod.synth(
        seed=seed,
        num_accounts=num_accounts,
        per_kind=per_kind,
        out_dir=out_dir,
        type_mix=type_mix,
        client=client if client is not None else build_fake_client(),
        rotation=rotation if rotation is not None else prose.TeacherRotation(list(FAKE_TEACHERS)),
        limiter=limiter
        if limiter is not None
        else prose.TokenBucketLimiter(requests_per_minute=10_000, requests_per_day=10_000_000),
    )
