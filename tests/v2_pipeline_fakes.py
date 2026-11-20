"""Stub teacher, label judge and writer for the v2 runner's dry run
(`scripts/data/synth_v2.py --dry-run`) and its tests. No network, $0.

`LabelEchoTeacher` is blind in form only: for each envelope it returns one
fact per WP4 label whose text is in the envelope (the label's type, a fixed
claim with no name or number, the span as the quote), so WP5 confirms every
scripted label and finds nothing else. `keep_all_judge` keeps every silver
fact. `dry_run_reply` is WP4's obedient writer that also answers WP6's
near-duplicate prompt. Every word is a fixed test token, never corpus text.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from corpus.generator.v2.writer import WrittenDoc
from portolan.extract.schema import Coverage, ExtractionResult, Fact
from portolan.metering import metered
from tests.v2_noise_fakes import neardup_reply
from tests.v2_writer_fakes import ScriptedBackend, obedient

__all__ = ["LabelEchoTeacher", "ScriptedBackend", "dry_run_reply", "keep_all_judge"]

Row = dict[str, Any]


@dataclass
class LabelEchoTeacher:
    docs: dict[str, WrittenDoc]
    # Extra rows per envelope (`{type, claim, quote}`), for tests that need a silver fact.
    extra: Callable[[Any], list[Row]] | None = None
    cost: float = 0.0
    calls: list[tuple[str, str]] = field(default_factory=list)

    def rows(self, envelope: Any) -> list[Row]:
        out: list[Row] = []
        for doc in self.docs.values():
            for label in doc.labels:
                if label.doc_id == envelope.source_id and label.text in envelope.text:
                    out.append(
                        {
                            "type": label.fact_type.value,
                            "claim": f"fixture claim about a {label.fact_type.value} fact",
                            "quote": label.text,
                            "locator": label.prepared_locator,
                        }
                    )
        return out + (self.extra(envelope) if self.extra else [])

    def factory(self, model: str):
        teacher = self

        class _Backend:
            def extract(self, envelope, system_prompt, sink, tenant):  # noqa: ANN001
                teacher.calls.append((model, envelope.source_id))
                with metered(sink, tenant, "extract", model) as usage:
                    usage.cost_override = teacher.cost
                    rows = teacher.rows(envelope)
                return ExtractionResult(
                    source_id=envelope.source_id,
                    summary="fixture summary",
                    facts=[
                        Fact(
                            id=f"f{i}",
                            type=r["type"],
                            claim=r["claim"],
                            quote=r["quote"],
                            entities=[],
                            locator=r.get("locator", "L1"),
                            sentiment=0,
                            confidence=0.9,
                        )
                        for i, r in enumerate(rows)
                    ],
                    coverage=Coverage(segments_read=1, segments_total=1),
                )

        return _Backend()


def keep_all_judge(model: str, system: str, user: str) -> tuple[dict[str, Any], float]:
    """A label judge that keeps every fact with its own type."""
    facts = json.loads(user.split("## Facts to check\n", 1)[1])
    return {
        "facts": [
            {
                "id": f["id"],
                "reason": "fixture keep",
                "stated_as_true": True,
                "durable": True,
                "claim_supported": True,
                "repeats": None,
                "best_type": f["fact_type"],
            }
            for f in facts
        ]
    }, 0.0


def dry_run_reply(model: str, system: str, user: str, n: int) -> str:
    if "\nORIGINAL (" in user and "\n\nCHANGE:" in user:
        return neardup_reply(user)
    return obedient(model, system, user, n)
