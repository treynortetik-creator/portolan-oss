"""The manifest: schema, writer, and validator.

`corpus/out/manifest.json` is the bench's other input besides the corpus
itself — for every planted fact it records the exact source file and
locator where it landed, so scoring a model's extraction against it is a
set comparison, not a re-read of the truth table's prose.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

MANIFEST_SCHEMA_VERSION = "0.1"

LOCATOR_PATTERNS: dict[str, re.Pattern[str]] = {
    "call": re.compile(r"^\d{2}:\d{2}:\d{2}$"),
    "ticket": re.compile(r"^S-ticket-[A-Za-z0-9]+#\d+$"),
    "email": re.compile(r"^\S+@halvardmedical\.com#p\d+$"),
    "rss": re.compile(r"^url:\S+ \d{4}-\d{2}-\d{2}$"),
    # Widened 2026-09-19 (synthetic-data pipeline) from `\.csv#` to
    # `\.(csv|json)#` so a CRM activity note (its own small JSON file, same
    # "structured container, prose body" split as a ticket) validates
    # alongside the original account/opportunity/case CSV rows — additive,
    # every existing `...csv#ID` locator still matches.
    "crm": re.compile(r"^[\w./\-]+\.(csv|json)#[\w\-]+$"),
    # Chat sources — added 2026-09-19 for the Slack/Teams source kinds
    # (`corpus/generator/structured.py`'s `slack_locator`/`teams_locator`):
    # `<channel>/<YYYY-MM-DD>.json#m<1-based message index>`.
    "slack": re.compile(r"^slack/[\w\-]+/\d{4}-\d{2}-\d{2}\.json#m\d+$"),
    "teams": re.compile(r"^teams/[\w\-]+/\d{4}-\d{2}-\d{2}\.json#m\d+$"),
    # World generator v2 prose kinds (added 2026-09-22, WP4; spec section 9:
    # "extended with ... meeting_notes and doc kinds"): a CRM activity note,
    # meeting notes, a doc, and the three planned-negative kinds are plain
    # paragraph containers with no native line/comment/message structure,
    # so their locator is `<source id>#p<1-based paragraph>` -- the shape
    # `corpus.generator.v2.renderers._render_prose_container_skeleton`
    # already emits. Additive: no existing key changes.
    "crm_note": re.compile(r"^S-[\w\-]+#p\d+$"),
    "meeting_notes": re.compile(r"^S-[\w\-]+#p\d+$"),
    "doc": re.compile(r"^S-[\w\-]+#p\d+$"),
    "near_duplicate": re.compile(r"^S-[\w\-]+#p\d+$"),
    "off_topic": re.compile(r"^S-[\w\-]+#p\d+$"),
    "hard_negative": re.compile(r"^S-[\w\-]+#p\d+$"),
}


@dataclass
class FactPlacement:
    fact_id: str
    locator: str
    line_or_comment: str
    excerpt: str


@dataclass
class SourceRecord:
    id: str
    type: str
    path: str
    date: str
    participants: list[str]
    accounts: list[str]
    token_estimate: int
    facts: list[FactPlacement] = field(default_factory=list)
    is_noise: bool = False
    is_distractor: bool = False
    distractor_id: str | None = None
    # The free-teacher model id that wrote this source -- `None` for the
    # original `generate` pipeline (which never sets it), for a noise
    # source (deterministic filler, no model call), and for a structured
    # (no-prose) CRM row. Added 2026-09-21 for `training_export.py`'s
    # `"teachers"` field -- see `corpus.generator.cli`'s `_synth_process_*`.
    model: str | None = None


def truth_table_hash(truth_table_path: str | Path) -> str:
    data = Path(truth_table_path).read_bytes()
    return hashlib.sha256(data).hexdigest()


def build_manifest(
    *,
    mode: str,
    reduced: bool,
    generated_at: str,
    truth_table_path: str | Path,
    sources: list[SourceRecord],
) -> dict[str, Any]:
    facts_index: dict[str, list[dict[str, str]]] = {}
    for s in sources:
        for f in s.facts:
            facts_index.setdefault(f.fact_id, []).append(
                {
                    "source_id": s.id,
                    "locator": f.locator,
                    "line_or_comment": f.line_or_comment,
                    "excerpt": f.excerpt,
                }
            )
    return {
        "generator_version": "0.1.0",
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "mode": mode,
        "reduced": reduced,
        "generated_at": generated_at,
        "truth_table_hash": truth_table_hash(truth_table_path),
        "sources": [
            {
                "id": s.id,
                "type": s.type,
                "path": s.path,
                "date": s.date,
                "participants": s.participants,
                "accounts": s.accounts,
                "token_estimate": s.token_estimate,
                "is_noise": s.is_noise,
                "is_distractor": s.is_distractor,
                "distractor_id": s.distractor_id,
                "model": s.model,
                "facts": [asdict(f) for f in s.facts],
            }
            for s in sources
        ],
        "facts": facts_index,
    }


def write_manifest(manifest: dict[str, Any], out_path: str | Path) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=False)
        fh.write("\n")


def validate(manifest: dict[str, Any], truth_table: dict[str, Any]) -> list[str]:
    """Check the manifest against the truth table. Returns a list of
    problems; empty means the manifest is valid.

    Checks:
    1. Every planted (non-distractor) fact has placements whose source-id
       set matches the truth table's `sources` list for that fact exactly.
    2. Every distractor (type `distractor`) has NO placements.
    3. Every locator matches the format for its source's type.
    """
    problems: list[str] = []
    source_type_by_id = {s["id"]: s["type"] for s in manifest["sources"]}
    facts_index: dict[str, list[dict[str, str]]] = manifest.get("facts", {})

    for fact_row in truth_table["facts"]:
        fact_id = fact_row["id"]
        placements = facts_index.get(fact_id, [])
        placed_source_ids = {p["source_id"] for p in placements}

        if fact_row["type"] == "distractor":
            if placed_source_ids:
                problems.append(
                    f"{fact_id} is a distractor but has placements: {sorted(placed_source_ids)}"
                )
            continue

        expected_source_ids = {src["id"] for src in fact_row.get("sources", [])}
        if placed_source_ids != expected_source_ids:
            problems.append(
                f"{fact_id} expected placements in {sorted(expected_source_ids)}, "
                f"got {sorted(placed_source_ids)}"
            )

    for fact_id, placements in facts_index.items():
        for p in placements:
            source_type = source_type_by_id.get(p["source_id"])
            if source_type is None:
                problems.append(f"{fact_id} placement cites unknown source_id {p['source_id']!r}")
                continue
            pattern = LOCATOR_PATTERNS.get(source_type)
            if pattern is None:
                problems.append(f"no locator pattern registered for source type {source_type!r}")
                continue
            if not pattern.match(p["locator"]):
                problems.append(
                    f"{fact_id} on {p['source_id']}: locator {p['locator']!r} does not match "
                    f"the {source_type} format ({pattern.pattern})"
                )

    return problems
