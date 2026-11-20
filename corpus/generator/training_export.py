"""Turn a finished `synth` batch into JSONL training records.

Reads `<batch_dir>/manifest.json` + `<batch_dir>/truth-table.yaml` (never
touches `.synth_progress.json` -- the internal resumability sidecar,
which may or may not still be on disk) plus the batch's own written
source files, and writes one JSON object per line to
`var/data/silver/synthetic/<batch-id>.jsonl` (or wherever `--out` points),
matching exactly the shape the silver-label pipeline (built in parallel,
a different agent's `portolan/training/records.py`) also produces:

    {"record_id": str, "corpus": "synthetic", "source_id": str,
     "kind": "ticket|call|email|chat", "license": "synthetic-owned",
     "text": str, "facts": [{"fact_type": str, "claim": str, "quote": str,
     "locator": str, "entities": [str]}], "teachers": [str, ...],
     "agreement": 1.0, "created": str}

One exported record per DOCUMENT:
  - call / ticket / email: one manifest `SourceRecord` == one document
    (the file it was written to).
  - slack / teams (`kind: "chat"`): manifest records multiple
    `SourceRecord`s per (channel, date) day-file, one per original
    message -- this module groups every `SourceRecord` sharing a `path`
    back into ONE exported document (the whole day's channel transcript),
    `source_id` = that file's relative path.
  - crm (structured rows and activity notes): never exported -- `kind`
    only ever takes `ticket|call|email|chat`.

Only VERIFIED planted facts are exported (`SourceRecord.facts` already
holds only verified placements -- `cli._synth_generate_unit` falls back to
`facts=[]` filler on a verification failure). A document (or, for chat, a
whole grouped day-file) is exported with `facts: []` only when every
source in it was genuinely intended to carry none (`is_noise=True` on
every one) -- a true negative. A document where a fact was INTENDED
(`is_noise=False`) but verification failed (so `facts` came back empty
anyway) is skipped entirely, not exported as a false true-negative.

`quote` is always re-derived from the document's own rendered `text` (the
exact line / ticket-comment body / email paragraph / chat message the
fact's locator points at), never taken verbatim from the manifest's own
(whitespace-collapsed, 160-char-truncated) `excerpt` field, so `quote in
text` always holds -- asserted per fact.

Usage:
    uv run python -m corpus.generator.training_export --batch var/data/synth/<batch-id>
    uv run python -m corpus.generator.training_export --batch var/data/synth/<batch-id> --out some/other.jsonl
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path
from typing import Any

import yaml

from corpus.generator import markers

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SILVER_DIR = PROJECT_ROOT / "var" / "data" / "silver" / "synthetic"

KIND_BY_SOURCE_TYPE: dict[str, str] = {
    "call": "call",
    "ticket": "ticket",
    "email": "email",
    "slack": "chat",
    "teams": "chat",
}


def default_export_path(batch_dir: Path) -> Path:
    return DEFAULT_SILVER_DIR / f"{batch_dir.name}.jsonl"


def _load_manifest(batch_dir: Path) -> dict[str, Any]:
    return json.loads((batch_dir / "manifest.json").read_text(encoding="utf-8"))


def _load_truth_table(batch_dir: Path) -> dict[str, Any]:
    with (batch_dir / "truth-table.yaml").open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _facts_by_id(truth: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {f["id"]: f for f in truth["facts"]}


def _should_export(records: list[dict[str, Any]]) -> tuple[bool, bool]:
    """`(should_export, is_true_negative)` for one document (one source
    record, or a grouped list of chat message records sharing a file).

    - Any record with verified facts -> export the whole document (its
      other, unrelated sources are just surrounding context).
    - No verified facts anywhere, but every record was genuinely intended
      to carry none (`is_noise`) -> a true negative, export with an empty
      `facts` list.
    - Otherwise: at least one record intended a fact that failed
      verification and fell back to filler -- skip, never a false
      true-negative.
    """
    if any(r["facts"] for r in records):
        return True, False
    all_noise = all(r["is_noise"] for r in records)
    return all_noise, all_noise


def _build_fact_entries(
    placements: list[dict[str, Any]],
    facts_by_id: dict[str, dict[str, Any]],
    quotes: dict[str, str],
    text: str,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for p in placements:
        fact_row = facts_by_id[p["fact_id"]]
        quote = quotes.get(p["fact_id"], p["excerpt"])
        assert quote in text, (
            f"quote for {p['fact_id']!r} is not an exact substring of the document text: {quote!r}"
        )
        entries.append(
            {
                "fact_type": fact_row["type"],
                "claim": fact_row["claim"],
                "quote": quote,
                "locator": p["locator"],
                "entities": list(fact_row.get("entities", [])),
            }
        )
    return entries


# --------------------------------------------------------------------------
# Per-kind text + quote reconstruction. Each returns (text, quote_by_fact_id)
# -- `quote` always a literal substring of `text` by construction (either
# read straight off disk, for call/email, or built by this module itself
# from the same parsed fields it quotes from, for ticket/chat, so there is
# no JSON-escaping or whitespace-collapse mismatch to worry about).
# --------------------------------------------------------------------------


def _call_text_and_quotes(batch_dir: Path, source: dict[str, Any]) -> tuple[str, dict[str, str]]:
    text = (batch_dir / source["path"]).read_text(encoding="utf-8")
    lines = text.splitlines()
    quotes: dict[str, str] = {}
    for f in source["facts"]:
        n = int(f["line_or_comment"])
        line = lines[n - 1] if 1 <= n <= len(lines) else None
        if line is None:
            quotes[f["fact_id"]] = f["excerpt"]
            continue
        m = markers.CALL_LINE_RE.match(line)
        quotes[f["fact_id"]] = m.group(3) if m else line
    return text, quotes


def _ticket_text_and_quotes(batch_dir: Path, source: dict[str, Any]) -> tuple[str, dict[str, str]]:
    payload = json.loads((batch_dir / source["path"]).read_text(encoding="utf-8"))
    lines = [
        f"Subject: {payload['subject']}",
        f"Requester: {payload['requester']} | Account: {payload['account']} | "
        f"Created: {payload['created']}",
        "",
    ]
    comment_bodies: dict[int, str] = {}
    for c in payload["comments"]:
        lines.append(f"[{c['number']}] {c['author']}: {c['body']}")
        comment_bodies[c["number"]] = c["body"]
    text = "\n".join(lines)
    quotes = {}
    for f in source["facts"]:
        n = int(f["line_or_comment"])
        quotes[f["fact_id"]] = comment_bodies.get(n, f["excerpt"])
    return text, quotes


def _email_text_and_quotes(batch_dir: Path, source: dict[str, Any]) -> tuple[str, dict[str, str]]:
    text = (batch_dir / source["path"]).read_text(encoding="utf-8")
    _header, _, body = text.partition("\n\n")
    paragraphs = body.split("\n\n")
    quotes = {}
    for f in source["facts"]:
        n = int(f["line_or_comment"])
        if 1 <= n <= len(paragraphs):
            quotes[f["fact_id"]] = paragraphs[n - 1].rstrip("\n")
        else:
            quotes[f["fact_id"]] = f["excerpt"]
    return text, quotes


def _group_chat_sources(
    sources: list[dict[str, Any]],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for s in sources:
        if s["type"] in ("slack", "teams"):
            groups.setdefault((s["type"], s["path"]), []).append(s)
    return groups


def _chat_group_text_and_quotes(
    batch_dir: Path,
    path: str,
    kind: str,
    name_by_id: dict[str, str],
    group: list[dict[str, Any]],
) -> tuple[str, dict[str, str]]:
    messages = json.loads((batch_dir / path).read_text(encoding="utf-8"))
    lines: list[str] = []
    text_by_index: dict[int, str] = {}
    for i, m in enumerate(messages, start=1):
        if kind == "slack":
            speaker = name_by_id.get(m["user"], m["user"])
            msg_text = m["text"]
        else:
            speaker = m["from"]["user"]["displayName"]
            msg_text = m["body"]["content"]
        lines.append(f"{speaker}: {msg_text}")
        text_by_index[i] = msg_text
    text = "\n".join(lines)
    quotes: dict[str, str] = {}
    for source in group:
        for f in source["facts"]:
            n = int(f["line_or_comment"])
            quotes[f["fact_id"]] = text_by_index.get(n, f["excerpt"])
    return text, quotes


def _slack_name_by_id(batch_dir: Path) -> dict[str, str]:
    users_path = batch_dir / "slack" / "users.json"
    if not users_path.exists():
        return {}
    users = json.loads(users_path.read_text(encoding="utf-8"))
    return {u["id"]: u["real_name"] for u in users}


def _teachers_for(records: list[dict[str, Any]]) -> list[str]:
    return sorted({r["model"] for r in records if r.get("model")})


def export_batch(batch_dir: Path, out_path: Path | None = None) -> list[dict[str, Any]]:
    """Export `batch_dir` (a finished `synth` batch directory) to a list of
    training records, sorted by `record_id`. Also writes them as JSONL to
    `out_path` (default: `default_export_path(batch_dir)`) when given (pass
    `out_path=False`-ish -- actually just omit `out_path` to still get the
    default write; there is no "compute only" mode via this signature, use
    the returned list directly if a write is not wanted)."""
    manifest = _load_manifest(batch_dir)
    truth = _load_truth_table(batch_dir)
    facts_by_id = _facts_by_id(truth)
    batch_id = batch_dir.name
    created = manifest.get("generated_at") or _dt.datetime.now(_dt.UTC).isoformat()

    sources_by_type: dict[str, list[dict[str, Any]]] = {}
    for s in manifest["sources"]:
        sources_by_type.setdefault(s["type"], []).append(s)

    records: list[dict[str, Any]] = []

    for source_type, builder in (
        ("call", _call_text_and_quotes),
        ("ticket", _ticket_text_and_quotes),
        ("email", _email_text_and_quotes),
    ):
        for source in sources_by_type.get(source_type, []):
            should_export, _true_negative = _should_export([source])
            if not should_export:
                continue
            text, quotes = builder(batch_dir, source)
            fact_entries = _build_fact_entries(source["facts"], facts_by_id, quotes, text)
            records.append(
                {
                    "record_id": f"{batch_id}:{source['id']}",
                    "corpus": "synthetic",
                    "source_id": source["id"],
                    "kind": KIND_BY_SOURCE_TYPE[source_type],
                    "license": "synthetic-owned",
                    "text": text,
                    "facts": fact_entries,
                    "teachers": _teachers_for([source]),
                    "agreement": 1.0,
                    "created": created,
                }
            )

    name_by_id = _slack_name_by_id(batch_dir)
    for (source_type, path), group in sorted(_group_chat_sources(manifest["sources"]).items()):
        should_export, _true_negative = _should_export(group)
        if not should_export:
            continue
        text, quotes = _chat_group_text_and_quotes(batch_dir, path, source_type, name_by_id, group)
        all_facts = [f for source in group for f in source["facts"]]
        fact_entries = _build_fact_entries(all_facts, facts_by_id, quotes, text)
        records.append(
            {
                "record_id": f"{batch_id}:{path}",
                "corpus": "synthetic",
                "source_id": path,
                "kind": "chat",
                "license": "synthetic-owned",
                "text": text,
                "facts": fact_entries,
                "teachers": _teachers_for(group),
                "agreement": 1.0,
                "created": created,
            }
        )

    records.sort(key=lambda r: r["record_id"])

    target = out_path if out_path is not None else default_export_path(batch_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False))
            fh.write("\n")

    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m corpus.generator.training_export")
    parser.add_argument("--batch", required=True, help="Path to a finished synth batch directory.")
    parser.add_argument(
        "--out",
        default=None,
        help="Output .jsonl path (default: var/data/silver/synthetic/<batch-id>.jsonl)",
    )
    args = parser.parse_args(argv)

    batch_dir = Path(args.batch)
    out_path = Path(args.out) if args.out else None
    records = export_batch(batch_dir, out_path)
    target = out_path if out_path is not None else default_export_path(batch_dir)

    by_type: dict[str, int] = {}
    by_fact_type: dict[str, int] = {}
    total_facts = 0
    for r in records:
        by_type[r["kind"]] = by_type.get(r["kind"], 0) + 1
        for f in r["facts"]:
            total_facts += 1
            by_fact_type[f["fact_type"]] = by_fact_type.get(f["fact_type"], 0) + 1

    print(f"batch: {batch_dir}")
    print(f"exported: {target}")
    print(f"records: {len(records)} by kind: {by_type}")
    print(f"facts: {total_facts} by type: {by_fact_type}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
