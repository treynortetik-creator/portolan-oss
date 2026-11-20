"""Convert synthetic B2B SaaS sales conversations (`fetch_saas.py`'s
`records.jsonl`) into Portolan's call-transcript shape
(`portolan/connectors/file_connector.py`: `calls/*.txt`, one file per
conversation, `[HH:MM:SS] Speaker: text` lines -- the same shape
`convert_ami.py` produces).

**Account.** A raw `calls/*.txt` file has no field of its own for
"account" -- `FileConnector` only ever gets a call's account from the
*optional* `manifest.json` next to the calls directory
(`_account_hints()` -> `manifest.json`'s per-source `accounts` list; see
that module's docstring). So this script also writes
`var/data/converted/saas_sales/manifest.json`, one `{"id":
"S-call-saas-<id>", "accounts": ["<buyer company>"]}` entry per
conversation -- the buyer's own `company_name` field when present (it was
present on all 2,000 fetched rows; not one hit the "no company name"
case), else a stable `synthetic-account-<company_id>` id built from the
row's own `company_id` (stable across every conversation for the same
buyer, never invented), else (`company_id` missing too) a last-resort
`synthetic-account-<conversation_id>`. No `date` entry is written -- this
synthetic dataset carries no date field of any kind, so (same as
`convert_ami.py`'s own calls, which also write no manifest date)
`FileConnector._call_date` falls back to "today" for every one of these
sources, an accepted, pre-existing pattern, not new to this corpus.

**Speaker labels.** `conversation` is a JSON array of `{"speaker",
"message"}` turns. Three speaker keys occur in the fetched sample:
`customer` -> "Customer", `sales_rep` -> "Sales Rep", and a rare `system`
-> "System" (4/2000 rows in a 200-row spot check; the scenario's own
`conversation_flow` field marks these as `multi_session` conversations,
and the `system` turn is always the literal text "--- a week later ---",
a session-boundary marker, not a person). All three get the same
`[HH:MM:SS] Speaker: text` line shape so `FileConnector`'s speaker regex
matches uniformly; "System" is left as a real speaker line rather than
silently dropped, since it is genuine structure in the source (skipping
it would look like one continuous same-day conversation instead of the
multi-session one the scenario data says it is).

**Timestamps.** The source has no timestamps at all (no per-turn time,
no call date) -- `[HH:MM:SS]` is synthesized as a flat 30-second-per-turn
clock starting at `00:00:00`, the same kind of honest synthesis
`fetch_ubuntu_irc.py`'s per-minute serial counter is (preserves turn
order and gives every line a valid, strictly increasing timestamp; is
not a claim about real call pacing or duration).

Field mapping:
  - filename  <- "S-call-saas-<id>.txt", `<id>` = `conversation_id` with
    its redundant leading "saas-" stripped (`conversation_id` is already
    "saas-<company-num>-conv-<n>"; stripping it avoids "saas" appearing
    twice in the path while staying uniquely traceable back to the
    source row -- the full original `conversation_id` is also kept in
    `manifest.json` would be redundant with the id itself, so it isn't
    duplicated there).
  - speaker lines <- `conversation` (see above).
  - `manifest.json` `accounts` <- `company_name` / `company_id` (see
    above).

Usage:
    uv run python scripts/data/convert_saas.py [--raw PATH] [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW = REPO_ROOT / "var/data/raw/saas_sales/records.jsonl"
DEFAULT_OUT_DIR = REPO_ROOT / "var/data/converted/saas_sales"

STEP_SECONDS = 30

SPEAKER_LABELS = {
    "customer": "Customer",
    "sales_rep": "Sales Rep",
    "system": "System",
}


def fmt_time(seconds: int) -> str:
    seconds = seconds % 86400
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def render_speaker(key: str) -> str:
    if key in SPEAKER_LABELS:
        return SPEAKER_LABELS[key]
    return (key or "Unknown").replace("_", " ").title() or "Unknown"


def render_call(record: dict) -> str | None:
    try:
        turns = json.loads(record.get("conversation") or "[]")
    except json.JSONDecodeError:
        return None
    if not isinstance(turns, list):
        return None

    lines: list[str] = []
    t = 0
    for turn in turns:
        text = " ".join((turn.get("message") or "").split())
        if not text:
            continue
        speaker = render_speaker(turn.get("speaker", ""))
        lines.append(f"[{fmt_time(t)}] {speaker}: {text}")
        t += STEP_SECONDS

    if not lines:
        return None
    return "\n\n".join(lines) + "\n"


def render_account(record: dict) -> str:
    name = (record.get("company_name") or "").strip()
    if name:
        return name
    company_id = (record.get("company_id") or "").strip()
    if company_id:
        return f"synthetic-account-{company_id}"
    return f"synthetic-account-{record.get('conversation_id', 'unknown')}"


def short_id(conversation_id: str) -> str:
    prefix = "saas-"
    if conversation_id.startswith(prefix):
        return conversation_id[len(prefix) :]
    return conversation_id


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    if not args.raw.exists():
        print(f"missing {args.raw} -- run fetch_saas.py first", file=sys.stderr)
        return 1

    calls_dir = args.out_dir / "calls"
    calls_dir.mkdir(parents=True, exist_ok=True)

    manifest_sources: list[dict] = []
    written = 0
    skipped = 0
    with args.raw.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            text = render_call(record)
            if text is None:
                skipped += 1
                continue
            conversation_id = record.get("conversation_id")
            if not conversation_id:
                skipped += 1
                continue
            source_id = f"S-call-saas-{short_id(conversation_id)}"
            out_path = calls_dir / f"{source_id}.txt"
            out_path.write_text(text, encoding="utf-8")
            manifest_sources.append({"id": source_id, "accounts": [render_account(record)]})
            written += 1

    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps({"sources": manifest_sources}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"wrote {written} call transcripts to {calls_dir}")
    print(f"wrote {len(manifest_sources)}-entry manifest to {manifest_path}")
    print(f"skipped {skipped} records with no usable conversation/id")
    return 0


if __name__ == "__main__":
    sys.exit(main())
