"""Convert FDA MAUDE device adverse-event records (`fetch_fda_maude.py`'s
`records.jsonl`) into Portolan's ticket/complaint shape
(`portolan/connectors/file_connector.py`: `tickets/*.json`, one file per
ticket, `{id, requester, account, created, subject, comments:
[{number, author, body}]}`).

MAUDE isn't a support ticket -- it's a single incident report -- but most
reports carry more than one `mdr_text` entry (the reporter's own
"Description of Event or Problem" narrative, plus zero or more
"Additional Manufacturer Narrative" follow-ups added later), and that is
genuinely thread-shaped: an initial complaint followed by manufacturer
replies. Each `mdr_text` entry becomes one comment, ordered by the
source's own `mdr_text_key` (its sequence field), with `author` set to a
title-cased version of the record's own `text_type_code` -- "Description
of Event or Problem" -> "Reporter", "Additional Manufacturer Narrative"
-> "Manufacturer", anything else passed through title-cased. Nothing is
invented: every field maps to one openFDA field.

Field mapping:
  - `id`      <- "S-ticket-fda-<mdr_report_key>"
  - `requester` <- `reporter_occupation_code`, title-cased ("OTHER" ->
    "Other"); falls back to "Reporter" for the handful of legacy numeric
    codes openFDA hasn't backfilled to text.
  - `account`   <- first device's `manufacturer_d_name`, else the
    top-level `manufacturer_name`, else "Unknown Manufacturer".
  - `created`   <- `date_received` (YYYYMMDD) reformatted to YYYY-MM-DD.
  - `subject`   <- "<generic_name or brand_name> -- <event_type>".
  - `comments`  <- one per `mdr_text` entry, see above.

Records with no usable `mdr_text` entries (shouldn't occur given the
fetch script's search filter, but checked defensively) are skipped.

Usage:
    uv run python scripts/data/convert_fda_maude.py [--raw PATH] [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW = REPO_ROOT / "var/data/raw/fda_maude/records.jsonl"
DEFAULT_OUT_DIR = REPO_ROOT / "var/data/converted/fda_maude"

AUTHOR_MAP = {
    "description of event or problem": "Reporter",
    "additional manufacturer narrative": "Manufacturer",
}


def render_author(text_type_code: str) -> str:
    key = (text_type_code or "").strip().lower()
    if key in AUTHOR_MAP:
        return AUTHOR_MAP[key]
    return (text_type_code or "Reporter").strip().title() or "Reporter"


def render_requester(code: str) -> str:
    code = (code or "").strip()
    if not code or code.isdigit():
        return "Reporter"
    return code.title()


def render_date(raw: str) -> str | None:
    raw = (raw or "").strip()
    if len(raw) != 8 or not raw.isdigit():
        return None
    return f"{raw[0:4]}-{raw[4:6]}-{raw[6:8]}"


def render_account(record: dict) -> str:
    for device in record.get("device") or []:
        name = (device.get("manufacturer_d_name") or "").strip()
        if name:
            return name
    name = (record.get("manufacturer_name") or "").strip()
    return name or "Unknown Manufacturer"


def render_subject(record: dict) -> str:
    device_name = "Unknown Device"
    for device in record.get("device") or []:
        generic = (device.get("generic_name") or "").strip()
        brand = (device.get("brand_name") or "").strip()
        device_name = generic or brand or device_name
        if device_name != "Unknown Device":
            break
    event_type = (record.get("event_type") or "").strip() or "Report"
    return f"{device_name} -- {event_type}"


def render_ticket(record: dict) -> dict | None:
    texts = record.get("mdr_text") or []
    if not texts:
        return None

    def sort_key(t: dict) -> int:
        try:
            return int(t.get("mdr_text_key", 0))
        except (TypeError, ValueError):
            return 0

    comments = []
    for i, t in enumerate(sorted(texts, key=sort_key), start=1):
        body = (t.get("text") or "").strip()
        if not body:
            continue
        comments.append(
            {"number": i, "author": render_author(t.get("text_type_code", "")), "body": body}
        )
    if not comments:
        return None

    created = render_date(record.get("date_received", ""))
    if created is None:
        return None

    report_key = record.get("mdr_report_key") or record.get("event_key") or ""
    return {
        "id": f"S-ticket-fda-{report_key}",
        "requester": render_requester(record.get("reporter_occupation_code", "")),
        "account": render_account(record),
        "created": created,
        "subject": render_subject(record),
        "comments": comments,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    if not args.raw.exists():
        print(f"missing {args.raw} -- run fetch_fda_maude.py first", file=sys.stderr)
        return 1

    tickets_dir = args.out_dir / "tickets"
    tickets_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    with args.raw.open(encoding="utf-8") as fh:
        for line in fh:
            record = json.loads(line)
            ticket = render_ticket(record)
            if ticket is None:
                skipped += 1
                continue
            out_path = tickets_dir / f"{ticket['id']}.json"
            out_path.write_text(json.dumps(ticket, indent=2, ensure_ascii=False), encoding="utf-8")
            written += 1

    print(f"wrote {written} tickets to {tickets_dir}")
    print(f"skipped {skipped} records with no usable text/date")
    return 0


if __name__ == "__main__":
    sys.exit(main())
