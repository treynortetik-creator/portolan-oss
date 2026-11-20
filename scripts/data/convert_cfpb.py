"""Convert CFPB consumer complaints (`fetch_cfpb.py`'s `records.jsonl`)
into Portolan's ticket shape (`portolan/connectors/file_connector.py`:
`tickets/*.json`, one file per ticket, `{id, requester, account, created,
subject, comments: [{number, author, body}]}`).

Field mapping (nothing invented -- every field maps to one CFPB field
from `fetch_cfpb.py`'s `records.jsonl`):
  - `id`        <- "S-ticket-cfpb-<complaint_id>"
  - `requester` <- the literal string "consumer" (per task brief -- CFPB
    complaints have no named consumer, just "a consumer" throughout).
  - `account`   <- `company` (the company the complaint concerns).
  - `created`   <- `date_received` (already `YYYY-MM-DD`, CFPB's own
    format -- no reformatting needed).
  - `subject`   <- `product` (CFPB's own taxonomy value, e.g. "Debt
    collection" -- the closest thing to a ticket subject line this
    dataset has; `sub_product`/`issue` were not fetched -- see
    `fetch_cfpb.py`'s `COLUMNS`).
  - `comments`  <- comment 1 is always the consumer's own narrative
    (`complaint_what_happened`), author "Consumer". Comment 2, only when
    `company_public_response` is non-empty, is the company's own canned
    public-response category text (CFPB's fixed ~11-value vocabulary,
    e.g. "Company believes it acted appropriately as authorized by
    contract or law" -- never freeform, but it is a real second
    company-authored statement on the ticket, and ~54% of the fetched
    sample carries one), author `company` (the same value used for
    `account`, matching how a real support ticket's own company would
    reply on its own ticket).

Usage:
    uv run python scripts/data/convert_cfpb.py [--raw PATH] [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW = REPO_ROOT / "var/data/raw/cfpb/records.jsonl"
DEFAULT_OUT_DIR = REPO_ROOT / "var/data/converted/cfpb"


def render_ticket(record: dict) -> dict | None:
    narrative = (record.get("complaint_what_happened") or "").strip()
    if not narrative:
        return None
    complaint_id = record.get("complaint_id")
    if not complaint_id:
        return None
    created = (record.get("date_received") or "").strip()
    if not created:
        return None

    company = (record.get("company") or "Unknown Company").strip() or "Unknown Company"

    comments = [{"number": 1, "author": "Consumer", "body": narrative}]
    company_response = (record.get("company_public_response") or "").strip()
    if company_response:
        comments.append({"number": 2, "author": company, "body": company_response})

    return {
        "id": f"S-ticket-cfpb-{complaint_id}",
        "requester": "consumer",
        "account": company,
        "created": created,
        "subject": (record.get("product") or "Complaint").strip() or "Complaint",
        "comments": comments,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    if not args.raw.exists():
        print(f"missing {args.raw} -- run fetch_cfpb.py first", file=sys.stderr)
        return 1

    tickets_dir = args.out_dir / "tickets"
    tickets_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    with args.raw.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            ticket = render_ticket(record)
            if ticket is None:
                skipped += 1
                continue
            out_path = tickets_dir / f"{ticket['id']}.json"
            out_path.write_text(json.dumps(ticket, indent=2, ensure_ascii=False), encoding="utf-8")
            written += 1

    print(f"wrote {written} tickets to {tickets_dir}")
    print(f"skipped {skipped} records with no usable narrative/id/date")
    return 0


if __name__ == "__main__":
    sys.exit(main())
