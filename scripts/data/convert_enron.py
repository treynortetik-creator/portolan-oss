"""Convert the Enron email slice (`fetch_enron.py`'s `rows.jsonl`) into
Portolan's email shape (`portolan/connectors/file_connector.py`:
`email/*.txt`, one message per file, an RFC-2822-ish header block --
`Message-ID`, `From`, `To`, `Date`, `Subject` -- then a blank line, then
the body).

Deliberately does NOT clean up the source data's own messiness: real
Enron `From`/`To` fields are full of Outlook forward artifacts
("Phillip K Allen [mailto:...] Sent: ..."), Lotus Notes directory names
with no `@` in them ("Richard Burchfield/HOU/ECT@ECT"), and inline
"cc:" lists jammed into the To field. That messiness is exactly what
this corpus is for, so it passes through to the output files unchanged
except for two things the file connector actually requires to parse the
file at all:

  - `Date` is reformatted from Enron's original
    "Mon, 14 May 2001 16:39:00 -0700 (PDT)" into
    "2001-05-14 16:39:00 -0700" -- `file_connector.py` takes the date
    token before the first space and parses it with
    `date.fromisoformat`, which needs `YYYY-MM-DD` first, not a weekday
    name. Rows whose date string doesn't parse at all (~4% of the
    sample -- itself a form of real-world messiness) are skipped rather
    than guessed at.
  - `From` gets a display name synthesized from the email's local part
    ("phillip.allen@enron.com" -> "Phillip Allen <phillip.allen@enron.com>")
    only when the raw field is already a clean bare address; anything
    already containing a name, or too mangled to be a bare address, is
    left exactly as the source had it.

`Message-ID` doesn't exist in the parsed source columns, so one is
synthesized as `<enron-<row_idx>@mail.enron.com>` -- deterministic and
idempotent across re-runs, not a real header from the original message.

Usage:
    uv run python scripts/data/convert_enron.py [--raw PATH] [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import email.utils
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW = REPO_ROOT / "var/data/raw/enron/rows.jsonl"
DEFAULT_OUT_DIR = REPO_ROOT / "var/data/converted/enron"

BARE_EMAIL_RE = re.compile(r"^[^\s<>@]+@[^\s<>@]+$")


def humanize_local_part(email_addr: str) -> str:
    local = email_addr.split("@", 1)[0]
    words = re.split(r"[._+-]+", local)
    words = [w for w in words if w]
    return " ".join(w.capitalize() for w in words) if words else email_addr


def render_from(raw: str) -> str:
    raw = raw.strip()
    if not raw:
        return "Unknown Sender <unknown@enron.com>"
    if "<" in raw and ">" in raw:
        return raw  # already "Name <email>" or similar -- leave as-is
    if BARE_EMAIL_RE.match(raw):
        return f"{humanize_local_part(raw)} <{raw}>"
    return raw  # mangled Outlook/Notes artifact -- pass through unchanged


def render_date(raw: str) -> str | None:
    try:
        dt = email.utils.parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None
    if dt is None:
        return None
    offset = dt.strftime("%z") or "+0000"
    return dt.strftime("%Y-%m-%d %H:%M:%S ") + offset


def render_message(row: dict) -> str | None:
    date_header = render_date(row.get("date", ""))
    if date_header is None:
        return None
    message_id = f"enron-{row['row_idx']}@mail.enron.com"
    from_header = render_from(row.get("from", ""))
    to_header = (row.get("to") or "").strip()
    subject = (row.get("subject") or "").strip()
    body = (row.get("body") or "").strip("\n")

    lines = [
        f"Message-ID: <{message_id}>",
        f"From: {from_header}",
        f"To: {to_header}",
        f"Date: {date_header}",
        f"Subject: {subject}",
        "",
        body,
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    if not args.raw.exists():
        print(f"missing {args.raw} -- run fetch_enron.py first", file=sys.stderr)
        return 1

    email_dir = args.out_dir / "email"
    email_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped_bad_date = 0
    with args.raw.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            text = render_message(row)
            if text is None:
                skipped_bad_date += 1
                continue
            out_path = email_dir / f"S-email-enron-{row['row_idx']:06d}.txt"
            out_path.write_text(text, encoding="utf-8")
            written += 1

    print(f"wrote {written} emails to {email_dir}")
    print(f"skipped {skipped_bad_date} rows with unparseable Date")
    return 0


if __name__ == "__main__":
    sys.exit(main())
