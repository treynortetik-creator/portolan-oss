"""Fetch raw #ubuntu IRC support-channel logs from irclogs.ubuntu.com.

Ubuntu's own infrastructure has logged its public support channels since
2004 and published the logs at https://irclogs.ubuntu.com/. Per
`wiki.ubuntu.com/IRC/TermsOfService`, "by participating in one of these
channels you agree to have anything you send to the channel publicly
stored or processed on http://irclogs.ubuntu.com/"; the curated
`common-pile/ubuntu_irc` release built from the same source states
plainly that "Logs of all discussions on the Ubuntu-hosted Internet
Relay Chat (IRC) since 2004 have been archived and released into the
Public Domain." This is the same underlying source the (now much
harder to obtain in raw per-message form) academic "Ubuntu Dialogue
Corpus" was built from -- fetching directly here instead, because the
easily-reachable processed mirrors of that corpus either strip
timestamps/usernames (needed for the Slack-export shape this task
converts to) or pre-split messages into response-selection pairs.

Downloads one plain-text log per sampled date for channel `#ubuntu`
(`https://irclogs.ubuntu.com/<year>/<month>/<day>/%23ubuntu.txt`), each
a full day of that channel: `[HH:MM] <nick> message` lines plus IRC
system lines (`=== nick is now known as ...`, joins/parts). Dates are
sampled on the 15th of each month across a fixed range so the slice
spans several years of real support traffic rather than one narrow
window. Idempotent: an existing day file is not re-downloaded.

Usage:
    uv run python scripts/data/fetch_ubuntu_irc.py [--out-dir DIR] \
        [--start 2016-01] [--end 2018-06]
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

CHANNEL = "#ubuntu"
BASE = "https://irclogs.ubuntu.com"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = REPO_ROOT / "var/data/raw/ubuntu_irc"


def month_range(start: str, end: str) -> list[date]:
    sy, sm = (int(x) for x in start.split("-"))
    ey, em = (int(x) for x in end.split("-"))
    out = []
    y, m = sy, sm
    while (y, m) <= (ey, em):
        out.append(date(y, m, 15))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return out


def fetch_day(day: date, out_dir: Path) -> bool:
    out_path = out_dir / f"{day.isoformat()}.txt"
    if out_path.exists() and out_path.stat().st_size > 0:
        return False
    url = f"{BASE}/{day.year:04d}/{day.month:02d}/{day.day:02d}/%23ubuntu.txt"
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = resp.read()
            out_path.write_bytes(data)
            return True
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                print(f"  {day}: no log (404) -- skipping")
                return False
            time.sleep(2**attempt)
        except (urllib.error.URLError, TimeoutError):
            time.sleep(2**attempt)
    print(f"  {day}: failed after retries", file=sys.stderr)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--start", default="2016-01", help="YYYY-MM, first month (15th sampled)")
    parser.add_argument("--end", default="2018-06", help="YYYY-MM, last month (15th sampled)")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    days = month_range(args.start, args.end)
    print(f"sampling {len(days)} days from {args.start} to {args.end}")

    fetched = 0
    for day in days:
        if fetch_day(day, args.out_dir):
            fetched += 1
            print(f"  {day}: fetched")
            time.sleep(0.3)
        else:
            existing = args.out_dir / f"{day.isoformat()}.txt"
            if existing.exists():
                print(f"  {day}: already have it")

    files = sorted(args.out_dir.glob("*.txt"))
    total_bytes = sum(f.stat().st_size for f in files)
    print(
        f"fetched {fetched} new day-logs this run; {len(files)} files, {total_bytes:,} bytes total"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
