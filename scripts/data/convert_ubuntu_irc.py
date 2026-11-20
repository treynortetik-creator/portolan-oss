"""Convert raw #ubuntu IRC day-logs (`fetch_ubuntu_irc.py`) into a
Slack-standard export layout, since no Slack connector exists in
Portolan yet (per the task brief): `channels.json`, `users.json`, and
`<channel>/<YYYY-MM-DD>.json` files of `{type, user, text, ts,
[thread_ts]}` message objects -- the shape Slack's own "export workspace"
feature produces.

Source line shapes (only three occur in this log format; the Ubuntu log
bot already filters out joins/parts/quits/topic changes):
  - `[HH:MM] <nick> message text`      -> a normal message
  - `[HH:MM]  * nick does something`   -> an IRC `/me` action -> Slack
    `subtype: "me_message"` (Slack's own convention: `text` is the
    action without the nick, e.g. "waves").
  - `=== oldnick is now known as newnick` (no timestamp of its own) ->
    dropped. A Slack export has no equivalent event (Slack identities
    don't get renamed mid-channel), and keeping it would need a
    fabricated timestamp; it carries no conversational content anyway.

`ts` is synthesized: each source line only carries HH:MM (no seconds),
so this script assigns the day's UTC midnight as the epoch base and
counts whole seconds forward one per message within an unchanged HH:MM
bucket, formatted as Slack does, "<unix_seconds>.<6-digit serial>" --
close enough to preserve message order and rough time-of-day, not a
literal recovered timestamp (the source never had second-level
resolution to recover).

`user` IDs are deterministic, `U` + the first 8 hex chars of
`sha1(nick)`, stable across every converted day so the same nick maps to
the same id throughout -- `users.json` is one row per distinct nick seen
across the whole converted slice, with `profile.real_name` set to the
nick (this corpus has no separate real-name field, only nicks).

Usage:
    uv run python scripts/data/convert_ubuntu_irc.py [--raw-dir DIR] [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import calendar
import hashlib
import json
import re
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = REPO_ROOT / "var/data/raw/ubuntu_irc"
DEFAULT_OUT_DIR = REPO_ROOT / "var/data/converted/ubuntu_irc"
CHANNEL_NAME = "ubuntu"

MSG_RE = re.compile(r"^\[(\d{2}):(\d{2})\] <([^>]+)> (.*)$")
ACTION_RE = re.compile(r"^\[(\d{2}):(\d{2})\]  \* (\S+) (.*)$")


def user_id(nick: str) -> str:
    return "U" + hashlib.sha1(nick.encode("utf-8")).hexdigest()[:8].upper()


def day_epoch(day: date) -> int:
    return calendar.timegm((day.year, day.month, day.day, 0, 0, 0, 0, 0, 0))


def convert_day(raw_path: Path, day: date) -> tuple[list[dict], set[str]]:
    base = day_epoch(day)
    messages: list[dict] = []
    nicks: set[str] = set()
    last_minute = None
    second_counter = 0

    for raw_line in raw_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = MSG_RE.match(raw_line)
        a = ACTION_RE.match(raw_line) if m is None else None
        if m is None and a is None:
            continue  # nick-change line or anything else not in the three known shapes

        if m:
            hh, mm, nick, text = m.groups()
            subtype = None
        else:
            hh, mm, nick, text = a.groups()
            subtype = "me_message"

        minute_key = (hh, mm)
        if minute_key != last_minute:
            second_counter = 0
            last_minute = minute_key
        else:
            second_counter += 1
        ts_seconds = base + int(hh) * 3600 + int(mm) * 60 + min(second_counter, 59)
        ts = f"{ts_seconds}.{second_counter:06d}"

        nicks.add(nick)
        msg = {"type": "message", "user": user_id(nick), "text": text, "ts": ts}
        if subtype:
            msg["subtype"] = subtype
        messages.append(msg)

    return messages, nicks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    raw_files = sorted(args.raw_dir.glob("*.txt"))
    if not raw_files:
        print(f"no raw files in {args.raw_dir} -- run fetch_ubuntu_irc.py first", file=sys.stderr)
        return 1

    export_dir = args.out_dir / "slack_export"
    channel_dir = export_dir / CHANNEL_NAME
    channel_dir.mkdir(parents=True, exist_ok=True)

    all_nicks: set[str] = set()
    total_messages = 0
    days_written = 0
    for raw_path in raw_files:
        day = date.fromisoformat(raw_path.stem)
        messages, nicks = convert_day(raw_path, day)
        if not messages:
            continue
        out_path = channel_dir / f"{day.isoformat()}.json"
        out_path.write_text(json.dumps(messages, indent=2, ensure_ascii=False), encoding="utf-8")
        all_nicks |= nicks
        total_messages += len(messages)
        days_written += 1

    earliest = date.fromisoformat(raw_files[0].stem)
    channels = [
        {
            "id": "C" + hashlib.sha1(CHANNEL_NAME.encode()).hexdigest()[:8].upper(),
            "name": CHANNEL_NAME,
            "created": day_epoch(earliest),
            "purpose": {
                "value": "Ubuntu community support channel (irclogs.ubuntu.com, public domain)"
            },
        }
    ]
    (export_dir / "channels.json").write_text(json.dumps(channels, indent=2), encoding="utf-8")

    users = [
        {"id": user_id(nick), "name": nick, "profile": {"real_name": nick}}
        for nick in sorted(all_nicks)
    ]
    (export_dir / "users.json").write_text(
        json.dumps(users, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"wrote {days_written} day-files, {total_messages} messages, {len(users)} users")
    print(f"output: {export_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
