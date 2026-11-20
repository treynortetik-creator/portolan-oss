"""Convert a slice of the AMI Meeting Corpus manual annotations into
Portolan's call-transcript shape (`portolan/connectors/file_connector.py`:
`calls/*.txt`, one file per meeting, `[HH:MM:SS] Speaker: text` lines).

Reads directly out of `ami_public_manual_1.6.2.zip` (no extraction to
disk) -- for each selected meeting it parses that speaker's
`words/<meeting>.<letter>.words.xml` (word-level tokens, punctuation
flags, `<vocalsound>` and `<disfmarker>` disfluency events) and
`segments/<meeting>.<letter>.segments.xml` (turn-level start/end times
that reference ranges of word ids), renders each segment to text, then
interleaves every speaker's segments by start time into one chronological
transcript per meeting.

Speakers are labeled "Speaker A".."Speaker D" -- the AMI meeting-role
letters from the file names, not real names (the manual annotation
release does not carry a reliable global per-meeting name field; several
speakers do introduce themselves by first name inside the transcript
text itself, which this script does not alter).

Disfluency rendering:
  - `<vocalsound type="laugh">` -> "[laughs]" (similarly cough/sniff/yawn/
    throat-clear); any other vocalsound type -> "[noise]".
  - `<disfmarker>` (a marked restart/false-start point, no text of its
    own) -> "--".
  - `<w punc="true">` attaches directly to the previous token (no
    leading space), matching how the AMI annotation itself marks
    punctuation as not a separate word.

Usage:
    uv run python scripts/data/convert_ami.py [--zip PATH] [--out-dir DIR] [--limit N]

`--limit` selects an evenly spaced sample of N meetings (default 40) out
of the corpus's 171 annotated meetings, so a full run stays fast and the
output stays a "manageable slice" rather than the whole corpus.
"""

from __future__ import annotations

import argparse
import re
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ZIP = REPO_ROOT / "var/data/raw/ami/ami_public_manual_1.6.2.zip"
DEFAULT_OUT_DIR = REPO_ROOT / "var/data/converted/ami"

NITE_ID = "{http://nite.sourceforge.net/}id"
WORD_ID_RE = re.compile(r"(\d+)$")
CHILD_ID_RE = re.compile(r"#id\([^)]*words(\d+)\)(?:\.\.id\([^)]*words(\d+)\))?")
SPEAKER_LETTERS = ("A", "B", "C", "D")

VOCAL_MAP = {
    "laugh": "[laughs]",
    "cough": "[coughs]",
    "sniff": "[sniffs]",
    "yawn": "[yawns]",
    "throat": "[clears throat]",
}


def vocal_text(vtype: str | None) -> str:
    if not vtype:
        return "[noise]"
    key = vtype.split()[0].lower()
    return VOCAL_MAP.get(key, "[noise]")


def parse_words(raw: bytes) -> dict[int, dict]:
    """id(int) -> {"text": str, "punc": bool, "bracket": bool}"""
    root = ET.fromstring(raw)
    out: dict[int, dict] = {}
    for el in root:
        m = WORD_ID_RE.search(el.get(NITE_ID, ""))
        if not m:
            continue
        idx = int(m.group(1))
        tag = el.tag
        if tag == "w":
            out[idx] = {"text": el.text or "", "punc": el.get("punc") == "true", "bracket": False}
        elif tag == "vocalsound":
            out[idx] = {"text": vocal_text(el.get("type")), "punc": False, "bracket": True}
        elif tag == "disfmarker":
            out[idx] = {"text": "--", "punc": False, "bracket": True}
        # other rare NXT tags (e.g. gap/pause markers) carry no useful
        # rendered text and are skipped.
    return out


def parse_segments(raw: bytes) -> list[dict]:
    root = ET.fromstring(raw)
    segments = []
    for seg in root.findall("segment"):
        try:
            start_t = float(seg.get("transcriber_start", "0") or 0)
        except ValueError:
            start_t = 0.0
        word_ids: list[int] = []
        for child in seg:
            href = child.get("href", "")
            m = CHILD_ID_RE.search(href)
            if not m:
                continue
            lo = int(m.group(1))
            hi = int(m.group(2)) if m.group(2) else lo
            word_ids.extend(range(lo, hi + 1))
        if word_ids:
            segments.append({"start": start_t, "word_ids": word_ids})
    return segments


def render_segment(word_ids: list[int], words: dict[int, dict]) -> str:
    parts: list[str] = []
    for wid in word_ids:
        tok = words.get(wid)
        if tok is None or not tok["text"]:
            continue
        if tok["punc"]:
            parts.append(tok["text"])
        else:
            parts.append(" " + tok["text"])
    return "".join(parts).strip()


def fmt_time(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def meeting_ids(names: set[str]) -> list[str]:
    ids = set()
    for name in names:
        m = re.match(r"^words/([A-Za-z0-9]+)\.[A-D]\.words\.xml$", name)
        if m:
            ids.add(m.group(1))
    return sorted(ids)


def select_meetings(all_ids: list[str], limit: int) -> list[str]:
    if limit <= 0 or limit >= len(all_ids):
        return all_ids
    stride = len(all_ids) / limit
    selected = []
    seen = set()
    i = 0.0
    while len(selected) < limit and int(i) < len(all_ids):
        mid = all_ids[int(i)]
        if mid not in seen:
            selected.append(mid)
            seen.add(mid)
        i += stride
    return selected


def convert_meeting(zf: zipfile.ZipFile, names: set[str], meeting_id: str) -> str | None:
    turns: list[tuple[float, str, str]] = []
    for letter in SPEAKER_LETTERS:
        wpath = f"words/{meeting_id}.{letter}.words.xml"
        spath = f"segments/{meeting_id}.{letter}.segments.xml"
        if wpath not in names or spath not in names:
            continue
        words = parse_words(zf.read(wpath))
        for seg in parse_segments(zf.read(spath)):
            text = render_segment(seg["word_ids"], words)
            if text:
                turns.append((seg["start"], letter, text))
    if not turns:
        return None
    turns.sort(key=lambda t: t[0])
    lines = [f"[{fmt_time(start)}] Speaker {letter}: {text}" for start, letter, text in turns]
    return "\n\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--limit", type=int, default=40)
    args = parser.parse_args()

    if not args.zip.exists():
        print(f"missing {args.zip} -- run fetch_ami.py first", file=sys.stderr)
        return 1

    calls_dir = args.out_dir / "calls"
    calls_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(args.zip) as zf:
        names = set(zf.namelist())
        all_ids = meeting_ids(names)
        selected = select_meetings(all_ids, args.limit)
        print(f"{len(all_ids)} meetings in corpus, converting {len(selected)}")

        written = 0
        for meeting_id in selected:
            text = convert_meeting(zf, names, meeting_id)
            if text is None:
                print(f"  skip {meeting_id}: no segments found")
                continue
            out_path = calls_dir / f"{meeting_id}.txt"
            out_path.write_text(text, encoding="utf-8")
            written += 1

    print(f"wrote {written} call transcripts to {calls_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
