"""Fetch a slice of the Enron email corpus.

The Enron corpus is public record: the Federal Energy Regulatory
Commission made ~0.5M Enron staff emails public during its investigation
of the company, and CMU has mirrored successive cleaned releases since
2004 (https://www.cs.cmu.edu/~enron/, "I am distributing this dataset as
a resource for researchers..."). The raw CMU tarball is ~1.7 GB, far past
this task's 200 MB-per-dataset budget and this task only wants a few
thousand messages, not all 500k -- so this script instead pulls a slice
through the Hugging Face `datasets-server` "rows" API against
`Hellisotherpeople/enron_emails_parsed` (535,703 rows, already parsed
into `date`/`from`/`to`/`subject`/`body` columns from the same CMU
maildir release; no license tag on the HF repo card, but the underlying
data's public-record status is CMU's, not this repackaging's, to grant or
withhold). The API returns JSON a page at a time with no need to download
the repo's ~667 MB parquet file.

Rows are sampled at evenly spaced offsets across the full 535,703-row
table (not the first N) so the slice isn't just one or two custodians'
mailboxes back to back -- the table is grouped by sender.

Output: `var/data/raw/enron/rows.jsonl`, one JSON object per line
(`row_idx`, `date`, `from`, `to`, `subject`, `body`), plus
`fetched_offsets.json` recording which 100-row batches are already down
so a re-run resumes instead of re-fetching.

Usage:
    uv run python scripts/data/fetch_enron.py [--out-dir DIR] [--target-rows 3000]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DATASET = "Hellisotherpeople/enron_emails_parsed"
API = "https://datasets-server.huggingface.co/rows"
BATCH_SIZE = 100
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = REPO_ROOT / "var/data/raw/enron"


def fetch_batch(offset: int, length: int) -> list[dict]:
    url = (
        f"{API}?dataset={DATASET.replace('/', '%2F')}"
        f"&config=default&split=train&offset={offset}&length={length}"
    )
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                payload = json.loads(resp.read())
            return [r["row"] for r in payload.get("rows", [])]
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            wait = 2**attempt
            print(f"  offset {offset}: {exc} -- retrying in {wait}s", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"failed to fetch offset {offset} after retries")


def total_rows() -> int:
    url = f"https://datasets-server.huggingface.co/size?dataset={DATASET.replace('/', '%2F')}"
    with urllib.request.urlopen(url, timeout=30) as resp:
        payload = json.loads(resp.read())
    return payload["size"]["dataset"]["num_rows"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--target-rows", type=int, default=3000)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.out_dir / "rows.jsonl"
    state_path = args.out_dir / "fetched_offsets.json"

    done_offsets: set[int] = set()
    if state_path.exists():
        done_offsets = set(json.loads(state_path.read_text()))

    n_batches = max(1, args.target_rows // BATCH_SIZE)
    n_total = total_rows()
    stride = max(BATCH_SIZE, n_total // n_batches)
    offsets = [i * stride for i in range(n_batches) if i * stride < n_total]

    print(f"corpus has {n_total:,} rows; sampling {len(offsets)} batches of {BATCH_SIZE}")

    rows_out = rows_path.open("a", encoding="utf-8")
    fetched = 0
    try:
        for offset in offsets:
            if offset in done_offsets:
                continue
            rows = fetch_batch(offset, BATCH_SIZE)
            for i, row in enumerate(rows):
                row_out = {"row_idx": offset + i, **row}
                rows_out.write(json.dumps(row_out, ensure_ascii=False) + "\n")
            fetched += len(rows)
            done_offsets.add(offset)
            state_path.write_text(json.dumps(sorted(done_offsets)))
            print(f"  offset {offset}: +{len(rows)} rows")
    finally:
        rows_out.close()

    total_lines = sum(1 for _ in rows_path.open(encoding="utf-8"))
    print(f"fetched {fetched} new rows this run; {rows_path} has {total_lines} rows total")
    return 0


if __name__ == "__main__":
    sys.exit(main())
