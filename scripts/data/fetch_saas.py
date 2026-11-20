"""Fetch synthetic B2B SaaS sales conversations
(`DeepMostInnovations/saas-sales-conversations` on Hugging Face --
`01-spec/research/free-data-and-open-models-2026-09-22.md`, Track A:
"Apache 2.0 ... 100,000 rows, GPT-4-generated").

**License check (task brief: "verify the license file says Apache-2.0
before converting, and stop and report if it does not").** This repo has
no separate `LICENSE` file -- the license declaration lives in the
dataset card's own YAML front matter in `README.md`
(`https://huggingface.co/datasets/DeepMostInnovations/saas-sales-
conversations/raw/main/README.md`), which is the standard place a
Hugging Face dataset declares its license. `check_license()` below fetches
that file fresh every run and asserts the `license:` line reads
`apache-2.0` -- if it doesn't (or the file/field is missing), the script
prints an error and exits 1 rather than converting anything, per the
brief.

**Column pruning, not `datasets.load_dataset`.** The repo's one data file
(`cleaned_custom_dataset.csv`) carries ~1,536 `embedding_<N>` float
columns per row alongside the fields this task needs -- that pushes the
auto-converted parquet to ~2.95 GB across 5 shards (20,000 rows each) for
what is otherwise a fairly small text dataset. Reading the full CSV or
calling `load_dataset(...)` without column selection would pull all of
that. This script instead reads every shard directly with `pyarrow` (same
column-pruning approach as `fetch_cfpb.py`), naming only the columns this
task's conversion needs -- no embedding column is ever fetched.

**Known quirk, checked before deciding how to sample:** the whole
100,000-row dataset only has **20 distinct buyer companies**
(`company_name`/`company_id`) -- confirmed by scanning `company_name`
alone across all 5 shards (5, 6, 6, 6, 5 unique names per shard) before
writing this script. Each company contributes ~5,000 conversations
(different personas/scenarios/reps against the same buyer). Reading only
shard 0 would have meant just 4-5 of the 20 companies in the sample --
this script instead reads all 5 shards and takes an evenly spaced stride
across the full 100,000-row pool (`select_evenly`, same pattern
`scripts/data/convert_ami.py`'s `select_meetings` uses), so the 2,000-row
sample sees close to the full 20-company roster instead of a handful of
them repeated.

Usage:
    uv run python scripts/data/fetch_saas.py [--out-dir DIR] [--target-rows 2000]
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import fsspec
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = REPO_ROOT / "var/data/raw/saas_sales"

README_URL = (
    "https://huggingface.co/datasets/DeepMostInnovations/saas-sales-conversations"
    "/raw/main/README.md"
)
SHARD_URL_TEMPLATE = (
    "https://huggingface.co/datasets/DeepMostInnovations/saas-sales-conversations"
    "/resolve/refs%2Fconvert%2Fparquet/default/train/{:04d}.parquet"
)
N_SHARDS = 5
COLUMNS = [
    "conversation_id",
    "company_id",
    "company_name",
    "product_name",
    "product_type",
    "communication_channel",
    "conversation_length",
    "outcome",
    "conversation",
]
REQUIRED_LICENSE = "apache-2.0"


def parse_license_from_front_matter(text: str, *, source: str = "<readme>") -> str:
    """Pure parser (no network): return the `license:` value from a
    dataset card's YAML front matter. Raises `RuntimeError` if the text
    has no front matter or no `license:` field -- never assumes a
    license. Split out from `check_license()` below so it's testable
    with a fake string, no fetch required."""
    if not text.startswith("---"):
        raise RuntimeError(f"{source} has no YAML front matter -- cannot verify license")
    front_matter = text.split("---", 2)[1]
    for line in front_matter.splitlines():
        line = line.strip()
        if line.startswith("license:"):
            return line.split(":", 1)[1].strip()
    raise RuntimeError(f"no 'license:' field found in {source}'s front matter")


def check_license() -> str:
    """Fetch the dataset card's YAML front matter and return the
    `license:` value (see `parse_license_from_front_matter`)."""
    with urllib.request.urlopen(README_URL, timeout=30) as resp:
        text = resp.read().decode("utf-8", errors="replace")
    return parse_license_from_front_matter(text, source=README_URL)


def select_evenly(n_total: int, limit: int) -> list[int]:
    if limit <= 0 or limit >= n_total:
        return list(range(n_total))
    stride = n_total / limit
    indices: list[int] = []
    seen: set[int] = set()
    i = 0.0
    while len(indices) < limit and int(i) < n_total:
        idx = int(i)
        if idx not in seen:
            indices.append(idx)
            seen.add(idx)
        i += stride
    return indices


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--target-rows", type=int, default=2000)
    args = parser.parse_args()

    print(f"checking license at {README_URL} ...")
    license_value = check_license()
    print(f"  license: {license_value}")
    if license_value != REQUIRED_LICENSE:
        print(
            f"STOP: expected license '{REQUIRED_LICENSE}', found '{license_value}' -- "
            "refusing to fetch/convert. Report this to the task owner.",
            file=sys.stderr,
        )
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    for shard_idx in range(N_SHARDS):
        url = SHARD_URL_TEMPLATE.format(shard_idx)
        print(
            f"reading shard {shard_idx:04d}.parquet (column-pruned -- no embedding_* columns) ..."
        )
        with fsspec.open(url, mode="rb") as f:
            table = pq.read_table(f, columns=COLUMNS)
        print(f"  {table.num_rows:,} rows read")
        all_rows.extend(table.to_pylist())

    n_total = len(all_rows)
    print(f"{n_total:,} rows read across {N_SHARDS} shards")

    indices = select_evenly(n_total, args.target_rows)
    print(f"  selecting {len(indices)} rows at an evenly spaced stride across all shards")

    rows = [all_rows[i] for i in indices]

    out_path = args.out_dir / "records.jsonl"
    written = 0
    with out_path.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

    print(f"wrote {written} records to {out_path}")
    print(f"license recorded: {license_value} (verified against {README_URL})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
