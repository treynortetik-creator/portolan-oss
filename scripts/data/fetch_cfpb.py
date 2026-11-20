"""Fetch CFPB consumer complaints with narratives and company public
responses (`01-spec/research/free-data-and-open-models-2026-09-22.md`,
Track A row 1: CFPB Consumer Complaint Database).

**Deviation from the task brief's URL -- verified 2026-09-22, before this
script was written, not assumed.** The brief's endpoint
(`consumerfinance.gov/data-research/consumer-complaints/search/api/v1/
?has_narrative=true...`) is live and reachable, but as of **2026-08-14
CFPB stopped publishing complaint narrative text entirely** -- confirmed
three independent ways on 2026-09-22:
  1. The live search API's `field` parameter now rejects the narrative
     field outright: `?field=complaint_what_happened` ->
     `{"field": ["\\"complaint_what_happened\\" is not a valid choice."]}`.
  2. `?has_narrative=true&size=2` (the brief's own filter) returns real
     hits, but every hit's `_source` has no narrative field at all --
     `_source` is a fixed field list (`ccdb5-api`'s own
     `complaint_search/defaults.py: SOURCE_FIELDS`) that no longer
     includes it, regardless of query params.
  3. The live bulk CSV export (`https://files.consumerfinance.gov/ccdb/
     complaints.csv.zip`, downloaded fresh 2026-09-22) has a 15-column
     header with no narrative or free-text company-response column:
     `Date received,Product,Sub-product,Issue,Sub-issue,Company public
     response,Company,State,ZIP code,Tags,Submitted via,Date sent to
     company,Company response to consumer,Timely response?,Complaint ID`
     -- "Company public response" here is a fixed enum ("Company
     believes it acted appropriately", not free text).
  4. CFPB's own newsroom confirms the policy change: "The CFPB to Cease
     Discretionary Publication of Complaint Narratives and
     Visualizations" (2026-08-14), stating prior narratives move to the
     FOIA Reading Room rather than staying in the live database/API.

This script instead reads `davidheineman/consumer-finance-complaints-
large` on Hugging Face: a static, columnar (parquet) mirror of the same
CFPB complaint database captured **while CFPB still published
narratives** (this slice's own `date_received` values run through
2024-12-24 -- see the printed range at fetch time -- well before the
2026-08-14 cutoff). It carries exactly the fields the task asked for:
`complaint_what_happened` (free-text narrative) and
`company_public_response` -- CFPB's own controlled vocabulary of ~11
canned company-response categories (e.g. "Company believes it acted
appropriately as authorized by contract or law"), never freeform text --
plus `company`, `date_received`, `product`, `complaint_id`. This is still
CFPB's own public-domain US-government data -- the underlying government
work's public-domain status does not depend on this mirror repo's own
`license: apache-2.0` tag (see `var/data/MANIFEST.md` section 6 for the
full licensing note).

Reads shard 0 of 11 directly with `pyarrow` (column-pruned over HTTP via
`fsspec`, not the full `datasets` streaming path, and not the HF
`datasets-server` `/rows` or `/filter` APIs -- `/rows` has no server-side
predicate so pulling enough narrative-bearing rows would mean paging
almost the whole 7.18M-row table, and `/filter` returned HTTP 422
"Parameter 'where' contains errors or invalid symbols" for straightforward
SQL like `complaint_what_happened IS NOT NULL` when this script was being
built). One shard alone is ~652,667 rows and yields ~197K
narrative-bearing complaints across every product category -- comfortably
more than the 3,000-record target -- so later shards are never read.
(Shard 0 turned out to already span a wide date range on its own --
2015-03-31 through 2024-12-23 in the sampled output -- so unlike Enron's
deliberate evenly-spaced offsets, this script does not need to reach into
other shards for date diversity; see MANIFEST.md for the exact range.)

Sampling: buckets narrative-bearing rows by their own literal `product`
string (CFPB's taxonomy has near-duplicate wording across years, e.g.
"Credit reporting or other personal consumer reports" vs. "Credit
reporting, credit repair services, or other personal consumer reports" --
left as two distinct buckets since the source itself uses two distinct
strings, not merged), gives each bucket a base quota of
`target_rows // n_products` (remainder spread one-per-bucket over the
first buckets, alphabetically), then tops off any shortfall (a bucket
with fewer available rows than its quota) from whichever buckets have the
largest surplus, so the run lands on exactly `--target-rows` whenever
enough narratives exist overall.

Usage:
    uv run python scripts/data/fetch_cfpb.py [--out-dir DIR] [--target-rows 3000]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import fsspec
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = REPO_ROOT / "var/data/raw/cfpb"

SHARD_URL = (
    "https://huggingface.co/datasets/davidheineman/consumer-finance-complaints-large"
    "/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"
)
COLUMNS = [
    "complaint_id",
    "product",
    "company",
    "date_received",
    "complaint_what_happened",
    "company_public_response",
]


def load_shard_rows() -> list[dict]:
    with fsspec.open(SHARD_URL, mode="rb") as f:
        table = pq.read_table(f, columns=COLUMNS)
    return table.to_pylist()


def bucket_by_product(rows: list[dict]) -> dict[str, list[dict]]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        narrative = (row.get("complaint_what_happened") or "").strip()
        if not narrative:
            continue
        product = (row.get("product") or "Unknown").strip() or "Unknown"
        buckets[product].append(row)
    return buckets


def sample_evenly(buckets: dict[str, list[dict]], target: int) -> list[dict]:
    products = sorted(buckets)
    n = len(products)
    if n == 0 or target <= 0:
        return []
    base_quota = target // n
    remainder = target - base_quota * n
    picked: dict[str, list[dict]] = {}
    for i, product in enumerate(products):
        quota = base_quota + (1 if i < remainder else 0)
        picked[product] = buckets[product][:quota]

    total = sum(len(v) for v in picked.values())
    if total < target:
        shortfall = target - total
        surplus_order = sorted(
            products, key=lambda p: len(buckets[p]) - len(picked[p]), reverse=True
        )
        for product in surplus_order:
            if shortfall <= 0:
                break
            already = len(picked[product])
            available = buckets[product][already:]
            take = min(shortfall, len(available))
            if take:
                picked[product].extend(available[:take])
                shortfall -= take

    result: list[dict] = []
    for product in products:
        result.extend(picked[product])
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--target-rows", type=int, default=3000)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(
        "reading shard 0000.parquet of davidheineman/consumer-finance-complaints-large "
        "(column-pruned: complaint_id, product, company, date_received, "
        "complaint_what_happened, company_public_response) ..."
    )
    rows = load_shard_rows()
    print(f"  {len(rows):,} rows read")

    buckets = bucket_by_product(rows)
    total_narratives = sum(len(v) for v in buckets.values())
    print(
        f"  {total_narratives:,} rows have a non-empty narrative, across {len(buckets)} products:"
    )
    for product in sorted(buckets, key=lambda p: -len(buckets[p])):
        print(f"    {len(buckets[product]):7,d}  {product}")

    sampled = sample_evenly(buckets, args.target_rows)
    dates = sorted(r["date_received"] for r in sampled if r.get("date_received"))

    out_path = args.out_dir / "records.jsonl"
    with out_path.open("w", encoding="utf-8") as out:
        for record in sampled:
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"wrote {len(sampled)} records to {out_path}")
    if dates:
        print(f"date_received range in sample: {dates[0]} .. {dates[-1]}")
    if len(sampled) < args.target_rows:
        print(
            f"WARNING: only {len(sampled)}/{args.target_rows} rows available "
            "(shard exhausted before reaching the target)",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
