"""Fetch device adverse-event narratives from the FDA's MAUDE database via
openFDA (public domain US government data,
https://open.fda.gov/license/ -- "there are no restrictions on the use
of this data"). Hits the *device* event endpoint, not drugs -- the task
brief's URL (`api.open.fda.gov`) is a typo for the real host, `api.fda.gov`.

Only records that carry a "Description of Event or Problem" narrative are
fetched (`search=mdr_text.text_type_code:"Description of Event or
Problem"`), sorted newest-first, paginated with `skip`/`limit` (openFDA
caps `limit` at 100 per request for this endpoint). Each 100-record page
is written to `var/data/raw/fda_maude/pages/skip_<n>.json` (the page
already at that skip is not re-fetched on a re-run, so this is
resumable) and also appended, one JSON object per line, to
`records.jsonl`.

De-identification note: MAUDE narratives are already de-identified by
FDA before publication (no patient name/MRN; reporter is identified only
by occupation code, e.g. "Health Professional"). This script does not
attempt any additional redaction.

Usage:
    uv run python scripts/data/fetch_fda_maude.py [--out-dir DIR] [--target-rows 3000]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://api.fda.gov/device/event.json"
PAGE_SIZE = 100
SEARCH = 'mdr_text.text_type_code:"Description of Event or Problem"'
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = REPO_ROOT / "var/data/raw/fda_maude"


def fetch_page(skip: int) -> list[dict]:
    params = {
        "search": SEARCH,
        "sort": "date_received:desc",
        "limit": str(PAGE_SIZE),
        "skip": str(skip),
    }
    url = f"{API}?{urllib.parse.urlencode(params)}"
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                payload = json.loads(resp.read())
            return payload.get("results", [])
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                # openFDA returns 404 (NOT_FOUND) once skip runs past the
                # matching result set -- treat as "no more pages".
                return []
            wait = 2**attempt
            print(f"  skip {skip}: HTTP {exc.code} -- retrying in {wait}s", file=sys.stderr)
            time.sleep(wait)
        except (urllib.error.URLError, TimeoutError) as exc:
            wait = 2**attempt
            print(f"  skip {skip}: {exc} -- retrying in {wait}s", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"failed to fetch skip={skip} after retries")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--target-rows", type=int, default=3000)
    args = parser.parse_args()

    pages_dir = args.out_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    records_path = args.out_dir / "records.jsonl"

    n_pages = max(1, -(-args.target_rows // PAGE_SIZE))  # ceil div
    fetched = 0
    with records_path.open("w", encoding="utf-8") as out:
        for i in range(n_pages):
            skip = i * PAGE_SIZE
            page_path = pages_dir / f"skip_{skip}.json"
            if page_path.exists():
                results = json.loads(page_path.read_text(encoding="utf-8"))
                print(f"  skip {skip}: cached, {len(results)} records")
            else:
                results = fetch_page(skip)
                page_path.write_text(json.dumps(results), encoding="utf-8")
                print(f"  skip {skip}: fetched {len(results)} records")
                time.sleep(0.3)  # be polite; openFDA allows 240 req/min unauthenticated
            if not results:
                print("  no more results -- stopping")
                break
            for r in results:
                out.write(json.dumps(r, ensure_ascii=False) + "\n")
            fetched += len(results)

    print(f"wrote {fetched} records to {records_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
