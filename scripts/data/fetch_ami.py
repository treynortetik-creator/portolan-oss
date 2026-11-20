"""Fetch the AMI Meeting Corpus manual annotations (CC BY 4.0).

Downloads the official manual-annotation bundle from the University of
Edinburgh (`ami_public_manual_1.6.2.zip`, ~23 MB) to
`var/data/raw/ami/`. Idempotent: skips the download if the file already
exists with a non-zero size. Nothing is extracted here -- `convert_ami.py`
reads directly out of the zip (via Python's `zipfile`) so the raw
footprint stays at ~23 MB instead of the ~200 MB the fully unzipped
bundle would take.

Usage:
    uv run python scripts/data/fetch_ami.py [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

URL = "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = REPO_ROOT / "var/data/raw/ami"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    out_path = args.out_dir / "ami_public_manual_1.6.2.zip"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"already have {out_path} ({out_path.stat().st_size:,} bytes) -- skipping")
        return 0

    print(f"downloading {URL}")
    tmp = out_path.with_suffix(".zip.part")
    with urllib.request.urlopen(URL, timeout=120) as resp, tmp.open("wb") as fh:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
    tmp.rename(out_path)
    print(f"wrote {out_path} ({out_path.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
