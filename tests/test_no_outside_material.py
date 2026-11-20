"""Two hard rules with no other enforcer: no material from an employer's (or any third
party's) systems, and no customer data. Both read the files git tracks, so they need a checkout
(the app image has no `.git`).

1. Forbidden names (set `PORTOLAN_FORBIDDEN_NAMES`, comma-separated, e.g. your employer's name
   and domain) appear nowhere in tracked files. Unset, the live check is skipped; the planted
   check below still proves the scanner works.
2. Data-shaped files (exports, mailboxes, spreadsheets, PDFs) and anything over 200 KB live
   only under directories that are synthetic by construction, or are named here with why.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

from tests._paths import require_repo_path

REPO_ROOT = Path(__file__).resolve().parent.parent
require_repo_path(".git")


def forbidden_pattern(names: str) -> re.Pattern[bytes] | None:
    parts = [re.escape(n.strip()) for n in names.split(",") if n.strip()]
    return re.compile("|".join(parts).encode(), re.IGNORECASE) if parts else None


EMPLOYER = forbidden_pattern(os.environ.get("PORTOLAN_FORBIDDEN_NAMES", ""))

# path -> the most lines that may mention the employer (each one states the rule or lists
# what was kept out). Raising a number needs the same review as adding a path.
EMPLOYER_MENTIONS_ALLOWED: dict[str, int] = {}

DATA_SUFFIXES = frozenset(
    {
        ".jsonl", ".csv", ".tsv", ".eml", ".mbox", ".msg", ".pdf",
        ".parquet", ".xlsx", ".xls", ".sqlite", ".db", ".dump",
    }
)  # fmt: skip
MAX_BYTES = 200 * 1024
# Everything under these is generated or hand-written for this repo, never a customer's.
SYNTHETIC_DIRS = ("corpus/out/", "corpus/generator/", "corpus/demo/", "tests/fixtures/")
LARGE_FILES_ALLOWED = {
    "01-spec/how-portolan-works.html": "explainer page with inlined assets",
    "uv.lock": "dependency lock file",
    "portolan/report/themes.py": "source",
    "tests/test_report_themes.py": "test source",
    "tests/test_web.py": "test source",
    "corpus/verdicts.yaml": "hand-labeled verdicts over the synthetic Halvard exam",
    "01-spec/bench-erb-search-2026-09-22.results.json": "ERB exam scores: ids and numbers only",
    "01-spec/bench-erb-search-2026-09-22-keyword-leg.results.json": (
        "ERB exam scores after the keyword-leg fix: ids and numbers only"
    ),
}


def tracked_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "-z"], cwd=REPO_ROOT, capture_output=True, check=True
    ).stdout
    return [p for p in out.decode("utf-8").split("\0") if p]


def employer_mentions(
    files: list[str], root: Path = REPO_ROOT, pattern: re.Pattern[bytes] | None = None
) -> dict[str, int]:
    pattern = pattern or EMPLOYER
    counts: dict[str, int] = {}
    if pattern is None:
        return counts
    for rel in files:
        path = root / rel
        if not path.is_file():
            continue
        lines = sum(1 for line in path.read_bytes().splitlines() if pattern.search(line))
        if lines:
            counts[rel] = lines
    return counts


def test_the_forbidden_pattern_matches_each_name() -> None:
    pat = forbidden_pattern("Acme Widgets, acmewidgets.com")
    assert pat is not None
    for text in ("ACME WIDGETS", "ops@acmewidgets.com"):
        assert pat.search(text.encode()), text
    assert not pat.search(b"acme rockets")
    assert forbidden_pattern("") is None


def mentions_over_allowance(
    files: list[str], root: Path = REPO_ROOT, pattern: re.Pattern[bytes] | None = None
) -> dict[str, int]:
    found = employer_mentions(files, root, pattern)
    return {rel: n for rel, n in found.items() if n > EMPLOYER_MENTIONS_ALLOWED.get(rel, 0)}


def misplaced_data_files(files: list[str], root: Path = REPO_ROOT) -> list[str]:
    misplaced = []
    for rel in files:
        path = root / rel
        if rel.startswith(SYNTHETIC_DIRS) or not path.is_file():
            continue
        if Path(rel).suffix.lower() in DATA_SUFFIXES:
            misplaced.append(f"{rel} (data file)")
        elif path.stat().st_size > MAX_BYTES and rel not in LARGE_FILES_ALLOWED:
            misplaced.append(f"{rel} ({path.stat().st_size:,} bytes)")
    return misplaced


def test_the_checks_catch_planted_files(tmp_path: Path) -> None:
    planted = {
        "notes/handoff.md": "Met the Acme Widgets ops team today.\n",
        "CLAUDE.md": "rule line one: acmewidgets\nrule line two: acmewidgets\n",
        "exports/accounts.csv": "id,name\n1,Acme\n",
        "tests/fixtures/connectors/x/export.csv": "id\n1\n",
        "notes/big.txt": "x" * (MAX_BYTES + 1),
    }
    for rel, text in planted.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(text, encoding="utf-8")
    files = sorted(planted)
    assert mentions_over_allowance(
        files, tmp_path, forbidden_pattern("acme widgets,acmewidgets")
    ) == {"notes/handoff.md": 1, "CLAUDE.md": 2}
    assert misplaced_data_files(files, tmp_path) == [
        "exports/accounts.csv (data file)",
        f"notes/big.txt ({MAX_BYTES + 1:,} bytes)",
    ]


@pytest.mark.skipif(EMPLOYER is None, reason="PORTOLAN_FORBIDDEN_NAMES is not set")
def test_the_employer_appears_only_where_the_rule_is_stated() -> None:
    over = mentions_over_allowance(tracked_files())
    assert not over, (
        f"the employer's name or domain in {over} (file: lines). no material from "
        "any employer's systems, data or documents in this repo."
    )


def test_no_data_files_outside_synthetic_directories() -> None:
    misplaced = misplaced_data_files(tracked_files())
    assert not misplaced, (
        f"tracked outside {SYNTHETIC_DIRS}: {misplaced}. CLAUDE.md: no customer data in this "
        "repo. Synthetic data goes under one of those directories; a large non-data file "
        "gets a line in LARGE_FILES_ALLOWED saying what it is."
    )
