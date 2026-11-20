"""Shared provenance-stamp helpers for every command that writes a
`BenchReport` -- `portolan.bench.__main__` (`extract`/`--rescore`/
`--rescore-dir`) and `portolan.bench.judge_cli` (`judge --rescore-dir`).

Split out of `__main__.py` (2026-09-18 bench-provenance follow-up,
`01-spec/bench-provenance-2026-09-18.md`) so `judge_cli.py` can stamp the
SAME fields on a judged report without importing from `__main__` (a
`python -m` entry-point module) or duplicating the logic. This module has
no CLI of its own and makes no judge/scoring decision of any kind -- pure
metadata (git commit, file hashes), never touching `portolan.bench.judge`'s
verdict logic, prompts, or model choice.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

# The repo root this file lives under (`.../portolan/bench/provenance.py`
# -> parents[2]) -- used only to ask git for the current commit.
REPO_ROOT = Path(__file__).resolve().parents[2]


def harness_commit() -> str | None:
    """`git rev-parse HEAD` of the checkout this process is running from
    -- `None` (never raises) when git isn't available or this isn't a git
    checkout at all, so a missing commit never fails a bench run over a
    provenance nicety."""
    try:
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def sha256_file(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def corpus_listing_sha256(corpus_root: Path) -> str | None:
    """sha256 of the sorted, newline-joined list of relative file paths
    under `corpus_root` -- a cheap, reproducible stand-in for "which
    corpus snapshot this run scored against" that changes if any file is
    added, removed or renamed, without hashing every file's content."""
    try:
        paths = sorted(
            str(p.relative_to(corpus_root)) for p in corpus_root.rglob("*") if p.is_file()
        )
    except OSError:
        return None
    if not paths:
        return None
    return hashlib.sha256("\n".join(paths).encode("utf-8")).hexdigest()


def provenance_kwargs(
    *,
    corpus: str,
    truth_table: str,
    prompt_variant: str | None = None,
    extractor_model: str | None = None,
) -> dict[str, object]:
    """Shared by every command that constructs a `BenchReport` --
    `_cmd_extract`/`_cmd_rescore`/`_cmd_rescore_dir` in `__main__.py`, and
    `judge_cli._cmd_judge` -- so the provenance stamp (`BenchReport`'s own
    docstring comments) is computed the same way regardless of which
    command wrote the report."""
    truth_table_path = Path(truth_table)
    return {
        "prompt_variant": prompt_variant,
        "extractor_model": extractor_model,
        "harness_commit": harness_commit(),
        "truth_table_path": str(truth_table_path),
        "truth_table_sha256": sha256_file(truth_table_path),
        "corpus_listing_sha256": corpus_listing_sha256(Path(corpus)),
    }
