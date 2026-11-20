"""Dollar spend meter for the silver-labeling pipeline's paid calls
(coordinator course-correction, 2026-09-21, item 1: "Add a spend meter
... accumulate it in `var/data/silver/spend.json` (per model, per
corpus, total), print the running total every 25 documents, and stop
the run cleanly when total spend reaches the `--budget-usd` ceiling").

One file per machine: `default_spend_path` maps any checkout's `var/`, a
worktree's included, to the main checkout's (guardrails review 2026-09-22,
row 33: a worktree agent's spend used to land in a meter nobody read).

Same file-lock pattern as `portolan.training.free_tier.FreeTierBudget`
(one JSON file, `fcntl.flock`-guarded read-modify-write) but tracking
dollars, not a request count, and with no daily reset -- a spend
ceiling is a per-run budget, not a calendar-day one.
"""

from __future__ import annotations

import fcntl
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["SpendMeter", "default_spend_path", "shared_var_root"]


def _git(checkout: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args], cwd=checkout, capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def shared_var_root(var_root: Path) -> Path:
    """The main checkout's `var/` when `var_root` is a git checkout's own `var/` (a worktree
    included, found through `git rev-parse --git-common-dir`), else `var_root` unchanged (a
    snapshot directory, a test's tmp_path, a container with no git). One meter per machine:
    a worktree agent's paid calls count against the same ceiling as the main checkout's."""
    var_root = Path(var_root).resolve()
    checkout = var_root.parent
    if var_root.name != "var" or not checkout.is_dir():
        return var_root
    toplevel = _git(checkout, "rev-parse", "--show-toplevel")
    common = _git(checkout, "rev-parse", "--git-common-dir")
    if not toplevel or not common or Path(toplevel).resolve() != checkout:
        return var_root
    common_dir = Path(common)
    if not common_dir.is_absolute():
        common_dir = checkout / common_dir
    return common_dir.resolve().parent / "var"


def default_spend_path(var_root: Path) -> Path:
    return shared_var_root(var_root) / "data" / "silver" / "spend.json"


@dataclass
class SpendMeter:
    path: Path

    def _read_locked(self, fh: Any) -> dict[str, Any]:
        fh.seek(0)
        raw = fh.read()
        if not raw.strip():
            return {"total_usd": 0.0, "per_model_usd": {}, "per_corpus_usd": {}}
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return {"total_usd": 0.0, "per_model_usd": {}, "per_corpus_usd": {}}
        if not isinstance(data, dict):
            return {"total_usd": 0.0, "per_model_usd": {}, "per_corpus_usd": {}}
        data.setdefault("total_usd", 0.0)
        data.setdefault("per_model_usd", {})
        data.setdefault("per_corpus_usd", {})
        return data

    def _write_locked(self, fh: Any, data: dict[str, Any]) -> None:
        fh.seek(0)
        fh.truncate()
        fh.write(json.dumps(data, indent=2))
        fh.flush()

    def _with_lock(self, fn: Any) -> Any:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        with self.path.open("r+", encoding="utf-8") as fh:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            try:
                return fn(fh)
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)

    def total_usd(self) -> float:
        return float(self._with_lock(lambda fh: self._read_locked(fh))["total_usd"])

    def snapshot(self) -> dict[str, Any]:
        return self._with_lock(lambda fh: self._read_locked(fh))

    def is_over_budget(self, budget_usd: float) -> bool:
        return self.total_usd() >= budget_usd

    def corpus_usd(self, corpus: str) -> float:
        """This meter's `per_corpus_usd[corpus]` -- 0.0 if `corpus` has
        never had anything recorded against it. Unlike `total_usd()`, this
        is unaffected by spend recorded under a DIFFERENT corpus (e.g. a
        concurrent silver-labeling run against `fda_maude`/`ami`) sharing
        the same meter file -- see `is_corpus_over_budget`."""
        data = self._with_lock(lambda fh: self._read_locked(fh))
        return float(data["per_corpus_usd"].get(corpus, 0.0))

    def is_corpus_over_budget(self, corpus: str, budget_usd: float) -> bool:
        """True once `corpus_usd(corpus) >= budget_usd` -- the per-corpus
        counterpart to `is_over_budget`. Use this for a ceiling that is
        meant to apply to ONE corpus's spend only (e.g. the paid-teacher
        synthetic-data run's `--budget-usd`, fixed 2026-09-22 after a run
        stopped immediately because the shared meter's grand `total_usd`
        had already been pushed past the ceiling by an unrelated,
        concurrently-running silver-labeling corpus). `is_over_budget`
        stays correct as-is for a ceiling that really is meant to be the
        meter's whole total across every corpus (the silver-labeling
        pipeline's own `--budget-usd`, by original design -- see this
        module's docstring)."""
        return self.corpus_usd(corpus) >= budget_usd

    def record(
        self, *, model: str, corpus: str, amount_usd: float, allow_zero: bool = False
    ) -> float:
        """Add `amount_usd` (a single call's real cost, e.g. one
        `UsageEvent.cost_estimate` delta) to `total_usd`, `per_model_usd
        [model]`, and `per_corpus_usd[corpus]`. Returns the new
        `total_usd`. A non-positive amount is a no-op (nothing to
        record, and never lets a negative correction silently reduce the
        total), except that `allow_zero=True` with exactly 0 writes a $0
        line for `model` and `corpus`: a local model's calls are free but
        still named on the meter."""
        if amount_usd < 0 or (amount_usd == 0 and not allow_zero):
            return self.total_usd()

        def _op(fh: Any) -> float:
            data = self._read_locked(fh)
            data["total_usd"] = round(data["total_usd"] + amount_usd, 8)
            data["per_model_usd"][model] = round(
                data["per_model_usd"].get(model, 0.0) + amount_usd, 8
            )
            data["per_corpus_usd"][corpus] = round(
                data["per_corpus_usd"].get(corpus, 0.0) + amount_usd, 8
            )
            self._write_locked(fh, data)
            return float(data["total_usd"])

        return self._with_lock(_op)
