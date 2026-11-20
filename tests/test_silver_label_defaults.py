"""Pins `scripts/data/silver_label.py`'s `--mode single` defaults
(fix/overlap-gate-paraphrase, 2026-09-22: the real Kimi K2 run on 12
SaaS sales calls showed the content-overlap rule dropping 23 of 49
emitted facts, every one a faithful restatement of its quote defeated
by ordinary paraphrase -- and the calibration table it was tuned
against never showed it doing any discriminating work at all (flat
0.20-0.95). Disabled by default, `0.0`, rather than re-tuned -- see
`portolan/training/overlap_gate.py`'s module docstring for the full
reasoning). A regression test, not a behavior test: if someone changes
`DEFAULT_OVERLAP_THRESHOLD` back without updating this test, the change
gets noticed.

`scripts/` isn't a Python package and is excluded from the container
image, so this loads the module by file path and skips when absent --
the same pattern `tests/test_set_owners.py` uses for `scripts/set_owners.py`.
"""

from __future__ import annotations

import importlib.util
from types import ModuleType

from tests._paths import require_repo_path

_SCRIPT = require_repo_path("scripts", "data", "silver_label.py")


def _load_silver_label() -> ModuleType:
    spec = importlib.util.spec_from_file_location("silver_label_script", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_default_overlap_threshold_is_disabled() -> None:
    module = _load_silver_label()
    assert module.DEFAULT_OVERLAP_THRESHOLD == 0.0


def test_default_budget_usd_unchanged() -> None:
    module = _load_silver_label()
    assert module.DEFAULT_BUDGET_USD == 10.00
