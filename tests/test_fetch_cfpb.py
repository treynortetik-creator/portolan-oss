"""Tests for `scripts/data/fetch_cfpb.py`'s pure sampling logic
(`bucket_by_product`, `sample_evenly`) -- no network.

`scripts/` isn't a Python package (no `__init__.py`) and is excluded from
the container image, so this loads the module by file path and skips when
absent -- the same pattern `tests/test_set_owners.py` uses for
`scripts/set_owners.py`.
"""

from __future__ import annotations

import importlib.util
from types import ModuleType

from tests._paths import require_repo_path

_SCRIPT = require_repo_path("scripts", "data", "fetch_cfpb.py")


def _load_fetch_cfpb() -> ModuleType:
    spec = importlib.util.spec_from_file_location("fetch_cfpb_script", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_bucket_by_product_drops_rows_without_narrative() -> None:
    module = _load_fetch_cfpb()
    rows = [
        {"product": "Debt collection", "complaint_what_happened": "they called me"},
        {"product": "Debt collection", "complaint_what_happened": ""},
        {"product": "Debt collection", "complaint_what_happened": "   "},
        {"product": "Mortgage", "complaint_what_happened": None},
    ]
    buckets = module.bucket_by_product(rows)
    assert list(buckets.keys()) == ["Debt collection"]
    assert len(buckets["Debt collection"]) == 1


def test_bucket_by_product_falls_back_to_unknown() -> None:
    module = _load_fetch_cfpb()
    rows = [{"product": "", "complaint_what_happened": "narrative text"}]
    buckets = module.bucket_by_product(rows)
    assert list(buckets.keys()) == ["Unknown"]


def test_sample_evenly_splits_base_quota_across_products() -> None:
    module = _load_fetch_cfpb()
    buckets = {
        "A": [{"i": i} for i in range(10)],
        "B": [{"i": i} for i in range(10)],
    }
    sampled = module.sample_evenly(buckets, 6)
    assert len(sampled) == 6


def test_sample_evenly_tops_off_shortfall_from_surplus_buckets() -> None:
    module = _load_fetch_cfpb()
    buckets = {
        "Rare": [{"i": 0}],  # only 1 available, quota would be 5
        "Common": [{"i": i} for i in range(20)],
    }
    sampled = module.sample_evenly(buckets, 10)
    # Rare can only contribute its single row; the rest comes from Common.
    assert len(sampled) == 10


def test_sample_evenly_caps_at_available_total() -> None:
    module = _load_fetch_cfpb()
    buckets = {"A": [{"i": 0}], "B": [{"i": 0}]}
    sampled = module.sample_evenly(buckets, 100)
    assert len(sampled) == 2


def test_sample_evenly_empty_buckets_returns_empty() -> None:
    module = _load_fetch_cfpb()
    assert module.sample_evenly({}, 10) == []
