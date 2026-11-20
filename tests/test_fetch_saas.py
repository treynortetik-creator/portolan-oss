"""Tests for `scripts/data/fetch_saas.py`'s pure logic
(`parse_license_from_front_matter`, `select_evenly`) -- no network.

`scripts/` isn't a Python package (no `__init__.py`) and is excluded from
the container image, so this loads the module by file path and skips when
absent -- the same pattern `tests/test_set_owners.py` uses for
`scripts/set_owners.py`.
"""

from __future__ import annotations

import importlib.util
from types import ModuleType

import pytest

from tests._paths import require_repo_path

_SCRIPT = require_repo_path("scripts", "data", "fetch_saas.py")


def _load_fetch_saas() -> ModuleType:
    spec = importlib.util.spec_from_file_location("fetch_saas_script", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------
# License parsing
# --------------------------------------------------------------------------


def test_parse_license_from_front_matter_apache() -> None:
    module = _load_fetch_saas()
    text = "---\nlanguage:\n- en\nlicense: apache-2.0\ntags:\n- sales\n---\n\n# body\n"
    assert module.parse_license_from_front_matter(text) == "apache-2.0"


def test_parse_license_from_front_matter_raises_without_front_matter() -> None:
    module = _load_fetch_saas()
    with pytest.raises(RuntimeError, match="no YAML front matter"):
        module.parse_license_from_front_matter("# just a heading, no front matter\n")


def test_parse_license_from_front_matter_raises_without_license_field() -> None:
    module = _load_fetch_saas()
    text = "---\nlanguage:\n- en\n---\n\n# body\n"
    with pytest.raises(RuntimeError, match="no 'license:' field"):
        module.parse_license_from_front_matter(text)


def test_parse_license_from_front_matter_rejects_non_apache() -> None:
    module = _load_fetch_saas()
    text = "---\nlicense: cc-by-nc-4.0\n---\n"
    assert module.parse_license_from_front_matter(text) == "cc-by-nc-4.0"
    assert module.parse_license_from_front_matter(text) != module.REQUIRED_LICENSE


# --------------------------------------------------------------------------
# Even sampling
# --------------------------------------------------------------------------


def test_select_evenly_returns_all_indices_when_limit_exceeds_total() -> None:
    module = _load_fetch_saas()
    assert module.select_evenly(5, 10) == [0, 1, 2, 3, 4]


def test_select_evenly_selects_requested_count() -> None:
    module = _load_fetch_saas()
    indices = module.select_evenly(100, 10)
    assert len(indices) == 10
    assert len(set(indices)) == 10
    assert all(0 <= i < 100 for i in indices)
    assert indices == sorted(indices)


def test_select_evenly_spans_the_full_range() -> None:
    module = _load_fetch_saas()
    indices = module.select_evenly(100000, 2000)
    assert len(indices) == 2000
    # an evenly spaced stride should reach well past the first few percent
    assert indices[-1] > 90000
