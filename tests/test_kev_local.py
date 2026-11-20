"""Tests for `scripts/kev/probe.py` (`docs/kev-local.md`): the three
judge-shaped probes against a locally running Kev (https://github.com/
jaredpalmer/kev) System One server.

`scripts/` isn't a Python package and is excluded from the container image
(`tests/test_set_owners.py` documents this), so this loads the probe module
by file path, the same way. It also needs a live server at `KEV_BASE_URL`
(default `http://127.0.0.1:8787`) -- there is no weights-free fake here, so
the whole module skips when nothing answers there, the same "resource
absent -> skip, don't fail" pattern `tests/test_search_embed.py` uses for
`PORTOLAN_REAL_EMBED`.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import httpx
import pytest

from tests._paths import require_repo_path

REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = require_repo_path("scripts", "kev", "probe.py")
_BASE_URL = os.environ.get("KEV_BASE_URL", "http://127.0.0.1:8787")


def _server_reachable(base_url: str) -> bool:
    try:
        resp = httpx.get(base_url.rstrip("/") + "/v1/models", timeout=2.0)
        return resp.status_code == 200
    except httpx.HTTPError:
        return False


pytestmark = pytest.mark.skipif(
    not _server_reachable(_BASE_URL),
    reason=f"no Kev server answering at {_BASE_URL} -- see docs/kev-local.md to start one",
)


def _load_probe_module():
    spec = importlib.util.spec_from_file_location("kev_probe_script", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["kev_probe_script"] = module
    spec.loader.exec_module(module)
    return module


def test_three_probes_return_typed_answers_with_probabilities():
    probe = _load_probe_module()
    rows = probe.run_probes(_BASE_URL)
    assert len(rows) == 3

    by_id = {row["probe_id"]: row for row in rows}

    faithful = by_id["faithful"]["primary_answer"]
    assert faithful["type"] == "noul"
    assert 0.0 <= faithful["noul"] <= 1.0

    fact_type = by_id["fact_type"]["primary_answer"]
    assert fact_type["type"] == "choice"
    assert fact_type["choice"] in probe.FACT_TYPE_CRITERIA
    # Probabilities are rounded to 2dp on the wire, so allow for rounding drift
    # across up to 12 fact-type options rather than requiring an exact sum.
    assert abs(sum(fact_type["probabilities"].values()) - 1.0) < 0.02

    overreach = by_id["overreach"]["primary_answer"]
    assert overreach["type"] == "noul"
    assert 0.0 <= overreach["noul"] <= 1.0
