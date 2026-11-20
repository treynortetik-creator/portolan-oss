"""Tests for `railway.json` and `docs/deploy-railway.md` (bug hunt
2026-09-18, P1 finding 8: "the production environment has no
infrastructure-as-code and no runbook in the repo" -- start command,
healthcheck path, and volume mount lived only in the Railway dashboard).
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RAILWAY_JSON = REPO_ROOT / "railway.json"
RUNBOOK = REPO_ROOT / "docs" / "deploy-railway.md"


def test_railway_json_exists_and_points_the_healthcheck_at_healthz() -> None:
    assert RAILWAY_JSON.exists(), "railway.json is missing (bug hunt finding 8)"
    config = json.loads(RAILWAY_JSON.read_text(encoding="utf-8"))
    assert config["deploy"]["healthcheckPath"] == "/healthz"
    assert config["build"]["dockerfilePath"] == "docker/Dockerfile"


def test_runbook_exists_and_names_the_required_env_vars() -> None:
    assert RUNBOOK.exists(), "docs/deploy-railway.md is missing (bug hunt finding 8)"
    text = RUNBOOK.read_text(encoding="utf-8")
    for required in ("DATABASE_URL", "PORTOLAN_ORG_KEY", "PORTOLAN_SESSION_SECRET", "STORE_ROOT"):
        assert required in text
