"""Static checks on the `secrets` job in `.github/workflows/ci.yml` (guardrails plan
2026-09-22, rank 5): gitleaks runs over full history, from a pinned and checksum-verified
binary, with the repo's allowlist. Not a substitute for the job running; it proves the lines
that make it a full-history, pinned scan are the ones in the file.

`.dockerignore` excludes `.github/`, so this skips inside the image (as
`tests/test_ci_workflow.py` does)."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from tests._paths import require_repo_path

REPO_ROOT = Path(__file__).resolve().parent.parent
CI_YML = require_repo_path(".github", "workflows", "ci.yml")


def _secrets_job() -> dict:
    return yaml.safe_load(CI_YML.read_text(encoding="utf-8"))["jobs"]["secrets"]


def test_secrets_job_checks_out_full_history() -> None:
    checkout = _secrets_job()["steps"][0]
    assert checkout["uses"].startswith("actions/checkout@")
    assert checkout["with"]["fetch-depth"] == 0


def test_gitleaks_binary_is_pinned_and_checksum_verified() -> None:
    job = _secrets_job()
    assert re.fullmatch(r"\d+\.\d+\.\d+", job["env"]["GITLEAKS_VERSION"])
    assert re.fullmatch(r"[0-9a-f]{64}", job["env"]["GITLEAKS_SHA256"])
    install = next(s["run"] for s in job["steps"] if "Install gitleaks" in s.get("name", ""))
    assert "sha256sum -c" in install
    assert install.index("sha256sum -c") < install.index("tar -xzf")


def test_scan_covers_git_history_with_the_allowlist_and_fails_the_build() -> None:
    scan = next(s["run"] for s in _secrets_job()["steps"] if "Scan" in s.get("name", ""))
    assert "gitleaks git" in scan
    assert "--no-git" not in scan
    assert "--config .gitleaks.toml" in scan and "--redact" in scan
    assert "--exit-code 1" in scan
    assert (REPO_ROOT / ".gitleaks.toml").exists()
