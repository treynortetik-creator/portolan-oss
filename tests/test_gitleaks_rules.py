"""The custom rules in `.gitleaks.toml` catch the secret shapes this repo holds, and pass the
fixed test credentials. Runs the real gitleaks binary over files written to `tmp_path`.

Every secret here is generated at run time (`secrets.token_*`), so no real value and no
key-shaped literal is ever in this file. gitleaks runs with `--redact`, and only rule ids and
file names are read back from its report.

Skips where gitleaks is not installed (a dev machine without it, the docker image), except in
GitHub Actions, where the `test` job installs the pinned binary: there a missing binary fails."""

from __future__ import annotations

import json
import os
import secrets
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG = REPO_ROOT / ".gitleaks.toml"

if shutil.which("gitleaks") is None:
    if os.environ.get("GITHUB_ACTIONS") == "true":
        raise RuntimeError("gitleaks is not on PATH; CI's test job must install it")
    pytest.skip("gitleaks is not installed", allow_module_level=True)

DSN_RULE = "portolan-postgres-dsn-password"
OPENROUTER_RULE = "portolan-openrouter-key"
ANTHROPIC_RULE = "portolan-anthropic-key"


def _scan(directory: Path) -> dict[str, set[str]]:
    report = directory.parent / "gitleaks-report.json"
    proc = subprocess.run(
        [
            "gitleaks",
            "dir",
            "--config",
            str(CONFIG),
            "--redact",
            "--no-banner",
            "--report-format",
            "json",
            "--report-path",
            str(report),
            str(directory),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode in (0, 1), proc.stderr[-2000:]
    found: dict[str, set[str]] = {}
    for f in json.loads(report.read_text(encoding="utf-8") or "[]"):
        found.setdefault(Path(f["File"]).name, set()).add(f["RuleID"])
    return found


@pytest.fixture(scope="module")
def findings(tmp_path_factory: pytest.TempPathFactory) -> dict[str, set[str]]:
    d = tmp_path_factory.mktemp("gl") / "fixtures"
    d.mkdir()
    password = secrets.token_urlsafe(24)
    openrouter = "sk-or-" + "v1-" + secrets.token_hex(32)
    anthropic = "sk-" + "ant-api03-" + secrets.token_urlsafe(60)
    files = {
        # should be caught
        "dsn_in_code.py": f'DSN = "postgresql://postgres:{password}@shinkansen.proxy.rlwy.net:41234/railway"\n',
        "dsn_in_log.txt": f"connecting to postgres://postgres:{password}@postgres.railway.internal:5432/railway\n",
        "openrouter_in_prose.md": f"the key was {openrouter} yesterday\n",
        "anthropic_in_prose.md": f"ANTHROPIC key {anthropic}\n",
        # should pass
        "ci.yml": "      DATABASE_URL: postgresql://portolan:portolan@localhost:5432/portolan_test\n",
        "compose.yml": "      DATABASE_URL: postgresql://portolan:portolan@db:5432/portolan_test\n",
        "runbook.md": (
            "`railway run --service Postgres -- sh -c 'psql \"postgresql://$PGUSER:$PGPASSWORD@"
            "$RAILWAY_TCP_PROXY_DOMAIN:$RAILWAY_TCP_PROXY_PORT/$PGDATABASE\"'`\n"
        ),
        "redacted.txt": "extract: connection failed postgresql://portolan:***@db.internal:5432/portolan\n",
        "fstring.py": 'dsn = f"postgresql://u:{FAKE_PASSWORD}@{REMOTE_HOST}:5432/portolan_test"\n',
        "hand_typed_fixture.py": 'bad = "postgresql://portolan:super-secret-password@localhost:1/x"\n',
    }
    for name, body in files.items():
        (d / name).write_text(body, encoding="utf-8")
    return _scan(d)


@pytest.mark.parametrize(
    ("filename", "rule"),
    [
        ("dsn_in_code.py", DSN_RULE),
        ("dsn_in_log.txt", DSN_RULE),
        ("openrouter_in_prose.md", OPENROUTER_RULE),
        ("anthropic_in_prose.md", ANTHROPIC_RULE),
    ],
)
def test_custom_rule_catches_the_shape(
    findings: dict[str, set[str]], filename: str, rule: str
) -> None:
    assert rule in findings.get(filename, set()), findings


@pytest.mark.parametrize(
    "filename",
    ["ci.yml", "compose.yml", "runbook.md", "redacted.txt", "fstring.py", "hand_typed_fixture.py"],
)
def test_known_non_secrets_pass(findings: dict[str, set[str]], filename: str) -> None:
    assert filename not in findings, findings[filename]


def test_config_extends_the_default_rules() -> None:
    text = CONFIG.read_text(encoding="utf-8")
    assert "useDefault = true" in text
