"""`python -m portolan.signals`: the flag gate and the targets commands."""

from __future__ import annotations

import json

import pytest

from portolan.signals.__main__ import main


def test_scan_is_refused_before_anything_runs_when_the_flag_is_off(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("PORTOLAN_EXTERNAL_SIGNALS", "false")

    assert main(["scan", "--llm", "fake", "--embed", "none"]) == 2
    assert "PORTOLAN_EXTERNAL_SIGNALS=true" in capsys.readouterr().err


def test_targets_import_and_list(db, tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    targets = tmp_path / "targets.yaml"
    targets.write_text(
        "accounts:\n"
        "  - crm_id: PUB-001\n"
        "    feeds:\n"
        "      - url: https://rss.app/feeds/abc.xml\n"
        "        kind: linkedin\n"
        "        linkedin_company_url: https://www.linkedin.com/company/example-health/\n"
        "    pages:\n"
        "      - https://www.example-health.test/newsroom\n"
    )

    assert main(["targets", "import", str(targets)]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "added": 2,
        "updated": 0,
        "unchanged": 0,
        "deactivated": 0,
    }
    assert main(["targets", "list"]) == 0
    listed = capsys.readouterr().out.splitlines()
    assert [line.split("\t")[1:4] for line in listed] == [
        ["PUB-001", "feed", "active"],
        ["PUB-001", "page", "active"],
    ]
