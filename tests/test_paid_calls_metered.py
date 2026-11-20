"""Every `OpenRouterClient` in the repo names a spend key, and one built without a
`spend_meter` sits in a module that meters its calls another way (CLAUDE.md: every paid
model call is metered). Guardrails review 2026-09-22, row 33: three scripts made paid
calls through this client into an in-memory sink and no meter at all.

The constructor already refuses a missing key at run time; this reads the source, so a
script that no test ever runs is held to the same rule.
"""

from __future__ import annotations

import ast
from pathlib import Path

from tests._paths import require_repo_path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCANNED_ROOTS = (REPO_ROOT / "portolan", REPO_ROOT / "corpus", require_repo_path("scripts"))

# Modules whose OpenRouterClient carries no spend_meter, and what meters those calls.
METERED_ELSEWHERE = {
    "portolan/llm/factory.py": "the app's usage sink (Postgres usage_events)",
    "portolan/pipeline.py": "the app's usage sink; worker clones copy the parent's meter",
    "corpus/generator/prose.py": "corpus/generator/cli.py records each call's cost",
    "corpus/generator/v2/writer.py": "RunBudget records each call's cost",
    "corpus/generator/v2/completeness.py": "TeacherRunBudget records each teacher call's cost",
    "scripts/data/repair_silver.py": "records each call's cost itself",  # path-guard: data
    "scripts/data/silver_label.py": "records each call's cost itself",  # path-guard: data
}


def _client_calls() -> list[tuple[str, int, set[str]]]:
    calls = []
    for root in SCANNED_ROOTS:
        for path in sorted(root.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
                if name == "OpenRouterClient":
                    keywords = {kw.arg for kw in node.keywords if kw.arg}
                    calls.append((path.relative_to(REPO_ROOT).as_posix(), node.lineno, keywords))
    return calls


def test_the_scan_finds_the_known_clients() -> None:
    files = {file for file, _, _ in _client_calls()}
    assert set(METERED_ELSEWHERE) <= files
    assert "scripts/bench_two_pass.py" in files  # path-guard: data


def test_every_openrouter_client_names_a_spend_key() -> None:
    missing = [f"{file}:{line}" for file, line, kws in _client_calls() if "spend_key" not in kws]
    assert not missing, f"OpenRouterClient built with no spend_key: {missing}"


def test_a_client_without_a_meter_is_metered_elsewhere() -> None:
    unmetered = [
        f"{file}:{line}"
        for file, line, kws in _client_calls()
        if "spend_meter" not in kws and file not in METERED_ELSEWHERE
    ]
    assert not unmetered, (
        f"OpenRouterClient with no spend_meter: {unmetered}. Pass "
        "spend_meter=SpendMeter(default_spend_path(REPO_ROOT / 'var')) so the call counts "
        "against the machine's one ceiling."
    )
