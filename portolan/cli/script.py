"""`portolan script <file.json>`: a scenario is a list of steps, each a
command to run and, optionally, what its result must look like.

    {"name": "demo",
     "steps": [
       {"do": "ask \\"Which accounts asked for Tidewater Link to work with Epic?\\"",
        "expect": {"outcome": "answered", "citations": ">=2", "cited_notes": "contains signal-"}},
       {"do": "note no-such-note", "exit": 3},
       {"expect": ["citations>=2"]}
     ]}

A step with no `do` checks the previous step's result again. `exit` is the
exit code the step must end with (default 0). The scenario stops at the
first step that fails: exit 1 for a failed expectation, or the command's
own exit code (2, 3 or 4) when a step that should have worked did not.
"""

from __future__ import annotations

import json
import shlex
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from portolan.cli.core import EXIT_FAILED, UsageError
from portolan.cli.expect import ExpectError, check, parse_expect


@dataclass
class Outcome:
    """What one command did: its data (None when it failed before producing
    any), its exit code, an explanation, and how to print the data."""

    data: dict[str, Any] | None
    code: int
    message: str | None = None
    render: Callable[[dict[str, Any]], str] | None = None

    def text(self) -> str:
        if self.data is not None and self.render is not None:
            return self.render(self.data)
        return self.message or ""


def load_scenario(path: Path) -> dict[str, Any]:
    try:
        spec = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise UsageError(f"cannot read the script {path}: {exc}") from exc
    except ValueError as exc:
        raise UsageError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(spec, dict) or not isinstance(spec.get("steps"), list) or not spec["steps"]:
        raise UsageError(f"{path} needs an object with a non-empty 'steps' list")
    for n, step in enumerate(spec["steps"], start=1):
        if not isinstance(step, dict) or not ({"do", "expect"} & set(step)):
            raise UsageError(f"{path}: step {n} needs a 'do' or an 'expect'")
    return spec


def _tokens(do: str) -> list[str]:
    try:
        tokens = shlex.split(do)
    except ValueError as exc:
        raise UsageError(f"cannot split the command {do!r}: {exc}") from exc
    if tokens and tokens[0] == "portolan":
        tokens = tokens[1:]
    if not tokens:
        raise UsageError("a step's 'do' is empty")
    return tokens


def run_scenario(
    spec: dict[str, Any], run: Callable[[list[str]], Outcome], *, restart: bool = True
) -> tuple[int, dict[str, Any]]:
    """Run the steps in order. `run(tokens)` executes one command. Returns
    the exit code and a report: name, ok, and one record per step."""
    report: dict[str, Any] = {"name": spec.get("name", "script"), "ok": True, "steps": []}
    if restart:
        run(["new"])
    last: dict[str, Any] = {}
    for number, step in enumerate(spec["steps"], start=1):
        record: dict[str, Any] = {"step": number, "do": step.get("do"), "ok": True, "checks": []}
        report["steps"].append(record)
        expected_exit = int(step.get("exit", 0))
        try:
            conditions = parse_expect(step.get("expect"))
        except ExpectError as exc:
            raise UsageError(f"step {number}: {exc}") from exc
        if step.get("do"):
            outcome = run(_tokens(step["do"]))
            record["exit"] = outcome.code
            last = {**(outcome.data or {}), "exit": outcome.code}
            if outcome.message:
                last.setdefault("message", outcome.message)
        else:
            outcome = Outcome(last, 0)
            last = {**last, "exit": 0}
        if outcome.code != expected_exit:
            record["ok"] = report["ok"] = False
            record["error"] = f"exit {outcome.code}, expected {expected_exit}"
            record["output"] = outcome.text()
            unexpected = (
                outcome.code if expected_exit == 0 and outcome.code != EXIT_FAILED else EXIT_FAILED
            )
            return unexpected, report
        record["checks"] = check(conditions, last)
        failed = [c for c in record["checks"] if not c["ok"]]
        if failed:
            record["ok"] = report["ok"] = False
            record["error"] = "; ".join(
                f"{c['expect']} (got {c['got']!r}{': ' + c['error'] if c['error'] else ''})"
                for c in failed
            )
            record["output"] = outcome.text()
            return EXIT_FAILED, report
    return 0, report


def render_report(report: dict[str, Any]) -> str:
    lines = [f"script: {report['name']}"]
    for record in report["steps"]:
        mark = "ok  " if record["ok"] else "FAIL"
        what = record["do"] or "(check the previous result)"
        lines.append(f"  {mark} step {record['step']}: {what}")
        for c in record["checks"]:
            lines.append(f"       {'ok' if c['ok'] else 'FAIL'}  {c['expect']}  (got {c['got']!r})")
        if not record["ok"]:
            lines.append(f"       {record['error']}")
            if record.get("output"):
                lines.extend(f"       | {line}" for line in record["output"].splitlines()[:40])
    verdict = "PASS" if report["ok"] else "FAIL"
    passed = sum(1 for r in report["steps"] if r["ok"])
    lines.append(f"{verdict}: {passed}/{len(report['steps'])} step(s)")
    return "\n".join(lines)
