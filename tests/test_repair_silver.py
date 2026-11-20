"""Tests for `scripts/data/repair_silver.py`: snapshot reading, outputs
written next to the live files without touching them, the call cache, and
the budget stop. Every teacher call is a stub."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

from portolan.training.records import (
    DroppedFact,
    TrainingFact,
    TrainingRecord,
    dropped_path,
    records_path,
    repaired_records_path,
    repairs_log_path,
)
from tests._paths import require_repo_path

_SCRIPT = require_repo_path("scripts", "data", "repair_silver.py")


def _load():
    spec = importlib.util.spec_from_file_location("repair_silver_script", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["repair_silver_script"] = module
    spec.loader.exec_module(module)
    return module


script = _load()

KIMI = "moonshotai/kimi-k2"
TEXT = (
    "[L1] Consumer: Comerica sent my debit card loaded with funds to somewhere else. "
    "[L2] They put me on a 5 year payment plan and charged $5k per month."
)


def _record(record_id: str, facts: list[TrainingFact], agreement: float) -> TrainingRecord:
    return TrainingRecord(
        record_id=record_id,
        corpus="cfpb",
        source_id=record_id,
        kind="ticket",
        license="public domain",
        text=TEXT,
        facts=facts,
        teachers=[KIMI],
        agreement=agreement,
        created="2026-09-22T12:00:00-07:00",
        mode="single",
    )


def _kept() -> TrainingFact:
    quote = "Comerica sent my debit card"
    start = TEXT.index(quote)
    return TrainingFact(
        fact_type="complaint",
        claim="Comerica sent the consumer's card elsewhere",
        quote=quote,
        quote_span=(start, start + len(quote)),
        locator="L1",
    )


def _drop(source_id: str, claim: str, quote: str, reason: str) -> DroppedFact:
    return DroppedFact(
        source_id=source_id,
        reason=reason,
        fact={"fact_type": "complaint", "claim": claim, "quote": quote, "locator": "L2"},
        teacher=KIMI,
        created="2026-09-22T11:59:00-07:00",
    )


def _write_var(tmp_path: Path) -> Path:
    var = tmp_path / "var"
    records = records_path(var, "cfpb")
    records.parent.mkdir(parents=True)
    rows = [
        # 1 kept of 3: one false drop (5-year) and one real failure
        _record("S-1", [_kept()], 1 / 3),
        # a gate-made negative: its one fact still fails
        _record("S-2", [], 0.0),
    ]
    records.write_text("".join(r.model_dump_json() + "\n" for r in rows))
    drops = [
        _drop(
            "S-1",
            "The consumer is on a 5-year payment plan",
            "a 5 year payment plan",
            "overlap_gate_numeral",
        ),
        _drop(
            "S-1",
            "The consumer said Chase Bank charged them",
            "charged $5k per month",
            "overlap_gate_name",
        ),
        _drop(
            "S-2",
            "The consumer said Wells Fargo lost the card",
            "Comerica sent my debit card",
            "overlap_gate_name",
        ),
    ]
    dropped_path(var, "cfpb").write_text("".join(d.model_dump_json() + "\n" for d in drops))
    return var


def _stub(replies: dict[str, dict[str, Any]], calls: list[str]):
    def call(model: str, system: str, user: str):
        for needle, reply in replies.items():
            if needle in user:
                calls.append(needle)
                return reply, 0.001
        raise AssertionError("unexpected repair call")

    return call


_CORRECT = {
    "decision": "correct",
    "reason": "the text names no bank",
    "fact_type": "complaint",
    "claim": "The company charged the consumer $5,000 per month",
    "quote": "charged $5k per month",
    "locator": "L2",
    "entities": [],
}
_WITHDRAW = {**_CORRECT, "decision": "withdraw", "reason": "the text never names Wells Fargo"}
_REPLIES = {"Chase Bank": _CORRECT, "Wells Fargo": _WITHDRAW}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_partial_last_line_is_left_out(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    path.write_bytes(b'{"a": 1}\n{"b": 2}\n{"c": ')
    assert script.read_complete_lines(path) == b'{"a": 1}\n{"b": 2}\n'


def test_repair_writes_new_files_and_never_touches_the_live_ones(tmp_path: Path) -> None:
    var = _write_var(tmp_path)
    live = {p: _digest(p) for p in (records_path(var, "cfpb"), dropped_path(var, "cfpb"))}
    calls: list[str] = []
    result = script.repair_corpus(
        var_root=var,
        out_root=var,
        corpus="cfpb",
        call=_stub(_REPLIES, calls),
        budget_left=lambda: True,
        company_context=None,
        now=lambda: "2026-09-22T16:00:00-07:00",
    )
    assert {p: _digest(p) for p in live} == live
    assert sorted(calls) == ["Chase Bank", "Wells Fargo"]
    assert (result.regated_facts, result.repaired_facts, result.withdrawn_facts) == (1, 1, 1)

    out = {
        r["record_id"]: r
        for r in map(json.loads, repaired_records_path(var, "cfpb").read_text().splitlines())
    }
    s1 = out["S-1"]
    assert s1["label_audit"]["status"] == "complete"
    assert [(f["regated"], f["repaired"]) for f in s1["facts"]] == [
        (False, False),
        (True, False),
        (False, True),
    ]
    assert s1["facts"][2]["claim"] == _CORRECT["claim"]
    # every teacher fact of S-2 was withdrawn: complete, and still never a negative
    assert out["S-2"]["facts"] == []
    assert out["S-2"]["label_audit"]["teacher_facts"] == 1
    assert len(repairs_log_path(var, "cfpb").read_text().splitlines()) == 2
    summary = json.loads((var / "data" / "silver" / "cfpb" / "repair_summary.json").read_text())
    assert summary["inputs"]["records.jsonl"]["sha256"] == live[records_path(var, "cfpb")]


def test_rerun_uses_the_repair_log_instead_of_calling_again(tmp_path: Path) -> None:
    var = _write_var(tmp_path)
    kwargs = dict(
        var_root=var, out_root=var, corpus="cfpb", budget_left=lambda: True, company_context=None
    )
    first: list[str] = []
    script.repair_corpus(call=_stub(_REPLIES, first), **kwargs)
    second: list[str] = []
    result = script.repair_corpus(call=_stub(_REPLIES, second), **kwargs)
    assert len(first) == 2 and second == []
    assert result.cached == 2 and result.repaired_facts == 1


def test_spent_budget_leaves_facts_unresolved_and_records_incomplete(tmp_path: Path) -> None:
    var = _write_var(tmp_path)
    calls: list[str] = []
    result = script.repair_corpus(
        var_root=var,
        out_root=var,
        corpus="cfpb",
        call=_stub(_REPLIES, calls),
        budget_left=lambda: False,
        company_context=None,
    )
    assert calls == []
    assert result.outcomes == {"not_attempted": 2}
    statuses = [
        json.loads(line)["label_audit"]["status"]
        for line in repaired_records_path(var, "cfpb").read_text().splitlines()
    ]
    assert statuses == ["incomplete", "incomplete"]


def test_teacher_override_sends_the_call_to_that_model(tmp_path: Path) -> None:
    var = _write_var(tmp_path)
    models: list[str] = []

    def call(model: str, system: str, user: str):
        models.append(model)
        return _WITHDRAW, 0.0

    script.repair_corpus(
        var_root=var,
        out_root=tmp_path / "out",
        corpus="cfpb",
        call=call,
        budget_left=lambda: True,
        company_context=None,
        teacher_override="deepseek/deepseek-v3.2",
    )
    assert models == ["deepseek/deepseek-v3.2", "deepseek/deepseek-v3.2"]
    assert repaired_records_path(tmp_path / "out", "cfpb").exists()
    assert not repaired_records_path(var, "cfpb").exists()
