"""Tests for the label judge (`portolan.training.label_judge`) and its silver
path (`scripts/data/repair_silver.py --label-judge`, the builder's provenance
check). Every judge call is a stub; nothing here touches the network."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from portolan.extract.prompt import render_labeling_rules
from portolan.training.label_judge import (
    JUDGE_SCHEMA,
    JudgeFact,
    JudgeLogEntry,
    judge_facts,
    judge_record,
    judge_system_prompt,
    judge_user_message,
    parse_judge_reply,
    retype_allowed,
)
from portolan.training.records import (
    LabelAudit,
    TrainingFact,
    TrainingRecord,
    label_judge_log_path,
    records_path,
    repaired_records_path,
)
from tests._paths import require_repo_path

MODEL = "deepseek/deepseek-v4.1-flash"
TEXT = (
    "[L1] Customer: The pump alarms failed twice this week on the ward. "
    "[L2] Vendor: Thanks so much, we really value working with you. "
    "[L3] Customer: Please add a quiet-hours setting to the alarm panel."
)
FACTS = [
    JudgeFact("complaint", "The pump alarms failed twice", "The pump alarms failed twice"),
    JudgeFact("praise", "The vendor values the customer", "we really value working with you"),
    JudgeFact("commitment", "The customer asked for quiet hours", "Please add a quiet-hours"),
]


def answer(fid: int, **over: Any) -> dict[str, Any]:
    row = {
        "id": fid,
        "reason": "the source says so",
        "stated_as_true": True,
        "durable": True,
        "claim_supported": True,
        "repeats": None,
        "best_type": FACTS[fid].fact_type if fid < len(FACTS) else "other",
    }
    return row | over


REPLY = {
    "facts": [
        answer(0),
        answer(1, durable=False, reason="a courtesy"),
        answer(2, best_type="request", reason="it is an ask"),
    ]
}


def test_answers_become_keep_drop_and_retype_by_rule() -> None:
    out = parse_judge_reply(REPLY, FACTS)
    assert [(j.verdict, j.fact_type) for j in out] == [
        ("keep", "complaint"),
        ("drop", "praise"),
        ("retype", "request"),
    ]
    assert out[1].reason == "a courtesy"


@pytest.mark.parametrize(
    "flag", [{"stated_as_true": False}, {"claim_supported": False}, {"repeats": 0}]
)
def test_any_failed_answer_or_an_earlier_repeat_drops(flag: dict[str, Any]) -> None:
    [_, second, _] = parse_judge_reply({"facts": [answer(0), answer(1, **flag), answer(2)]}, FACTS)
    assert second is not None and second.verdict == "drop"


def test_malformed_missing_and_repeated_answers_give_no_verdict() -> None:
    rows = [
        answer(0, durable="yes"),  # not a boolean
        answer(1, best_type="gossip"),  # not a type
        answer(1),  # the same id twice: neither answer is trusted
    ]
    assert parse_judge_reply({"facts": rows}, FACTS) == [None, None, None]
    assert parse_judge_reply(None, FACTS) == [None, None, None]
    assert parse_judge_reply({"facts": "nope"}, FACTS) == [None, None, None]


def test_a_repeat_must_point_at_an_earlier_fact() -> None:
    out = parse_judge_reply({"facts": [answer(0, repeats=2), answer(1), answer(2)]}, FACTS)
    assert out[0] is not None and out[0].verdict == "keep"


def test_system_prompt_carries_the_production_rules_verbatim() -> None:
    system = judge_system_prompt("Company: Veltrona Water Systems")
    assert render_labeling_rules() in system
    assert "## What's worth extracting (the durability bar)" in system
    assert "Not this:" in system  # the checklist's counter-examples
    assert "## Company context\nCompany: Veltrona Water Systems" in system
    assert judge_system_prompt(None).count("## Company context") == 0
    user = judge_user_message(TEXT, FACTS, "2026-03-06")
    assert user.startswith("Source date: 2026-03-06")
    assert json.loads(user.split("## Facts to check\n", 1)[1])[2]["claim"] == FACTS[2].claim
    assert set(JUDGE_SCHEMA["properties"]["facts"]["items"]["required"]) >= {
        "stated_as_true",
        "durable",
        "claim_supported",
        "repeats",
        "best_type",
    }


def test_judge_facts_calls_once_then_reads_the_cache() -> None:
    calls: list[str] = []

    def call(model: str, system: str, user: str):
        calls.append(model)
        return REPLY, 0.0004

    result, how = judge_facts(TEXT, FACTS, model=MODEL, call=call, company_context=None)
    assert how == "called" and calls == [MODEL] and result.cost_usd == 0.0004
    entry = JudgeLogEntry.from_result(result, subject="S-1", facts=FACTS, created="t")
    again, how = judge_facts(
        TEXT,
        FACTS,
        model=MODEL,
        call=call,
        company_context=None,
        cache={entry.key: JudgeLogEntry.model_validate_json(entry.model_dump_json())},
    )
    assert how == "cached" and calls == [MODEL]
    assert [j.verdict for j in again.judgements] == ["keep", "drop", "retype"]
    # another model is another key
    _r, how = judge_facts(
        TEXT, FACTS, model="qwen/qwen3-235b-a22b-2507", call=call, company_context=None
    )
    assert how == "called"


def test_judge_facts_never_raises_and_respects_the_budget() -> None:
    def broken(model: str, system: str, user: str):
        raise RuntimeError("status 503")

    result, how = judge_facts(TEXT, FACTS, model=MODEL, call=broken, company_context=None)
    assert how == "called" and "503" in (result.error or "")
    assert result.judgements == [None, None, None] and not result.complete
    skipped, how = judge_facts(
        TEXT, FACTS, model=MODEL, call=broken, company_context=None, budget_left=lambda: False
    )
    assert how == "skipped" and skipped.error == "not_attempted"


def test_retype_must_pass_the_forty_word_quote_rule() -> None:
    assert retype_allowed("other", "word " * 60)
    assert retype_allowed("complaint", "word " * 40)
    assert not retype_allowed("complaint", "word " * 41)
    assert not retype_allowed("commitment", None)


def _fact(fact_type: str, claim: str, quote: str, text: str = TEXT) -> TrainingFact:
    start = text.index(quote)
    return TrainingFact(
        fact_type=fact_type,
        claim=claim,
        quote=quote,
        quote_span=(start, start + len(quote)),
        locator="L1",
        sentiment=0,
        confidence=0.9,
    )


def _record(facts: list[TrainingFact], record_id: str = "S-1", text: str = TEXT) -> TrainingRecord:
    return TrainingRecord(
        record_id=record_id,
        corpus="cfpb",
        source_id=record_id,
        kind="ticket",
        license="public domain",
        text=text,
        facts=facts,
        teachers=["moonshotai/kimi-k2"],
        agreement=1.0,
        created="2026-09-22T12:00:00-07:00",
        mode="single",
        user_message_inputs={"source_date": "2026-03-06"},
        label_audit=LabelAudit(
            status="complete",
            teacher_facts=len(facts),
            kept=len(facts),
            regated=0,
            repaired=0,
            withdrawn=0,
            unresolved=0,
            unlogged=0,
            audited="t",
        ),
    )


RECORD_FACTS = [_fact(f.fact_type, f.claim, f.quote) for f in FACTS]


def test_judge_record_drops_and_retypes_without_touching_claims() -> None:
    record = _record(RECORD_FACTS)
    result, _how = judge_facts(
        TEXT, FACTS, model=MODEL, call=lambda m, s, u: (REPLY, 0.0), company_context=None
    )
    out = judge_record(record, result)
    assert [(f.fact_type, f.claim, f.quote, f.teacher_fact_type) for f in out.facts] == [
        ("complaint", FACTS[0].claim, FACTS[0].quote, None),
        ("request", FACTS[2].claim, FACTS[2].quote, "commitment"),
    ]
    audit = out.label_audit
    assert (audit.judge_model, audit.judge_dropped, audit.judge_retyped) == (MODEL, 1, 1)
    assert (audit.judge_unresolved, audit.status) == (0, "complete")


def test_a_fact_without_a_verdict_makes_the_record_incomplete() -> None:
    long_quote = " ".join(f"word{i}" for i in range(45))
    text = f"{TEXT} [L4] Customer: {long_quote}."
    facts = [
        *[_fact(f.fact_type, f.claim, f.quote, text) for f in FACTS[:2]],
        _fact("other", "A long statement", long_quote, text),
    ]
    reply = {"facts": [answer(0), answer(2, best_type="complaint")]}  # no answer for id 1
    result, _how = judge_facts(
        text,
        [JudgeFact(f.fact_type, f.claim, f.quote) for f in facts],
        model=MODEL,
        call=lambda m, s, u: (reply, 0.0),
        company_context=None,
    )
    out = judge_record(_record(facts, text=text), result)
    # id 1 had no answer; id 2's retype to a quote-required type fails the 41+ word rule
    assert len(long_quote.split()) > 40
    assert [f.fact_type for f in out.facts] == ["complaint", "praise", "other"]
    assert out.label_audit.judge_unresolved == 2 and out.label_audit.status == "incomplete"


# --------------------------------------------------------------------------
# Silver path: repair_silver.py --label-judge, and the builder's check
# --------------------------------------------------------------------------


_REPAIR = require_repo_path("scripts", "data", "repair_silver.py")
_BUILDER = require_repo_path("scripts", "data", "build_training_set.py")


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


repair_script = _load("repair_silver_for_judge_tests", _REPAIR)
builder = _load("build_training_set_for_judge_tests", _BUILDER)


def _write_var(tmp_path: Path) -> Path:
    var = tmp_path / "var"
    path = records_path(var, "cfpb")
    path.parent.mkdir(parents=True)
    rows = [_record(RECORD_FACTS, "S-1"), _record([], "S-2")]
    for row in rows:
        row.label_audit = None
        row.agreement = 1.0
    path.write_text("".join(r.model_dump_json() + "\n" for r in rows))
    return var


def _run(var: Path, call, budget_left=lambda: True):
    return repair_script.repair_corpus(
        var_root=var,
        out_root=var,
        corpus="cfpb",
        call=None,
        budget_left=lambda: True,
        company_context="Company: a consumer bank",
        judge_model=MODEL,
        judge_call=call,
        judge_budget_left=budget_left,
        now=lambda: "2026-09-22T16:00:00-07:00",
    )


def test_repair_silver_applies_the_judge_and_caches_its_calls(tmp_path: Path) -> None:
    var = _write_var(tmp_path)
    seen: list[str] = []

    def call(model: str, system: str, user: str):
        seen.append(user)
        return REPLY, 0.0003

    result = _run(var, call)
    assert len(seen) == 1  # S-2 has no facts: nothing to judge
    assert "Source date: 2026-03-06" in seen[0]
    assert (result.judge_calls, result.judge_dropped, result.judge_retyped) == (1, 1, 1)
    out = {
        r["record_id"]: r
        for r in map(json.loads, repaired_records_path(var, "cfpb").read_text().splitlines())
    }
    assert [f["fact_type"] for f in out["S-1"]["facts"]] == ["complaint", "request"]
    assert out["S-1"]["label_audit"]["judge_model"] == MODEL
    assert out["S-1"]["label_audit"]["status"] == "complete"
    assert out["S-2"]["label_audit"]["judge_model"] is None
    [entry] = label_judge_log_path(var, "cfpb").read_text().splitlines()
    assert json.loads(entry)["subject"] == "S-1"
    summary = json.loads((var / "data" / "silver" / "cfpb" / "repair_summary.json").read_text())
    assert summary["judge_dropped"] == 1 and summary["judge_spend_usd"] == pytest.approx(0.0003)

    rerun: list[str] = []
    again = _run(var, lambda m, s, u: (rerun.append(u), (REPLY, 0.0))[1])
    assert rerun == [] and again.judge_cached == 1 and again.judge_dropped == 1


def test_spent_judge_budget_leaves_the_record_incomplete(tmp_path: Path) -> None:
    var = _write_var(tmp_path)
    result = _run(var, lambda m, s, u: (REPLY, 0.0), budget_left=lambda: False)
    assert (result.judge_calls, result.judge_skipped, result.judge_unresolved) == (0, 1, 3)
    first = json.loads(repaired_records_path(var, "cfpb").read_text().splitlines()[0])
    assert first["label_audit"]["status"] == "incomplete"
    assert len(first["facts"]) == 3
    assert not label_judge_log_path(var, "cfpb").exists()


def test_builder_refuses_a_judge_off_the_allowlist() -> None:
    record = _record(RECORD_FACTS)
    ok = record.model_copy(
        update={"label_audit": record.label_audit.model_copy(update={"judge_model": MODEL})}
    )
    closed = record.model_copy(
        update={
            "label_audit": record.label_audit.model_copy(
                update={"judge_model": "anthropic/claude-sonnet-5"}
            )
        }
    )
    assert "judge_not_allowed" not in builder.provenance_failures(ok)
    assert "judge_not_allowed" in builder.provenance_failures(closed)
