"""Tests for `scripts/data/v2_estimate.py` (the v2 cost model, WP11).

The measured logs are small synthetic copies of the real ones' shapes
(WP4 `calls.jsonl` and `run_report.json`, WP5 `teacher_calls.jsonl` and
`docs/*.json`, WP6 calls, WP9 disputes, an MLX log), written under
`tmp_path`, so every number the model prints can be checked by hand.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from corpus.generator.v2.models import Split
from corpus.generator.v2.rounds import PILOT, WorldSpec
from tests._paths import require_repo_path

_SCRIPT = require_repo_path("scripts", "data", "v2_estimate.py")


def _load():
    spec = importlib.util.spec_from_file_location("v2_estimate_script", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["v2_estimate_script"] = module
    spec.loader.exec_module(module)
    return module


E = _load()
T0 = "2026-09-23T01:00:"


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _writer(plan_id, kind, sec, cost, pt=1000, ct=500):
    return {
        "plan_id": plan_id,
        "call_kind": kind,
        "model": "qwen/qwen3-32b",
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "cost_usd": cost,
        "error": None,
        "at": f"{T0}{sec:02d}+00:00",
    }


def _teacher(plan_id, sec, cost, role="completeness", attempt=0):
    return {
        "plan_id": plan_id,
        "role": role,
        "attempt": attempt,
        "model": "moonshotai/kimi-k2",
        "cost_usd": cost,
        "at": f"{T0}{sec:02d}+00:00",
    }


def _doc_state(path: Path, status: str, attempts: int, candidates: int = 0) -> None:
    facts = [{"role": "completeness", "triage": "candidate"}] * candidates + [
        {"role": "completeness", "triage": "confirmed"}
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"status": status, "attempts": [{"facts": facts}] * attempts}))


@pytest.fixture()
def var_root(tmp_path: Path) -> Path:
    root = tmp_path / "var"
    wp4 = root / "data/synth_v2/smoke_wp4fix"
    # call doc: initial 00 -> repair 50 -> email initial 58 (last call: median of initial)
    _jsonl(
        wp4 / "calls.jsonl",
        [
            _writer("S-call-00001", "initial", 0, 0.002),
            _writer("S-call-00001", "repair", 50, 0.001, pt=400, ct=100),
            _writer("S-email-00002", "initial", 58, 0.001, pt=600, ct=300),
        ],
    )
    (wp4 / "run_report.json").write_text(
        json.dumps(
            {
                "per_document": [
                    {"plan_id": "S-call-00001", "facts_planted": 4, "facts_labeled": 3},
                    {"plan_id": "S-email-00002", "facts_planted": 2, "facts_labeled": 1},
                ]
            }
        )
    )
    measure = wp4 / "wp5_measure"
    _jsonl(
        measure / "teacher_calls.jsonl",
        [
            _teacher("S-call-00001", 0, 0.01),
            _teacher("S-email-00002", 30, 0.005),
            _teacher("S-email-00002", 40, 0.005),
            _teacher("S-email-00002", 45, 0.5, attempt=1),  # a re-check, not first pass
        ],
    )
    _doc_state(measure / "docs/S-call-00001.json", "needs_regeneration", 1, candidates=2)
    _doc_state(measure / "docs/S-email-00002.json", "accepted", 1, candidates=0)
    wp5 = root / "data/synth_v2/smoke_wp5/final"
    _jsonl(
        wp5 / "teacher_calls.jsonl",
        [_teacher("S-call-00001", 0, 0.02), _teacher("S-call-00001", 10, 0.003, role="audit")],
    )
    _doc_state(wp5 / "docs/S-call-00001.json", "accepted", 2)
    _doc_state(wp5 / "docs/S-ticket-00001.json", "dropped", 2)
    # A run of the current rule (pass .10 or later): one of four documents
    # regenerated and kept, two label-judge calls ten seconds apart.
    current = root / "data/synth_v2/smoke_pilot_prep/train/world-x/completeness"
    (current / "qa_report.json").parent.mkdir(parents=True, exist_ok=True)
    (current / "qa_report.json").write_text(json.dumps({"pass_version": "wp5-2026-09-23.13"}))
    for i in range(4):
        _doc_state(current / f"docs/S-call-0000{i}.json", "accepted", 2 if i == 0 else 1)
    regenerated = current / "docs/S-call-00000.json"
    regenerated.write_text(
        json.dumps({**json.loads(regenerated.read_text()), "regeneration_tried": True})
    )
    _jsonl(
        current / "label_judge.jsonl",
        [
            {"cost_usd": 0.0004, "created": f"{T0}00+00:00"},
            {"cost_usd": 0.0004, "created": f"{T0}10+00:00"},
        ],
    )
    # An earlier rule's run (pass .9 sorts after .10 as text, not as a number).
    old = root / "data/synth_v2/smoke_wp5/pass9"
    (old / "qa_report.json").parent.mkdir(parents=True, exist_ok=True)
    (old / "qa_report.json").write_text(json.dumps({"pass_version": "wp5-2026-09-22.9"}))
    _doc_state(old / "docs/S-call-00009.json", "accepted", 1)
    (old / "docs/S-call-00009.json").write_text(
        json.dumps({"status": "accepted", "attempts": [{}], "regeneration_tried": True})
    )
    _jsonl(
        root / "data/synth_v2/smoke_wp6/calls.jsonl",
        [_writer("S-near-duplicate-00001", "initial", 0, 0.0005, pt=800, ct=200)],
    )
    votes = [{"cost_usd": 0.004}] * 3
    _jsonl(
        root / "bench/wp9-smoke-2026-09-22/recheck-ticket/disputes.jsonl",
        [{"created": f"{T0}00+00:00", "votes": votes}] * 2,
    )
    log = root / "bench/local-teachers-2026-09-22/label.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("noise\nmlx call 1: in=100 cached=0 out=1000 100.0s finish=json_complete\n")
    return root


def test_seconds_come_from_gaps_and_a_long_pause_splits_runs(tmp_path) -> None:
    rows = [
        _writer("S-call-00001", "initial", 0, 0.001),
        _writer("S-call-00001", "repair", 20, 0.001),
        {**_writer("S-email-00001", "initial", 0, 0.001), "at": "2026-09-23T01:30:00+00:00"},
    ]
    _jsonl(tmp_path / "c.jsonl", rows)
    calls = E.load_calls(tmp_path / "c.jsonl")
    assert [c.seconds for c in calls] == [20.0, 20.0, 20.0]  # repair and last: medians
    assert [c.run for c in calls] == [0, 0, 1]
    assert [c.kind for c in calls] == ["call", "call", "email"]
    assert E.kind_of("S-hard-negative-00004") == "hard-negative"


def test_measure_reads_every_log_by_hand_numbers(var_root) -> None:
    m = E.measure(var_root)
    call, email = m.writer_by_kind["call"], m.writer_by_kind["email"]
    assert (call.n, call.cost_usd, call.calls, call.seconds) == (1, pytest.approx(0.003), 2, 58.0)
    assert (email.cost_usd, email.seconds) == (pytest.approx(0.001), 50.0)
    assert m.teacher_by_kind["email"].cost_usd == pytest.approx(0.01)
    assert m.teacher_by_kind["email"].calls == 2  # the attempt-1 re-check is not first pass
    assert m.teacher_all.cost_usd == pytest.approx(0.01)
    assert m.wp4_labeled_share.value == pytest.approx(4 / 6) and m.wp4_labeled_share.n == 6
    assert (m.wp5_regeneration_share.value, m.wp5_regeneration_share.n) == (0.5, 2)
    # The current rule (pass .10 or later) only: one of four; pass .9 is not counted.
    current = m.wp5_regeneration_share_current
    assert (current.value, current.n) == (0.25, 4)
    assert (m.wp5_kept_after_regeneration.value, m.wp5_kept_after_regeneration.n) == (0.5, 2)
    judge = m.label_judge
    assert (judge.n, judge.calls) == (4, 0.5)
    assert judge.cost_usd == pytest.approx(0.0008 / 4) and judge.seconds == pytest.approx(5.0)
    assert m.wp5_candidates_per_doc.value == 1.0
    assert m.audit.cost_usd == pytest.approx(0.003) and m.audit.n == 1
    assert m.neardup.cost_usd == pytest.approx(0.0005)
    assert m.judge_per_dispute.cost_usd == pytest.approx(0.012)
    assert m.judge_per_dispute.seconds == E.UNMEASURED_JUDGE_SECONDS_PER_DISPUTE
    assert any("judge seconds per dispute are not measured" in n for n in m.notes)
    assert m.local_decode_tps.value == pytest.approx(10.0)


def _counts(strictness: str) -> E.WorldCounts:
    split = Split.TEST if strictness == "test" else Split.TRAIN
    spec = WorldSpec("pharmacy_services", split, 1, strictness=strictness)
    return E.WorldCounts(
        spec, {"call": 2, "email": 1}, neardup_docs=1, planted_facts=10, crm_code_labels=3
    )


def test_estimate_world_train_strictness(var_root) -> None:
    m = E.measure(var_root)
    est = E.estimate_world(_counts("train"), m)
    c = est.cost_usd
    assert c["write (WP4)"] == pytest.approx(2 * 0.003 + 0.001)
    assert c["completeness teacher (WP5)"] == pytest.approx(3 * 0.01)
    # The current rule's regeneration share (0.25), not the section 3.4 rule's.
    assert c["completeness regeneration (WP5)"] == pytest.approx(2 * 0.25 * 0.013 + 0.25 * 0.011)
    # WP6 writes near-duplicates after WP5: no teacher reads them.
    assert c["near-duplicates (WP6)"] == pytest.approx(0.0005)
    # The audit reads every WP4 document (completeness.AUDIT_SHARE), no near-duplicate.
    assert c["audit teacher (WP5)"] == pytest.approx(3 * 1.0 * 0.003)
    assert c["label judge (WP5)"] == pytest.approx(3 * 0.0002)
    assert "three-judge disputes (test)" not in c and not est.unmeasured
    assert est.labeled_facts == pytest.approx(10 * (4 / 6) * (0.75 + 0.25 * 0.5))
    assert est.usd_per_labeled_fact == pytest.approx(est.total_usd / est.labeled_facts)
    assert est.seconds["write (WP4)"] == pytest.approx(2 * 58 + 50)


def test_test_strictness_audits_every_document_and_judges_candidates(var_root) -> None:
    m = E.measure(var_root)
    est = E.estimate_world(_counts("test"), m)
    assert est.cost_usd["audit teacher (WP5)"] == pytest.approx(3 * 0.003)
    assert est.cost_usd["three-judge disputes (test)"] == pytest.approx(4 * 1.0 * 0.012)
    assert est.cost_usd["questions (WP10, spec figure)"] == E.SPEC_QUESTIONS_USD_PER_TEST_WORLD
    assert est.unmeasured and "WP10 not built" in est.unmeasured[0]


def test_local_writers_cost_nothing_and_take_token_time(var_root) -> None:
    m = E.measure(var_root, local_prefill_tps=100.0)
    paid = E.estimate_world(_counts("train"), m)
    local = E.estimate_world(_counts("train"), m, writers_local=True)
    assert local.cost_usd["write (WP4)"] == 0.0
    assert (
        local.cost_usd["completeness teacher (WP5)"] == paid.cost_usd["completeness teacher (WP5)"]
    )
    # call doc: 1,400 prompt tokens at 100/s plus 600 completion at the measured 10/s
    call_secs = 1400 / 100 + 600 / 10
    email_secs = 600 / 100 + 300 / 10
    assert local.seconds["write (WP4)"] == pytest.approx(2 * call_secs + email_secs)
    assert local.total_usd < paid.total_usd


def test_world_counts_come_from_the_planner() -> None:
    counts = E.world_counts(PILOT)
    assert sum(counts.docs_by_kind.values()) == 222 and counts.neardup_docs == 8
    assert counts.planted_facts > 300 and counts.crm_code_labels > 0


def test_main_prints_both_writer_lines_and_writes_json(var_root, tmp_path, capsys) -> None:
    out = tmp_path / "est.json"
    code = E.main(
        [
            "--var-root",
            str(var_root),
            "--worlds",
            "pharmacy_services:train:1,medical_devices:test:201",
            "--json",
            str(out),
        ]
    )
    assert code == 0
    text = capsys.readouterr().out
    assert "| paid rotation |" in text and "| local at $0 |" in text
    assert "**assumed, not measured**" in text
    data = json.loads(out.read_text())
    assert len(data["paid_writers"]["worlds"]) == 2
    assert data["local_writers"]["total_usd"] < data["paid_writers"]["total_usd"]


def test_main_refuses_without_logs(tmp_path, capsys) -> None:
    assert E.main(["--var-root", str(tmp_path / "empty"), "--worlds", "pilot"]) == 2
    assert "no writer or teacher call logs" in capsys.readouterr().err
