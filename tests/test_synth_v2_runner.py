"""Tests for `scripts/data/synth_v2.py` (the v2 world batch runner, WP11) and
`corpus/generator/v2/rounds.py`.

Every writer call is stubbed with the WP4 fixture writer
(`tests/v2_writer_fakes.py`), every teacher and judge call with the pipeline
fixtures (`tests/v2_pipeline_fakes.py`); every output, spend meter and
contamination reference set lives under `tmp_path`. Worlds are cut to a few
documents with `max_docs` so each test runs in about a second.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from corpus.generator.v2 import writer as w
from corpus.generator.v2.models import Split
from corpus.generator.v2.names import NameGuard
from corpus.generator.v2.rounds import PILOT, ROUND_1, WorldSpec, parse_worlds
from portolan.training import contamination as C
from portolan.training.spend import SpendMeter
from tests._paths import require_repo_path
from tests.v2_pipeline_fakes import LabelEchoTeacher, dry_run_reply, keep_all_judge
from tests.v2_writer_fakes import ScriptedBackend

_SCRIPT = require_repo_path("scripts", "data", "synth_v2.py")


def _load():
    spec = importlib.util.spec_from_file_location("synth_v2_script", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["synth_v2_script"] = module
    spec.loader.exec_module(module)
    return module


S = _load()


class _Teacher(S.DryRunTeacher):
    """The dry run's teacher, with extra rows per envelope and a record of
    every teacher call (model, source id)."""

    def __init__(self, extra=None) -> None:
        super().__init__()
        self.extra = extra
        self.calls: list[tuple[str, str]] = []

    def bind(self, written):
        teacher = LabelEchoTeacher(dict(written), extra=self.extra, calls=self.calls)
        return teacher.factory


def _ctx(
    tmp_path: Path,
    *,
    backend=None,
    cap: float = 1.0,
    refs=None,
    max_docs: int | None = 4,
    writers=None,
    teacher=None,
    judge=keep_all_judge,
    guard: NameGuard | None = None,
):
    meter = SpendMeter(tmp_path / "spend.json")
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    budget = S.open_budget(
        out, meter, cap_usd=cap, line_cap_usd=None, reset_baseline=False, now=S._utc_now
    )
    lines: list[str] = []
    ctx = S.RunContext(
        out_root=out,
        writers=tuple(writers or w.DEFAULT_ROTATION),
        backend=backend or ScriptedBackend(dry_run_reply, cost=0.0),
        budget=budget,
        refs=refs,
        refs_dir=None,
        teacher_factory=teacher or _Teacher(),
        judge_call=judge,
        name_guard=guard,
        max_docs=max_docs,
        progress=lines.append,
    )
    return ctx, lines


def _report(ctx, spec: WorldSpec) -> dict:
    return json.loads((S.world_dir(ctx.out_root, spec) / "run_report.json").read_text())


# --------------------------------------------------------------------------
# rounds
# --------------------------------------------------------------------------


def test_round1_is_12_train_2_validation_3_test_with_the_held_out_vertical_only_in_test() -> None:
    splits = [w_.split for w_ in ROUND_1]
    assert (
        splits.count(Split.TRAIN),
        splits.count(Split.VALIDATION),
        splits.count(Split.TEST),
    ) == (12, 2, 3)
    assert len({w_.world_id for w_ in ROUND_1}) == 17
    held = [w_ for w_ in ROUND_1 if w_.pack == "lab_and_imaging_services"]
    assert [w_.split for w_ in held] == [Split.TEST]
    assert all(w_.strictness == ("test" if w_.split == Split.TEST else "train") for w_ in ROUND_1)
    assert PILOT.strictness == "test" and (PILOT.pack, PILOT.split, PILOT.seed) == (
        ROUND_1[0].pack,
        ROUND_1[0].split,
        ROUND_1[0].seed,
    )


def test_parse_worlds_forms_and_errors() -> None:
    assert [x.seed for x in parse_worlds("3", pack="telehealth", split="train", seed=40)] == [
        40,
        41,
        42,
    ]
    one = parse_worlds("", pack="telehealth", split="test", seed=9)
    assert one == [WorldSpec("telehealth", Split.TEST, 9, strictness="test")]
    both = parse_worlds(
        "telehealth:train:1,ehr_addons:validation:2", pack="x", split="train", seed=0
    )
    assert [x.label() for x in both] == ["telehealth:train:1", "ehr_addons:validation:2"]
    assert (
        parse_worlds("round1", pack="x", split="train", seed=0, strictness="test")[0].strictness
        == "test"
    )
    with pytest.raises(ValueError, match="unknown pack"):
        parse_worlds("nope:train:1", pack="x", split="train", seed=0)
    with pytest.raises(ValueError, match="repeat"):
        parse_worlds("telehealth:train:1,telehealth:test:1", pack="x", split="train", seed=0)
    with pytest.raises(ValueError, match="pack:split:seed"):
        parse_worlds("telehealth:train", pack="x", split="train", seed=0)


# --------------------------------------------------------------------------
# preflight
# --------------------------------------------------------------------------


def test_local_writer_ids_are_refused_without_the_local_backend(monkeypatch) -> None:
    monkeypatch.delattr(w, "LocalWriterBackend", raising=False)
    with pytest.raises(S.RunnerError, match="feature/local-teachers"):
        S.check_writers(("local:mlx-community/Qwen3.5-4B-MLX-4bit",))
    with pytest.raises(S.RunnerError, match="refused"):
        S.check_writers(("nvidia/nemotron-3.5-lightning",))
    with pytest.raises(S.RunnerError, match="not in ALLOWED_WRITER_MODELS"):
        S.check_writers(("some/unknown-model",))


def test_main_refuses_before_any_spend(tmp_path, capsys) -> None:
    out = ["--out", str(tmp_path / "o"), "--no-contamination-check"]
    three = ",".join(w.DEFAULT_ROTATION[:3])
    assert S.main(["--check", "--writers", three, *out]) == 2
    assert "cannot cover" in capsys.readouterr().err
    assert S.main(["--check", "--worlds", "lab_and_imaging_services:train:1", *out]) == 2
    assert "held-out" in capsys.readouterr().err
    assert S.main(["--worlds", "pilot", *out]) == 2  # a real run needs --budget-usd
    assert "--budget-usd is required" in capsys.readouterr().err
    assert S.main(["--check", "--worlds", "pilot", *out]) == 0
    assert not (tmp_path / "o").exists()


def test_dry_run_and_real_roots_never_mix(tmp_path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / S.DRY_RUN_MARKER).write_text("x")
    with pytest.raises(S.RunnerError, match="dry-run root"):
        S.check_output_root(root, dry_run=False)
    real = tmp_path / "real"
    real.mkdir()
    (real / "budget.json").write_text("{}")
    with pytest.raises(S.RunnerError, match="holds a real run"):
        S.check_output_root(real, dry_run=True)


def test_holdout_output_root_is_refused(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="hold-out"):
        S.check_output_root(tmp_path / "corpus" / "out" / "x", dry_run=True)


# --------------------------------------------------------------------------
# the pipeline
# --------------------------------------------------------------------------


def test_dry_run_cli_runs_every_stage_and_no_stub(tmp_path, capsys) -> None:
    out = tmp_path / "dry"
    code = S.main(
        [
            "--dry-run",
            "--worlds",
            "pilot",
            "--max-docs",
            "5",
            "--no-contamination-check",
            "--no-name-guard",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    assert "stub" not in capsys.readouterr().err.lower()
    wdir = out / "train" / "world-pharmacy_services-1"
    for rel in (
        "world.yaml",
        "truth-table.yaml",
        "world_plan.json",
        "plan/crm/accounts.csv",
        "plan/code_labels.jsonl",
        "completeness/qa_report.json",
        "noise/noise_manifest.json",
        "manifest.json",
    ):
        assert (wdir / rel).exists(), rel
    assert (out / S.DRY_RUN_MARKER).exists() and (out / "dry_run_spend.json").exists()
    report = json.loads((wdir / "run_report.json").read_text())
    assert report["dry_run"] is True and report["complete"] is False  # --max-docs
    assert report["stubbed_stages"] == []
    assert report["strictness"] == "test"
    statuses = {s["name"]: s["status"] for s in report["stages"]}
    assert statuses == {
        "simulate": "done",
        "plan": "done",
        "render": "done",
        "write": "partial",
        "completeness": "done",
        "noise": "done",
        "export": "done",
        "contamination": "skipped",
        "report": "done",
    }
    assert not any("STUB" in x for x in report["warnings"])
    assert any(x.startswith("test strictness:") for x in report["warnings"])
    labels = (wdir / "writer" / "labels.jsonl").read_text().splitlines()
    assert report["labels"]["scripted_writer"] == len(labels) > 0
    assert report["labels"]["silver_completeness"] == 0
    assert report["gates"]["section_3_5"]["teacher_recall_on_scripted"] == "PASS"
    assert report["label_judge"] == S.DEFAULT_LABEL_JUDGE_MODEL
    completeness = next(s for s in report["stages"] if s["name"] == "completeness")
    assert completeness["detail"]["teacher_recall_r"] == 1.0
    export = next(s for s in report["stages"] if s["name"] == "export")
    assert export["detail"]["noise_not_applied"] == 0
    assert export["detail"]["training_records"] == export["detail"]["sft_ready"] > 0
    assert sum(len(v) for v in report["review_sample"].values()) == 5


def test_resume_makes_no_call_and_reports_resumed_stages(tmp_path) -> None:
    backend = ScriptedBackend(dry_run_reply, cost=0.0)
    teacher = _Teacher()
    ctx, _ = _ctx(tmp_path, backend=backend, teacher=teacher)
    S.run_worlds([PILOT], ctx)
    first_calls, first_teacher = len(backend.calls), len(teacher.calls)
    assert first_calls >= 4 and first_teacher >= 4
    ctx2, _ = _ctx(tmp_path, backend=backend, teacher=teacher)
    S.run_worlds([PILOT], ctx2)
    assert len(backend.calls) == first_calls
    assert len(teacher.calls) == first_teacher
    report = _report(ctx2, PILOT)
    statuses = {s["name"]: s["status"] for s in report["stages"]}
    assert statuses["simulate"] == statuses["plan"] == "resumed"
    write = next(s for s in report["stages"] if s["name"] == "write")
    assert (
        write["detail"]["documents"]["resumed"] == 4
        and write["detail"]["documents"]["this_run"] == 0
    )


def test_budget_counts_other_spend_on_the_shared_line_and_stops_before_the_call(tmp_path) -> None:
    meter = SpendMeter(tmp_path / "spend.json")
    meter.record(model="m", corpus="synthetic_v2", amount_usd=5.0)  # earlier work on the line
    out = tmp_path / "out"
    budget = S.open_budget(
        out, meter, cap_usd=0.05, line_cap_usd=None, reset_baseline=False, now=S._utc_now
    )
    assert budget.baseline_usd == 5.0 and budget.run_spent_usd() == 0.0
    model = w.DEFAULT_ROTATION[0]
    budget.precheck(model, 10_000, 1_000)  # fits
    meter.record(model="other-session", corpus="synthetic_v2", amount_usd=0.049)
    with pytest.raises(w.WriterBudgetExceeded, match=r"0\.05 cap"):
        budget.precheck(model, 10_000, 1_000)
    again = S.open_budget(
        out, meter, cap_usd=0.05, line_cap_usd=None, reset_baseline=False, now=S._utc_now
    )
    assert again.baseline_usd == 5.0  # the ledger keeps the first baseline
    fresh = S.open_budget(
        out, meter, cap_usd=0.05, line_cap_usd=None, reset_baseline=True, now=S._utc_now
    )
    assert fresh.baseline_usd == pytest.approx(5.049)
    line = S.SharedMeterBudget(cap_usd=100.0, meter=meter, corpus_cap_usd=5.05, baseline_usd=5.049)
    with pytest.raises(w.WriterBudgetExceeded, match="line cap"):
        line.precheck(model, 10_000, 1_000)
    # The label judge's line counts against the same cap.
    judged = S.open_budget(
        out, meter, cap_usd=0.05, line_cap_usd=None, reset_baseline=True, now=S._utc_now
    )
    assert judged.can_spend(0.01)
    meter.record(model="judge", corpus="label_judge", amount_usd=0.045)
    assert judged.run_spent_usd() == pytest.approx(0.045) and not judged.can_spend(0.01)
    with pytest.raises(w.WriterBudgetExceeded, match=r"0\.05 cap"):
        judged.precheck_usd(0.01)
    with pytest.raises(S.RunnerError, match="meter"):
        S.open_budget(
            out,
            SpendMeter(tmp_path / "other.json"),
            cap_usd=1,
            line_cap_usd=None,
            reset_baseline=False,
            now=S._utc_now,
        )


def test_a_budget_stop_halts_the_world_and_leaves_later_worlds_unstarted(tmp_path) -> None:
    backend = ScriptedBackend(dry_run_reply, cost=0.0)
    ctx, _ = _ctx(tmp_path, backend=backend, cap=0.0)
    second = WorldSpec("telehealth", Split.TRAIN, 12)
    round_report = S.run_worlds([PILOT, second], ctx)
    assert backend.calls == []  # stopped before the first call
    assert round_report["stop_reason"] and "cap" in round_report["stop_reason"]
    assert [x["status"] for x in round_report["worlds"]] == ["stopped", "not started"]
    report = _report(ctx, PILOT)
    statuses = {s["name"]: s["status"] for s in report["stages"]}
    assert statuses["write"] == "stopped"
    for later in ("completeness", "noise", "export", "contamination"):
        assert statuses[later] == "skipped", later
    assert report["complete"] is False
    assert not S.world_dir(ctx.out_root, second).exists()


def test_paid_calls_land_on_the_shared_meter_under_synthetic_v2(tmp_path) -> None:
    backend = ScriptedBackend(dry_run_reply, cost=0.001)
    ctx, _ = _ctx(tmp_path, backend=backend, max_docs=3)
    round_report = S.run_worlds([PILOT], ctx)
    spent = SpendMeter(tmp_path / "spend.json").corpus_usd("synthetic_v2")
    assert spent == pytest.approx(0.001 * len(backend.calls))
    assert round_report["budget"]["spent_by_root_usd"] == pytest.approx(spent)
    report = _report(ctx, PILOT)
    assert report["cost_usd"]["total_known"] == pytest.approx(spent)
    labeled = sum(
        n for s, n in report["labels"]["exported_by_label_source"].items() if s != "scripted_code"
    )
    assert report["cost_usd"]["per_labeled_fact"] == pytest.approx(spent / labeled, abs=1e-6)


def _refs(names=(), canary: str | None = None) -> C.ContaminationReferences:
    params = C.ContaminationParams(**({"canary_guid": canary} if canary else {}))
    docs = [
        (
            "ref-1",
            "erb_test",
            "an unrelated reference document about harbor logistics and tide tables",
        )
    ]
    matrix, meta, ngrams = C.build_reference_matrix(docs, params)
    blocklist = C.build_name_blocklist([("erb_test", n) for n in names], params)
    return C.ContaminationReferences(matrix, meta, ngrams, blocklist, params)


def test_contamination_stage_reports_excluded_sources(tmp_path) -> None:
    from corpus.generator.v2.packs.schema import load_pack
    from corpus.generator.v2.simulator import simulate_world

    world = simulate_world(
        load_pack(S.PACKS_DIR / "pharmacy_services.yaml"), seed=1, split=Split.TRAIN
    )
    ctx, _ = _ctx(tmp_path, refs=_refs(names=[world.vendor.name]))
    S.run_worlds([PILOT], ctx)
    report = _report(ctx, PILOT)
    stage = next(s for s in report["stages"] if s["name"] == "contamination")
    assert stage["status"] == "done"
    detail = stage["detail"]
    assert detail["excluded"] == len(detail["excluded_source_ids"]) > 0
    assert detail["hits_by_reason"]["blocklisted_name"] >= detail["excluded"]
    wdir = S.world_dir(ctx.out_root, PILOT)
    exported = (wdir / "sources.jsonl").read_text().splitlines()
    assert detail["sources_checked"] == len(exported)
    crm = len(json.loads((wdir / "world_plan.json").read_text())["crm_accounts"])
    assert sum(1 for x in exported if json.loads(x)["kind"] == "crm_account") == crm
    rows = (wdir / "contamination" / "contamination.jsonl").read_text().splitlines()
    assert len(rows) == detail["excluded"]
    assert world.vendor.name not in "".join(rows)  # hashes only, never the matched text


def test_a_canary_fails_the_world(tmp_path) -> None:
    ctx, _ = _ctx(tmp_path, refs=_refs(canary="fixture opening words"))
    round_report = S.run_worlds([PILOT], ctx)
    assert round_report["worlds"][0]["status"] == "failed"
    stage = next(s for s in _report(ctx, PILOT)["stages"] if s["name"] == "contamination")
    assert stage["status"] == "failed" and "canary" in stage["message"]


def test_local_writers_run_at_zero_dollars_under_a_zero_cap(tmp_path, monkeypatch) -> None:
    local = tuple(f"local:test/model-{i}" for i in range(4))
    for model in local:
        monkeypatch.setitem(
            w.ALLOWED_WRITER_MODELS, model, w.WriterModelInfo("local", 0.0, 0.0, False)
        )
    monkeypatch.setattr(w, "LocalWriterBackend", object, raising=False)
    S.check_writers(local)
    backend = ScriptedBackend(dry_run_reply, cost=0.0)
    ctx, _ = _ctx(tmp_path, backend=backend, cap=0.0, writers=local, max_docs=4)
    round_report = S.run_worlds([PILOT], ctx)
    # The writers cost nothing; the paid teacher is refused before its first call.
    statuses = {s["name"]: s["status"] for s in _report(ctx, PILOT)["stages"]}
    assert statuses["write"] == "partial" and statuses["completeness"] == "stopped"
    assert "cap" in (round_report["stop_reason"] or "")
    assert {c.model for c in backend.calls} <= set(local)
    labels = [
        json.loads(x)
        for x in (S.world_dir(ctx.out_root, PILOT) / "writer" / "labels.jsonl")
        .read_text()
        .splitlines()
    ]
    assert labels and {x["writer_model"] for x in labels} <= set(local)


def test_the_label_judge_reads_silver_labels_and_its_log(tmp_path) -> None:
    """A teacher fact nobody scripted becomes a silver label; the judge is
    called on it and its verdicts reach the run report."""

    def extra(envelope):
        return [
            {"type": "request", "claim": "fixture silver claim", "quote": "fixture reply words"}
        ][: int("fixture reply words" in envelope.text)]

    calls: list[str] = []

    def judge(model, system, user):
        calls.append(model)
        return keep_all_judge(model, system, user)

    ctx, _ = _ctx(tmp_path, teacher=_Teacher(extra=extra), judge=judge)
    S.run_worlds([PILOT], ctx)
    report = _report(ctx, PILOT)
    completeness = next(s for s in report["stages"] if s["name"] == "completeness")
    assert completeness["detail"]["silver"]["labels"] > 0
    assert calls and set(calls) == {S.DEFAULT_LABEL_JUDGE_MODEL}
    assert completeness["detail"]["label_judge"]["verdicts"] == {
        "keep": completeness["detail"]["silver"]["labels"]
    }
    wdir = S.world_dir(ctx.out_root, PILOT)
    assert (wdir / "completeness" / "label_judge.jsonl").read_text().strip()
    assert report["labels"]["silver_completeness"] == completeness["detail"]["silver"]["labels"]
    assert report["cost_usd"]["label_judge_calls"] == len(calls)


def test_noise_runs_on_wp5_output_and_ships_silver_labels(tmp_path) -> None:
    def extra(envelope):
        return [
            {"type": "request", "claim": "fixture silver claim", "quote": "fixture reply words"}
        ][: int("fixture reply words" in envelope.text)]

    ctx, _ = _ctx(tmp_path, teacher=_Teacher(extra=extra), max_docs=12)
    S.run_worlds([PILOT], ctx)
    wdir = S.world_dir(ctx.out_root, PILOT)
    manifest = json.loads((wdir / "manifest.json").read_text())
    assert manifest["upstream"]["noise"]["ran"] and manifest["upstream"]["completeness"]["ran"]
    assert manifest["documents"]["noise_not_applied"] == {}
    assert manifest["documents"]["noised"] == manifest["documents"]["exported"]
    labels = [json.loads(x) for x in (wdir / "labels.jsonl").read_text().splitlines()]
    assert any(x["label_source"] == "silver_completeness" for x in labels)
    plans = {p["id"]: p for p in json.loads((wdir / "world_plan.json").read_text())["doc_plans"]}
    docs = [json.loads(p.read_text()) for p in (wdir / "noise" / "docs").glob("*.json")]
    shipped = [d for d in docs if plans[d["plan_id"]]["role"] != "near_duplicate"]
    assert shipped and all(d["noised_from_sha256"] for d in shipped)


def test_the_name_guard_reaches_the_simulated_world(tmp_path) -> None:
    from corpus.generator.v2.packs.schema import load_pack
    from corpus.generator.v2.simulator import simulate_world

    pack = load_pack(S.PACKS_DIR / "pharmacy_services.yaml")
    plain = simulate_world(pack, seed=1, split=Split.TRAIN)
    words = C.normalize_name_words(plain.vendor.name.split()[0])
    guard = NameGuard(
        {"erb_test": {len(words): {C.word_sequence_hash(words)}}},
        max_name_words=4,
        fingerprint="f" * 64,
    )
    ctx, _ = _ctx(tmp_path, guard=guard, max_docs=2)
    S.run_worlds([PILOT], ctx)
    report = _report(ctx, PILOT)
    simulate = next(s for s in report["stages"] if s["name"] == "simulate")
    assert simulate["detail"]["name_guard"] == "f" * 64
    assert simulate["detail"]["vendor"] != plain.vendor.name
    assert not guard.blocked(simulate["detail"]["vendor"])


def test_writer_share_gate_is_reported(tmp_path) -> None:
    ctx, _ = _ctx(tmp_path, max_docs=2, writers=w.DEFAULT_ROTATION[:4])
    S.run_worlds([PILOT], ctx)
    gate = _report(ctx, PILOT)["gates"]["writer_max_share_of_written"]
    # two documents, four writers: each wrote one of two, 50% of what shipped
    assert gate == {"value": 0.5, "max": 0.25, "pass": False}
    assert any("writer share gate" in x for x in _report(ctx, PILOT)["warnings"])
