"""WP9: three-judge dispute resolution with scripted judges (no network),
on the fixture world and on the Halvard exam (report only)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import portolan.bench.__main__ as main_module
import portolan.bench.world_cli as world_cli
from portolan.bench.__main__ import main
from portolan.bench.disputes import (
    BudgetEstimateExceeded,
    Dispute,
    JudgePanel,
    JudgeReply,
    resolve_disputes,
    summarize,
    write_dispute_logs,
)
from portolan.bench.world_cli import _world_disputes
from portolan.bench.world_score import score_world
from portolan.bench.world_v2 import load_world
from portolan.extract.backends import FakeBackend
from portolan.training.spend import SpendMeter
from tests.bench_world_fixture import (
    CALL_ID,
    JUDGES,
    PRAISE,
    PRICES,
    ScriptedJudges,
    build_world,
    scripted_results,
    verdict,
    world_script,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
TRUTH_TABLE = REPO_ROOT / "corpus" / "truth-table.yaml"
MANIFEST = REPO_ROOT / "corpus" / "out" / "manifest.json"
CORPUS = REPO_ROOT / "corpus" / "out"


def _world_disputes_and_texts(tmp_path: Path) -> tuple[list[Dispute], dict[str, str]]:
    world = load_world(build_world(tmp_path / "w"))
    score = score_world(world, scripted_results(), backend="stub")
    return _world_disputes(world, score), {sid: s.text for sid, s in world.sources.items()}


def test_disputes_are_the_decoy_hit_and_off_span_facts(tmp_path: Path) -> None:
    disputes, _ = _world_disputes_and_texts(tmp_path)
    assert [(d.produced_id, d.category) for d in disputes] == [
        ("f3", "decoy_hit"),
        ("f4", "off_span"),
        ("f5", "off_span"),
    ]
    assert disputes[0].decoy_class_id == "hypothetical"


def test_majority_early_exit_and_correction_log(tmp_path: Path) -> None:
    disputes, texts = _world_disputes_and_texts(tmp_path)
    judges = ScriptedJudges(world_script())
    meter = SpendMeter(tmp_path / "spend.json")
    run = resolve_disputes(
        disputes, texts, JudgePanel(JUDGES, PRICES, judges), max_usd=0.5, meter=meter
    )
    decisions = {o.dispute.produced_id: (o.decision, o.tally) for o in run.outcomes}
    assert decisions == {
        "f3": ("not_supported", "0-2"),
        "f4": ("supported", "2-1"),
        "f5": ("not_supported", "0-2"),
    }
    # Two non-supported votes settle it: judge c only ran on the praise line.
    assert [m for m, _ in judges.calls].count("judge/c") == 1
    assert run.calls == 7 == len(judges.calls)
    assert meter.corpus_usd("bench_judges") == pytest.approx(0.007)
    assert summarize(run) == {
        "decoy_hit": {"disputed": 1, "judged": 1, "overturned": 0, "upheld": 1, "not_judged": 0},
        "off_span": {"disputed": 2, "judged": 2, "overturned": 1, "upheld": 1, "not_judged": 0},
    }

    corrections, everything = write_dispute_logs(run, tmp_path / "out", context={"world_id": "w"})
    rows = [json.loads(line) for line in corrections.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert row["fact"]["quote"] == "the new dashboard saves my nurses an hour a day"
    assert row["tally"] == "2-1"
    assert row["judge_models"] == list(JUDGES)
    assert [v["effective"] for v in row["votes"]] == ["supported", "unsupported", "supported"]
    assert row["status"] == "proposed" and row["applied"] is False
    assert "judge/a" in row["rationale"] and "judge/b" not in row["rationale"]
    assert len(everything.read_text(encoding="utf-8").splitlines()) == 3


def test_supported_vote_without_resolvable_evidence_counts_as_partial(tmp_path: Path) -> None:
    disputes, texts = _world_disputes_and_texts(tmp_path)
    praise = [d for d in disputes if d.produced_id == "f4"]
    script = {
        ("judge/a", "Claim: The new dashboard"): verdict(
            "supported", "words that are not in the call"
        ),
        ("judge/b", "Claim: The new dashboard"): verdict("supported", ""),
        ("judge/c", "Claim: The new dashboard"): verdict("supported", PRAISE),
    }
    run = resolve_disputes(
        praise, texts, JudgePanel(JUDGES, PRICES, ScriptedJudges(script)), max_usd=0.5
    )
    outcome = run.outcomes[0]
    assert [v.verdict for v in outcome.votes] == ["supported", "supported"]
    assert [v.effective for v in outcome.votes] == ["partial", "partial"]
    assert outcome.decision == "not_supported"


def test_a_failed_judge_call_is_a_non_supported_vote(tmp_path: Path) -> None:
    disputes, texts = _world_disputes_and_texts(tmp_path)
    praise = [d for d in disputes if d.produced_id == "f4"]

    class Flaky(ScriptedJudges):
        def judge(self, *, model: str, system: str, user: str, max_tokens: int) -> JudgeReply:
            if model == "judge/a":
                self.calls.append((model, user))
                return JudgeReply(data=None, cost_usd=0.001, error="HTTP 502")
            return super().judge(model=model, system=system, user=user, max_tokens=max_tokens)

    script = {("judge/b", "dashboard"): verdict("supported", PRAISE)}
    run = resolve_disputes(praise, texts, JudgePanel(JUDGES, PRICES, Flaky(script)), max_usd=0.5)
    votes = run.outcomes[0].votes
    assert [(v.judge_model, v.effective) for v in votes] == [
        ("judge/a", "error"),
        ("judge/b", "supported"),
        ("judge/c", "unsupported"),
    ]
    assert votes[0].error == "HTTP 502"
    assert run.outcomes[0].decision == "not_supported"


def test_estimate_over_the_cap_calls_no_judge(tmp_path: Path) -> None:
    disputes, texts = _world_disputes_and_texts(tmp_path)
    judges = ScriptedJudges(world_script())
    with pytest.raises(BudgetEstimateExceeded) as exc:
        resolve_disputes(disputes, texts, JudgePanel(JUDGES, PRICES, judges), max_usd=1e-6)
    assert judges.calls == []
    assert exc.value.estimate_usd > 1e-6


def test_runtime_cap_stops_before_a_call_that_could_cross_it(tmp_path: Path) -> None:
    disputes, texts = _world_disputes_and_texts(tmp_path)
    judges = ScriptedJudges({}, cost=0.3)  # bills far above the estimate
    meter = SpendMeter(tmp_path / "spend.json")
    run = resolve_disputes(
        disputes, texts, JudgePanel(JUDGES, PRICES, judges), max_usd=0.5, meter=meter
    )
    assert run.calls == 2
    assert run.stopped is not None and run.stopped.startswith("budget")
    assert [o.decision for o in run.outcomes] == ["not_supported", "not_judged", "not_judged"]
    assert meter.corpus_usd("bench_judges") == pytest.approx(0.6)


def test_no_call_starts_when_its_ceiling_could_cross_the_cap(tmp_path: Path) -> None:
    """Expected cost fits the cap, but one call's ceiling (full max_tokens,
    three billed requests) does not: no judge is called at all."""
    disputes, texts = _world_disputes_and_texts(tmp_path)
    judges = ScriptedJudges(world_script())
    prices = dict.fromkeys(JUDGES, (1e-9, 1e-4))  # expected ~$0.05 a call, ceiling ~$0.60
    run = resolve_disputes(disputes, texts, JudgePanel(JUDGES, prices, judges), max_usd=0.5)
    assert judges.calls == []
    assert run.estimate_usd <= 0.5 < run.ceiling_usd
    assert [o.decision for o in run.outcomes] == ["not_judged"] * 3


def test_panel_needs_three_different_priced_models() -> None:
    judges = ScriptedJudges({})
    with pytest.raises(ValueError):
        JudgePanel(("a", "a", "b"), {"a": (0, 0), "b": (0, 0)}, judges)
    with pytest.raises(ValueError):
        JudgePanel(("a", "b", "c"), {"a": (0, 0), "b": (0, 0)}, judges)


# --------------------------------------------------------------------------
# CLI, world
# --------------------------------------------------------------------------


def _patch(monkeypatch, canned, script) -> ScriptedJudges:
    judges = ScriptedJudges(script)
    monkeypatch.setattr(main_module, "_live_backend", lambda name, model: FakeBackend(canned))
    monkeypatch.setattr(
        world_cli, "make_judge_panel", lambda models: JudgePanel(models, PRICES, judges)
    )
    return judges


def test_extract_world_with_judges_end_to_end(monkeypatch, tmp_path: Path) -> None:
    world_dir = build_world(tmp_path / "w")
    judges = _patch(monkeypatch, scripted_results(), world_script())
    out = tmp_path / "out"
    spend = tmp_path / "spend.json"
    code = main(
        [
            "extract",
            "--world",
            str(world_dir),
            "--backend",
            "openrouter",
            "--model",
            "amazon/nova-lite-v1",
            "--dispute-judges",
            ",".join(JUDGES),
            "--max-usd",
            "0.25",
            "--spend-path",
            str(spend),
            "--out",
            str(out),
        ]
    )
    assert code == 0
    assert len(judges.calls) == 7
    report = json.loads(next(out.glob("world-bench-*-openrouter.json")).read_text("utf-8"))
    assert report["overall"]["judge_supported"] == 1
    assert report["overall_rates"]["substantiated"] == 5 / 8
    assert report["disputes"]["by_category"]["off_span"]["overturned"] == 1
    assert report["disputes"]["judge_models"] == list(JUDGES)
    rows = (out / "corrections.jsonl").read_text("utf-8").splitlines()
    assert len(rows) == 1 and json.loads(rows[0])["source_id"] == CALL_ID
    meter = SpendMeter(spend)
    assert meter.corpus_usd("bench_judges") == pytest.approx(0.007)
    assert meter.corpus_usd("bench_extract") > 0
    md = next(out.glob("world-bench-*-openrouter.md")).read_text("utf-8")
    assert "## Three-judge disputes" in md and "| off_span | 2 | 2 | 1 | 1 | 0 |" in md


def test_extract_world_with_nothing_disputed_calls_no_judge(monkeypatch, tmp_path: Path) -> None:
    from portolan.bench.world_score import perfect_world_results

    world_dir = build_world(tmp_path / "w")
    judges = _patch(monkeypatch, perfect_world_results(load_world(world_dir)), {})
    monkeypatch.setattr(world_cli, "make_judge_panel", lambda models: pytest.fail("no panel"))
    code = main(
        [
            "extract",
            "--world",
            str(world_dir),
            "--backend",
            "openrouter",
            "--dispute-judges",
            "default",
            "--spend-path",
            str(tmp_path / "spend.json"),
            "--out",
            str(tmp_path / "out"),
        ]
    )
    assert code == 0
    assert judges.calls == []
    assert (tmp_path / "out" / "corrections.jsonl").read_text("utf-8") == ""


def test_extract_world_refuses_judges_over_budget(monkeypatch, tmp_path: Path, capsys) -> None:
    world_dir = build_world(tmp_path / "w")
    judges = _patch(monkeypatch, scripted_results(), world_script())
    code = main(
        [
            "extract",
            "--world",
            str(world_dir),
            "--backend",
            "openrouter",
            "--dispute-judges",
            ",".join(JUDGES),
            "--max-usd",
            "0.000001",
            "--spend-path",
            str(tmp_path / "spend.json"),
            "--out",
            str(tmp_path / "out"),
        ]
    )
    assert code == world_cli.EXIT_JUDGES_OVER_BUDGET
    assert judges.calls == []
    assert not (tmp_path / "out" / "corrections.jsonl").exists()
    assert "judge estimate: expected $" in capsys.readouterr().out


# --------------------------------------------------------------------------
# Halvard exam, report only
# --------------------------------------------------------------------------

HALVARD_SOURCES = ["S-ticket-019", "S-ticket-024"]


def _halvard_canned() -> tuple[dict, str]:
    """Perfect results for two tickets plus one extra fact each: one quoting
    a real ticket line (judges will call it real), one invented."""
    from portolan.bench.corpus_loader import load_prepared_source
    from portolan.bench.fixtures import perfect_results
    from portolan.bench.truth import Manifest, Truth
    from portolan.extract.schema import Fact

    manifest = Manifest.load(MANIFEST)
    truth = Truth.for_manifest(manifest, truth_table_path=TRUTH_TABLE)
    canned = {sid: perfect_results(truth)[sid] for sid in HALVARD_SOURCES}
    text = load_prepared_source(manifest.source(HALVARD_SOURCES[0]), corpus_root=CORPUS).text
    real_line = text.splitlines()[1].split(": ", 1)[1][:120]
    canned[HALVARD_SOURCES[0]].facts.append(
        Fact(
            id="x-real",
            type="other",
            claim="Extra real statement zzmarker.",
            locator="S-ticket-019#1",
            quote=real_line,
            sentiment=0,
            confidence=0.7,
        )
    )
    canned[HALVARD_SOURCES[1]].facts.append(
        Fact(
            id="x-invented",
            type="praise",
            claim="The customer loves the new portal yymarker.",
            locator="S-ticket-024#1",
            quote="We love the new portal",
            sentiment=1,
            confidence=0.7,
        )
    )
    return canned, real_line


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_extract_halvard_with_judges_is_report_only(monkeypatch, tmp_path: Path) -> None:
    canned, real_line = _halvard_canned()
    script = {m: verdict("supported", real_line) for m in JUDGES}
    judges = _patch(monkeypatch, canned, {(m, "zzmarker"): v for m, v in script.items()})
    before = _sha(TRUTH_TABLE)
    out = tmp_path / "out"
    code = main(
        [
            "extract",
            "--backend",
            "openrouter",
            "--model",
            "amazon/nova-lite-v1",
            "--corpus",
            str(CORPUS),
            "--manifest",
            str(MANIFEST),
            "--truth-table",
            str(TRUTH_TABLE),
            "--source-ids",
            *HALVARD_SOURCES,
            "--dispute-judges",
            ",".join(JUDGES),
            "--spend-path",
            str(tmp_path / "spend.json"),
            "--out",
            str(out),
        ]
    )
    assert code == 0
    assert _sha(TRUTH_TABLE) == before
    rows = [json.loads(x) for x in (out / "corrections.jsonl").read_text("utf-8").splitlines()]
    assert [r["produced_id"] for r in rows] == ["f" + str(len(canned[HALVARD_SOURCES[0]].facts))]
    assert rows[0]["exam"] == "halvard" and rows[0]["tally"] == "3-0"
    everything = (out / "disputes.jsonl").read_text("utf-8").splitlines()
    assert len(everything) == 2
    assert len(judges.calls) == 5  # 3 on the real line, 2 on the invented one
    summary = json.loads(next(out.glob("disputes-*.json")).read_text("utf-8"))["summary"]
    assert sum(r["overturned"] for r in summary["by_category"].values()) == 1
    assert SpendMeter(tmp_path / "spend.json").corpus_usd("bench_judges") == pytest.approx(0.005)


def test_disputes_subcommand_on_a_saved_halvard_report(monkeypatch, tmp_path: Path) -> None:
    from portolan.bench.extraction import run_extraction, score_extraction
    from portolan.bench.report import BenchReport, write
    from portolan.bench.truth import Manifest, Truth

    canned, _ = _halvard_canned()
    manifest = Manifest.load(MANIFEST)
    truth = Truth.for_manifest(manifest, truth_table_path=TRUTH_TABLE)
    run = run_extraction(
        manifest, FakeBackend(canned), source_ids=HALVARD_SOURCES, corpus_root=CORPUS, truth=truth
    )
    score = score_extraction(truth, run)
    import datetime as dt

    _, saved = write(BenchReport(backend="fake", date=dt.date.today(), extraction=score), tmp_path)
    judges = _patch(monkeypatch, canned, {})
    before = _sha(TRUTH_TABLE)
    code = main(
        [
            "disputes",
            "--report",
            str(saved),
            "--corpus",
            str(CORPUS),
            "--manifest",
            str(MANIFEST),
            "--dispute-judges",
            ",".join(JUDGES),
            "--spend-path",
            str(tmp_path / "spend.json"),
            "--out",
            str(tmp_path / "d"),
        ]
    )
    assert code == 0
    assert _sha(TRUTH_TABLE) == before
    assert len(judges.calls) == 2 * len(score.unplanted)
    assert (tmp_path / "d" / "corrections.jsonl").read_text("utf-8") == ""
    assert len((tmp_path / "d" / "disputes.jsonl").read_text("utf-8").splitlines()) == len(
        score.unplanted
    )
