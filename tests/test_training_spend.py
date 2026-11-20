"""Tests for `portolan.training.spend` -- the dollar spend meter
(coordinator course-correction, 2026-09-21, item 1: "accumulate it in
`var/data/silver/spend.json` (per model, per corpus, total)")."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from portolan.training.spend import SpendMeter, default_spend_path, shared_var_root


def test_default_spend_path(tmp_path: Path) -> None:
    assert default_spend_path(tmp_path) == tmp_path / "data" / "silver" / "spend.json"


def test_starts_at_zero(tmp_path: Path) -> None:
    meter = SpendMeter(path=default_spend_path(tmp_path))
    assert meter.total_usd() == 0.0
    assert meter.is_over_budget(0.0) is True  # >= 0 with nothing spent
    assert meter.is_over_budget(0.01) is False


def test_record_accumulates_total_per_model_per_corpus(tmp_path: Path) -> None:
    meter = SpendMeter(path=default_spend_path(tmp_path))
    meter.record(model="mistralai/mistral-nemo", corpus="fda_maude", amount_usd=0.01)
    meter.record(model="mistralai/mistral-nemo", corpus="fda_maude", amount_usd=0.02)
    meter.record(model="judge:jev", corpus="fda_maude", amount_usd=0.001)
    meter.record(model="mistralai/mistral-nemo", corpus="ami", amount_usd=0.05)

    snapshot = meter.snapshot()
    assert snapshot["total_usd"] == round(0.01 + 0.02 + 0.001 + 0.05, 8)
    assert snapshot["per_model_usd"]["mistralai/mistral-nemo"] == round(0.01 + 0.02 + 0.05, 8)
    assert snapshot["per_model_usd"]["judge:jev"] == 0.001
    assert snapshot["per_corpus_usd"]["fda_maude"] == round(0.01 + 0.02 + 0.001, 8)
    assert snapshot["per_corpus_usd"]["ami"] == 0.05


def test_record_non_positive_amount_is_a_no_op(tmp_path: Path) -> None:
    meter = SpendMeter(path=default_spend_path(tmp_path))
    meter.record(model="m", corpus="c", amount_usd=0.0)
    meter.record(model="m", corpus="c", amount_usd=-1.0)
    assert meter.total_usd() == 0.0


def test_is_over_budget(tmp_path: Path) -> None:
    meter = SpendMeter(path=default_spend_path(tmp_path))
    meter.record(model="m", corpus="c", amount_usd=0.49)
    assert meter.is_over_budget(0.50) is False
    meter.record(model="m", corpus="c", amount_usd=0.01)
    assert meter.is_over_budget(0.50) is True


def test_corpus_usd_and_is_corpus_over_budget(tmp_path: Path) -> None:
    """Regression for the 2026-09-22 paid-teacher-budget bug: a ceiling
    meant for ONE corpus must not trip on another corpus's spend sharing
    the same meter file."""
    meter = SpendMeter(path=default_spend_path(tmp_path))
    assert meter.corpus_usd("synthetic") == 0.0
    assert meter.is_corpus_over_budget("synthetic", 0.0) is True  # >= 0 with nothing spent

    meter.record(model="m", corpus="fda_maude", amount_usd=5.89)  # unrelated corpus, big spend
    meter.record(model="m", corpus="synthetic", amount_usd=0.0025)  # tiny synthetic spend

    assert meter.total_usd() >= 5.0
    assert meter.is_over_budget(5.0) is True  # the (correctly) unchanged whole-meter check

    assert meter.corpus_usd("synthetic") == 0.0025
    assert meter.is_corpus_over_budget("synthetic", 5.0) is False  # must NOT trip

    meter.record(model="m", corpus="synthetic", amount_usd=4.9975)  # now at the ceiling
    assert meter.corpus_usd("synthetic") == 5.0
    assert meter.is_corpus_over_budget("synthetic", 5.0) is True


def test_shared_file_accumulates_across_two_meter_instances(tmp_path: Path) -> None:
    path = default_spend_path(tmp_path)
    silver_label_meter = SpendMeter(path=path)
    synth_runner_meter = SpendMeter(path=path)

    silver_label_meter.record(model="a", corpus="c1", amount_usd=0.1)
    synth_runner_meter.record(model="b", corpus="c2", amount_usd=0.2)

    assert silver_label_meter.total_usd() == 0.3
    assert synth_runner_meter.total_usd() == 0.3


def test_persists_across_instances(tmp_path: Path) -> None:
    path = default_spend_path(tmp_path)
    SpendMeter(path=path).record(model="m", corpus="c", amount_usd=1.23)
    assert SpendMeter(path=path).total_usd() == 1.23


def test_corrupt_file_treated_as_empty(tmp_path: Path) -> None:
    path = default_spend_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    meter = SpendMeter(path=path)
    assert meter.total_usd() == 0.0
    assert meter.record(model="m", corpus="c", amount_usd=1.0) == 1.0


# -- one meter per machine (guardrails review 2026-09-22, row 33) ----------


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
    )


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git (not in the app image)")
def test_a_worktrees_var_maps_to_the_main_checkouts(tmp_path: Path) -> None:
    main = tmp_path / "main"
    main.mkdir()
    _git(main, "init", "-q")
    _git(main, "commit", "-q", "--allow-empty", "-m", "init")
    worktree = tmp_path / "agent-x"
    _git(main, "worktree", "add", "-q", "--detach", str(worktree))

    main_var = main.resolve() / "var"
    assert shared_var_root(worktree / "var") == main_var
    assert shared_var_root(main / "var") == main_var
    assert default_spend_path(worktree / "var") == main_var / "data" / "silver" / "spend.json"
    # Not a checkout's own var/: left alone (snapshots, tmp dirs, a nested var/).
    (main / "sub").mkdir()
    assert shared_var_root(main / "sub" / "var") == (main / "sub" / "var").resolve()
    assert shared_var_root(tmp_path / "var") == (tmp_path / "var").resolve()
