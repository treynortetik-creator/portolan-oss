"""Tests for `scripts/data/synth_batches.py` (the multi-world synthetic-
data batch runner, added 2026-09-21).

`scripts/` isn't a Python package (no `__init__.py`, matching every other
file there), so this loads the module by file path -- same pattern as
`tests/test_set_owners.py`. No network: `run_batches` is called with a
fake, no-network teacher client (`tests/synth_fakes.py`) and a permissive
rate limiter throughout.

Every test passes an explicit `export_dir` (under its own `tmp_path`,
alongside `out_root`) to `run_batches` -- `training_export.export_batch`'s
default (`DEFAULT_SILVER_DIR`, the real `var/data/silver/synthetic/` tree)
is a PROJECT-ROOT path, not test-isolated on its own; a test that calls
`run_batches`/`export_batch` without an explicit path writes fake
`faketeacher/*`-authored JSONL into the real training directory (caught
live 2026-09-21: `world-101.jsonl` .. `world-501.jsonl` and several
`batch-*.jsonl` files leaked there from an earlier, less careful version
of this file). `test_run_batches_and_export_never_touch_project_var`
below is the regression guard for that.

Run with:
    TZ=UTC PORTOLAN_ROLE=test uv run pytest -q tests/test_synth_batches.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from corpus.generator import prose, training_export
from portolan.training.spend import SpendMeter
from tests._paths import require_repo_path
from tests.synth_fakes import FAKE_TEACHERS, build_fake_client

_SCRIPT = require_repo_path("scripts", "data", "synth_batches.py")


def _load_synth_batches():
    spec = importlib.util.spec_from_file_location("synth_batches_script", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["synth_batches_script"] = module
    spec.loader.exec_module(module)
    return module


def _fakes():
    return (
        build_fake_client(),
        prose.TeacherRotation(list(FAKE_TEACHERS)),
        prose.TokenBucketLimiter(requests_per_minute=10_000, requests_per_day=10_000_000),
    )


def _project_var_snapshot() -> set[Path]:
    """Every file currently under the real project's `var/` -- taken
    before a test runs the (fake-teacher) pipeline, compared after, so a
    test that accidentally writes to the project-root default is caught
    immediately rather than silently polluting `var/`."""
    var_dir = training_export.PROJECT_ROOT / "var"
    if not var_dir.exists():
        return set()
    return {p for p in var_dir.rglob("*") if p.is_file()}


@pytest.fixture(autouse=True)
def _stub_key_check(monkeypatch: pytest.MonkeyPatch) -> None:
    """No test in this file may reach the real, network-touching
    `prose.free_model_daily_requests_remaining` (the pre-world OpenRouter
    key check added 2026-09-21 for the daily-limit fix). Most tests below
    predate that check and pass neither `no_key_check` nor `key_check` to
    `run_batches`, so without this stub they fall through to `_free_
    requests_remaining`'s real-check branch and hit OpenRouter's live
    `GET /api/v1/key` using whatever `OPENROUTER_API_KEY` `_env_or_dotenv`
    finds -- checked in the real environment OR `REPO_ROOT/.env`, so the
    check still runs even when the shell's own `OPENROUTER_API_KEY` is
    unset, as long as a `.env` file sets it. That is exactly how these
    tests started failing on 2026-09-22 with `DailyBudgetExhausted`: the
    account's real remaining free-tier budget was 182 when the daily-limit
    fix was tested and is 0 now, so the suite's result started depending
    on a number outside the test's control.

    Stubbing the module-level function (not `_free_requests_remaining`
    itself, and not `run_batches`'s `no_key_check`/`key_check` parameters)
    keeps the runner's production behavior exactly as it was -- a test
    that explicitly injects its own `key_check` callable, or passes
    `no_key_check=True`, is unaffected; this only replaces what a real
    key check would otherwise have called."""
    monkeypatch.setattr(prose, "free_model_daily_requests_remaining", lambda *args, **kwargs: None)


def test_run_batches_never_calls_real_key_check_with_fakes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    """Regression guard for the bug `_stub_key_check` above fixes, and for
    the "one note per run, not per world" fix (2026-09-22).

    A spy that RAISES on any call (the original version of this test) is
    too brittle: on a checkout whose `.env` carries a real `OPENROUTER_
    API_KEY` (this repo's own dev checkout does -- see `_stub_key_check`'s
    docstring), `_env_or_dotenv` finds it via `.env` even with the
    shell's own `OPENROUTER_API_KEY` deleted, `run_batches` reaches the
    key check exactly as designed, and a raising spy then fails the test
    for the wrong reason (that is exactly what broke this test on
    2026-09-22). This version records calls in a list instead and checks
    two things:

    1. With NEITHER key source available (env var deleted, and `REPO_
       ROOT` repointed at an empty directory so the `.env` fallback finds
       nothing either -- `.env` is otherwise outside this test's control),
       `run_batches` with fakes must never call `prose.free_model_daily_
       requests_remaining` at all: `_free_requests_remaining` should
       short-circuit on `_env_or_dotenv` finding nothing.
    2. With a (fake, non-secret) key present instead, the spy -- patched
       onto the CANONICAL `corpus.generator.prose` module object -- must
       be the thing `synth_batches.py` actually reaches: it imports
       `prose` as a module and calls `prose.free_model_daily_requests_
       remaining(...)` through that module attribute (never a name
       captured at import time), so patching the attribute here must be
       visible to it. The check still runs once per world (2 worlds
       here), but the "continuing without it" note must print to stdout
       only ONCE for the whole run, not once per world, and must never
       become one of the per-world summary lines `run_batches` returns."""
    mod = _load_synth_batches()
    client, rotation, limiter = _fakes()

    # -- 1. No key anywhere: the check must not even be attempted. --
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path / "no-env-here")

    no_key_calls: list[object] = []

    def _record_no_key_call(*args: object, **kwargs: object) -> int | None:
        no_key_calls.append(args)
        return None

    monkeypatch.setattr(prose, "free_model_daily_requests_remaining", _record_no_key_call)

    lines = mod.run_batches(
        worlds_count=1,
        num_accounts=2,
        per_kind=4,
        type_mix=None,
        out_root=tmp_path / "synth-nokey",
        base_seed=1001,
        client=client,
        rotation=rotation,
        limiter=limiter,
        export_dir=tmp_path / "silver-nokey",
    )
    assert no_key_calls == []
    assert any("seed=1001" in line for line in lines)

    # -- 2. A key present: the runner must reach it through the module
    # attribute, and print the failure note only once across N worlds. --
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-test-key-not-real")
    calls: list[str] = []

    def _record_key_call(api_key: str, **kwargs: object) -> int | None:
        calls.append(api_key)
        return None

    monkeypatch.setattr(prose, "free_model_daily_requests_remaining", _record_key_call)

    lines2 = mod.run_batches(
        worlds_count=2,
        num_accounts=2,
        per_kind=4,
        type_mix=None,
        out_root=tmp_path / "synth-key",
        base_seed=1101,
        client=client,
        rotation=rotation,
        limiter=limiter,
        export_dir=tmp_path / "silver-key",
    )
    assert calls == ["fake-test-key-not-real", "fake-test-key-not-real"]
    assert len(lines2) == 2  # the note never becomes a per-world summary line
    out = capsys.readouterr().out
    assert out.count("key check: OpenRouter key endpoint failed") == 1


def test_run_batches_runs_every_world(tmp_path: Path):
    mod = _load_synth_batches()
    client, rotation, limiter = _fakes()
    out_root = tmp_path / "synth"
    export_dir = tmp_path / "silver"

    lines = mod.run_batches(
        worlds_count=2,
        num_accounts=2,
        per_kind=5,
        type_mix=None,
        out_root=out_root,
        base_seed=201,
        client=client,
        rotation=rotation,
        limiter=limiter,
        export_dir=export_dir,
    )

    assert len(lines) == 2
    assert "seed=201" in lines[0]
    assert "seed=202" in lines[1]
    for seed in (201, 202):
        batch_dir = mod.batch_dir_for(out_root, seed)
        assert (batch_dir / "synth_report.json").exists()
        assert (batch_dir / "manifest.json").exists()
        assert (export_dir / f"{batch_dir.name}.jsonl").exists()
        # Never the real project-root default when export_dir is given.
        assert not training_export.default_export_path(batch_dir).exists()


def test_run_batches_shares_one_rate_limiter_across_worlds(tmp_path: Path):
    mod = _load_synth_batches()
    client, rotation, limiter = _fakes()
    out_root = tmp_path / "synth"

    mod.run_batches(
        worlds_count=3,
        num_accounts=2,
        per_kind=4,
        type_mix=None,
        out_root=out_root,
        base_seed=301,
        client=client,
        rotation=rotation,
        limiter=limiter,
        export_dir=tmp_path / "silver",
    )
    # One shared limiter instance across all 3 worlds -- its day_count is
    # the SUM of every request across every world, not per-world.
    assert limiter.day_count > 0
    assert rotation.total_requests == limiter.day_count


def test_run_batches_is_resumable_and_does_not_regenerate(tmp_path: Path):
    mod = _load_synth_batches()
    client, rotation, limiter = _fakes()
    out_root = tmp_path / "synth"
    export_dir = tmp_path / "silver"

    first_lines = mod.run_batches(
        worlds_count=2,
        num_accounts=2,
        per_kind=4,
        type_mix=None,
        out_root=out_root,
        base_seed=401,
        client=client,
        rotation=rotation,
        limiter=limiter,
        export_dir=export_dir,
    )
    assert not any("resumed" in line for line in first_lines)

    requests_after_first_run = rotation.total_requests

    second_lines = mod.run_batches(
        worlds_count=2,
        num_accounts=2,
        per_kind=4,
        type_mix=None,
        out_root=out_root,
        base_seed=401,
        client=client,
        rotation=rotation,
        limiter=limiter,
        export_dir=export_dir,
    )
    assert all("resumed: already done" in line for line in second_lines)
    # No new teacher requests made on the resumed run -- both worlds were
    # already complete, so nothing was regenerated.
    assert rotation.total_requests == requests_after_first_run

    # The two runs report identical numbers (same underlying files).
    for first, second in zip(first_lines, second_lines, strict=True):
        assert first.split(" [resumed")[0] == second.split(" [resumed")[0]


def test_run_batches_with_type_mix_reaches_minimums(tmp_path: Path):
    mod = _load_synth_batches()
    client, rotation, limiter = _fakes()
    out_root = tmp_path / "synth"
    type_mix = {"pricing": 4, "risk": 3}

    mod.run_batches(
        worlds_count=1,
        num_accounts=3,
        per_kind=15,
        type_mix=type_mix,
        out_root=out_root,
        base_seed=501,
        client=client,
        rotation=rotation,
        limiter=limiter,
        export_dir=tmp_path / "silver",
    )
    batch_dir = mod.batch_dir_for(out_root, 501)
    report = json.loads((batch_dir / "synth_report.json").read_text(encoding="utf-8"))
    for fact_type, minimum in type_mix.items():
        assert report["planted_by_type"].get(fact_type, 0) >= minimum


def test_main_rejects_invalid_type_mix(tmp_path: Path, capsys):
    mod = _load_synth_batches()
    rc = mod.main(
        [
            "--worlds",
            "1",
            "--per-kind",
            "5",
            "--out",
            str(tmp_path / "synth"),
            "--export-dir",
            str(tmp_path / "silver"),
            "--type-mix",
            "not_a_real_type=5",
        ]
    )
    assert rc == 1
    out = capsys.readouterr().out
    assert "invalid --type-mix" in out


def test_run_batches_and_export_never_touch_project_var(tmp_path: Path):
    """End-to-end regression guard: running the batch runner (which itself
    calls `training_export.export_batch` internally) with fakes, plus a
    direct `training_export.export_batch` call, must not create or modify
    a single file under the real project's `var/` when explicit,
    tmp_path-scoped directories are given throughout."""
    before = _project_var_snapshot()

    mod = _load_synth_batches()
    client, rotation, limiter = _fakes()
    out_root = tmp_path / "synth"
    export_dir = tmp_path / "silver"

    mod.run_batches(
        worlds_count=2,
        num_accounts=2,
        per_kind=4,
        type_mix=None,
        out_root=out_root,
        base_seed=601,
        client=client,
        rotation=rotation,
        limiter=limiter,
        export_dir=export_dir,
    )

    # A second, direct export call (mirrors what a test of
    # training_export.py alone would do) into its own explicit path.
    batch_dir = mod.batch_dir_for(out_root, 601)
    training_export.export_batch(batch_dir, tmp_path / "direct-export.jsonl")

    after = _project_var_snapshot()
    assert after == before, sorted(str(p) for p in after - before)


# --------------------------------------------------------------------------
# Daily free-tier budget handling, added 2026-09-21 -- see the module
# docstring above. A real run hit the account-wide 1,000-requests/day cap
# mid-run and silently produced fifteen empty worlds before this fix;
# these tests use a tiny `requests_per_day` (still via the fake, no-
# network client) to reproduce that `stopped_early` report shape exactly,
# rather than hand-fabricating one.
# --------------------------------------------------------------------------


def test_run_batches_stops_on_daily_budget_and_never_starts_next_world(tmp_path, capsys):
    mod = _load_synth_batches()
    client = build_fake_client()
    rotation = prose.TeacherRotation(list(FAKE_TEACHERS))
    # requests_per_day=1: the first teacher call in world 701 succeeds,
    # the next one hits `prose.DailyLimitExceeded` -- exactly the shape
    # `cli.synth()` swallows internally into a `stopped_early` report.
    limiter = prose.TokenBucketLimiter(requests_per_minute=10_000, requests_per_day=1)
    out_root = tmp_path / "synth"
    export_dir = tmp_path / "silver"

    with pytest.raises(mod.DailyBudgetExhausted):
        mod.run_batches(
            worlds_count=3,
            num_accounts=2,
            per_kind=4,
            type_mix=None,
            out_root=out_root,
            base_seed=701,
            requests_per_day=1,
            client=client,
            rotation=rotation,
            limiter=limiter,
            export_dir=export_dir,
            no_key_check=True,
        )

    batch_701 = mod.batch_dir_for(out_root, 701)
    assert batch_701.exists()
    report = json.loads((batch_701 / "synth_report.json").read_text(encoding="utf-8"))
    assert report["stopped_early"] is True
    assert "requests/day budget" in report["stop_reason"]

    # The next seeds were never started -- no directories for them at all.
    assert not mod.batch_dir_for(out_root, 702).exists()
    assert not mod.batch_dir_for(out_root, 703).exists()

    out = capsys.readouterr().out
    assert "STOPPED (daily free-tier limit) after world 701" in out
    assert "rerun the same command after the cap resets" in out


def test_run_batches_resume_redoes_stopped_world_and_skips_clean_one(tmp_path):
    mod = _load_synth_batches()
    client, rotation, limiter = _fakes()
    out_root = tmp_path / "synth"
    export_dir = tmp_path / "silver"

    mod.run_batches(
        worlds_count=2,
        num_accounts=2,
        per_kind=4,
        type_mix=None,
        out_root=out_root,
        base_seed=801,
        client=client,
        rotation=rotation,
        limiter=limiter,
        export_dir=export_dir,
        no_key_check=True,
    )

    # Simulate world 801 having stopped early on the daily budget (the
    # exact scenario this fix targets) after the fact, leaving world 802
    # clean -- a deterministic way to test the reset/skip logic itself.
    stopped_batch = mod.batch_dir_for(out_root, 801)
    clean_batch = mod.batch_dir_for(out_root, 802)
    report_path = stopped_batch / "synth_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["stopped_early"] = True
    report["stop_reason"] = "TokenBucketLimiter: 1000 requests/day budget used up"
    report["verified"] = 0
    report_path.write_text(json.dumps(report), encoding="utf-8")

    resumed_lines = mod.run_batches(
        worlds_count=2,
        num_accounts=2,
        per_kind=4,
        type_mix=None,
        out_root=out_root,
        base_seed=801,
        client=client,
        rotation=rotation,
        limiter=limiter,
        export_dir=export_dir,
        no_key_check=True,
    )

    assert len(resumed_lines) == 2
    # World 801: regenerated fresh, not treated as already-done.
    assert "seed=801" in resumed_lines[0]
    assert "[resumed: already done]" not in resumed_lines[0]
    new_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert new_report["stopped_early"] is False

    # The old, incomplete attempt was moved aside, not silently reused.
    incomplete_dirs = list(out_root.glob("world-801.incomplete-*"))
    assert len(incomplete_dirs) == 1

    # World 802 was already clean -- skipped, not regenerated.
    assert "seed=802" in resumed_lines[1]
    assert "[resumed: already done]" in resumed_lines[1]
    assert clean_batch.exists()


def test_run_batches_key_check_stops_below_threshold_and_continues_on_error(tmp_path, capsys):
    mod = _load_synth_batches()
    client, rotation, limiter = _fakes()
    out_root = tmp_path / "synth"
    export_dir = tmp_path / "silver"

    # Below threshold -- must stop before world 901 is ever attempted, no
    # real network call (a plain callable stands in for the OpenRouter
    # key check).
    with pytest.raises(mod.DailyBudgetExhausted):
        mod.run_batches(
            worlds_count=1,
            num_accounts=2,
            per_kind=4,
            type_mix=None,
            out_root=out_root,
            base_seed=901,
            client=client,
            rotation=rotation,
            limiter=limiter,
            export_dir=export_dir,
            key_check=lambda: 5,
        )
    assert not mod.batch_dir_for(out_root, 901).exists()
    out = capsys.readouterr().out
    assert "daily free-tier budget nearly spent (5 left); rerun after reset" in out

    # The endpoint erroring (`key_check` returns None) must NOT stop the
    # run -- log and continue.
    lines = mod.run_batches(
        worlds_count=1,
        num_accounts=2,
        per_kind=4,
        type_mix=None,
        out_root=out_root,
        base_seed=902,
        client=client,
        rotation=rotation,
        limiter=limiter,
        export_dir=export_dir,
        key_check=lambda: None,
    )
    assert mod.batch_dir_for(out_root, 902).exists()
    assert any("seed=902" in line for line in lines)
    out2 = capsys.readouterr().out
    assert "key check" in out2
    assert "continuing" in out2


def test_main_returns_3_on_daily_budget_exhausted(tmp_path, monkeypatch):
    mod = _load_synth_batches()

    def fake_run_batches(**kwargs):
        raise mod.DailyBudgetExhausted("boom")

    monkeypatch.setattr(mod, "run_batches", fake_run_batches)
    rc = mod.main(
        [
            "--worlds",
            "1",
            "--per-kind",
            "5",
            "--out",
            str(tmp_path / "synth"),
            "--export-dir",
            str(tmp_path / "silver"),
            "--no-key-check",
        ]
    )
    assert rc == 3


# --------------------------------------------------------------------------
# Paid teachers -- added 2026-09-22 (`--teachers`, shared across worlds).
# --------------------------------------------------------------------------


def test_run_batches_paid_teachers_skip_key_check_and_stop_at_budget(tmp_path, capsys):
    mod = _load_synth_batches()
    client = build_fake_client()
    teachers = ["deepseek/deepseek-v4-flash-0731"]
    spend = SpendMeter(path=tmp_path / "spend.json")
    spend.record(model=teachers[0], corpus="synthetic", amount_usd=10.0)  # already over budget

    key_check_calls: list[int] = []

    def _key_check() -> int | None:
        key_check_calls.append(1)
        return 999

    out_root = tmp_path / "synth"
    export_dir = tmp_path / "silver"

    with pytest.raises(mod.PaidBudgetExhausted):
        mod.run_batches(
            worlds_count=2,
            num_accounts=2,
            per_kind=4,
            type_mix=None,
            out_root=out_root,
            base_seed=1101,
            client=client,
            export_dir=export_dir,
            teachers=teachers,
            budget_usd=1.0,
            spend=spend,
            key_check=_key_check,  # must never be called -- paid mode forces no_key_check
        )

    assert key_check_calls == []
    batch_1101 = mod.batch_dir_for(out_root, 1101)
    assert batch_1101.exists()
    report = json.loads((batch_1101 / "synth_report.json").read_text(encoding="utf-8"))
    assert report["stopped_early"] is True
    assert prose.PAID_BUDGET_STOP_MARKER in report["stop_reason"]
    assert not mod.batch_dir_for(out_root, 1102).exists()

    out = capsys.readouterr().out
    assert "STOPPED (paid-teacher budget): synthetic spend $10.00 of $1.00 after world 1101" in out


def test_run_batches_paid_budget_checks_synthetic_corpus_not_meter_total(tmp_path, capsys):
    """Regression for the 2026-09-22 bug: a concurrent silver-labeling run
    sharing the same `spend.json` meter pushed `total_usd` to $5.89 while
    the synthetic corpus itself had spent only $0.0025, and the paid-
    teacher run stopped immediately anyway because the old check compared
    `spend.total_usd()` (every corpus combined) against `--budget-usd`
    instead of `per_corpus_usd["synthetic"]`. A big *unrelated*-corpus
    spend must never stop a paid-teacher run whose own synthetic spend is
    still under the ceiling."""
    mod = _load_synth_batches()
    client = build_fake_client()
    teachers = ["deepseek/deepseek-v4-flash-0731"]
    spend = SpendMeter(path=tmp_path / "spend.json")
    # A concurrent silver-labeling run against a different corpus: well
    # past any reasonable synthetic-run budget, entirely unrelated.
    spend.record(model="some-silver-model", corpus="fda_maude", amount_usd=5.89)
    assert spend.total_usd() >= 5.0  # sanity: the bug's trigger condition holds

    out_root = tmp_path / "synth"
    export_dir = tmp_path / "silver"

    lines = mod.run_batches(
        worlds_count=1,
        num_accounts=2,
        per_kind=4,
        type_mix=None,
        out_root=out_root,
        base_seed=1301,
        client=client,
        export_dir=export_dir,
        teachers=teachers,
        budget_usd=5.0,
        spend=spend,
    )

    assert len(lines) == 1
    assert "seed=1301" in lines[0]
    assert "STOPPED" not in lines[0]
    report = json.loads(
        (mod.batch_dir_for(out_root, 1301) / "synth_report.json").read_text(encoding="utf-8")
    )
    assert report["stopped_early"] is False

    out = capsys.readouterr().out
    # Run-start print names the ceiling and the (near-zero) synthetic spend.
    assert "paid-teacher budget: synthetic spend $0.00 of $5.00 ceiling" in out
    # This world's own cost shows up in its summary line.
    assert "synthetic_cost=$" in lines[0]
    # The synthetic corpus line moved; the unrelated corpus line didn't.
    assert spend.corpus_usd("synthetic") > 0.0
    assert spend.corpus_usd("fda_maude") == 5.89


def test_run_batches_paid_budget_stops_when_synthetic_corpus_line_at_ceiling(tmp_path):
    """The other half of the regression guard: when the SYNTHETIC line
    itself is at or above the ceiling, the run must still stop (exit 4 at
    the `main()` level), even though nothing here is a meter-total check
    any more."""
    mod = _load_synth_batches()
    client = build_fake_client()
    teachers = ["deepseek/deepseek-v4-flash-0731"]
    spend = SpendMeter(path=tmp_path / "spend.json")
    spend.record(model=teachers[0], corpus="synthetic", amount_usd=2.50)  # == budget-usd below

    with pytest.raises(mod.PaidBudgetExhausted):
        mod.run_batches(
            worlds_count=1,
            num_accounts=2,
            per_kind=4,
            type_mix=None,
            out_root=tmp_path / "synth",
            base_seed=1401,
            client=client,
            export_dir=tmp_path / "silver",
            teachers=teachers,
            budget_usd=2.50,
            spend=spend,
        )


def test_run_batches_teachers_all_free_still_respects_key_check(tmp_path):
    mod = _load_synth_batches()
    client = build_fake_client()

    key_check_calls: list[int] = []

    def _key_check() -> int | None:
        key_check_calls.append(1)
        return 5  # below MIN_FREE_REQUESTS_REMAINING -- must still stop the run

    with pytest.raises(mod.DailyBudgetExhausted):
        mod.run_batches(
            worlds_count=1,
            num_accounts=2,
            per_kind=4,
            type_mix=None,
            out_root=tmp_path / "synth",
            base_seed=1201,
            client=client,
            export_dir=tmp_path / "silver",
            teachers=list(FAKE_TEACHERS),  # all ':free' -- unchanged free path
            key_check=_key_check,
        )
    assert key_check_calls == [1]


def test_main_rejects_mixed_teachers(tmp_path, capsys):
    mod = _load_synth_batches()
    rc = mod.main(
        [
            "--worlds",
            "1",
            "--per-kind",
            "5",
            "--out",
            str(tmp_path / "synth"),
            "--export-dir",
            str(tmp_path / "silver"),
            "--no-key-check",
            "--teachers",
            "deepseek/deepseek-v4-flash-0731:free,deepseek/deepseek-v4.1-flash",
        ]
    )
    assert rc == 1
    assert "invalid --teachers" in capsys.readouterr().out


def test_main_returns_4_on_paid_budget_exhausted(tmp_path, monkeypatch):
    mod = _load_synth_batches()

    def fake_run_batches(**kwargs):
        raise mod.PaidBudgetExhausted("boom")

    monkeypatch.setattr(mod, "run_batches", fake_run_batches)
    rc = mod.main(
        [
            "--worlds",
            "1",
            "--per-kind",
            "5",
            "--out",
            str(tmp_path / "synth"),
            "--export-dir",
            str(tmp_path / "silver"),
            "--teachers",
            "deepseek/deepseek-v4-flash-0731",
        ]
    )
    assert rc == 4
