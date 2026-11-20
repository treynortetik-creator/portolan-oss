"""The v2 world batch runner (`01-spec/world-generator-v2-2026-09-22.md`
section 9, WP11): one world, or a round of them, from simulation to an
exported, checked world.

Stages per world, in order:

| Stage | Work package | What it does |
|---|---|---|
| simulate | WP1 | `simulate_world` with the contamination name guard; writes `world.yaml` and `truth-table.yaml` |
| plan | WP3 | `plan_world` (the per-type cap applies here); writes `world_plan.json`, `plan/crm/accounts.csv`, `plan/code_labels.jsonl` |
| render | WP3 | renders every document skeleton; a skeleton that fails stops the world |
| write | WP4 | `writer.run_world` into `writer/` (document checkpoints, calls, labels, decoys) |
| completeness | WP5 | `completeness.run_completeness` into `completeness/`: the blind teacher, the audit teacher and the label judge |
| noise | WP6 | `noise.noise_completed_world` into `noise/`: every section 5 operation on WP5's shipped documents and labels, and the near-duplicates |
| export | WP7 | `export.export_world`: `sources.jsonl`, `labels.jsonl`, `decoys.jsonl`, `manifest.json`, `files/`, and `training_records.jsonl` for train and validation worlds |
| contamination | WP8 | every exported source's text against `var/contamination/` |
| report | WP11 | `run_report.json` for the world; `round_report.json` for the invocation |

**Resume.** Stage state lives in `<world>/stages.json`; the deterministic
stages are recomputed and compared by hash, and every paid stage resumes
per document through its own checkpoints (a finished document makes no
call).

**Budget.** `--budget-usd` is a hard cap on what this output root spends,
checked before every paid call against the shared spend meter: the growth
of the `synthetic_v2` line (writer, teacher, audit, rewrites and
near-duplicates) and the `label_judge` line since the root's recorded
baselines (`<out>/budget.json`), plus the call's worst case, may not pass
it. Spend by any other session on those lines during the run counts
against it too (conservative). `--line-cap-usd` also caps the
`synthetic_v2` line's absolute total.

**Dry run.** `--dry-run` writes every document with the WP4 test fixture
writer and runs WP5 with a fixture teacher and judge (`tests/v2_writer_fakes.py`,
`tests/v2_pipeline_fakes.py`: fixed test tokens, no network, $0) into its own
root marked `DRY_RUN`; that text must never reach training.

    uv run python scripts/data/synth_v2.py --worlds medical_devices:train:3 --budget-usd 6.10
    uv run python scripts/data/synth_v2.py --worlds round1 --budget-usd 55
    uv run python scripts/data/synth_v2.py --dry-run --worlds pilot --no-contamination-check
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from corpus.generator import prose  # noqa: E402
from corpus.generator.v2 import completeness as wp5  # noqa: E402
from corpus.generator.v2 import export as wp7  # noqa: E402
from corpus.generator.v2 import noise as wp6  # noqa: E402
from corpus.generator.v2 import writer as w  # noqa: E402
from corpus.generator.v2.models import Split, World  # noqa: E402
from corpus.generator.v2.names import NameGuard  # noqa: E402
from corpus.generator.v2.noise_neardup import NearDupConfig, write_near_duplicates  # noqa: E402
from corpus.generator.v2.packs.schema import VerticalPack, load_pack  # noqa: E402
from corpus.generator.v2.planner import (  # noqa: E402
    WorldDocPlan,
    plan_world,
)
from corpus.generator.v2.renderers import materialize_crm_accounts, render  # noqa: E402
from corpus.generator.v2.rounds import PACKS_DIR, WorldSpec, parse_worlds  # noqa: E402
from corpus.generator.v2.simulator import simulate_world  # noqa: E402
from portolan.training import contamination as contam  # noqa: E402
from portolan.training.label_judge import (  # noqa: E402
    DEFAULT_LABEL_JUDGE_MODEL,
    LABEL_JUDGE_SPEND_KEY,
    JudgeCall,
    judge_model_allowed,
)
from portolan.training.pipeline import assert_not_holdout  # noqa: E402
from portolan.training.spend import SpendMeter, default_spend_path  # noqa: E402

__all__ = [
    "RUNNER_VERSION",
    "STAGES",
    "RunContext",
    "RunnerError",
    "SharedMeterBudget",
    "main",
    "run_one_world",
    "run_worlds",
]

RUNNER_VERSION = "wp11-2026-09-23.2"
STAGES = (
    "simulate",
    "plan",
    "render",
    "write",
    "completeness",
    "noise",
    "export",
    "contamination",
    "report",
)
DEFAULT_OUT = REPO_ROOT / "var" / "data" / "synth_v2" / "worlds"
DEFAULT_DRY_RUN_OUT = REPO_ROOT / "var" / "data" / "synth_v2" / "dryrun"
DEFAULT_CONTAMINATION_DIR = REPO_ROOT / "var" / "contamination"
FIXTURE_PIPELINE = REPO_ROOT / "tests" / "v2_pipeline_fakes.py"
DRY_RUN_MARKER = "DRY_RUN"
WRITER_SHARE_CAP = 0.25  # spec section 4: no model writes more than 25% of a world
# The label judge's worst case per call (3,000 output tokens at a flash
# model's list price, times three), checked before each call.
JUDGE_WORST_CASE_USD = 0.01
# Section 7's test-strictness extras that no stage runs yet.
TEST_STRICTNESS_NOT_BUILT = (
    "test strictness: three-judge disputes (section 7) and search questions (WP10) are not "
    "run by this runner; the audit teacher already reads every document"
)


class RunnerError(RuntimeError):
    """A check before any spend failed (bad writers, bad world list, mixed
    dry-run and real output)."""


# --------------------------------------------------------------------------
# Budget
# --------------------------------------------------------------------------


@dataclass
class SharedMeterBudget(w.RunBudget):
    """WP4's `RunBudget`, with the cap checked against the shared meter:
    the growth of the `synthetic_v2` line since `baseline_usd`, plus the
    growth of each `extra_lines` key since its baseline, plus this call's
    worst case, may not pass `cap_usd`. `extra_prices` (model -> prompt and
    completion $ per million) prices non-writer models (WP5's teachers)."""

    baseline_usd: float = 0.0
    extra_prices: dict[str, tuple[float, float]] = field(default_factory=dict)
    extra_lines: dict[str, float] = field(default_factory=dict)

    def run_spent_usd(self) -> float:
        if self.meter is None:
            return self.spent_usd
        spent = max(0.0, self.meter.corpus_usd(self.corpus_key) - self.baseline_usd)
        for key, baseline in self.extra_lines.items():
            spent += max(0.0, self.meter.corpus_usd(key) - baseline)
        return spent

    def worst_case_usd(self, model: str, prompt_chars: int, max_tokens: int) -> float:
        price = self.extra_prices.get(model)
        if price is None:
            return super().worst_case_usd(model, prompt_chars, max_tokens)
        list_price = (math.ceil(prompt_chars / 3.0) * price[0] + max_tokens * price[1]) / 1e6
        return list_price * self.price_safety_factor

    def precheck(self, model: str, prompt_chars: int, max_tokens: int) -> None:
        worst = self.worst_case_usd(model, prompt_chars, max_tokens)
        self.precheck_usd(worst)
        if self.meter is not None and self.corpus_cap_usd is not None:
            line = self.meter.corpus_usd(self.corpus_key)
            if line + worst > self.corpus_cap_usd:
                raise w.WriterBudgetExceeded(
                    f"{prose.PAID_BUDGET_STOP_MARKER}: {self.corpus_key} line ${line:.4f} plus "
                    f"this call's worst case would pass the ${self.corpus_cap_usd:.2f} line cap"
                )

    def precheck_usd(self, worst: float) -> None:
        spent = self.run_spent_usd()
        if spent + worst > self.cap_usd:
            raise w.WriterBudgetExceeded(
                f"{prose.PAID_BUDGET_STOP_MARKER}: spend ${spent:.4f} since the run's baseline "
                f"plus this call's worst case ${worst:.4f} would pass the ${self.cap_usd:.2f} cap"
            )

    def can_spend(self, worst: float) -> bool:
        return self.run_spent_usd() + worst <= self.cap_usd


def open_budget(
    out_root: Path,
    meter: SpendMeter,
    *,
    cap_usd: float,
    line_cap_usd: float | None,
    reset_baseline: bool,
    now: Callable[[], datetime],
) -> SharedMeterBudget:
    """The root's budget ledger (`budget.json`): each counted meter line's
    value when this root first ran, so a resumed run keeps counting from
    there. A line the ledger does not know yet starts counting now."""
    ledger = out_root / "budget.json"
    data: dict[str, Any] = {}
    if ledger.exists() and not reset_baseline:
        data = json.loads(ledger.read_text(encoding="utf-8"))
        if Path(data.get("spend_path", "")).resolve() != meter.path.resolve():
            raise RunnerError(
                f"{ledger} was opened against the meter {data.get('spend_path')}, not "
                f"{meter.path}; pass --reset-budget-baseline to start counting again"
            )
    else:
        data = {
            "corpus_key": w.CORPUS_KEY,
            "spend_path": str(meter.path),
            "baseline_usd": meter.corpus_usd(w.CORPUS_KEY),
            "created_at": now().isoformat(),
        }
    lines = dict(data.get("extra_baselines_usd") or {})
    lines.setdefault(LABEL_JUDGE_SPEND_KEY, meter.corpus_usd(LABEL_JUDGE_SPEND_KEY))
    data["extra_baselines_usd"] = lines
    data["cap_usd"] = cap_usd
    data["line_cap_usd"] = line_cap_usd
    data["opened_at"] = now().isoformat()
    _atomic_write(ledger, json.dumps(data, indent=2))
    return SharedMeterBudget(
        cap_usd=cap_usd,
        meter=meter,
        corpus_cap_usd=line_cap_usd,
        baseline_usd=float(data["baseline_usd"]),
        extra_prices=dict(wp5.TEACHER_PRICES),
        extra_lines={k: float(v) for k, v in lines.items()},
    )


# --------------------------------------------------------------------------
# Context and stage bookkeeping
# --------------------------------------------------------------------------


@dataclass
class RunContext:
    out_root: Path
    writers: tuple[str, ...]
    backend: w.WriterBackend
    budget: SharedMeterBudget
    refs: contam.ContaminationReferences | None
    refs_dir: Path | None
    # WP5's teachers (model -> extraction backend). `client_meters`: the
    # backend's client records each call on the meter itself.
    teacher_factory: Callable[[str], Any] | None = None
    client_meters: bool = False
    # The label judge on WP5's silver labels; `None` runs WP5 without it.
    judge_model: str | None = DEFAULT_LABEL_JUDGE_MODEL
    judge_call: JudgeCall | None = None
    name_guard: NameGuard | None = None
    dry_run: bool = False
    max_docs: int | None = None
    progress: Callable[[str], None] = print
    now: Callable[[], datetime] = field(default=lambda: datetime.now(UTC))
    stop_reason: str | None = None


@dataclass
class StageResult:
    name: str
    status: str  # done / resumed / partial / skipped / stopped / failed
    seconds: float = 0.0
    message: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)
    fingerprint: str | None = None

    def as_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "seconds": round(self.seconds, 2),
            "message": self.message,
            "fingerprint": self.fingerprint,
            "detail": self.detail,
        }


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def world_dir(out_root: Path, spec: WorldSpec) -> Path:
    return out_root / spec.split.value / spec.world_id


# --------------------------------------------------------------------------
# Preflight (before any spend)
# --------------------------------------------------------------------------


def check_writers(writers: tuple[str, ...]) -> None:
    if not writers:
        raise RunnerError("--writers is empty")
    for model in writers:
        if model.lower().startswith("local:") and getattr(w, "LocalWriterBackend", None) is None:
            raise RunnerError(
                f"{model}: local writers need corpus.generator.v2.writer.LocalWriterBackend "
                "(branch feature/local-teachers), which is not on this checkout"
            )
        try:
            w._check_allowed(model)
        except ValueError as exc:
            raise RunnerError(str(exc)) from exc


def writer_coverage(writers: tuple[str, ...], doc_total: int) -> tuple[int, int]:
    """(documents the rotation can write under the 25% cap, the cap)."""
    cap = max(1, math.floor(WRITER_SHARE_CAP * doc_total))
    return len(set(writers)) * cap, cap


def check_output_root(out_root: Path, *, dry_run: bool) -> None:
    assert_not_holdout(out_root)
    marker = out_root / DRY_RUN_MARKER
    if dry_run and (out_root / "budget.json").exists() and not marker.exists():
        raise RunnerError(f"{out_root} holds a real run; a dry run needs its own --out")
    if not dry_run and marker.exists():
        raise RunnerError(
            f"{out_root} is a dry-run root (fixture text); a real run needs its own --out"
        )


def load_world_pack(spec: WorldSpec) -> VerticalPack:
    pack = load_pack(PACKS_DIR / f"{spec.pack}.yaml")
    if pack.held_out and spec.split != Split.TEST:
        raise RunnerError(
            f"{spec.label()}: {spec.pack} is the held-out vertical; it only builds test worlds"
        )
    return pack


# --------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------


def _yaml(data: Any) -> str:
    dumper = getattr(yaml, "CSafeDumper", yaml.SafeDumper)
    return yaml.dump(data, Dumper=dumper, sort_keys=False, allow_unicode=True)


def stage_simulate(
    spec: WorldSpec, pack: VerticalPack, wdir: Path, prior: dict, guard: NameGuard | None = None
) -> tuple[World, StageResult]:
    world = simulate_world(
        pack, seed=spec.seed, months=spec.months, split=spec.split, name_guard=guard
    )
    digest = _sha(world.model_dump_json())
    same = prior.get("fingerprint") == digest and (wdir / "world.yaml").exists()
    if not same:
        _atomic_write(wdir / "world.yaml", _yaml(world.model_dump(mode="json")))
        _atomic_write(wdir / "truth-table.yaml", _yaml(world.to_truth_table()))
    detail = {
        "world_id": world.id,
        "window": [world.window_start.isoformat(), world.window_end.isoformat()],
        "vendor": world.vendor.name,
        "accounts": len(world.accounts),
        "people": len(world.people),
        "events": len(world.events),
        "facts": len(world.facts),
        "name_guard": guard.fingerprint if guard is not None else None,
        "changed_since_last_run": bool(prior.get("fingerprint")) and not same,
    }
    return world, StageResult(
        "simulate", "resumed" if same else "done", detail=detail, fingerprint=digest
    )


def stage_plan(
    world: World, pack: VerticalPack, wdir: Path, prior: dict
) -> tuple[WorldDocPlan, StageResult]:
    world_plan = plan_world(world, pack)
    digest = _sha(world_plan.model_dump_json())
    same = prior.get("fingerprint") == digest and (wdir / "world_plan.json").exists()
    if not same:
        _atomic_write(wdir / "world_plan.json", world_plan.model_dump_json(indent=2))
        materialize_crm_accounts(world_plan.crm_accounts, wdir / "plan")
        _atomic_write(
            wdir / "plan" / "code_labels.jsonl",
            "".join(label.model_dump_json() + "\n" for label in world_plan.labels),
        )
    cap = world_plan.type_cap
    detail = {
        "documents": len(world_plan.doc_plans),
        "by_kind": dict(Counter(d.kind.value for d in world_plan.doc_plans)),
        "by_role": dict(Counter(d.role.value for d in world_plan.doc_plans)),
        "planted_facts": sum(len(d.fact_ids) for d in world_plan.doc_plans),
        "decoys": len(world_plan.decoys),
        "crm_records": len(world_plan.crm_accounts),
        "crm_code_labels": len(world_plan.labels),
        "type_cap": {
            "applied": cap.applied,
            "reason": cap.reason,
            "facts_capped": len(cap.capped_fact_ids),
            "documents_emptied": len(cap.documents_emptied),
            "max_share_before": cap.max_share(cap.facts_before),
            "max_share_after": cap.max_share(cap.facts_after),
            "unmet_types": cap.unmet_types,
        }
        if cap is not None
        else None,
        "changed_since_last_run": bool(prior.get("fingerprint")) and not same,
    }
    return world_plan, StageResult(
        "plan", "resumed" if same else "done", detail=detail, fingerprint=digest
    )


def stage_render(world: World, world_plan: WorldDocPlan, pack: VerticalPack) -> StageResult:
    slots: Counter = Counter()
    failures: dict[str, str] = {}
    for plan in world_plan.doc_plans:
        try:
            rendered = render(plan, world, world_plan, pack)
        except Exception as exc:  # noqa: BLE001 -- every broken skeleton is reported, then the world stops
            failures[plan.id] = f"{type(exc).__name__}: {exc}"[:300]
            continue
        slots.update(s.kind.value for s in rendered.slots)
    status = "failed" if failures else "done"
    detail = {"documents": len(world_plan.doc_plans), "slots": dict(slots), "failures": failures}
    message = f"{len(failures)} skeletons failed to render" if failures else None
    return StageResult("render", status, message=message, detail=detail)


def _writer_ids(
    world_plan: WorldDocPlan, config: w.WriterConfig, max_docs: int | None
) -> list[str] | None:
    if max_docs is None:
        return None
    return [p.id for p in w.writer_doc_plans(world_plan, config)[:max_docs]]


def _session(
    ctx: RunContext, world_plan: WorldDocPlan, call_log: Path, written: list[w.WrittenDoc] = ()
) -> w.WriterSession:
    """A writer session on the run's budget; the 25% cap counts `written`."""
    config = w.WriterConfig(models=ctx.writers)
    pool = w.WriterPool(config.models, world_doc_total=len(w.writer_doc_plans(world_plan, config)))
    for doc in written:
        if doc.status == "written" and doc.writer_model in config.models:
            pool.assign(doc.writer_model)
    return w.WriterSession(
        backend=ctx.backend,
        pool=pool,
        budget=ctx.budget,
        config=config,
        call_log=call_log,
        now=ctx.now,
    )


def stage_write(
    spec: WorldSpec,
    world: World,
    world_plan: WorldDocPlan,
    pack: VerticalPack,
    wdir: Path,
    ctx: RunContext,
) -> StageResult:
    config = w.WriterConfig(models=ctx.writers)
    owned = w.writer_doc_plans(world_plan, config)
    session = _session(ctx, world_plan, wdir / "writer" / "calls.jsonl")
    doc_ids = _writer_ids(world_plan, config, ctx.max_docs)

    def progress(line: str) -> None:
        ctx.progress(f"[{spec.world_id}] write {line}")

    report = w.run_world(
        world,
        world_plan,
        pack,
        session=session,
        out_dir=wdir / "writer",
        doc_ids=doc_ids,
        progress=progress,
    )
    shares = report.get("writer_max_share_of_written") or 0.0
    detail = {
        "documents_owned": len(owned),
        "documents_selected": len(doc_ids) if doc_ids is not None else len(owned),
        "documents": report["documents"],
        "totals": report["totals"],
        "pass_rates": report["pass_rates"],
        "cost_usd": report["cost_usd"],
        "run_spend_usd": report["run_spend_usd"],
        "calls": report["calls"],
        "writer_docs": report["writer_docs"],
        "writer_max_share_of_written": shares,
        "writer_cap_documents": report["writer_cap_documents"],
        "prompt_version": report["prompt_version"],
        "checks_version": report["checks_version"],
        "failures_by_check_final": report["failures_by_check_final"],
    }
    if report["stop_reason"]:
        ctx.stop_reason = report["stop_reason"]
        return StageResult("write", "stopped", message=report["stop_reason"], detail=detail)
    status = "partial" if doc_ids is not None and len(doc_ids) < len(owned) else "done"
    message = (
        f"--max-docs: {len(doc_ids)} of {len(owned)} writer documents selected"
        if status == "partial" and doc_ids is not None
        else None
    )
    return StageResult("write", status, message=message, detail=detail)


def _written_docs(wdir: Path) -> list[w.WrittenDoc]:
    """The written documents of the latest write stage: those WP4's own
    `writer/run_report.json` lists (a checkpoint left from older code is on
    disk but not in that list)."""
    return list(wp7.load_stage_outputs(wdir / "writer").written.values())


def stage_completeness(
    spec: WorldSpec,
    world: World,
    world_plan: WorldDocPlan,
    pack: VerticalPack,
    wdir: Path,
    ctx: RunContext,
) -> StageResult:
    out_dir = wdir / "completeness"
    written = {d.plan_id: d for d in _written_docs(wdir)}
    if ctx.teacher_factory is None:
        raise RunnerError("the completeness stage needs a teacher (RunContext.teacher_factory)")
    # A fixture teacher is bound to the world's own documents (`DryRunTeacher`).
    bind = getattr(ctx.teacher_factory, "bind", None)
    teacher_factory = bind(written) if bind is not None else ctx.teacher_factory
    session = _session(ctx, world_plan, out_dir / "writer_calls.jsonl", list(written.values()))
    judge = None
    if ctx.judge_model is not None:
        budget = ctx.budget
        judge = wp5.LabelJudgeSettings(
            model=ctx.judge_model,
            call=ctx.judge_call,
            budget_left=lambda: budget.can_spend(JUDGE_WORST_CASE_USD),
            log_path=out_dir / "label_judge.jsonl",
        )
    cctx = wp5.CompletenessContext(
        world=world,
        world_plan=world_plan,
        backend_factory=teacher_factory,
        budget=ctx.budget,
        company_context=wp5.world_company_context(world),
        regenerator=wp5.WriterRegenerator(world, world_plan, pack, session, finished=written),
        call_log=out_dir / "teacher_calls.jsonl",
        client_meters=ctx.client_meters,
        label_judge=judge,
        now=ctx.now,
    )

    def progress(line: str) -> None:
        ctx.progress(f"[{spec.world_id}] completeness {line}")

    report = wp5.run_completeness(cctx, written, out_dir=out_dir, progress=progress)
    gates = {row["gate"]: row["result"] for row in report["gates"]}
    judge_summary = report.get("label_judge") or {}
    detail = {
        "documents": report["documents"],
        "spend": report["spend"],
        "silver": {k: report["silver"][k] for k in ("labels", "new_facts", "recovered_planted")},
        "teacher_recall_r": report["teacher"]["recall_r"],
        "residual_share": report["residual"]["share"],
        "label_judge": {
            "model": ctx.judge_model,
            "verdicts": judge_summary.get("verdicts"),
            "documents_not_fully_judged": judge_summary.get("documents_not_fully_judged"),
        },
        "gates": gates,
    }
    if report["stop_reason"]:
        ctx.stop_reason = report["stop_reason"]
        return StageResult("completeness", "stopped", message=report["stop_reason"], detail=detail)
    return StageResult("completeness", "done", detail=detail)


def stage_noise(
    spec: WorldSpec,
    world: World,
    world_plan: WorldDocPlan,
    pack: VerticalPack,
    wdir: Path,
    ctx: RunContext,
) -> StageResult:
    out_dir = wdir / "noise"
    stage = wp7.load_stage_outputs(wdir / "writer", wdir / "completeness")
    states = stage.states or {}
    config = wp6.NoiseConfig(seed=world.seed)
    session = _session(ctx, world_plan, out_dir / "calls.jsonl", list(stage.written.values()))
    near_report: dict[str, Any] = {}

    def near(finals: dict[str, w.WrittenDoc]) -> Any:
        result = write_near_duplicates(
            world,
            world_plan,
            pack,
            finals,
            session=session,
            config=NearDupConfig(seed=world.seed),
            out_dir=out_dir,
            progress=lambda line: ctx.progress(f"[{spec.world_id}] noise {line}"),
        )
        near_report.update(result.report)
        return result

    run = wp6.noise_completed_world(
        world,
        world_plan,
        stage.written,
        states,
        shipped=wp5.SHIPPED_STATUSES,
        config=config,
        near_duplicates=near,
    )
    wp6.write_outputs(run, out_dir)
    report = run.report
    detail = {
        "noise_version": report["noise_version"],
        "documents": len(run.docs),
        "documents_per_operation": report["documents_per_operation"],
        "rates": report["rates"],
        "labels": report["labels"],
        "drop_reasons": report["drop_reasons"],
        "near_duplicates": {
            k: near_report.get(k) for k in ("planned", "attempted", "by_status", "cost_usd")
        },
    }
    if near_report.get("stop_reason"):
        ctx.stop_reason = near_report["stop_reason"]
        return StageResult("noise", "stopped", message=near_report["stop_reason"], detail=detail)
    return StageResult("noise", "done", detail=detail)


def stage_export(
    spec: WorldSpec,
    world: World,
    world_plan: WorldDocPlan,
    pack: VerticalPack,
    wdir: Path,
    ctx: RunContext,
) -> StageResult:
    manifest = wp7.export_world(
        world,
        world_plan,
        split=spec.split,
        pack=spec.pack,
        seed=spec.seed,
        world_dir=wdir,
        held_out=pack.held_out,
        dry_run=ctx.dry_run,
        now=ctx.now,
    )
    training = manifest["training"]
    per_type = manifest["caps"]["per_type"]
    detail = {
        "documents": manifest["documents"]["exported"],
        "documents_noised": manifest["documents"]["noised"],
        "noise_not_applied": len(manifest["documents"]["noise_not_applied"]),
        "documents_excluded_by_reason": manifest["documents"]["excluded_by_reason"],
        "sources": manifest["counts"]["sources"],
        "labels": manifest["counts"]["labels"],
        "labels_by_label_source": manifest["counts"]["labels_by_label_source"],
        "decoys": manifest["counts"]["decoys"],
        "per_type_cap": {
            k: per_type[k]
            for k in ("applied", "max_share_before", "max_share_after", "labels_dropped", "met")
        }
        | {"documents_dropped": len(per_type["documents_dropped"])},
        "training_records": training.get("records") if training.get("written") else None,
        "sft_ready": training.get("sft_ready"),
        "training_ready": manifest["training_ready"],
        "training_ready_reasons": manifest["training_ready_reasons"],
    }
    return StageResult("export", "done", detail=detail)


def contamination_sources(wdir: Path) -> list[tuple[str, str, str]]:
    """(source_id, plan_id, text) for every source the export ships, CRM
    rows included: the text the builder gates."""
    return [
        (row["source_id"], row.get("plan_id") or "", row["text"])
        for row in _read_jsonl(wdir / "sources.jsonl")
    ]


def stage_contamination(wdir: Path, ctx: RunContext, prior: dict) -> StageResult:
    if ctx.refs is None:
        return StageResult(
            "contamination",
            "skipped",
            message="SKIPPED: --no-contamination-check was passed; nothing was checked",
        )
    sources = contamination_sources(wdir)
    refs_manifest = _load_json((ctx.refs_dir or Path()) / "manifest.json")
    digest = _sha(
        json.dumps(
            [refs_manifest.get("created"), sorted((s, _sha(t)) for s, _p, t in sources)],
            sort_keys=True,
        )
    )
    if prior.get("fingerprint") == digest and prior.get("status") in ("done", "resumed", "failed"):
        status = "failed" if prior["status"] == "failed" else "resumed"
        return StageResult(
            "contamination",
            status,
            message=prior.get("message"),
            detail=prior.get("detail", {}),
            fingerprint=digest,
        )
    rows = []
    by_reason: Counter = Counter()
    excluded: list[str] = []
    review: list[str] = []
    canary: list[str] = []
    for source_id, plan_id, text in sources:
        hits = contam.check_text(text, ctx.refs)
        if not hits:
            continue
        for h in hits:
            by_reason[h.reason] += 1
        if contam.is_contaminated(hits):
            excluded.append(source_id)
        else:
            review.append(source_id)
        if any(h.reason == "canary_string" for h in hits):
            canary.append(source_id)
        rows.append(
            {
                "source_id": source_id,
                "plan_id": plan_id,
                "excluded": contam.is_contaminated(hits),
                "hits": [dataclasses.asdict(h) for h in hits],
            }
        )
    _atomic_write(
        wdir / "contamination" / "contamination.jsonl",
        "".join(json.dumps(r, default=str) + "\n" for r in rows),
    )
    detail = {
        # The builder gates the final record text again.
        "checked_text": "exported sources (sources.jsonl)",
        "references": str(ctx.refs_dir) if ctx.refs_dir else None,
        "references_created": refs_manifest.get("created"),
        "sources_checked": len(sources),
        "excluded": len(excluded),
        "excluded_source_ids": excluded,
        "review_band": len(review),
        "review_source_ids": review,
        "hits_by_reason": dict(by_reason),
        "canary_source_ids": canary,
    }
    _atomic_write(wdir / "contamination" / "report.json", json.dumps(detail, indent=2))
    if canary:
        return StageResult(
            "contamination",
            "failed",
            message=f"ERB canary string in {len(canary)} source(s): {canary[:5]}",
            detail=detail,
            fingerprint=digest,
        )
    return StageResult("contamination", "done", detail=detail, fingerprint=digest)


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def _log_cost(path: Path) -> tuple[float, int]:
    """Cost and count of the calls in one call log; a call reused from an
    earlier run (`reused`) is not new spend."""
    cost, n = 0.0, 0
    for row in _read_jsonl(path):
        if row.get("reused"):
            continue
        cost += float(row.get("cost_usd") or 0.0)
        n += 1
    return cost, n


def review_sample(spec: WorldSpec, wdir: Path, per_kind: int = 3) -> dict[str, list[dict]]:
    """Up to `per_kind` exported documents per kind, in a seeded order, for
    the pilot's hand review (spec Build notes, WP11 pilot checklist), with
    the files and label counts as they ship."""
    sources = _read_jsonl(wdir / "sources.jsonl")
    labels = Counter(x["doc_id"] for x in _read_jsonl(wdir / "labels.jsonl"))
    decoys = Counter(x["doc_id"] for x in _read_jsonl(wdir / "decoys.jsonl"))
    by_plan: dict[str, list[dict]] = {}
    for row in sources:
        if row.get("kind") == "crm_account":
            continue
        by_plan.setdefault(row["plan_id"], []).append(row)
    by_kind: dict[str, list[str]] = {}
    for plan_id, rows in by_plan.items():
        by_kind.setdefault(rows[0]["kind"], []).append(plan_id)
    out: dict[str, list[dict]] = {}
    for kind, plan_ids in sorted(by_kind.items()):
        plan_ids.sort(key=lambda p: _sha(f"{spec.world_id}:{p}"))
        out[kind] = [
            {
                "plan_id": p,
                "files": sorted({f"files/{r['path']}" for r in by_plan[p]}),
                "writer_model": by_plan[p][0].get("writer_model"),
                "role": by_plan[p][0].get("role"),
                "labels": sum(labels[r["source_id"]] for r in by_plan[p]),
                "decoys": sum(decoys[r["source_id"]] for r in by_plan[p]),
                "noise": sorted({str(op.get("type")) for r in by_plan[p] for op in r["noise"]}),
            }
            for p in plan_ids[:per_kind]
        ]
    return out


def build_world_report(
    spec: WorldSpec, wdir: Path, results: list[StageResult], ctx: RunContext
) -> dict[str, Any]:
    by_name = {r.name: r for r in results}
    warnings = [f"{r.name}: {r.message}" for r in results if r.status == "skipped" and r.message]
    for r in results:
        if r.status in ("stopped", "failed", "partial") and r.message:
            warnings.append(f"{r.name} {r.status}: {r.message}")
    if ctx.dry_run:
        warnings.insert(
            0, "DRY RUN: fixture writer and teacher text (test tokens); never training data"
        )
    if spec.strictness == "test":
        warnings.append(TEST_STRICTNESS_NOT_BUILT)
    if ctx.judge_model is None:
        warnings.append("completeness: the label judge was off (--label-judge none)")
    if ctx.name_guard is None:
        warnings.append("simulate: no name guard; generated names were not checked")
    write = by_name.get("write")
    share = float(write.detail.get("writer_max_share_of_written", 0.0)) if write else 0.0
    completeness = by_name.get("completeness")
    qa = _load_json(wdir / "completeness" / "qa_report.json")
    gates = {
        "writer_max_share_of_written": {
            "value": share,
            "max": WRITER_SHARE_CAP,
            "pass": share <= WRITER_SHARE_CAP,
        },
        "section_3_5": {row["gate"]: row["result"] for row in qa.get("gates") or []}
        or "not computed: the completeness stage did not finish",
    }
    if share > WRITER_SHARE_CAP:
        warnings.append(
            f"writer share gate: one writer wrote {share:.0%} of this world's written documents "
            f"(spec cap {WRITER_SHARE_CAP:.0%})"
        )
    export = by_name.get("export")
    if export is not None and export.status == "done" and not export.detail["training_ready"]:
        warnings.append(
            "export: not training-ready: " + "; ".join(export.detail["training_ready_reasons"])
        )
    blocking = any(r.status in ("stopped", "failed") for r in results)
    written = _written_docs(wdir)
    writer_cost, writer_calls = _log_cost(wdir / "writer" / "calls.jsonl")
    teacher_cost, teacher_calls = _log_cost(wdir / "completeness" / "teacher_calls.jsonl")
    rewrite_cost, _ = _log_cost(wdir / "completeness" / "writer_calls.jsonl")
    judge_cost, judge_calls = _log_cost(wdir / "completeness" / "label_judge.jsonl")
    near_cost, near_calls = _log_cost(wdir / "noise" / "calls.jsonl")
    total = writer_cost + teacher_cost + rewrite_cost + judge_cost + near_cost
    exported = _read_jsonl(wdir / "labels.jsonl")
    by_source = Counter(str(x.get("label_source")) for x in exported)
    document_labels = sum(n for s, n in by_source.items() if s != "scripted_code")
    finished_cost = float(write.detail.get("cost_usd", 0.0)) if write else 0.0
    plan = by_name.get("plan")
    return {
        "runner_version": RUNNER_VERSION,
        "generated_at": ctx.now().isoformat(),
        "world_id": spec.world_id,
        "pack": spec.pack,
        "split": spec.split.value,
        "seed": spec.seed,
        "months": spec.months,
        "strictness": spec.strictness,
        "dry_run": ctx.dry_run,
        "complete": not blocking and all(r.status != "partial" for r in results),
        "stubbed_stages": [],
        "warnings": warnings,
        "stages": [r.as_json() for r in results],
        "writers": list(ctx.writers),
        "teacher": wp5.TEACHER_MODEL,
        "label_judge": ctx.judge_model,
        "labels": {
            "scripted_writer": sum(len(d.labels) for d in written),
            "crm_code": int(plan.detail.get("crm_code_labels", 0)) if plan else 0,
            "silver_completeness": by_source.get("silver_completeness", 0)
            if completeness is not None
            else None,
            "exported": len(exported) if export is not None and export.status == "done" else None,
            "exported_by_label_source": dict(by_source),
        },
        "cost_usd": {
            "write_finished_documents": round(finished_cost, 6),
            "write_all_calls_in_log": round(writer_cost, 6),
            "write_calls_in_log": writer_calls,
            "completeness_teacher_and_audit": round(teacher_cost, 6),
            "completeness_teacher_calls": teacher_calls,
            "completeness_rewrites": round(rewrite_cost, 6),
            "label_judge": round(judge_cost, 6),
            "label_judge_calls": judge_calls,
            "noise_near_duplicates": round(near_cost, 6),
            "noise_calls": near_calls,
            "total_known": round(total, 6),
            "per_labeled_fact": round(total / document_labels, 6) if document_labels else None,
        },
        "gates": gates,
        "review_sample": review_sample(spec, wdir),
    }


# --------------------------------------------------------------------------
# Driving worlds
# --------------------------------------------------------------------------


def _save_stages(wdir: Path, results: list[StageResult], ctx: RunContext) -> None:
    state = _load_json(wdir / "stages.json")
    for r in results:
        state[r.name] = {**r.as_json(), "finished_at": ctx.now().isoformat()}
    _atomic_write(wdir / "stages.json", json.dumps(state, indent=2, default=str))


def run_one_world(spec: WorldSpec, ctx: RunContext) -> dict[str, Any]:
    wdir = world_dir(ctx.out_root, spec)
    assert_not_holdout(wdir)
    wdir.mkdir(parents=True, exist_ok=True)
    prior = _load_json(wdir / "stages.json")
    results: list[StageResult] = []

    def timed(fn: Callable[[], Any]) -> Any:
        start = time.monotonic()
        out = fn()
        result = out[1] if isinstance(out, tuple) else out
        result.seconds = time.monotonic() - start
        results.append(result)
        _save_stages(wdir, results, ctx)
        ctx.progress(
            f"[{spec.world_id}] {result.name}: {result.status}"
            + (f" -- {result.message}" if result.message else "")
        )
        return out

    pack = load_world_pack(spec)
    world, _ = timed(
        lambda: stage_simulate(spec, pack, wdir, prior.get("simulate", {}), ctx.name_guard)
    )
    world_plan, _ = timed(lambda: stage_plan(world, pack, wdir, prior.get("plan", {})))
    halted = timed(lambda: stage_render(world, world_plan, pack)).status == "failed"
    paid_stages = (
        ("write", stage_write),
        ("completeness", stage_completeness),
        ("noise", stage_noise),
        ("export", stage_export),
    )
    for name, stage in paid_stages:
        if halted:
            results.append(
                StageResult(name, "skipped", message="SKIPPED: an earlier stage stopped or failed")
            )
            continue
        result = timed(
            lambda stage=stage: stage(spec, world, world_plan, pack, wdir, ctx)  # type: ignore[operator]
        )
        halted = result.status in ("stopped", "failed")
    if halted:
        results.append(
            StageResult(
                "contamination", "skipped", message="SKIPPED: an earlier stage stopped or failed"
            )
        )
    else:
        timed(lambda: stage_contamination(wdir, ctx, prior.get("contamination", {})))
    report = build_world_report(spec, wdir, results, ctx)
    results.append(StageResult("report", "done"))
    report["stages"].append(results[-1].as_json())
    _save_stages(wdir, results, ctx)
    _atomic_write(wdir / "run_report.json", json.dumps(report, indent=2, default=str))
    return report


def run_worlds(specs: list[WorldSpec], ctx: RunContext) -> dict[str, Any]:
    started = ctx.now()
    spent_before = ctx.budget.run_spent_usd()
    worlds: list[dict[str, Any]] = []
    for spec in specs:
        if ctx.stop_reason:
            worlds.append(
                {"world_id": spec.world_id, "split": spec.split.value, "status": "not started"}
            )
            continue
        report = run_one_world(spec, ctx)
        failed = any(s["status"] == "failed" for s in report["stages"])
        export = next((s for s in report["stages"] if s["name"] == "export"), None)
        worlds.append(
            {
                "world_id": spec.world_id,
                "split": spec.split.value,
                "status": "failed" if failed else ("stopped" if ctx.stop_reason else "ran"),
                "complete": report["complete"],
                "training_ready": bool(
                    export and export["status"] == "done" and export["detail"]["training_ready"]
                ),
                "labels_exported": report["labels"]["exported"],
                "cost_usd": report["cost_usd"]["total_known"],
                "report": str(world_dir(ctx.out_root, spec) / "run_report.json"),
            }
        )
    round_report = {
        "runner_version": RUNNER_VERSION,
        "started_at": started.isoformat(),
        "finished_at": ctx.now().isoformat(),
        "dry_run": ctx.dry_run,
        "writers": list(ctx.writers),
        "label_judge": ctx.judge_model,
        "budget": {
            "cap_usd": ctx.budget.cap_usd,
            "line_cap_usd": ctx.budget.corpus_cap_usd,
            "baseline_usd": ctx.budget.baseline_usd,
            "extra_baselines_usd": ctx.budget.extra_lines,
            "spent_by_root_usd": round(ctx.budget.run_spent_usd(), 6),
            "spent_this_invocation_usd": round(ctx.budget.run_spent_usd() - spent_before, 6),
        },
        "stop_reason": ctx.stop_reason,
        "worlds": worlds,
    }
    _atomic_write(
        ctx.out_root / "round_report.json", json.dumps(round_report, indent=2, default=str)
    )
    return round_report


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _pipeline_fakes() -> Any:
    """`tests/v2_pipeline_fakes.py`, loaded by path for --dry-run."""
    name = "_synth_v2_fixture_pipeline"
    if name in sys.modules:
        return sys.modules[name]
    if not FIXTURE_PIPELINE.exists():
        raise RunnerError(f"--dry-run needs {FIXTURE_PIPELINE}, which is not in this checkout")
    module_spec = importlib.util.spec_from_file_location(name, FIXTURE_PIPELINE)
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[name] = module  # its dataclasses look the module up here
    module_spec.loader.exec_module(module)
    return module


def fixture_backend() -> w.WriterBackend:
    """The WP4 tests' obedient stub writer, which also answers WP6's
    near-duplicate prompt."""
    fakes = _pipeline_fakes()
    return fakes.ScriptedBackend(reply=fakes.dry_run_reply, cost=0.0)


class DryRunTeacher:
    """The fixture teacher (`LabelEchoTeacher`), bound per world to that
    world's WP4 documents by the completeness stage."""

    def __init__(self) -> None:
        self.fakes = _pipeline_fakes()

    def bind(self, written: dict[str, w.WrittenDoc]) -> Callable[[str], Any]:
        return self.fakes.LabelEchoTeacher(dict(written)).factory

    def __call__(self, model: str) -> Any:
        raise RunnerError("DryRunTeacher is bound to a world's documents first (bind)")


def paid_backend(writers: tuple[str, ...], meter: SpendMeter) -> w.WriterBackend:
    routing = getattr(w, "RoutingWriterBackend", None)
    local = getattr(w, "LocalWriterBackend", None)
    if routing is not None and local is not None:
        return routing(local(meter=meter))
    return w.OpenRouterWriterBackend()


def _api_key(env_root: Path) -> str:
    key = os.environ.get("OPENROUTER_API_KEY") or prose._env_or_dotenv(
        env_root, "OPENROUTER_API_KEY"
    )
    if not key:
        raise RunnerError(f"OPENROUTER_API_KEY is not set (environment or {env_root}/.env)")
    os.environ["OPENROUTER_API_KEY"] = key  # the OpenRouter writer backend reads it here
    return key


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="scripts/data/synth_v2.py",
        description="Run v2 worlds from simulation to an exported, checked world.",
    )
    p.add_argument("--worlds", default="", help="N, round1, pilot, or pack:split:seed[,...]")
    p.add_argument("--pack", default="pharmacy_services")
    p.add_argument("--split", default="train", choices=[s.value for s in Split])
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--months", type=int, default=8)
    p.add_argument("--strictness", choices=("train", "test"), default=None)
    p.add_argument(
        "--writers",
        default=",".join(w.DEFAULT_ROTATION),
        help="comma-separated writer ids; local:<repo id> for local MLX models",
    )
    p.add_argument(
        "--label-judge",
        default=DEFAULT_LABEL_JUDGE_MODEL,
        metavar="MODEL",
        help="judge WP5's silver labels with this allowlisted open-weight model; 'none' "
        f"turns it off (default {DEFAULT_LABEL_JUDGE_MODEL})",
    )
    p.add_argument("--budget-usd", type=float, default=None, help="hard cap for this output root")
    p.add_argument(
        "--line-cap-usd", type=float, default=None, help="cap on the synthetic_v2 line total"
    )
    p.add_argument("--reset-budget-baseline", action="store_true")
    p.add_argument("--spend-path", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument(
        "--max-docs", type=int, default=None, help="write only the first N documents per world"
    )
    p.add_argument("--contamination-dir", type=Path, default=DEFAULT_CONTAMINATION_DIR)
    p.add_argument("--no-contamination-check", action="store_true")
    p.add_argument(
        "--no-name-guard",
        action="store_true",
        help="do not redraw generated names the contamination name blocklist would block",
    )
    p.add_argument(
        "--env-root",
        type=Path,
        default=REPO_ROOT,
        help="checkout whose .env holds OPENROUTER_API_KEY when the environment does not",
    )
    p.add_argument("--dry-run", action="store_true", help="fixture writer and teacher, $0")
    p.add_argument(
        "--check", action="store_true", help="preflight only: world sizes and writer coverage"
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        specs = parse_worlds(
            args.worlds,
            pack=args.pack,
            split=args.split,
            seed=args.seed,
            months=args.months,
            strictness=args.strictness,
        )
        writers = tuple(m.strip() for m in args.writers.split(",") if m.strip())
        check_writers(writers)
        judge_model = None if args.label_judge.lower() == "none" else args.label_judge
        if judge_model is not None and not judge_model_allowed(judge_model):
            raise RunnerError(f"--label-judge {judge_model} is not on the teacher allowlist")
        packs = {s.pack: load_world_pack(s) for s in specs}
        out_root = args.out or (DEFAULT_DRY_RUN_OUT if args.dry_run else DEFAULT_OUT)
        check_output_root(out_root, dry_run=args.dry_run)
        guard = None if args.no_name_guard else NameGuard.from_dir(args.contamination_dir)
        for spec in specs:
            world = simulate_world(
                packs[spec.pack],
                seed=spec.seed,
                months=spec.months,
                split=spec.split,
                name_guard=guard,
            )
            owned = len(
                w.writer_doc_plans(
                    plan_world(world, packs[spec.pack]), w.WriterConfig(models=writers)
                )
            )
            can, cap = writer_coverage(writers, owned)
            print(
                f"{spec.label()} ({spec.strictness}): {owned} writer documents, cap {cap} per writer"
            )
            if can < owned:
                raise RunnerError(
                    f"{spec.label()}: {len(set(writers))} writers x {cap} documents each cannot cover "
                    f"{owned} documents under the 25% cap; pass at least "
                    f"{math.ceil(owned / cap)} writers"
                )
        if args.check:
            return 0
        if args.budget_usd is None and not args.dry_run:
            raise RunnerError("--budget-usd is required for a real run")
        if guard is None and not args.no_name_guard and not args.dry_run:
            raise RunnerError(
                f"no name blocklist at {args.contamination_dir}; build it "
                "(scripts/data/build_contamination_signatures.py) or pass --no-name-guard"
            )
        if args.no_contamination_check:
            refs, refs_dir = None, None
        else:
            refs_dir = args.contamination_dir
            refs = contam.load_references_or_refuse(refs_dir, allow_missing=False)
    except (RunnerError, ValueError, contam.ContaminationSignaturesMissing) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    out_root.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        (out_root / DRY_RUN_MARKER).write_text(
            "Fixture writer and teacher output (tests/v2_writer_fakes.py, "
            "tests/v2_pipeline_fakes.py). Never training data.\n",
            encoding="utf-8",
        )
    spend_path = args.spend_path or (
        out_root / "dry_run_spend.json" if args.dry_run else default_spend_path(REPO_ROOT / "var")
    )
    meter = SpendMeter(spend_path)
    try:
        budget = open_budget(
            out_root,
            meter,
            cap_usd=args.budget_usd if args.budget_usd is not None else 1.0,
            line_cap_usd=args.line_cap_usd,
            reset_baseline=args.reset_budget_baseline,
            now=_utc_now,
        )
        if args.dry_run:
            backend = fixture_backend()
        else:
            api_key = _api_key(args.env_root)
            backend = paid_backend(writers, meter)
    except RunnerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    ctx = RunContext(
        out_root=out_root,
        writers=writers,
        backend=backend,
        budget=budget,
        refs=refs,
        refs_dir=refs_dir,
        judge_model=judge_model,
        name_guard=guard,
        dry_run=args.dry_run,
        max_docs=args.max_docs,
        progress=lambda line: print(line, flush=True),
        now=_utc_now,
    )
    if args.dry_run:
        ctx.teacher_factory = DryRunTeacher()
        ctx.judge_call = _pipeline_fakes().keep_all_judge
    else:
        from portolan.training.label_judge import openrouter_judge_call

        ctx.teacher_factory = wp5.openrouter_teacher_factory(api_key, meter)
        ctx.client_meters = True
        ctx.judge_call = openrouter_judge_call(api_key, meter) if judge_model else None
    round_report = run_worlds(specs, ctx)
    print(json.dumps({k: round_report[k] for k in ("budget", "stop_reason", "worlds")}, indent=2))
    if round_report["stop_reason"]:
        return 4
    if any(x.get("status") == "failed" for x in round_report["worlds"]):
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
