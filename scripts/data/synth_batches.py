"""Batch runner for the synthetic-data training pipeline: runs
`corpus.generator.cli.synth` across N worlds (seeds), one after another,
under ONE SHARED free-tier rate-limit budget (`corpus.generator.prose.
TokenBucketLimiter`/`TeacherRotation` -- build instructions: "one shared
rate limiter across worlds", not a fresh 20/min-1,000/day budget reset for
every world), then runs `corpus.generator.training_export.export_batch` on
each finished world, and prints one summary line per world: documents,
planted, verified, exported records, and a per-fact_type verified-count
breakdown.

Each world gets its own output directory, `<out>/world-<seed>/` -- that
directory name IS the world's "batch id". Resumable by batch id: a world
whose `synth_report.json` already exists, did NOT stop early (no
`stopped_early: true`), and already has its training-export JSONL on disk
is skipped entirely (no new `synth()` call, no re-generation) -- its
existing report is just re-exported (idempotent, no network) and
summarized.

A world whose report says `stopped_early: true` -- for ANY reason -- is
NOT resumed and NOT skipped: its directory is renamed out of the way
(`<dir>.incomplete-<timestamp>`) and its exported JSONL deleted, then it
is regenerated from scratch on the next run (`_reset_incomplete_world`).
This is deliberately more conservative than `cli.synth()`'s own PARTIAL-
progress resume via `.synth_progress.json` (still used within ONE world
that hits a transient issue mid-build) -- a world this runner itself
marked incomplete always starts clean.

The SHARED daily free-tier budget is watched two ways, fixed 2026-09-21
after a run silently produced fifteen empty (`verified: 0`) worlds in a
row once the account-wide 1,000-requests/day cap was hit mid-run (`cli.
synth()` catches `prose.DailyLimitExceeded` internally and returns a
graceful `stopped_early` report, so it never propagates here to catch):
(1) after each world, its own report is inspected for a daily-budget
`stopped_early` -- if so, the run stops immediately (exit code 3) and the
next seed is never started; (2) before each world, an OpenRouter key
check (`GET /api/v1/key`, skippable with `--no-key-check`) stops the run
the same way when too few free requests plausibly remain for another
world. Both print one line and raise `DailyBudgetExhausted`.

Surviving a flaky free model (build instructions item 4) needs no extra
retry logic here: `cli._synth_generate_unit` already retries a
verification failure once on a DIFFERENT teacher, then falls back to
filler text and records the source in `failed_sources` -- this script
just surfaces that count in each world's summary line.

Usage:
    uv run python scripts/data/synth_batches.py --worlds 20 --accounts 5 \\
        --per-kind 60 --type-mix balanced --out var/data/synth/

    # Resume a partially-completed or budget-stopped run: same command,
    # same --out. Worlds already finished cleanly are skipped; any
    # `stopped_early` world is regenerated from scratch.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from corpus.generator import cli as cli_mod  # noqa: E402
from corpus.generator import prose, training_export, worlds  # noqa: E402
from portolan.training.spend import SpendMeter, default_spend_path  # noqa: E402

DEFAULT_OUT = REPO_ROOT / "var" / "data" / "synth"

# One world's LLM calls cost ~111 free-model requests (observed 2026-09-21,
# `--accounts 5 --per-kind 60`). 120 leaves a little headroom for the
# pre-world key check (build instructions item 3).
MIN_FREE_REQUESTS_REMAINING = 120


class DailyBudgetExhausted(RuntimeError):
    """Raised by `run_batches` -- and mapped to exit code 3 by `main()` --
    when either (a) a world's own `synth_report.json` shows the run
    stopped early on the SHARED free-tier daily request budget, or (b) the
    pre-world OpenRouter key check found too few free requests left to
    safely start another world. In both cases `run_batches` has already
    printed its one STOPPED/nearly-spent line and must not start the next
    world -- caught 2026-09-21: `corpus.generator.cli.synth` catches
    `prose.DailyLimitExceeded` INTERNALLY and returns a graceful
    `stopped_early` report instead of letting the exception reach this
    runner, so a plain `except prose.DailyLimitExceeded` around the
    `cli_mod.synth()` call below never fires -- the report itself must be
    inspected (see `_stopped_for_daily_budget`)."""


def batch_dir_for(out_root: Path, seed: int) -> Path:
    return out_root / f"world-{seed}"


def _report_path(batch_dir: Path) -> Path:
    return batch_dir / "synth_report.json"


def _load_report(batch_dir: Path) -> dict[str, Any] | None:
    path = _report_path(batch_dir)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def export_path_for(batch_dir: Path, export_dir: Path | None) -> Path:
    """Where this world's training-export JSONL goes: `<export_dir>/
    <batch_dir.name>.jsonl` when the caller gave one, else `training_
    export.default_export_path(batch_dir)` (the real `var/data/silver/
    synthetic/` tree). `export_dir` exists so a test can redirect BOTH the
    batch directory (via its own `out_root`) and the exported JSONL under
    one `tmp_path`, never touching the project's real `var/`."""
    if export_dir is not None:
        return export_dir / f"{batch_dir.name}.jsonl"
    return training_export.default_export_path(batch_dir)


def _already_done(batch_dir: Path, export_dir: Path | None) -> dict[str, Any] | None:
    """The existing `synth_report.json` if this world already finished
    cleanly and was already exported -- `None` if it must (re)run. In
    practice `report.get("stopped_early")` is never true here any more --
    `_reset_incomplete_world` (called on every seed before this) already
    moved any `stopped_early` world's directory out of the way -- this
    check stays as a defensive second line."""
    report = _load_report(batch_dir)
    if report is None or report.get("stopped_early"):
        return None
    if not export_path_for(batch_dir, export_dir).exists():
        return None
    return report


def _reset_incomplete_world(batch_dir: Path, export_dir: Path | None) -> None:
    """A world directory whose `synth_report.json` says `stopped_early:
    true` -- for ANY reason, not just the daily budget -- is redone from
    scratch on the next run, never partially reused: rename the directory
    out of the way (`<dir>.incomplete-<timestamp>`) and delete its
    already-exported training JSONL, so the next `cli_mod.synth()` call
    for this seed starts with a clean `out_dir` instead of silently
    resuming from the old `.synth_progress.json` sidecar left by the
    interrupted attempt. A world with a clean report, or no report at all
    yet, is left untouched."""
    report = _load_report(batch_dir)
    if report is None or not report.get("stopped_early"):
        return
    timestamp = _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    incomplete_dir = batch_dir.with_name(f"{batch_dir.name}.incomplete-{timestamp}")
    batch_dir.rename(incomplete_dir)
    export_path = export_path_for(batch_dir, export_dir)
    if export_path.exists():
        export_path.unlink()


def _stopped_for_daily_budget(report: dict[str, Any], requests_per_day: int) -> bool:
    """True when `report` (a just-finished world's own `synth_report.
    json`) shows the world ended early because the SHARED free-tier daily
    request budget ran out -- see `DailyBudgetExhausted`'s docstring for
    why this can't just be an `except prose.DailyLimitExceeded`.

    Primary signal: `stop_reason` names the daily budget, matching
    `TokenBucketLimiter.acquire`'s own message verbatim ("N requests/day
    budget used up"). Fallback, for a `stopped_early` report whose
    `stop_reason` text might read differently: `verified == 0` (nothing
    got planted) while the shared rotation's cumulative `total_requests`
    had already reached the day's budget."""
    if not report.get("stopped_early"):
        return False
    stop_reason = report.get("stop_reason") or ""
    if "requests/day budget" in stop_reason:
        return True
    return report.get("verified", -1) == 0 and report.get("total_requests", 0) >= requests_per_day


def _daily_budget_stop_line(seed: int, report: dict[str, Any]) -> str:
    verified = report.get("verified", 0)
    return (
        f"STOPPED (daily free-tier limit) after world {seed}: {verified} verified; "
        "rerun the same command after the cap resets"
    )


class PaidBudgetExhausted(RuntimeError):
    """Raised by `run_batches` -- mapped to exit code 4 by `main()` --
    when a world's own `synth_report.json` shows the run stopped early on
    the shared paid-teacher dollar budget (`--budget-usd`, default `cli.
    DEFAULT_PAID_BUDGET_USD`). Same "the report is inspected, not caught
    off the exception" rationale as `DailyBudgetExhausted` -- `cli.synth()`
    catches `prose.PaidBudgetExceeded` internally too."""


def _stopped_for_paid_budget(report: dict[str, Any]) -> bool:
    """True when `report` (a just-finished world's own `synth_report.
    json`) shows the world ended early because the shared paid-teacher
    dollar budget ran out -- `prose.PAID_BUDGET_STOP_MARKER` is the exact
    substring `prose.PaidBudgetExceeded`'s message always contains."""
    if not report.get("stopped_early"):
        return False
    return prose.PAID_BUDGET_STOP_MARKER in (report.get("stop_reason") or "")


def _paid_budget_stop_line(
    seed: int, report: dict[str, Any], budget_usd: float, spend_usd: float
) -> str:
    """`spend_usd` (fixed 2026-09-22, was `report`-less `${budget_usd:.2f}
    budget-usd ceiling` phrasing that never said what actually tripped it)
    is THIS run's own `SYNTH_CORPUS_KEY` ("synthetic") spend at the moment
    of the stop -- `SpendMeter.corpus_usd`, never the shared meter's
    `total_usd()` across every corpus -- so the printed line always says
    which figure was compared against `--budget-usd`."""
    verified = report.get("verified", 0)
    return (
        f"STOPPED (paid-teacher budget): synthetic spend ${spend_usd:.2f} of "
        f"${budget_usd:.2f} after world {seed}: {verified} verified; "
        "rerun with a larger --budget-usd to continue"
    )


def _free_requests_remaining(
    key_check: Callable[[], int | None] | None,
) -> tuple[int | None, bool]:
    """`(remaining, attempted)` for the pre-world key check (build
    instructions item 3). `attempted` is False only when there was
    nothing to check with -- no injected `key_check` (tests) and no
    `OPENROUTER_API_KEY` set -- the one case where staying silent is
    correct rather than logging an "endpoint failed" note; every other
    case (`key_check` given, or a real key present) means a check was
    actually attempted, so a `None` result means the endpoint itself
    failed and should be logged."""
    if key_check is not None:
        return key_check(), True
    api_key = prose._env_or_dotenv(REPO_ROOT, "OPENROUTER_API_KEY")
    if not api_key:
        return None, False
    return prose.free_model_daily_requests_remaining(api_key), True


def summary_line(
    seed: int,
    batch_dir: Path,
    report: dict[str, Any],
    exported: list[dict[str, Any]],
    *,
    resumed: bool,
    world_cost_usd: float | None = None,
) -> str:
    """`world_cost_usd` (added 2026-09-22, paid-teacher runs only): THIS
    world's own contribution to the run's synthetic-corpus spend -- the
    spend meter's `SYNTH_CORPUS_KEY` line, sampled right before and right
    after this world's `cli_mod.synth()` call. `None` (the default, and
    always the case for a free-teacher run or a resumed/already-done
    world, which makes no new calls) omits the field entirely."""
    per_type = report.get("verified_by_type", {})
    per_type_str = ", ".join(f"{t}={n}" for t, n in sorted(per_type.items())) or "(none)"
    planted = sum(report.get("planted_by_type", {}).values())
    line = (
        f"world seed={seed} [{batch_dir.name}]: "
        f"documents={report.get('total_sources', 0)} "
        f"planted={planted} "
        f"verified={report.get('verified', 0)} "
        f"failed={len(report.get('failed_sources', []))} "
        f"exported={len(exported)} "
        f"per_type=[{per_type_str}]"
    )
    if world_cost_usd is not None:
        line += f" synthetic_cost=${world_cost_usd:.4f}"
    if resumed:
        line += " [resumed: already done]"
    return line


def run_batches(
    *,
    worlds_count: int,
    num_accounts: int,
    per_kind: int,
    type_mix: dict[str, int] | None,
    out_root: Path,
    base_seed: int,
    requests_per_minute: int = 20,
    requests_per_day: int = 1000,
    client: Any = None,
    rotation: prose.TeacherRotation | None = None,
    limiter: prose.TokenBucketLimiter | None = None,
    export_dir: Path | None = None,
    no_key_check: bool = False,
    key_check: Callable[[], int | None] | None = None,
    key_check_threshold: int = MIN_FREE_REQUESTS_REMAINING,
    teachers: list[str] | None = None,
    budget_usd: float = cli_mod.DEFAULT_PAID_BUDGET_USD,
    spend: Any = None,
) -> list[str]:
    """Run `worlds_count` worlds (seeds `base_seed .. base_seed +
    worlds_count - 1`) in sequence, sharing ONE `rotation`/`limiter`/
    `client` across every world unless the caller already supplied its
    own (tests inject fakes this way -- see `tests/synth_fakes.py`).

    `export_dir`: where each world's training-export JSONL is written
    (`export_path_for` -- default `None` keeps the real `var/data/silver/
    synthetic/` tree, matching `cli.py synth`'s own default). A test MUST
    pass an explicit `export_dir` under its own `tmp_path` -- otherwise
    the exported JSONL leaks into the project's real training directory
    even when `out_root` (the batch directories) is already isolated.

    `teachers` (added 2026-09-22): exact OpenRouter ids shared across
    EVERY world in this run, bypassing free-model discovery. All-paid
    (`prose.teachers_are_paid`, `MixedTeacherListError` on a mixed list):
    overrides `requests_per_minute`/`requests_per_day` to 60/unlimited,
    forces `no_key_check = True` (a paid call has no free-tier daily
    budget for the key endpoint to check), and defaults `spend` to the
    shared `var/data/silver/spend.json` meter -- ONE meter for the WHOLE
    multi-world run, same "shared budget across worlds" rule the free
    path already follows. All-free: unchanged discovery-bypass behavior.

    `--budget-usd` is a ceiling on THIS run's own `cli_mod.SYNTH_CORPUS_
    KEY` ("synthetic") spend line in that meter, never the meter's shared
    `total_usd` across every corpus (fixed 2026-09-22 -- a concurrent
    silver-labeling run sharing the same `spend.json` file had pushed
    `total_usd` past a $5 ceiling the synthetic corpus itself was nowhere
    near, stopping a paid-teacher run immediately). See `SpendMeter.
    is_corpus_over_budget`/`corpus_usd`.

    Before EVERY seed: `_reset_incomplete_world` redoes any `stopped_
    early` world from scratch (build instructions item 2), then -- unless
    `no_key_check` -- a pre-world OpenRouter key check (item 3) stops the
    run when too few free requests remain today. `key_check` overrides
    the real `prose.free_model_daily_requests_remaining` call entirely
    (tests inject a fake here instead of mocking HTTP).

    Raises `DailyBudgetExhausted` the moment either the key check or a
    just-finished world's own report shows the shared daily free-tier
    budget is spent (item 1: never start the next world after that).
    Otherwise returns the printed summary lines (one per world, printed
    as each completes so a long run's progress is visible as it goes)."""
    out_root.mkdir(parents=True, exist_ok=True)
    if export_dir is not None:
        export_dir.mkdir(parents=True, exist_ok=True)

    # Paid teachers (added 2026-09-22): `--teachers` with every id lacking
    # ':free' bypasses discovery AND the free-tier daily budget/key check
    # below -- `paid_mode` gates both. A mixed free/paid list is an error
    # (`prose.teachers_are_paid` raises `MixedTeacherListError`); `main()`
    # already validates this before calling in, so it's re-raised here
    # only as a defensive check for a direct (non-CLI) caller.
    paid_mode = False
    if teachers is not None:
        paid_mode = prose.teachers_are_paid(list(teachers))
        if rotation is None:
            rotation = prose.TeacherRotation(list(teachers))
        if paid_mode:
            requests_per_minute = cli_mod.PAID_TEACHER_RPM
            requests_per_day = cli_mod.PAID_TEACHER_RPD
            no_key_check = True
            if spend is None:
                spend = SpendMeter(path=default_spend_path(REPO_ROOT / "var"))
            print(
                f"paid-teacher budget: synthetic spend "
                f"${spend.corpus_usd(cli_mod.SYNTH_CORPUS_KEY):.2f} of ${budget_usd:.2f} ceiling"
            )

    if rotation is None:
        api_key = prose._env_or_dotenv(REPO_ROOT, "OPENROUTER_API_KEY")
        all_free_models = prose.discover_free_models(api_key)
        if not all_free_models:
            raise RuntimeError(
                "synth_batches: no free (':free') OpenRouter models discovered -- check "
                "OPENROUTER_API_KEY and network access; refusing to fall back to a paid model."
            )
        rotation = prose.TeacherRotation(all_free_models[: prose.MAX_TEACHER_ROTATION_SIZE])
    if limiter is None:
        limiter = prose.TokenBucketLimiter(
            requests_per_minute=requests_per_minute, requests_per_day=requests_per_day
        )
    if client is None:
        client = prose.build_openrouter_client(REPO_ROOT)

    lines: list[str] = []
    key_check_note_printed = False
    for i in range(worlds_count):
        seed = base_seed + i
        this_batch_dir = batch_dir_for(out_root, seed)

        # Item 2: any world left over from a run that stopped early
        # (whatever the reason) is redone from scratch, never resumed.
        _reset_incomplete_world(this_batch_dir, export_dir)
        this_export_path = export_path_for(this_batch_dir, export_dir)

        existing = _already_done(this_batch_dir, export_dir)
        if existing is not None:
            exported = training_export.export_batch(this_batch_dir, this_export_path)
            line = summary_line(seed, this_batch_dir, existing, exported, resumed=True)
            print(line)
            lines.append(line)
            continue

        # Item 3: before spending any requests on this world, check
        # whether today's free-tier budget can plausibly cover it. A
        # failed/unavailable check (`attempted and remaining is None`) is
        # NOT a per-world event worth a summary line -- it is one fact
        # about the whole run ("the key endpoint isn't answering, or
        # `key_check` isn't wired up") that stays true every subsequent
        # world too. Printed at most once per run, straight to stdout,
        # and deliberately never added to `lines` -- callers (and this
        # module's own tests) treat `lines` as one entry per world.
        if not no_key_check:
            remaining, attempted = _free_requests_remaining(key_check)
            if attempted and remaining is None:
                if not key_check_note_printed:
                    print(
                        "key check: OpenRouter key endpoint failed or returned an "
                        "unexpected response; continuing without it"
                    )
                    key_check_note_printed = True
            elif remaining is not None and remaining < key_check_threshold:
                line = f"daily free-tier budget nearly spent ({remaining} left); rerun after reset"
                print(line)
                lines.append(line)
                raise DailyBudgetExhausted(line)

        # Paid teachers: sampled before this world's `cli.synth()` call so
        # the summary line below can show THIS world's own contribution to
        # the run's synthetic-corpus spend, not the cumulative total. The
        # ceiling itself is still enforced the same place it always was --
        # inside `cli.synth()`'s `_synth_generate_unit`, before every live
        # call, never here -- so an already-exhausted budget is caught
        # before the next real dollar is spent, whether that happens on
        # this world's very first call or partway through it.
        spend_before = (
            spend.corpus_usd(cli_mod.SYNTH_CORPUS_KEY) if paid_mode and spend is not None else None
        )
        try:
            report = cli_mod.synth(
                seed=seed,
                num_accounts=num_accounts,
                per_kind=per_kind,
                out_dir=this_batch_dir,
                requests_per_minute=requests_per_minute,
                requests_per_day=requests_per_day,
                type_mix=type_mix,
                client=client,
                rotation=rotation,
                limiter=limiter,
                spend=spend,
                budget_usd=budget_usd,
            )
        except prose.PaidBudgetExceeded as exc:
            # Defense in depth, same rationale as the `DailyLimitExceeded`
            # catch below: `cli.synth()` swallows this internally too and
            # returns a `stopped_early` report instead -- not expected to
            # fire, kept in case a future code path lets it through.
            line = _paid_budget_stop_line(
                seed, {"verified": 0}, budget_usd, spend.corpus_usd(cli_mod.SYNTH_CORPUS_KEY)
            )
            print(line)
            lines.append(line)
            raise PaidBudgetExhausted(line) from exc
        except prose.DailyLimitExceeded as exc:
            # Defense in depth: `cli.synth()` currently swallows this
            # internally and returns a `stopped_early` report instead (see
            # `DailyBudgetExhausted`'s docstring) -- this branch is not
            # expected to fire, kept in case a future code path lets the
            # exception through uncaught.
            line = _daily_budget_stop_line(seed, {"verified": 0})
            print(line)
            lines.append(line)
            raise DailyBudgetExhausted(line) from exc

        # Paid teachers: a world that itself stopped early because the
        # shared dollar budget ran out ends the whole run here, same
        # never-start-the-next-seed rule as the free daily budget below --
        # checked FIRST since a paid run never touches the free-tier
        # daily-budget signal `_stopped_for_daily_budget` looks for.
        if paid_mode and _stopped_for_paid_budget(report):
            line = _paid_budget_stop_line(
                seed, report, budget_usd, spend.corpus_usd(cli_mod.SYNTH_CORPUS_KEY)
            )
            print(line)
            lines.append(line)
            raise PaidBudgetExhausted(line)

        # Item 1: a world that itself stopped early on the shared daily
        # budget ends the whole run here -- the next seed is never started.
        if _stopped_for_daily_budget(report, requests_per_day):
            line = _daily_budget_stop_line(seed, report)
            print(line)
            lines.append(line)
            raise DailyBudgetExhausted(line)

        world_cost_usd = (
            round(spend.corpus_usd(cli_mod.SYNTH_CORPUS_KEY) - spend_before, 8)
            if spend_before is not None
            else None
        )
        exported = training_export.export_batch(this_batch_dir, this_export_path)
        line = summary_line(
            seed, this_batch_dir, report, exported, resumed=False, world_cost_usd=world_cost_usd
        )
        print(line)
        lines.append(line)

    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python scripts/data/synth_batches.py")
    parser.add_argument(
        "--worlds", type=int, required=True, help="Number of worlds (seeds) to run."
    )
    parser.add_argument("--accounts", type=int, default=5)
    parser.add_argument("--per-kind", type=int, default=60)
    parser.add_argument(
        "--type-mix",
        default=None,
        help="Same syntax as `cli.py synth --type-mix`: 'balanced', or 'type=count,...'.",
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument(
        "--export-dir",
        default=None,
        help="Where each world's training-export JSONL goes. Default: var/data/silver/synthetic/.",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=None,
        help="First world's seed; worlds use base_seed..base_seed+worlds-1. Default: today as YYYYMMDD.",
    )
    parser.add_argument("--rpm", type=int, default=20, help="Free-tier requests-per-minute budget.")
    parser.add_argument("--rpd", type=int, default=1000, help="Free-tier requests-per-day budget.")
    parser.add_argument(
        "--no-key-check",
        action="store_true",
        help=(
            "Skip the pre-world OpenRouter key check (GET /api/v1/key) that stops the run "
            "when too few free-tier requests remain today."
        ),
    )
    parser.add_argument(
        "--teachers",
        default=None,
        help=(
            "Comma-separated exact OpenRouter model ids, shared across every world, "
            "bypassing free-model discovery. Same all-paid/all-free rules as "
            "`cli.py synth --teachers`; mixing free and paid ids is an error."
        ),
    )
    parser.add_argument(
        "--budget-usd",
        type=float,
        default=cli_mod.DEFAULT_PAID_BUDGET_USD,
        help="Paid-teacher dollar ceiling for the WHOLE run (only meaningful with --teachers).",
    )
    args = parser.parse_args(argv)

    try:
        type_mix = worlds.parse_type_mix(args.type_mix, args.per_kind)
    except ValueError as exc:
        print(f"invalid --type-mix: {exc}")
        return 1

    teachers = [t.strip() for t in args.teachers.split(",") if t.strip()] if args.teachers else None
    if teachers is not None:
        try:
            prose.teachers_are_paid(teachers)
        except prose.MixedTeacherListError as exc:
            print(f"invalid --teachers: {exc}")
            return 1

    base_seed = (
        args.base_seed if args.base_seed is not None else int(_dt.date.today().strftime("%Y%m%d"))
    )

    try:
        run_batches(
            worlds_count=args.worlds,
            num_accounts=args.accounts,
            per_kind=args.per_kind,
            type_mix=type_mix,
            out_root=Path(args.out),
            base_seed=base_seed,
            requests_per_minute=args.rpm,
            requests_per_day=args.rpd,
            export_dir=Path(args.export_dir) if args.export_dir else None,
            no_key_check=args.no_key_check,
            teachers=teachers,
            budget_usd=args.budget_usd,
        )
    except PaidBudgetExhausted:
        # The stop line was already printed by `run_batches` itself.
        return 4
    except DailyBudgetExhausted:
        # The stop line was already printed by `run_batches` itself.
        return 3
    except RuntimeError as exc:
        print(str(exc))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
