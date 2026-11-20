"""`python -m portolan.bench extract --backend fake|agent_sdk|bedrock|mlx --model <id> \
    --corpus corpus/out --out var/bench/`
(`--backend mlx` also takes `--adapter-path <dir>` for a local LoRA student.)
`python -m portolan.bench graph --out var/bench/`
"""

from __future__ import annotations

import argparse
import datetime as dt
import inspect
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path

from portolan.bench.corpus_loader import DEFAULT_CORPUS_ROOT
from portolan.bench.extraction import (
    ExtractionScore,
    JudgeClient,
    rescore_full,
    rescore_unplanted_classifications,
    run_extraction,
    score_extraction,
)
from portolan.bench.graph import score_graph
from portolan.bench.provenance import provenance_kwargs
from portolan.bench.report import BenchReport, write
from portolan.bench.truth import DEFAULT_MANIFEST_PATH, DEFAULT_TRUTH_TABLE_PATH, Manifest, Truth
from portolan.extract.backends import Backend, FakeBackend
from portolan.extract.prompt import build_system_prompt
from portolan.llm.base import TEMPERATURE_ENV_VAR

# Bench-provenance wave, 2026-09-18 (01-spec/bench-provenance-2026-09-18.md):
# `build_system_prompt`'s own `variant` parameter default, read via
# `inspect` rather than hardcoded here -- so `--variant`'s own default
# (and the stamp it feeds into a written report) can never drift out of
# sync with the prompt module's actual default the way a copy-pasted
# string literal could.
_DEFAULT_PROMPT_VARIANT: str = inspect.signature(build_system_prompt).parameters["variant"].default


def _fake_backend(truth: Truth, source_ids: list[str] | None) -> Backend:
    # No network, no model: emits exactly what the truth table planted.
    # Useful as a wiring smoke test for this CLI and for local report-
    # rendering iteration without a model call.
    from portolan.bench.fixtures import perfect_results

    canned = perfect_results(truth)
    if source_ids is not None:
        canned = {sid: canned[sid] for sid in source_ids if sid in canned}
    return FakeBackend(canned)


def _live_backend(name: str, model: str | None) -> Backend:
    """Construct the `agent_sdk` or `bedrock` backend. Imported lazily,
    inside this function, because `portolan.llm` (wave A1) and
    `portolan.extract.backends.LLMClientBackend` (also wave A1, per
    build-contracts.md's "LLM access" section) may not exist yet in this
    checkout -- bench (wave B4) does not own either.

    `name` (the CLI's own `--backend` choice) is now passed straight
    through to `get_llm(provider=name)` (infra-2b wave, 2026-09-18 -- real
    bug found live: `get_llm()` called with no argument obeys ONLY
    `settings.llm_provider` (`LLM_PROVIDER` in `.env`/the environment),
    completely ignoring `name` -- so a checkout with `.env` set to
    `LLM_PROVIDER=openrouter` silently ran every `--backend agent_sdk`
    request through OpenRouter instead, with a bare `claude-sonnet-5`
    model id OpenRouter doesn't recognize as an Anthropic-prefixed id,
    producing unusable output at real OpenRouter cost -- $9.98 across
    three runs before anyone noticed). `--backend`'s own choices
    (`agent_sdk`, `bedrock`, `anthropic`, `openrouter` -- `fake` is
    handled by `_fake_backend`, never reaches this function) are already
    exactly `Settings.llm_provider`'s literal values, so no mapping is
    needed.
    """
    try:
        from portolan.extract.backends import LLMClientBackend
        from portolan.llm import get_llm
    except ImportError as exc:
        raise SystemExit(
            f"backend {name!r} needs portolan.llm.get_llm and "
            "portolan.extract.backends.LLMClientBackend, which aren't available "
            f"in this checkout yet ({exc}). See build-contracts.md 'LLM access' "
            "(wave A1) -- bench does not own either module."
        ) from exc

    llm = get_llm(provider=name)

    # Fix instructions item 4 ("choose the JSON strategy up front rather
    # than by trial, caching per model id for the process"): if `llm` is
    # an `OpenRouterClient` (the only client with this method, as of this
    # wave), prime its process-wide JSON-strategy cache from OpenRouter's
    # live `/models/{id}/endpoints` listing before the first source is
    # ever extracted, instead of discovering it (or mis-attempting into
    # discovering it) reactively on that first call. Every other backend
    # has no `warm_json_strategy` attribute and is unaffected.
    # `run_extraction`'s `concurrency > 1` path builds one fresh backend
    # per worker thread via `_live_backend_factory` below -- each of those
    # calls this again, but `prime_json_strategy` is a cheap no-op past
    # the first (module-level cache in `portolan.llm.openrouter`), so
    # every worker thread shares the one answer this call already primed.
    warm_json_strategy = getattr(llm, "warm_json_strategy", None)
    if warm_json_strategy is not None:
        warm_json_strategy(model)

    # F17: LLMClientBackend takes `model` directly now; the earlier
    # `except TypeError` fallback (for a constructor that hadn't grown the
    # `model` parameter yet) is gone.
    return LLMClientBackend(llm, model=model)


def _mlx_backend(model: str | None, adapter_path: str | None, max_tokens: int | None) -> Backend:
    """`--backend mlx`: a local student model run in-process by `mlx-lm`
    (`portolan.llm.mlx_local.MLXClient`, fine-tune shakedown 2026-09-22),
    wrapped in the same `LLMClientBackend` every hosted provider uses, so
    its output goes through the same strict/retry/`parse_lenient` chain.
    Needs `uv sync --group mlx` on Apple Silicon; no network, no key."""
    from portolan.extract.backends import LLMClientBackend
    from portolan.llm.mlx_local import MLXClient, mlx_available

    if not model:
        raise SystemExit("--backend mlx needs --model (a local path or Hugging Face repo id)")
    if not mlx_available():
        raise SystemExit("--backend mlx needs mlx and mlx-lm: run `uv sync --group mlx`")
    client = MLXClient(model, adapter_path=adapter_path, max_tokens=max_tokens, progress=True)
    return LLMClientBackend(client, model=model)


def _parser_counts(run_result: object) -> dict[str, object]:
    """What `ExtractionResult.parse_lenient` did across a run, which the
    scorecard itself does not carry: facts it dropped (`rejected`), fields
    it defaulted or coerced (`coercions`, by field), and sources that
    produced no usable JSON at all (`failed_sources`). Plus model-call
    counts from the run's own sink (calls, and retries where the client
    labels them -- `MLXClient` does)."""
    results = dict(run_result)  # type: ignore[call-overload]
    coercions_by_field: dict[str, int] = {}
    for result in results.values():
        for coercion in result.coercions:
            coercions_by_field[coercion.field] = coercions_by_field.get(coercion.field, 0) + 1
    events = [
        e for e in getattr(getattr(run_result, "sink", None), "events", []) if e.input_tokens > 0
    ]
    retries = sum(1 for e in events if (e.extra.get("mlx") or {}).get("retry"))
    finish: dict[str, int] = {}
    for e in events:
        reason = (e.extra.get("mlx") or {}).get("finish_reason")
        if reason:
            finish[reason] = finish.get(reason, 0) + 1
    return {
        "sources_with_result": len(results),
        "failed_sources": dict(getattr(run_result, "failed_sources", {})),
        "rejected_facts": sum(len(r.rejected) for r in results.values()),
        "coercions_total": sum(coercions_by_field.values()),
        "coercions_by_field": coercions_by_field,
        "model_calls": len(events),
        "retry_calls": retries,
        "finish_reasons": finish,
    }


def _live_backend_factory(name: str, model: str | None) -> Callable[[], Backend]:
    """`--concurrency N > 1` needs one backend (and, for `agent_sdk`, one
    underlying `AgentSDKClient` -- one `claude_agent_sdk.query()` chain) per
    worker thread, never one shared across threads. Returns a zero-arg
    callable that builds a brand new one on every call; `run_extraction`
    calls it exactly once per worker thread and keeps the result for that
    thread's remaining sources."""
    return lambda: _live_backend(name, model)


def _judge_client(name: str, model: str | None) -> JudgeClient:
    """`--judge`: off by default (`01-spec/quality-loop.md`, "a judge call
    grades only the unmatched remainder"). Reuses whatever `portolan.llm.
    get_llm(provider=name)` resolves to -- an `LLMClient` already satisfies
    `JudgeClient` structurally (both are just `.complete(...)`), so this is
    `_live_backend` minus the `LLMClientBackend` wrapping.

    `name` is passed through to `get_llm(provider=name)` for the same
    reason `_live_backend` does (infra-2b wave, 2026-09-18 -- see that
    function's docstring): without it, the judge call silently rode
    `settings.llm_provider` regardless of `--backend`. `model` stays
    ignored/deleted -- unchanged, deliberate existing behavior (the judge
    uses the resolved provider's own default-configured model, per this
    function's own docstring; no bug report named this half)."""
    try:
        from portolan.llm import get_llm
    except ImportError as exc:
        raise SystemExit(
            f"--judge needs portolan.llm.get_llm, which isn't available in "
            f"this checkout yet ({exc})."
        ) from exc
    del model  # the judge always uses get_llm()'s default-configured model
    return get_llm(provider=name)


def _cmd_rescore(args: argparse.Namespace) -> int:
    """`--rescore <saved-report.json>`: no backend, no model call. Re-runs
    ONLY the invented/`evidence_basis` classification of an already-saved
    bench report against the corpus on disk (`rescore_unplanted_
    classifications`), carrying every other field -- recall, precision,
    matched, unmatched, tokens, cost, wall time -- forward unchanged, then
    writes a normal report (`.md`/`.json` twin) the same way a live run
    does. 2026-09-18 fix instructions item 4: "check python -m
    portolan.bench extract --help for a re-score-from-saved-output mode
    and add one if it does not exist."
    """
    saved_path = Path(args.rescore)
    saved = json.loads(saved_path.read_text(encoding="utf-8"))
    extraction_data = saved.get("extraction")
    if extraction_data is None:
        raise SystemExit(f"{saved_path} has no 'extraction' section to rescore")

    manifest = Manifest.load(args.manifest)
    # 2026-09-18 sanity audit: load `Truth` too so the fragment fallback
    # gets each source's `EvidenceContext` (registry_context/vocabulary/
    # source_date), same as `--rescore-dir`'s `rescore_full` already does
    # via `score_extraction`.
    truth = Truth.for_manifest(manifest, truth_table_path=args.truth_table)
    score = ExtractionScore.model_validate(extraction_data)
    rescored = rescore_unplanted_classifications(
        score, manifest, corpus_root=args.corpus, truth=truth
    )

    backend_label = f"{saved.get('backend', 'unknown')}-rescored"
    # The original run's own model/variant didn't change -- only the
    # SCORER did -- so carry them forward from the saved report's own
    # stamp (present on any report saved after the bench-provenance wave;
    # `.get` on an older report with no such field is simply `None`).
    report = BenchReport(
        backend=backend_label,
        date=dt.date.today(),
        extraction=rescored,
        **provenance_kwargs(
            corpus=args.corpus,
            truth_table=args.truth_table,
            prompt_variant=saved.get("prompt_variant"),
            extractor_model=saved.get("extractor_model"),
        ),
    )
    md_path, json_path = write(report, args.out)

    print(f"rescored {saved_path} against corpus at {args.corpus!r} (no backend call)")
    print(f"wrote {md_path}")
    print(f"wrote {json_path}")
    old_invented = sum(
        1 for u in score.unplanted if not u.is_distractor_leak and not u.in_source_text
    )
    new_invented = sum(
        1 for u in rescored.unplanted if not u.is_distractor_leak and not u.in_source_text
    )
    print(f"invented: {old_invented} (as saved) -> {new_invented} (corrected)")
    return 0


def _cmd_rescore_dir(args: argparse.Namespace) -> int:
    """`--rescore-dir <dir>`: full re-score (`portolan.bench.extraction.
    rescore_full`) of every saved bench report found under `<dir>`
    (recursively, `bench-*.json`, skipping any prior `*-rescored.json`
    output so re-running this command is idempotent). No backend, no model
    call.

    Unlike `--rescore <file>` (which touches ONLY the invented/evidence_
    basis classification and leaves recall untouched by design), this
    rebuilds a `Fact` per saved produced record and re-runs the ENTIRE
    match against the CURRENT matcher code -- so a locator/matcher fix
    (2026-09-18: the RSS locator format bug in `portolan.bench.locators`)
    changes recall on a re-score of old saved output, which is the whole
    point of this mode: `01-spec/harness-review-2026-09-18.md`'s
    before/after table is built from exactly this. Writes one rescored
    report per input (mirroring its subpath under `--out`) plus
    `<out>/rescore-summary.json`, one row per input, so a write-up can read
    the before/after numbers without re-deriving anything.
    """
    src_dir = Path(args.rescore_dir)
    manifest = Manifest.load(args.manifest)
    truth = Truth.for_manifest(manifest, truth_table_path=args.truth_table)
    out_root = Path(args.out)

    found = sorted(p for p in src_dir.rglob("bench-*.json") if not p.stem.endswith("-rescored"))
    if not found:
        raise SystemExit(f"no bench-*.json files found under {src_dir}")

    rows: list[dict[str, object]] = []
    for saved_path in found:
        saved = json.loads(saved_path.read_text(encoding="utf-8"))
        extraction_data = saved.get("extraction")
        if extraction_data is None:
            print(f"skip {saved_path} (no 'extraction' section)")
            continue
        score = ExtractionScore.model_validate(extraction_data)
        rescored = rescore_full(score, manifest, truth, corpus_root=args.corpus)

        rel = saved_path.relative_to(src_dir)
        dest_dir = out_root / rel.parent
        backend_label = f"{saved.get('backend', 'unknown')}-rescored"
        report = BenchReport(
            backend=backend_label,
            date=dt.date.today(),
            extraction=rescored,
            **provenance_kwargs(
                corpus=args.corpus,
                truth_table=args.truth_table,
                prompt_variant=saved.get("prompt_variant"),
                extractor_model=saved.get("extractor_model"),
            ),
        )
        md_path, json_path = write(report, dest_dir)

        def _invented(sc: ExtractionScore) -> int:
            return sum(1 for u in sc.unplanted if not u.is_distractor_leak and not u.in_source_text)

        row = {
            "source": str(saved_path),
            "rescored_md": str(md_path),
            "rescored_json": str(json_path),
            "recall_before": score.recall,
            "recall_after": rescored.recall,
            "lenient_recall_after": rescored.lenient_recall,
            "near_miss_credited_count": rescored.near_miss_credited_count,
            "near_miss_credited_fact_ids": rescored.near_miss_credited_fact_ids,
            "invented_before": _invented(score),
            "invented_after": _invented(rescored),
        }
        rows.append(row)
        print(
            f"{saved_path}: recall {row['recall_before']:.3f} -> {row['recall_after']:.3f} "
            f"(lenient {row['lenient_recall_after']:.3f}, "
            f"near_miss_credited={row['near_miss_credited_count']}) "
            f"invented {row['invented_before']} -> {row['invented_after']}"
        )

    out_root.mkdir(parents=True, exist_ok=True)
    summary_path = out_root / "rescore-summary.json"
    summary_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {summary_path}")
    return 0


def _cmd_extract(args: argparse.Namespace) -> int:
    if getattr(args, "world", None):
        from portolan.bench.world_cli import cmd_extract_world

        return cmd_extract_world(args)
    if args.rescore_dir:
        return _cmd_rescore_dir(args)
    if args.rescore:
        return _cmd_rescore(args)
    if args.backend is None:
        raise SystemExit("--backend is required unless --rescore/--rescore-dir is given")

    manifest = Manifest.load(args.manifest)
    truth = Truth.for_manifest(manifest, truth_table_path=args.truth_table)

    backend_factory: Callable[[], Backend] | None = None
    if args.backend == "fake":
        backend = _fake_backend(truth, args.source_ids)
    elif args.backend == "mlx":
        if args.concurrency > 1:
            raise SystemExit("--backend mlx holds one model in memory; use --concurrency 1")
        backend = _mlx_backend(args.model, args.adapter_path, args.max_tokens)
        print(
            f"backend resolved: provider='mlx' model={args.model!r} adapter={args.adapter_path!r}",
            file=sys.stderr,
        )
        os.environ[TEMPERATURE_ENV_VAR] = str(args.temperature)
    else:
        backend = _live_backend(args.backend, args.model)
        # infra-2b wave, 2026-09-18: named explicitly, every run, so a
        # provider/env mismatch (the `--backend agent_sdk` / `LLM_PROVIDER=
        # openrouter` bug this wave fixed in `get_llm`/`_live_backend`) is
        # visible at a glance instead of discovered from an unusable
        # output and a real dollar cost after the fact.
        print(
            f"backend resolved: provider={args.backend!r} "
            f"model={args.model or '<provider default>'!r}",
            file=sys.stderr,
        )
        if args.concurrency > 1:
            backend_factory = _live_backend_factory(args.backend, args.model)
        # Determinism, wired for real (infra-2b wave, 2026-09-18,
        # `01-spec/infra-2b-2026-09-18.md`; supersedes the harness-review
        # wave's recorded-only stub, `01-spec/harness-review-2026-09-18.md`
        # determinism section). `portolan.extract.backends.
        # LLMClientBackend` and the judge call site (`portolan.bench.
        # extraction`) are still outside this wave's ownership and forward
        # no `temperature=` keyword of their own -- so instead of a new
        # parameter threaded through either, this sets the
        # `PORTOLAN_TEMPERATURE` env var (`portolan.llm.base.
        # TEMPERATURE_ENV_VAR`), which every client's own `complete()`
        # reads fresh, via `resolve_temperature()`, for the `extract`/
        # `audit` surfaces only -- the exact same "flag plumbing sets an
        # env var the client reads per-call" seam `portolan.llm.openrouter`
        # already uses for `PORTOLAN_OPENROUTER_REASONING_EFFORT`. Report
        # prose and chat are untouched by this or any prior wave.
        os.environ[TEMPERATURE_ENV_VAR] = str(args.temperature)
        agent_sdk_note = (
            " (AgentSDKClient still can't honour this -- the Claude Agent SDK's "
            "ClaudeAgentOptions exposes no temperature field at all; see that module)"
            if args.backend == "agent_sdk"
            else ""
        )
        print(
            f"--temperature={args.temperature} is wired to every extract/judge call on "
            f"this run via {TEMPERATURE_ENV_VAR}{agent_sdk_note}.",
            file=sys.stderr,
        )

    # Company-context wave, 2026-09-17 (01-spec/company-context-eval-
    # 2026-09-17.md): `--company-context` builds the system prompt with
    # the synthetic Halvard Medical company-context block rendered in
    # (`portolan.context.HALVARD_MEDICAL_CONTEXT_BODY`, no database
    # needed -- the bench runs against the corpus, not a live tenant).
    # `--variant` (bench-provenance wave, 2026-09-18): explicitly passed
    # through every time now, defaulting to `_DEFAULT_PROMPT_VARIANT`
    # (`build_system_prompt`'s own default) -- so omitting BOTH flags
    # reproduces the exact pre-existing behavior byte-for-byte
    # (`run_extraction`'s own former default, `system_prompt=None` ->
    # `build_system_prompt()`), while the CLI can now also stamp
    # `prompt_variant` onto the written report truthfully instead of
    # leaving it undiscoverable outside a directory name.
    company_context_body = None
    if args.company_context:
        from portolan.context import HALVARD_MEDICAL_CONTEXT_BODY

        company_context_body = HALVARD_MEDICAL_CONTEXT_BODY
    system_prompt = build_system_prompt(company_context=company_context_body, variant=args.variant)

    run_result = run_extraction(
        manifest,
        backend,
        source_ids=args.source_ids,
        segment=not args.no_segment,
        corpus_root=args.corpus,
        system_prompt=system_prompt,
        concurrency=args.concurrency,
        backend_factory=backend_factory,
        truth=truth,
        samples=args.samples,
    )
    if args.judge and args.backend == "mlx":
        raise SystemExit("--judge is not wired for --backend mlx (no hosted judge model)")
    judge = _judge_client(args.backend, args.model) if args.judge else None
    score = score_extraction(truth, run_result, judge=judge)
    extractor_model = "fake" if args.backend == "fake" else (args.model or "<provider default>")
    if args.backend == "mlx" and args.adapter_path:
        extractor_model = f"{extractor_model}+adapter:{args.adapter_path}"
    report = BenchReport(
        backend=args.backend,
        date=dt.date.today(),
        extraction=score,
        temperature_requested=args.temperature,
        samples=args.samples,
        **provenance_kwargs(
            corpus=args.corpus,
            truth_table=args.truth_table,
            prompt_variant=args.variant,
            extractor_model=extractor_model,
        ),
    )
    md_path, json_path = write(report, args.out)

    print(f"wrote {md_path}")
    print(f"wrote {json_path}")
    # Named `parser-*.json`, not `bench-*.json`, so `--rescore-dir`'s glob
    # never mistakes it for a saved report.
    parser_counts = _parser_counts(run_result)
    parser_path = md_path.with_name(f"parser-{md_path.stem}.json")
    parser_path.write_text(json.dumps(parser_counts, indent=2, sort_keys=True) + "\n")
    print(f"wrote {parser_path}")
    print(
        f"parser: rejected_facts={parser_counts['rejected_facts']} "
        f"coercions={parser_counts['coercions_total']} "
        f"failed_sources={len(parser_counts['failed_sources'])} "
        f"model_calls={parser_counts['model_calls']} retry_calls={parser_counts['retry_calls']}"
    )
    precision = score.precision if score.precision is not None else float("nan")
    fact_type_accuracy = (
        score.fact_type_accuracy if score.fact_type_accuracy is not None else float("nan")
    )
    lenient_recall = score.lenient_recall if score.lenient_recall is not None else float("nan")
    print(
        f"recall={score.recall:.3f} lenient_recall={lenient_recall:.3f} "
        f"precision={precision:.3f} "
        f"fact_type_accuracy={fact_type_accuracy:.3f} "
        f"distractor_leakage={score.distractor_leakage}"
    )
    if getattr(args, "dispute_judges", None):
        from portolan.bench.world_cli import halvard_dispute_step

        return halvard_dispute_step(args, score, run_result, manifest, extractor_model)
    return 0


def _cmd_graph(args: argparse.Namespace) -> int:
    from portolan import db as db_module

    manifest = Manifest.load(args.manifest)
    truth = Truth.for_manifest(manifest, truth_table_path=args.truth_table)

    conn = db_module.connect(args.database_url)
    try:
        score = score_graph(conn, truth)
    finally:
        conn.close()

    report = BenchReport(backend=args.label, date=dt.date.today(), graph=score)
    md_path, json_path = write(report, args.out)

    print(f"wrote {md_path}")
    print(f"wrote {json_path}")
    print(f"graph: {score.passed_count}/{score.passed_count + score.failed_count} passed")
    for failure in score.failures():
        print(f"  FAIL [{failure.category}] {failure.name}: {failure.detail}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m portolan.bench")
    sub = parser.add_subparsers(dest="command", required=True)

    extract_p = sub.add_parser("extract", help="run extraction over the corpus and score it")
    extract_p.add_argument(
        "--backend",
        default=None,
        choices=["fake", "agent_sdk", "bedrock", "anthropic", "openrouter", "mlx"],
        help="required unless --rescore is given; `mlx` runs a local model in-process "
        "(needs `uv sync --group mlx`)",
    )
    extract_p.add_argument("--model", default=None, help="model id (agent_sdk/bedrock only)")
    extract_p.add_argument(
        "--adapter-path",
        default=None,
        help="--backend mlx only: a LoRA adapter directory (adapters.safetensors + "
        "adapter_config.json) from scripts/train/mlx_lora.py; omit for the base model",
    )
    extract_p.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="--backend mlx only: cap generated tokens per call below the extractor's own "
        "4096 (default: no extra cap)",
    )
    extract_p.add_argument(
        "--rescore",
        default=None,
        metavar="SAVED_REPORT_JSON",
        help="re-score mode: no backend, no model call. Path to a previously saved bench "
        "report's JSON (the '<name>.json' twin `write()` produces alongside its .md). "
        "Re-runs ONLY the invented/evidence_basis classification "
        "(portolan.bench.extraction.rescore_unplanted_classifications) against the corpus "
        "on disk (--corpus/--manifest); recall, precision, matched, unmatched, tokens, "
        "cost and wall time are carried forward from the saved report unchanged. Every "
        "other extract flag is ignored when this is given.",
    )
    extract_p.add_argument(
        "--rescore-dir",
        default=None,
        metavar="DIR",
        help="full re-score mode: no backend, no model call. Recursively finds every "
        "saved bench-*.json report under DIR and re-runs the ENTIRE match "
        "(portolan.bench.extraction.rescore_full) against the CURRENT matcher code -- "
        "recall, precision, fact-type accuracy, strict AND lenient recall, "
        "near-miss-credited, matched/unmatched/unplanted, all of it. Unlike --rescore "
        "(evidence_basis only, recall untouched by design), this DOES change recall when "
        "a matcher/locator fix changed since the saved run. Writes one rescored report "
        "per input under --out (mirroring its subpath) plus <out>/rescore-summary.json. "
        "Takes precedence over --rescore when both are given.",
    )
    extract_p.add_argument("--corpus", default=str(DEFAULT_CORPUS_ROOT))
    extract_p.add_argument("--out", default="var/bench")
    extract_p.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH))
    extract_p.add_argument("--truth-table", default=str(DEFAULT_TRUTH_TABLE_PATH))
    extract_p.add_argument("--source-ids", nargs="*", default=None, help="score only these sources")
    extract_p.add_argument("--no-segment", action="store_true")
    extract_p.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="run sources through a thread pool of this many workers "
        "(agent_sdk/bedrock: a fresh backend/client per worker, not shared)",
    )
    extract_p.add_argument(
        "--samples",
        type=int,
        default=1,
        help="run each source's extraction this many independent times and union the "
        "facts (portolan.extract.segment.merge_segment_results's existing dedup rule) -- "
        "default 1, today's unchanged single-pass behavior. Every sample is a real "
        "backend call: tokens/cost/wall time scale with N. Measures whether repeated "
        "sampling recovers more planted facts than one run alone, and what it costs in "
        "invented facts (determinism section, 01-spec/harness-review-2026-09-18.md).",
    )
    extract_p.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="recorded into the written report (BenchReport.temperature_requested) AND, "
        "as of the infra-2b wave (01-spec/infra-2b-2026-09-18.md), actually sent on every "
        "live extract/judge call via portolan.llm.base.resolve_temperature (set through "
        "the PORTOLAN_TEMPERATURE env var -- portolan.extract.backends.LLMClientBackend "
        "and the judge call site aren't owned by this wave, so this is the same "
        "'flag-plumbing sets an env var the client reads fresh' seam "
        "PORTOLAN_OPENROUTER_REASONING_EFFORT already uses). A client that has no request "
        "field for temperature at all (AgentSDKClient -- see that module) still can't "
        "honour it; see the determinism section of 01-spec/harness-review-2026-09-18.md "
        "for the history.",
    )
    extract_p.add_argument(
        "--judge",
        action="store_true",
        help="grade the unmatched remainder with a strong-model judge call "
        "(off by default; adds one call per unmatched planted fact)",
    )
    extract_p.add_argument(
        "--company-context",
        action="store_true",
        help="render the synthetic Halvard Medical company-context block into the system "
        "prompt (off by default; company-context wave, 01-spec/company-context-eval-"
        "2026-09-17.md's A/B run)",
    )
    extract_p.add_argument(
        "--variant",
        default=_DEFAULT_PROMPT_VARIANT,
        help="prompt A/B variant passed to portolan.extract.prompt.build_system_prompt "
        f"(default: that module's own default, currently {_DEFAULT_PROMPT_VARIANT!r} -- read "
        "via inspect, never hardcoded here, so this can't drift out of sync with it). "
        "2026-09-18 bench-provenance wave: added so a saved run's JSON/report header can "
        "truthfully stamp which variant produced these numbers -- before this flag, the only "
        "way to run a non-default variant through the CLI was a scratchpad driver outside the "
        "repo (01-spec/facttype-4a-2026-09-18.md, 01-spec/prompt-ab-2026-09-18.md).",
    )
    from portolan.bench.world_cli import add_disputes_subcommand, add_world_args

    add_world_args(extract_p)
    extract_p.set_defaults(func=_cmd_extract)

    graph_p = sub.add_parser("graph", help="score the database against expectations.merge/filter")
    graph_p.add_argument("--out", default="var/bench")
    graph_p.add_argument("--database-url", default=None, help="defaults to $DATABASE_URL")
    graph_p.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH))
    graph_p.add_argument("--truth-table", default=str(DEFAULT_TRUTH_TABLE_PATH))
    graph_p.add_argument(
        "--label", default="graph", help="backend label used in the report filename"
    )
    graph_p.set_defaults(func=_cmd_graph)

    from portolan.bench.judge_cli import add_judge_subcommand  # wave 3A two-line hook

    add_judge_subcommand(sub)
    add_disputes_subcommand(sub)

    from portolan.bench.world_questions_cli import add_questions_subcommand

    add_questions_subcommand(sub)

    from portolan.bench.chat_exam_cli import add_chat_exam_subcommand

    add_chat_exam_subcommand(sub)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
