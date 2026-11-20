#!/usr/bin/env python3
"""Two-pass extraction bench: pass 1 is FROZEN (the saved Nova Lite run 1
at `var/bench/2026-09-18-corrected/amazon_nova-lite-v1/run1/`); this
script runs the SECOND pass (one reviewer fork) over every source pass 1
covered, combines the reviewer's verdicts and sweep with pass 1's facts
three ways (`combined`, `verify_only`, `sweep_only`,
`portolan.bench.review_bench`), scores each with the bench's existing
scorer, and writes one JSON artifact per run.

`01-spec/extraction-accuracy-campaign-2026-09-18.md` wave 1D. See
`01-spec/two-pass-2026-09-18.md` for the results this produces.

Two subcommands, so a slow reviewer (the local `agent_sdk` Haiku fork:
one Claude Agent SDK CLI subprocess per source, ~100s+ per call even
under concurrency) can be run in several sequential foreground calls
without ever polling or backgrounding -- `collect` is resumable (it skips
any source_id already recorded in the run's `reviews.json`), `score` is
pure computation over whatever `collect` has saved so far and can be
re-run for free.

    uv run python scripts/bench_two_pass.py collect --reviewer nova-self --run-index 1 \\
        --concurrency 12
    uv run python scripts/bench_two_pass.py collect --reviewer haiku --run-index 1 \\
        --concurrency 16 --source-ids S-call-001 S-call-002 ...   # one chunk
    uv run python scripts/bench_two_pass.py collect --reviewer jev --run-index 1
    uv run python scripts/bench_two_pass.py score --reviewer nova-self --run-index 1
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from portolan.bench.corpus_loader import DEFAULT_CORPUS_ROOT, load_prepared_source
from portolan.bench.extraction import (
    ExtractionScore,
    ProducedFactRecord,
    _build_envelope,
    _registry_context_for,
    _vocabulary_from_truth,
)
from portolan.bench.review_bench import (
    COMBINE_MODES,
    combine_source,
    rejected_fact_classification,
    score_combined,
    sweep_recovered_fact_ids,
)
from portolan.bench.truth import DEFAULT_MANIFEST_PATH, DEFAULT_TRUTH_TABLE_PATH, Manifest, Truth
from portolan.context import HALVARD_MEDICAL_CONTEXT_BODY
from portolan.extract.review_pass import ReviewResult, build_review_system_prompt, review_source
from portolan.extract.schema import Fact
from portolan.pricing import estimate_cost

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_ROOT = REPO_ROOT / "var" / "bench" / "2026-09-18-twopass"
DEFAULT_PASS1_REPORT = (
    REPO_ROOT
    / "var"
    / "bench"
    / "2026-09-18-corrected"
    / "amazon_nova-lite-v1"
    / "run1"
    / "bench-2026-09-18-openrouter.json"
)

# Jev pricing, same source as scripts/jev_eval/judge_and_lint.py:
# var/jev-typesafe-brief-2026-09-16.md sec 4.
JEV_PRICE_PER_MILLION_INPUT = 0.042
JEV_URL = "https://api.typesafe.ai/v1/systemone"
JEV_MODEL = "jev-latest"
JEV_REJECT_THRESHOLD = 0.7  # 01-spec/jev-eval-2026-09-16/A-judge-and-lint.md verdict

REVIEWER_FORKS: dict[str, dict[str, str]] = {
    "nova-self": {"kind": "openrouter", "model": "amazon/nova-lite-v1", "label": "nova-lite-self"},
    "deepseek": {
        "kind": "openrouter",
        "model": "deepseek/deepseek-v4.1-flash",
        "label": "deepseek-v4.1-flash",
    },
    "kimi": {"kind": "openrouter", "model": "moonshotai/kimi-k2.5", "label": "kimi-k2.5"},
    "haiku": {
        "kind": "agent_sdk",
        "model": "claude-haiku-4-5-20251001",
        "label": "claude-haiku-4-5-agent-sdk",
    },
    "jev": {"kind": "jev", "model": JEV_MODEL, "label": "jev-verify-only"},
}


# --------------------------------------------------------------------------
# Loading pass 1 (frozen)
# --------------------------------------------------------------------------


def _load_pass1(path: Path) -> ExtractionScore:
    data = json.loads(path.read_text(encoding="utf-8"))
    return ExtractionScore.model_validate(data["extraction"])


def _pass1_by_source(score: ExtractionScore) -> dict[str, list[ProducedFactRecord]]:
    return dict(score.produced_by_source)


# --------------------------------------------------------------------------
# collect -- one reviewer call per source
# --------------------------------------------------------------------------


@dataclass
class _WorkResult:
    source_id: str
    text: str
    review: ReviewResult
    dropped_count: int
    usage: dict[str, Any] | None
    error: str | None


SPEND_KEY = "bench_two_pass"


def _build_openrouter_client():
    from portolan.llm.openrouter import OpenRouterClient
    from portolan.metering import InMemorySink
    from portolan.training.spend import SpendMeter, default_spend_path

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY not set (source .env first)")
    return OpenRouterClient(
        spend_key=SPEND_KEY,
        spend_meter=SpendMeter(default_spend_path(REPO_ROOT / "var")),
        api_key=api_key,
        sink=InMemorySink(),
        tenant="bench-review",
    )


def _build_agent_sdk_client():
    from portolan.llm.agent_sdk import AgentSDKClient
    from portolan.metering import InMemorySink

    return AgentSDKClient(sink=InMemorySink(), tenant="bench-review")


def _collect_llm(
    *,
    fork: dict[str, str],
    truth: Truth,
    manifest: Manifest,
    pass1_by_source: dict[str, list[ProducedFactRecord]],
    source_ids: list[str],
    corpus_root: Path,
    concurrency: int,
    max_tokens: int,
) -> dict[str, dict[str, Any]]:
    system_prompt = build_review_system_prompt(company_context=HALVARD_MEDICAL_CONTEXT_BODY)
    vocabulary = _vocabulary_from_truth(truth)
    model = fork["model"]
    kind = fork["kind"]

    local = threading.local()

    def _client_for_thread():
        client = getattr(local, "client", None)
        if client is None:
            client = (
                _build_openrouter_client() if kind == "openrouter" else _build_agent_sdk_client()
            )
            local.client = client
        return client

    def _work(source_id: str) -> _WorkResult:
        manifest_source = manifest.source(source_id)
        prepared = load_prepared_source(manifest_source, corpus_root=corpus_root)
        registry_context = _registry_context_for(manifest_source, truth)
        envelope = _build_envelope(
            prepared.participants,
            prepared.source_id,
            prepared.source_type,
            prepared.source_date,
            prepared.text,
            None,
            registry_context,
            vocabulary,
        )
        pass1_facts = pass1_by_source.get(source_id, [])
        client = _client_for_thread()
        try:
            review, dropped, _llm_result = review_source(
                client,
                envelope=envelope,
                pass1_facts=pass1_facts,
                system_prompt=system_prompt,
                model=model,
                max_tokens=max_tokens,
            )
        except Exception as exc:  # noqa: BLE001 - one bad source must not sink the run
            return _WorkResult(
                source_id=source_id,
                text=prepared.text,
                review=ReviewResult(),
                dropped_count=0,
                usage=None,
                error=f"{type(exc).__name__}: {exc}",
            )
        sink_events = getattr(client, "sink", None)
        usage: dict[str, Any] | None = None
        if sink_events is not None and sink_events.events:
            ev = sink_events.events[-1]
            usage = {
                "input_tokens": ev.input_tokens,
                "output_tokens": ev.output_tokens,
                "cached_tokens": ev.cached_tokens,
                "cost_estimate": ev.cost_estimate,
            }
            # A fresh InMemorySink per thread-local client means this is
            # always exactly the event this call just recorded -- no
            # cross-thread interleaving to worry about (see module
            # docstring: one client, and one sink, per worker thread).
            sink_events.events.clear()
        return _WorkResult(
            source_id=source_id,
            text=prepared.text,
            review=review,
            dropped_count=len(dropped),
            usage=usage,
            error=None,
        )

    out: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for r in pool.map(_work, source_ids):
            out[r.source_id] = {
                "source_id": r.source_id,
                "pass1_fact_ids": [f.id for f in pass1_by_source.get(r.source_id, [])],
                "review": r.review.model_dump(mode="json"),
                "dropped_count": r.dropped_count,
                "usage": r.usage,
                "error": r.error,
            }
    return out


def _jev_state_and_questions(text: str, facts: list[ProducedFactRecord]) -> tuple[str, dict]:
    state = f"Source text:\n{text}\n\nCandidate claims to check against the source above."
    questions = {}
    for f in facts:
        questions[f.id] = {
            "type": "noul",
            "instructions": (
                f'Does the source state this claim (type: {f.type})? Claim: "{f.claim}"'
            ),
        }
    return state, questions


def _call_jev(client: httpx.Client, api_key: str, state: str, questions: dict) -> dict:
    body = {"state": state, "model": JEV_MODEL, "questions": questions}
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            resp = client.post(
                JEV_URL, json=body, headers={"Authorization": f"Bearer {api_key}"}, timeout=60.0
            )
        except httpx.HTTPError as exc:
            last_exc = exc
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
                continue
            raise RuntimeError(f"Jev call failed after retries: {exc}") from exc
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code in (429, 529) and attempt < 2:
            time.sleep(2 * (attempt + 1))
            continue
        raise RuntimeError(f"Jev HTTP {resp.status_code}: {resp.text[:500]}")
    raise RuntimeError(f"Jev call failed after retries: {last_exc}")


def _collect_jev(
    *,
    truth: Truth,
    manifest: Manifest,
    pass1_by_source: dict[str, list[ProducedFactRecord]],
    source_ids: list[str],
    corpus_root: Path,
    concurrency: int,
) -> dict[str, dict[str, Any]]:
    """Jev verify-only: one batched Jev call per source (one `noul`
    question per pass-1 fact in that source -- '01-spec/jev-eval-2026-09-
    16/README.md's "noul yes/no questions batched per input"), reject
    below `JEV_REJECT_THRESHOLD`. No sweep -- Jev cannot write, so
    `review.missed` is always empty for this fork (the DESIGN's "Jev
    verify-only ... no sweep since Jev cannot write")."""
    api_key = os.environ.get("TYPESAFE_API_KEY")
    if not api_key:
        raise SystemExit("TYPESAFE_API_KEY not set (source .env first)")

    def _work(source_id: str) -> _WorkResult:
        manifest_source = manifest.source(source_id)
        prepared = load_prepared_source(manifest_source, corpus_root=corpus_root)
        facts = pass1_by_source.get(source_id, [])
        if not facts:
            return _WorkResult(
                source_id=source_id,
                text=prepared.text,
                review=ReviewResult(),
                dropped_count=0,
                usage={
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cached_tokens": 0,
                    "cost_estimate": 0.0,
                },
                error=None,
            )
        state, questions = _jev_state_and_questions(prepared.text, facts)
        client = httpx.Client()
        try:
            resp = _call_jev(client, api_key, state, questions)
        except Exception as exc:  # noqa: BLE001
            return _WorkResult(
                source_id=source_id,
                text=prepared.text,
                review=ReviewResult(),
                dropped_count=0,
                usage=None,
                error=f"{type(exc).__name__}: {exc}",
            )
        finally:
            client.close()

        verdicts = []
        for f in facts:
            answer = resp.get("answers", {}).get(f.id, {})
            noul = answer.get("noul")
            if noul is None or noul >= JEV_REJECT_THRESHOLD:
                verdicts.append({"fact_id": f.id, "verdict": "keep"})
            else:
                verdicts.append(
                    {
                        "fact_id": f.id,
                        "verdict": "reject",
                        "reject_reason": "not_in_source",
                        "reason": f"jev noul={noul}",
                    }
                )
        review = ReviewResult.model_validate({"verdicts": verdicts, "missed": []})
        usage_block = resp.get("usage", {})
        input_tokens = int(usage_block.get("input_tokens", 0) or 0)
        cost = input_tokens * JEV_PRICE_PER_MILLION_INPUT / 1_000_000
        usage = {
            "input_tokens": input_tokens,
            "output_tokens": int(usage_block.get("output_tokens", 0) or 0),
            "cached_tokens": 0,
            "cost_estimate": cost,
        }
        return _WorkResult(
            source_id=source_id,
            text=prepared.text,
            review=review,
            dropped_count=0,
            usage=usage,
            error=None,
        )

    out: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for r in pool.map(_work, source_ids):
            out[r.source_id] = {
                "source_id": r.source_id,
                "pass1_fact_ids": [f.id for f in pass1_by_source.get(r.source_id, [])],
                "review": r.review.model_dump(mode="json"),
                "dropped_count": r.dropped_count,
                "usage": r.usage,
                "error": r.error,
            }
    return out


def cmd_collect(args: argparse.Namespace) -> int:
    fork = REVIEWER_FORKS[args.reviewer]
    out_dir = Path(args.out_root) / fork["label"] / f"run{args.run_index}"
    out_dir.mkdir(parents=True, exist_ok=True)
    reviews_path = out_dir / "reviews.json"
    existing: dict[str, dict[str, Any]] = {}
    if reviews_path.exists():
        existing = json.loads(reviews_path.read_text(encoding="utf-8"))

    truth = Truth.load(args.truth_table, args.manifest)
    manifest = truth.manifest
    pass1_score = _load_pass1(Path(args.pass1_report))
    pass1_by_source = _pass1_by_source(pass1_score)

    all_source_ids = sorted(pass1_by_source.keys())
    requested = args.source_ids if args.source_ids else all_source_ids
    todo = [sid for sid in requested if sid not in existing]
    print(
        f"reviewer={args.reviewer} run={args.run_index}: {len(todo)} sources to collect "
        f"({len(existing)}/{len(all_source_ids)} already done)"
    )
    if not todo:
        return 0

    t0 = time.perf_counter()
    if fork["kind"] == "jev":
        new_results = _collect_jev(
            truth=truth,
            manifest=manifest,
            pass1_by_source=pass1_by_source,
            source_ids=todo,
            corpus_root=args.corpus,
            concurrency=args.concurrency,
        )
    else:
        new_results = _collect_llm(
            fork=fork,
            truth=truth,
            manifest=manifest,
            pass1_by_source=pass1_by_source,
            source_ids=todo,
            corpus_root=args.corpus,
            concurrency=args.concurrency,
            max_tokens=args.max_tokens,
        )
    elapsed = time.perf_counter() - t0

    existing.update(new_results)
    reviews_path.write_text(json.dumps(existing, indent=2, default=str), encoding="utf-8")
    n_errors = sum(1 for v in new_results.values() if v.get("error"))
    print(
        f"collected {len(new_results)} sources in {elapsed:.1f}s "
        f"({n_errors} errors) -> {reviews_path} ({len(existing)}/{len(all_source_ids)} total)"
    )
    return 0


# --------------------------------------------------------------------------
# score -- combine + score all three modes from a saved reviews.json
# --------------------------------------------------------------------------


def cmd_score(args: argparse.Namespace) -> int:
    fork = REVIEWER_FORKS[args.reviewer]
    out_dir = Path(args.out_root) / fork["label"] / f"run{args.run_index}"
    reviews_path = out_dir / "reviews.json"
    if not reviews_path.exists():
        raise SystemExit(f"{reviews_path} does not exist -- run collect first")
    reviews_raw: dict[str, dict[str, Any]] = json.loads(reviews_path.read_text(encoding="utf-8"))

    truth = Truth.load(args.truth_table, args.manifest)
    manifest = truth.manifest
    pass1_score = _load_pass1(Path(args.pass1_report))
    pass1_by_source = _pass1_by_source(pass1_score)

    all_source_ids = sorted(pass1_by_source.keys())
    missing = [sid for sid in all_source_ids if sid not in reviews_raw]
    if missing:
        print(
            f"WARNING: {len(missing)} sources have no collected review yet (treated as no "
            f"reviewer output -- pass 1 facts kept unchanged, no sweep): {missing[:5]}..."
        )

    source_texts: dict[str, str] = {}
    for sid in all_source_ids:
        prepared = load_prepared_source(manifest.source(sid), corpus_root=args.corpus)
        source_texts[sid] = prepared.text

    reviews_by_source: dict[str, ReviewResult] = {}
    n_errors = 0
    total_usage = {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0, "cost_estimate": 0.0}
    for sid in all_source_ids:
        entry = reviews_raw.get(sid)
        if entry is None or entry.get("error"):
            reviews_by_source[sid] = ReviewResult()
            if entry is not None:
                n_errors += 1
            continue
        reviews_by_source[sid] = ReviewResult.model_validate(entry["review"])
        usage = entry.get("usage")
        if usage:
            total_usage["input_tokens"] += usage.get("input_tokens", 0) or 0
            total_usage["output_tokens"] += usage.get("output_tokens", 0) or 0
            total_usage["cached_tokens"] += usage.get("cached_tokens", 0) or 0
            if usage.get("cost_estimate") is not None:
                total_usage["cost_estimate"] += usage["cost_estimate"]

    # Pass 1: compute every mode's combined facts + score first (needed
    # before sweep-recovery can be computed -- `combined`'s baseline is
    # THIS SAME RUN's `verify_only` score, so verify_only must exist
    # before combined's sweep-recovery diff can be taken).
    raw_scores: dict[str, ExtractionScore] = {}
    mode_diag: dict[str, dict[str, Any]] = {}
    for mode in COMBINE_MODES:
        combined_by_source: dict[str, list[Fact]] = {}
        rejected_all: list[tuple[str, str, str | None]] = []  # (source_id, fact_id, reason)
        fixed_all: list[tuple[str, str]] = []
        missed_added_total = 0
        for sid in all_source_ids:
            facts, diag = combine_source(
                sid, pass1_by_source.get(sid, []), reviews_by_source[sid], mode=mode
            )
            combined_by_source[sid] = facts
            for fact_id, reason in diag.rejected:
                rejected_all.append((sid, fact_id, reason))
            for fact_id in diag.fixed:
                fixed_all.append((sid, fact_id))
            missed_added_total += diag.missed_added

        raw_scores[mode] = score_combined(truth, combined_by_source, source_texts)
        mode_diag[mode] = {
            "rejected_all": rejected_all,
            "fixed_all": fixed_all,
            "missed_added_total": missed_added_total,
        }

    # Pass 2: sweep-recovery, mode-appropriate baseline (see
    # sweep_recovered_fact_ids's own docstring for why each baseline is
    # what it is -- isolating the sweep's own marginal contribution from
    # what a `fix` alone could already recover).
    sweep_baseline = {
        "combined": raw_scores["verify_only"],
        "verify_only": None,  # verify_only ran no sweep at all
        "sweep_only": pass1_score,
    }

    mode_results: dict[str, Any] = {}
    for mode in COMBINE_MODES:
        score = raw_scores[mode]
        diag = mode_diag[mode]

        rejected_that_were_invented = 0
        rejected_classification_counts: dict[str, int] = {}
        for sid, fact_id, _reason in diag["rejected_all"]:
            classification = rejected_fact_classification(pass1_score, sid, fact_id)
            rejected_classification_counts[classification] = (
                rejected_classification_counts.get(classification, 0) + 1
            )
            if classification == "invented":
                rejected_that_were_invented += 1

        baseline = sweep_baseline[mode]
        sweep_recovered = sweep_recovered_fact_ids(score, baseline) if baseline is not None else []

        mode_results[mode] = {
            "score": score.model_dump(mode="json"),
            "rejected_count": len(diag["rejected_all"]),
            "rejected_that_were_invented": rejected_that_were_invented,
            "rejected_classification_counts": rejected_classification_counts,
            "fixed_count": len(diag["fixed_all"]),
            "missed_added_total": diag["missed_added_total"],
            "sweep_recovered_fact_ids": sweep_recovered,
            "sweep_recovered_count": len(sweep_recovered),
        }

    list_price_equivalent = None
    if fork["kind"] == "agent_sdk":
        # Haiku via the local claude_agent_sdk backend runs on the maintainer's
        # Claude subscription, not billed per token -- report the
        # equivalent Anthropic list price (portolan.pricing's own
        # "claude-haiku-4-5" row, the same $/M figure OpenRouter passes
        # through unmarked up for this model) from the recorded token
        # counts, so the accuracy-per-dollar column stays comparable to
        # every OpenRouter fork's real charge.
        list_price_equivalent = estimate_cost(
            "claude-haiku-4-5",
            total_usage["input_tokens"],
            total_usage["output_tokens"],
            total_usage["cached_tokens"],
        )

    out = {
        "reviewer": args.reviewer,
        "run_index": args.run_index,
        "label": fork["label"],
        "model": fork["model"],
        "pass1_report": str(args.pass1_report),
        "sources_reviewed": len(all_source_ids),
        "sources_missing_review": len(missing),
        "sources_review_error": n_errors,
        "usage": total_usage,
        "billing": "subscription (no API charge)" if fork["kind"] == "agent_sdk" else "openrouter",
        "list_price_equivalent_usd": list_price_equivalent,
        "modes": mode_results,
    }
    scored_path = out_dir / "scored.json"
    scored_path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")

    print(f"wrote {scored_path}")
    for mode in COMBINE_MODES:
        s = mode_results[mode]["score"]
        invented_count = sum(
            1 for u in s["unplanted"] if not u["is_distractor_leak"] and not u["in_source_text"]
        )
        print(
            f"{args.reviewer} run{args.run_index} [{mode}]: recall={s['recall']:.4f} "
            f"fact_type_accuracy={(s['fact_type_accuracy'] or 0):.4f} "
            f"invented={invented_count} "
            f"produced={s['produced_total']} "
            f"rejected={mode_results[mode]['rejected_count']} "
            f"rejected_invented={mode_results[mode]['rejected_that_were_invented']} "
            f"sweep_recovered={mode_results[mode]['sweep_recovered_count']}"
        )
    print(
        f"reviewer usage: input={total_usage['input_tokens']} output={total_usage['output_tokens']} "
        f"cost=${total_usage['cost_estimate']:.4f}"
    )
    return 0


# --------------------------------------------------------------------------
# report -- aggregate every fork's scored.json plus pass 1 alone into the
# exact numbers 01-spec/two-pass-2026-09-18.md quotes. A report helper,
# not part of the reviewer/combine/score pipeline; every number it prints
# is read from a saved JSON file (median/sum arithmetic on those numbers
# only), per the project's numbers-by-code rule -- nothing hand-computed.
# --------------------------------------------------------------------------

# Wall-clock seconds `collect` itself printed for each run (read off the
# terminal output of the actual 2026-09-18 collection calls -- this
# script does not re-measure wall time, since `score` makes no model
# calls). Filled in as runs are collected; a run not yet here just omits
# a wall figure rather than failing. Haiku's figure sums its four
# sequential chunks (52 sources each, --concurrency 20): a single Claude
# Agent SDK CLI subprocess call runs ~100s+ even under concurrency (see
# `run_extraction`'s own docstring on `agent_sdk`), so one source set was
# split across four foreground calls, none of which polled or backgrounded.
COLLECT_WALL_S: dict[tuple[str, int], float] = {
    ("nova-self", 1): 152.7,
    ("nova-self", 2): 70.0,
    ("nova-self", 3): 70.0,
    ("deepseek", 1): 401.3,
    ("kimi", 1): 531.8,
    ("jev", 1): 10.5,
    ("haiku", 1): 383.7 + 232.8 + 379.9 + 508.7,
}


def _pass1_numbers(pass1_report: Path) -> dict[str, Any]:
    d = json.loads(pass1_report.read_text(encoding="utf-8"))["extraction"]
    produced = d["produced_total"]
    invented = sum(
        1 for u in d["unplanted"] if not u["is_distractor_leak"] and not u["in_source_text"]
    )
    return {
        "recall": d["recall"],
        "produced_total": produced,
        "invented_count": invented,
        "invented_rate": invented / produced if produced else None,
        "fact_type_accuracy": d["fact_type_accuracy"],
        "matched_count": d["matched_count"],
        "planted_total": d["planted_total"],
    }


def _mode_numbers(scored: dict[str, Any], mode: str) -> dict[str, Any]:
    m = scored["modes"][mode]
    s = m["score"]
    produced = s["produced_total"]
    invented = sum(
        1 for u in s["unplanted"] if not u["is_distractor_leak"] and not u["in_source_text"]
    )
    return {
        "recall": s["recall"],
        "produced_total": produced,
        "invented_count": invented,
        "invented_rate": invented / produced if produced else None,
        "fact_type_accuracy": s["fact_type_accuracy"],
        "rejected_count": m["rejected_count"],
        "rejected_that_were_invented": m["rejected_that_were_invented"],
        "rejected_classification_counts": m["rejected_classification_counts"],
        "sweep_recovered_count": m["sweep_recovered_count"],
        "sweep_recovered_fact_ids": m["sweep_recovered_fact_ids"],
    }


def cmd_report(args: argparse.Namespace) -> int:
    import statistics

    print("=== pass 1 alone (frozen) ===")
    p1 = _pass1_numbers(Path(args.pass1_report))
    print(json.dumps(p1, indent=2))

    for reviewer, fork in REVIEWER_FORKS.items():
        label = fork["label"]
        run_dirs = sorted((Path(args.out_root) / label).glob("run*"))
        run_indices = [int(p.name.removeprefix("run")) for p in run_dirs if p.name[3:].isdigit()]
        if not run_indices:
            continue
        print(f"\n=== {reviewer} ({label}) ===")
        per_run_usage = []
        per_run_modes: dict[str, list[dict[str, Any]]] = {
            "combined": [],
            "verify_only": [],
            "sweep_only": [],
        }
        for run_index in sorted(run_indices):
            scored_path = Path(args.out_root) / label / f"run{run_index}" / "scored.json"
            if not scored_path.exists():
                continue
            scored = json.loads(scored_path.read_text(encoding="utf-8"))
            usage = scored["usage"]
            per_run_usage.append(usage)
            wall = COLLECT_WALL_S.get((reviewer, run_index))
            print(
                f"run{run_index}: usage input={usage['input_tokens']} "
                f"output={usage['output_tokens']} cost=${usage['cost_estimate']:.4f} "
                f"billing={scored['billing']} "
                f"list_price_equiv={scored.get('list_price_equivalent_usd')} wall={wall}s"
            )
            for mode in per_run_modes:
                nums = _mode_numbers(scored, mode)
                per_run_modes[mode].append(nums)
                print(f"  [{mode}] {json.dumps(nums)}")

        total_cost = sum(u["cost_estimate"] for u in per_run_usage)
        total_wall = sum(COLLECT_WALL_S.get((reviewer, i), 0.0) for i in run_indices)
        print(
            f"TOTAL across {len(per_run_usage)} run(s): cost=${total_cost:.4f} "
            f"wall={total_wall:.1f}s"
        )
        for mode, rows in per_run_modes.items():
            if not rows:
                continue
            recalls = [r["recall"] for r in rows]
            rates = [r["invented_rate"] for r in rows if r["invented_rate"] is not None]
            median_rate = statistics.median(rates) if rates else None
            # 2026-09-18 bench-provenance wave (01-spec/
            # bench-provenance-2026-09-18.md, FIX 2a): every aggregated
            # figure this script prints must say how many runs it's a
            # median of, right next to the number -- "median of 1 runs"
            # reads oddly but is exactly the honest label for n=1 (a
            # single-run "median" is just that run's own value, not a
            # real median; "(single run)" is used instead when n==1 to
            # say so plainly).
            basis = "single run" if len(recalls) == 1 else f"median of {len(recalls)} runs"
            print(
                f"  [{mode}] recall={statistics.median(recalls):.4f} "
                f"invented_rate={median_rate} ({basis})"
            )
    return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bench_two_pass")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--reviewer", required=True, choices=list(REVIEWER_FORKS))
    common.add_argument("--run-index", type=int, default=1)
    common.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    common.add_argument("--pass1-report", default=str(DEFAULT_PASS1_REPORT))
    common.add_argument("--corpus", default=str(DEFAULT_CORPUS_ROOT))
    common.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH))
    common.add_argument("--truth-table", default=str(DEFAULT_TRUTH_TABLE_PATH))

    collect_p = sub.add_parser("collect", parents=[common])
    collect_p.add_argument("--concurrency", type=int, default=8)
    # 4096 (the extractor's own default) was too small for a reviewer
    # call on a source with 20+ pass-1 facts: a reasoning-capable model
    # (moonshotai/kimi-k2.5, smoke-tested 2026-09-18 on S-call-004, 22
    # pass-1 facts) burned the entire budget and produced 62 malformed
    # verdict items rather than 22 real ones. 8192 fixed it with room to
    # spare (that same call used 337 real output tokens once given the
    # room).
    collect_p.add_argument("--max-tokens", type=int, default=8192)
    collect_p.add_argument("--source-ids", nargs="*", default=None)
    collect_p.set_defaults(func=cmd_collect)

    score_p = sub.add_parser("score", parents=[common])
    score_p.set_defaults(func=cmd_score)

    report_p = sub.add_parser("report")
    report_p.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    report_p.add_argument("--pass1-report", default=str(DEFAULT_PASS1_REPORT))
    report_p.set_defaults(func=cmd_report)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
