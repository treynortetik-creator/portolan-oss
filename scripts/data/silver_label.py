"""Silver-label the public corpora under `var/data/converted/`
(silver-labels build instructions, coordinator course-corrections
2026-09-21). Wires `portolan.training.pipeline` to real corpus
directories and real network clients -- every function that actually
does the work (document reading, gating, resumability, negative
capping, spend/budget tracking) lives in `portolan.training`, unit-
tested there with fakes; this script is the thin, network-touching
driver.

**Two teacher designs, selected by `--judge`.**

`--judge jev|kev|both` (default `jev`, the primary design as of
2026-09-21): teacher one is a single CHEAP PAID open-weight model,
picked live by price (`portolan.training.paid_teacher.
discover_cheapest_paid_teacher` -- `GET /api/v1/models`, cheapest
qualifying Qwen/DeepSeek/Llama/Mistral/GLM/Kimi id with a >32k context
window and structured-JSON support). Every fact it emits that clears the
mechanical quote gate goes to a typed judge (`portolan.training.
typed_judge`, Jev and/or Kev) instead of a second extraction pass. A
dollar spend meter (`portolan.training.spend.SpendMeter`,
`var/data/silver/spend.json`) tracks BOTH the extractor's real OpenRouter
cost and the judge's cost against one `--budget-usd` ceiling (default
$10); the run stops cleanly the moment it's spent.

`--judge none`: the original two-extractor agreement design
(`portolan.training.pipeline.run_corpus`, unchanged since it was first
built), now over two different cheap PAID open-weight models by default
(`_run_paid_agreement_mode`) -- the fallback the coordinator specified
2026-09-21 after both Jev and Kev scored under 85% on
`scripts/data/judge_calibration.py`'s real run (see that script's output
in this task's own report for the numbers and the harness-audit findings
behind them). Every configured judge still runs and its verdict is
recorded on every kept fact (`facts[].judge`), purely informationally --
it never gates keep/drop in this mode. `--judge none --free-teachers`
opts back into the ORIGINAL two-FREE-open-weight-teacher design
(rotation/cooldown/free-tier-budget machinery, `_run_free_agreement_mode`)
if free-tier congestion ever needs revisiting instead.

**API keys**, all read the same way (env, else `.env`, never printed --
`_env_or_dotenv`, reused from `corpus.generator.prose`):
`OPENROUTER_API_KEY` (both designs), `TYPESAFE_API_KEY` (`--judge jev`
or `both`), `KEV_BASE_URL`/`KEV_API_KEY` (`--judge kev` or `both` --
`docs/kev-local.md`, once documented, names Kev's local URL).

**Rate limits.** OpenRouter's `:free` cap is account-wide, shared across
every `:free` model: 20 requests/minute, 50 or 1000/day depending on
lifetime credits purchased (confirmed against openrouter.ai/docs/api-
reference/limits and this account's own live `GET /api/v1/auth/key`,
`portolan.training.free_tier`'s module docstring) -- still enforced this
way for `--judge none`. Paid models are rate-limited at 60 requests/
minute here (OpenRouter does not publish a fixed number for arbitrary
paid models; 60/min is this script's own conservative default, not a
number read from a header -- no header-reading channel exists at this
layer without deeper surgery into `OpenRouterClient`, noted as a known
gap).

Usage:
    uv run python scripts/data/silver_label.py --corpus fda_maude --max-docs 8 --budget-usd 0.50
    uv run python scripts/data/silver_label.py --corpus ami
    uv run python scripts/data/silver_label.py --corpus fda_maude --judge none  # two-free-teacher mode
    uv run python scripts/data/silver_label.py --stats
    uv run python scripts/data/silver_label.py --stats --corpus fda_maude
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from corpus.generator.prose import TokenBucketLimiter, _env_or_dotenv
from portolan.extract.backends import LLMClientBackend
from portolan.extract.prompt import DEFAULT_PROMPT_VARIANT, build_system_prompt
from portolan.llm.openrouter import OpenRouterClient
from portolan.metering import InMemorySink
from portolan.training.free_tier import (
    FREE_DAILY_LIMIT_HIGH_CREDIT,
    FREE_MINUTE_LIMIT,
    FreeTierBudget,
    default_budget_path,
    fetch_daily_limit,
    seconds_until_utc_midnight,
)
from portolan.training.local_teacher import is_local_teacher
from portolan.training.paid_teacher import (
    discover_cheapest_paid_teacher,
    discover_two_cheapest_paid_teachers,
)
from portolan.training.pipeline import (
    CORPUS_KINDS,
    CORPUS_LICENSES,
    DEFAULT_CHAT_WINDOW_SIZE,
    documents_for_corpus,
    run_corpus,
    run_corpus_judged,
    run_corpus_single,
)
from portolan.training.records import (
    compute_stats,
    dropped_by_corpus,
    dropped_path,
    failures_path,
    iter_all_dropped,
    iter_all_records,
    print_stats,
    progress_path,
    records_path,
)
from portolan.training.single_teacher import discover_single_teacher
from portolan.training.spend import SpendMeter, default_spend_path
from portolan.training.teachers import discover_open_weight_free_models
from portolan.training.typed_judge import JEV_URL_DEFAULT, JevClient
from scripts.data.silver_contexts import context_for_corpus

REPO_ROOT = Path(__file__).resolve().parents[2]
VAR_ROOT = REPO_ROOT / "var"
CORPORA = tuple(CORPUS_LICENSES.keys())
DEFAULT_BUDGET_USD = 10.00
PAID_TEACHER_RPM = (
    60  # coordinator: "60 requests per minute for paid models unless headers say otherwise"
)
KEV_LOCAL_URL = "http://127.0.0.1:8787/v1/systemone"
# scripts/data/overlap_calibration.py's real run, 2026-09-21: gold-keep-
# rate (95%) and every corruption's drop rate are FLAT across the whole
# 0.20-0.95 sweep -- the content-word threshold does no discriminating
# work on this calibration set at all; the numeral/name "must all
# appear" rule is what actually catches quote_swapped/added_detail.
# Coordinator course-correction 2026-09-21: since the threshold ties
# across the whole range, picking the strictest tied value (0.95) buys
# nothing on gold or on corruption-catching, and it cost 5 of 7 real
# facts in the first --mode single smoke run -- too strict for real
# extractor claims, which routinely restate a quote's content in
# different words. Set that day to the LEAST strict value in the tied
# range, 0.5.
#
# 2026-09-22 real Kimi K2 run on 12 SaaS sales calls (fix/overlap-gate-
# paraphrase, `01-spec/training-data-2026-09-19.md`'s "Kimi K2 smoke and
# gate finding"): even 0.5 dropped 23 of 49 emitted facts on
# `overlap_gate_content` alone, and every one read by hand was a
# faithful restatement of its quote -- chat-speak ("bc", "$299/month")
# and normal paraphrase the content-word bar was never calibrated
# against (`overlap_calibration.py`'s own cases are single-sentence
# gold-claim-plus-quote pairs, not real chat transcripts). The gate's
# OWN calibration table never showed this rule earning its keep: gold-
# keep-rate and every corruption's drop rate are flat across the entire
# 0.20-0.95 sweep, so no threshold in that range was ever doing
# discriminating work -- the numeral/name "must all appear" rule (never
# "mostly", unchanged here) is what actually catches
# quote_swapped/added_detail, and it alone is what the calibration table
# credits. Disabled by default (0.0 -- content_word_score's own local
# and full-text bars both pass trivially at 0.0, so the rule never
# fires unless a caller opts back in with `--overlap-threshold`).
DEFAULT_OVERLAP_THRESHOLD = 0.0

# Rotation width for the discovered open-weight list -- enough models to
# spread load across several free tiers without pulling in every `:free`
# id OpenRouter happens to host that day.
MAX_TEACHERS = 8

# `OpenRouterClient`'s own internal per-model retry budget for THIS
# pipeline only (default elsewhere in the codebase stays 5) -- confirmed
# live 2026-09-21 that a fully-exhausted 5-retry attempt against one
# throttled model burns ~70s before `portolan.training.rotation` ever
# gets a chance to move to a different model. One quick retry stays as a
# cheap safety net for a genuine one-off blip; rotation, not repeated
# same-model retry, is what actually helps under sustained congestion.
SILVER_LABEL_MAX_RETRIES = 1


def _print_summary_line(stats) -> None:
    print(
        f"  [{stats.corpus}] docs={stats.documents_processed} "
        f"envelopes={stats.envelopes_processed} kept={stats.facts_kept} "
        f"disagree={stats.facts_dropped_disagreement} "
        f"quote_gate={stats.facts_dropped_quote_gate} "
        f"negatives={stats.negatives_written} capped={stats.negatives_capped} "
        f"failures={stats.failures} congestion_pauses={stats.congestion_pauses} "
        f"requests_used_today={stats.requests_used_today}"
    )


def _print_judged_summary_line(stats) -> None:
    print(
        f"  [{stats.corpus}] docs={stats.documents_processed} "
        f"envelopes={stats.envelopes_processed} emitted={stats.facts_emitted} "
        f"quote_gate_dropped={stats.facts_dropped_quote_gate} "
        f"sent_to_judge={stats.facts_sent_to_judge} kept={stats.facts_kept} "
        f"judge_dropped={stats.facts_dropped_judge} negatives={stats.negatives_written} "
        f"capped={stats.negatives_capped} failures={stats.failures} "
        f"spend_usd={stats.total_spend_usd:.4f}"
    )


def _run_stats(args: argparse.Namespace) -> int:
    corpora = args.corpus or None
    records = iter_all_records(VAR_ROOT, corpora)
    dropped = iter_all_dropped(VAR_ROOT, corpora)
    stats = compute_stats(records, dropped)
    print_stats(stats, dropped_by_corpus(VAR_ROOT, corpora))
    return 0


def _run_judged_mode(args: argparse.Namespace, corpus: str) -> int:
    openrouter_key = _env_or_dotenv(REPO_ROOT, "OPENROUTER_API_KEY")
    if not openrouter_key:
        print(
            "error: OPENROUTER_API_KEY not set in the environment or .env "
            "(portolan.config.Settings.openrouter_api_key)",
            file=sys.stderr,
        )
        return 2

    # Every configured judge is queried and recorded on every fact
    # (coordinator: "Record BOTH verdicts ... when both endpoints are
    # configured") -- Jev is configured whenever `--judge` names it;
    # Kev is configured whenever `KEV_BASE_URL` is set, REGARDLESS of
    # `--judge`, so a `--judge jev` run still records Kev's verdict for
    # free comparison once Kev exists locally. `--judge` only picks
    # which configured judge(s) the KEEP decision requires.
    judges: dict[str, JevClient] = {}
    if args.judge in ("jev", "both"):
        jev_key = _env_or_dotenv(REPO_ROOT, "TYPESAFE_API_KEY")
        if not jev_key:
            print("error: TYPESAFE_API_KEY not set in the environment or .env", file=sys.stderr)
            return 2
        judges["jev"] = JevClient(api_key=jev_key, base_url=JEV_URL_DEFAULT)

    kev_base_url = _env_or_dotenv(REPO_ROOT, "KEV_BASE_URL")
    if kev_base_url:
        kev_key = _env_or_dotenv(REPO_ROOT, "KEV_API_KEY")  # may be None for a local Kev
        judges["kev"] = JevClient(api_key=kev_key, base_url=kev_base_url)
    elif args.judge in ("kev", "both"):
        print(
            "error: --judge kev/both needs KEV_BASE_URL (docs/kev-local.md, once documented)",
            file=sys.stderr,
        )
        return 2

    if args.judge not in judges and args.judge != "both":
        print(f"error: --judge {args.judge!r} has no configured endpoint", file=sys.stderr)
        return 2

    print(f"discovering cheapest paid open-weight teacher for corpus={corpus!r}...")
    candidate = discover_cheapest_paid_teacher(openrouter_key)
    if candidate is None:
        print(
            "error: no qualifying paid open-weight model found (Qwen/DeepSeek/Llama/Mistral/"
            "GLM/Kimi, >32k context, structured JSON output)",
            file=sys.stderr,
        )
        return 2
    print(
        f"  paid teacher: {candidate.model_id}  "
        f"prompt=${candidate.prompt_price_per_token * 1_000_000:.4f}/M tokens  "
        f"completion=${candidate.completion_price_per_token * 1_000_000:.4f}/M tokens  "
        f"context={candidate.context_length}"
    )

    client = OpenRouterClient(
        spend_key=corpus,
        api_key=openrouter_key,
        default_model=candidate.model_id,
        sink=InMemorySink(),
    )
    try:
        client.warm_json_strategy(candidate.model_id)
    except Exception as exc:  # noqa: BLE001 - warming is best-effort
        print(f"  warm_json_strategy failed, continuing: {exc}")

    system_prompt = build_system_prompt(
        company_context=context_for_corpus(corpus), variant=DEFAULT_PROMPT_VARIANT
    )
    limiter = TokenBucketLimiter(requests_per_minute=PAID_TEACHER_RPM, requests_per_day=10**9)

    def backend_factory(model: str) -> LLMClientBackend:
        return LLMClientBackend(client, model=model, surface="extract")

    converted_root = VAR_ROOT / "data" / "converted" / corpus
    documents = documents_for_corpus(corpus, converted_root, window_size=args.chat_window)

    spend = SpendMeter(path=default_spend_path(VAR_ROOT))
    print(
        f"  spend so far (all corpora, this meter): ${spend.total_usd():.4f} / ${args.budget_usd:.2f}"
    )

    records_file = records_path(VAR_ROOT, corpus)
    failures_file = failures_path(VAR_ROOT, corpus)
    progress_file = progress_path(VAR_ROOT, corpus)

    print(
        f"running corpus={corpus!r} max_docs={args.max_docs} judge={args.judge!r} "
        f"judges_configured={list(judges)} budget_usd={args.budget_usd} -> {records_file}"
    )
    stats = run_corpus_judged(
        corpus=corpus,
        kind=CORPUS_KINDS[corpus],
        license_text=CORPUS_LICENSES[corpus],
        documents=documents,
        system_prompt=system_prompt,
        prompt_variant=DEFAULT_PROMPT_VARIANT,
        paid_model=candidate.model_id,
        backend_factory=backend_factory,
        limiter=limiter,
        sink=client.sink,
        tenant="silver-label",
        records_file=records_file,
        failures_file=failures_file,
        progress_file=progress_file,
        judges=judges,
        judge_mode=args.judge,
        spend=spend,
        budget_usd=args.budget_usd,
        max_docs=args.max_docs,
        sleep=time.sleep,
        on_summary=_print_judged_summary_line,
    )

    print(
        f"done: corpus={corpus} documents_processed={stats.documents_processed} "
        f"documents_skipped_resumed={stats.documents_skipped_resumed} "
        f"envelopes={stats.envelopes_processed} facts_emitted={stats.facts_emitted} "
        f"facts_dropped_quote_gate={stats.facts_dropped_quote_gate} "
        f"facts_sent_to_judge={stats.facts_sent_to_judge} facts_kept={stats.facts_kept} "
        f"facts_dropped_judge={stats.facts_dropped_judge} "
        f"records_written={stats.records_written} negatives_written={stats.negatives_written} "
        f"negatives_capped={stats.negatives_capped} failures={stats.failures} "
        f"extractor_spend_usd={stats.extractor_spend_usd:.4f} "
        f"judge_spend_usd={stats.judge_spend_usd:.4f} "
        f"total_spend_usd={stats.total_spend_usd:.4f} stopped_reason={stats.stopped_reason}"
    )
    if stats.stopped_reason == "budget_usd":
        print(f"  budget spent: ${stats.total_spend_usd:.4f} >= ${args.budget_usd:.2f} ceiling")
    return 0


def _run_free_agreement_mode(args: argparse.Namespace, corpus: str) -> int:
    """The original two-FREE-open-weight-teacher agreement design.
    Opt-in via `--judge none --free-teachers`; plain `--judge none`
    defaults to `_run_paid_agreement_mode` instead (coordinator course-
    correction, 2026-09-21, item 3)."""
    api_key = _env_or_dotenv(REPO_ROOT, "OPENROUTER_API_KEY")
    if not api_key:
        print(
            "error: OPENROUTER_API_KEY not set in the environment or .env "
            "(portolan.config.Settings.openrouter_api_key)",
            file=sys.stderr,
        )
        return 2

    print(f"discovering free open-weight OpenRouter models for corpus={corpus!r}...")
    models = discover_open_weight_free_models(api_key)[:MAX_TEACHERS]
    if len(models) < 2:
        print(
            f"error: only {len(models)} open-weight ':free' model(s) discovered; need at "
            "least two from different families -- check OPENROUTER_API_KEY and network access",
            file=sys.stderr,
        )
        return 2
    print(f"  teachers: {models}")

    daily_limit = fetch_daily_limit(api_key)
    if daily_limit is None:
        print(
            "  could not confirm the live daily :free budget "
            f"(GET /api/v1/auth/key) -- assuming {FREE_DAILY_LIMIT_HIGH_CREDIT}/day"
        )
        daily_limit = FREE_DAILY_LIMIT_HIGH_CREDIT
    else:
        print(f"  confirmed account-wide daily :free budget: {daily_limit}/day")
    budget = FreeTierBudget(path=default_budget_path(VAR_ROOT), daily_limit=daily_limit)
    print(
        f"  requests used today (account-wide, every :free caller): "
        f"{budget.used_today()}/{daily_limit}"
    )

    client = OpenRouterClient(
        spend_key=corpus,
        api_key=api_key,
        default_model=models[0],
        sink=InMemorySink(),
        max_retries=SILVER_LABEL_MAX_RETRIES,
    )
    for model in models:
        try:
            client.warm_json_strategy(model)
        except Exception as exc:  # noqa: BLE001 - warming is best-effort
            print(f"  warm_json_strategy({model!r}) failed, continuing: {exc}")

    system_prompt = build_system_prompt(
        company_context=context_for_corpus(corpus), variant=DEFAULT_PROMPT_VARIANT
    )
    # The account-wide minute cap (20/min, FREE_MINUTE_LIMIT) is enforced
    # in-process here; the account-wide DAILY cap is `budget` above,
    # file-locked and shared across processes -- `requests_per_day` is
    # set effectively unlimited so `TokenBucketLimiter` never raises its
    # own `DailyLimitExceeded` and short-circuits `budget`'s real
    # accounting.
    limiter = TokenBucketLimiter(requests_per_minute=args.rpm, requests_per_day=10**9)

    def backend_factory(model: str) -> LLMClientBackend:
        return LLMClientBackend(client, model=model, surface="extract")

    converted_root = VAR_ROOT / "data" / "converted" / corpus
    documents = documents_for_corpus(corpus, converted_root, window_size=args.chat_window)

    records_file = records_path(VAR_ROOT, corpus)
    failures_file = failures_path(VAR_ROOT, corpus)
    progress_file = progress_path(VAR_ROOT, corpus)

    print(f"running corpus={corpus!r} max_docs={args.max_docs} -> {records_file}")
    stats = run_corpus(
        corpus=corpus,
        kind=CORPUS_KINDS[corpus],
        license_text=CORPUS_LICENSES[corpus],
        documents=documents,
        system_prompt=system_prompt,
        prompt_variant=DEFAULT_PROMPT_VARIANT,
        models=models,
        backend_factory=backend_factory,
        limiter=limiter,
        budget=budget,
        sink=client.sink,
        tenant="silver-label",
        records_file=records_file,
        failures_file=failures_file,
        progress_file=progress_file,
        max_docs=args.max_docs,
        sleep=time.sleep,
        on_summary=_print_summary_line,
    )

    print(
        f"done: corpus={corpus} documents_processed={stats.documents_processed} "
        f"documents_skipped_resumed={stats.documents_skipped_resumed} "
        f"envelopes={stats.envelopes_processed} records_written={stats.records_written} "
        f"facts_kept={stats.facts_kept} "
        f"facts_dropped_disagreement={stats.facts_dropped_disagreement} "
        f"facts_dropped_quote_gate={stats.facts_dropped_quote_gate} "
        f"negatives_written={stats.negatives_written} negatives_capped={stats.negatives_capped} "
        f"failures={stats.failures} congestion_pauses={stats.congestion_pauses} "
        f"stopped_reason={stats.stopped_reason} "
        f"requests_used_today={stats.requests_used_today}/{daily_limit} "
        f"teacher_pairs_used={dict(stats.teacher_pairs_used)} "
        f"rate_limit_events_per_model={dict(stats.rate_limit_events)}"
    )
    if stats.stopped_reason == "daily_budget":
        reset_seconds = seconds_until_utc_midnight()
        print(
            f"  daily :free budget spent ({daily_limit}/day, account-wide) -- resets in "
            f"{reset_seconds / 3600:.1f}h (next UTC midnight)"
        )
    return 0


def _attach_judge_verdicts(
    records_file: Path, judges: dict[str, JevClient], start_line: int
) -> tuple[int, float]:
    """Post-hoc, informational only: for every record this run appended
    (lines `[start_line:]` -- lines already on disk before the run are
    left untouched), ask every configured judge the three typed
    questions for each of the record's facts and attach the verdict(s)
    to `TrainingFact.judge` -- never changes which facts were kept
    (coordinator: "Keep the judge code and record its verdicts in the
    records anyway ... so we learn from it without gating on it").
    Returns `(facts_judged, judge_spend_usd)`."""
    from portolan.training.records import TrainingRecord
    from portolan.training.typed_judge import ask_typed_questions, fact_type_criteria

    if not records_file.exists():
        return 0, 0.0
    lines = records_file.read_text(encoding="utf-8").splitlines()
    criteria = fact_type_criteria()
    facts_judged = 0
    judge_spend = 0.0

    for i in range(start_line, len(lines)):
        if not lines[i].strip():
            continue
        record = TrainingRecord.model_validate_json(lines[i])
        if not record.facts:
            continue
        record_dict = record.model_dump()
        for fact_dict, fact in zip(record_dict["facts"], record.facts, strict=True):
            verdicts: dict[str, dict] = {}
            for judge_name, client in judges.items():
                verdict = ask_typed_questions(
                    client, fact.claim, fact.quote or "", criteria=criteria
                )
                verdicts[judge_name] = verdict.to_record_dict()
                cost = verdict.cost_usd()
                judge_spend += cost
            fact_dict["judge"] = verdicts
            facts_judged += 1
        lines[i] = TrainingRecord.model_validate(record_dict).model_dump_json()

    records_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return facts_judged, judge_spend


def _run_paid_agreement_mode(args: argparse.Namespace, corpus: str) -> int:
    """Fallback smoke design (coordinator course-correction, 2026-09-21,
    item 3, after both Jev and Kev scored under 85% on
    `scripts/data/judge_calibration.py`): the ORIGINAL two-extractor
    agreement gate (`portolan.training.pipeline.run_corpus`, unchanged),
    but with two different cheap PAID open-weight models
    (`portolan.training.paid_teacher.discover_two_cheapest_paid_teachers`)
    instead of two free ones -- no free-tier daily budget applies, so
    `budget=None`; every real OpenRouter cost is read back off `sink.
    events` and recorded into the same `SpendMeter` the judged mode uses,
    so `--budget-usd` still means something (checked after the run
    completes, then printed as a cost-per-document projection -- this
    mode does not interrupt mid-run the way `run_corpus_judged` does,
    since `run_corpus` has no budget-aware stop condition to hook into).

    Every configured judge (Jev/Kev) still runs, but purely
    informationally: `_attach_judge_verdicts` asks it about every kept
    fact AFTER the agreement gate has already decided what to keep, and
    records the verdict without ever using it to drop anything.
    """
    openrouter_key = _env_or_dotenv(REPO_ROOT, "OPENROUTER_API_KEY")
    if not openrouter_key:
        print(
            "error: OPENROUTER_API_KEY not set in the environment or .env "
            "(portolan.config.Settings.openrouter_api_key)",
            file=sys.stderr,
        )
        return 2

    print(f"discovering two cheapest different-family paid open-weight teachers for {corpus!r}...")
    pair = discover_two_cheapest_paid_teachers(openrouter_key)
    if pair is None:
        print(
            "error: could not find two paid open-weight models from different families "
            "(Qwen/DeepSeek/Llama/Mistral/GLM/Kimi, >32k context, structured JSON output)",
            file=sys.stderr,
        )
        return 2
    model_a, model_b = pair
    print(
        f"  teacher A: {model_a.model_id}  prompt=${model_a.prompt_price_per_token * 1_000_000:.4f}/M"
    )
    print(
        f"  teacher B: {model_b.model_id}  prompt=${model_b.prompt_price_per_token * 1_000_000:.4f}/M"
    )

    client = OpenRouterClient(
        spend_key=corpus,
        api_key=openrouter_key,
        default_model=model_a.model_id,
        sink=InMemorySink(),
    )
    for model_id in (model_a.model_id, model_b.model_id):
        try:
            client.warm_json_strategy(model_id)
        except Exception as exc:  # noqa: BLE001 - warming is best-effort
            print(f"  warm_json_strategy({model_id!r}) failed, continuing: {exc}")

    system_prompt = build_system_prompt(
        company_context=context_for_corpus(corpus), variant=DEFAULT_PROMPT_VARIANT
    )
    limiter = TokenBucketLimiter(requests_per_minute=PAID_TEACHER_RPM, requests_per_day=10**9)

    def backend_factory(model: str) -> LLMClientBackend:
        return LLMClientBackend(client, model=model, surface="extract")

    converted_root = VAR_ROOT / "data" / "converted" / corpus
    documents = documents_for_corpus(corpus, converted_root, window_size=args.chat_window)

    records_file = records_path(VAR_ROOT, corpus)
    failures_file = failures_path(VAR_ROOT, corpus)
    progress_file = progress_path(VAR_ROOT, corpus)
    lines_before = (
        len(records_file.read_text(encoding="utf-8").splitlines()) if records_file.exists() else 0
    )
    events_before = len(client.sink.events)

    print(
        f"running corpus={corpus!r} max_docs={args.max_docs} judge=none (paid agreement) -> {records_file}"
    )
    stats = run_corpus(
        corpus=corpus,
        kind=CORPUS_KINDS[corpus],
        license_text=CORPUS_LICENSES[corpus],
        documents=documents,
        system_prompt=system_prompt,
        prompt_variant=DEFAULT_PROMPT_VARIANT,
        models=[model_a.model_id, model_b.model_id],
        backend_factory=backend_factory,
        limiter=limiter,
        budget=None,
        sink=client.sink,
        tenant="silver-label",
        records_file=records_file,
        failures_file=failures_file,
        progress_file=progress_file,
        max_docs=args.max_docs,
        sleep=time.sleep,
        on_summary=_print_summary_line,
    )

    extractor_spend = sum((e.cost_estimate or 0.0) for e in client.sink.events[events_before:])
    spend = SpendMeter(path=default_spend_path(VAR_ROOT))
    if extractor_spend > 0:
        spend.record(
            model=f"{model_a.model_id}+{model_b.model_id}",
            corpus=corpus,
            amount_usd=extractor_spend,
        )

    judges: dict[str, JevClient] = {}
    jev_key = _env_or_dotenv(REPO_ROOT, "TYPESAFE_API_KEY")
    if jev_key:
        judges["jev"] = JevClient(api_key=jev_key, base_url=JEV_URL_DEFAULT)
    kev_base_url = _env_or_dotenv(REPO_ROOT, "KEV_BASE_URL")
    if kev_base_url:
        judges["kev"] = JevClient(
            api_key=_env_or_dotenv(REPO_ROOT, "KEV_API_KEY"), base_url=kev_base_url
        )

    facts_judged = 0
    judge_spend = 0.0
    if judges:
        print(
            f"  attaching informational judge verdicts from {list(judges)} (not gating keep/drop)..."
        )
        facts_judged, judge_spend = _attach_judge_verdicts(records_file, judges, lines_before)
        if judge_spend > 0:
            spend.record(model="judge:info", corpus=corpus, amount_usd=judge_spend)

    total_spend = extractor_spend + judge_spend
    docs = stats.documents_processed
    cost_per_doc = total_spend / docs if docs else 0.0
    projected_docs = (args.budget_usd / cost_per_doc) if cost_per_doc > 0 else float("inf")

    print(
        f"done: corpus={corpus} documents_processed={docs} "
        f"documents_skipped_resumed={stats.documents_skipped_resumed} "
        f"envelopes={stats.envelopes_processed} records_written={stats.records_written} "
        f"facts_kept={stats.facts_kept} "
        f"facts_dropped_disagreement={stats.facts_dropped_disagreement} "
        f"facts_dropped_quote_gate={stats.facts_dropped_quote_gate} "
        f"negatives_written={stats.negatives_written} negatives_capped={stats.negatives_capped} "
        f"failures={stats.failures} facts_judged_informationally={facts_judged} "
        f"extractor_spend_usd={extractor_spend:.6f} judge_spend_usd={judge_spend:.6f} "
        f"total_spend_usd={total_spend:.6f} cost_per_doc_usd={cost_per_doc:.6f}"
    )
    print(
        f"  at this cost/doc, a ${args.budget_usd:.2f} budget buys roughly {projected_docs:.0f} documents"
    )
    return 0


def _print_single_summary_line(stats) -> None:
    print(
        f"  [{stats.corpus}] docs={stats.documents_processed} "
        f"envelopes={stats.envelopes_processed} emitted={stats.facts_emitted} "
        f"quote_gate={stats.facts_dropped_quote_gate} overlap_dropped={stats.facts_dropped_overlap} "
        f"quote_not_exact={stats.facts_dropped_quote_not_exact} "
        f"kept={stats.facts_kept} negatives={stats.negatives_written} "
        f"capped={stats.negatives_capped} failures={stats.failures} "
        f"spend_usd={stats.total_spend_usd:.4f}"
    )


def _run_single_local(args: argparse.Namespace, corpus: str) -> int:
    """`--mode single --teacher local:<repo id>`: the same run as
    `_run_single_mode`, with the production prompt answered by a local MLX
    model (`portolan.training.local_teacher`). No key, no budget (each call
    writes a $0 spend line naming the model), no rate limit, and no Kev
    judge: Kev is another local model, and this Mac holds one at a time.
    Wall time per document goes to `timings.jsonl` next to the records."""
    from portolan.llm.mlx_local import mlx_available
    from portolan.training.local_teacher import canonical_local_teacher, make_local_client

    try:
        teacher_model = canonical_local_teacher(args.teacher)
        client = make_local_client(
            teacher_model, sink=InMemorySink(), tenant="silver-label", progress=True
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not mlx_available():
        print("error: local teachers need mlx and mlx-lm: `uv sync --group mlx`", file=sys.stderr)
        return 2
    print(f"using local teacher: {teacher_model!r} (MLX, in-process)")

    system_prompt = build_system_prompt(
        company_context=context_for_corpus(corpus), variant=DEFAULT_PROMPT_VARIANT
    )
    limiter = TokenBucketLimiter(requests_per_minute=10**9, requests_per_day=10**9)

    def backend_factory(model: str) -> LLMClientBackend:
        return LLMClientBackend(client, model=client.model_id, surface="extract")

    var_root = VAR_ROOT
    converted_root = var_root / "data" / "converted" / corpus
    documents = documents_for_corpus(corpus, converted_root, window_size=args.chat_window)
    records_file = records_path(var_root, corpus)

    print(
        f"running corpus={corpus!r} max_docs={args.max_docs} mode=single teacher=local "
        f"overlap_threshold={args.overlap_threshold} -> {records_file}"
    )
    stats = run_corpus_single(
        corpus=corpus,
        kind=CORPUS_KINDS[corpus],
        license_text=CORPUS_LICENSES[corpus],
        documents=documents,
        system_prompt=system_prompt,
        prompt_variant=DEFAULT_PROMPT_VARIANT,
        teacher_model=teacher_model,
        overlap_threshold=args.overlap_threshold,
        backend_factory=backend_factory,
        limiter=limiter,
        sink=client.sink,
        tenant="silver-label",
        records_file=records_file,
        failures_file=failures_path(var_root, corpus),
        progress_file=progress_path(var_root, corpus),
        dropped_file=dropped_path(var_root, corpus),
        spend=SpendMeter(path=default_spend_path(var_root)),
        budget_usd=float("inf"),
        judges={},
        max_docs=args.max_docs,
        sleep=time.sleep,
        on_summary=_print_single_summary_line,
        timings_file=records_file.parent / "timings.jsonl",
    )
    print(
        f"done: corpus={corpus} teacher={teacher_model} "
        f"documents_processed={stats.documents_processed} "
        f"documents_skipped_resumed={stats.documents_skipped_resumed} "
        f"envelopes={stats.envelopes_processed} facts_emitted={stats.facts_emitted} "
        f"facts_dropped_quote_gate={stats.facts_dropped_quote_gate} "
        f"facts_dropped_overlap={stats.facts_dropped_overlap} "
        f"facts_dropped_quote_not_exact={stats.facts_dropped_quote_not_exact} "
        f"facts_kept={stats.facts_kept} records_written={stats.records_written} "
        f"negatives_written={stats.negatives_written} failures={stats.failures} "
        f"peak_memory_gb={client.peak_memory_bytes() / 1e9:.2f} "
        f"stopped_reason={stats.stopped_reason}"
    )
    return 0


def _run_single_mode(args: argparse.Namespace, corpus: str) -> int:
    """`--mode single` (coordinator course-correction, 2026-09-21, final
    design): one teacher, mechanical overlap gate
    (`portolan.training.overlap_gate.passes_overlap_gate`, threshold from
    `scripts/data/overlap_calibration.py`'s real run), Kev recorded on
    every kept fact informationally, Jev skipped entirely to save its
    (real, billed) cost."""
    if args.teacher and is_local_teacher(args.teacher):
        return _run_single_local(args, corpus)
    openrouter_key = _env_or_dotenv(REPO_ROOT, "OPENROUTER_API_KEY")
    if not openrouter_key:
        print(
            "error: OPENROUTER_API_KEY not set in the environment or .env "
            "(portolan.config.Settings.openrouter_api_key)",
            file=sys.stderr,
        )
        return 2

    if args.teacher:
        teacher_model = args.teacher
        print(f"using override teacher: {teacher_model!r}")
    else:
        print(
            "discovering single teacher (deepseek-v3.x > qwen3-235b-a22b > kimi-k2, cheapest under budget)..."
        )
        candidate = discover_single_teacher(openrouter_key)
        if candidate is None:
            print(
                "error: no qualifying single teacher under "
                "$0.0015/doc (deepseek-v3.x/qwen3-235b-a22b/kimi-k2) -- pass --teacher",
                file=sys.stderr,
            )
            return 2
        teacher_model = candidate.model_id
        print(
            f"  teacher: {teacher_model}  "
            f"prompt=${candidate.prompt_price_per_token * 1_000_000:.4f}/M  "
            f"completion=${candidate.completion_price_per_token * 1_000_000:.4f}/M"
        )

    # Coordinator item 3: "store Kev's three probabilities in judge.kev
    # ... skip Jev for the run to save its cost." Kev only, unconditional
    # (not gated by --judge, which --mode single doesn't read at all).
    judges: dict[str, JevClient] = {"kev": JevClient(api_key=None, base_url=KEV_LOCAL_URL)}

    client = OpenRouterClient(
        spend_key=corpus, api_key=openrouter_key, default_model=teacher_model, sink=InMemorySink()
    )
    try:
        client.warm_json_strategy(teacher_model)
    except Exception as exc:  # noqa: BLE001 - warming is best-effort
        print(f"  warm_json_strategy failed, continuing: {exc}")

    system_prompt = build_system_prompt(
        company_context=context_for_corpus(corpus), variant=DEFAULT_PROMPT_VARIANT
    )
    limiter = TokenBucketLimiter(requests_per_minute=PAID_TEACHER_RPM, requests_per_day=10**9)

    def backend_factory(model: str) -> LLMClientBackend:
        return LLMClientBackend(client, model=model, surface="extract")

    converted_root = VAR_ROOT / "data" / "converted" / corpus
    documents = documents_for_corpus(corpus, converted_root, window_size=args.chat_window)

    spend = SpendMeter(path=default_spend_path(VAR_ROOT))
    records_file = records_path(VAR_ROOT, corpus)
    failures_file = failures_path(VAR_ROOT, corpus)
    progress_file = progress_path(VAR_ROOT, corpus)
    dropped_file = dropped_path(VAR_ROOT, corpus)

    print(
        f"running corpus={corpus!r} max_docs={args.max_docs} mode=single "
        f"overlap_threshold={args.overlap_threshold} budget_usd={args.budget_usd} -> {records_file}"
    )
    stats = run_corpus_single(
        corpus=corpus,
        kind=CORPUS_KINDS[corpus],
        license_text=CORPUS_LICENSES[corpus],
        documents=documents,
        system_prompt=system_prompt,
        prompt_variant=DEFAULT_PROMPT_VARIANT,
        teacher_model=teacher_model,
        overlap_threshold=args.overlap_threshold,
        backend_factory=backend_factory,
        limiter=limiter,
        sink=client.sink,
        tenant="silver-label",
        records_file=records_file,
        failures_file=failures_file,
        progress_file=progress_file,
        dropped_file=dropped_file,
        spend=spend,
        budget_usd=args.budget_usd,
        judges=judges,
        max_docs=args.max_docs,
        sleep=time.sleep,
        on_summary=_print_single_summary_line,
    )

    docs = stats.documents_processed
    cost_per_doc = stats.total_spend_usd / docs if docs else 0.0
    print(
        f"done: corpus={corpus} documents_processed={docs} "
        f"documents_skipped_resumed={stats.documents_skipped_resumed} "
        f"envelopes={stats.envelopes_processed} facts_emitted={stats.facts_emitted} "
        f"facts_dropped_quote_gate={stats.facts_dropped_quote_gate} "
        f"facts_dropped_overlap={stats.facts_dropped_overlap} "
        f"facts_dropped_quote_not_exact={stats.facts_dropped_quote_not_exact} "
        f"facts_kept={stats.facts_kept} "
        f"facts_kept_by_type={dict(stats.facts_kept_by_type)} "
        f"records_written={stats.records_written} negatives_written={stats.negatives_written} "
        f"negatives_capped={stats.negatives_capped} failures={stats.failures} "
        f"extractor_spend_usd={stats.extractor_spend_usd:.6f} "
        f"judge_spend_usd={stats.judge_spend_usd:.6f} total_spend_usd={stats.total_spend_usd:.6f} "
        f"cost_per_doc_usd={cost_per_doc:.6f} stopped_reason={stats.stopped_reason}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    global VAR_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        action="append",
        choices=CORPORA,
        help="corpus to run (required unless --stats); --stats may repeat this to filter",
    )
    parser.add_argument("--max-docs", type=int, default=None, help="stop after N new documents")
    parser.add_argument("--stats", action="store_true", help="print stats and exit")
    parser.add_argument(
        "--mode",
        choices=["agreement", "single"],
        default="agreement",
        help=(
            "agreement (default): --judge decides jev/kev/both-gated or two-extractor-agreement "
            "(--judge none). single: one teacher, mechanical overlap gate, Kev recorded "
            "informationally -- the primary design as of 2026-09-21 (neither the typed judge nor "
            "cross-extractor agreement reliably gated a kept fact in real runs)."
        ),
    )
    parser.add_argument(
        "--teacher",
        default=None,
        help="--mode single only: override model id (default: cheapest of "
        "deepseek-v3.x/qwen3-235b-a22b/kimi-k2 under $0.0015/doc). "
        "`local:<hugging face repo id>` runs a local MLX model listed in "
        "portolan.training.local_teacher.LOCAL_MODELS instead of OpenRouter",
    )
    parser.add_argument(
        "--var-root",
        type=Path,
        default=VAR_ROOT,
        help="root holding data/converted/ (input) and data/silver/ (records, spend meter)",
    )
    parser.add_argument(
        "--overlap-threshold",
        type=float,
        default=DEFAULT_OVERLAP_THRESHOLD,
        help=f"--mode single only: mechanical gate threshold (default {DEFAULT_OVERLAP_THRESHOLD}, "
        "from scripts/data/overlap_calibration.py's real run)",
    )
    parser.add_argument(
        "--judge",
        choices=["jev", "kev", "both", "none"],
        default="jev",
        help=(
            "--mode agreement only. jev|kev|both: paid-teacher-plus-typed-judge mode, judge(s) "
            "gate the keep decision. none: two-extractor agreement mode (paid models unless "
            "--free-teachers)."
        ),
    )
    parser.add_argument(
        "--free-teachers",
        action="store_true",
        help="with --judge none, use the original two-FREE-open-weight-teacher agreement mode",
    )
    parser.add_argument(
        "--budget-usd",
        type=float,
        default=DEFAULT_BUDGET_USD,
        help="dollar spend ceiling (paid-teacher/judge cost, shared meter) -- default $10.00",
    )
    parser.add_argument(
        "--rpm",
        type=int,
        default=FREE_MINUTE_LIMIT,
        help="account-wide requests/minute budget for --free-teachers (OpenRouter's :free cap is 20)",
    )
    parser.add_argument(
        "--chat-window",
        type=int,
        default=DEFAULT_CHAT_WINDOW_SIZE,
        help="messages per ubuntu_irc document window",
    )
    args = parser.parse_args(argv)
    VAR_ROOT = args.var_root

    if args.stats:
        return _run_stats(args)

    corpus = args.corpus[0] if args.corpus else None
    if not corpus:
        print("error: --corpus NAME is required outside --stats mode", file=sys.stderr)
        return 2
    if corpus not in CORPORA:
        print(f"error: unknown corpus {corpus!r}; choose one of {sorted(CORPORA)}", file=sys.stderr)
        return 2

    if args.mode == "single":
        return _run_single_mode(args, corpus)
    if args.judge == "none":
        if args.free_teachers:
            return _run_free_agreement_mode(args, corpus)
        return _run_paid_agreement_mode(args, corpus)
    return _run_judged_mode(args, corpus)


if __name__ == "__main__":
    sys.exit(main())
