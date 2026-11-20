"""GPT-6 Luna evaluation, 2026-09-23 (`01-spec/bench-gpt6-luna-2026-09-23.md`).

Two measurements, each metered under corpus key `luna_eval` with a hard cap:

- `exam`: the Halvard extraction exam on the fixed 24-source slice (the
  fine-tune shakedown's slice, 29 planted facts), neutral production prompt,
  no company context (the bench default). Same run and scoring code as
  `python -m portolan.bench extract`; adds a spend cap, reasoning effort,
  service tier, a larger output cap, per-call records, and `--batch`, which
  sends the slice through OpenRouter's Batch API instead of the sync API.
- `labels`: the label-judge hand-verdict set (27 documents, 103 teacher facts
  with by-eye verdicts, `var/data/label_judge/2026-09-22/`). The model labels
  the same documents with the exact prompt the teacher saw; its facts are
  matched to the verdicted facts by quote span.

`table` and `label-table` aggregate saved runs. Nothing here writes a label
or a training record: the provenance rule (CLAUDE.md) keeps closed-model text
out of training data, and these outputs live under `var/bench/`.

Usage:
    uv run python scripts/bench/luna_eval.py exam --model openai/gpt-6-luna-pro \\
        --effort max --max-tokens 32000 --out var/bench/luna-2026-09-23/pro-max
    uv run python scripts/bench/luna_eval.py exam --model openai/gpt-6-luna-pro \\
        --effort max --batch --out var/bench/luna-2026-09-23/pro-max-batch
    uv run python scripts/bench/luna_eval.py labels --model openai/gpt-6-luna \\
        --out var/bench/luna-2026-09-23/labels-luna
    uv run python scripts/bench/luna_eval.py table --run kimi=<dir> --run luna=<dir>
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import statistics
import sys
import threading
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from portolan.extract.backends import LLMClientBackend, _build_retry_message  # noqa: E402
from portolan.extract.prompt import build_system_prompt, build_user_message  # noqa: E402
from portolan.extract.schema import (  # noqa: E402
    Coverage,
    ExtractionResult,
    SourceEnvelope,
    extraction_result_json_schema,
)
from portolan.llm.json_util import extract_json_object  # noqa: E402
from portolan.llm.openrouter import OpenRouterClient  # noqa: E402
from portolan.llm.openrouter_batch import (  # noqa: E402
    BATCH_SUFFIX,
    OpenRouterBatchClient,
    batch_results,
    chat_request_body,
)
from portolan.metering import InMemorySink, UsageEvent, UsageSink  # noqa: E402
from portolan.training.spend import SpendMeter, default_spend_path, shared_var_root  # noqa: E402

SPEND_KEY = "luna_eval"
DEFAULT_CAP_USD = 6.00
MODELS_URL = "https://openrouter.ai/api/v1/models"
# Characters per token for the pre-call estimate: below the ~4 of English text, so the
# estimate errs high.
CHARS_PER_TOKEN = 3.0

# The fine-tune shakedown's fixed slice (01-spec/finetune-shakedown-2026-09-22.md).
SLICE = [
    "S-call-002", "S-call-003", "S-call-004", "S-call-005",
    "S-call-006", "S-call-007", "S-call-008", "S-call-009",
    "S-email-005", "S-email-021", "S-email-021-thread", "S-email-030",
    "S-email-040", "S-email-052", "S-email-058", "S-email-061",
    "S-ticket-019", "S-ticket-024", "S-ticket-027", "S-ticket-031",
    "S-ticket-038", "S-ticket-039", "S-ticket-040", "S-ticket-041",
]  # fmt: skip


class BudgetExceeded(RuntimeError):
    """A call's worst case would take `luna_eval` spend past the cap."""


# --------------------------------------------------------------------------
# Key, prices, budget
# --------------------------------------------------------------------------


def openrouter_key() -> str:
    """The key from the environment or this checkout's `.env`, else the main
    checkout's `.env` (a worktree has none). Never printed."""
    from portolan.config import Settings, secret_value

    key = secret_value(Settings().openrouter_api_key)
    if not key:
        main_env = shared_var_root(REPO_ROOT / "var").parent / ".env"
        if main_env.is_file():
            key = secret_value(Settings(_env_file=main_env).openrouter_api_key)
    if not key or not key.strip():
        raise SystemExit("OPENROUTER_API_KEY is not set")
    return key.strip()


@dataclass(frozen=True)
class Prices:
    """US$ per token, from OpenRouter's public model listing."""

    prompt: float
    completion: float
    cache_read: float

    def worst_case(self, prompt_chars: int, max_tokens: int) -> float:
        return (prompt_chars / CHARS_PER_TOKEN) * self.prompt + max_tokens * self.completion

    def expected(self, prompt_chars: int, output_tokens: int) -> float:
        return (prompt_chars / CHARS_PER_TOKEN) * self.prompt + output_tokens * self.completion


_price_cache: dict[str, Prices] = {}


def model_prices(model: str, *, transport: httpx.BaseTransport | None = None) -> Prices:
    if model in _price_cache:
        return _price_cache[model]
    with httpx.Client(transport=transport, timeout=30.0) as client:
        listing = client.get(MODELS_URL).json()
    for entry in listing.get("data", []):
        pricing = entry.get("pricing") or {}
        _price_cache[entry["id"]] = Prices(
            prompt=float(pricing.get("prompt") or 0),
            completion=float(pricing.get("completion") or 0),
            cache_read=float(pricing.get("input_cache_read") or pricing.get("prompt") or 0),
        )
    if model not in _price_cache:
        raise SystemExit(f"{model!r} is not in OpenRouter's model listing")
    return _price_cache[model]


@dataclass
class Budget:
    """Hard cap on the `luna_eval` key: before each call, spent + worst case of
    every call in flight + this call's worst case must stay within the cap."""

    meter: SpendMeter
    cap_usd: float
    key: str = SPEND_KEY
    _reserved: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def spent(self) -> float:
        return self.meter.corpus_usd(self.key)

    def reserve(self, worst_case_usd: float) -> float:
        with self._lock:
            spent = self.spent()
            if spent + self._reserved + worst_case_usd > self.cap_usd:
                raise BudgetExceeded(
                    f"{self.key}: spent ${spent:.4f} + in flight ${self._reserved:.4f} + "
                    f"next call up to ${worst_case_usd:.4f} would pass the ${self.cap_usd:.2f} cap"
                )
            self._reserved += worst_case_usd
            return worst_case_usd

    def release(self, amount: float) -> None:
        with self._lock:
            self._reserved = max(0.0, self._reserved - amount)


class ResponseCache:
    """Append-only JSONL of finished calls, keyed by everything that shapes the
    request, so a rerun of a stopped run pays only for calls it never finished.
    Each entry keeps the call's usage events, replayed into the run's sink so the
    report still shows what the answer cost to produce."""

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._entries: dict[str, dict[str, Any]] = {}
        if path is not None and path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    entry = json.loads(line)
                    self._entries[entry["key"]] = entry

    @staticmethod
    def key(parts: dict[str, Any]) -> str:
        return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()

    def get(self, key: str) -> dict[str, Any] | None:
        return self._entries.get(key)

    def put(self, key: str, result: Any, events: list[UsageEvent]) -> None:
        entry = {
            "key": key,
            "result": result.model_dump(mode="json"),
            "events": [asdict(e) for e in events],
        }
        with self._lock:
            self._entries[key] = entry
            if self.path is not None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(entry) + "\n")


class GuardedClient:
    """An `LLMClient` that answers from `ResponseCache` when it can, else
    reserves the call's worst case on `Budget` and delegates to the wrapped
    client, which meters the real cost."""

    def __init__(
        self,
        client: OpenRouterClient,
        budget: Budget,
        prices: Prices,
        cache: ResponseCache | None = None,
        reserve_factor: float = 1.0,
    ) -> None:
        self.client = client
        self.budget = budget
        self.prices = prices
        self.cache = cache or ResponseCache(None)
        self.reserve_factor = reserve_factor

    def complete(
        self,
        *,
        system: str,
        user: str,
        json_schema: dict[str, Any] | None = None,
        model: str | None = None,
        surface: Any,
        max_tokens: int = 4096,
        validate_json: bool = True,
        temperature: float | None = None,
    ) -> Any:
        from portolan.llm.base import LLMResult

        key = ResponseCache.key(
            {
                "model": model or self.client.default_model,
                "reasoning": self.client.reasoning,
                "service_tier": self.client.service_tier,
                "max_tokens": max_tokens,
                "system": system,
                "user": user,
                "schema": json_schema,
            }
        )
        outer_sink = self.client.sink
        hit = self.cache.get(key)
        if hit is not None:
            for event in hit["events"]:
                outer_sink.record(
                    UsageEvent(**{**event, "extra": {**event["extra"], "cache_replay": True}})
                )
            return LLMResult.model_validate(hit["result"])

        prompt_chars = len(system) + len(user) + len(json.dumps(json_schema or {}))
        held = self.budget.reserve(
            self.reserve_factor * self.prices.worst_case(prompt_chars, max_tokens)
        )
        private = InMemorySink()
        started = time.perf_counter()
        try:
            with self.client.using_sink(private, self.client.tenant):
                result = self.client.complete(
                    system=system,
                    user=user,
                    json_schema=json_schema,
                    model=model,
                    surface=surface,
                    max_tokens=max_tokens,
                    validate_json=validate_json,
                    temperature=temperature,
                )
        except Exception as exc:
            print(f"  call failed after {time.perf_counter() - started:.0f}s: {exc}"[:300])
            raise
        finally:
            self.budget.release(held)
            for event in private.events:
                outer_sink.record(event)
        self.cache.put(key, result, list(private.events))
        print(
            f"  call {result.model} {time.perf_counter() - started:.0f}s "
            f"in={result.input_tokens} out={result.output_tokens} "
            f"spent=${self.budget.spent():.4f}",
            flush=True,
        )
        return result

    def using_sink(self, sink: UsageSink, tenant: str) -> Any:
        return self.client.using_sink(sink, tenant)


def make_client_factory(
    *,
    model: str,
    effort: str | None,
    service_tier: str | None,
    budget: Budget,
    prices: Prices,
    api_key: str,
    read_timeout_s: float,
    cache: ResponseCache | None = None,
    reserve_factor: float = 1.0,
) -> Callable[[], GuardedClient]:
    reasoning = {"effort": effort} if effort else None

    def build() -> GuardedClient:
        client = OpenRouterClient(
            spend_key=SPEND_KEY,
            spend_meter=budget.meter,
            api_key=api_key,
            default_model=model,
            reasoning=reasoning,
            service_tier=service_tier,
            timeout=httpx.Timeout(30.0, read=read_timeout_s),
            max_retries=3,
        )
        client.warm_json_strategy(model)
        return GuardedClient(client, budget, prices, cache, reserve_factor)

    return build


# --------------------------------------------------------------------------
# Batch path: record the requests, submit one batch, replay the answers
# --------------------------------------------------------------------------


def request_key(system_prompt: str, user_message: str) -> str:
    digest = hashlib.sha256(f"{system_prompt}\n\x00\n{user_message}".encode()).hexdigest()
    return f"req-{digest[:32]}"


@dataclass
class RecordedRequest:
    custom_id: str
    source_id: str
    system: str
    user: str


class RecordingBackend:
    """Phase 1: records each extract request and returns an empty result, so
    `run_extraction` walks every source and segment without a model call."""

    def __init__(self) -> None:
        self.requests: dict[str, RecordedRequest] = {}
        self._lock = threading.Lock()

    def extract(
        self, envelope: SourceEnvelope, system_prompt: str, sink: UsageSink, tenant: str
    ) -> ExtractionResult:
        user = build_user_message(envelope)
        key = request_key(system_prompt, user)
        with self._lock:
            self.requests.setdefault(
                key, RecordedRequest(key, envelope.source_id, system_prompt, user)
            )
        return ExtractionResult(
            source_id=envelope.source_id,
            summary="",
            coverage=Coverage(segments_read=1, segments_total=1),
        )


class ReplayBackend:
    """Phase 3: answers each extract request from the batch results, through
    the same strict-then-retry-then-lenient chain as `LLMClientBackend`. The
    retry (a strict-validation failure) goes to `retry_backend` over the sync
    API when given, else straight to `parse_lenient`."""

    def __init__(
        self,
        answers: dict[str, str],
        usage: dict[str, dict[str, Any]],
        *,
        model: str,
        retry_client: Any = None,
        retry_model: str | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self.answers = answers
        self.usage = usage
        self.model = model
        self.retry_client = retry_client
        self.retry_model = retry_model
        self.max_tokens = max_tokens
        self.strict_failures: list[str] = []

    def extract(
        self, envelope: SourceEnvelope, system_prompt: str, sink: UsageSink, tenant: str
    ) -> ExtractionResult:
        user = build_user_message(envelope)
        key = request_key(system_prompt, user)
        if key not in self.answers:
            raise ValueError(f"no batch answer for {envelope.source_id} ({key})")
        usage = self.usage.get(key) or {}
        sink.record(
            UsageEvent(
                tenant=tenant,
                surface="extract",
                model_key=self.model,
                input_tokens=int(usage.get("prompt_tokens", 0) or 0),
                output_tokens=int(usage.get("completion_tokens", 0) or 0),
                cached_tokens=int(
                    (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
                ),
                cost_estimate=usage.get("cost") if isinstance(usage.get("cost"), float) else None,
                request_id=key,
                timestamp=dt.datetime.now(dt.UTC).isoformat(),
                extra={
                    "batch": True,
                    "reasoning_tokens": int(
                        (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
                        or 0
                    ),
                },
            )
        )
        data = extract_json_object(self.answers[key]) or {}
        try:
            return ExtractionResult.model_validate(data)
        except Exception as exc:  # noqa: BLE001 -- the strict attempt, as LLMClientBackend
            self.strict_failures.append(envelope.source_id)
            retry_data: dict[str, Any] = data
            if self.retry_client is not None:
                retry = LLMClientBackend(
                    self.retry_client, model=self.retry_model, max_tokens=self.max_tokens
                )
                with retry._sink_scope(sink, tenant):
                    result = retry._complete(
                        system_prompt,
                        _build_retry_message(user, exc),  # type: ignore[arg-type]
                        extraction_result_json_schema(),
                    )
                retry_data = result.data if result.data is not None else data
            try:
                return ExtractionResult.model_validate(retry_data)
            except Exception:  # noqa: BLE001
                parsed, _rejected = ExtractionResult.parse_lenient(
                    retry_data, default_source_id=envelope.source_id
                )
                return parsed


def run_batch(
    requests: list[RecordedRequest],
    *,
    batch_model: str,
    effort: str | None,
    max_tokens: int,
    api_key: str,
    meter: SpendMeter,
    out: Path,
    poll_seconds: float,
    timeout_seconds: float,
) -> tuple[dict[str, str], dict[str, dict[str, Any]], dict[str, Any]]:
    """Submit one batch, wait, and return (answers, usage by key, timing)."""
    client = OpenRouterBatchClient(spend_key=SPEND_KEY, spend_meter=meter, api_key=api_key)
    schema = extraction_result_json_schema()
    reasoning = {"effort": effort} if effort else None
    lines = [
        {
            "custom_id": r.custom_id,
            "body": chat_request_body(
                system=r.system,
                user=r.user,
                max_tokens=max_tokens,
                json_schema=schema,
                reasoning=reasoning,
            ),
        }
        for r in requests
    ]
    marker = out / "batch_submitted.json"
    if marker.is_file():
        # Resume polling a batch this directory already submitted (the driver was stopped).
        saved = json.loads(marker.read_text(encoding="utf-8"))
        batch_id, submitted_at = saved["batch_id"], saved["submitted_at"]
        print(f"resuming batch {batch_id}")
    else:
        submitted_at = time.time()
        batch = client.submit(batch_model, lines)
        batch_id = batch["id"]
        marker.write_text(
            json.dumps({"batch_id": batch_id, "submitted_at": submitted_at}) + "\n",
            encoding="utf-8",
        )
        print(f"batch {batch_id} submitted: {len(lines)} requests, status {batch.get('status')}")
    polls: list[dict[str, Any]] = []

    def on_poll(b: dict[str, Any]) -> None:
        row = {
            "t": round(time.time() - submitted_at, 1),
            "status": b.get("status"),
            "counts": b.get("request_counts"),
        }
        polls.append(row)
        print(f"  {row['t']:>8.1f}s {row['status']} {row['counts']}", flush=True)

    final = client.wait(
        batch_id, poll_seconds=poll_seconds, timeout_seconds=timeout_seconds, on_poll=on_poll
    )
    finished_at = time.time()
    (out / "batch.json").write_text(json.dumps(final, indent=2) + "\n", encoding="utf-8")
    results = batch_results(final)
    answers = {cid: r.text for cid, r in results.items() if r.body is not None}
    usage = {cid: r.usage for cid, r in results.items()}
    timing = {
        "batch_id": batch_id,
        "status": final.get("status"),
        "submitted_at": dt.datetime.fromtimestamp(submitted_at, dt.UTC).isoformat(),
        "created_at": final.get("created_at"),
        "finalized_at": final.get("finalized_at"),
        "wall_s": round(finished_at - submitted_at, 1),
        "request_counts": final.get("request_counts"),
        "usage": final.get("usage"),
        "errors": {cid: r.error for cid, r in results.items() if r.error is not None},
        "polls": polls,
    }
    return answers, usage, timing


# --------------------------------------------------------------------------
# exam
# --------------------------------------------------------------------------


def _events_rows(sink: InMemorySink) -> list[dict[str, Any]]:
    return [
        {
            "model": e.model_key,
            "input_tokens": e.input_tokens,
            "output_tokens": e.output_tokens,
            "cached_tokens": e.cached_tokens,
            "cost": e.cost_estimate,
            "reasoning_tokens": e.extra.get("reasoning_tokens"),
            "provider": e.extra.get("provider"),
            "service_tier": e.extra.get("service_tier"),
            "error": e.extra.get("error"),
            "batch": e.extra.get("batch", False),
        }
        for e in sink.events
        if e.input_tokens > 0 or e.extra.get("error")
    ]


def cmd_exam(args: argparse.Namespace) -> int:
    from portolan.bench.__main__ import _parser_counts
    from portolan.bench.corpus_loader import DEFAULT_CORPUS_ROOT
    from portolan.bench.extraction import run_extraction, score_extraction
    from portolan.bench.provenance import provenance_kwargs
    from portolan.bench.report import BenchReport, write
    from portolan.bench.truth import (
        DEFAULT_MANIFEST_PATH,
        DEFAULT_TRUTH_TABLE_PATH,
        Manifest,
        Truth,
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = Manifest.load(REPO_ROOT / DEFAULT_MANIFEST_PATH)
    truth = Truth.for_manifest(manifest, truth_table_path=REPO_ROOT / DEFAULT_TRUTH_TABLE_PATH)
    corpus_root = REPO_ROOT / DEFAULT_CORPUS_ROOT
    source_ids = args.source_ids or SLICE
    system_prompt = build_system_prompt(variant=args.variant)

    # The exact requests, from the same run code with a recording backend.
    recorder = RecordingBackend()
    run_extraction(
        manifest,
        recorder,
        source_ids=source_ids,
        corpus_root=corpus_root,
        system_prompt=system_prompt,
        truth=truth,
    )
    requests = list(recorder.requests.values())
    schema_chars = len(json.dumps(extraction_result_json_schema()))
    prompt_chars = sum(len(r.system) + len(r.user) + schema_chars for r in requests)
    model = args.model + (BATCH_SUFFIX if args.batch else "")
    prices = model_prices(model)
    meter = SpendMeter(default_spend_path(REPO_ROOT / "var"))
    budget = Budget(meter, args.max_usd)
    expected = prices.expected(prompt_chars, args.expect_output_tokens * len(requests))
    worst = prices.worst_case(prompt_chars, args.max_tokens * len(requests))
    spent = budget.spent()
    print(
        f"estimate: {model} effort={args.effort} tier={args.service_tier} "
        f"{len(source_ids)} sources, {len(requests)} requests, ~{prompt_chars / CHARS_PER_TOKEN:,.0f} "
        f"input tokens; expected ${expected:.4f} (at {args.expect_output_tokens} output tokens "
        f"per request), worst case ${worst:.4f}; {SPEND_KEY} spent ${spent:.4f} of "
        f"${args.max_usd:.2f}",
        flush=True,
    )
    if spent + expected > args.max_usd:
        raise SystemExit("stop: the expected cost would pass the cap")
    if args.dry_run:
        return 0

    api_key = openrouter_key()
    sink = InMemorySink()
    batch_timing: dict[str, Any] | None = None
    if args.batch:
        # Worst case for the whole batch is held up front; a batch cannot be stopped midway.
        held = budget.reserve(worst)
        try:
            answers, usage, batch_timing = run_batch(
                requests,
                batch_model=model,
                effort=args.effort,
                max_tokens=args.max_tokens,
                api_key=api_key,
                meter=meter,
                out=out,
                poll_seconds=args.poll_seconds,
                timeout_seconds=args.batch_timeout_s,
            )
        finally:
            budget.release(held)
        retry_client = make_client_factory(
            model=args.model,
            effort=args.effort,
            service_tier=None,
            budget=budget,
            prices=model_prices(args.model),
            api_key=api_key,
            read_timeout_s=args.read_timeout_s,
        )()
        replay = ReplayBackend(
            answers,
            usage,
            model=model,
            retry_client=retry_client,
            retry_model=args.model,
            max_tokens=args.max_tokens,
        )
        run_result = run_extraction(
            manifest,
            replay,
            source_ids=source_ids,
            corpus_root=corpus_root,
            system_prompt=system_prompt,
            truth=truth,
            sink=sink,
        )
        batch_timing["strict_failures"] = replay.strict_failures
    else:
        factory = make_client_factory(
            model=model,
            effort=args.effort,
            service_tier=args.service_tier,
            budget=budget,
            prices=prices,
            api_key=api_key,
            read_timeout_s=args.read_timeout_s,
            cache=ResponseCache(out / "cache.jsonl"),
            reserve_factor=args.reserve_factor,
        )

        def backend_factory() -> LLMClientBackend:
            return LLMClientBackend(factory(), model=model, max_tokens=args.max_tokens)

        run_result = run_extraction(
            manifest,
            backend_factory(),
            source_ids=source_ids,
            corpus_root=corpus_root,
            system_prompt=system_prompt,
            truth=truth,
            sink=sink,
            concurrency=args.concurrency,
            backend_factory=backend_factory,
        )

    score = score_extraction(truth, run_result)
    label = f"{model}{'@' + args.effort if args.effort else ''}"
    report = BenchReport(
        backend="luna-eval",
        date=dt.date.today(),
        extraction=score,
        samples=1,
        **provenance_kwargs(
            corpus=corpus_root,
            truth_table=REPO_ROOT / DEFAULT_TRUTH_TABLE_PATH,
            prompt_variant=args.variant,
            extractor_model=label,
        ),
    )
    md_path, json_path = write(report, out)
    parser = _parser_counts(run_result)
    (out / f"parser-{md_path.stem}.json").write_text(
        json.dumps(parser, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    run_info = {
        "model": model,
        "effort": args.effort,
        "service_tier": args.service_tier,
        "max_tokens": args.max_tokens,
        "concurrency": args.concurrency,
        "source_ids": source_ids,
        "requests": len(requests),
        "wall_time_s": dict(run_result.wall_time_s),
        "total_wall_time_s": run_result.total_wall_time_s,
        "failed_sources": dict(run_result.failed_sources),
        "calls": _events_rows(sink),
        "batch": batch_timing,
        "luna_eval_spent_after": budget.spent(),
    }
    (out / "run.json").write_text(json.dumps(run_info, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {md_path}\nwrote {json_path}")
    print(
        f"recall={score.recall:.3f} cost=${score.cost_estimate or 0:.4f} "
        f"failed={len(run_result.failed_sources)} {SPEND_KEY} spent ${budget.spent():.4f}"
    )
    return 0


# --------------------------------------------------------------------------
# table (exam)
# --------------------------------------------------------------------------


def _norm(text: str) -> str:
    return " ".join(text.split()).lower()


def _token_cost(calls: list[dict[str, Any]], prices: Prices) -> float:
    return sum(
        (c["input_tokens"] - (c.get("cached_tokens") or 0)) * prices.prompt
        + (c.get("cached_tokens") or 0) * prices.cache_read
        + c["output_tokens"] * prices.completion
        for c in calls
    )


def standard_price_cost(model: str | None, calls: list[dict[str, Any]]) -> float | None:
    """The same calls' tokens at the model's standard (non-batch, non-flex) prices."""
    if not model or not calls:
        return None
    return _token_cost(calls, model_prices(model.removesuffix(BATCH_SUFFIX)))


def batch_price_cost(
    model: str | None, billed_usd: float | None, calls: list[dict[str, Any]]
) -> float | None:
    """What the same calls' tokens cost at the model's `:batch` prices
    (OpenRouter's listing; cache writes not counted). None when there is no
    `:batch` slug or no billed run."""
    if not model or billed_usd is None or not calls:
        return None
    try:
        batch = model_prices(model.removesuffix(BATCH_SUFFIX) + BATCH_SUFFIX)
    except SystemExit:
        return None
    return _token_cost(calls, batch)


def _load_run(directory: Path) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    from portolan.bench.extraction import ExtractionScore

    (path,) = sorted(directory.glob("bench-*.json"))
    saved = json.loads(path.read_text(encoding="utf-8"))
    run_info_path = directory / "run.json"
    run_info = json.loads(run_info_path.read_text()) if run_info_path.is_file() else {}
    return ExtractionScore.model_validate(saved["extraction"]), saved, run_info


def exam_rows(specs: list[tuple[str, Path]]) -> dict[str, Any]:
    """Per-run metrics on the sources every run scored (a source that failed
    in any run is left out of all of them), as the prompt-neutral table does."""
    from portolan.bench.corpus_loader import DEFAULT_CORPUS_ROOT, load_prepared_source
    from portolan.bench.truth import DEFAULT_MANIFEST_PATH, Manifest

    manifest = Manifest.load(REPO_ROOT / DEFAULT_MANIFEST_PATH)
    runs = [(name, d, *_load_run(d)) for name, d in specs]
    scored = [set(s.produced_by_source) | set(s.failed_sources) for _, _, s, _, _ in runs]
    in_slice = set.intersection(*scored) if scored else set()
    failed = set().union(*(set(s.failed_sources) for _, _, s, _, _ in runs))
    common = in_slice - failed
    by_id = {s.id: s for s in manifest.sources}
    texts = {
        sid: load_prepared_source(by_id[sid], REPO_ROOT / DEFAULT_CORPUS_ROOT).text
        for sid in common
    }
    placements: dict[str, set[str]] = defaultdict(set)
    for source in manifest.sources:
        for f in source.facts:
            placements[f.fact_id].add(source.id)
    fact_ids = sorted(f for f, srcs in placements.items() if srcs & common and srcs <= in_slice)

    rows = []
    for name, directory, score, _saved, info in runs:
        credit: dict[str, float] = defaultdict(float)
        for m in score.matched:
            if m.source_id in common:
                credit[m.fact_id] = max(credit[m.fact_id], m.credit)
        lenient_ids = set(credit) | set(score.near_miss_credited_fact_ids)
        produced = [
            (sid, f)
            for sid, facts in score.produced_by_source.items()
            if sid in common
            for f in facts
        ]
        quoted = [(sid, f.quote) for sid, f in produced if f.quote]
        unplanted = [u for u in score.unplanted if u.source_id in common]
        matched_produced = sum(1 for m in score.matched if m.source_id in common)
        calls = info.get("calls") or []
        n_docs = len(common)
        latencies = [v for k, v in (info.get("wall_time_s") or {}).items() if k in common]
        reasoning = [c["reasoning_tokens"] for c in calls if c.get("reasoning_tokens") is not None]
        batch = info.get("batch") or {}
        cost = (batch.get("usage") or {}).get("cost") if batch else None
        if cost is None:
            cost = score.cost_estimate
        rows.append(
            {
                "name": name,
                "dir": str(directory),
                "model": info.get("model"),
                "effort": info.get("effort"),
                "planted": len(fact_ids),
                "recall_strict": sum(credit[f] for f in fact_ids) / len(fact_ids),
                "recall_lenient": sum(1 for f in fact_ids if f in lenient_ids) / len(fact_ids),
                "omitted_strict": sum(1 for f in fact_ids if credit.get(f, 0) == 0),
                "produced": len(produced),
                "quote_verbatim_exact": (
                    sum(1 for sid, q in quoted if q in texts[sid]) / len(quoted) if quoted else None
                ),
                "substantiated_matched": matched_produced / len(produced) if produced else None,
                "invented": sum(
                    1 for u in unplanted if not u.in_source_text and not u.is_distractor_leak
                ),
                "distractor_leaks": sum(1 for u in unplanted if u.is_distractor_leak),
                "type_confusions": sum(
                    1
                    for m in score.matched
                    if m.source_id in common and m.fact_type != m.produced_type
                ),
                "failed_sources": sorted(score.failed_sources),
                "cost_usd": cost,
                # Every attempted source cost money, failed ones included.
                "cost_per_doc": (cost or 0) / len(info.get("source_ids") or common),
                "cost_per_doc_at_standard_price": (
                    at_std / len(info.get("source_ids") or common)
                    if (at_std := standard_price_cost(info.get("model"), calls)) is not None
                    else None
                ),
                "cost_per_doc_at_batch_price": (
                    at_batch / len(info.get("source_ids") or common)
                    if (at_batch := batch_price_cost(info.get("model"), cost, calls)) is not None
                    else None
                ),
                "latency_median_s": statistics.median(latencies) if latencies else None,
                "latency_p90_s": (
                    sorted(latencies)[max(0, round(0.9 * len(latencies)) - 1)]
                    if latencies
                    else None
                ),
                "batch_wall_s": batch.get("wall_s") if batch else None,
                "reasoning_tokens_per_doc": sum(reasoning) / n_docs if reasoning and n_docs else 0,
                "output_tokens_per_doc": (
                    sum(c.get("output_tokens") or 0 for c in calls) / n_docs if n_docs else None
                ),
                "providers": sorted({c.get("provider") for c in calls if c.get("provider")}),
                "service_tiers": sorted(
                    {c.get("service_tier") for c in calls if c.get("service_tier")}
                ),
                "matched_ids": sorted(credit),
            }
        )
    return {
        "common_sources": sorted(common),
        "excluded_sources": sorted(failed),
        "fact_ids": fact_ids,
        "rows": rows,
    }


def cmd_table(args: argparse.Namespace) -> int:
    specs = []
    for spec in args.run:
        name, _, directory = spec.partition("=")
        specs.append((name, Path(directory)))
    table = exam_rows(specs)
    print(
        f"common sources {len(table['common_sources'])}; excluded {table['excluded_sources']}; "
        f"planted facts {len(table['fact_ids'])}"
    )
    keys = [
        "recall_strict", "recall_lenient", "omitted_strict", "produced", "quote_verbatim_exact",
        "substantiated_matched", "invented", "distractor_leaks", "type_confusions", "cost_per_doc",
        "cost_per_doc_at_standard_price", "cost_per_doc_at_batch_price", "latency_median_s", "latency_p90_s", "batch_wall_s",
        "reasoning_tokens_per_doc",
        "output_tokens_per_doc",
    ]  # fmt: skip
    for row in table["rows"]:
        cells = []
        for k in keys:
            v = row[k]
            cells.append(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}")
        print(f"{row['name']}: " + " ".join(cells) + f" providers={row['providers']}")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(table, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.json_out}")
    return 0


# --------------------------------------------------------------------------
# labels (hand-verdict set)
# --------------------------------------------------------------------------


@dataclass
class LabelSource:
    source_key: str
    corpus: str
    text: str
    system_prompt: str
    # (envelope, offset of the envelope text in `text`)
    envelopes: list[tuple[SourceEnvelope, int]]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def label_sources(eval_dir: Path, v2_docs: Path) -> list[LabelSource]:
    """Each evaluation document with the envelopes and system prompt its
    teacher was given: a public record's stored envelope (its system prompt
    checked against the record's sha), a v2 document's WP5 envelopes."""
    from portolan.training.records import TrainingRecord
    from portolan.training.sft_format import envelope_for_record
    from scripts.data.silver_contexts import CORPUS_CONTEXTS

    sources = {s["source_key"]: s for s in _load_jsonl(eval_dir / "eval_sources.jsonl")}
    out: list[LabelSource] = []
    for raw in _load_jsonl(eval_dir / "build" / "public_sample_records.jsonl"):
        record = TrainingRecord.model_validate(raw)
        system = build_system_prompt(
            company_context=CORPUS_CONTEXTS[record.corpus], variant=record.prompt_variant
        )
        if hashlib.sha256(system.encode()).hexdigest() != record.system_prompt_sha256:
            raise SystemExit(f"{record.record_id}: system prompt differs from the teacher's")
        envelope = envelope_for_record(record)
        src = sources[record.record_id]
        if envelope.text != src["text"]:
            raise SystemExit(f"{record.record_id}: envelope text differs from the eval text")
        out.append(
            LabelSource(record.record_id, record.corpus, src["text"], system, [(envelope, 0)])
        )

    v2_keys = {k for k, s in sources.items() if s["source"] == "synth_v2"}
    if v2_keys:
        from corpus.generator.v2.completeness import (
            PROMPT_VARIANT,
            _source_envelopes,
            load_written_docs,
            source_inputs,
            world_company_context,
        )
        from corpus.generator.v2.models import Split
        from corpus.generator.v2.packs.schema import load_pack
        from corpus.generator.v2.planner import plan_world
        from corpus.generator.v2.simulator import simulate_world

        pack = load_pack(
            REPO_ROOT / "corpus" / "generator" / "v2" / "packs" / "medical_devices.yaml"
        )
        world = simulate_world(pack, seed=42, months=8, split=Split("train"))
        context = world_company_context(world)
        system = build_system_prompt(company_context=context, variant=PROMPT_VARIANT)
        plans = plan_world(world, pack).doc_by_id()
        for plan_id, doc in load_written_docs(v2_docs).items():
            for source in source_inputs(doc, plans[plan_id]):
                if source.source_id not in v2_keys:
                    continue
                src = sources[source.source_id]
                if source.text != src["text"] or context != src["company_context"]:
                    raise SystemExit(f"{source.source_id}: v2 text or context differs from eval")
                envs = [(env, off) for _i, env, off in _source_envelopes(source)]
                out.append(LabelSource(source.source_id, "synth_v2", src["text"], system, envs))
    missing = set(sources) - {s.source_key for s in out}
    if missing:
        raise SystemExit(f"no envelopes rebuilt for {sorted(missing)}")
    return out


MIN_CLAIM_CHARS = 20  # a claim is a fallback needle only when it could not match by accident


_FOLD = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"',
                       "—": "-", "–": "-"})  # fmt: skip


def _folded(text: str) -> tuple[str, list[int]]:
    """`text` with curly quotes and dashes made plain and whitespace runs made
    one space, plus each folded character's index in `text`."""
    out: list[str] = []
    index: list[int] = []
    for i, ch in enumerate(text.translate(_FOLD)):
        if ch.isspace():
            if out and out[-1] == " ":
                continue
            ch = " "
        out.append(ch)
        index.append(i)
    return "".join(out), index


def _span(text: str, quote: str | None, claim: str) -> tuple[int, int] | None:
    """Where the quote (else a long enough claim) sits in `text`: exact first,
    then with quotes, dashes and whitespace folded (the silver gate's own
    resolver also tolerates these)."""
    folded_text, index = _folded(text)
    for needle in (quote, claim if len(claim or "") >= MIN_CLAIM_CHARS else None):
        if not needle:
            continue
        at = text.find(needle)
        if at >= 0:
            return at, at + len(needle)
        stripped = needle.strip().strip('"').strip(".")
        if stripped and (at := text.find(stripped)) >= 0:
            return at, at + len(stripped)
        folded = _folded(stripped)[0].strip()
        if folded and (at := folded_text.find(folded)) >= 0:
            return index[at], index[at + len(folded) - 1] + 1
    return None


def _verdict_span(row: dict[str, Any], text: str) -> tuple[int, int] | None:
    if row.get("quote_span"):
        return tuple(row["quote_span"])  # type: ignore[return-value]
    if row.get("char_start") is not None:
        return row["char_start"], row["char_end"]
    return _span(text, row.get("quote"), row.get("claim", ""))


def _overlap(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def match_label_facts(
    produced: list[dict[str, Any]], verdicts: list[dict[str, Any]], text: str
) -> list[dict[str, Any]]:
    """Each produced fact matched to the verdicted fact its quote overlaps most
    (any shared character, the WP5 decoy rule's test); unmatched facts stay
    unverdicted."""
    spans = [(row, _verdict_span(row, text)) for row in verdicts]
    out = []
    for fact in produced:
        span = _span(text, fact.get("quote"), fact.get("claim", ""))
        best, best_overlap = None, 0
        if span is not None:
            for row, vspan in spans:
                if vspan is None:
                    continue
                ov = _overlap(span, vspan)
                if ov > best_overlap:
                    best, best_overlap = row, ov
        out.append(
            {
                **fact,
                "span": list(span) if span else None,
                "eval_id": best["eval_id"] if best else None,
                "verdict": best["verdict"] if best else None,
                "type_ok": (fact["fact_type"] in best["ok_types"]) if best else None,
            }
        )
    return out


def cmd_labels(args: argparse.Namespace) -> int:
    eval_dir = Path(args.eval_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    verdicts = _load_jsonl(eval_dir / "eval_verdicts.jsonl")
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    sources_meta = {s["source_key"]: s for s in _load_jsonl(eval_dir / "eval_sources.jsonl")}
    for key, meta in sources_meta.items():
        ids = set(meta["eval_ids"])
        by_source[key] = [v for v in verdicts if v["eval_id"] in ids]
    sources = label_sources(eval_dir, Path(args.v2_docs))
    envelopes = sum(len(s.envelopes) for s in sources)
    schema_chars = len(json.dumps(extraction_result_json_schema()))
    prompt_chars = sum(
        len(s.system_prompt) + len(build_user_message(e)) + schema_chars
        for s in sources
        for e, _ in s.envelopes
    )
    prices = model_prices(args.model)
    meter = SpendMeter(default_spend_path(REPO_ROOT / "var"))
    budget = Budget(meter, args.max_usd)
    expected = prices.expected(prompt_chars, args.expect_output_tokens * envelopes)
    print(
        f"estimate: {args.model} effort={args.effort} {len(sources)} documents, {envelopes} "
        f"requests; expected ${expected:.4f}; {SPEND_KEY} spent ${budget.spent():.4f} of "
        f"${args.max_usd:.2f}",
        flush=True,
    )
    if budget.spent() + expected > args.max_usd:
        raise SystemExit("stop: the expected cost would pass the cap")
    if args.dry_run:
        return 0

    factory = make_client_factory(
        model=args.model,
        effort=args.effort,
        service_tier=args.service_tier,
        budget=budget,
        prices=prices,
        api_key=openrouter_key(),
        read_timeout_s=args.read_timeout_s,
        cache=ResponseCache(out / "cache.jsonl"),
        reserve_factor=args.reserve_factor,
    )
    sink = InMemorySink()
    rows: list[dict[str, Any]] = []
    lock = threading.Lock()

    def run(source: LabelSource) -> None:
        backend = LLMClientBackend(factory(), model=args.model, max_tokens=args.max_tokens)
        produced: list[dict[str, Any]] = []
        started = time.perf_counter()
        error = None
        for envelope, _offset in source.envelopes:
            try:
                result = backend.extract(envelope, source.system_prompt, sink, "luna-eval")
            except Exception as exc:  # noqa: BLE001 -- recorded (a budget stop too); rerun uses the cache
                error = f"{type(exc).__name__}: {exc}"[:500]
                continue
            for fact in result.facts:
                produced.append(
                    {
                        "fact_type": fact.type,
                        "claim": fact.claim,
                        "quote": fact.quote,
                        "locator": fact.locator,
                    }
                )
        matched = match_label_facts(produced, by_source[source.source_key], source.text)
        with lock:
            rows.append(
                {
                    "source_key": source.source_key,
                    "corpus": source.corpus,
                    "latency_s": round(time.perf_counter() - started, 2),
                    "error": error,
                    "facts": matched,
                }
            )

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        list(pool.map(run, sources))
    rows.sort(key=lambda r: r["source_key"])
    (out / "labels.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    (out / "run.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "effort": args.effort,
                "max_tokens": args.max_tokens,
                "calls": _events_rows(sink),
                "cost_usd": sum(e.cost_estimate or 0 for e in sink.events),
                "luna_eval_spent_after": budget.spent(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {out / 'labels.jsonl'}; {SPEND_KEY} spent ${budget.spent():.4f}")
    return 0


def v2_references(wp5_dir: Path, keys: set[str]) -> list[dict[str, Any]]:
    """The v2 documents' scripted facts (true by construction; the verdict set
    holds only the silver labels beyond them) and planted decoys (never a
    fact), as reference rows shaped like verdict rows."""
    refs: list[dict[str, Any]] = []
    for label in _load_jsonl(wp5_dir / "labels.jsonl"):
        if label["label_source"] == "scripted_writer" and label["doc_id"] in keys:
            refs.append(
                {
                    "eval_id": label["fact_id"],
                    "source_key": label["doc_id"],
                    "char_start": label["char_start"],
                    "char_end": label["char_end"],
                    "verdict": "correct",
                    "ok_types": [label["fact_type"]],
                    "origin": "scripted",
                }
            )
    for decoy in _load_jsonl(wp5_dir / "decoys.jsonl"):
        if decoy["doc_id"] in keys:
            refs.append(
                {
                    "eval_id": decoy["decoy_id"],
                    "source_key": decoy["doc_id"],
                    "char_start": decoy["char_start"],
                    "char_end": decoy["char_end"],
                    "verdict": "not_a_fact",
                    "ok_types": [],
                    "origin": f"decoy:{decoy['decoy_class_id']}",
                }
            )
    return refs


def label_metrics(
    rows: list[dict[str, Any]],
    references: list[dict[str, Any]],
    extra_verdicts: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, Any]:
    """Counts over one model's labels of the verdict set. `references` are the
    verdicted teacher facts plus, on v2 documents, scripted facts and decoys;
    each row's facts must already be matched against them. `extra_verdicts`
    holds by-eye verdicts on facts that match no reference, keyed by
    (source_key, index within the source's facts)."""
    real_ids = {v["eval_id"] for v in references if v["verdict"] in ("correct", "wrong_type")}
    produced = 0
    on_wrong = 0
    on_real_ok = 0
    on_real_bad_type = 0
    unverdicted = 0
    extra_ok = 0
    extra_bad = 0
    extra_missing = 0
    duplicates = 0
    unresolvable = 0
    found: set[str] = set()
    found_ok_type: set[str] = set()
    failed_docs = []
    for row in rows:
        if row.get("error") and not row["facts"]:
            failed_docs.append(row["source_key"])
        for index, fact in enumerate(row["facts"]):
            produced += 1
            if fact["eval_id"] is None and not fact.get("span"):
                # Neither quote nor claim is in the text: the silver gate drops these.
                unresolvable += 1
                continue
            if fact["eval_id"] is None:
                unverdicted += 1
                extra = extra_verdicts.get((row["source_key"], index))
                if extra is None:
                    extra_missing += 1
                elif extra["verdict"] == "correct":
                    extra_ok += 1
                else:
                    extra_bad += 1
                continue
            if fact["verdict"] == "not_a_fact":
                on_wrong += 1
                continue
            if fact["eval_id"] in found:
                # A second label of the same fact: the label judge drops repeats.
                duplicates += 1
                continue
            found.add(fact["eval_id"])
            if fact["type_ok"]:
                on_real_ok += 1
                found_ok_type.add(fact["eval_id"])
            else:
                on_real_bad_type += 1
    good = on_real_ok + extra_ok
    judged = produced - extra_missing
    return {
        "documents": len(rows),
        "failed_documents": failed_docs,
        "produced": produced,
        "on_human_wrong": on_wrong,
        "on_real_ok_type": on_real_ok,
        "on_real_wrong_type": on_real_bad_type,
        "duplicates": duplicates,
        "quote_not_in_text": unresolvable,
        "unverdicted": unverdicted,
        "unverdicted_correct": extra_ok,
        "unverdicted_wrong": extra_bad,
        "unverdicted_not_read": extra_missing,
        "precision": good / judged if judged else None,
        "real_facts": len(real_ids),
        "real_found": len(found & real_ids),
        "real_found_ok_type": len(found_ok_type & real_ids),
        "recall_of_real": len(found & real_ids) / len(real_ids) if real_ids else None,
    }


def _references_by_source(
    eval_dir: Path, wp5_dir: Path | None
) -> tuple[dict[str, str], dict[str, list[dict[str, Any]]]]:
    sources = {s["source_key"]: s for s in _load_jsonl(eval_dir / "eval_sources.jsonl")}
    verdicts = {v["eval_id"]: v for v in _load_jsonl(eval_dir / "eval_verdicts.jsonl")}
    refs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for key, source in sources.items():
        refs[key] = [verdicts[i] for i in source["eval_ids"]]
    if wp5_dir is not None:
        v2_keys = {k for k, s in sources.items() if s["source"] == "synth_v2"}
        for ref in v2_references(wp5_dir, v2_keys):
            refs[ref["source_key"]].append(ref)
    return {k: s["text"] for k, s in sources.items()}, refs


def cmd_teacher_labels(args: argparse.Namespace) -> int:
    """The labeling teacher's own stored output on the verdict set, in the
    `labels.jsonl` shape, so it is scored by the same code as a new run: a
    public record's kept facts, and a v2 document's WP5 completeness-pass
    facts (the gate's `kept`, original text), decoy hits and repeats included."""
    eval_dir = Path(args.eval_dir)
    sources = {s["source_key"]: s for s in _load_jsonl(eval_dir / "eval_sources.jsonl")}
    rows: dict[str, list[dict[str, Any]]] = {k: [] for k in sources}
    for record in _load_jsonl(eval_dir / "build" / "public_sample_records.jsonl"):
        for fact in record["facts"]:
            rows[record["record_id"]].append(
                {
                    "fact_type": fact.get("fact_type"),
                    "claim": fact.get("claim", ""),
                    "quote": fact.get("quote"),
                    "locator": fact.get("locator"),
                }
            )
    for fact in _load_jsonl(Path(args.wp5_dir) / "teacher_facts.jsonl"):
        if (
            fact["doc_id"] in rows
            and sources[fact["doc_id"]]["source"] == "synth_v2"
            and fact["role"] == "completeness"
            and fact["version"] == "original"
            and fact["teacher_model"] == args.teacher
            and fact["gate"] == "kept"
        ):
            rows[fact["doc_id"]].append(
                {
                    "fact_type": fact["fact_type"],
                    "claim": fact["claim"],
                    "quote": fact.get("span_text") or fact.get("quote"),
                    "locator": fact.get("locator"),
                }
            )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lines = []
    for key in sorted(rows):
        produced = rows[key]
        matched = match_label_facts(produced, [], sources[key]["text"])
        lines.append(
            json.dumps(
                {
                    "source_key": key,
                    "corpus": sources[key]["source"],
                    "error": None,
                    "facts": matched,
                },
                ensure_ascii=False,
            )
        )
    (out / "labels.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "run.json").write_text(
        json.dumps({"model": args.teacher, "stored": True, "cost_usd": None}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {out / 'labels.jsonl'} ({sum(len(v) for v in rows.values())} facts)")
    return 0


def _rematch(
    runs: dict[str, list[dict[str, Any]]],
    refs: dict[str, list[dict[str, Any]]],
    texts: dict[str, str],
) -> None:
    for rows in runs.values():
        for row in rows:
            row["facts"] = match_label_facts(
                [
                    {k: f.get(k) for k in ("fact_type", "claim", "quote", "locator")}
                    for f in row["facts"]
                ],
                refs[row["source_key"]],
                texts[row["source_key"]],
            )


def pool_extra_verdicts(
    runs: dict[str, list[dict[str, Any]]],
    refs: dict[str, list[dict[str, Any]]],
    extra: dict[str, dict[tuple[str, int], dict[str, Any]]],
) -> int:
    """Add each by-eye-verdicted fact that matched no reference to the
    references (pooling, as in shared-task evaluation), so a true fact any
    model found counts in every model's recall and a second model's copy of a
    judged fact is matched without a second reading. Returns rows added."""
    added = 0
    for name, rows in runs.items():
        for row in rows:
            for index, fact in enumerate(row["facts"]):
                verdict = extra.get(name, {}).get((row["source_key"], index))
                if fact["eval_id"] is not None or verdict is None or not fact.get("span"):
                    continue
                span = tuple(fact["span"])
                same_as = verdict.get("same_as")
                if same_as:
                    # The same fact as a reference, quoted from another sentence: an alias
                    # span carrying the reference's id, so it counts once.
                    target = next(r for r in refs[row["source_key"]] if r["eval_id"] == same_as)
                    refs[row["source_key"]].append(
                        {
                            **target,
                            "char_start": span[0],
                            "char_end": span[1],
                            "quote_span": None,
                            "origin": "alias",
                        }
                    )
                    added += 1
                    continue
                if any(
                    _overlap(span, (r["char_start"], r["char_end"])) > 0
                    for r in refs[row["source_key"]]
                    if r.get("origin") == "pooled"
                ):
                    continue
                ok_types = verdict.get("ok_types") or (
                    [fact["fact_type"]] if verdict["verdict"] == "correct" else []
                )
                refs[row["source_key"]].append(
                    {
                        "eval_id": f"pool:{name}:{row['source_key']}:{index}",
                        "source_key": row["source_key"],
                        "char_start": span[0],
                        "char_end": span[1],
                        "verdict": verdict["verdict"],
                        "ok_types": ok_types,
                        "origin": "pooled",
                        "reason": verdict.get("reason"),
                    }
                )
                added += 1
    return added


def cmd_label_table(args: argparse.Namespace) -> int:
    texts, refs = _references_by_source(
        Path(args.eval_dir), Path(args.wp5_dir) if args.wp5_dir else None
    )
    extra: dict[str, dict[tuple[str, int], dict[str, Any]]] = defaultdict(dict)
    if args.extra_verdicts:
        for row in _load_jsonl(Path(args.extra_verdicts)):
            extra[row["run"]][(row["source_key"], row["index"])] = row
    runs: dict[str, list[dict[str, Any]]] = {}
    dirs: dict[str, Path] = {}
    for spec in args.run:
        name, _, directory = spec.partition("=")
        dirs[name] = Path(directory)
        runs[name] = _load_jsonl(Path(directory) / "labels.jsonl")
    _rematch(runs, refs, texts)
    pooled = pool_extra_verdicts(runs, refs, extra)
    _rematch(runs, refs, texts)
    all_refs = [r for rows in refs.values() for r in rows]
    print(
        f"references: {len(all_refs)} ({pooled} pooled by-eye); real facts "
        f"{len({r['eval_id'] for r in all_refs if r['verdict'] in ('correct', 'wrong_type')})}"
    )
    table = {}
    unread: list[dict[str, Any]] = []
    for name, rows in runs.items():
        directory = dirs[name]
        for row in rows:
            for index, fact in enumerate(row["facts"]):
                if fact["eval_id"] is None:
                    unread.append(
                        {
                            "run": name,
                            "source_key": row["source_key"],
                            "index": index,
                            **{k: fact.get(k) for k in ("fact_type", "claim", "quote")},
                        }
                    )
        info = json.loads((directory / "run.json").read_text())
        metrics = label_metrics(rows, all_refs, {})
        metrics["cost_usd"] = info.get("cost_usd")
        metrics["cost_per_doc"] = (
            info["cost_usd"] / metrics["documents"] if info.get("cost_usd") is not None else None
        )
        at_std = standard_price_cost(info.get("model"), info.get("calls") or [])
        metrics["cost_per_doc_at_standard_price"] = (
            at_std / metrics["documents"] if at_std is not None else None
        )
        at_batch = batch_price_cost(
            info.get("model"), info.get("cost_usd"), info.get("calls") or []
        )
        metrics["cost_per_doc_at_batch_price"] = (
            at_batch / metrics["documents"] if at_batch is not None else None
        )
        table[name] = metrics
        print(name, json.dumps(metrics))
    if args.unread_out:
        Path(args.unread_out).write_text(
            "".join(json.dumps(u, ensure_ascii=False) + "\n" for u in unread), encoding="utf-8"
        )
        print(f"wrote {args.unread_out} ({len(unread)} facts with no verdict yet)")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(table, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.json_out}")
    return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", required=True, help="OpenRouter model id (sync slug)")
    p.add_argument("--effort", default=None, help="reasoning effort (e.g. max); default: off")
    p.add_argument("--service-tier", default=None, help="e.g. flex (half price, sync)")
    p.add_argument("--max-tokens", type=int, default=4096, help="output cap per call")
    p.add_argument("--expect-output-tokens", type=int, default=1500, help="for the estimate")
    p.add_argument("--max-usd", type=float, default=DEFAULT_CAP_USD, help="cap on luna_eval")
    p.add_argument("--concurrency", type=int, default=6)
    p.add_argument("--read-timeout-s", type=float, default=1800.0)
    p.add_argument(
        "--reserve-factor", type=float, default=1.0, help="worst cases held per call, for the cap"
    )
    p.add_argument("--dry-run", action="store_true", help="print the estimate and stop")
    p.add_argument("--out", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="luna_eval")
    sub = parser.add_subparsers(dest="command", required=True)

    exam = sub.add_parser("exam", help="Halvard exam slice")
    _add_model_args(exam)
    exam.add_argument("--variant", default="v6-type-precision")
    exam.add_argument("--source-ids", nargs="*", default=None)
    exam.add_argument("--batch", action="store_true", help="send through the Batch API")
    exam.add_argument("--poll-seconds", type=float, default=20.0)
    exam.add_argument("--batch-timeout-s", type=float, default=3 * 3600.0)
    exam.set_defaults(func=cmd_exam)

    table = sub.add_parser("table", help="aggregate exam runs")
    table.add_argument("--run", action="append", required=True, help="name=dir")
    table.add_argument("--json-out", default=None)
    table.set_defaults(func=cmd_table)

    default_eval = shared_var_root(REPO_ROOT / "var") / "data" / "label_judge" / "2026-09-22"
    default_v2 = shared_var_root(REPO_ROOT / "var") / "data" / "synth_v2" / "smoke_wp4fix"
    labels = sub.add_parser("labels", help="label the hand-verdict set")
    _add_model_args(labels)
    labels.add_argument("--eval-dir", default=str(default_eval))
    labels.add_argument("--v2-docs", default=str(default_v2))
    labels.set_defaults(func=cmd_labels)

    default_wp5 = shared_var_root(REPO_ROOT / "var") / "data" / "synth_v2" / "smoke_wp5" / "wp4fix"
    tlabels = sub.add_parser("teacher-labels", help="the teacher's stored labels, no call")
    tlabels.add_argument("--teacher", default="moonshotai/kimi-k2")
    tlabels.add_argument("--eval-dir", default=str(default_eval))
    tlabels.add_argument("--wp5-dir", default=str(default_wp5))
    tlabels.add_argument("--out", required=True)
    tlabels.set_defaults(func=cmd_teacher_labels)

    ltable = sub.add_parser("label-table", help="aggregate label runs")
    ltable.add_argument("--run", action="append", required=True, help="name=dir")
    ltable.add_argument("--eval-dir", default=str(default_eval))
    ltable.add_argument("--wp5-dir", default=str(default_wp5), help="v2 scripted facts, decoys")
    ltable.add_argument("--extra-verdicts", default=None, help="by-eye verdicts, jsonl")
    ltable.add_argument("--unread-out", default=None, help="write facts still needing a verdict")
    ltable.add_argument("--json-out", default=None)
    ltable.set_defaults(func=cmd_label_table)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
