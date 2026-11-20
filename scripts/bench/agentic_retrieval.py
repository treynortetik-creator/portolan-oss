"""Agentic retrieval bench driver (`portolan.bench.agentic`); write-up
`01-spec/bench-agentic-retrieval-2026-09-23.md`.

    uv run python scripts/bench/agentic_retrieval.py check
    uv run python scripts/bench/agentic_retrieval.py run --set demo --modes chat,single,agent \\
        --models claude-sonnet-5
    uv run python scripts/bench/agentic_retrieval.py run --set erb --modes agent \\
        --models claude-opus-5-5
    uv run python scripts/bench/agentic_retrieval.py judge --env-file <main checkout>/.env
    uv run python scripts/bench/agentic_retrieval.py report

- `check`: both question sets load, every gold note exists, the ERB sample is
  the 2026-09-22 answer sample.
- `run`: answers each question per mode and model, appending one row per
  answer to `<out-dir>/runs.jsonl` (resumable: a row already there is skipped).
  Claude runs through `claude_agent_sdk` (the local Claude Code login); search
  embeds queries with local fastembed; no paid OpenRouter call. Except modes
  `web_single` and `web_agent` (`run_web_chat`, 01-spec/chat-agent-2026-09-25.md):
  the web chat path as deployed, over a metered `OpenRouterClient` (`--spend-key`,
  default `erb_exam`; `--budget-usd` stops starting questions), e.g.
  `run --set erb --modes web_agent,web_single --models anthropic/claude-sonnet-5
  --env-root <main checkout>`.
- `judge`: ERB's correctness and completeness judges (`portolan.bench.
  erb_answers`) over OpenRouter, with an estimate printed first and a cap on
  this run's new spend (metered under `erb_exam` for ERB questions and
  `agentic_demo` for demo ones).
- `report`: aggregates and writes `summary.json`, plus the ids-and-numbers-only
  results file for `01-spec/`.

Databases (local only): the demo graph in a `portolan_test*` database seeded by
`scripts/bench/demo_search.py --embedder fastembed`, and the ERB exam's
`portolan_erb`. The bench only reads them: the door's usage row per tool call is
not written (`tools._log_usage` is replaced for this process), and for
`portolan_erb`, which predates the AI-access migration, the door's access check
is skipped. ERB text (questions, answers, the agent's queries) stays in the
gitignored `var/`; ERB never enters training.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import shutil
import statistics
import sys
import threading
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

from portolan import db as db_module
from portolan.bench.agentic import (
    DEFAULT_MAX_TOOL_CALLS,
    DEFAULT_WALL_SECONDS,
    WEB_MODES,
    BenchQuestion,
    DoorSession,
    agent_system_prompt,
    door_caller,
    door_tool_specs,
    load_note_evidence,
    run_agent,
    run_chat_path,
    run_single_top10,
    run_web_chat,
)
from portolan.bench.agentic_questions import demo_question_set
from portolan.util.redact import redact_dsn

REPO = Path(__file__).resolve().parents[2]
DEMO_URL = "postgresql://localhost:5432/portolan_test_agentic_demo"
DEMO_TENANT = "test"
ERB_SAMPLE_FILE = REPO / "01-spec" / "bench-erb-answers-2026-09-22.results.json"
SPEND_KEYS = {"erb": "erb_exam", "demo": "agentic_demo"}
JUDGE_MODEL = "anthropic/claude-haiku-4.5"
RESULTS_OUT = REPO / "01-spec" / "bench-agentic-retrieval-2026-09-23.results.json"


def _log(msg: str) -> None:
    print(f"[{datetime.now(UTC):%H:%M:%S}Z] {msg}", flush=True)


# ---------------------------------------------------------------------
# Question sets
# ---------------------------------------------------------------------


def erb_questions(erb_root: Path) -> list[BenchQuestion]:
    """The 2026-09-22 answer sample: 20 of the 157 in-slice questions,
    `random.Random(20260922)`, exactly as `scripts/bench/erb_exam.py answers`."""
    from portolan.bench.erb import (
        ERB_SLICES,
        load_questions,
        load_uuid_index,
        select_in_slice_questions,
    )
    from portolan.bench.erb_answers import sample_questions

    uuid_index = load_uuid_index(erb_root)
    questions = load_questions(erb_root / "questions.jsonl")
    sel = select_in_slice_questions(questions, ERB_SLICES, uuid_index=uuid_index)
    sample = sample_questions(sel.questions, 20)
    return [
        BenchQuestion(
            qid=q["question_id"],
            question=q["question"],
            gold_answer=q.get("gold_answer", ""),
            answer_facts=tuple(q.get("answer_facts") or []),
            expected=tuple(q["expected_doc_ids"]),
            kind=q.get("question_type", ""),
        )
        for q in sample
    ]


def question_set(name: str, args: argparse.Namespace) -> list[BenchQuestion]:
    if name == "demo":
        return list(demo_question_set())
    return erb_questions(Path(args.erb_root).expanduser())


def database_url(name: str, args: argparse.Namespace) -> str:
    if name == "demo":
        url = args.demo_url
        dbname = urlsplit(url).path.lstrip("/")
        if not dbname.startswith("portolan_test") or urlsplit(url).hostname not in {
            "localhost",
            "127.0.0.1",
        }:
            sys.exit(f"refusing {redact_dsn(url)}: local portolan_test* only")
        return url
    from portolan.bench.erb_search import assert_erb_database, erb_database_url

    url = erb_database_url(args.pg_host, args.pg_port)
    assert_erb_database(url)
    return url


def cmd_check(args: argparse.Namespace) -> None:
    for name in ("demo", "erb"):
        qs = question_set(name, args)
        url = database_url(name, args)
        conn = db_module.connect(url, autocommit=True)
        try:
            with conn.cursor() as cur:
                gold = sorted({s for q in qs for s in q.expected})
                cur.execute("SELECT slug FROM notes WHERE slug = ANY(%s)", (gold,))
                found = {r[0] for r in cur.fetchall()}
        finally:
            conn.close()
        missing = sorted(set(gold) - found)
        print(f"{name}: {len(qs)} questions, {len(gold)} gold notes, missing {missing}")
        if missing:
            sys.exit("gold notes missing")
    sample = json.loads(ERB_SAMPLE_FILE.read_text(encoding="utf-8"))["sample"]
    ids = [q.qid for q in question_set("erb", args)]
    print(f"erb sample matches 2026-09-22: {ids == sample}")
    if ids != sample:
        sys.exit("the ERB sample differs from the 2026-09-22 answer sample")


# ---------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------


def _key(row: dict) -> tuple[str, str, str, str]:
    return (row["set"], row["mode"], row["model"], row["question_id"])


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _answer_sha(row: dict) -> str:
    return hashlib.sha256((row.get("answer_text") or "").encode("utf-8")).hexdigest()[:16]


def _failed(row: dict) -> bool:
    """A harness failure (the CLI refused, a crash), not a model's empty answer."""
    return bool(row.get("error")) and "tool_calls" not in row


def latest_runs(path: Path) -> dict[tuple[str, str, str, str], dict]:
    """The newest row per (set, mode, model, question)."""
    out: dict[tuple[str, str, str, str], dict] = {}
    for row in _read_jsonl(path):
        out[_key(row)] = row
    return out


class _Runner:
    """Builds, per worker thread, what one answer needs: a door over the set's
    database, an agent-SDK client, a connection."""

    def __init__(self, set_name: str, url: str, args: argparse.Namespace) -> None:
        from portolan.search.embed import FastEmbedEmbedder

        self.set_name = set_name
        self.url = url
        self.args = args
        self.embedder = FastEmbedEmbedder()
        self.store_root = Path(args.out_dir) / "store"
        self.store_root.mkdir(parents=True, exist_ok=True)
        self.bypass_access = set_name == "erb"
        self.meter = None
        self.api_key = None
        self.spend_key = args.spend_key
        if any(m in WEB_MODES for m in args.modes.split(",")):
            from portolan.bench.erb_search import read_openrouter_key
            from portolan.training.spend import SpendMeter, default_spend_path

            env_file = Path(args.env_root) / ".env" if args.env_root else None
            self.api_key = read_openrouter_key(env_file)
            self.meter = SpendMeter(default_spend_path(REPO / "var"))

    def _door(self):
        from portolan.config import Settings
        from portolan.mcp_server.server import create_server
        from portolan.store import LocalStore

        settings = Settings(
            database_url=self.url,
            tenant=DEMO_TENANT,
            embed_provider="fake",
            store_root=self.store_root,
        )
        door = create_server(settings, embedder=self.embedder, store=LocalStore(self.store_root))
        if self.bypass_access:
            door._access_gate = lambda method: None
        return door

    def _evidence(self, slugs: Iterable[str]):
        conn = db_module.connect(self.url, autocommit=True)
        try:
            return load_note_evidence(conn, slugs)
        finally:
            conn.close()

    def web_client(self, model: str):
        """A metered OpenRouter client for one question: its own sink, so the
        question's calls and cost can be read back."""
        from portolan.llm.openrouter import OpenRouterClient
        from portolan.metering import InMemorySink

        return OpenRouterClient(
            spend_key=self.spend_key,
            spend_meter=self.meter,
            api_key=self.api_key,
            default_model=model,
            sink=InMemorySink(),
            tenant="bench",
        )

    def answer(self, mode: str, model: str, q: BenchQuestion) -> dict:
        if mode in WEB_MODES:
            from portolan.web.chat_agent import AgentLimits

            limits = AgentLimits(
                max_tool_calls=self.args.agent_max_calls,
                max_usd=self.args.agent_max_usd,
                timeout_seconds=self.args.agent_timeout,
            )
            conn = db_module.connect(self.url, autocommit=True)
            try:
                run = run_web_chat(
                    conn,
                    self.web_client(model),
                    self.embedder,
                    q,
                    mode=mode,
                    model=model,
                    limits=limits,
                )
            finally:
                conn.close()
            agent = (run.raw or {}).get("agent") or {}
            row = run.as_row(q)
            row.update(
                {
                    "set": self.set_name,
                    "question_type": q.kind,
                    "answer_text": run.answer_text,
                    "raw": run.raw,
                    "stop": agent.get("stop"),
                    "fallback": agent.get("fallback"),
                    "calls_by_tool": agent.get("by_tool"),
                }
            )
            return row
        if mode in ("agent", "agent_seeded"):
            door = self._door()
            session = DoorSession(
                tool_specs=door_tool_specs(door),
                call_door=door_caller(door),
                evidence_for=self._evidence,
                max_tool_calls=self.args.max_calls,
            )
            prompt = agent_system_prompt(door.instructions or "", self.args.max_calls)
            run = asyncio.run(
                run_agent(
                    q,
                    session=session,
                    model=model,
                    system_prompt=prompt,
                    wall_seconds=self.args.wall_seconds,
                    cli_path=self.args.cli_path or None,
                    seed_search=mode == "agent_seeded",
                )
            )
            extra = {"calls": [asdict(c) for c in run.calls]}
        else:
            from portolan.llm.agent_sdk import AgentSDKClient
            from portolan.metering import InMemorySink

            llm = AgentSDKClient(
                default_model=model,
                sink=InMemorySink(),
                tenant="bench",
                cli_path=self.args.cli_path or None,
            )
            conn = db_module.connect(self.url, autocommit=True)
            try:
                fn = run_chat_path if mode == "chat" else run_single_top10
                run = fn(conn, llm, self.embedder, q, model=model)
            finally:
                conn.close()
            extra = {}
        row = run.as_row(q)
        row.update(
            {
                "set": self.set_name,
                "question_type": q.kind,
                "answer_text": run.answer_text,
                "raw": run.raw,
                **extra,
            }
        )
        return row


def _disable_usage_rows() -> None:
    """The bench reads the graphs and writes nothing to them."""
    from portolan.mcp_server import tools

    tools._log_usage = lambda conn, **kwargs: None


def cmd_run(args: argparse.Namespace) -> None:
    _disable_usage_rows()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    runs_path = out / "runs.jsonl"
    # A row that failed before answering is run again; `latest_runs` keeps the newest.
    done = {k for k, r in latest_runs(runs_path).items() if not _failed(r)}
    url = database_url(args.set, args)
    questions = question_set(args.set, args)
    if args.only:
        wanted = set(args.only.split(","))
        questions = [q for q in questions if q.qid in wanted]
    if args.limit:
        questions = questions[: args.limit]
    if args.set == "demo":
        from portolan.mcp_server.access import get_ai_access, set_ai_access

        conn = db_module.connect(url, autocommit=True)
        try:
            if not get_ai_access(conn, DEMO_TENANT).enabled:
                set_ai_access(
                    conn, DEMO_TENANT, enabled=True, changed_by="agentic-bench", reason="local"
                )
        finally:
            conn.close()
    runner = _Runner(args.set, url, args)
    runner.embedder.embed(["warm up"])  # load the model once, before the threads start
    jobs = [
        (mode, model, q)
        for model in args.models.split(",")
        for mode in args.modes.split(",")
        for q in questions
        if (args.set, mode, model, q.qid) not in done
    ]
    _log(f"run: {len(jobs)} answers to produce on {args.set} ({redact_dsn(url)})")
    lock = threading.Lock()

    baseline = runner.meter.corpus_usd(runner.spend_key) if runner.meter is not None else 0.0

    def one(job):
        mode, model, q = job
        started = time.monotonic()
        if runner.meter is not None:
            spent = runner.meter.corpus_usd(runner.spend_key) - baseline
            if spent >= args.budget_usd:
                _log(f"  budget: ${spent:.4f} of ${args.budget_usd:.2f} spent; skipping {q.qid}")
                return
        try:
            row = runner.answer(mode, model, q)
        except Exception as exc:  # noqa: BLE001 -- recorded; the run goes on
            row = {
                "set": args.set,
                "mode": mode,
                "model": model,
                "question_id": q.qid,
                "error": f"{type(exc).__name__}: {str(exc)[:300]}",
                "answer_text": "",
            }
        with lock, runs_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, default=str) + "\n")
        _log(
            f"  {args.set} {mode:6s} {model:16s} {q.qid}: {time.monotonic() - started:5.1f}s "
            f"calls={row.get('tool_calls')} kept={row.get('gate', {}).get('kept')} "
            f"recall_cited={row.get('recall_cited_pct')} err={row.get('error')}"
        )

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(one, jobs))
    if runner.meter is not None:
        spent = runner.meter.corpus_usd(runner.spend_key) - baseline
        _log(f"run: ${spent:.4f} metered under {runner.spend_key} this run")


# ---------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------


# Every judge dollar this bench spends, across invocations and out-dirs, so the
# $2 cap holds for the whole study (the shared meter's `erb_exam` also carries
# other runs' spend).
JUDGE_LEDGER = REPO / "var" / "bench" / "agentic" / "judge_spend.json"


def _judge_estimate(
    rows: list[dict], questions: dict[tuple[str, str], BenchQuestion]
) -> tuple[float, float]:
    """`(expected, upper)` dollars, chars/3 tokens, Haiku 4.5 at $1 / $5, full
    `max_tokens` out; `upper` also pays ERB's two retries of an unparseable
    correctness verdict on every answer."""
    from portolan.bench.erb_prompts import (
        ANSWER_WHOLISTIC_EVALUATION_PROMPT,
        INDIVIDUAL_FACT_VALIDATOR_PROMPT,
    )

    expected = upper = 0.0
    for r in rows:
        answer = r.get("answer_text") or ""
        if not answer.strip():
            continue
        q = questions[(r["set"], r["question_id"])]
        tokens_in = (
            len(ANSWER_WHOLISTIC_EVALUATION_PROMPT) + len(q.question) + len(q.gold_answer)
        ) / 3 + len(answer) / 3
        correctness = (tokens_in * 1.0 + 300 * 5.0) / 1e6
        facts = sum(
            ((len(INDIVIDUAL_FACT_VALIDATOR_PROMPT) + len(answer) + len(f)) / 3 + 20 * 5.0) / 1e6
            for f in q.answer_facts
        )
        expected += correctness + facts
        upper += 3 * correctness + facts
    return expected, upper


def _ledger_spent() -> float:
    if not JUDGE_LEDGER.is_file():
        return 0.0
    return float(json.loads(JUDGE_LEDGER.read_text(encoding="utf-8")).get("usd", 0.0))


def _ledger_add(usd: float) -> None:
    JUDGE_LEDGER.parent.mkdir(parents=True, exist_ok=True)
    JUDGE_LEDGER.write_text(json.dumps({"usd": round(_ledger_spent() + usd, 6)}) + "\n")


def cmd_judge(args: argparse.Namespace) -> None:
    from portolan.bench.erb_answers import OpenRouterChat, score_answer_text
    from portolan.bench.erb_search import BudgetExceeded, read_openrouter_key
    from portolan.training.spend import SpendMeter, default_spend_path

    out = Path(args.out_dir)
    judged_path = out / "judged.jsonl"
    # A verdict belongs to one answer text: a re-run row with new text is judged again.
    judged = {(_key(r), r.get("answer_sha")) for r in _read_jsonl(judged_path)}
    rows = [
        r
        for k, r in latest_runs(out / "runs.jsonl").items()
        if (k, _answer_sha(r)) not in judged and not _failed(r)
    ]
    questions = {(name, q.qid): q for name in ("demo", "erb") for q in question_set(name, args)}
    expected, upper = _judge_estimate(rows, questions)
    meter = SpendMeter(default_spend_path(REPO / "var"))
    keys = {name: args.spend_key or key for name, key in SPEND_KEYS.items()}
    already = {name: meter.corpus_usd(key) for name, key in keys.items()}
    remaining = args.budget_usd - _ledger_spent()
    print(
        f"ESTIMATE (judge): ${expected:.4f} expected, ${upper:.4f} upper bound, for "
        f"{len(rows)} answers; judge {JUDGE_MODEL}; study cap ${args.budget_usd:.2f}, "
        f"${remaining:.4f} left; already metered "
        + ", ".join(f"{keys[n]} ${v:.4f}" for n, v in already.items()),
        flush=True,
    )
    if expected > remaining:
        sys.exit("STOP: the expected spend exceeds what is left of the cap")
    if args.dry_run:
        return
    env_file = Path(args.env_root) / ".env" if args.env_root else None
    if args.env_file:
        env_file = Path(args.env_file)
    key = read_openrouter_key(env_file)
    chats = {
        name: OpenRouterChat(
            key,
            meter=meter,
            corpus=keys[name],
            budget_usd=already[name] + remaining,
        )
        for name in ("demo", "erb")
    }
    lock = threading.Lock()

    def spent() -> float:
        return sum(c.spent_usd for c in chats.values())

    def one(row: dict) -> None:
        if spent() >= remaining:
            raise BudgetExceeded(f"the study's judge spend reached ${args.budget_usd}")
        q = questions[(row["set"], row["question_id"])]
        from portolan.bench.erb_answers import strip_dsid_references

        answer = strip_dsid_references(row.get("answer_text") or "")
        verdict = score_answer_text(chats[row["set"]], JUDGE_MODEL, q.as_judge_dict(), answer)
        with lock, judged_path.open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {
                        **{k: row[k] for k in ("set", "mode", "model", "question_id")},
                        "answer_sha": _answer_sha(row),
                        **verdict,
                    }
                )
                + "\n"
            )

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(one, rows))
    except BudgetExceeded as exc:
        sys.exit(f"STOP: {exc}")
    finally:
        for c in chats.values():
            c.close()
        _ledger_add(spent())
    print(
        f"judge: {sum(c.calls for c in chats.values())} calls, "
        f"${spent():.4f} this run ("
        + ", ".join(f"{keys[n]} ${c.spent_usd:.4f}" for n, c in chats.items())
        + ")"
    )


# ---------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------


def _mean(values: list[float]) -> float | None:
    vals = [v for v in values if v is not None]
    return round(statistics.fmean(vals), 2) if vals else None


def _pct(values: list[float], q: float) -> float | None:
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    idx = min(len(vals) - 1, max(0, round(q * (len(vals) - 1))))
    return round(vals[idx], 1)


def summarize_group(rows: list[dict]) -> dict:
    n = len(rows)
    judged = [r for r in rows if "answer_correct" in r]
    out = {
        "count": n,
        "judged": len(judged),
        "correct_pct": round(sum(r["answer_correct"] for r in judged) / len(judged) * 100, 1)
        if judged
        else None,
        "completeness_pct": _mean([r["completeness_pct"] for r in judged]),
        "combined": _mean([r["completeness_pct"] if r["answer_correct"] else 0.0 for r in judged]),
        "answered_pct": round(sum(bool(r.get("answered")) for r in rows) / n * 100, 1),
        "recall_retrieved_pct": _mean([r.get("recall_retrieved_pct") for r in rows]),
        "recall_in_context_pct": _mean([r.get("recall_in_context_pct") for r in rows]),
        "recall_cited_pct": _mean([r.get("recall_cited_pct") for r in rows]),
        "tool_calls_mean": _mean([r.get("tool_calls") for r in rows]),
        "wall_s_mean": _mean([r.get("wall_s") for r in rows]),
        "wall_s_p50": _pct([r.get("wall_s") for r in rows], 0.5),
        "wall_s_p95": _pct([r.get("wall_s") for r in rows], 0.95),
        "cost_usd_mean": round(statistics.fmean(c), 5)
        if (c := [r["cost_usd"] for r in rows if r.get("cost_usd") is not None])
        else None,
        "input_tokens_mean": _mean(
            [
                sum(
                    (r.get("usage") or {}).get(k, 0)
                    for k in (
                        "input_tokens",
                        "cache_read_input_tokens",
                        "cache_creation_input_tokens",
                    )
                )
                for r in rows
            ]
        ),
        "output_tokens_mean": _mean([(r.get("usage") or {}).get("output_tokens", 0) for r in rows]),
        "errors": sum(1 for r in rows if r.get("error")),
        "fallbacks": sum(1 for r in rows if r.get("fallback")),
    }
    agent = [r for r in rows if r.get("mode") in ("agent", "agent_seeded")]
    if agent:
        kinds = [r.get("calls_by_kind") or {} for r in agent]
        out.update(
            {
                "search_calls_mean": _mean([k.get("search", 0) for k in kinds]),
                "land_calls_mean": _mean([k.get("land", 0) for k in kinds]),
                "hop_calls_mean": _mean([k.get("hop", 0) for k in kinds]),
                "recall_searched_pct": _mean([r.get("recall_searched_pct") for r in agent]),
                "questions_gold_via_hop_only": sum(1 for r in agent if r.get("gold_via_hop_only")),
                "questions_cited_via_hop_only": sum(
                    1 for r in agent if r.get("cited_via_hop_only")
                ),
                "submissions_mean": _mean([r.get("submissions") for r in agent]),
                "resubmitted": sum(1 for r in agent if (r.get("submissions") or 0) > 1),
                "over_budget_calls": sum(r.get("over_budget_calls") or 0 for r in agent),
                "error_calls": sum(r.get("error_calls") or 0 for r in agent),
                "stops": _counts(r.get("stop") for r in agent),
                "sdk_cost_usd_mean": _mean([r.get("sdk_cost_usd") for r in agent]),
                "api_s_mean": _mean([r.get("api_s") for r in agent]),
                "tool_s_mean": _mean([r.get("tool_s") for r in agent]),
            }
        )
    gates = [r.get("gate") or {} for r in rows]
    out["gate"] = {
        k: sum(g.get(k, 0) or 0 for g in gates) for k in sorted({k for g in gates for k in g})
    }
    return out


def _counts(values: Iterable) -> dict:
    out: dict = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return out


def cmd_report(args: argparse.Namespace) -> None:
    out = Path(args.out_dir)
    # `label=dir` pairs: each run directory is one version of the door.
    sources = [s.split("=", 1) for s in args.sources.split(",")] if args.sources else []
    sources = sources or [["run", str(out)]]
    rows = []
    for label, directory in sources:
        verdicts = {
            (_key(r), r.get("answer_sha")): r for r in _read_jsonl(Path(directory) / "judged.jsonl")
        }
        for r in latest_runs(Path(directory) / "runs.jsonl").values():
            v = verdicts.get((_key(r), _answer_sha(r)))
            judged = (
                {k: v[k] for k in ("answer_correct", "completeness_pct", "facts_valid")}
                if v
                else {}
            )
            search_s = sum(
                c["seconds"]
                for c in r.get("calls") or []
                if c["tool"] == "search" and not c.get("over_budget")
            )
            near = sum(
                1
                for c in r.get("calls") or []
                if c["tool"] == "search" and c["arguments"].get("near")
            )
            rows.append(
                {
                    **r,
                    **judged,
                    "door": label,
                    "search_s": round(search_s, 2),
                    "near_searches": near,
                }
            )
    groups: dict[str, list[dict]] = {}
    for r in rows:
        mode = f"{r['door']}|{r['mode']}|{r['model']}"
        groups.setdefault(f"{r['set']}|{mode}", []).append(r)
        if r["set"] == "demo" and r["question_id"] != "live_q7":
            groups.setdefault(f"demo_multihop|{mode}", []).append(r)
        if r["set"] == "erb":
            groups.setdefault(f"erb_type|{mode}|{r.get('question_type')}", []).append(r)
    summary = {k: summarize_group(v) for k, v in sorted(groups.items())}
    tool_stats: dict[str, dict] = {}
    for r in rows:
        for c in r.get("calls") or []:
            if c.get("over_budget"):
                continue
            s = tool_stats.setdefault(
                f"{r['door']}|{r['set']}|{r['model']}|{c['tool']}",
                {"calls": 0, "chars": 0, "seconds": 0.0, "errors": 0, "empty": 0},
            )
            s["calls"] += 1
            s["chars"] += c["result_chars"]
            s["seconds"] += c["seconds"]
            s["errors"] += int(bool(c["is_error"]))
            s["empty"] += int(c.get("detail", "").startswith("empty result"))
    for s in tool_stats.values():
        s["mean_chars"] = round(s.pop("chars") / s["calls"])
        s["mean_seconds"] = round(s.pop("seconds") / s["calls"], 2)
    must = {
        f"{r['door']}|{r['mode']}|{r['model']}": {
            "answer_correct": r.get("answer_correct"),
            "completeness_pct": r.get("completeness_pct"),
            "recall_cited_pct": r.get("recall_cited_pct"),
            "cited": r.get("cited") if "cited" in r else None,
            "answered": r.get("answered"),
        }
        for r in rows
        if r["question_id"] == "live_q7"
    }
    (out / "summary.json").write_text(
        json.dumps({"groups": summary, "tool_stats": tool_stats, "live_q7": must}, indent=2) + "\n",
        encoding="utf-8",
    )
    # Ids and numbers only: no question, answer, query or ERB text.
    keep = (
        "set",
        "mode",
        "model",
        "question_id",
        "question_type",
        "answer_correct",
        "completeness_pct",
        "recall_retrieved_pct",
        "recall_in_context_pct",
        "recall_cited_pct",
        "recall_searched_pct",
        "tool_calls",
        "calls_by_tool",
        "calls_by_kind",
        "gold_first_seen_call",
        "door",
        "wall_s",
        "api_s",
        "search_s",
        "near_searches",
        "tool_s",
        "usage",
        "cost_usd",
        "sdk_cost_usd",
        "gate",
        "submissions",
        "stop",
        "answered",
        "over_budget_calls",
        "error_calls",
        "fallback",
    )
    slim = [{k: r[k] for k in keep if k in r} for r in rows]
    for r, full in zip(slim, rows, strict=True):
        if full.get("set") == "demo":
            r["gold_via_hop_only"] = full.get("gold_via_hop_only")
            r["cited_via_hop_only"] = full.get("cited_via_hop_only")
        else:
            r["gold_via_hop_only"] = len(full.get("gold_via_hop_only") or [])
            r["cited_via_hop_only"] = len(full.get("cited_via_hop_only") or [])
    slim.sort(key=lambda r: (r["door"], r["set"], r["mode"], r["model"], r["question_id"]))
    # Rows as a table (column names once) and compact separators: the file stays under the
    # repo's 200 KB limit for tracked data files (`tests/test_no_outside_material.py`).
    columns = sorted({k for r in slim for k in r})
    results = {
        "run_at": datetime.now(UTC).isoformat(),
        "judge": JUDGE_MODEL,
        "groups": summary,
        "tool_stats": tool_stats,
        "live_q7": must,
        "row_columns": columns,
        "rows": [[r.get(k) for k in columns] for r in slim],
    }
    Path(args.results_out).write_text(
        json.dumps(results, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    for name, s in summary.items():
        print(
            f"{name:42s} n={s['count']:3d} correct={s['correct_pct']} compl={s['completeness_pct']} "
            f"recall(ret/ctx/cited)={s['recall_retrieved_pct']}/{s['recall_in_context_pct']}/"
            f"{s['recall_cited_pct']} calls={s['tool_calls_mean']} "
            f"hops={s.get('hop_calls_mean')} wall p50/p95={s['wall_s_p50']}/{s['wall_s_p95']} "
            f"cost={s['cost_usd_mean']}"
        )
    print(json.dumps(must, indent=1))


def cmd_overhead(args: argparse.Namespace) -> None:
    """The Claude Code CLI's own cost per session (process start-up and
    teardown), outside the model: wall time minus the CLI's `duration_api_ms`
    for a one-word reply. Every mode here pays it once per model session; a
    production path calling the API directly would not."""
    from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query

    async def once(model: str) -> tuple[float, float]:
        options = ClaudeAgentOptions(
            system_prompt="Reply with the single word OK.",
            max_turns=1,
            tools=[],
            model=model,
            setting_sources=[],
            skills=[],
            strict_mcp_config=True,
            mcp_servers={},
            cli_path=args.cli_path or None,
        )
        started = time.monotonic()
        api = 0.0
        async for message in query(prompt="OK?", options=options):
            if isinstance(message, ResultMessage):
                api = (message.duration_api_ms or 0) / 1000
        return time.monotonic() - started, api

    out: dict = {}
    for model in args.models.split(","):
        samples = [asyncio.run(once(model)) for _ in range(args.n)]
        overhead = [w - a for w, a in samples]
        out[model] = {
            "wall_s": [round(w, 2) for w, _ in samples],
            "api_s": [round(a, 2) for _, a in samples],
            "overhead_s_median": round(statistics.median(overhead), 2),
        }
        print(model, out[model], flush=True)
    (Path(args.out_dir) / "overhead.json").write_text(json.dumps(out, indent=2) + "\n")


def main(argv: list[str] | None = None) -> None:
    from portolan.bench.erb import default_erb_root

    parser = argparse.ArgumentParser(prog="agentic_retrieval")
    parser.add_argument("--out-dir", default=str(REPO / "var" / "bench" / "agentic"))
    parser.add_argument("--erb-root", default=str(default_erb_root()))
    parser.add_argument("--demo-url", default=DEMO_URL)
    parser.add_argument("--pg-host", default="localhost")
    parser.add_argument("--pg-port", type=int, default=5432)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    run = sub.add_parser("run")
    run.add_argument("--set", choices=("demo", "erb"), required=True)
    run.add_argument("--modes", default="chat,single,agent")
    run.add_argument("--models", default="claude-sonnet-5")
    run.add_argument("--workers", type=int, default=3)
    run.add_argument("--limit", type=int, default=0)
    run.add_argument("--only", default="", help="comma list of question ids")
    run.add_argument("--max-calls", type=int, default=DEFAULT_MAX_TOOL_CALLS)
    web = run.add_argument_group("web_single / web_agent: the live chat path over OpenRouter")
    web.add_argument("--spend-key", default="erb_exam", help="meter paid calls under this key")
    web.add_argument("--budget-usd", type=float, default=2.5, help="stop starting questions here")
    web.add_argument("--env-root", default="", help="a checkout whose env file holds the key")
    web.add_argument("--agent-max-calls", type=int, default=6)
    web.add_argument("--agent-max-usd", type=float, default=0.15)
    web.add_argument("--agent-timeout", type=float, default=45.0)
    run.add_argument("--wall-seconds", type=float, default=DEFAULT_WALL_SECONDS)
    run.add_argument(
        "--cli-path",
        default=shutil.which("claude") or "",
        help="the Claude Code CLI to drive (default: the installed `claude`; the SDK's "
        "bundled one refuses claude-opus-5-5)",
    )
    judge = sub.add_parser("judge")
    judge.add_argument("--env-file", default="", help="read OPENROUTER_API_KEY from here")
    judge.add_argument(
        "--env-root", default="", help="a checkout whose .env holds the key (from a worktree)"
    )
    judge.add_argument("--budget-usd", type=float, default=2.0)
    judge.add_argument("--workers", type=int, default=4)
    judge.add_argument("--dry-run", action="store_true")
    judge.add_argument(
        "--spend-key", default="", help="meter every set under this corpus key instead"
    )
    report = sub.add_parser("report")
    report.add_argument("--results-out", default=str(RESULTS_OUT))
    report.add_argument(
        "--sources",
        default="",
        help="comma list of label=run-dir (default: run=<out-dir>)",
    )
    overhead = sub.add_parser("overhead")
    overhead.add_argument("--models", default="claude-sonnet-5,claude-opus-5-5")
    overhead.add_argument("--n", type=int, default=5)
    overhead.add_argument("--cli-path", default=shutil.which("claude") or "")
    args = parser.parse_args(argv)
    commands = {
        "check": cmd_check,
        "run": cmd_run,
        "judge": cmd_judge,
        "report": cmd_report,
        "overhead": cmd_overhead,
    }
    commands[args.cmd](args)


if __name__ == "__main__":
    main()
