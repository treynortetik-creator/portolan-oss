"""Run and grade the chat exam (01-spec/chat-exam-2026-09-29.md; cases in
`portolan.bench.chat_exam_cases`).

1. **Answer.** Each case goes through the real chat path in-process against its local graph:
   `run_chat_agent` (agent mode) or `answer_question` (single mode), exactly as
   `portolan/web/routes/chat.py` calls them, on a metered `OpenRouterClient`.
2. **Check, in code.** Every citation resolves (the note exists; a claim id is on that note);
   every passage quote is verbatim (glyph-folded containment) in its note's page text; an
   abstain case must end with no cited sentence, and an answerable one must not.
3. **Judge.** A model reads the question, the gold claims and the answer's sentences with the
   evidence each cites, and says per gold claim whether the answer states it and per
   sentence whether its evidence supports it. It never sees the mode, model or run.
   Case score = gold claims stated / gold claims x supported sentences / sentences; an
   abstain case scores 1 for abstaining, 0 otherwise; a failed check scores 0.

Test-half transcripts and verdicts are written only under `var/evals/locked/` (refused to
worktree agents by `.claude/hooks/no-secrets.py`); the readable results carry, for a test
case, its id, type, graph, score and pass. Every paid call is metered under `SPEND_KEY`.
"""

from __future__ import annotations

import datetime as dt
import gzip
import html
import json
import math
import random
import statistics
import subprocess
import threading
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psycopg

from portolan.bench.chat_exam_cases import REPO_ROOT, ExamCase
from portolan.training.spend import SpendMeter, default_spend_path, shared_var_root

__all__ = [
    "DEFAULT_JUDGE_MODEL",
    "DEFAULT_MODEL",
    "HARD_CAP_USD",
    "SPEND_KEY",
    "Budget",
    "bootstrap_ci",
    "check_answer",
    "compare_results",
    "judge_prompt",
    "min_difference",
    "noise_sd_for_rule",
    "parse_verdict",
    "score_case",
    "sd_upper_bound",
]

SPEND_KEY = "chat_exam"
# The exam's whole budget under SPEND_KEY (orchestrator, 2026-09-29); no run passes it.
HARD_CAP_USD = 3.0
DEFAULT_MODEL = "anthropic/claude-sonnet-5"
# Graded twice, Haiku 4.5 flipped no verdict; the DeepSeek flashes flipped 2 to 3% and marked
# claims the answer states verbatim as missing (01-spec/chat-exam-2026-09-29.md, "The judge").
DEFAULT_JUDGE_MODEL = "anthropic/claude-haiku-4.5"
HEADROOM_WARN = 0.95
# The most one judgement can cost (about 3,000 tokens in, 900 out, on the dearest judge tried).
JUDGE_RESERVE_USD = 0.02
BOOTSTRAP_ROUNDS = 10_000
Z95 = 1.96
Z90_ONE_SIDED = 1.2816

_VAR = REPO_ROOT / "var"


def results_root() -> Path:
    return shared_var_root(_VAR) / "evals" / "chat-exam"


def locked_results_root() -> Path:
    return shared_var_root(_VAR) / "evals" / "locked" / "chat-exam"


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------


class Budget:
    """Stops new paid work before `SPEND_KEY` passes this run's cap: the lower of the key's
    spend at start plus `max_usd`, and `HARD_CAP_USD`. `reserve` is the most one unit of
    work (an answer, a judgement) can cost; in-flight units hold their reserve, and a unit
    that would fit once they finish waits for them instead of being refused."""

    def __init__(self, meter: SpendMeter, max_usd: float, hard_cap: float = HARD_CAP_USD) -> None:
        self.meter = meter
        self.start_usd = meter.corpus_usd(SPEND_KEY)
        self.cap_usd = min(self.start_usd + max_usd, hard_cap)
        self._cond = threading.Condition()
        self._held = 0.0

    def spent(self) -> float:
        return self.meter.corpus_usd(SPEND_KEY) - self.start_usd

    def acquire(self, reserve: float) -> bool:
        with self._cond:
            while True:
                spent = self.meter.corpus_usd(SPEND_KEY)
                if spent + reserve > self.cap_usd:
                    return False
                if spent + self._held + reserve <= self.cap_usd:
                    self._held += reserve
                    return True
                self._cond.wait(timeout=5.0)

    def release(self, reserve: float) -> None:
        with self._cond:
            self._held = max(0.0, self._held - reserve)
            self._cond.notify_all()


# ---------------------------------------------------------------------------
# Answering through the real chat path
# ---------------------------------------------------------------------------


class _Recorder:
    """Delegates to the chat's client and notes what the answer transcript needs: each
    turn's finish reason (a `length` stop is a truncation) and the tokens sent."""

    def __init__(self, client: Any) -> None:
        self._client = client
        self.finish_reasons: list[str | None] = []
        self.truncated = False

    def complete(self, **kwargs: Any) -> Any:
        result = self._client.complete(**kwargs)
        if result.output_tokens >= kwargs.get("max_tokens", 4096):
            self.truncated = True
        return result

    def complete_with_tools(self, **kwargs: Any) -> Any:
        turn = self._client.complete_with_tools(**kwargs)
        self.finish_reasons.append(turn.finish_reason)
        if turn.finish_reason == "length":
            self.truncated = True
        return turn


class _LockedEmbedder:
    """One embedding model shared by the worker threads, one call at a time."""

    def __init__(self, embedder: Any) -> None:
        self._embedder = embedder
        self._lock = threading.Lock()
        self.dim = getattr(embedder, "dim", None)
        self.model_name = getattr(embedder, "model_name", None)

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._embedder, name)
        if not callable(attr):
            return attr

        def locked(*args: Any, **kwargs: Any) -> Any:
            with self._lock:
                return attr(*args, **kwargs)

        return locked


@dataclass
class RunConfig:
    mode: str
    model: str
    judge_model: str
    limits: dict[str, Any]
    label: str = ""

    def public(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "model": self.model,
            "judge_model": self.judge_model,
            "limits": self.limits,
            "label": self.label,
        }


def answer_case(
    case: ExamCase,
    conn: psycopg.Connection,
    embedder: Any,
    client: Any,
    cfg: RunConfig,
    *,
    cost_of: Callable[[], float],
) -> dict[str, Any]:
    """The transcript of one answer: sentences with citations, the answer's usage ledger,
    what search returned, wall time, cost, truncation, and any error."""
    from portolan.web.answer import answer_question
    from portolan.web.chat_agent import AgentLimits, answer_question_agent

    llm = _Recorder(client)
    started = time.monotonic()
    error = None
    answer = None
    try:
        if cfg.mode == "agent":
            answer = answer_question_agent(
                conn,
                llm,
                embedder,
                case.question,
                model=cfg.model,
                fallback_model=cfg.model,
                limits=AgentLimits(**cfg.limits),
                tenant="test",
            )
        else:
            answer = answer_question(
                conn, llm, embedder, case.question, k=8, expand=1, model=cfg.model
            )
    except Exception as exc:  # noqa: BLE001 -- an error is a result: recorded, scored 0
        error = f"{type(exc).__name__}: {str(exc)[:300]}"
    wall = time.monotonic() - started
    sentences = []
    usage: dict[str, Any] = {}
    retrieved: list[str] = []
    if answer is not None:
        usage = answer.usage
        retrieved = [h.note_slug for h in answer.retrieved]
        sentences = [
            {
                "text": s.text,
                "citations": [c.model_dump(exclude_none=True) for c in s.citations],
            }
            for s in answer.sentences
        ]
    agent = usage.get("agent") or {}
    return {
        "id": case.id,
        "question": case.question,
        "sentences": sentences,
        "abstained": answer is not None and not any(s["citations"] for s in sentences),
        "usage": usage,
        "retrieved": retrieved,
        "wall_s": round(wall, 2),
        "cost_usd": round(cost_of(), 6),
        "truncated": llm.truncated,
        "fallback": agent.get("fallback"),
        "error": error,
    }


# ---------------------------------------------------------------------------
# Checks in code
# ---------------------------------------------------------------------------


def _page_texts(conn: psycopg.Connection, slug: str) -> list[str]:
    """What the note page shows (reader text of the body), plus its chunks as the chat saw
    them; a verbatim quote is in one of these."""
    from portolan.web.answer import link_titles, reader_text

    with conn.cursor() as cur:
        cur.execute("SELECT body_md FROM notes WHERE slug = %s", (slug,))
        row = cur.fetchone()
        cur.execute("SELECT text FROM chunks WHERE note_slug = %s ORDER BY id", (slug,))
        chunks = [r[0] for r in cur.fetchall()]
    texts = [row[0] or ""] if row else []
    texts += chunks
    titles = link_titles(conn, texts)
    return [reader_text(t, titles) for t in texts]


def check_answer(conn: psycopg.Connection, transcript: dict[str, Any]) -> dict[str, Any]:
    """Citations resolve and passage quotes are verbatim. Returns counts and the failures."""
    from portolan.extract.quote_gate import _quote_supported_by_text

    unresolved: list[str] = []
    not_verbatim: list[str] = []
    n = 0
    pages: dict[str, list[str]] = {}
    with conn.cursor() as cur:
        for sentence in transcript["sentences"]:
            for c in sentence["citations"]:
                n += 1
                slug = c.get("note_slug", "")
                cur.execute("SELECT 1 FROM notes WHERE slug = %s", (slug,))
                if cur.fetchone() is None:
                    unresolved.append(f"note {slug}")
                    continue
                claim_id = c.get("claim_id")
                if claim_id is not None:
                    cur.execute("SELECT note_slug FROM claims WHERE id = %s", (claim_id,))
                    row = cur.fetchone()
                    if row is None or row[0] != slug:
                        unresolved.append(f"claim {claim_id} on {slug}")
                quote = c.get("quote")
                if c.get("passage_id") is not None:
                    if slug not in pages:
                        pages[slug] = _page_texts(conn, slug)
                    if not quote or not any(
                        _quote_supported_by_text(quote, t) for t in pages[slug]
                    ):
                        not_verbatim.append(f"{slug}: {str(quote)[:80]}")
    return {
        "citations": n,
        "unresolved": unresolved,
        "not_verbatim": not_verbatim,
        "ok": not unresolved and not not_verbatim,
    }


def evidence_for(conn: psycopg.Connection, transcript: dict[str, Any]) -> list[list[str]]:
    """Per sentence, what each citation stands on, as the note page shows it: a claim's text
    with its date and source, a passage quote with its note's title, or, for a note-level
    citation, the note's title."""
    from portolan.bench.chat_exam_cases import _base_text

    out: list[list[str]] = []
    with conn.cursor() as cur:
        for sentence in transcript["sentences"]:
            ev: list[str] = []
            for c in sentence["citations"]:
                cur.execute("SELECT title FROM notes WHERE slug = %s", (c["note_slug"],))
                row = cur.fetchone()
                title = row[0] if row else c["note_slug"]
                if c.get("claim_id") is not None:
                    cur.execute(
                        "SELECT claim_text, recorded::date, source_slug FROM claims WHERE id = %s",
                        (c["claim_id"],),
                    )
                    claim = cur.fetchone()
                    if claim:
                        text, recorded, source = claim
                        ev.append(f"{_base_text(text)} (recorded {recorded}, source {source})")
                elif c.get("quote"):
                    ev.append(f'"{c["quote"]}" (quoted from the note "{title}")')
                else:
                    ev.append(f'(the whole note "{title}")')
            out.append(ev)
    return out


# ---------------------------------------------------------------------------
# The judge
# ---------------------------------------------------------------------------

JUDGE_SYSTEM = (
    "You grade one answer from a company's knowledge-graph chat. You are strict and literal. "
    "You judge only what the answer's sentences say; you do not know or care how the answer "
    "was produced."
)

JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "checklist": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "stated": {"type": "string", "enum": ["yes", "no"]},
                },
                "required": ["id", "stated"],
            },
        },
        "sentences": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "supported": {"type": "string", "enum": ["yes", "no"]},
                },
                "required": ["id", "supported"],
            },
        },
    },
    "required": ["checklist", "sentences"],
}


def judge_prompt(
    question: str, gold: Sequence[str], sentences: Sequence[str], evidence: Sequence[list[str]]
) -> str:
    lines = [f"Question: {question}", "", "Checklist (facts a complete answer states):"]
    lines += [f"G{i}. {g}" for i, g in enumerate(gold, 1)]
    lines += ["", "Answer, one sentence per line, each with the evidence it cites:"]
    for i, (s, ev) in enumerate(zip(sentences, evidence, strict=True), 1):
        lines.append(f"S{i}. {s}")
        for e in ev:
            lines.append(f"    evidence: {e}")
    lines += [
        "",
        "For each checklist fact G: stated = yes only if the answer says it (same meaning, "
        "any wording). Names, dates, numbers and the direction of a change must match. A fact "
        "the answer only hints at, or that a reader would have to infer, is no.",
        "For each answer sentence S: supported = yes when every fact it states (a name, "
        "number, date, event, attribute or relationship) is given by its evidence; a claim's "
        "recorded date counts as evidence of when it was said. Framing, summing up and plain "
        "conclusions drawn from the evidence are fine. supported = no when the sentence states "
        "a fact its evidence does not give, or contradicts it.",
        'Reply with JSON only: {"checklist": [{"id": "G1", "stated": "yes"}, ...], '
        '"sentences": [{"id": "S1", "supported": "yes"}, ...]} with every G and every S.',
    ]
    return "\n".join(lines)


def parse_verdict(data: Any, n_gold: int, n_sentences: int) -> dict[str, list[bool]] | None:
    """`{"stated": [...], "supported": [...]}` in G/S order, or None when a G or S is
    missing or malformed."""
    if not isinstance(data, dict):
        return None

    def read(key: str, prefix: str, field_name: str, n: int) -> list[bool] | None:
        rows = data.get(key)
        if not isinstance(rows, list):
            return None
        got: dict[int, bool] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            raw_id = str(row.get("id", "")).strip().upper().lstrip(prefix)
            value = str(row.get(field_name, "")).strip().lower()
            if raw_id.isdigit() and value in ("yes", "no"):
                got[int(raw_id)] = value == "yes"
        if set(got) != set(range(1, n + 1)):
            return None
        return [got[i] for i in range(1, n + 1)]

    stated = read("checklist", "G", "stated", n_gold)
    supported = read("sentences", "S", "supported", n_sentences)
    if stated is None or supported is None:
        return None
    return {"stated": stated, "supported": supported}


def judge_case(
    client: Any,
    judge_model: str,
    case: ExamCase,
    transcript: dict[str, Any],
    evidence: list[list[str]],
    *,
    attempts: int = 2,
) -> dict[str, list[bool]] | None:
    from portolan.llm.json_util import extract_json_object

    sentences = [s["text"] for s in transcript["sentences"]]
    prompt = judge_prompt(case.question, [g.text for g in case.gold], sentences, evidence)
    for _ in range(attempts):
        result = client.complete(
            system=JUDGE_SYSTEM,
            user=prompt,
            json_schema=JUDGE_SCHEMA,
            model=judge_model,
            surface="audit",
            max_tokens=900,
            validate_json=False,
            temperature=0.0,
        )
        data = result.data if result.data is not None else extract_json_object(result.text or "")
        verdict = parse_verdict(data, len(case.gold), len(sentences))
        if verdict is not None:
            return verdict
    return None


def needs_judge(case: ExamCase, transcript: dict[str, Any], checks: dict[str, Any]) -> bool:
    return (
        bool(case.gold)
        and not case.expects_abstain
        and transcript["error"] is None
        and checks["ok"]
        and not transcript["abstained"]
    )


def score_case(
    case: ExamCase,
    transcript: dict[str, Any],
    checks: dict[str, Any],
    verdict: dict[str, list[bool]] | None,
) -> dict[str, Any]:
    """`score` in [0, 1] (None when the judge failed twice: left out of the mean, counted),
    `pass` when it is 1, and the outcome that decided it."""
    row: dict[str, Any] = {"recall": None, "faithfulness": None}
    if transcript["error"] is not None:
        return {**row, "score": 0.0, "pass": False, "outcome": "error"}
    if not checks["ok"]:
        return {**row, "score": 0.0, "pass": False, "outcome": "check_failed"}
    if case.expects_abstain:
        ok = transcript["abstained"]
        return {**row, "score": float(ok), "pass": ok, "outcome": "abstained" if ok else "answered"}
    if transcript["abstained"]:
        return {**row, "score": 0.0, "pass": False, "outcome": "abstained"}
    if not case.gold:
        return {**row, "score": 1.0, "pass": True, "outcome": "answered"}
    if verdict is None:
        return {**row, "score": None, "pass": None, "outcome": "judge_failed"}
    recall = sum(verdict["stated"]) / len(verdict["stated"])
    faith = sum(verdict["supported"]) / len(verdict["supported"]) if verdict["supported"] else 1.0
    score = recall * faith
    return {
        "recall": round(recall, 4),
        "faithfulness": round(faith, 4),
        "score": round(score, 4),
        "pass": score >= 1.0 - 1e-9,
        "outcome": "answered",
    }


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def bootstrap_ci(
    values: Sequence[float], *, rounds: int = BOOTSTRAP_ROUNDS, seed: int = 0
) -> tuple[float, float]:
    """95% percentile interval of the mean, resampling cases with replacement."""
    if not values:
        return (math.nan, math.nan)
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(rounds))
    return (means[int(0.025 * rounds)], means[int(0.975 * rounds) - 1])


def min_difference(per_case_sd: float, n_cases: int) -> float:
    """The smallest mean-score change (0 to 1) two runs of `n_cases` cases can show beyond
    their noise at 95%: each run's mean carries `per_case_sd / sqrt(n)` of noise, so their
    difference carries sqrt(2) times that. Never below one case's weight."""
    if n_cases <= 0:
        return math.nan
    return max(Z95 * math.sqrt(2) * per_case_sd / math.sqrt(n_cases), 1.0 / n_cases)


def per_case_noise_sd(first: Sequence[float], second: Sequence[float]) -> float:
    """One run's per-case SD from two runs (or gradings) of the same cases: the sample SD of
    their difference over sqrt(2)."""
    diffs = [a - b for a, b in zip(first, second, strict=True)]
    if len(diffs) < 2:
        return 0.0
    return statistics.stdev(diffs) / math.sqrt(2)


def sd_upper_bound(sd: float, n_pairs: int) -> float:
    """The one-sided 90% upper confidence bound of an SD estimated from `n_pairs` pairs
    (chi-square with n-1 degrees of freedom, Wilson-Hilferty's approximation): a noise
    figure from eight re-runs is itself noisy, so the acceptance rule uses its high end."""
    if n_pairs < 2:
        return sd
    k = n_pairs - 1
    chi2_low = k * (1 - 2 / (9 * k) - Z90_ONE_SIDED * math.sqrt(2 / (9 * k))) ** 3
    return sd * math.sqrt(k / chi2_low)


def noise_sd_for_rule(noise: dict[str, Any] | None) -> float | None:
    """The per-case SD the acceptance rule uses: the stored upper bound, else the bound of
    the stored estimate over the pairs it came from."""
    if not noise:
        return None
    if noise.get("per_case_sd_upper90") is not None:
        return noise["per_case_sd_upper90"]
    if noise.get("per_case_sd") is None:
        return None
    pairs = noise.get("repeated") if noise.get("answer_sd") is not None else noise.get("regraded")
    return sd_upper_bound(noise["per_case_sd"], pairs or 0)


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    scored = [r["score"] for r in rows if r.get("score") is not None]
    lo, hi = bootstrap_ci(scored)
    by_type: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if r.get("score") is not None:
            by_type[r["type"]].append(r["score"])
    return {
        "n": len(rows),
        "scored": len(scored),
        "mean_pct": round(100 * sum(scored) / len(scored), 1) if scored else None,
        "ci95_pct": [round(100 * lo, 1), round(100 * hi, 1)] if scored else None,
        "passed": sum(1 for r in rows if r.get("pass")),
        "by_type": {
            t: {"n": len(v), "mean_pct": round(100 * sum(v) / len(v), 1)}
            for t, v in sorted(by_type.items())
        },
        "outcomes": dict(Counter(r.get("outcome") for r in rows)),
    }


# ---------------------------------------------------------------------------
# Compare two runs: the acceptance rule
# ---------------------------------------------------------------------------


def _paired(a: dict[str, Any], b: dict[str, Any], split: str) -> list[tuple[float, float]]:
    ra = {r["id"]: r for r in a["rows"] if r["split"] == split and r.get("score") is not None}
    rb = {r["id"]: r for r in b["rows"] if r["split"] == split and r.get("score") is not None}
    return [(ra[i]["score"], rb[i]["score"]) for i in sorted(set(ra) & set(rb))]


def compare_results(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """KEEP only if train and test both improve by more than the minimum difference; REVERT
    if either falls by more than it, or train gains while test does not (overfitting);
    otherwise NOISE."""
    if a["config"]["judge_model"] != b["config"]["judge_model"]:
        raise ValueError("the two runs were graded by different judges; scores do not compare")
    out: dict[str, Any] = {"splits": {}}
    for split in ("train", "test"):
        pairs = _paired(a, b, split)
        if not pairs:
            out["splits"][split] = None
            continue
        diffs = [y - x for x, y in pairs]
        mean = sum(diffs) / len(diffs)
        lo, hi = bootstrap_ci(diffs, seed=1)
        sds = [s for s in (noise_sd_for_rule(r.get("noise")) for r in (a, b)) if s is not None]
        # With no noise measured on either run, one case is the least that can count.
        min_pct = 100 * min_difference(max(sds, default=0.0), len(pairs))
        out["splits"][split] = {
            "n": len(pairs),
            "a_pct": round(100 * sum(x for x, _ in pairs) / len(pairs), 1),
            "b_pct": round(100 * sum(y for _, y in pairs) / len(pairs), 1),
            "diff_pct": round(100 * mean, 1),
            "diff_ci95_pct": [round(100 * lo, 1), round(100 * hi, 1)],
            "min_diff_pct": round(min_pct, 1),
        }
    tr, te = out["splits"].get("train"), out["splits"].get("test")
    if tr is None or te is None:
        out["decision"] = "NOISE"
        out["why"] = "both runs need train and test rows to apply the rule"
        return out

    def up(s: dict[str, Any]) -> bool:
        return s["diff_pct"] > s["min_diff_pct"] and s["diff_ci95_pct"][0] > 0

    def down(s: dict[str, Any]) -> bool:
        return s["diff_pct"] < -s["min_diff_pct"]

    if up(tr) and up(te):
        out["decision"], out["why"] = "KEEP", "train and test both improved beyond the noise"
    elif down(tr) or down(te):
        out["decision"], out["why"] = "REVERT", "a split fell by more than the noise"
    elif up(tr):
        out["decision"], out["why"] = "REVERT", "train improved and test did not: overfitting"
    else:
        out["decision"], out["why"] = "NOISE", "no change beyond the noise"
    return out


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def public_row(row: dict[str, Any]) -> dict[str, Any]:
    """What a readable file may carry about a test case."""
    keys = ("id", "split", "type", "graph", "score", "pass", "outcome", "cost_usd", "wall_s")
    return {k: row.get(k) for k in keys} | {"infra": row.get("infra") or []}


def write_locked(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(json.dumps(payload, ensure_ascii=False).encode(), mtime=0))


def read_locked(path: Path) -> Any:
    return json.loads(gzip.decompress(path.read_bytes()).decode("utf-8"))


_CSS = """
:root{--bg:#f7f5f0;--fg:#1d2433;--muted:#5b6475;--line:#d9d4c7;--ok:#1f7a4d;--bad:#b3261e;
--card:#fff}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#12161f;--fg:#e7e3da;
--muted:#9aa3b5;--line:#2c3444;--ok:#5cc58f;--bad:#f08a80;--card:#1a202c}}
body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:0;
padding:16px;max-width:1100px;margin:auto}
table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid var(--line);
padding:4px 6px;text-align:left;vertical-align:top}.ok{color:var(--ok)}.bad{color:var(--bad)}
.muted{color:var(--muted)}details{background:var(--card);border:1px solid var(--line);
border-radius:6px;padding:8px 12px;margin:8px 0}code{font-size:13px}
"""


def review_html(results: dict[str, Any], transcripts: dict[str, dict[str, Any]]) -> str:
    """One page: the summary, a row per case, and a transcript section per train case. A
    test row shows its id, type, graph and score only."""
    esc = html.escape
    cfg = results["config"]
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        f"<title>Chat exam review</title><style>{_CSS}</style></head><body>",
        "<h1>Chat exam</h1>",
        f"<p class='muted'>{esc(results['started_at'])} · {esc(cfg['mode'])} · "
        f"{esc(cfg['model'])} · judge {esc(cfg['judge_model'])} · {esc(results['git_sha'])}</p>",
    ]
    for split, s in results["summary"].items():
        if s is None:
            continue
        parts.append(
            f"<p><b>{esc(split)}</b>: {s['mean_pct']}% (95% CI {s['ci95_pct']}), "
            f"{s['passed']}/{s['n']} passed</p>"
        )
    parts.append(
        "<table><tr><th>case</th><th>split</th><th>type</th><th>graph</th><th>score</th>"
        "<th>outcome</th><th>$</th><th>s</th></tr>"
    )
    for r in results["rows"]:
        link = (
            f"<a href='#{esc(r['id'])}'>{esc(r['id'])}</a>"
            if r["split"] != "test"
            else esc(r["id"])
        )
        score = "-" if r.get("score") is None else f"{r['score']:.2f}"
        cls = "ok" if r.get("pass") else "bad"
        parts.append(
            f"<tr><td>{link}</td><td>{esc(r['split'])}</td><td>{esc(r['type'])}</td>"
            f"<td>{esc(r['graph'])}</td><td class='{cls}'>{score}</td>"
            f"<td>{esc(str(r.get('outcome')))}</td><td>{r.get('cost_usd') or 0:.4f}</td>"
            f"<td>{r.get('wall_s') or 0}</td></tr>"
        )
    parts.append("</table><h2>Train transcripts</h2>")
    for r in results["rows"]:
        if r["split"] == "test" or r["id"] not in transcripts:
            continue
        t = transcripts[r["id"]]
        parts.append(
            f"<details id='{esc(r['id'])}'><summary>{esc(r['id'])}: {esc(t['question'])}</summary>"
        )
        parts.append("<p><b>Answer</b></p><ol>")
        verdict = r.get("verdict") or {}
        supported = verdict.get("supported") or []
        for i, s in enumerate(t["sentences"]):
            mark = ""
            if i < len(supported):
                mark = (
                    " <span class='ok'>supported</span>"
                    if supported[i]
                    else " <span class='bad'>unsupported</span>"
                )
            cites = ", ".join(
                esc(c.get("note_slug", ""))
                + (f"#claim-{c['claim_id']}" if c.get("claim_id") else "")
                + (f" “{esc(str(c.get('quote')))}”" if c.get("quote") else "")
                for c in s["citations"]
            )
            parts.append(f"<li>{esc(s['text'])}{mark}<br><span class='muted'>{cites}</span></li>")
        parts.append("</ol><p><b>Gold claims</b></p><ol>")
        stated = verdict.get("stated") or []
        for i, g in enumerate(r.get("gold") or []):
            mark = ""
            if i < len(stated):
                mark = (
                    " <span class='ok'>stated</span>"
                    if stated[i]
                    else " <span class='bad'>missing</span>"
                )
            notes = esc(", ".join(g["notes"]))
            parts.append(f"<li>{esc(g['text'])}{mark} <span class='muted'>{notes}</span></li>")
        parts.append("</ol>")
        ledger = esc(json.dumps(t.get("usage", {}).get("agent", {}), sort_keys=True))
        parts.append(
            f"<p class='muted'>checks {esc(json.dumps(r.get('checks')))} · agent "
            f"<code>{ledger}</code></p></details>"
        )
    parts.append("</body></html>")
    return "\n".join(parts)


@dataclass
class RunPaths:
    run_dir: Path
    locked_dir: Path

    @classmethod
    def new(cls, label: str) -> RunPaths:
        stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
        name = f"{stamp}-{label}" if label else stamp
        return cls(results_root() / name, locked_results_root() / name)


def pool_run(items: Sequence[Any], fn: Callable[[Any], Any], workers: int) -> list[Any]:
    if workers <= 1:
        return [fn(i) for i in items]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, items))


def default_meter() -> SpendMeter:
    return SpendMeter(default_spend_path(_VAR))


def _sink_total(sink: Any) -> float:
    return sum(e.cost_estimate or 0.0 for e in sink.events)
